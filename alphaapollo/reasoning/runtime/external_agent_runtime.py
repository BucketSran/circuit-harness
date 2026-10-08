# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a third-party coding agent behind the shared :class:`AgentRuntime` contract.

An external agent owns its model loop.  It may either own its tools too, in
which case transitions are explicitly synthesized, or call AlphaApollo's tools
through a Runtime-owned Environment and retain those real transitions.

Two properties are deliberate:

* ``AgentTurn.environment_transition`` stays mandatory. Native tool activity
  carries explicitly synthesized transitions. In ``final_response`` mode only
  the final answer is evaluated by the Environment. An Environment-backed
  ``tool_loop`` requires executed tools to pass through that Environment.
  Adapter-observed pre-dispatch schema rejections retain a synthetic failed
  attempt explicitly marked ``environment_stepped=False``.
* The generation records are honest.  An external agent returns text, not a
  trainable token span, so ``provenance`` and every token/logprob field stay
  ``None`` and ``AgentResult.metadata`` is labelled ``policy_source=external_agent``
  with ``trainable=False``.  ``GenerationResponse.is_trainable`` is therefore
  ``False`` by construction rather than by convention.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from alphaapollo.common.environment.base import EnvironmentObservation, EnvironmentTransition
from alphaapollo.common.generation.base import (
    GenerationRequest,
    GenerationResponse,
    SamplingOptions,
    ToolCall,
)
from alphaapollo.common.trajectory.recorder import ensure_safe_environment_input
from alphaapollo.reasoning.runtime.agent_runtime import (
    AgentResult,
    AgentRuntime,
    AgentTask,
    AgentTurn,
)
from alphaapollo.reasoning.runtime.environment_lifecycle import (
    EnvironmentInitialization,
    initialize_environment,
)

__all__ = [
    "EVENT_KINDS",
    "POLICY_SOURCE_EXTERNAL_AGENT",
    "TERMINATION_REASONS",
    "ExternalAgentRuntime",
    "ExternalAgentSession",
    "ExternalAuxiliaryError",
    "ExternalEvent",
    "ExternalRunOutcome",
    "encode_arguments",
    "flatten_content_blocks",
    "project_outcome",
    "workflow_workspace_path",
]

#: Marks a trajectory whose text was produced by an agent we do not control.
POLICY_SOURCE_EXTERNAL_AGENT = "external_agent"

#: ``message``/``reasoning``/``tool_call`` describe what the agent produced;
#: ``tool_result``/``error`` describe what came back to it.
EVENT_KINDS = frozenset({"message", "reasoning", "tool_call", "tool_result", "error"})

TERMINATION_REASONS = frozenset({"final", "truncated", "external_error", "cancelled", "timeout"})

_GENERATION_KINDS = frozenset({"message", "reasoning", "tool_call"})


def _plain_json_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return copy.deepcopy(dict(value))


@dataclass(frozen=True, slots=True)
class ExternalEvent:
    """One normalized event an external agent already emitted.

    ``raw`` keeps the provider record verbatim so a projection bug never destroys
    the only copy of what actually happened.
    """

    kind: str
    content: str = ""
    tool_call: ToolCall | None = None
    call_id: str | None = None
    tool_id: str | None = None
    failed: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)
    # Adapter-observed schema rejection; no tool execution took place.
    rejected_before_dispatch: bool = False

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"kind must be one of {sorted(EVENT_KINDS)}, got {self.kind!r}")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if self.kind == "tool_call":
            if not isinstance(self.tool_call, ToolCall):
                raise TypeError("tool_call events require a ToolCall record")
        elif self.tool_call is not None:
            raise ValueError("tool_call is only valid on a tool_call event")
        for name in ("call_id", "tool_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided")
        if not isinstance(self.failed, bool):
            raise TypeError("failed must be a bool")
        if not isinstance(self.rejected_before_dispatch, bool):
            raise TypeError("rejected_before_dispatch must be a bool")
        if self.rejected_before_dispatch and (
            self.kind != "tool_result" or not self.failed or not self.call_id or not self.tool_id
        ):
            raise ValueError("pre-dispatch rejection requires a failed, identified tool result")
        object.__setattr__(self, "raw", _plain_json_mapping(self.raw, name="raw"))


@dataclass(frozen=True, slots=True)
class ExternalRunOutcome:
    """Everything one external agent invocation is known to have produced."""

    final_text: str
    events: tuple[ExternalEvent, ...] = ()
    termination_reason: str = "final"
    usage: Mapping[str, Any] = field(default_factory=dict)
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.final_text, str):
            raise TypeError("final_text must be a string")
        events = tuple(self.events)
        if any(not isinstance(event, ExternalEvent) for event in events):
            raise TypeError("events must contain ExternalEvent records")
        if self.termination_reason not in TERMINATION_REASONS:
            raise ValueError(
                f"termination_reason must be one of {sorted(TERMINATION_REASONS)}, "
                f"got {self.termination_reason!r}"
            )
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "usage", _plain_json_mapping(self.usage, name="usage"))
        object.__setattr__(
            self,
            "provider_metadata",
            _plain_json_mapping(self.provider_metadata, name="provider_metadata"),
        )


class ExternalAuxiliaryError(RuntimeError):
    """An unsuccessful planning attempt with observed evidence for the search journal.

    An absent outcome is represented by response=None and empty usage/events;
    it is never projected as a successful or trainable generation.
    """

    def __init__(self, message: str, *, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = copy.deepcopy(dict(evidence))


@runtime_checkable
class ExternalAgentSession(Protocol):
    """One external agent invocation channel, owned by :class:`ExternalAgentRuntime`."""

    def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _TurnDraft:
    """The generation half of one turn plus the observations that closed it."""

    text_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[ExternalEvent] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.text_parts or self.reasoning_parts or self.tool_calls or self.observations)


def _split_turns(events: Sequence[ExternalEvent]) -> list[_TurnDraft]:
    """Group a flat event stream into ``generation -> observation`` turns.

    A turn stays open while the agent is producing (text, reasoning, tool calls)
    and while consecutive results come back.  The first generation event after an
    observation starts the next turn, so parallel tool calls answered by several
    results collapse into one turn rather than inventing generations.
    """

    drafts: list[_TurnDraft] = []
    current = _TurnDraft()
    for event in events:
        if event.kind in _GENERATION_KINDS:
            if current.observations:
                drafts.append(current)
                current = _TurnDraft()
            if event.kind == "message":
                if event.content:
                    current.text_parts.append(event.content)
            elif event.kind == "reasoning":
                if event.content:
                    current.reasoning_parts.append(event.content)
            else:
                assert event.tool_call is not None  # guaranteed by ExternalEvent
                current.tool_calls.append(event.tool_call)
        else:
            current.observations.append(event)
    if not current.empty:
        drafts.append(current)
    return drafts


def _observation(draft: _TurnDraft) -> EnvironmentObservation:
    """Project the results that closed one turn onto the canonical observation."""

    results = draft.observations
    if not results:
        return EnvironmentObservation(kind="external_agent_turn", content="")
    failed = [event for event in results if event.failed]
    content = "\n".join(event.content for event in results if event.content)
    first = results[0]
    single = len(results) == 1
    return EnvironmentObservation(
        kind="external_tool_result",
        content=content,
        call_id=first.call_id if single else None,
        tool_id=first.tool_id if single else None,
        attempted=True,
        error_stage="external_tool" if failed else None,
        error_code="external_tool_error" if failed else None,
    )


def _transition(
    draft: _TurnDraft,
    *,
    terminal: bool,
    termination_reason: str,
    agent: str,
) -> EnvironmentTransition:
    raw_events = [event.raw for event in draft.observations if event.raw]
    metadata: dict[str, Any] = {
        "external": {
            "agent": agent,
            "synthesized": True,
            "tool_results": len(draft.observations),
        }
    }
    if raw_events:
        metadata["external"]["raw"] = raw_events
    return EnvironmentTransition(
        observation=_observation(draft),
        reward=0.0,
        done=terminal,
        success=None,
        termination_reason=termination_reason if terminal else None,
        metadata=metadata,
    )


def _aggregate_environment_transitions(
    transitions: Sequence[EnvironmentTransition],
) -> EnvironmentTransition:
    """Represent parallel calls from one generation without losing any step."""

    marked = tuple(_mark_environment_transition(transition) for transition in transitions)
    if len(transitions) == 1:
        return marked[0]
    if not transitions:
        raise ValueError("at least one Environment transition is required")
    if any(transition.done for transition in marked[:-1]):
        raise RuntimeError("Environment stepped again after a terminal transition")
    last = marked[-1]
    metadata = copy.deepcopy(last.metadata)
    metadata["external_environment"] = {
        "environment_stepped": True,
        "synthesized": True,
        "aggregated_parallel_steps": len(marked),
        "transitions": [
            {
                "observation": _json_observation(transition.observation),
                "reward": transition.reward,
                "done": transition.done,
                "success": transition.success,
                "response_format_valid": transition.response_format_valid,
                "env_action_valid": transition.env_action_valid,
                "termination_reason": transition.termination_reason,
                "metadata": copy.deepcopy(original.metadata),
            }
            for transition, original in zip(marked, transitions, strict=True)
        ],
    }
    return EnvironmentTransition(
        observation=last.observation,
        reward=sum(transition.reward for transition in transitions),
        done=last.done,
        success=last.success,
        response_format_valid=all(transition.response_format_valid for transition in transitions),
        env_action_valid=all(transition.env_action_valid for transition in transitions),
        termination_reason=last.termination_reason,
        metadata=metadata,
    )


def _mark_environment_transition(transition: EnvironmentTransition) -> EnvironmentTransition:
    """Make a Runtime-observed step distinguishable without relying on absence."""

    metadata = copy.deepcopy(transition.metadata)
    marker = metadata.get("external_environment")
    marker = dict(marker) if isinstance(marker, Mapping) else {}
    marker.update(environment_stepped=True, synthesized=False)
    metadata["external_environment"] = marker
    return replace(transition, metadata=metadata)


def _json_observation(observation: Any) -> Any:
    renderer = getattr(observation, "to_dict", None)
    return copy.deepcopy(renderer() if callable(renderer) else observation)


def _bind_environment_transitions(
    drafts: Sequence[_TurnDraft],
    transitions: Sequence[EnvironmentTransition],
    *,
    rejected_post_termination_calls: int,
) -> tuple[list[_TurnDraft], list[EnvironmentTransition], int]:
    """Match real Environment steps to generations and reject unbridged tools."""

    remaining = tuple(transitions)
    if any(not isinstance(item, EnvironmentTransition) for item in remaining):
        raise TypeError("environment_transitions must contain EnvironmentTransition records")
    bound_drafts: list[_TurnDraft] = []
    bound_transitions: list[EnvironmentTransition] = []
    cursor = 0
    for index, draft in enumerate(drafts):
        rejections = [event for event in draft.observations if event.rejected_before_dispatch]
        rejected_ids = {event.call_id for event in rejections}
        calls_by_id = {call.id: call for call in draft.tool_calls}
        if len(rejected_ids) != len(rejections) or any(
            event.call_id not in calls_by_id or calls_by_id[event.call_id].name != event.tool_id
            for event in rejections
        ):
            raise RuntimeError("pre-dispatch rejection has no unique matching tool call")
        dispatched_calls = [call for call in draft.tool_calls if call.id not in rejected_ids]
        step_count = len(dispatched_calls) if draft.tool_calls else 1
        end = cursor + step_count
        if end > len(remaining):
            missing = end - len(remaining)
            terminal_prefix = bool(remaining[cursor:] and remaining[-1].done)
            if not terminal_prefix or missing > rejected_post_termination_calls:
                raise RuntimeError(
                    "external agent reported tool activity that did not step the "
                    "Runtime-owned Environment"
                )
            end = len(remaining)
        step_transitions = remaining[cursor:end]
        environment_tool_ids = [_environment_tool_id(item) for item in step_transitions]
        if draft.tool_calls:
            reported_tool_ids = [_unprefix_tool_id(call.name) for call in dispatched_calls]
            stepped = Counter(tool_id or "" for tool_id in environment_tool_ids)
            reported = Counter(reported_tool_ids)
            complete_generation = len(step_transitions) == len(dispatched_calls)
            if (complete_generation and stepped != reported) or (stepped - reported):
                raise RuntimeError(
                    "external agent reported tool activity that did not match the "
                    "Runtime-owned Environment steps"
                )
        elif any(tool_id is not None for tool_id in environment_tool_ids):
            raise RuntimeError(
                "Runtime-owned Environment tool step has no matching external tool call"
            )
        if step_transitions:
            transition = _aggregate_environment_transitions(step_transitions)
        else:
            # Preserve the failed attempt without inventing an Environment step.
            transition = EnvironmentTransition(
                observation=replace(_observation(draft), error_stage="external_tool_validation"),
                reward=0.0,
                done=False,
                success=None,
                env_action_valid=False,
                metadata={
                    "external_environment": {"environment_stepped": False, "synthesized": True}
                },
            )
        if rejections:
            metadata = copy.deepcopy(transition.metadata)
            metadata["external_environment"]["pre_dispatch_rejections"] = [
                {
                    "call_id": event.call_id,
                    "tool_id": event.tool_id,
                    "content": event.content,
                    "raw": event.raw,
                }
                for event in rejections
            ]
            transition = replace(transition, metadata=metadata, env_action_valid=False)
        bound_drafts.append(draft)
        bound_transitions.append(transition)
        cursor = end
        if transition.done:
            if cursor != len(remaining):
                raise RuntimeError("Environment produced transitions after termination")
            return bound_drafts, bound_transitions, len(drafts) - index - 1
    if cursor != len(remaining):
        raise RuntimeError(
            "Runtime-owned Environment stepped without a matching external agent turn"
        )
    return bound_drafts, bound_transitions, 0


def _environment_tool_id(transition: EnvironmentTransition) -> str | None:
    request = transition.metadata.get("tool_request")
    if not isinstance(request, Mapping):
        return None
    tool_id = request.get("tool_id")
    return tool_id if isinstance(tool_id, str) else None


def _unprefix_tool_id(tool_id: str) -> str:
    """Remove the vendor-specific MCP server prefix from a canonical tool id."""

    return tool_id.rsplit("__", 1)[-1].rsplit(".", 1)[-1]


def _continuation_messages(draft: _TurnDraft) -> list[dict[str, Any]]:
    """Reconstruct the transcript this turn added, for the next turn's request."""

    messages: list[dict[str, Any]] = []
    text = "".join(draft.text_parts)
    if text or draft.tool_calls:
        assistant: dict[str, Any] = {"role": "assistant", "content": text}
        if draft.tool_calls:
            assistant["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in draft.tool_calls
            ]
        messages.append(assistant)
    for event in draft.observations:
        message: dict[str, Any] = {"role": "tool", "content": event.content}
        if event.call_id is not None:
            message["tool_call_id"] = event.call_id
        if event.tool_id is not None:
            message["name"] = event.tool_id
        messages.append(message)
    return messages


def project_outcome(
    task: AgentTask,
    outcome: ExternalRunOutcome,
    *,
    agent: str,
    model: str,
    sampling: SamplingOptions,
    environment_transitions: Sequence[EnvironmentTransition] | None = None,
    environment_init: EnvironmentInitialization | None = None,
    environment_mode: str = "tool_loop",
    rejected_post_termination_calls: int = 0,
) -> AgentResult:
    """Project one external outcome onto the canonical Runtime trajectory records.

    Pure and separately testable: no subprocess, filesystem, or session state.
    """

    if not isinstance(task, AgentTask):
        raise TypeError("task must be an AgentTask")
    if not isinstance(outcome, ExternalRunOutcome):
        raise TypeError("outcome must be an ExternalRunOutcome")
    if not isinstance(sampling, SamplingOptions):
        raise TypeError("sampling must be a SamplingOptions")
    if (
        isinstance(rejected_post_termination_calls, bool)
        or not isinstance(rejected_post_termination_calls, int)
        or rejected_post_termination_calls < 0
    ):
        raise ValueError("rejected_post_termination_calls must be a non-negative integer")

    _validate_environment_mode(environment_mode)
    drafts = _split_turns(outcome.events)
    if environment_transitions is not None and not drafts:
        drafts = [_TurnDraft(text_parts=[outcome.final_text])]
    bound_transitions: list[EnvironmentTransition] | None = None
    discarded_turns = 0
    if environment_mode == "final_response":
        if environment_transitions is None or len(environment_transitions) != 1:
            raise ValueError("final_response requires exactly one Environment transition")
        final_transition = environment_transitions[0]
        if not isinstance(final_transition, EnvironmentTransition):
            raise TypeError("environment_transitions must contain EnvironmentTransition records")
        # Bind only the actual final answer to evaluation. Native tool activity
        # retains synthesized zero-reward transitions, even in a tool-ended stream.
        if not drafts or (
            drafts[-1].tool_calls
            or drafts[-1].observations
            or not drafts[-1].text_parts
            or "".join(drafts[-1].text_parts) != outcome.final_text
        ):
            drafts.append(_TurnDraft(text_parts=[outcome.final_text]))
        bound_transitions = [
            _transition(
                draft, terminal=False, termination_reason=outcome.termination_reason, agent=agent
            )
            for draft in drafts[:-1]
        ]
        bound_transitions.append(_mark_environment_transition(final_transition))
    elif environment_transitions is not None:
        drafts, bound_transitions, discarded_turns = _bind_environment_transitions(
            drafts,
            environment_transitions,
            rejected_post_termination_calls=rejected_post_termination_calls,
        )
    termination_reason = outcome.termination_reason
    final_text = outcome.final_text
    if bound_transitions and bound_transitions[-1].done:
        if outcome.termination_reason == "final" or discarded_turns:
            termination_reason = bound_transitions[-1].termination_reason or termination_reason
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": task.system},
        {"role": "user", "content": task.prompt},
    ]
    # The external agent owns its own context; this transcript is reconstructed
    # from the events we observed, which is why the request says so explicitly.
    provider_options = {
        "external_runtime": agent,
        "sampling_controlled_by": "external_agent",
        "messages_reconstructed": True,
        **outcome.provider_metadata,
    }

    turns: list[AgentTurn] = []
    for index, draft in enumerate(drafts):
        terminal = index == len(drafts) - 1
        request = GenerationRequest(
            request_id=f"{task.task_id}:{index}",
            model=model,
            messages=tuple(copy.deepcopy(message) for message in messages),
            sampling=sampling,
            group_id=task.task_id,
            sample_id=task.sample_id,
            provider_options=copy.deepcopy(provider_options),
            routing_key=task.routing_key,
        )
        response = GenerationResponse(
            request_id=request.request_id,
            content="".join(draft.text_parts),
            group_id=request.group_id,
            sample_id=request.sample_id,
            reasoning_content="".join(draft.reasoning_parts) or None,
            finish_reason=termination_reason if terminal else None,
            tool_calls=tuple(draft.tool_calls),
            # Usage is reported once per invocation, so attribute it to the turn
            # that carries the terminal transition instead of duplicating it.
            usage=copy.deepcopy(dict(outcome.usage)) if terminal else {},
            backend_metadata={"agent": agent, **copy.deepcopy(dict(outcome.provider_metadata))},
            prompt_token_ids=None,
            response_token_ids=None,
            response_logprobs=None,
            provenance=None,
        )
        turns.append(
            AgentTurn(
                index=index,
                generation_request=request,
                generation_response=response,
                environment_transition=(
                    bound_transitions[index]
                    if bound_transitions is not None
                    else _transition(
                        draft,
                        terminal=terminal,
                        termination_reason=outcome.termination_reason,
                        agent=agent,
                    )
                ),
            )
        )
        messages.extend(_continuation_messages(draft))

    metadata = dict(task.metadata)
    metadata.update(
        policy_source=POLICY_SOURCE_EXTERNAL_AGENT,
        trainable=False,
        external={"agent": agent, "model": model, **copy.deepcopy(dict(outcome.provider_metadata))},
    )
    if environment_mode == "final_response":
        metadata["external"].update(
            instruction_delivery="user_context",
            events=[asdict(event) for event in outcome.events],
        )
    if environment_init is not None and environment_init.metadata:
        metadata["environment_init"] = copy.deepcopy(dict(environment_init.metadata))
    if bound_transitions is not None:
        metadata["environment"] = {
            "stepped": True,
            "transitions": len(environment_transitions or ()),
            "discarded_post_termination_turns": discarded_turns,
            "rejected_post_termination_tool_calls": rejected_post_termination_calls,
            "bounded_after_termination": bool(discarded_turns or rejected_post_termination_calls),
        }
    if environment_mode == "final_response":
        metadata["environment"]["mode"] = environment_mode
    return AgentResult(
        task_id=task.task_id,
        final_text=final_text,
        turns=tuple(turns),
        termination_reason=termination_reason,
        metadata=metadata,
    )


def _validate_environment_mode(mode: str) -> None:
    if mode not in ("tool_loop", "final_response"):
        raise ValueError("environment_mode must be 'tool_loop' or 'final_response'")


def _safe_directory_name(task_id: str, index: int) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in task_id)
    return f"{index:04d}-{cleaned[:80] or 'task'}"


def workflow_workspace_path(
    root: Path | str,
    *,
    workflow_name: str,
    input_id: str,
    branch_index: int,
) -> Path:
    """Return the stable isolated workspace for one Workflow input branch."""

    for name, value in (("workflow_name", workflow_name), ("input_id", input_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if isinstance(branch_index, bool) or not isinstance(branch_index, int) or branch_index < 0:
        raise ValueError("branch_index must be a non-negative int")
    identity = f"{workflow_name}\0{input_id}\0{branch_index}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    readable = "-".join(
        part
        for part in (
            _safe_path_component(workflow_name, limit=32),
            _safe_path_component(input_id, limit=48),
            f"branch-{branch_index}",
        )
        if part
    )
    return Path(root) / f"{readable}-{digest}"


def _safe_path_component(value: str, *, limit: int) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    return cleaned[:limit].strip("-")


def _workflow_workspace_identity(task: AgentTask) -> tuple[str, str, int]:
    metadata = task.metadata
    workflow_name = metadata.get("workflow_name")
    input_id = metadata.get("workflow_input_id")
    branch_index = metadata.get("branch_index")
    if not isinstance(workflow_name, str) or not workflow_name.strip():
        raise ValueError(
            "reuse_workspace requires AgentTask.metadata.workflow_name to be a non-empty string"
        )
    if not isinstance(input_id, str) or not input_id.strip():
        raise ValueError(
            "reuse_workspace requires AgentTask.metadata.workflow_input_id to be a non-empty string"
        )
    if isinstance(branch_index, bool) or not isinstance(branch_index, int) or branch_index < 0:
        raise ValueError(
            "reuse_workspace requires AgentTask.metadata.branch_index to be a non-negative int"
        )
    return workflow_name, input_id, branch_index


class ExternalAgentRuntime(AgentRuntime):
    """Run tasks through isolated or explicitly continued external sessions.

    Unlike :class:`AlphaApolloAgentRuntime`, ``run_batch`` is bounded concurrency
    rather than a shared forward pass: each task is an independent subprocess or
    remote session, so the batch exists only to preserve order and identity.
    Failures propagate (fail-fast), matching the AlphaApollo Runtime; per-cell
    tolerance belongs to the evaluation application that already reruns cells.
    """

    def __init__(
        self,
        session_factory: Callable[[], ExternalAgentSession],
        *,
        agent: str,
        model: str,
        workspace_root: Path | str | None = None,
        keep_workspaces: bool = False,
        reuse_workspace: bool = False,
        reuse_session: bool = False,
        concurrency: int = 1,
        environment_factory: Callable[[AgentTask], Any] | None = None,
        environment_mode: str = "tool_loop",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        workspace_prepare: Callable[[AgentTask, Path], None] | None = None,
        workspace_observe: Callable[[AgentTask, Path], None] | None = None,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        if environment_factory is not None and not callable(environment_factory):
            raise TypeError("environment_factory must be callable when provided")
        _validate_environment_mode(environment_mode)
        if environment_mode == "final_response" and environment_factory is None:
            raise ValueError("final_response requires environment_factory")
        for name, value in (("agent", agent), ("model", model)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("concurrency must be at least one")
        if not isinstance(keep_workspaces, bool):
            raise TypeError("keep_workspaces must be a bool")
        if not isinstance(reuse_workspace, bool):
            raise TypeError("reuse_workspace must be a bool")
        if not isinstance(reuse_session, bool):
            raise TypeError("reuse_session must be a bool")
        if reuse_session and not reuse_workspace:
            raise ValueError("reuse_session requires reuse_workspace")
        for name, hook in (
            ("workspace_prepare", workspace_prepare),
            ("workspace_observe", workspace_observe),
        ):
            if hook is not None and not callable(hook):
                raise TypeError(f"{name} must be callable when provided")
        self._session_factory = session_factory
        self._agent = agent
        self._model = model
        self._workspace_root = Path(workspace_root) if workspace_root is not None else None
        self._keep_workspaces = keep_workspaces
        self._reuse_workspace = reuse_workspace
        self._reuse_session = reuse_session
        self._concurrency = concurrency
        self._environment_factory = environment_factory
        self._environment_mode = environment_mode
        self._workspace_prepare = workspace_prepare
        self._workspace_observe = workspace_observe
        # Declared, not enforced: an external agent decides its own decoding, so
        # this records what AlphaApollo asked for, never what it guarantees.
        self._sampling = SamplingOptions(
            temperature=float(temperature), max_tokens=max_tokens, top_p=float(top_p)
        )
        self._closed = False
        self._workspace_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._reused_root: Path | None = None
        self._cleanup_reused_root: Callable[[], None] | None = None
        self._reused_workspaces: set[Path] = set()
        self._active_reused_workspaces: set[Path] = set()
        self._reused_sessions: dict[tuple[str, str, int], ExternalAgentSession] = {}

    def configure_workspace_hooks(
        self,
        *,
        prepare: Callable[[AgentTask, Path], None] | None,
        observe: Callable[[AgentTask, Path], None] | None,
    ) -> None:
        """Bind one composition-owned per-task workspace lifecycle before execution."""

        if self._closed:
            raise RuntimeError("ExternalAgentRuntime is closed")
        if self._workspace_prepare is not None or self._workspace_observe is not None:
            raise RuntimeError("workspace hooks are already configured")
        if prepare is not None and not callable(prepare):
            raise TypeError("prepare must be callable when provided")
        if observe is not None and not callable(observe):
            raise TypeError("observe must be callable when provided")
        self._workspace_prepare = prepare
        self._workspace_observe = observe

    def run_batch(self, tasks: Sequence[AgentTask]) -> list[AgentResult]:
        if self._closed:
            raise RuntimeError("ExternalAgentRuntime is closed")
        if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
            raise TypeError("tasks must be a sequence of AgentTask")
        materialized = tuple(tasks)
        if any(not isinstance(task, AgentTask) for task in materialized):
            raise TypeError("tasks must contain AgentTask records")
        identities = [task.task_id for task in materialized]
        if len(set(identities)) != len(identities):
            raise ValueError("tasks must have unique task_id values")
        if not materialized:
            return []
        if self._environment_factory is None and any(task.task_payload for task in materialized):
            raise ValueError(
                "task_payload requires environment_factory; an external agent session "
                "must not receive Environment-only input"
            )
        for task in materialized:
            if self._environment_mode == "final_response":
                if task.tools:
                    raise ValueError("final_response does not accept AlphaApollo task tools")
                if task.model is not None and task.model != self._model:
                    raise ValueError(
                        "task.model conflicts with the configured external runtime model"
                    )
            # The same gate DefaultEnvironment applies before a model sees an
            # observation. An external process is further from us, not closer,
            # so evaluator-only material must not reach it either.
            ensure_safe_environment_input(
                {"system": task.system, "prompt": task.prompt, "metadata": dict(task.metadata)}
            )

        workspaces, cleanup_workspaces = self._prepare_workspaces(materialized)
        sessions: list[ExternalAgentSession] = []
        environments: list[Any] = []
        environments_lock = threading.Lock()
        primary_error: BaseException | None = None
        try:
            for workspace in workspaces:
                workspace.mkdir(parents=True, exist_ok=True)

            def execute(index: int) -> AgentResult:
                task = materialized[index]
                session, retained_session = self._session_for_task(task)
                if not isinstance(session, ExternalAgentSession):
                    raise TypeError("session_factory must return an ExternalAgentSession")
                if not retained_session:
                    with self._session_lock:
                        sessions.append(session)
                session_task = replace(task, task_payload={})
                environment = None
                environment_socket = None
                environment_init = None
                rejected_post_termination_calls = 0
                execution_error: BaseException | None = None
                try:
                    if self._workspace_prepare is not None:
                        self._workspace_prepare(task, workspaces[index])
                    if self._environment_factory is not None:
                        environment = self._environment_factory(task)
                        if environment is None:
                            raise TypeError("environment_factory must return an Environment")
                        with environments_lock:
                            if any(environment is existing for existing in environments):
                                environment = None
                                raise TypeError(
                                    "environment_factory must return a distinct Environment "
                                    "for each task"
                                )
                            environments.append(environment)
                        environment_init = initialize_environment(environment, task, seed=None)
                        initial_observation = environment_init.observation
                        if initial_observation not in (None, ""):
                            if not isinstance(initial_observation, str):
                                raise TypeError(
                                    "External Environment init observation must be a string"
                                )
                            ensure_safe_environment_input(initial_observation)
                            memory_context = (
                                task.metadata.get("agent_memory_context")
                                if isinstance(task.metadata, Mapping)
                                else None
                            )
                            if isinstance(memory_context, str) and memory_context.strip():
                                initial_observation = f"{initial_observation}\n\n{memory_context}"
                            session_task = replace(
                                session_task,
                                prompt=initial_observation,
                            )
                        from alphaapollo.reasoning.runtime.external.bridge import (
                            environment_socket as socket_bridge,
                        )

                        if self._environment_mode == "tool_loop":
                            environment_socket = socket_bridge.RuntimeEnvironmentSocket(
                                environment,
                                workspaces[index] / socket_bridge.ENVIRONMENT_SOCKET_NAME,
                            )
                        else:
                            prompt = session_task.prompt
                            memory_context = task.metadata.get("agent_memory_context")
                            if (
                                isinstance(memory_context, str)
                                and memory_context.strip()
                                and memory_context not in prompt
                            ):
                                prompt = f"{prompt}\n\n{memory_context}"
                            # Empty session.system preserves the harness's native base
                            # instructions; role instructions are explicit user context.
                            session_task = replace(
                                session_task,
                                system="",
                                prompt="\n\n".join(part for part in (task.system, prompt) if part),
                            )
                    started = time.perf_counter()
                    outcome = session.run(session_task, workspace=workspaces[index])
                    runtime_seconds = time.perf_counter() - started
                    if not isinstance(outcome, ExternalRunOutcome):
                        raise TypeError(
                            f"session for task {task.task_id!r} returned "
                            f"{type(outcome).__name__}; expected ExternalRunOutcome"
                        )
                    transitions: list[EnvironmentTransition] | None = None
                    if environment is not None:
                        transitions = []
                        if environment_socket is not None:
                            environment_socket.close()
                            if environment_socket.errors:
                                raise environment_socket.errors[0]
                            rejected_post_termination_calls = (
                                environment_socket.rejected_post_termination_calls
                            )
                            transitions = list(environment_socket.transitions)
                        drafts = _split_turns(outcome.events)
                        tool_ended = bool(drafts and drafts[-1].tool_calls)
                        # A CLI can stop at a tool result (budget, timeout or
                        # normal exit). Its final_text may still hold prose from
                        # before that call; it is not another observed action.
                        needs_final_step = (
                            self._environment_mode == "final_response" or not tool_ended
                        )
                        if needs_final_step and (not transitions or not transitions[-1].done):
                            response = GenerationResponse(
                                request_id=f"{session_task.task_id}:final",
                                content=outcome.final_text,
                                group_id=session_task.task_id,
                                sample_id=session_task.sample_id,
                            )
                            projector = getattr(environment, "project_response", None)
                            action = (
                                projector(response) if callable(projector) else outcome.final_text
                            )
                            transitions.append(
                                socket_bridge.normalize_environment_transition(
                                    environment.step(action)
                                )
                            )
                    result = project_outcome(
                        session_task,
                        outcome,
                        agent=self._agent,
                        model=self._model,
                        sampling=self._sampling,
                        environment_transitions=transitions,
                        environment_init=environment_init,
                        environment_mode=self._environment_mode,
                        rejected_post_termination_calls=rejected_post_termination_calls,
                    )
                    result = replace(
                        result,
                        metadata={**dict(result.metadata), "runtime_seconds": runtime_seconds},
                    )
                    if self._workspace_observe is not None:
                        self._workspace_observe(task, workspaces[index])
                    return result
                except BaseException as exc:
                    execution_error = exc
                    raise
                finally:
                    cleanup_errors: list[BaseException] = []
                    if environment_socket is not None:
                        try:
                            environment_socket.close()
                        except BaseException as exc:  # noqa: BLE001 - preserve primary error
                            cleanup_errors.append(exc)
                    bridge_stopped = environment_socket is None or not environment_socket.is_alive
                    if environment is not None and bridge_stopped:
                        close = getattr(environment, "close", None)
                        if callable(close):
                            try:
                                close()
                            except BaseException as exc:  # noqa: BLE001 - preserve primary error
                                cleanup_errors.append(exc)
                    if cleanup_errors and execution_error is None:
                        raise cleanup_errors[0]

            if self._concurrency == 1:
                return [execute(index) for index in range(len(materialized))]
            with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
                return list(pool.map(execute, range(len(materialized))))
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            errors = _close_all(sessions)
            cleanup_workspaces()
            # A cleanup failure must never mask the failure that caused it.
            if errors and primary_error is None:
                raise errors[0]

    def generate_auxiliary(
        self,
        *,
        request_id: str,
        system: str,
        prompt: str,
        response_format: Mapping[str, Any] | None = None,
    ) -> GenerationResponse:
        """Run controller planning through the same harness without a candidate episode.

        Native base instructions remain in force. Structured output is requested
        in user context and validated locally; no provider decoding guarantee is
        implied. Auxiliary workspaces never reuse a candidate branch or its hooks.
        """

        if self._closed:
            raise RuntimeError("ExternalAgentRuntime is closed")
        for name, value in (("request_id", request_id), ("system", system), ("prompt", prompt)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        format_request, validator = _auxiliary_response_format(response_format)
        session_prompt = f"{system}\n\n{prompt}"
        if format_request is not None:
            session_prompt += (
                "\n\nReturn only the final JSON value, without Markdown fences or commentary. "
                "It must satisfy this response format (validated locally):\n"
                + json.dumps(format_request, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
        ensure_safe_environment_input(session_prompt)
        task = AgentTask(
            task_id=request_id,
            system="",
            prompt=session_prompt,
            model=self._model,
            routing_key=f"auxiliary:{request_id}",
        )
        external = {
            "agent": self._agent,
            "model": self._model,
            "configured_model": self._model,
            "instruction_delivery": "user_context",
            "events": [],
        }
        if format_request is not None:
            external["response_format"] = {
                "requested": format_request,
                "delivery": "prompt",
                "enforcement": f"local_{format_request['type']}",
                "provider_native": False,
            }
        evidence = {
            "response": None,
            "usage": {},
            "backend_metadata": {
                "policy_source": POLICY_SOURCE_EXTERNAL_AGENT,
                "trainable": False,
                "external": external,
            },
        }
        root, cleanup_root = self._workspace_base()
        workspace: Path | None = None
        sessions: list[ExternalAgentSession] = []
        primary_error: BaseException | None = None
        started = time.perf_counter()
        try:
            workspace = Path(tempfile.mkdtemp(prefix="auxiliary-", dir=root))
            session = self._session_factory()
            if not isinstance(session, ExternalAgentSession):
                raise TypeError("session_factory must return an ExternalAgentSession")
            sessions.append(session)
            started = time.perf_counter()
            outcome = session.run(task, workspace=workspace)
            evidence["backend_metadata"]["runtime_seconds"] = time.perf_counter() - started
            if not isinstance(outcome, ExternalRunOutcome):
                raise TypeError("auxiliary session must return ExternalRunOutcome")
            # Capture the paid invocation before checking its answer. Failed
            # attempts must remain journalable when the search controller retries.
            evidence["response"] = outcome.final_text
            evidence["usage"] = copy.deepcopy(dict(outcome.usage))
            evidence["backend_metadata"]["external"] = {
                **copy.deepcopy(dict(outcome.provider_metadata)),
                **external,
                "model": outcome.provider_metadata.get("model") or self._model,
                "events": [asdict(event) for event in outcome.events],
                "termination_reason": outcome.termination_reason,
            }
            if outcome.termination_reason != "final":
                raise RuntimeError(
                    f"auxiliary session did not finish: {outcome.termination_reason}"
                )
            if not outcome.final_text.strip():
                raise ValueError("auxiliary session returned an empty final response")
            if format_request is not None:
                value = _auxiliary_json(outcome.final_text)
                if format_request["type"] == "json_object" and not isinstance(value, dict):
                    raise ValueError("auxiliary response must be a JSON object")
                if validator is not None:
                    errors = validator.iter_errors(value)
                    error = next(errors, None)
                    if error is not None:
                        raise ValueError(
                            f"auxiliary response does not match JSON schema: {error.message}"
                        )
            return GenerationResponse(
                request_id=request_id,
                content=outcome.final_text,
                finish_reason=outcome.termination_reason,
                usage=evidence["usage"],
                backend_metadata=evidence["backend_metadata"],
            )
        except BaseException as exc:
            primary_error = exc
            evidence["backend_metadata"].setdefault(
                "runtime_seconds", time.perf_counter() - started
            )
            if isinstance(exc, Exception):
                raise ExternalAuxiliaryError(str(exc), evidence=evidence) from exc
            raise
        finally:
            errors = _close_all(sessions)
            if not self._keep_workspaces:
                if workspace is not None:
                    shutil.rmtree(workspace, ignore_errors=True)
                cleanup_root()
            if errors and primary_error is None:
                if not isinstance(errors[0], Exception):
                    raise errors[0]
                raise ExternalAuxiliaryError(str(errors[0]), evidence=evidence) from errors[0]

    def close(self) -> None:
        with self._workspace_lock:
            if self._closed:
                return
            if self._active_reused_workspaces:
                raise RuntimeError("cannot close ExternalAgentRuntime while a batch is active")
            self._closed = True
            if self._reuse_workspace and not self._keep_workspaces:
                for workspace in self._reused_workspaces:
                    shutil.rmtree(workspace, ignore_errors=True)
                if self._cleanup_reused_root is not None:
                    self._cleanup_reused_root()
            self._reused_workspaces.clear()
            self._active_reused_workspaces.clear()
            self._reused_root = None
            self._cleanup_reused_root = None
        with self._session_lock:
            sessions = tuple(self._reused_sessions.values())
            self._reused_sessions.clear()
        errors = _close_all(sessions)
        if errors:
            raise errors[0]

    def terminate(self, reason: str = "workflow_shutdown") -> None:
        del reason
        self.close()

    def _session_for_task(
        self,
        task: AgentTask,
    ) -> tuple[ExternalAgentSession, bool]:
        """Return one fresh session or the conversation owned by this branch."""

        if not self._reuse_session:
            return self._session_factory(), False
        identity = _workflow_workspace_identity(task)
        with self._session_lock:
            existing = self._reused_sessions.get(identity)
            if existing is not None:
                return existing, True
            session = self._session_factory()
            if not isinstance(session, ExternalAgentSession):
                raise TypeError("session_factory must return an ExternalAgentSession")
            if getattr(session, "supports_continuation", False) is not True:
                try:
                    session.close()
                finally:
                    raise TypeError(
                        "reuse_session requires an external session with verified "
                        "conversation continuation support"
                    )
            self._reused_sessions[identity] = session
            return session, True

    def _workspace_base(self) -> tuple[Path, Callable[[], None]]:
        if self._workspace_root is not None:
            root = self._workspace_root
            root.mkdir(parents=True, exist_ok=True)
            return root, lambda: None
        temporary = tempfile.mkdtemp(prefix="alphaapollo-external-")
        return Path(temporary), lambda: shutil.rmtree(temporary, ignore_errors=True)

    def _prepare_workspaces(
        self,
        tasks: Sequence[AgentTask],
    ) -> tuple[list[Path], Callable[[], None]]:
        if not self._reuse_workspace:
            root, cleanup_root = self._workspace_base()
            workspaces: list[Path] = []
            try:
                for index, task in enumerate(tasks):
                    prefix = f"{_safe_directory_name(task.task_id, index)}-"
                    workspaces.append(Path(tempfile.mkdtemp(prefix=prefix, dir=root)))
            except BaseException:
                for workspace in workspaces:
                    shutil.rmtree(workspace, ignore_errors=True)
                cleanup_root()
                raise

            def cleanup() -> None:
                if self._keep_workspaces:
                    return
                for workspace in workspaces:
                    shutil.rmtree(workspace, ignore_errors=True)
                cleanup_root()

            return workspaces, cleanup

        identities = [_workflow_workspace_identity(task) for task in tasks]
        if len(set(identities)) != len(identities):
            raise ValueError(
                "reuse_workspace requires unique workflow_input_id and branch_index "
                "pairs within one batch"
            )
        with self._workspace_lock:
            if self._reused_root is None:
                self._reused_root, self._cleanup_reused_root = self._workspace_base()
            root = self._reused_root
            workspaces = [
                workflow_workspace_path(
                    root,
                    workflow_name=workflow_name,
                    input_id=input_id,
                    branch_index=branch_index,
                )
                for workflow_name, input_id, branch_index in identities
            ]
            overlapping = sorted(set(workspaces) & self._active_reused_workspaces)
            if overlapping:
                raise RuntimeError(
                    "reuse_workspace cannot run the same Workflow input branch concurrently: "
                    f"{overlapping[0]}"
                )
            self._reused_workspaces.update(workspaces)
            self._active_reused_workspaces.update(workspaces)

        def release() -> None:
            with self._workspace_lock:
                self._active_reused_workspaces.difference_update(workspaces)

        return workspaces, release


def _auxiliary_response_format(
    response_format: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, Any]:
    """Validate the supported prompt/local response contract before starting a harness."""

    if response_format is None:
        return None, None
    requested = _plain_json_mapping(response_format, name="response_format")
    kind = requested.get("type")
    if kind == "json_object" and set(requested) == {"type"}:
        return requested, None
    if kind != "json_schema" or set(requested) != {"type", "json_schema"}:
        raise ValueError("response_format must be json_object or json_schema")
    spec = requested["json_schema"]
    if (
        not isinstance(spec, Mapping)
        or set(spec) - {"name", "strict", "schema", "description"}
        or not isinstance(spec.get("name"), str)
        or not spec["name"].strip()
        or not isinstance(spec.get("schema"), Mapping)
        or ("strict" in spec and not isinstance(spec["strict"], bool))
        or ("description" in spec and not isinstance(spec["description"], str))
    ):
        raise ValueError("response_format.json_schema requires a name and schema")
    try:
        json.dumps(requested, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("response_format must contain valid JSON values") from exc

    def check_references(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                    not isinstance(child, str) or not child.startswith("#")
                ):
                    raise ValueError("response_format only supports local JSON schema references")
                check_references(child)
        elif isinstance(value, list):
            for child in value:
                check_references(child)

    check_references(spec["schema"])
    try:
        from jsonschema.exceptions import SchemaError
        from jsonschema.validators import validator_for
    except ImportError as exc:
        raise ImportError(
            "JSON schema auxiliary planning requires AlphaApollo[json-schema]"
        ) from exc
    validator_type = validator_for(spec["schema"], default=None)
    if validator_type is None:
        if "$schema" in spec["schema"]:
            raise ValueError("response_format uses an unsupported JSON schema dialect")
        validator_type = validator_for(spec["schema"])
    try:
        validator_type.check_schema(spec["schema"])
    except SchemaError as exc:
        raise ValueError(f"response_format contains an invalid JSON schema: {exc.message}") from exc
    return requested, validator_type(spec["schema"])


def _auxiliary_json(text: str) -> Any:
    """Parse exactly one JSON value, refusing ambiguous keys and non-JSON constants."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    try:
        return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
    except ValueError as exc:
        raise ValueError(f"auxiliary final response must be valid JSON: {exc}") from exc


def _close_all(sessions: Sequence[ExternalAgentSession]) -> list[BaseException]:
    errors: list[BaseException] = []
    for session in sessions:
        try:
            session.close()
        except BaseException as exc:  # noqa: BLE001 - report every close failure
            errors.append(exc)
    return errors


def encode_arguments(value: Any) -> str:
    """Encode structured tool arguments the way ``ToolCall.arguments`` expects."""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def flatten_content_blocks(content: Any) -> str:
    """Flatten MCP-style content blocks into the text an observation carries.

    Claude Code tool results and Codex MCP tool results share this shape, so a
    non-text block is preserved as JSON instead of being dropped.
    """

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(json.dumps(copy.deepcopy(block), ensure_ascii=False, default=str))
        return "\n".join(parts)
    return str(content)
