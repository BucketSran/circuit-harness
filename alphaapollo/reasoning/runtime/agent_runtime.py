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

"""Stable contracts for complete Reasoning agent trajectories.

An :class:`AgentRuntime` owns the model-to-Environment loop for one configured
agent role.  Its primary operation is batched: one input task produces one
ordered :class:`AgentResult`, and every :class:`AgentTurn` retains the complete
single-generation request/response and the Environment transition that followed
it.  Durable persistence is deliberately outside this in-process contract.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from alphaapollo.common.trajectory.episode import (
    EpisodeTurn,
    validate_ordered_episode_turns,
)
from alphaapollo.reasoning._immutable import _freeze_json_mapping

__all__ = [
    "AgentResult",
    "AgentRuntime",
    "AgentTask",
    "AgentTurn",
]


# The one ``GenerationResponse.finish_reason`` that means "the sampler stopped
# this reply at ``max_tokens``" rather than "the model chose to stop".  Both
# shipped Generation backends normalize onto it: the OpenAI-compatible backend
# passes the provider's value through, and the verl server maps its
# ``stop_reason``.  ``ExternalAgentRuntime`` synthesizes a ``finish_reason``
# from its own episode-termination vocabulary instead and so never matches --
# correctly, because that runtime does not own the sampler and cannot observe a
# token ceiling it never set.
_TRUNCATED_FINISH_REASON = "length"


def _require_non_negative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_transition_shape(value: Any) -> None:
    """Validate the stable transition facts retained by a public ``AgentTurn``."""

    required = ("observation", "reward", "done", "termination_reason", "metadata")
    if any(not hasattr(value, name) for name in required):
        raise TypeError("environment_transition must satisfy the EnvironmentTransition shape")
    if not isinstance(value.done, bool):
        raise TypeError("environment_transition.done must be a bool")
    reward = value.reward
    if isinstance(reward, (bool, str, bytes)):
        raise TypeError("environment_transition.reward must be a finite number")
    try:
        normalized_reward = float(reward)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("environment_transition.reward must be a finite number") from exc
    if not math.isfinite(normalized_reward):
        raise ValueError("environment_transition.reward must be finite")
    reason = value.termination_reason
    if value.done:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("terminal environment transitions require a termination reason")
    elif reason is not None:
        raise ValueError("non-terminal environment transitions cannot have a termination reason")
    if not isinstance(value.metadata, Mapping):
        raise TypeError("environment_transition.metadata must be a mapping")


@dataclass(frozen=True, slots=True)
class AgentTask:
    """One role invocation with isolated Environment-only input.

    ``task_id`` is the stable identity echoed by :class:`AgentResult`. ``system``
    and ``prompt`` are the initial chat turns. Model routing, Workflow branch,
    round, sample, seed, and granted tool names are explicit policy fields; an
    empty ``tools`` tuple grants no tools, regardless of which schemas the Runtime
    has available. ``metadata`` is public invocation attribution. ``task_payload``
    is the isolated, recursively immutable input owned by the Runtime's
    Environment; it is never model input or result metadata. Environment slot
    and seed select the Learning-owned episode without exposing private payloads.
    """

    task_id: str
    system: str
    prompt: str
    model: str | None = None
    routing_key: str = ""
    tools: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    branch_id: str = "main"
    round_index: int = 0
    sample_id: int = 0
    sampling_seed: int | None = None
    environment_slot: int = 0
    environment_seed: int | None = None
    task_payload: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        for name in ("system", "prompt"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("model must be a non-empty string when provided")
        if not isinstance(self.routing_key, str):
            raise TypeError("routing_key must be a string")
        if not isinstance(self.branch_id, str) or not self.branch_id.strip():
            raise ValueError("branch_id must be a non-empty string")
        _require_non_negative_int(self.round_index, name="round_index")
        _require_non_negative_int(self.sample_id, name="sample_id")
        if self.sampling_seed is not None and (
            isinstance(self.sampling_seed, bool) or not isinstance(self.sampling_seed, int)
        ):
            raise TypeError("sampling_seed must be an integer when provided")
        _require_non_negative_int(self.environment_slot, name="environment_slot")
        if self.environment_seed is not None and (
            isinstance(self.environment_seed, bool) or not isinstance(self.environment_seed, int)
        ):
            raise TypeError("environment_seed must be an integer when provided")
        if isinstance(self.tools, (str, bytes)) or not isinstance(self.tools, Sequence):
            raise TypeError("tools must be a sequence of names")
        tools = tuple(self.tools)
        if any(not isinstance(tool, str) or not tool.strip() for tool in tools):
            raise ValueError("tools must contain non-empty strings")
        if len(set(tools)) != len(tools):
            raise ValueError("tools must not contain duplicate names")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if not isinstance(self.task_payload, Mapping):
            raise TypeError("task_payload must be a mapping")
        metadata = dict(self.metadata)
        task_payload = self.task_payload
        if "task_payload" in metadata:
            legacy_payload = metadata.pop("task_payload")
            if task_payload:
                raise ValueError(
                    "task_payload cannot be provided both explicitly and through metadata"
                )
            if not isinstance(legacy_payload, Mapping):
                raise TypeError("metadata task_payload must be a mapping")
            task_payload = legacy_payload
        object.__setattr__(self, "tools", tools)
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_mapping(metadata, where="metadata"),
        )
        object.__setattr__(
            self,
            "task_payload",
            _freeze_json_mapping(task_payload, where="task_payload"),
        )


@dataclass(frozen=True, slots=True)
class AgentTurn(EpisodeTurn):
    """One complete ``Generation -> Environment`` trajectory step.

    The request and response are intentionally retained as the concrete objects
    exchanged with the injected Generation backend.  This avoids reducing token
    traces, tool calls, provider metadata, or future backend-specific provenance
    to a lossy text-only record.  ``environment_transition`` is normalized to the
    canonical current Common value even when a legacy ``Env`` was adapted.
    """

    def __post_init__(self) -> None:
        EpisodeTurn.__post_init__(self)
        _validate_transition_shape(self.environment_transition)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """One task's terminal outcome, immutable metadata, and ordered trajectory."""

    task_id: str
    final_text: str
    turns: tuple[AgentTurn, ...] = ()
    termination_reason: str = "final"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    initial_observation: Any = None
    # Whether the reply that produced ``final_text`` was cut off at the
    # sampler's ``max_tokens``, rather than the model choosing to stop.
    #
    # This is a second fact beside ``termination_reason``, never a replacement
    # for it. ``termination_reason`` is the Environment's answer to "is this
    # episode over"; the Environment is handed the model's text as an action and
    # never sees a ``GenerationResponse``, so a reply cut mid-sentence that still
    # looks answer-shaped ends the episode ``completed``/``final``. Both can be
    # true at once -- an episode stopped at ``max_steps`` whose last generation
    # was also truncated -- so collapsing them into one value would lose one.
    #
    # Derived, never supplied. ``init=False`` leaves exactly one authority for
    # the fact: the turns this result already carries. No producer can record a
    # truncation flag that its own trajectory contradicts, and no Runtime has to
    # remember to set it.
    #
    # It reflects the LAST turn. That is the turn whose text became
    # ``final_text``, and ``final_text`` is what every consumer downstream reads:
    # answer extraction, verdict parsing, and scoring see no other turn. A turn
    # cut off earlier in the loop is real evidence that ``max_tokens`` is too
    # low, but it did not damage what was recorded, and it stays visible
    # per-turn in the trajectory projection, which persists each turn's
    # ``finish_reason`` verbatim. Widening this to "some turn was cut off" would
    # make it unable to answer the question its consumers actually ask.
    output_truncated: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.final_text, str):
            raise TypeError("final_text must be a string")
        turns = validate_ordered_episode_turns(self.turns)
        if any(not isinstance(turn, AgentTurn) for turn in turns):
            raise TypeError("turns must contain AgentTurn records")
        if not isinstance(self.termination_reason, str) or not self.termination_reason.strip():
            raise ValueError("termination_reason must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "turns", turns)
        object.__setattr__(
            self,
            "output_truncated",
            bool(turns)
            and getattr(turns[-1].generation_response, "finish_reason", None)
            == _TRUNCATED_FINISH_REASON,
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_mapping(self.metadata, where="metadata"),
        )
        object.__setattr__(self, "initial_observation", copy.deepcopy(self.initial_observation))


class AgentRuntime(ABC):
    """Interface implemented by AlphaApollo-managed and external agent runtimes."""

    @abstractmethod
    def run_batch(self, tasks: Sequence[AgentTask]) -> list[AgentResult]:
        """Run one complete trajectory per task, preserving input order."""

    def run(self, task: AgentTask) -> AgentResult:
        """Batch-size-one convenience operation with the same validation semantics."""

        if not isinstance(task, AgentTask):
            raise TypeError("task must be AgentTask")
        results = self.run_batch((task,))
        if len(results) != 1:
            raise RuntimeError(
                f"run_batch returned {len(results)} results for one task; expected exactly one"
            )
        result = results[0]
        if not isinstance(result, AgentResult):
            raise TypeError("run_batch must return AgentResult records")
        if result.task_id != task.task_id:
            raise RuntimeError(
                f"run_batch returned task_id {result.task_id!r}; expected {task.task_id!r}"
            )
        return result
