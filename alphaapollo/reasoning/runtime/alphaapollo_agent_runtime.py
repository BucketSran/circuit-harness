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

"""Batched AlphaApollo model-to-Environment trajectory runtime.

One turn is exactly one ``GenerationBackend.generate_batch`` call for the
currently active slots, followed by one masked Environment batch step.  The
requests are Common's canonical :class:`GenerationRequest`/
:class:`SamplingOptions` records, so a strict Common backend never receives a
private duck type.

Generation responses are deliberately opaque to action projection.  An
Environment-owned :class:`EnvironmentProjector` converts a response to an
action and supplies the messages for a following turn.  This module never
decodes, selects, or rewrites tool calls.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Protocol, runtime_checkable

from alphaapollo.common.generation.base import GenerationRequest, SamplingOptions
from alphaapollo.reasoning._immutable import _freeze_json_mapping
from alphaapollo.reasoning.runtime.agent_runtime import (
    AgentResult,
    AgentRuntime,
    AgentTask,
    AgentTurn,
)
from alphaapollo.reasoning.runtime.environment_lifecycle import (
    EnvironmentInitialization,
    effective_task_seed,
    invoke_environment_init,
    normalize_environment_init,
)
from alphaapollo.reasoning.runtime.trajectory_slot import _TrajectorySlot

__all__ = ["AlphaApolloAgentRuntime", "EnvironmentProjector"]

# Preserve the existing private test/debugging name while shared lifecycle code
# owns the implementation.
_EnvironmentInit = EnvironmentInitialization
_normalize_init = normalize_environment_init

_IMAGE_BLOCK_TYPES = frozenset({"image_url", "input_image", "image"})
_OMITTED_IMAGES_TEXT = "[earlier image content omitted]"


def _message_has_images(message: Mapping[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, (str, bytes)) or not isinstance(content, Sequence):
        return False
    return any(
        isinstance(block, Mapping) and block.get("type") in _IMAGE_BLOCK_TYPES for block in content
    )


def _strip_image_blocks(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    assert isinstance(content, Sequence) and not isinstance(content, (str, bytes))
    # Responses pairs input_image with input_text; Chat-style messages use text.
    placeholder_type = (
        "input_text"
        if any(
            isinstance(block, Mapping) and block.get("type") == "input_image" for block in content
        )
        else "text"
    )
    kept = [
        block
        for block in content
        if not (isinstance(block, Mapping) and block.get("type") in _IMAGE_BLOCK_TYPES)
    ]
    kept.append({"type": placeholder_type, "text": _OMITTED_IMAGES_TEXT})
    message["content"] = kept
    return message


def _window_image_history(
    messages: Sequence[Mapping[str, Any]],
    keep: int | None,
) -> tuple[Mapping[str, Any], ...]:
    """Keep image blocks only in the newest ``keep`` image-bearing messages.

    Long multimodal episodes accumulate image-bearing messages, and providers
    bound the images accepted in a single request. Older messages keep their
    text (tool results, observations) and get a neutral omission placeholder;
    the stored trajectory is unaffected because this runs on the per-request
    snapshot only. ``keep=None`` preserves the historical unbounded behavior.
    """

    if keep is None:
        return tuple(messages)
    remaining = keep
    windowed: list[Mapping[str, Any]] = []
    for message in reversed(messages):
        if _message_has_images(message):
            if remaining > 0:
                remaining -= 1
            else:
                message = _strip_image_blocks(dict(message))
        windowed.append(message)
    windowed.reverse()
    return tuple(windowed)


@runtime_checkable
class _GenerationBatch(Protocol):
    """Structural subset of Common ``GenerationBackend`` consumed here."""

    def generate_batch(self, requests: Sequence[Any]) -> Sequence[Any]: ...


@runtime_checkable
class EnvironmentProjector(Protocol):
    """Environment-owned Generation interaction boundary.

    ``project_response`` may inspect structured tool calls and choose the
    Environment action.  ``continuation_messages`` owns the inverse projection
    from the response plus transition back to model input, including exact tool
    call ID echoing.  Runtime treats both return values as opaque structures.
    """

    def project_response(self, response: Any) -> Any: ...

    def continuation_messages(
        self, response: Any, transition: Any
    ) -> Sequence[Mapping[str, Any]]: ...


@runtime_checkable
class _EnvironmentBatch(Protocol):
    def init(self, tasks: Sequence[AgentTask]) -> Sequence[Any]: ...

    def step(
        self,
        actions: Sequence[Any | None],
        active_mask: Sequence[bool],
    ) -> Sequence[Any | None]: ...

    def terminate(self, active_mask: Sequence[bool], reason: str) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class _EnvironmentProjectionSource(Protocol):
    """Optional batch hook for selecting a slot's Environment projector."""

    def projector(self, slot_index: int) -> EnvironmentProjector: ...


@dataclass(frozen=True, slots=True)
class _EnvironmentTransition:
    """Canonical lossless-enough projection of a legacy ``Env`` mapping."""

    observation: Any
    reward: float
    done: bool
    success: bool | None = None
    response_format_valid: bool = True
    env_action_valid: bool = True
    termination_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    previous_observation: Any | None = None
    raw_observation: Any | None = None
    executed_action: Any | None = None

    @property
    def action_valid(self) -> bool:
        return self.response_format_valid and self.env_action_valid

    def __post_init__(self) -> None:
        if not isinstance(self.done, bool):
            raise TypeError("Environment transition done must be a bool")
        reward = _transition_reward(self.reward)
        if self.success is not None and not isinstance(self.success, bool):
            raise TypeError("Environment transition success must be a bool when provided")
        if not self.done and self.success is not None:
            raise ValueError("success is defined only for terminal Environment transitions")
        for name in ("response_format_valid", "env_action_valid"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Environment transition {name} must be a bool")
        if self.done:
            if not isinstance(self.termination_reason, str) or not self.termination_reason.strip():
                raise ValueError("terminal Environment transitions require a termination reason")
        elif self.termination_reason is not None:
            raise ValueError(
                "non-terminal Environment transitions cannot have a termination reason"
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("Environment transition metadata must be a mapping")
        object.__setattr__(self, "reward", reward)
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_mapping(self.metadata, where="Environment transition metadata"),
        )


def _close_best_effort(resource: Any) -> None:
    """Release a rejected resource without hiding its contract error."""

    try:
        close = getattr(resource, "close", None)
        if callable(close):
            close()
    except BaseException:
        pass


class _PerSlotEnvironmentBatch:
    """Lift homogeneous single-episode Environments to an active-mask batch."""

    def __init__(
        self,
        tasks: Sequence[AgentTask],
        factory: Callable[[AgentTask], Any],
        *,
        seed: int | None,
    ) -> None:
        self._tasks = tuple(tasks)
        self._environments: list[Any] = []
        self._initialized = [False] * len(tasks)
        self._seed = seed
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(len(tasks), 1),
            thread_name_prefix="alphaapollo-environment",
        )
        try:
            for task in tasks:
                self._environments.append(factory(task))
        except BaseException:
            _close_best_effort(self)
            raise
        if any(environment is None for environment in self._environments):
            _close_best_effort(self)
            raise TypeError("environment_factory must return an Environment")
        if len({id(environment) for environment in self._environments}) != len(self._environments):
            _close_best_effort(self)
            raise TypeError("environment_factory must return a distinct Environment for each task")
        if len({type(environment) for environment in self._environments}) > 1:
            _close_best_effort(self)
            raise TypeError("run_batch requires a homogeneous Environment batch")

    def init(self, tasks: Sequence[AgentTask]) -> list[Any]:
        if tuple(tasks) != self._tasks:
            raise RuntimeError("Environment batch initialized with different tasks")
        futures = [
            self._executor.submit(invoke_environment_init, environment, task, seed=self._seed)
            for task, environment in zip(tasks, self._environments, strict=True)
        ]
        results: list[Any | None] = [None] * len(futures)
        first_error: BaseException | None = None
        for index, future in enumerate(futures):
            try:
                raw_result = future.result()
                self._initialized[index] = True
                results[index] = normalize_environment_init(raw_result)
            except BaseException as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
        if any(result is None for result in results):
            raise RuntimeError("Environment init completed without a result")
        return [result for result in results if result is not None]

    def step(
        self,
        actions: Sequence[Any | None],
        active_mask: Sequence[bool],
    ) -> list[Any | None]:
        if len(actions) != len(self._environments) or len(active_mask) != len(self._environments):
            raise RuntimeError("Environment batch step cardinality mismatch")
        transitions: list[Any | None] = [None] * len(self._environments)
        futures: dict[int, concurrent.futures.Future[Any]] = {}
        for index, (environment, action, active) in enumerate(
            zip(self._environments, actions, active_mask, strict=True)
        ):
            if not active:
                continue
            if action is None:
                raise RuntimeError("active Environment slot received no action")
            futures[index] = self._executor.submit(environment.step, action)
        for index, future in futures.items():
            transitions[index] = future.result()
        return transitions

    def projector(self, slot_index: int) -> EnvironmentProjector:
        try:
            projector = self._environments[slot_index]
        except IndexError as exc:
            raise RuntimeError(f"invalid Environment slot index {slot_index}") from exc
        return _require_projector(projector)

    def terminate(self, active_mask: Sequence[bool], reason: str) -> None:
        if len(active_mask) != len(self._environments):
            raise RuntimeError("Environment batch terminate cardinality mismatch")
        futures: list[concurrent.futures.Future[Any]] = []
        for initialized, environment, active in zip(
            self._initialized, self._environments, active_mask, strict=True
        ):
            if not initialized or not active:
                continue
            terminate = getattr(environment, "terminate", None)
            if callable(terminate):
                futures.append(self._executor.submit(terminate, reason))
        for future in futures:
            future.result()

    def close(self) -> None:
        futures: list[concurrent.futures.Future[Any]] = []
        closed: set[int] = set()
        for environment in self._environments:
            identity = id(environment)
            if identity in closed:
                continue
            closed.add(identity)
            close = getattr(environment, "close", None)
            if not callable(close):
                continue
            futures.append(self._executor.submit(close))
        first_error: BaseException | None = None
        for future in futures:
            try:
                future.result()
            except BaseException as exc:
                first_error = first_error or exc
        self._executor.shutdown(wait=True)
        if first_error is not None:
            raise first_error


class AlphaApolloAgentRuntime(AgentRuntime):
    """Run independent slots through batched Generation and Environment steps."""

    def __init__(
        self,
        backend: Any,
        *,
        model: str,
        environment_factory: Callable[[AgentTask], Any] | None = None,
        environment_batch_factory: Callable[[Sequence[AgentTask]], Any] | None = None,
        environment_projector: EnvironmentProjector | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        max_turns: int = 6,
        seed: int | None = None,
        tools: Sequence[Mapping[str, Any]] = (),
        tool_choice: str | None = None,
        provider_options: Mapping[str, Any] | None = None,
        image_history_messages: int | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty")
        _require_finite_number(temperature, name="temperature")
        _require_finite_number(top_p, name="top_p")
        if float(temperature) < 0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be at least one")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if image_history_messages is not None and (
            isinstance(image_history_messages, bool)
            or not isinstance(image_history_messages, int)
            or image_history_messages < 1
        ):
            raise ValueError("image_history_messages must be a positive integer when provided")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer when provided")
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
            raise TypeError("tools must be a sequence of mappings")
        if any(not isinstance(tool, Mapping) for tool in tools):
            raise TypeError("tools must contain mappings")
        if tool_choice is not None and (
            not isinstance(tool_choice, str) or not tool_choice.strip()
        ):
            raise ValueError("tool_choice must be non-empty when provided")
        if tool_choice is not None and not tools:
            raise ValueError("tool_choice requires at least one tool schema")
        if provider_options is not None and not isinstance(provider_options, Mapping):
            raise TypeError("provider_options must be a mapping")
        if environment_batch_factory is not None and environment_factory is not None:
            raise TypeError("pass a per-slot or batched Environment factory, not both")
        if environment_batch_factory is None and environment_factory is None:
            raise TypeError("an Environment factory is required")
        if not isinstance(backend, _GenerationBatch):
            raise TypeError("backend must expose generate_batch()")
        if environment_projector is not None:
            _require_projector(environment_projector)

        self._generation: _GenerationBatch = backend
        self._model = model
        self._environment_batch_factory = environment_batch_factory
        self._environment_factory = environment_factory
        self._environment_projector = environment_projector
        self._sampling = SamplingOptions(
            temperature=float(temperature),
            max_tokens=max_tokens,
            top_p=float(top_p),
        )
        self._max_turns = max_turns
        self._image_history_messages = image_history_messages
        self._seed = seed
        tool_schemas = tuple(copy.deepcopy(dict(tool)) for tool in tools)
        self._tool_schemas_by_name = _index_tool_schemas(tool_schemas)
        self._tool_choice = tool_choice
        self._provider_options = copy.deepcopy(dict(provider_options or {}))
        if seed is not None and "seed" not in self._provider_options:
            self._provider_options["seed"] = seed

    def run_batch(self, tasks: Sequence[AgentTask]) -> list[AgentResult]:
        tasks = _validate_tasks(tasks)
        if not tasks:
            return []
        self._validate_task_tool_grants(tasks)
        slots = [_TrajectorySlot(task=task, position=index) for index, task in enumerate(tasks)]
        environment_batch = self._make_environment_batch(tasks)
        primary_error: BaseException | None = None
        init_completed = False
        try:
            initial = list(environment_batch.init(tasks))
            if len(initial) != len(slots):
                raise RuntimeError(
                    f"Environment batch returned {len(initial)} init results for {len(slots)} tasks"
                )
            for slot, init_result in zip(slots, initial, strict=True):
                slot.initialize(normalize_environment_init(init_result))
            init_completed = True

            for _ in range(self._max_turns):
                active_mask = [slot.active for slot in slots]
                if not any(active_mask):
                    break
                active_slots = [slot for slot in slots if slot.active]
                requests = [self._request_for(slot) for slot in active_slots]
                _validate_generation_requests(requests)
                responses = list(self._generation.generate_batch(requests))
                _validate_generation_responses(requests, responses)

                projectors: dict[int, EnvironmentProjector] = {}
                actions: list[Any | None] = [None] * len(slots)
                for slot, response in zip(active_slots, responses, strict=True):
                    projector = self._projector_for(environment_batch, slot.position)
                    projectors[slot.position] = projector
                    action = projector.project_response(response)
                    if action is None:
                        raise RuntimeError(
                            "Environment projector returned no action for active slot"
                        )
                    actions[slot.position] = action

                # Exactly one masked Environment batch step follows each active
                # Generation batch. Inactive slots are never projected or stepped.
                transitions = _align_transitions(
                    list(environment_batch.step(actions, active_mask)), active_mask
                )
                for slot, request, response in zip(active_slots, requests, responses, strict=True):
                    transition = _normalize_transition(transitions[slot.position])
                    continuation = ()
                    if not transition.done and slot.next_turn_index + 1 < self._max_turns:
                        continuation = _normalize_messages(
                            projectors[slot.position].continuation_messages(response, transition)
                        )
                    slot.record(
                        AgentTurn(
                            index=slot.next_turn_index,
                            generation_request=request,
                            generation_response=response,
                            environment_transition=transition,
                        ),
                        final_text=_content_of(response),
                        continuation_messages=continuation,
                    )
                    if transition.done:
                        slot.finish(_termination_reason(transition))

            remaining_mask = [slot.active for slot in slots]
            if any(remaining_mask):
                _terminate_slots_once(environment_batch, slots, remaining_mask, "max_turns")
                for slot in slots:
                    if slot.active:
                        slot.finish("max_turns")
            return [slot.result() for slot in slots]
        except BaseException as exc:
            primary_error = exc
            active_mask = [slot.active for slot in slots] if init_completed else [True] * len(slots)
            reason = "cancelled" if _is_cancellation(exc) else "runtime_error"
            try:
                if any(active_mask):
                    _terminate_slots_once(environment_batch, slots, active_mask, reason)
            except BaseException:
                pass
            raise
        finally:
            try:
                environment_batch.close()
            except BaseException:
                if primary_error is None:
                    raise

    def generate_auxiliary(
        self,
        *,
        request_id: str,
        system: str,
        prompt: str,
        response_format: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run one controller-owned generation with this Runtime's model policy.

        This is intentionally narrower than :meth:`run_batch`: search controllers
        occasionally need a non-solution planning call (AdaEvolve paradigm
        generation). The call uses the configured backend, model, sampling,
        provider options, seed, and response identity, but does not create a fake
        Environment trajectory or score the planning text as a candidate.
        """

        for name, value in (("request_id", request_id), ("system", system), ("prompt", prompt)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if response_format is not None and not isinstance(response_format, Mapping):
            raise TypeError("response_format must be a mapping when provided")

        task = AgentTask(
            task_id=request_id,
            system=system,
            prompt=prompt,
            routing_key=f"auxiliary:{request_id}",
        )
        messages = tuple(
            message
            for message in (
                {"role": "system", "content": system} if system else None,
                {"role": "user", "content": prompt},
            )
            if message is not None
        )
        provider_options = copy.deepcopy(self._provider_options)
        if response_format is not None:
            if "response_format" in provider_options:
                raise ValueError(
                    "configured provider_options already define response_format for an "
                    "AdaEvolve auxiliary generation"
                )
            provider_options["response_format"] = copy.deepcopy(dict(response_format))
        effective_seed = effective_task_seed(task, self._seed)
        if effective_seed is not None:
            provider_options["seed"] = effective_seed
        request = GenerationRequest(
            request_id=request_id,
            model=self._model,
            messages=messages,
            sampling=self._sampling,
            group_id=_generation_group_id(
                task=task,
                turn_index=0,
                model=self._model,
                messages=messages,
                tools=(),
            ),
            provider_options=provider_options,
            routing_key=task.routing_key,
        )
        responses = list(self._generation.generate_batch((request,)))
        _validate_generation_responses((request,), responses)
        return responses[0]

    def _make_environment_batch(self, tasks: Sequence[AgentTask]) -> _EnvironmentBatch:
        if self._environment_batch_factory is not None:
            batch = self._environment_batch_factory(tasks)
            if not isinstance(batch, _EnvironmentBatch):
                _close_best_effort(batch)
                raise TypeError(
                    "environment_batch_factory must return init/step/terminate/close batch"
                )
            return batch
        assert self._environment_factory is not None
        return _PerSlotEnvironmentBatch(tasks, self._environment_factory, seed=self._seed)

    def _projector_for(
        self, environment_batch: _EnvironmentBatch, slot_index: int
    ) -> EnvironmentProjector:
        if self._environment_projector is not None:
            return self._environment_projector
        if not isinstance(environment_batch, _EnvironmentProjectionSource):
            raise TypeError(
                "Environment batch must expose projector(slot_index), or "
                "environment_projector must be provided"
            )
        return _require_projector(environment_batch.projector(slot_index))

    def _request_for(self, slot: _TrajectorySlot) -> Any:
        task_model = slot.task.model or self._model
        turn_index = slot.next_turn_index
        granted_schemas = tuple(
            copy.deepcopy(self._tool_schemas_by_name[name]) for name in slot.task.tools
        )
        provider_options = copy.deepcopy(self._provider_options)
        effective_seed = effective_task_seed(slot.task, self._seed)
        if effective_seed is not None:
            provider_options["seed"] = effective_seed
        messages = _window_image_history(slot.message_snapshot(), self._image_history_messages)
        return GenerationRequest(
            request_id=f"{slot.task.task_id}:{turn_index}",
            model=task_model,
            messages=messages,
            sampling=self._sampling,
            group_id=_generation_group_id(
                task=slot.task,
                turn_index=turn_index,
                model=task_model,
                messages=messages,
                tools=granted_schemas,
            ),
            sample_id=slot.task.sample_id,
            tools=granted_schemas,
            tool_choice=self._tool_choice if granted_schemas else None,
            provider_options=provider_options,
            routing_key=slot.task.routing_key,
        )

    def _validate_task_tool_grants(self, tasks: Sequence[AgentTask]) -> None:
        available = self._tool_schemas_by_name.keys()
        for task in tasks:
            unknown = sorted(set(task.tools).difference(available))
            if unknown:
                raise ValueError(
                    f"AgentTask {task.task_id!r} grants tools without Runtime schemas: {unknown}"
                )


def _terminate_slots_once(
    environment_batch: _EnvironmentBatch,
    slots: Sequence[_TrajectorySlot],
    candidate_mask: Sequence[bool],
    reason: str,
) -> None:
    """Invoke Runtime-driven termination at most once for every trajectory slot."""

    if len(candidate_mask) != len(slots):
        raise RuntimeError("Runtime termination mask cardinality mismatch")
    termination_mask = [
        bool(candidate) and slot.claim_termination()
        for slot, candidate in zip(slots, candidate_mask, strict=True)
    ]
    if any(termination_mask):
        environment_batch.terminate(termination_mask, reason)


def _index_tool_schemas(
    schemas: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index available OpenAI-compatible schemas without granting them to a task."""

    indexed: dict[str, dict[str, Any]] = {}
    for index, schema in enumerate(schemas):
        function = schema.get("function")
        if schema.get("type") != "function" or not isinstance(function, Mapping):
            raise ValueError(f"tools[{index}] must be an OpenAI-compatible function tool schema")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tools[{index}].function.name must be a non-empty string")
        if name in indexed:
            raise ValueError(f"tool schemas must not contain duplicate name {name!r}")
        indexed[name] = copy.deepcopy(dict(schema))
    return indexed


def _require_projector(value: Any) -> EnvironmentProjector:
    if not isinstance(value, EnvironmentProjector):
        raise TypeError(
            "Environment projector must expose project_response() and continuation_messages()"
        )
    return value


def _require_finite_number(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_tasks(tasks: Sequence[AgentTask]) -> tuple[AgentTask, ...]:
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
        raise TypeError("tasks must be a sequence of AgentTask records")
    result = tuple(tasks)
    if any(not isinstance(task, AgentTask) for task in result):
        raise TypeError("tasks must contain AgentTask records")
    task_ids = [task.task_id for task in result]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_id values must be unique within one run_batch call")
    return result


def _normalize_transition(result: Any) -> _EnvironmentTransition:
    """Project a pre-#195 transition through one strict temporary adapter.

    Current Environments return either the legacy mapping contract or an object
    record matching ``EnvironmentTransition``.  Runtime never retains either
    object directly: both paths validate every lifecycle field and produce the
    same private transition record.  Once #195 lands, this adapter can be
    replaced by an exact canonical type check at this single boundary.
    """

    if result is None:
        raise RuntimeError("active Environment slot returned no transition")
    if isinstance(result, _EnvironmentTransition):
        return result
    if is_dataclass(result) and not isinstance(result, type):
        return _adapt_transition_record(result)
    if isinstance(result, Mapping):
        return _adapt_legacy_transition(result)
    return _adapt_transition_record(result)


def _adapt_legacy_transition(result: Mapping[str, Any]) -> _EnvironmentTransition:
    """Validate and convert the one supported pre-#195 mapping contract."""

    try:
        observation = result["observations"]
        reward = result["reward"]
        done = result["done"]
    except KeyError as exc:
        raise TypeError("legacy Environment transition is malformed") from exc
    metadata_value = result.get("metadata")
    metadata_value = {} if metadata_value is None else metadata_value
    if not isinstance(metadata_value, Mapping):
        raise TypeError("Environment transition metadata must be a mapping")
    metadata = copy.deepcopy(dict(metadata_value))
    termination_reason = metadata.get("termination_reason")
    if done and not termination_reason:
        termination_reason = "completed"
    return _EnvironmentTransition(
        observation=copy.deepcopy(observation),
        reward=reward,
        done=done,
        success=metadata.get("success"),
        response_format_valid=metadata.get("response_format_valid", True),
        env_action_valid=metadata.get("env_action_valid", True),
        termination_reason=termination_reason,
        metadata=metadata,
        previous_observation=metadata.get("previous_observation"),
        raw_observation=metadata.get("raw_observation"),
        executed_action=metadata.get("executed_action"),
    )


def _adapt_transition_record(result: Any) -> _EnvironmentTransition:
    """Strictly adapt the current object record without retaining a duck type."""

    if isinstance(result, type) or not is_dataclass(result):
        raise TypeError("Environment step must return a typed transition record or legacy mapping")
    required = (
        "observation",
        "reward",
        "done",
        "success",
        "response_format_valid",
        "env_action_valid",
        "termination_reason",
        "metadata",
    )
    missing = [name for name in required if not hasattr(result, name)]
    if missing:
        raise TypeError("Environment transition record is missing field(s): " + ", ".join(missing))
    return _EnvironmentTransition(
        observation=copy.deepcopy(result.observation),
        reward=result.reward,
        done=result.done,
        success=result.success,
        response_format_valid=result.response_format_valid,
        env_action_valid=result.env_action_valid,
        termination_reason=result.termination_reason,
        metadata=result.metadata,
        previous_observation=copy.deepcopy(getattr(result, "previous_observation", None)),
        raw_observation=copy.deepcopy(getattr(result, "raw_observation", None)),
        executed_action=copy.deepcopy(getattr(result, "executed_action", None)),
    )


def _transition_reward(value: Any) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError("Environment transition reward must be a finite number")
    try:
        reward = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("Environment transition reward must be a finite number") from exc
    if not math.isfinite(reward):
        raise ValueError("Environment transition reward must be finite")
    return reward


def _generation_group_id(
    *,
    task: AgentTask,
    turn_index: int,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> str:
    """Identify one prompt group independently of batch position and branch id."""

    payload = json.dumps(
        _identity_json_value(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
            }
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    owner = task.routing_key or task.task_id
    return f"{owner}:{turn_index}:{digest}"


def _identity_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("generation identity cannot contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("generation identity mappings require string keys")
        return {key: _identity_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_identity_json_value(item) for item in value]
    raise TypeError(f"generation identity cannot encode a value of type {type(value).__name__}")


def _align_transitions(
    transitions: list[Any | None], active_mask: Sequence[bool]
) -> list[Any | None]:
    if len(transitions) == len(active_mask):
        return transitions
    active_count = sum(active_mask)
    if len(transitions) != active_count:
        raise RuntimeError(
            f"Environment batch returned {len(transitions)} transitions for "
            f"{active_count} active slots"
        )
    iterator = iter(transitions)
    return [next(iterator) if active else None for active in active_mask]


def _generation_identity(record: Any, *, label: str) -> tuple[str, str, int]:
    missing = [
        name for name in ("request_id", "group_id", "sample_id") if not hasattr(record, name)
    ]
    if missing:
        raise RuntimeError(f"{label} is missing Generation identity field(s): {', '.join(missing)}")
    request_id = record.request_id
    group_id = record.group_id
    sample_id = record.sample_id
    if not isinstance(request_id, str) or not request_id.strip():
        raise RuntimeError(f"{label} request_id must be a non-empty string")
    if not isinstance(group_id, str):
        raise RuntimeError(f"{label} group_id must be a string")
    if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0:
        raise RuntimeError(f"{label} sample_id must be a non-negative integer")
    return request_id, group_id, sample_id


def _validate_generation_requests(requests: Sequence[Any]) -> None:
    """Reject colliding request/sample identities before provider work starts."""

    seen_request_ids: set[str] = set()
    seen_samples: set[tuple[str, int]] = set()
    for index, request in enumerate(requests):
        request_id, group_id, sample_id = _generation_identity(
            request, label=f"Generation request {index}"
        )
        if request_id in seen_request_ids:
            raise RuntimeError(f"duplicate Generation request_id {request_id!r} in batch")
        seen_request_ids.add(request_id)
        if group_id:
            sample_key = (group_id, sample_id)
            if sample_key in seen_samples:
                raise RuntimeError(
                    f"duplicate Generation (group_id, sample_id) {sample_key!r} in batch"
                )
            seen_samples.add(sample_key)


def _validate_generation_responses(requests: Sequence[Any], responses: Sequence[Any]) -> None:
    if len(responses) != len(requests):
        raise RuntimeError(
            f"generate_batch returned {len(responses)} responses for {len(requests)} requests"
        )
    request_identities = [
        _generation_identity(request, label=f"Generation request {index}")
        for index, request in enumerate(requests)
    ]
    response_identities = [
        _generation_identity(response, label=f"Generation response {index}")
        for index, response in enumerate(responses)
    ]
    response_request_ids = [identity[0] for identity in response_identities]
    if len(set(response_request_ids)) != len(response_request_ids):
        raise RuntimeError("generate_batch returned duplicate response request_id values")
    if len(set(response_identities)) != len(response_identities):
        raise RuntimeError("generate_batch returned duplicate response identities")
    for index, (expected, actual) in enumerate(
        zip(request_identities, response_identities, strict=True)
    ):
        if actual != expected:
            raise RuntimeError(
                f"Generation response {index} identity {actual!r} does not echo "
                f"request identity {expected!r} in order"
            )
        _content_of(responses[index])


def _content_of(response: Any) -> str:
    if not hasattr(response, "content"):
        raise TypeError("Generation response must expose content directly (no candidates)")
    content = response.content
    if not isinstance(content, str):
        raise TypeError("Generation response content must be a string")
    return content


def _normalize_messages(messages: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("Environment continuation_messages must return a sequence of mappings")
    result: list[Mapping[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("Environment continuation messages must be mappings")
        result.append(copy.deepcopy(dict(message)))
    return tuple(result)


def _termination_reason(transition: Any) -> str:
    reason = transition.termination_reason
    if reason in (None, "", "completed", "model_output"):
        return "final"
    return str(reason)


def _is_cancellation(exc: BaseException) -> bool:
    return isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
