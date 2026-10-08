# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The agent-facing robotic environment for one manipulation episode.

One planner action is one tool call, mirroring the coding environment's turn
shape. The environment is backend-agnostic: a simulator (LIBERO, RoboCasa, ...)
or a physical robot is injected as a :class:`RobotBackend`, and the semantic
robotics tools (``vla_act``, ``segment``, ``finish``, ...) are executed through
the shared ``ToolBridge``.

The backend decides task completion and success. The environment also stops an
episode on explicit budgets or internal failures. Planner prose and
``finish(...)`` are recorded as agent claims but never scored.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from alphaapollo.common.environment.base import (
    BaseEnvironment,
    EnvironmentContext,
    EnvironmentInitResult,
    EnvironmentLifecycleError,
    EnvironmentSession,
    EnvironmentState,
    EnvironmentTransition,
)
from alphaapollo.common.environment.default.bridge import ToolBridge, ToolBridgeResult
from alphaapollo.common.environment.default.projection import FEEDBACK_MODES, FeedbackMode
from alphaapollo.common.environment.robotics.projection import (
    format_observation,
    load_observation_image_blocks,
    observation_payload,
    project_model_action,
    robot_continuation_messages,
    tool_call_payload,
)
from alphaapollo.common.execution import ToolError, ToolRequest, ToolResponse
from alphaapollo.common.execution.robotics import (
    RobotBackend,
    RobotObservation,
    RobotResetResult,
    RobotTask,
)
from alphaapollo.common.execution.tools import ToolCatalog
from alphaapollo.common.execution.tools.robotics import (
    FINISH_SPEC,
    ROBOTICS_EXECUTABLE_SPECS,
)
from alphaapollo.common.trajectory.recorder import (
    EnvironmentCapture,
    EnvironmentCaptureKind,
    EnvironmentEventKind,
    EnvironmentEventSink,
    NullEnvironmentEventSink,
    ensure_safe_environment_input,
)

__all__ = ["RobotEnvironment"]

ArtifactLoader = Callable[[Any], bytes]
logger = logging.getLogger(__name__)

_TERMINATION_ENVIRONMENT = "environment_terminated"
_TERMINATION_TURNS = "max_turns_exhausted"
_TERMINATION_STEPS = "max_episode_steps_exhausted"
_TERMINATION_INTERNAL = "internal_error"

_ROBOT_TASK_ENVELOPE_FIELDS = frozenset(
    {"instruction", "benchmark", "environment_version", "backend_metadata"}
)


def _positive_int_or_none(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer when provided")
    return value


def _require_non_empty_string(value: object, *, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be non-empty")
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _backend_metadata_from_payload(task_payload: Mapping[str, Any]) -> dict[str, Any]:
    nested = task_payload.get("backend_metadata", {})
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise TypeError("robotics task payload backend_metadata must be a mapping")

    flat = {
        key: value for key, value in task_payload.items() if key not in _ROBOT_TASK_ENVELOPE_FIELDS
    }
    duplicate_fields = sorted(set(nested).intersection(flat), key=repr)
    if duplicate_fields:
        joined = ", ".join(repr(field) for field in duplicate_fields)
        raise ValueError(
            "robotics task payload metadata fields must not appear both flat and "
            f"inside backend_metadata: {joined}"
        )
    return {**dict(nested), **flat}


def _action_validity(result: ToolResponse | ToolError | None) -> tuple[bool, bool]:
    if not isinstance(result, ToolError):
        return True, True
    if result.stage == "parse":
        return False, False
    if result.stage in {"catalog", "policy"}:
        return True, False
    return True, True


def _request_action(request: ToolRequest) -> dict[str, Any]:
    """Return the one structured call shape shared bridges can normalize."""

    return {
        "tool_calls": [
            {
                "id": request.call_id,
                "type": "function",
                "function": {
                    "name": request.tool_id,
                    "arguments": json.dumps(request.arguments, sort_keys=True),
                },
            }
        ]
    }


class RobotEnvironment(BaseEnvironment[str, Any]):
    """One backend-owned episode between an agent runtime and robotics tools.

    Two budgets are enforced because one tool call can span many simulator
    steps: ``max_turns`` bounds primitive calls and ``max_episode_steps`` bounds
    the simulator steps they consume in total.
    """

    def __init__(
        self,
        *,
        backend: RobotBackend,
        tool_bridge: ToolBridge,
        event_sink: EnvironmentEventSink | None = None,
        max_turns: int | None = None,
        max_episode_steps: int | None = None,
        tool_timeout_s: float | None = None,
        feedback_mode: FeedbackMode = "differentiated",
        finish_tool_id: str = "finish",
        load_artifact: ArtifactLoader | None = None,
        catalog: ToolCatalog | None = None,
    ) -> None:
        if not isinstance(backend, RobotBackend):
            raise TypeError("backend must implement RobotBackend")
        if not isinstance(tool_bridge, ToolBridge):
            raise TypeError("tool_bridge must implement ToolBridge")
        if tool_timeout_s is not None and (
            isinstance(tool_timeout_s, bool)
            or not isinstance(tool_timeout_s, (int, float))
            or not math.isfinite(tool_timeout_s)
            or tool_timeout_s <= 0
        ):
            raise ValueError("tool_timeout_s must be finite and positive when provided")
        if feedback_mode not in FEEDBACK_MODES:
            raise ValueError(
                f"unknown feedback_mode {feedback_mode!r}; expected one of {sorted(FEEDBACK_MODES)}"
            )
        if not isinstance(finish_tool_id, str) or not finish_tool_id.strip():
            raise ValueError("finish_tool_id must be non-empty")
        if load_artifact is not None and not callable(load_artifact):
            raise TypeError("load_artifact must be callable when provided")
        if catalog is not None and not isinstance(catalog, ToolCatalog):
            raise TypeError("catalog must be a ToolCatalog when provided")
        finish_spec = (
            FINISH_SPEC
            if finish_tool_id == FINISH_SPEC.tool_id
            else replace(FINISH_SPEC, tool_id=finish_tool_id)
        )
        # A caller-supplied catalog may legitimately declare the finish tool so
        # the model sees its schema -- `ROBOTICS_TOOL_SPECS` is exactly that,
        # `FINISH_SPEC` bundled with the executable specs, and
        # `build_robotics_tool_executors` registers no executor for it. That
        # declaration is not a collision. A *different* spec under the same id
        # is: the advisory claim would shadow it, so every call would validate
        # against the finish schema and never dispatch, silently disabling the
        # tool. Refuse that here rather than at runtime.
        supplied = catalog or ToolCatalog(ROBOTICS_EXECUTABLE_SPECS)
        supplied_specs = supplied.list_specs(include_internal=True)
        shadowed = next(
            (spec for spec in supplied_specs if spec.tool_id == finish_tool_id),
            None,
        )
        if shadowed is not None and shadowed != finish_spec:
            raise ValueError(
                f"finish_tool_id {finish_tool_id!r} collides with an executable "
                "robotics tool; choose an id outside the catalog"
            )
        # Dispatch never reaches a finish-id entry -- `step` routes that id to
        # `_finish_catalog` -- so drop it here instead of holding an
        # unreachable duplicate whose owner is ambiguous.
        executable_catalog = (
            ToolCatalog(tuple(spec for spec in supplied_specs if spec.tool_id != finish_tool_id))
            if shadowed is not None
            else supplied
        )
        self._backend = backend
        self._tool_bridge = tool_bridge
        self._finish_catalog = ToolCatalog((finish_spec,))
        self._catalog = executable_catalog
        self._events = event_sink or NullEnvironmentEventSink()
        self._max_turns = _positive_int_or_none(max_turns, name="max_turns")
        self._max_episode_steps = _positive_int_or_none(
            max_episode_steps,
            name="max_episode_steps",
        )
        self._tool_timeout_s = tool_timeout_s
        self._feedback_mode = feedback_mode
        self._finish_tool_id = finish_tool_id
        self._load_artifact = load_artifact
        self._session: EnvironmentSession | None = None
        self._closed = False
        self._turns = 0
        self._steps = 0
        self._task: RobotTask | None = None
        self._last_image_blocks: tuple[Mapping[str, Any], ...] = ()
        self._last_request: ToolRequest | None = None
        self._last_result: ToolResponse | ToolError | None = None
        self._last_call_id: str | None = None
        self._agent_claim: str | None = None
        self._last_observation_str: str | None = None

    @property
    def session(self) -> EnvironmentSession | None:
        return self._session

    @property
    def state(self) -> EnvironmentState:
        if self._session is None:
            return EnvironmentState.CLOSED if self._closed else EnvironmentState.CREATED
        return self._session.state

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init(self, context: EnvironmentContext) -> EnvironmentInitResult:
        if self._session is not None or self._closed:
            raise EnvironmentLifecycleError(f"init requires created state, got {self.state.value}")
        task = self._build_task(context)
        session = EnvironmentSession(
            session_id=context.session_id,
            actor=context.actor,
            branch_id=context.branch_id,
        )
        session.activate()
        self._session = session
        self._task = task
        try:
            reset = self._backend.reset(task)
            if not isinstance(reset, RobotResetResult):
                raise TypeError("backend.reset() must return RobotResetResult")
            self._steps = int(self._backend.steps_used)
            image_blocks = self._continuation_images(reset.observation)
            reset_payload = reset.to_dict()
            initial_observation = observation_payload(
                None,
                observation=reset.observation,
                environment_terminated=False,
                environment_success=None,
                next_action_required=True,
                feedback_mode=self._feedback_mode,
            )
            initial_observation["turns_used"] = 0
            initial_observation["episode_steps_used"] = self._steps
            observation_text = format_observation(initial_observation)
            model_content = _initial_model_content(
                instruction=task.instruction,
                observation_text=observation_text,
                image_blocks=image_blocks,
            )
            input_content = {"role": "environment", "content": model_content}
            ensure_safe_environment_input({"model_input": input_content})
        except Exception as exc:
            session.fail(_TERMINATION_INTERNAL)
            self._emit(
                EnvironmentEventKind.FAILED,
                {"stage": "init", "error_type": type(exc).__name__},
            )
            raise
        self._emit(
            EnvironmentEventKind.INITIALIZED,
            {"task_id": task.task_id, "benchmark": task.benchmark},
            capture=EnvironmentCapture(
                kind=EnvironmentCaptureKind.MODEL_INPUT,
                content={"role": "environment", "content": observation_text},
                artifact_refs=reset.observation.artifact_refs,
            ),
        )
        return EnvironmentInitResult(
            observation=model_content,
            metadata={
                "session": session.snapshot(),
                "task_id": task.task_id,
                "benchmark": task.benchmark,
                "reset_info": reset_payload["info"],
                "max_turns": self._max_turns,
                "max_episode_steps": self._max_episode_steps,
            },
        )

    def step(self, action: Any) -> EnvironmentTransition:
        session = self._require_session()
        session.advance()
        self._emit(
            EnvironmentEventKind.ACTION_RECEIVED,
            {"action_type": type(action).__name__},
            capture=EnvironmentCapture(
                kind=EnvironmentCaptureKind.MODEL_OUTPUT,
                content=_capture_model_action(action),
            ),
        )
        self._last_request = None
        self._last_result = None
        self._last_call_id = None
        try:
            projected = project_model_action(action)
        except Exception as exc:  # noqa: BLE001 - a malformed action is an internal step
            return self._internal_failure(exc)

        # Bound the episode before consuming another primitive call.
        budget_stop = self._budget_termination()
        if budget_stop is not None:
            return self._terminal(
                budget_stop,
                success=False,
                payload=_terminal_payload(self._turns, self._steps),
            )

        # Planner prose is never ground truth and does not consume a primitive
        # call, but a budget already exhausted by earlier work remains terminal.
        if projected is None:
            return self._planner_text(action)
        self._turns += 1

        if isinstance(projected, ToolError):
            return self._preflight_failure(projected, request=None)

        request = projected
        self._last_request = request
        self._last_call_id = request.call_id

        catalog = self._finish_catalog if request.tool_id == self._finish_tool_id else self._catalog
        resolved = catalog.resolve(request)
        if isinstance(resolved, ToolError):
            return self._preflight_failure(resolved, request=request)

        if request.tool_id == self._finish_tool_id:
            # Advisory only: record the claim and defer to the backend oracle.
            self._agent_claim = _agent_status(request)
            self._emit(
                EnvironmentEventKind.TOOL_REQUESTED,
                {"call_id": request.call_id, "tool_id": request.tool_id, "advisory": True},
                capture=EnvironmentCapture(
                    kind=EnvironmentCaptureKind.TOOL_REQUEST,
                    content=_tool_request_capture(request),
                ),
            )
            return self._after_action(None, request=request)

        try:
            result = self._tool_bridge.dispatch(
                _request_action(request),
                session.execution_context(timeout_s=self._tool_timeout_s),
            )
        except Exception as exc:  # noqa: BLE001 - a bridge failure is an internal step
            return self._internal_failure(exc)
        if not isinstance(result, ToolBridgeResult):
            return self._internal_failure(
                TypeError(
                    f"tool bridge returned {type(result).__name__}, expected ToolBridgeResult"
                )
            )
        self._emit_tool_result(result)
        outcome = self._outcome_of(result)
        if outcome is None:
            # A dispatched call must come back as a response, an error, or a
            # refusal. Reporting nothing would take ``observation_payload``'s
            # advisory branch, which is reserved for the ``finish`` claim the
            # environment deliberately does not execute, and tell the model an
            # unexecuted call succeeded.
            return self._internal_failure(
                RuntimeError(
                    f"tool bridge returned no outcome for {request.tool_id!r} "
                    f"call {request.call_id!r}"
                )
            )
        self._last_result = outcome
        return self._after_action(outcome, request=request)

    def terminate(self, reason: str = "runtime_requested") -> None:
        super().terminate(reason)
        session = self._session
        if session is not None and session.state is EnvironmentState.ACTIVE:
            session.terminate(reason)
            self._emit(EnvironmentEventKind.TERMINATED, {"reason": reason})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        session = self._session
        cleanup_errors = session.cleanup_errors if session is not None else []
        try:
            close = getattr(self._tool_bridge, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # noqa: BLE001 - later cleanup must still run
            message = f"{type(exc).__name__}: tool bridge cleanup failed"
            cleanup_errors.append(message)
            logger.exception("tool bridge cleanup failed before environment close")
        try:
            self._backend.close()
        except Exception as exc:  # noqa: BLE001 - the session must still close
            message = f"{type(exc).__name__}: backend cleanup failed"
            cleanup_errors.append(message)
            logger.exception("backend cleanup failed before environment close")
        if session is not None and session.state is not EnvironmentState.CLOSED:
            session.close()
        self._emit(
            EnvironmentEventKind.CLOSED,
            {
                "cleanup_errors": list(cleanup_errors),
                "event_errors": list(session.event_errors) if session is not None else [],
            },
        )

    # ------------------------------------------------------------------
    # Runtime projection
    # ------------------------------------------------------------------
    def project_response(self, response: object) -> Any:
        """Project a generation into the message shape ``normalize`` accepts."""

        tool_calls = getattr(response, "tool_calls", ())
        if tool_calls:
            projected: dict[str, Any] = {
                "content": getattr(response, "content", "") or "",
                "tool_calls": [tool_call_payload(call) for call in tool_calls],
            }
            reasoning = getattr(response, "reasoning_content", None)
            if isinstance(reasoning, str):
                projected["reasoning_content"] = reasoning
            return projected
        return getattr(response, "content", "") or ""

    def continuation_messages(
        self,
        response: object,
        transition: object,
    ) -> tuple[Mapping[str, Any], ...]:
        if bool(getattr(transition, "done", False)) or self._last_observation_str is None:
            return ()
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": getattr(response, "content", "") or "",
        }
        reasoning = getattr(response, "reasoning_content", None)
        if isinstance(reasoning, str):
            assistant["reasoning_content"] = reasoning
        raw_calls = getattr(response, "tool_calls", ())
        tool_calls = [tool_call_payload(call) for call in raw_calls] if raw_calls else []
        matching_calls = [call for call in tool_calls if call.get("id") == self._last_call_id]
        call_id = self._last_call_id if len(matching_calls) == 1 else None
        if call_id is not None:
            assistant["tool_calls"] = matching_calls
        return robot_continuation_messages(
            assistant_message=assistant,
            observation_text=self._last_observation_str,
            image_blocks=self._last_image_blocks,
            call_id=call_id,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_session(self) -> EnvironmentSession:
        if self._closed:
            raise EnvironmentLifecycleError("environment is closed")
        if self._session is None:
            raise EnvironmentLifecycleError("operation requires an initialized environment")
        self._session.require_active()
        return self._session

    def _build_task(self, context: EnvironmentContext) -> RobotTask:
        task_payload = context.task_payload or {}
        backend_metadata = _backend_metadata_from_payload(task_payload)

        instruction = _require_non_empty_string(
            context.user_prompt,
            name="robotics user_prompt",
        )
        payload_instruction = task_payload.get("instruction")
        if payload_instruction is not None:
            private_instruction = _require_non_empty_string(
                payload_instruction,
                name="robotics task payload instruction",
            )
            if private_instruction != instruction:
                raise ValueError(
                    "robotics task payload instruction must match the public user_prompt"
                )

        benchmark = _require_non_empty_string(
            task_payload.get("benchmark"),
            name="robotics task payload benchmark",
        ).strip()
        environment_version = _require_non_empty_string(
            task_payload.get("environment_version"),
            name="robotics task payload environment_version",
        ).strip()
        return RobotTask(
            task_id=context.task_id or context.session_id,
            benchmark=benchmark,
            instruction=instruction,
            environment_version=environment_version,
            backend_metadata=dict(backend_metadata),
        )

    def _refresh_oracle(self) -> RobotObservation:
        observation = self._backend.observe()
        if not isinstance(observation, RobotObservation):
            raise TypeError("backend.observe() must return RobotObservation")
        self._steps = int(self._backend.steps_used)
        return observation

    def _continuation_images(
        self,
        observation: RobotObservation,
    ) -> tuple[Mapping[str, Any], ...]:
        if self._load_artifact is None:
            if any((ref.type or "").startswith("image/") for ref in observation.artifact_refs):
                raise RuntimeError(
                    "robot observation contains image artifacts, but no artifact "
                    "loader is configured"
                )
            return ()
        return load_observation_image_blocks(observation, self._load_artifact)

    def _backend_state(self) -> tuple[bool, bool, bool | None, str | None]:
        """Snapshot the backend verdict once, normalized to the episode contract.

        A backend is foreign code: a gymnasium-derived simulator reports numpy
        scalars, and a physical rig can report anything. Every value is coerced
        here, inside the caller's guarded block, so a non-conforming verdict ends
        only this episode instead of escaping ``step`` and aborting every other
        episode in the batch.

        ``success`` is three-valued and only meaningful once the episode ends;
        see :meth:`RobotTransition.__post_init__`. A definitive ``False`` while
        the run continues would make consumers misreport it as failed, so a
        mid-episode verdict is reported as unknown.
        """

        terminated = bool(self._backend.terminated)
        truncated = bool(self._backend.truncated)
        raw_success = self._backend.success
        raw_reason = self._backend.termination_reason
        success = (
            None if raw_success is None or not (terminated or truncated) else bool(raw_success)
        )
        reason = raw_reason if isinstance(raw_reason, str) and raw_reason.strip() else None
        return terminated, truncated, success, reason

    def _planner_text(self, action: Any) -> EnvironmentTransition:
        try:
            observation = self._refresh_oracle()
            terminated, truncated, success, reason = self._backend_state()
            done = terminated or truncated
            image_blocks = () if done else self._continuation_images(observation)
        except Exception as exc:  # noqa: BLE001 - observation faults end only this episode
            return self._internal_failure(exc)
        self._last_image_blocks = image_blocks
        payload = observation_payload(
            None,
            request=None,
            observation=observation,
            environment_terminated=done,
            environment_success=success,
            next_action_required=not done,
            feedback_mode=self._feedback_mode,
        )
        payload["planner_text"] = _capture_model_action(action)
        payload["turns_used"] = self._turns
        payload["episode_steps_used"] = self._steps
        self._last_observation_str = format_observation(payload)
        if done:
            return self._terminal_from_backend(
                payload,
                terminated=terminated,
                success=success,
                reason=reason,
            )
        return self._nonterminal(payload)

    def _after_action(
        self,
        outcome: ToolResponse | ToolError | None,
        *,
        request: ToolRequest | None,
    ) -> EnvironmentTransition:
        try:
            observation = self._refresh_oracle()
            terminated, truncated, success, reason = self._backend_state()
            done = terminated or truncated
            budget_stop = None if done else self._steps_termination()
            image_blocks = (
                () if done or budget_stop is not None else self._continuation_images(observation)
            )
        except Exception as exc:  # noqa: BLE001 - observation faults end only this episode
            return self._internal_failure(exc)
        self._last_image_blocks = image_blocks
        payload = observation_payload(
            outcome,
            request=request,
            observation=observation,
            environment_terminated=done,
            environment_success=success,
            next_action_required=not done,
            feedback_mode=self._feedback_mode,
        )
        payload["turns_used"] = self._turns
        payload["episode_steps_used"] = self._steps
        if self._agent_claim is not None:
            payload["agent_claim"] = self._agent_claim
        self._last_observation_str = format_observation(payload)
        if done:
            return self._terminal_from_backend(
                payload,
                terminated=terminated,
                success=success,
                reason=reason,
            )
        if budget_stop is not None:
            return self._terminal(budget_stop, success=False, payload=payload)
        return self._nonterminal(payload)

    def _terminal_from_backend(
        self,
        payload: dict[str, Any],
        *,
        terminated: bool,
        success: bool | None,
        reason: str | None,
    ) -> EnvironmentTransition:
        if not reason:
            reason = _TERMINATION_ENVIRONMENT if terminated else "truncated"
        return self._terminal(reason, success=success, payload=payload)

    def _nonterminal(self, payload: dict[str, Any]) -> EnvironmentTransition:
        response_format_valid, env_action_valid = _action_validity(self._last_result)
        return EnvironmentTransition(
            observation=format_observation(payload),
            reward=0.0,
            done=False,
            response_format_valid=response_format_valid,
            env_action_valid=env_action_valid,
            metadata=self._metadata(payload),
        )

    def _terminal(
        self,
        reason: str,
        *,
        success: bool | None,
        payload: dict[str, Any],
    ) -> EnvironmentTransition:
        payload = dict(payload)
        # The payload the planner reads must agree with the transition. A budget
        # stop arrives with the still-running verdict the observation was built
        # from, so the terminal verdict is restated here. Every caller passes a
        # value already normalized by ``_backend_state`` or a literal.
        payload.update(
            {
                "environment_terminated": True,
                "environment_success": success,
                "next_action_required": False,
            }
        )
        session = self._session
        if session is not None and session.state is EnvironmentState.ACTIVE:
            session.terminate(reason)
        observation = format_observation(payload)
        self._last_observation_str = observation
        self._emit(
            EnvironmentEventKind.TERMINATED,
            {"reason": reason, "success": success},
            capture=EnvironmentCapture(
                kind=EnvironmentCaptureKind.FINAL_OUTPUT,
                content={"role": "assistant", "content": observation},
            ),
        )
        metadata = self._metadata(payload)
        metadata["termination_reason"] = reason
        response_format_valid, env_action_valid = _action_validity(self._last_result)
        return EnvironmentTransition(
            observation=observation,
            reward=1.0 if success is True else 0.0,
            done=True,
            success=success,
            response_format_valid=response_format_valid,
            env_action_valid=env_action_valid,
            termination_reason=reason,
            metadata=metadata,
        )

    def _internal_failure(self, exc: Exception) -> EnvironmentTransition:
        self._last_image_blocks = ()
        session = self._require_session()
        session.fail(_TERMINATION_INTERNAL)
        payload = _terminal_payload(self._turns, self._steps)
        payload["error"] = {
            "stage": "internal",
            "code": "environment_internal_error",
            "message": str(exc),
        }
        self._last_observation_str = format_observation(payload)
        self._emit(
            EnvironmentEventKind.FAILED,
            {"error_type": type(exc).__name__, "error_code": "environment_internal_error"},
        )
        metadata = self._metadata(payload)
        metadata["termination_reason"] = _TERMINATION_INTERNAL
        response_format_valid, env_action_valid = _action_validity(self._last_result)
        return EnvironmentTransition(
            observation=format_observation(payload),
            reward=0.0,
            done=True,
            success=False,
            response_format_valid=response_format_valid,
            env_action_valid=env_action_valid,
            termination_reason=_TERMINATION_INTERNAL,
            metadata=metadata,
        )

    def _preflight_failure(
        self,
        error: ToolError,
        *,
        request: ToolRequest | None,
    ) -> EnvironmentTransition:
        self._last_result = error
        self._last_call_id = error.call_id
        self._emit(
            EnvironmentEventKind.PREFLIGHT_FAILED,
            {
                "call_id": error.call_id,
                "tool_id": error.tool_id,
                "stage": error.stage,
                "code": error.code,
                "attempted": False,
            },
            capture=EnvironmentCapture(
                kind=EnvironmentCaptureKind.TOOL_ERROR,
                content=_tool_error_capture(error),
            ),
        )
        return self._after_action(error, request=request)

    def _budget_termination(self) -> str | None:
        # Pre-dispatch check: refuse to consume another primitive call.
        if self._max_turns is not None and self._turns >= self._max_turns:
            return _TERMINATION_TURNS
        if self._max_episode_steps is not None and self._steps >= self._max_episode_steps:
            return _TERMINATION_STEPS
        return None

    def _steps_termination(self) -> str | None:
        # Post-oracle check: a tool may have consumed simulator steps.
        if self._max_episode_steps is not None and self._steps >= self._max_episode_steps:
            return _TERMINATION_STEPS
        return None

    def _metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._session
        metadata: dict[str, Any] = {
            "observation": payload,
            "session": session.snapshot() if session is not None else {},
            "turns_used": self._turns,
            "episode_steps_used": self._steps,
        }
        if self._last_request is not None:
            metadata["tool_request"] = _tool_request_capture(self._last_request)
        if isinstance(self._last_result, ToolResponse):
            metadata["tool_response"] = _tool_response_capture(self._last_result)
        elif isinstance(self._last_result, ToolError):
            metadata["tool_error"] = _tool_error_capture(self._last_result)
        return metadata

    def _outcome_of(self, result: ToolBridgeResult) -> ToolResponse | ToolError | None:
        if result.refused:
            return ToolError(
                stage="policy",
                code=result.refusal_code or "tool_refused",
                message=result.refusal_message or "tool refused",
                call_id=result.request.call_id if result.request else None,
                tool_id=result.request.tool_id if result.request else None,
            )
        if result.error is not None:
            return result.error
        if result.response is not None:
            return result.response
        return None

    def _emit_tool_result(self, result: ToolBridgeResult) -> None:
        if result.refused:
            self._emit(
                EnvironmentEventKind.TOOL_REFUSED,
                {"code": result.refusal_code, "attempted": False},
            )
            return
        if result.request is not None:
            self._emit(
                EnvironmentEventKind.TOOL_REQUESTED,
                {
                    "call_id": result.request.call_id,
                    "tool_id": result.request.tool_id,
                    "source": result.request.source,
                },
                capture=EnvironmentCapture(
                    kind=EnvironmentCaptureKind.TOOL_REQUEST,
                    content=_tool_request_capture(result.request),
                ),
            )
        if isinstance(result.error, ToolError):
            self._emit(
                EnvironmentEventKind.PREFLIGHT_FAILED,
                {
                    "call_id": result.error.call_id,
                    "tool_id": result.error.tool_id,
                    "stage": result.error.stage,
                    "code": result.error.code,
                    "attempted": False,
                },
                capture=EnvironmentCapture(
                    kind=EnvironmentCaptureKind.TOOL_ERROR,
                    content=_tool_error_capture(result.error),
                ),
            )
            return
        if result.response is None:
            return
        self._emit(
            EnvironmentEventKind.EXECUTION_ATTEMPTED,
            {
                "call_id": result.response.call_id,
                "tool_id": result.response.tool_id,
                "attempted": True,
            },
        )
        self._emit(
            EnvironmentEventKind.EXECUTION_COMPLETED,
            {
                "call_id": result.response.call_id,
                "tool_id": result.response.tool_id,
                "ok": result.response.exit_code == 0,
                "attempted": True,
                "exit_code": result.response.exit_code,
            },
            capture=EnvironmentCapture(
                kind=EnvironmentCaptureKind.TOOL_RESPONSE,
                content=_tool_response_capture(result.response),
                artifact_refs=result.response.artifacts,
            ),
        )

    def _emit(
        self,
        kind: EnvironmentEventKind,
        payload: dict[str, Any],
        *,
        capture: EnvironmentCapture | None = None,
    ) -> None:
        session = self._session
        if session is None:
            return
        try:
            self._events.emit(kind, session, payload, capture)
        except Exception as exc:  # noqa: BLE001 - trajectory recording is best-effort
            message = f"{kind.value}: {type(exc).__name__}: event recording failed"
            session.event_errors.append(message)
            logger.exception(
                "failed to record %s for robot environment session %s",
                kind.value,
                session.session_id,
            )


def _terminal_payload(turns_used: int, steps_used: int) -> dict[str, Any]:
    return {
        "environment_terminated": True,
        "environment_success": False,
        "next_action_required": False,
        "turns_used": turns_used,
        "episode_steps_used": steps_used,
    }


def _initial_model_content(
    *,
    instruction: str,
    observation_text: str,
    image_blocks: tuple[Mapping[str, Any], ...],
) -> str | list[dict[str, Any]]:
    if not image_blocks:
        return f"{instruction}\n\n{observation_text}"
    return [
        {"type": "text", "text": instruction},
        {"type": "text", "text": observation_text},
        *[dict(block) for block in image_blocks],
    ]


def _agent_status(request: ToolRequest) -> str | None:
    status = request.arguments.get("status")
    return status if isinstance(status, str) and status.strip() else None


def _capture_model_action(action: Any) -> Any:
    if isinstance(action, (str, Mapping)):
        return action
    dump = getattr(action, "model_dump", None)
    if callable(dump):
        try:
            value = dump()
        except Exception:  # noqa: BLE001 - a capture should never fail the episode
            return ""
        return value if isinstance(value, (str, Mapping)) else ""
    return str(action)


def _tool_request_capture(request: ToolRequest) -> dict[str, Any]:
    return {
        "call_id": request.call_id,
        "tool_id": request.tool_id,
        "arguments": request.arguments,
        "source": request.source,
    }


def _tool_response_capture(response: ToolResponse) -> dict[str, Any]:
    return {
        "call_id": response.call_id,
        "tool_id": response.tool_id,
        "stdout": response.stdout,
        "stderr": response.stderr,
        "exit_code": response.exit_code,
        "artifacts": [ref.model_dump(mode="json") for ref in response.artifacts],
    }


def _tool_error_capture(error: ToolError) -> dict[str, Any]:
    return {
        "call_id": error.call_id,
        "tool_id": error.tool_id,
        "stage": error.stage,
        "code": error.code,
        "message": error.message,
    }
