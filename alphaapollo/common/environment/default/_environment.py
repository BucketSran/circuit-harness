# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Single-episode lifecycle for the default tool-capable Environment."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from alphaapollo.common.environment.base import (
    BaseEnvironment,
    EnvironmentContext,
    EnvironmentInitResult,
    EnvironmentLifecycleError,
    EnvironmentObservation,
    EnvironmentSession,
    EnvironmentState,
    EnvironmentTransition,
)
from alphaapollo.common.environment.default.bridge import ToolBridge, ToolBridgeResult
from alphaapollo.common.environment.default.outcome import (
    _capture_model_action,
    _episode_context_from,
    _execution_failure,
    _gold_from_payload,
    _runtime_tool_metadata,
    _tool_validity,
)
from alphaapollo.common.environment.default.projection import FEEDBACK_MODES, FeedbackMode
from alphaapollo.common.execution import ToolError
from alphaapollo.common.execution.workspace import WorkspaceLease, WorkspaceProvider
from alphaapollo.common.grader import (
    extract_final_answer,
    grade,
    require_offline_grader,
)
from alphaapollo.common.trajectory.recorder import (
    EnvironmentCapture,
    EnvironmentCaptureKind,
    EnvironmentEventKind,
    EnvironmentEventSink,
    NullEnvironmentEventSink,
    ensure_safe_environment_input,
    sanitize_capture_value,
)

logger = logging.getLogger(__name__)

Cleanup = Callable[[EnvironmentSession], None]


# Environment lifecycle ------------------------------------------------------
class DefaultEnvironment(BaseEnvironment[str, Any]):
    """One policy-free episode between an agent runtime and atomic tools.

    Ordinary model output is returned as a terminal candidate. Tool calls are
    delegated to ``ToolBridge`` and yield a non-terminal observation so the
    runtime can continue the same agent turn loop.
    """

    def __init__(
        self,
        *,
        tool_bridge: ToolBridge,
        event_sink: EnvironmentEventSink | None = None,
        max_steps: int | None = None,
        tool_timeout_s: float | None = None,
        cleanup: Cleanup | None = None,
        execution_mode: str = "default",
        workspace_provider: WorkspaceProvider | None = None,
        feedback_mode: FeedbackMode = "differentiated",
        grader_id: str | None = None,
    ) -> None:
        if max_steps is not None and (
            isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer when provided")
        if tool_timeout_s is not None and (
            isinstance(tool_timeout_s, bool)
            or not isinstance(tool_timeout_s, (int, float))
            or not math.isfinite(tool_timeout_s)
            or tool_timeout_s <= 0
        ):
            raise ValueError("tool_timeout_s must be finite and positive when provided")
        if not isinstance(execution_mode, str) or not execution_mode.strip():
            raise ValueError("execution_mode must be non-empty")
        if feedback_mode not in FEEDBACK_MODES:
            raise ValueError(
                f"unknown feedback_mode {feedback_mode!r}; expected one of {sorted(FEEDBACK_MODES)}"
            )
        if grader_id is not None:
            # Resolve now: a matrix must not run for an hour and then discover at
            # its first terminal step that the grader does not exist, or that it
            # is environment-graded and has no offline verdict to give.
            require_offline_grader(grader_id)
        self._tool_bridge = tool_bridge
        self._events = event_sink or NullEnvironmentEventSink()
        self._max_steps = max_steps
        self._tool_timeout_s = tool_timeout_s
        self._cleanup = cleanup
        self._execution_mode = execution_mode
        self._feedback_mode = feedback_mode
        self._workspace_provider = workspace_provider
        self._grader_id = grader_id
        self._gold: str | None = None
        self._workspace_lease: WorkspaceLease | None = None
        self._session: EnvironmentSession | None = None
        self._closed_without_session = False

    @property
    def session(self) -> EnvironmentSession | None:
        return self._session

    @property
    def state(self) -> EnvironmentState:
        if self._session is None:
            return (
                EnvironmentState.CLOSED
                if self._closed_without_session
                else EnvironmentState.CREATED
            )
        return self._session.state

    def init(
        self,
        context: EnvironmentContext | None = None,
        *,
        session_id: str | None = None,
        actor: str | None = None,
        branch_id: str = "main",
        round_index: int = 0,
        initial_observation: str = "",
        workspace_snapshot_ref: str | None = None,
        episode_context: dict[str, Any] | None = None,
    ) -> EnvironmentInitResult:
        if context is not None:
            if session_id is not None or actor is not None:
                raise TypeError("pass either EnvironmentContext or legacy init keywords")
            session_id = context.session_id
            actor = context.actor
            branch_id = context.branch_id
            round_index = context.round_index
            initial_observation = context.user_prompt
            workspace_snapshot_ref = context.metadata.get("workspace_snapshot_ref")
            episode_context = _episode_context_from(context)
            self._gold = _gold_from_payload(context.task_payload, self._grader_id)
        if session_id is None or actor is None:
            raise TypeError("session_id and actor are required")
        if self._session is not None or self._closed_without_session:
            raise EnvironmentLifecycleError(f"init requires created state, got {self.state.value}")
        if not isinstance(initial_observation, str):
            raise TypeError("initial_observation must be a string")
        if episode_context is not None and not isinstance(episode_context, dict):
            raise TypeError("episode_context must be a dict when provided")
        input_content = {
            "role": "environment",
            "content": initial_observation,
            "episode_context": episode_context or {},
        }
        ensure_safe_environment_input(
            {
                "model_input": input_content,
                "workspace_snapshot_ref": workspace_snapshot_ref,
            }
        )
        workspace_snapshot_ref = self._resolve_workspace_snapshot(
            workspace_snapshot_ref,
            session_id=session_id,
            actor=actor,
            branch_id=branch_id,
        )
        initial_capture = EnvironmentCapture(
            kind=EnvironmentCaptureKind.MODEL_INPUT,
            content=input_content,
        )
        try:
            session = EnvironmentSession(
                session_id=session_id,
                actor=actor,
                branch_id=branch_id,
                round_index=round_index,
                workspace_snapshot_ref=workspace_snapshot_ref,
                execution_mode=self._execution_mode,
            )
            session.activate()
            self._session = session
            self._emit(
                EnvironmentEventKind.INITIALIZED,
                {
                    "has_initial_observation": bool(initial_observation),
                    "workspace_snapshot_ref": workspace_snapshot_ref,
                },
                capture=initial_capture,
            )
        except Exception:
            self._session = None
            if self._workspace_lease is not None:
                self._workspace_lease.close()
                self._workspace_lease = None
            raise
        return EnvironmentInitResult(
            observation=initial_observation,
            metadata={"session": session.snapshot()},
        )

    def _resolve_workspace_snapshot(
        self,
        current_ref: str | None,
        *,
        session_id: str,
        actor: str,
        branch_id: str,
    ) -> str | None:
        if current_ref is not None or self._workspace_provider is None:
            return current_ref
        lease = self._workspace_provider.acquire(
            session_id=session_id,
            actor=actor,
            branch_id=branch_id,
        )
        if not isinstance(lease, WorkspaceLease):
            raise TypeError("workspace provider must return WorkspaceLease")
        try:
            ensure_safe_environment_input({"workspace_snapshot_ref": lease.ref})
        except Exception:
            lease.close()
            raise
        self._workspace_lease = lease
        return lease.ref

    def step(self, action: Any) -> EnvironmentTransition:
        session = self._require_session()
        session.require_active()
        session.advance()
        self._emit(
            EnvironmentEventKind.ACTION_RECEIVED,
            {"action_type": type(action).__name__},
            capture=EnvironmentCapture(
                kind=EnvironmentCaptureKind.MODEL_OUTPUT,
                content=_capture_model_action(action),
            ),
        )
        try:
            result = self._tool_bridge.dispatch(
                action,
                session.execution_context(timeout_s=self._tool_timeout_s),
            )
        except Exception as exc:
            return self._internal_failure(exc)
        if not isinstance(result, ToolBridgeResult):
            return self._internal_failure(
                TypeError(
                    f"tool bridge returned {type(result).__name__}, expected ToolBridgeResult"
                )
            )

        if not result.is_tool_call:
            observation = EnvironmentObservation(
                kind="model_output",
                content=result.model_output or "",
            )
            session.terminate("model_output")
            self._emit_observation(observation)
            self._emit(
                EnvironmentEventKind.TERMINATED,
                {"reason": session.termination_reason},
                capture=EnvironmentCapture(
                    kind=EnvironmentCaptureKind.FINAL_OUTPUT,
                    content={
                        "role": "assistant",
                        "content": observation.content,
                    },
                ),
            )
            return self._output(observation, done=True)

        self._emit_tool_result(result)
        observation = self._tool_observation(result)
        done = False
        if self._max_steps is not None and session.step_index >= self._max_steps:
            session.terminate("max_steps")
            done = True
        self._emit_observation(observation)
        if done:
            self._emit(
                EnvironmentEventKind.TERMINATED,
                {"reason": session.termination_reason},
            )
        return self._output(
            observation,
            done=done,
            success=False if done else None,
            tool_result=result,
        )

    def terminate(self, reason: str = "runtime_requested") -> None:
        session = self._require_session()
        session.terminate(reason)
        self._emit(EnvironmentEventKind.TERMINATED, {"reason": reason})

    def close(self) -> None:
        if self._session is None:
            close = getattr(self._tool_bridge, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception("tool bridge cleanup failed before environment init")
                    return
            self._closed_without_session = True
            return
        session = self._session
        if session.state is EnvironmentState.CLOSED:
            self._close_tool_bridge(session)
            self._release_workspace(session)
            return
        if self._cleanup is not None:
            try:
                self._cleanup(session)
            except Exception as exc:
                message = f"{type(exc).__name__}: cleanup failed"
                session.cleanup_errors.append(message)
                logger.exception("environment cleanup failed for session %s", session.session_id)
        self._close_tool_bridge(session)
        self._release_workspace(session)
        session.close()
        self._emit(
            EnvironmentEventKind.CLOSED,
            {
                "cleanup_errors": list(session.cleanup_errors),
                "event_errors": list(session.event_errors),
            },
        )

    def _close_tool_bridge(self, session: EnvironmentSession) -> None:
        close = getattr(self._tool_bridge, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:
            message = f"{type(exc).__name__}: tool bridge cleanup failed"
            if message not in session.cleanup_errors:
                session.cleanup_errors.append(message)
            logger.exception("tool bridge cleanup failed for session %s", session.session_id)

    def _release_workspace(self, session: EnvironmentSession) -> None:
        if self._workspace_lease is None or self._workspace_lease.released:
            return
        try:
            self._workspace_lease.close()
        except Exception as exc:
            message = f"{type(exc).__name__}: workspace cleanup failed"
            if message not in session.cleanup_errors:
                session.cleanup_errors.append(message)
            logger.exception("workspace cleanup failed for session %s", session.session_id)

    def _require_session(self) -> EnvironmentSession:
        if self._session is None:
            raise EnvironmentLifecycleError(
                f"operation requires initialized environment, got {self.state.value}"
            )
        return self._session

    def _emit(
        self,
        kind: EnvironmentEventKind,
        payload: dict[str, Any],
        *,
        capture: EnvironmentCapture | None = None,
    ) -> None:
        session = self._require_session()
        try:
            self._events.emit(kind, session, payload, capture)
        except Exception as exc:
            session.event_errors.append(
                f"{kind.value}: {type(exc).__name__}: event recording failed"
            )
            logger.exception(
                "failed to record %s for environment session %s",
                kind.value,
                session.session_id,
            )

    def _emit_tool_result(self, result: ToolBridgeResult) -> None:
        if result.refused:
            self._emit(
                EnvironmentEventKind.TOOL_REFUSED,
                {
                    "code": result.refusal_code,
                    "attempted": False,
                },
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
                    content={
                        "call_id": result.request.call_id,
                        "tool_id": result.request.tool_id,
                        "arguments": result.request.arguments,
                        "source": result.request.source,
                    },
                ),
            )
        if isinstance(result.error, ToolError):
            unresolved = result.error.tool_id is None
            self._emit(
                EnvironmentEventKind.PREFLIGHT_FAILED,
                {
                    "call_id": result.error.call_id,
                    "tool_id": result.error.tool_id,
                    "stage": result.error.stage,
                    "code": result.error.code,
                    "attempted": False,
                    **(
                        {"canonicalization_status": "rejected_before_resolution"}
                        if unresolved
                        else {}
                    ),
                },
                capture=EnvironmentCapture(
                    kind=EnvironmentCaptureKind.TOOL_ERROR,
                    content={
                        "call_id": result.error.call_id,
                        "tool_id": result.error.tool_id,
                        "stage": result.error.stage,
                        "code": result.error.code,
                        "message": result.error.message,
                        "attempted": False,
                        **(
                            {"canonicalization_status": "rejected_before_resolution"}
                            if unresolved
                            else {}
                        ),
                    },
                ),
            )
            return
        if result.response is None or result.record is None:
            raise TypeError("attempted tool result requires response and audit record")
        failure = _execution_failure(result.response.exit_code)
        payload: dict[str, Any] = {
            "call_id": result.response.call_id,
            "tool_id": result.response.tool_id,
            "ok": failure is None,
            "attempted": True,
            "exit_code": result.response.exit_code,
            "recorded": True,
        }
        if failure is not None:
            payload["stage"], payload["code"] = failure
        self._emit(
            EnvironmentEventKind.EXECUTION_ATTEMPTED,
            {
                "call_id": result.response.call_id,
                "tool_id": result.response.tool_id,
                "attempted": True,
            },
        )
        capture_content: dict[str, Any] = {
            "call_id": result.response.call_id,
            "tool_id": result.response.tool_id,
            "stdout": result.response.stdout,
            "stderr": result.response.stderr,
            "exit_code": result.response.exit_code,
            "artifacts": [ref.model_dump(mode="json") for ref in result.response.artifacts],
            "attempted": True,
            "record": result.record.model_dump(mode="json"),
        }
        if failure is not None:
            capture_content["error_stage"], capture_content["error_code"] = failure
        self._emit(
            EnvironmentEventKind.EXECUTION_COMPLETED,
            payload,
            capture=EnvironmentCapture(
                kind=EnvironmentCaptureKind.TOOL_RESPONSE,
                content=capture_content,
                artifact_refs=result.response.artifacts,
            ),
        )

    def _emit_observation(self, observation: EnvironmentObservation) -> None:
        self._emit(
            EnvironmentEventKind.OBSERVATION_EMITTED,
            {
                "kind": observation.kind,
                "call_id": observation.call_id,
                "tool_id": observation.tool_id,
                "attempted": observation.attempted,
                "exit_code": observation.exit_code,
                "error_stage": observation.error_stage,
                "error_code": observation.error_code,
                "contamination_flags": list(observation.contamination_flags),
            },
            capture=EnvironmentCapture(
                kind=EnvironmentCaptureKind.ENVIRONMENT_OBSERVATION,
                content=observation.to_dict(),
                contamination_flags=observation.contamination_flags,
            ),
        )

    def _tool_observation(self, result: ToolBridgeResult) -> EnvironmentObservation:
        call_id = result.request.call_id if result.request else None
        tool_id = result.request.tool_id if result.request else None
        if isinstance(result.error, ToolError):
            call_id = result.error.call_id or call_id
            tool_id = result.error.tool_id or tool_id
        error = result.error
        error_stage = error.stage if error else None
        error_code = error.code if error else None
        if result.refused:
            error_stage = "policy"
            error_code = result.refusal_code
        if result.response is not None:
            failure = _execution_failure(result.response.exit_code)
            if failure is not None:
                error_stage, error_code = failure
        sanitized_payload, contamination_flags, _ = sanitize_capture_value(
            result.observation_payload(feedback_mode=self._feedback_mode)
        )
        if not isinstance(sanitized_payload, dict):
            raise TypeError("sanitized tool observation payload must remain an object")
        if contamination_flags:
            sanitized_payload["contamination_flags"] = list(contamination_flags)
        return EnvironmentObservation(
            kind=(
                "tool_refusal"
                if result.refused
                else (
                    "tool_response"
                    if result.response is not None and result.response.exit_code == 0
                    else "tool_error"
                )
            ),
            content=ToolBridgeResult.format_observation(sanitized_payload),
            call_id=call_id,
            tool_id=tool_id,
            attempted=result.attempted,
            exit_code=result.response.exit_code if result.response else None,
            error_stage=error_stage,
            error_code=error_code,
            contamination_flags=contamination_flags,
        )

    def _internal_failure(self, exc: Exception) -> EnvironmentTransition:
        session = self._require_session()
        logger.error(
            "environment step failed for session %s with %s",
            session.session_id,
            type(exc).__name__,
            exc_info=exc,
        )
        session.fail("internal_error")
        observation = EnvironmentObservation(
            kind="environment_error",
            content="environment internal error",
            error_stage="internal",
            error_code="environment_internal_error",
        )
        self._emit(
            EnvironmentEventKind.FAILED,
            {
                "error_type": type(exc).__name__,
                "error_code": observation.error_code,
            },
        )
        self._emit_observation(observation)
        return self._output(observation, done=True, success=False)

    def _reward(self, observation: EnvironmentObservation, *, done: bool) -> float:
        """Grade the final answer in loop, when this episode was given gold.

        Only a terminal observation is graded: an intermediate tool result is not
        an answer. An unscoreable verdict is 0.0 rather than a penalty, because
        failing to parse an answer is not evidence the agent did worse than one
        that answered wrongly.
        """

        if not done or self._grader_id is None or self._gold is None:
            return 0.0
        verdict = grade(
            extract_final_answer(observation.content),
            self._gold,
            grader_id=self._grader_id,
        )
        return 1.0 if verdict else 0.0

    def _output(
        self,
        observation: EnvironmentObservation,
        *,
        done: bool,
        success: bool | None = None,
        tool_result: ToolBridgeResult | None = None,
    ) -> EnvironmentTransition:
        metadata: dict[str, Any] = {
            "observation": observation.to_dict(),
            "session": self._require_session().snapshot(),
        }
        if tool_result is not None:
            metadata.update(_runtime_tool_metadata(tool_result))
        session = self._require_session()
        response_format_valid, env_action_valid = _tool_validity(tool_result)
        return EnvironmentTransition(
            observation=observation.content,
            reward=self._reward(observation, done=done),
            done=done,
            success=success,
            response_format_valid=response_format_valid,
            env_action_valid=env_action_valid,
            termination_reason=session.termination_reason if done else None,
            metadata=metadata,
        )
