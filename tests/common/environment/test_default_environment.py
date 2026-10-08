from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alphaapollo.common.environment import (
    DefaultEnvironment,
    EnvironmentEventKind,
    EnvironmentLifecycleError,
    EnvironmentState,
    ExecutorToolBridge,
    FakeToolBridge,
    RecordingEnvironmentEventSink,
    ToolBridgeResult,
    TrajectoryEnvironmentEventSink,
)
from alphaapollo.common.execution import (
    ArtifactStore,
    ExecutionContext,
    ToolError,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.execution.tools.schemas import ToolCallRecord
from alphaapollo.common.trajectory import TrajectoryStore
from tests.common.environment.fakes import CanonicalTestToolBridge


class _PythonExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[ToolRequest, ExecutionContext]] = []

    def execute(
        self,
        request: ToolRequest,
        context: ExecutionContext,
    ) -> ToolResponse:
        self.calls.append((request, context))
        return ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stdout="204\n",
        )


class _ClosableBridge(FakeToolBridge):
    def __init__(self) -> None:
        super().__init__([])
        self.closed = False
        self.close_count = 0

    def close(self) -> None:
        if self.closed:
            return
        self.close_count += 1
        self.closed = True


def test_fake_agent_runtime_can_continue_after_tool_and_finish() -> None:
    executor = _PythonExecutor()
    events = RecordingEnvironmentEventSink()
    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(executor),
        event_sink=events,
        max_steps=3,
    )

    initial, metadata = env.init(
        session_id="aime-60-solver",
        actor="solver",
        initial_observation="Solve the problem.",
    )
    assert initial == "Solve the problem."
    assert metadata["session"]["state"] == "active"

    tool_step = env.step("<python_code>print(204)</python_code>")
    assert tool_step["done"] is False
    assert '"stdout": "204\\n"' in tool_step["observations"]
    assert tool_step["metadata"]["observation"]["kind"] == "tool_response"
    assert tool_step["metadata"]["tool_request"]["arguments"] == {"code": "print(204)"}
    assert tool_step["metadata"]["tool_response"]["stdout"] == "204\n"
    assert tool_step["metadata"]["tool_record"]["tool_id"] == "python"
    assert "tool_error" not in tool_step["metadata"]
    request, context = executor.calls[0]
    assert request.arguments == {"code": "print(204)"}
    assert context.actor == "solver"

    final_step = env.step("The answer is 204.")
    assert final_step["done"] is True
    assert final_step["observations"] == "The answer is 204."
    assert "tool_request" not in final_step["metadata"]
    assert env.state is EnvironmentState.TERMINATED
    with pytest.raises(EnvironmentLifecycleError):
        env.step("another action")

    env.close()
    env.close()
    assert env.state is EnvironmentState.CLOSED
    assert [event.kind for event in events.events] == [
        EnvironmentEventKind.INITIALIZED,
        EnvironmentEventKind.ACTION_RECEIVED,
        EnvironmentEventKind.TOOL_REQUESTED,
        EnvironmentEventKind.EXECUTION_ATTEMPTED,
        EnvironmentEventKind.EXECUTION_COMPLETED,
        EnvironmentEventKind.OBSERVATION_EMITTED,
        EnvironmentEventKind.ACTION_RECEIVED,
        EnvironmentEventKind.OBSERVATION_EMITTED,
        EnvironmentEventKind.TERMINATED,
        EnvironmentEventKind.CLOSED,
    ]


def test_preflight_failure_is_structured_and_not_attempted() -> None:
    env = DefaultEnvironment(tool_bridge=CanonicalTestToolBridge(_PythonExecutor()))
    env.init(session_id="bad-call", actor="solver")

    result = env.step("<python_code></python_code>")

    assert result["done"] is False
    observation = result["metadata"]["observation"]
    assert observation["kind"] == "tool_error"
    assert observation["attempted"] is False
    assert observation["error_stage"] == "parse"
    assert observation["error_code"] == "empty_python_code"
    assert '"attempted": false' in result["observations"]
    assert result["metadata"]["tool_request"]["stage"] == "parse"
    assert result["metadata"]["tool_error"] == {
        "stage": "parse",
        "code": "empty_python_code",
        "message": "<python_code> must contain source code",
        "attempted": False,
    }
    assert "tool_response" not in result["metadata"]
    assert "tool_record" not in result["metadata"]
    assert result.response_format_valid is False
    assert result.env_action_valid is False
    assert result.action_valid is False


def test_sandbox_acquisition_failure_is_structured_and_not_attempted() -> None:
    bridge = FakeToolBridge(
        [
            ToolBridgeResult(
                error=ToolError(
                    stage="acquire",
                    code="sandbox_unavailable",
                    message="sandbox unavailable",
                    call_id="call-acquire",
                    tool_id="python",
                )
            )
        ]
    )
    env = DefaultEnvironment(tool_bridge=bridge)
    env.init(session_id="acquire-failure", actor="solver")

    result = env.step("<python_code>print(1)</python_code>")

    observation = result["metadata"]["observation"]
    assert observation["kind"] == "tool_error"
    assert observation["attempted"] is False
    assert observation["error_stage"] == "acquire"
    assert observation["error_code"] == "sandbox_unavailable"
    assert result.response_format_valid is True
    assert result.env_action_valid is True
    assert result.action_valid is True


def test_tool_budget_refusal_continues_without_counting_as_failure() -> None:
    events = RecordingEnvironmentEventSink()
    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge(
            [
                ToolBridgeResult(
                    refusal_code="tool_budget_exhausted",
                    refusal_message=("tool budget exhausted; provide your final answer now"),
                )
            ]
        ),
        event_sink=events,
    )
    env.init(session_id="budget-refusal", actor="solver")

    result = env.step("<python_code>print(1)</python_code>")

    assert result["done"] is False
    assert result["metadata"]["observation"]["kind"] == "tool_refusal"
    assert result["metadata"]["tool_refused"] == "tool_budget_exhausted"
    assert "tool_request" not in result["metadata"]
    assert "tool_error" not in result["metadata"]
    assert result.response_format_valid is True
    assert result.env_action_valid is False
    assert result.action_valid is False
    assert [event.kind for event in events.events][-2:] == [
        EnvironmentEventKind.TOOL_REFUSED,
        EnvironmentEventKind.OBSERVATION_EMITTED,
    ]


def test_nonzero_execution_is_structured() -> None:
    request = ToolRequest(
        call_id="call-1",
        tool_id="python",
        arguments={"code": "raise RuntimeError"},
        source="python_code",
    )
    cases = [
        ToolBridgeResult(
            request=request,
            response=ToolResponse(
                call_id="call-1",
                tool_id="python",
                stderr="boom",
                exit_code=1,
            ),
            record=ToolCallRecord(
                tool_id="python",
                args=request.arguments,
                stderr="boom",
                exit_code=1,
            ),
        ),
    ]
    env = DefaultEnvironment(tool_bridge=FakeToolBridge(cases))
    env.init(session_id="failure", actor="verifier")

    result = env.step("ignored by deterministic fake")

    observation = result["metadata"]["observation"]
    assert observation["kind"] == "tool_error"
    assert observation["attempted"] is True
    assert observation["exit_code"] == 1
    assert observation["error_stage"] == "execute"
    assert observation["error_code"] == "nonzero_exit"
    assert result["metadata"]["tool_error"]["code"] == "nonzero_exit"
    assert result["metadata"]["tool_error"]["attempted"] is True
    assert result["metadata"]["tool_record"]["exit_code"] == 1
    assert '"code": "nonzero_exit"' in result["observations"]
    assert '"stage": "execute"' in result["observations"]
    assert '"stderr": "boom"' in result["observations"]
    assert result["done"] is False
    assert result.response_format_valid is True
    assert result.env_action_valid is True
    assert result.action_valid is True


def test_policy_rejection_is_an_invalid_environment_action() -> None:
    request = ToolRequest(
        call_id="call-policy",
        tool_id="python",
        arguments={"code": "print(1)"},
        source="python_code",
    )
    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge(
            [
                ToolBridgeResult(
                    request=request,
                    error=ToolError(
                        stage="policy",
                        code="tool_not_granted",
                        message="tool is not granted",
                        call_id=request.call_id,
                        tool_id=request.tool_id,
                    ),
                )
            ]
        )
    )
    env.init(session_id="policy-rejection", actor="solver")

    result = env.step("ignored by deterministic fake")

    assert result.response_format_valid is True
    assert result.env_action_valid is False
    assert result.action_valid is False


@pytest.mark.parametrize("feedback_mode", ["generic", "missing"])
def test_feedback_ablation_keeps_failed_tool_turn_non_terminal(
    feedback_mode: str,
) -> None:
    request = ToolRequest(
        call_id="call-ablation",
        tool_id="python",
        arguments={"code": "raise RuntimeError"},
        source="python_code",
    )
    result = ToolBridgeResult(
        request=request,
        response=ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stderr="secret host detail",
            exit_code=1,
        ),
        record=ToolCallRecord(
            tool_id=request.tool_id,
            args=request.arguments,
            stderr="secret host detail",
            exit_code=1,
        ),
    )
    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge([result]),
        feedback_mode=feedback_mode,  # type: ignore[arg-type]
    )
    env.init(session_id=f"feedback-{feedback_mode}", actor="solver")

    step = env.step("<python_code>raise RuntimeError</python_code>")

    assert step["done"] is False
    assert step["metadata"]["observation"]["kind"] == "tool_error"
    assert "secret host detail" not in step["observations"]
    if feedback_mode == "generic":
        assert "generic_failure" in step["observations"]
    else:
        assert '"attempted": true' in step["observations"]
        assert "nonzero_exit" not in step["observations"]


def test_timeout_is_structured_and_does_not_expose_exception_detail() -> None:
    class _TimeoutExecutor:
        def execute(
            self,
            request: ToolRequest,
            context: ExecutionContext,
        ) -> ToolResponse:
            raise TimeoutError("host path /secret/answer-key")

    env = DefaultEnvironment(tool_bridge=CanonicalTestToolBridge(_TimeoutExecutor()))
    env.init(session_id="timeout", actor="solver")

    result = env.step("<python_code>while True: pass</python_code>")

    observation = result["metadata"]["observation"]
    assert observation["kind"] == "tool_error"
    assert observation["attempted"] is True
    assert observation["error_stage"] == "timeout"
    assert observation["error_code"] == "execution_timeout"
    assert result["metadata"]["tool_error"]["code"] == "execution_timeout"
    assert result["metadata"]["tool_record"]["exit_code"] == 124
    assert "/secret/answer-key" not in result["observations"]


def test_cancellation_is_structured_and_retains_attempt_record() -> None:
    request = ToolRequest(
        call_id="cancelled-call",
        tool_id="python",
        arguments={"code": "print(1)"},
        source="python_code",
    )
    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge(
            [
                ToolBridgeResult(
                    request=request,
                    response=ToolResponse(
                        call_id=request.call_id,
                        tool_id=request.tool_id,
                        stderr="command cancelled",
                        exit_code=130,
                    ),
                    record=ToolCallRecord(
                        tool_id=request.tool_id,
                        args=request.arguments,
                        stderr="command cancelled",
                        exit_code=130,
                    ),
                )
            ]
        )
    )
    env.init(session_id="cancelled", actor="solver")

    result = env.step("ignored by deterministic fake")

    observation = result["metadata"]["observation"]
    assert observation["kind"] == "tool_error"
    assert observation["attempted"] is True
    assert observation["error_stage"] == "execute"
    assert observation["error_code"] == "execution_cancelled"
    assert result["metadata"]["tool_error"]["code"] == "execution_cancelled"
    assert result["metadata"]["tool_record"]["exit_code"] == 130


def test_executor_response_identity_mismatch_fails_closed() -> None:
    class _MismatchedExecutor:
        def execute(
            self,
            request: ToolRequest,
            context: ExecutionContext,
        ) -> ToolResponse:
            return ToolResponse(
                call_id="wrong-call",
                tool_id="wrong-tool",
                stdout="untrusted output",
            )

    env = DefaultEnvironment(tool_bridge=CanonicalTestToolBridge(_MismatchedExecutor()))
    env.init(session_id="mismatch", actor="solver")

    result = env.step("<python_code>print(204)</python_code>")

    observation = result["metadata"]["observation"]
    assert result["done"] is True
    assert result.success is False
    assert observation["kind"] == "environment_error"
    assert observation["error_stage"] == "internal"
    assert observation["error_code"] == "environment_internal_error"
    assert "wrong-call" not in result["observations"]
    assert "wrong-tool" not in result["observations"]
    assert "untrusted output" not in result["observations"]


def test_no_tool_episode_terminates_without_calling_executor() -> None:
    executor = _PythonExecutor()
    env = DefaultEnvironment(tool_bridge=CanonicalTestToolBridge(executor))
    env.init(session_id="no-tool", actor="verifier")

    result = env.step("PASS")

    assert result["done"] is True
    assert result["observations"] == "PASS"
    assert executor.calls == []


def test_openai_compatible_final_response_returns_message_content() -> None:
    executor = _PythonExecutor()
    env = DefaultEnvironment(tool_bridge=CanonicalTestToolBridge(executor))
    env.init(session_id="provider-final", actor="solver")
    response = {
        "choices": [
            {
                "message": {
                    "content": "FINAL-204",
                    "tool_calls": [],
                }
            }
        ]
    }

    result = env.step(response)

    assert result["done"] is True
    assert result["observations"] == "FINAL-204"
    assert executor.calls == []


def test_cleanup_failure_is_recorded_and_close_is_idempotent() -> None:
    events = RecordingEnvironmentEventSink()

    def fail_cleanup(session: Any) -> None:
        raise RuntimeError("release failed")

    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge([]),
        event_sink=events,
        cleanup=fail_cleanup,
    )
    env.init(session_id="cleanup", actor="solver")

    env.close()
    env.close()

    assert env.state is EnvironmentState.CLOSED
    assert env.session is not None
    assert env.session.cleanup_errors == ["RuntimeError: cleanup failed"]
    assert events.events[-1].payload["cleanup_errors"] == ["RuntimeError: cleanup failed"]


def test_environment_close_releases_optional_tool_bridge_lifecycle() -> None:
    bridge = _ClosableBridge()
    env = DefaultEnvironment(tool_bridge=bridge)
    env.init(session_id="bridge-cleanup", actor="solver")

    env.close()
    env.close()

    assert bridge.closed is True
    assert bridge.close_count == 1


def test_environment_close_before_init_releases_tool_bridge() -> None:
    bridge = _ClosableBridge()
    env = DefaultEnvironment(tool_bridge=bridge)

    env.close()
    env.close()

    assert env.state is EnvironmentState.CLOSED
    assert bridge.close_count == 1


def test_executor_tool_bridge_close_is_idempotent() -> None:
    class ClosableExecutor(_PythonExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    executor = ClosableExecutor()
    bridge = ExecutorToolBridge(executor)
    env = DefaultEnvironment(tool_bridge=bridge)
    env.init(session_id="executor-cleanup", actor="solver")

    env.close()
    env.close()

    assert executor.close_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        bridge.dispatch(
            "answer",
            ExecutionContext(session_id="executor-cleanup", actor="solver"),
        )


def test_executor_tool_bridge_failed_close_can_be_retried() -> None:
    class RetryableExecutor(_PythonExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1
            if self.close_count == 1:
                raise RuntimeError("temporary cleanup failure")

    executor = RetryableExecutor()
    bridge = ExecutorToolBridge(executor)

    with pytest.raises(RuntimeError, match="temporary cleanup failure"):
        bridge.close()
    bridge.close()
    bridge.close()

    assert executor.close_count == 2


def test_solver_and_verifier_sessions_are_isolated() -> None:
    solver = DefaultEnvironment(tool_bridge=FakeToolBridge([]))
    verifier = DefaultEnvironment(tool_bridge=FakeToolBridge([]))
    solver.init(
        session_id="solver-session",
        actor="solver",
        workspace_snapshot_ref="workspace-solver",
    )
    verifier.init(
        session_id="verifier-session",
        actor="verifier",
        workspace_snapshot_ref="workspace-verifier",
    )

    assert solver.session is not verifier.session
    assert solver.session is not None
    assert verifier.session is not None
    assert solver.session.workspace_snapshot_ref != verifier.session.workspace_snapshot_ref


def test_trajectory_sink_preserves_order_and_omits_action_content(tmp_path: Path) -> None:
    store = TrajectoryStore(
        jsonl_path=tmp_path / "events.jsonl",
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_PythonExecutor()),
        event_sink=TrajectoryEnvironmentEventSink(store),
    )
    env.init(session_id="trajectory", actor="solver")
    secret_marker = "expected_answer=204"
    env.step(f"<python_code>print(204) # {secret_marker}</python_code>")
    env.close()

    events = store.read_events()
    assert [event.sequence for event in events] == list(range(len(events)))
    serialized = "\n".join(event.model_dump_json() for event in events)
    assert secret_marker not in serialized
    assert all(event.payload["actor"] == "solver" for event in events)
    store.close()


def test_lifecycle_rejects_step_before_init_and_reinit() -> None:
    env = DefaultEnvironment(tool_bridge=FakeToolBridge([]))
    with pytest.raises(EnvironmentLifecycleError):
        env.step("answer")
    env.init(session_id="once", actor="solver")
    with pytest.raises(EnvironmentLifecycleError):
        env.init(session_id="twice", actor="solver")
