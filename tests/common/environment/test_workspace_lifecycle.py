from __future__ import annotations

from typing import Any

import pytest

from alphaapollo.common.environment import (
    DefaultEnvironment,
    FakeToolBridge,
    RecordingEnvironmentEventSink,
    SensitiveEnvironmentInputError,
    ToolBridgeResult,
    WorkspaceLease,
)
from alphaapollo.common.execution import ToolRequest, ToolResponse
from alphaapollo.common.execution.tools.schemas import ToolCallRecord


class _ToyWorkspaceProvider:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, str, str]] = []
        self.released: list[str] = []

    def acquire(
        self,
        *,
        session_id: str,
        actor: str,
        branch_id: str,
    ) -> WorkspaceLease:
        self.acquired.append((session_id, actor, branch_id))
        ref = f"toy-workspace://{session_id}/{branch_id}/{actor}"
        return WorkspaceLease(ref, lambda: self.released.append(ref))


def test_solver_and_verifier_get_distinct_leases_and_independent_cleanup() -> None:
    workspaces = _ToyWorkspaceProvider()
    solver = DefaultEnvironment(
        tool_bridge=FakeToolBridge([]),
        workspace_provider=workspaces,
    )
    verifier = DefaultEnvironment(
        tool_bridge=FakeToolBridge([]),
        workspace_provider=workspaces,
    )

    _, solver_metadata = solver.init(
        session_id="shared-run",
        actor="solver",
        branch_id="solver",
    )
    _, verifier_metadata = verifier.init(
        session_id="shared-run",
        actor="verifier",
        branch_id="verifier",
    )
    solver_ref = solver_metadata["session"]["workspace_snapshot_ref"]
    verifier_ref = verifier_metadata["session"]["workspace_snapshot_ref"]
    assert solver_ref != verifier_ref

    solver.close()
    assert workspaces.released == [solver_ref]
    assert verifier.session is not None
    assert verifier.session.state.value == "active"
    verifier.close()
    assert workspaces.released == [solver_ref, verifier_ref]


def test_explicit_resolved_workspace_ref_bypasses_provider() -> None:
    workspaces = _ToyWorkspaceProvider()
    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge([]),
        workspace_provider=workspaces,
    )

    _, metadata = env.init(
        session_id="resolved",
        actor="solver",
        workspace_snapshot_ref="upstream-resolved-workspace",
    )
    env.close()

    assert metadata["session"]["workspace_snapshot_ref"] == ("upstream-resolved-workspace")
    assert workspaces.acquired == []
    assert workspaces.released == []


class _SensitiveWorkspaceProvider:
    def __init__(self) -> None:
        self.released = False

    def acquire(
        self,
        *,
        session_id: str,
        actor: str,
        branch_id: str,
    ) -> WorkspaceLease:
        return WorkspaceLease(
            "api_key=workspace-secret",
            lambda: setattr(self, "released", True),
        )


def test_sensitive_acquired_workspace_is_released_before_init_fails() -> None:
    workspaces = _SensitiveWorkspaceProvider()
    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge([]),
        workspace_provider=workspaces,
    )

    with pytest.raises(SensitiveEnvironmentInputError):
        env.init(session_id="sensitive", actor="solver")

    assert workspaces.released is True
    assert env.session is None


def test_acquired_workspace_is_released_when_session_validation_fails() -> None:
    workspaces = _ToyWorkspaceProvider()
    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge([]),
        workspace_provider=workspaces,
    )

    with pytest.raises(ValueError, match="session_id"):
        env.init(session_id="", actor="solver")

    assert workspaces.released == ["toy-workspace:///main/solver"]
    assert env.session is None


class _FailingInitSink(RecordingEnvironmentEventSink):
    def emit(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("trajectory unavailable")


def test_event_persistence_failure_is_nonfatal_observable_and_cleanup_still_runs() -> None:
    workspaces = _ToyWorkspaceProvider()
    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge([]),
        workspace_provider=workspaces,
        event_sink=_FailingInitSink(),
    )

    _, metadata = env.init(session_id="event-failure", actor="solver")

    assert metadata["session"]["state"] == "active"
    assert metadata["session"]["event_errors"] == [
        "environment_initialized: RuntimeError: event recording failed"
    ]
    assert workspaces.released == []
    env.close()
    assert workspaces.released == ["toy-workspace://event-failure/main/solver"]
    assert env.session is not None
    assert env.session.state.value == "closed"


def test_event_failures_during_step_are_returned_in_session_metadata() -> None:
    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge([ToolBridgeResult(model_output="final")]),
        event_sink=_FailingInitSink(),
    )
    env.init(session_id="step-event-failure", actor="solver")

    result = env.step("final")

    assert result["done"] is True
    assert result["observations"] == "final"
    errors = result["metadata"]["session"]["event_errors"]
    assert [error.split(":", 1)[0] for error in errors] == [
        "environment_initialized",
        "agent_action_received",
        "observation_emitted",
        "environment_terminated",
    ]
    assert env.session is not None
    assert env.session.state.value == "terminated"


def test_event_failures_do_not_change_a_tool_observation() -> None:
    request = ToolRequest(
        call_id="resolved-call",
        tool_id="toy",
        arguments={"value": 42},
    )
    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge(
            [
                ToolBridgeResult(
                    request=request,
                    response=ToolResponse(
                        call_id=request.call_id,
                        tool_id=request.tool_id,
                        stdout="42\n",
                    ),
                    record=ToolCallRecord(
                        tool_id=request.tool_id,
                        args=request.arguments,
                        stdout="42\n",
                    ),
                )
            ]
        ),
        event_sink=_FailingInitSink(),
    )
    env.init(session_id="tool-event-failure", actor="solver")

    result = env.step({"resolved_tool_action": True})

    assert result["done"] is False
    assert result["metadata"]["observation"]["kind"] == "tool_response"
    assert '"stdout": "42\\n"' in result["observations"]
    assert [error.split(":", 1)[0] for error in result["metadata"]["session"]["event_errors"]] == [
        "environment_initialized",
        "agent_action_received",
        "tool_request_normalized",
        "tool_execution_attempted",
        "tool_execution_completed",
        "observation_emitted",
    ]


def test_workspace_cleanup_failure_is_observable_and_retried_until_released() -> None:
    calls = 0

    def fail_release() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("release failed")

    class _Provider:
        def acquire(
            self,
            *,
            session_id: str,
            actor: str,
            branch_id: str,
        ) -> WorkspaceLease:
            return WorkspaceLease("workspace-with-broken-release", fail_release)

    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge([ToolBridgeResult(model_output="done")]),
        workspace_provider=_Provider(),
    )
    env.init(session_id="cleanup-failure", actor="solver")
    env.step("done")

    env.close()
    env.close()
    env.close()

    assert calls == 2
    assert env.session is not None
    assert env.session.cleanup_errors == ["RuntimeError: workspace cleanup failed"]


def test_execution_mode_and_timeout_are_resolved_inputs_not_env_config() -> None:
    bridge = FakeToolBridge([ToolBridgeResult(model_output="done")])
    env = DefaultEnvironment(
        tool_bridge=bridge,
        execution_mode="isolated",
        tool_timeout_s=17,
    )
    _, metadata = env.init(session_id="resolved-policy", actor="verifier")

    env.step("done")

    assert metadata["session"]["execution_mode"] == "isolated"
    assert bridge.contexts[0].mode == "isolated"
    assert bridge.contexts[0].timeout_s == 17


def test_workspace_is_released_after_internal_step_failure() -> None:
    workspaces = _ToyWorkspaceProvider()
    env = DefaultEnvironment(
        tool_bridge=FakeToolBridge([]),
        workspace_provider=workspaces,
    )
    env.init(session_id="step-failure", actor="solver")

    result = env.step("bridge has no configured result")
    env.close()

    assert result["done"] is True
    assert result["metadata"]["observation"]["kind"] == "environment_error"
    assert workspaces.released == ["toy-workspace://step-failure/main/solver"]


def test_context_manager_releases_workspace() -> None:
    workspaces = _ToyWorkspaceProvider()

    with DefaultEnvironment(
        tool_bridge=FakeToolBridge([]),
        workspace_provider=workspaces,
    ) as env:
        env.init(session_id="context-manager", actor="verifier")

    assert workspaces.released == ["toy-workspace://context-manager/main/verifier"]
