from __future__ import annotations

import pytest

from alphaapollo.common.environment import (
    DefaultEnvironment,
    EnvironmentEventKind,
    EnvironmentRuntimeError,
    FakeToolBridge,
    RecordingEnvironmentEventSink,
    RuntimeEnvironmentAdapter,
    ToolBridgeResult,
)
from alphaapollo.common.execution import ToolRequest, ToolResponse
from alphaapollo.common.execution.tools.schemas import ToolCallRecord


def _tool_result() -> ToolBridgeResult:
    request = ToolRequest(
        call_id="runtime-tool-call",
        tool_id="python",
        arguments={"code": "6 * 7"},
        source="python_code",
    )
    return ToolBridgeResult(
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


def test_runtime_adapter_binds_resolved_identity_and_positional_init() -> None:
    events = RecordingEnvironmentEventSink()
    environment = DefaultEnvironment(
        tool_bridge=FakeToolBridge([ToolBridgeResult(model_output="FINAL-42")]),
        event_sink=events,
    )
    adapter = RuntimeEnvironmentAdapter(
        environment,
        session_id="aime-60-solver",
        actor="solver",
        branch_id="candidate-7/solver",
        workspace_snapshot_ref="solver-workspace",
        episode_context={"condition": "python_code"},
    )

    initial, metadata = adapter.init("Solve carefully.", "Compute six times seven.")
    final = adapter.step("FINAL-42")
    adapter.close()

    assert initial == "Compute six times seven."
    assert metadata["session"]["actor"] == "solver"
    assert metadata["session"]["branch_id"] == "candidate-7/solver"
    assert metadata["session"]["workspace_snapshot_ref"] == "solver-workspace"
    assert final["done"] is True
    assert final["observations"] == ""
    assert events.events[0].capture is not None
    assert events.events[0].capture["content"]["episode_context"] == {
        "condition": "python_code",
        "system_prompt": "Solve carefully.",
    }


def test_runtime_adapter_exposes_canonical_tool_exchange_metadata() -> None:
    environment = DefaultEnvironment(
        tool_bridge=FakeToolBridge([_tool_result(), ToolBridgeResult(model_output="FINAL-42")])
    )
    adapter = RuntimeEnvironmentAdapter(
        environment,
        session_id="tool-loop",
        actor="verifier",
    )
    adapter.init("Verify carefully.", "Candidate answer: 42")

    tool_step = adapter.step("<python_code>6 * 7</python_code>")
    final_step = adapter.step("VERDICT: PASS")

    assert tool_step["done"] is False
    assert tool_step["metadata"] is not None
    assert tool_step["metadata"]["tool_request"]["tool_id"] == "python"
    assert tool_step["metadata"]["tool_response"]["stdout"] == "42\n"
    assert tool_step["metadata"]["tool_record"]["tool_id"] == "python"
    assert final_step["done"] is True
    assert final_step["observations"] == ""


def test_runtime_adapter_preserves_environment_owned_non_final_termination() -> None:
    environment = DefaultEnvironment(
        tool_bridge=FakeToolBridge([_tool_result()]),
        max_steps=1,
    )
    adapter = RuntimeEnvironmentAdapter(
        environment,
        session_id="step-limit",
        actor="solver",
    )
    adapter.init("Solve.", "Problem.")

    result = adapter.step("<python_code>6 * 7</python_code>")

    assert result["done"] is True
    assert result["metadata"]["success"] is False
    assert result["metadata"]["termination_reason"] == "max_steps"
    assert environment.session is not None
    assert environment.session.termination_reason == "max_steps"


def test_runtime_adapter_raises_for_terminal_internal_error() -> None:
    environment = DefaultEnvironment(tool_bridge=FakeToolBridge([]))
    adapter = RuntimeEnvironmentAdapter(
        environment,
        session_id="internal-error",
        actor="solver",
    )
    adapter.init("Solve.", "Problem.")

    with pytest.raises(EnvironmentRuntimeError, match="internal_error") as error:
        adapter.step("<python_code>print(42)</python_code>")
    assert error.value.termination_reason == "internal_error"


def test_runtime_adapter_persists_runtime_termination_before_close() -> None:
    events = RecordingEnvironmentEventSink()
    environment = DefaultEnvironment(
        tool_bridge=FakeToolBridge([]),
        event_sink=events,
    )
    adapter = RuntimeEnvironmentAdapter(
        environment,
        session_id="turn-limit",
        actor="solver",
    )
    adapter.init("Solve.", "Problem.")

    adapter.terminate("max_turns")
    adapter.close()

    assert environment.session is not None
    assert environment.session.termination_reason == "max_turns"
    terminated = [event for event in events.events if event.kind is EnvironmentEventKind.TERMINATED]
    assert len(terminated) == 1
    assert terminated[0].payload["reason"] == "max_turns"
