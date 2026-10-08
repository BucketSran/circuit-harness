from __future__ import annotations

from typing import Any

import pytest

from alphaapollo.common.environment.default import RoutedToolBridge, ToolBridgeResult, normalize
from alphaapollo.common.execution import ExecutionContext, ToolError, ToolRequest, ToolResponse
from alphaapollo.common.execution.tools import ToolCallRecord, ToolCatalog, ToolSpec


def _spec(tool_id: str) -> ToolSpec:
    return ToolSpec(
        tool_id=tool_id,
        description=f"Execute {tool_id}",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )


def _action(tool_id: str, value: object = "ok") -> dict[str, Any]:
    return {
        "id": f"call-{tool_id}",
        "function": {"name": tool_id, "arguments": {"value": value}},
    }


class _RecordingBridge:
    def __init__(self) -> None:
        self.requests: list[ToolRequest] = []
        self.close_count = 0

    def dispatch(self, action: Any, context: ExecutionContext) -> ToolBridgeResult:
        del context
        request = normalize(action)
        assert isinstance(request, ToolRequest)
        self.requests.append(request)
        response = ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stdout=request.arguments["value"],
        )
        return ToolBridgeResult(
            request=request,
            response=response,
            record=ToolCallRecord(tool_id=request.tool_id, stdout=response.stdout),
        )

    def close(self) -> None:
        self.close_count += 1


class _FailOnceCloseBridge(_RecordingBridge):
    def close(self) -> None:
        super().close()
        if self.close_count == 1:
            raise RuntimeError("close failed once")


CONTEXT = ExecutionContext(session_id="routed-tools")


def test_routes_multiple_tool_ids_and_validates_before_dispatch() -> None:
    first = _RecordingBridge()
    second = _RecordingBridge()
    catalog = ToolCatalog((_spec("first"), _spec("second")))
    bridge = RoutedToolBridge({"first": first, "second": second}, catalog=catalog)

    result = bridge.dispatch(_action("second", "value"), CONTEXT)
    rejected = bridge.dispatch(_action("first", 7), CONTEXT)

    assert result.response is not None
    assert result.response.stdout == "value"
    assert [request.tool_id for request in second.requests] == ["second"]
    assert first.requests == []
    assert isinstance(rejected.error, ToolError)
    assert rejected.error.code == "invalid_arguments"


def test_global_budget_applies_across_child_bridges() -> None:
    first = _RecordingBridge()
    second = _RecordingBridge()
    catalog = ToolCatalog((_spec("first"), _spec("second")))
    bridge = RoutedToolBridge(
        {"first": first, "second": second},
        catalog=catalog,
        max_tool_calls=1,
    )

    accepted = bridge.dispatch(_action("first"), CONTEXT)
    refused = bridge.dispatch(_action("second"), CONTEXT)

    assert accepted.attempted is True
    assert refused.refusal_code == "tool_budget_exhausted"
    assert second.requests == []


def test_ungranted_catalogued_tool_is_a_policy_error() -> None:
    selected = _RecordingBridge()
    catalog = ToolCatalog((_spec("selected"), _spec("not_granted")))
    bridge = RoutedToolBridge({"selected": selected}, catalog=catalog)

    result = bridge.dispatch(_action("not_granted"), CONTEXT)

    assert isinstance(result.error, ToolError)
    assert result.error.stage == "policy"
    assert result.error.code == "tool_not_granted"


def test_shared_child_is_closed_once_and_close_is_idempotent() -> None:
    shared = _RecordingBridge()
    catalog = ToolCatalog((_spec("first"), _spec("second")))
    bridge = RoutedToolBridge({"first": shared, "second": shared}, catalog=catalog)

    bridge.close()
    bridge.close()

    assert shared.close_count == 1
    assert bridge.closed is True


def test_close_retries_only_the_child_whose_cleanup_failed() -> None:
    failing = _FailOnceCloseBridge()
    healthy = _RecordingBridge()
    catalog = ToolCatalog((_spec("first"), _spec("second")))
    bridge = RoutedToolBridge({"first": failing, "second": healthy}, catalog=catalog)

    with pytest.raises(RuntimeError, match="close failed once"):
        bridge.close()
    assert bridge.closed is False
    assert failing.close_count == 1
    assert healthy.close_count == 1

    bridge.close()

    assert failing.close_count == 2
    assert healthy.close_count == 1
    assert bridge.closed is True


def test_plain_model_output_bypasses_tool_routing() -> None:
    child = _RecordingBridge()
    bridge = RoutedToolBridge({"first": child}, catalog=ToolCatalog((_spec("first"),)))

    result = bridge.dispatch("final answer", CONTEXT)

    assert result.model_output == "final answer"
    assert child.requests == []
