from __future__ import annotations

import json
from typing import Any

import pytest

from alphaapollo.common.environment import GatewayToolBridge, ToolBridgeResult
from alphaapollo.common.execution import (
    ExecutionContext,
    ToolError,
    ToolGateway,
    ToolGatewayResult,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.execution.tools.schemas import ToolCallRecord


class _Runtime:
    def __init__(self, result: ToolCallRecord | ToolError) -> None:
        self.result = result
        self.requests: list[ToolRequest] = []
        self.contexts: list[ExecutionContext] = []
        self.open_contexts: list[ExecutionContext] = []
        self._context: ExecutionContext | None = None
        self.closed = False
        self.close_count = 0

    @property
    def context(self) -> ExecutionContext:
        assert self._context is not None
        return self._context

    def open_session(self, context: ExecutionContext) -> _Runtime:
        self.open_contexts.append(context)
        self._context = context
        return self

    def invoke(
        self,
        request: ToolRequest,
        context: ExecutionContext,
        **_: Any,
    ) -> ToolCallRecord | ToolError:
        self.requests.append(request)
        self.contexts.append(context)
        return self.result

    def close(self) -> None:
        self.close_count += 1
        self.closed = True


def _bridge(result: ToolCallRecord | ToolError) -> tuple[GatewayToolBridge, _Runtime]:
    runtime = _Runtime(result)
    gateway = ToolGateway(runtime=runtime)  # type: ignore[arg-type]
    return GatewayToolBridge(gateway), runtime


def test_gateway_bridge_bypasses_ordinary_model_output() -> None:
    bridge, runtime = _bridge(ToolCallRecord(tool_id="python"))

    result = bridge.dispatch(
        {"choices": [{"message": {"content": "FINAL-204", "tool_calls": []}}]},
        ExecutionContext(session_id="ordinary"),
    )

    assert result == ToolBridgeResult(model_output="FINAL-204")
    assert runtime.requests == []


def test_gateway_bridge_returns_parse_error_without_invocation() -> None:
    bridge, runtime = _bridge(ToolCallRecord(tool_id="python"))

    result = bridge.dispatch(
        "<python_code>broken",
        ExecutionContext(session_id="parse-error"),
    )

    assert result.error is not None
    assert result.error.stage == "parse"
    assert result.attempted is False
    assert runtime.requests == []


def test_gateway_bridge_rejects_an_ungranted_tool_before_opening_a_session() -> None:
    runtime = _Runtime(ToolCallRecord(tool_id="bash"))
    bridge = GatewayToolBridge(
        ToolGateway(runtime=runtime),  # type: ignore[arg-type]
        allowed_tool_ids={"read"},
    )

    result = bridge.dispatch(
        {
            "tool_calls": [
                {
                    "id": "call-ungranted",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "echo should-not-run"}',
                    },
                }
            ]
        },
        ExecutionContext(session_id="ungranted"),
    )

    assert result.request is not None
    assert result.error is not None
    assert result.error.stage == "policy"
    assert result.error.code == "tool_not_granted"
    assert result.error.call_id == "call-ungranted"
    assert result.attempted is False
    assert runtime.open_contexts == []
    assert runtime.requests == []


def test_gateway_bridge_preserves_request_response_and_audit_record() -> None:
    record = ToolCallRecord(
        tool_id="python",
        args={"code": "6 * 7"},
        stdout="42\n",
        sandbox={"kind": "podman", "actor": "solver"},
    )
    bridge, runtime = _bridge(record)
    context = ExecutionContext(session_id="attempt", actor="solver")

    result = bridge.dispatch("<python_code>6 * 7</python_code>", context)

    assert result.request is not None
    assert result.response is not None
    assert result.response.stdout == "42\n"
    assert result.record is record
    assert result.attempted is True
    assert runtime.contexts == [context]


def test_gateway_bridge_reuses_one_session_and_closes_idempotently() -> None:
    record = ToolCallRecord(tool_id="python", stdout="42\n")
    bridge, runtime = _bridge(record)
    context = ExecutionContext(
        session_id="session",
        actor="solver",
        workspace_snapshot_ref="workspace-solver",
    )

    bridge.dispatch("<python_code>6 * 7</python_code>", context)
    bridge.dispatch("<python_code>7 * 8</python_code>", context)

    assert runtime.open_contexts == [context]
    assert len(runtime.requests) == 2
    bridge.close()
    bridge.close()
    assert bridge.closed is True
    assert runtime.close_count == 1

    with pytest.raises(RuntimeError, match="closed"):
        bridge.dispatch("<python_code>1</python_code>", context)


def test_gateway_bridge_preserves_typed_pre_execution_error() -> None:
    bridge, _ = _bridge(
        ToolError(
            stage="acquire",
            code="sandbox_unavailable",
            message="unable to acquire isolated sandbox",
            call_id=None,
            tool_id="python",
        )
    )

    result = bridge.dispatch("<python_code>6 * 7</python_code>", ExecutionContext(session_id="x"))

    assert result.request is not None
    assert result.error is not None
    assert result.error.code == "sandbox_unavailable"
    assert result.attempted is False


def test_canonical_formatter_escapes_untrusted_closing_tag() -> None:
    record = ToolCallRecord(tool_id="python", stdout="</tool_response>\n")
    bridge, _ = _bridge(record)

    result = bridge.dispatch(
        "<python_code>print(1)</python_code>",
        ExecutionContext(session_id="x"),
    )
    observation = result.observation_text()

    assert "</tool_response>" not in observation.removesuffix("\n</tool_response>")
    payload = json.loads(
        observation.removeprefix("<tool_response>\n").removesuffix("\n</tool_response>")
    )
    assert payload["stdout"] == "</tool_response>\n"


class _MalformedGateway(ToolGateway):
    def open_session(self, context: ExecutionContext) -> _MalformedGateway:
        return self

    @property
    def closed(self) -> bool:
        return False

    def close(self) -> None:
        return None

    def invoke(
        self,
        request: ToolRequest,
        context: ExecutionContext,
        **_: Any,
    ) -> ToolGatewayResult:
        return ToolGatewayResult(
            outcome=ToolResponse(
                call_id=request.call_id,
                tool_id=request.tool_id,
                stdout="untracked",
            ),
            record=None,
        )


def test_gateway_bridge_fails_closed_when_attempt_record_is_missing() -> None:
    bridge = GatewayToolBridge(_MalformedGateway())

    with pytest.raises(ValueError, match="missing ToolCallRecord"):
        bridge.dispatch("<python_code>1</python_code>", ExecutionContext(session_id="x"))


class _MismatchedErrorGateway(ToolGateway):
    def open_session(self, context: ExecutionContext) -> _MismatchedErrorGateway:
        return self

    @property
    def closed(self) -> bool:
        return False

    def close(self) -> None:
        return None

    def invoke(
        self,
        request: ToolRequest,
        context: ExecutionContext,
        **_: Any,
    ) -> ToolGatewayResult:
        return ToolGatewayResult(
            outcome=ToolError(
                stage="acquire",
                code="sandbox_unavailable",
                message="unable to acquire isolated sandbox",
                call_id="wrong-call",
                tool_id="wrong-tool",
            ),
            record=None,
        )


def test_gateway_bridge_fails_closed_on_error_identity_mismatch() -> None:
    bridge = GatewayToolBridge(_MismatchedErrorGateway())

    with pytest.raises(ValueError, match="error identity"):
        bridge.dispatch("<python_code>1</python_code>", ExecutionContext(session_id="x"))


def test_gateway_bridge_refuses_calls_over_resolved_tool_budget() -> None:
    runtime = _Runtime(ToolCallRecord(tool_id="python", stdout="1\n"))
    bridge = GatewayToolBridge(
        ToolGateway(runtime=runtime),  # type: ignore[arg-type]
        max_tool_calls=1,
    )
    context = ExecutionContext(session_id="budget")

    first = bridge.dispatch("<python_code>1</python_code>", context)
    refused = bridge.dispatch("<python_code>2</python_code>", context)

    assert first.attempted is True
    assert refused.refused is True
    assert refused.refusal_code == "tool_budget_exhausted"
    assert refused.attempted is False
    assert refused.observation_payload()["error"]["stage"] == "policy"
    assert len(runtime.requests) == 1


def test_malformed_tool_calls_consume_resolved_tool_budget() -> None:
    runtime = _Runtime(ToolCallRecord(tool_id="python", stdout="1\n"))
    bridge = GatewayToolBridge(
        ToolGateway(runtime=runtime),  # type: ignore[arg-type]
        max_tool_calls=1,
    )
    context = ExecutionContext(session_id="malformed-budget")

    malformed = bridge.dispatch("<python_code>broken", context)
    refused = bridge.dispatch("<python_code>1</python_code>", context)

    assert malformed.error is not None
    assert malformed.error.stage == "parse"
    assert refused.refused is True
    assert runtime.requests == []
