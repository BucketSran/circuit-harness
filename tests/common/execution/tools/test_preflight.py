from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from alphaapollo.common.environment.default.projection import prepare_tool_request
from alphaapollo.common.execution import ExecutionContext, ToolError, ToolRequest
from alphaapollo.common.execution.tools import (
    ExecutionPolicy,
    ToolCatalog,
)


def _tool_call(
    name: str,
    arguments: object,
    *,
    call_id: str = "call-1",
) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _context(**kwargs: object) -> ExecutionContext:
    return ExecutionContext(session_id="session", **kwargs)  # type: ignore[arg-type]


def _assert_error(result: object, stage: str, code: str) -> ToolError:
    assert isinstance(result, ToolError)
    assert result.stage == stage
    assert result.code == code
    return result


@dataclass
class OpenAICompatibleResponse:
    payload: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return self.payload


def test_sdk_shaped_openai_or_qwen_response_passes_complete_m1_preflight() -> None:
    response = OpenAICompatibleResponse(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            _tool_call(
                                "write",
                                '{"path":"notes.txt","content":"hello"}',
                                call_id="call-qwen",
                            )
                        ],
                    }
                }
            ]
        }
    )

    result = prepare_tool_request(response, _context())

    assert result == ToolRequest(
        call_id="call-qwen",
        tool_id="write",
        arguments={"path": "notes.txt", "content": "hello"},
        source="openai_tool_call",
    )


@pytest.mark.parametrize(
    ("tool_id", "arguments"),
    [
        ("read", {"path": "README.md"}),
        ("bash", {"command": "pwd"}),
        (
            "edit",
            {
                "path": "notes.txt",
                "edits": [{"oldText": "before", "newText": "after"}],
            },
        ),
        ("write", {"path": "notes.txt", "content": "hello"}),
        ("grep", {"pattern": "ToolRequest"}),
        ("find", {"pattern": "**/*.py"}),
        ("ls", {}),
    ],
)
def test_all_public_tools_pass_catalog_and_policy(
    tool_id: str, arguments: dict[str, object]
) -> None:
    result = prepare_tool_request(_tool_call(tool_id, arguments), _context())

    assert isinstance(result, ToolRequest)
    assert result.tool_id == tool_id
    assert result.arguments == arguments


def test_python_code_passes_catalog_and_policy_as_internal_compatibility_call() -> None:
    result = prepare_tool_request("<python_code>print(1)</python_code>", _context())

    assert isinstance(result, ToolRequest)
    assert result.tool_id == "python"
    assert result.arguments == {"code": "print(1)"}
    assert result.source == "python_code"


def test_internal_python_cannot_be_called_as_a_public_structured_tool() -> None:
    result = prepare_tool_request(_tool_call("python", '{"code":"print(1)"}'), _context())

    _assert_error(result, "policy", "internal_tool_forbidden")


@pytest.mark.parametrize(
    ("payload", "stage", "code"),
    [
        (_tool_call("missing", {}), "catalog", "unknown_tool"),
        (_tool_call("read", {}), "catalog", "invalid_arguments"),
        (_tool_call("read", {"path": "../secret"}), "policy", "path_forbidden"),
        (_tool_call("read", {"path": "README.md"}), "policy", "actor_forbidden"),
    ],
)
def test_unknown_invalid_and_forbidden_calls_return_typed_pre_execution_errors(
    payload: object, stage: str, code: str
) -> None:
    context = _context(actor="guest") if code == "actor_forbidden" else _context()

    _assert_error(prepare_tool_request(payload, context), stage, code)


def test_non_tool_output_bypasses_preflight() -> None:
    assert prepare_tool_request("ordinary final answer", _context()) is None


class UnexpectedCatalog(ToolCatalog):
    def resolve(self, request: ToolRequest) -> object:  # type: ignore[override]
        raise AssertionError("catalog must not run after a parse error")


class UnexpectedPolicy(ExecutionPolicy):
    def authorize(  # type: ignore[override]
        self,
        request: ToolRequest,
        context: ExecutionContext,
        spec: object,
    ) -> None:
        raise AssertionError("policy must not run before catalog success")


def test_parse_errors_short_circuit_before_catalog_or_policy() -> None:
    result = prepare_tool_request(
        _tool_call("read", "{"),
        _context(),
        catalog=UnexpectedCatalog(),
        policy=UnexpectedPolicy(),
    )

    _assert_error(result, "parse", "invalid_arguments_json")


def test_catalog_errors_short_circuit_before_policy() -> None:
    result = prepare_tool_request(
        _tool_call("missing", {}),
        _context(),
        policy=UnexpectedPolicy(),
    )

    _assert_error(result, "catalog", "unknown_tool")
