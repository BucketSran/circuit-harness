from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from alphaapollo.common.environment.default.projection import (
    normalize,
    normalize_tool_input,
)
from alphaapollo.common.execution import (
    ExecutionContext,
    ToolError,
    ToolExecutor,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.execution.tools import INTERNAL_PYTHON_TOOL_ID


def _tool_call(
    *,
    call_id: str = "call-1",
    name: str = "read",
    arguments: object = '{"path": "README.md"}',
) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _assert_parse_error(result: object, code: str) -> ToolError:
    assert isinstance(result, ToolError)
    assert result.stage == "parse"
    assert result.code == code
    return result


@dataclass
class FaithfulOpenAIResponse:
    payload: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return self.payload


class RecordingExecutor:
    def __init__(self) -> None:
        self.request: ToolRequest | None = None

    def execute(self, request: ToolRequest, context: ExecutionContext) -> ToolResponse:
        self.request = request
        return ToolResponse(call_id=request.call_id, tool_id=request.tool_id)


def test_normalize_name_and_descriptive_alias_share_one_ingress() -> None:
    assert normalize_tool_input is normalize


def test_direct_openai_tool_call_round_trips_provider_metadata_and_arguments() -> None:
    payload = _tool_call(call_id="provider-call-7", name="grep", arguments={"pattern": "x"})

    result = normalize(payload)

    assert result == ToolRequest(
        call_id="provider-call-7",
        tool_id="grep",
        arguments={"pattern": "x"},
        source="openai_tool_call",
    )
    assert isinstance(result, ToolRequest)


def test_full_openai_response_round_trips_json_string_arguments() -> None:
    payload = {
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _tool_call(
                            call_id="call-write",
                            name="write",
                            arguments='{"path":"notes.txt","content":"hello"}',
                        )
                    ],
                },
            }
        ],
    }

    result = normalize(payload)

    assert result == ToolRequest(
        call_id="call-write",
        tool_id="write",
        arguments={"path": "notes.txt", "content": "hello"},
        source="openai_tool_call",
    )


def test_sdk_response_crosses_ingress_executor_boundary_without_conversion() -> None:
    response = FaithfulOpenAIResponse(
        {
            "id": "chatcmpl-sdk",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_tool_call(call_id="call-sdk", name="ls", arguments={})],
                    },
                }
            ],
        }
    )
    request = normalize(response)
    executor = RecordingExecutor()

    assert isinstance(executor, ToolExecutor)
    assert isinstance(request, ToolRequest)
    result = executor.execute(request, ExecutionContext(session_id="session-sdk"))

    assert executor.request is request
    assert result.call_id == "call-sdk"
    assert result.tool_id == "ls"


def test_message_and_tool_calls_envelopes_are_supported() -> None:
    call = _tool_call()

    assert normalize({"message": {"tool_calls": [call]}}) == normalize({"tool_calls": [call]})


def test_python_code_round_trip_is_case_insensitive_and_deterministic() -> None:
    payload = "Before\n<PYTHON_CODE>\nprint('hello')\n</PYTHON_CODE>\nAfter"

    first = normalize(payload)
    second = normalize(payload)

    assert first == second
    assert isinstance(first, ToolRequest)
    assert first.call_id.startswith("python-code-")
    assert len(first.call_id) == len("python-code-") + 16
    assert first.tool_id == INTERNAL_PYTHON_TOOL_ID
    assert first.arguments == {"code": "print('hello')"}
    assert first.source == "python_code"


def test_ordinary_non_tool_output_bypasses_ingress() -> None:
    assert normalize("A normal final answer") is None
    assert normalize({"message": {"content": "A normal final answer"}}) is None


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"choices": []}, "multiple_choices"),
        ({"choices": [{}, {}]}, "multiple_choices"),
        ({"choices": "bad"}, "invalid_choices"),
        ({"message": "bad"}, "invalid_message"),
        ({"tool_calls": "bad"}, "invalid_tool_calls"),
        ({"tool_calls": ["bad"]}, "invalid_tool_call"),
        ({"tool_calls": []}, "unexpected"),
    ],
)
def test_invalid_envelopes_return_typed_parse_errors_or_bypass(payload: object, code: str) -> None:
    result = normalize(payload)  # type: ignore[arg-type]
    if code == "unexpected":
        assert result is None
    else:
        _assert_parse_error(result, code)


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (_tool_call(call_id=""), "missing_call_id"),
        ({"id": "call", "type": "custom", "function": {}}, "unsupported_tool_call_type"),
        ({"id": "call", "type": "function", "function": "bad"}, "invalid_function"),
        (_tool_call(name=""), "missing_tool_id"),
        (_tool_call(arguments="{"), "invalid_arguments_json"),
        (_tool_call(arguments='{"timeout": NaN}'), "invalid_arguments_json"),
        (_tool_call(arguments='{"timeout": Infinity}'), "invalid_arguments_json"),
        (_tool_call(arguments="[]"), "invalid_arguments"),
        (_tool_call(arguments=["not", "an", "object"]), "invalid_arguments"),
        (_tool_call(arguments={"timeout": math.nan}), "invalid_arguments_json_value"),
    ],
)
def test_malformed_function_calls_return_typed_parse_errors(
    call: dict[str, object], code: str
) -> None:
    _assert_parse_error(normalize(call), code)


def test_argument_parse_error_preserves_safe_call_identity() -> None:
    error = _assert_parse_error(
        normalize(_tool_call(call_id="call-bad", name="write", arguments="{")),
        "invalid_arguments_json",
    )

    assert error.call_id == "call-bad"
    assert error.tool_id == "write"


def test_multiple_structured_calls_are_rejected_as_upper_layer_control_flow() -> None:
    result = normalize({"tool_calls": [_tool_call(call_id="one"), _tool_call(call_id="two")]})

    _assert_parse_error(result, "multiple_tool_calls")


def test_structured_call_and_python_token_are_rejected_together() -> None:
    payload = {
        "content": "<python_code>print(1)</python_code>",
        "tool_calls": [_tool_call()],
    }

    _assert_parse_error(normalize(payload), "multiple_tool_calls")


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ("<python_code>print(1)", "malformed_python_code"),
        ("print(1)</python_code>", "malformed_python_code"),
        ("<python_code></python_code>", "empty_python_code"),
        (
            "<python_code>print(1)</python_code><python_code>print(2)</python_code>",
            "multiple_tool_calls",
        ),
        ("<python_code><python_code>x</python_code></python_code>", "malformed_python_code"),
    ],
)
def test_malformed_or_multiple_python_tokens_return_typed_errors(payload: str, code: str) -> None:
    _assert_parse_error(normalize(payload), code)


def test_unsupported_payload_type_is_a_typed_error() -> None:
    _assert_parse_error(normalize(42), "unsupported_payload")  # type: ignore[arg-type]
