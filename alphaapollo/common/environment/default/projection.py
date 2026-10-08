# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"Projection between model actions and typed execution requests and responses."

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from alphaapollo.common.artifacts.schemas import require_json_value
from alphaapollo.common.execution.output import format_truncated_output, truncate_tail
from alphaapollo.common.execution.tools.base import (
    INTERNAL_PYTHON_TOOL_ID,
    ExecutionContext,
    ToolError,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.execution.tools.gateway import ExecutionPolicy
from alphaapollo.common.execution.tools.registry import ToolCatalog

# Per-stream ceiling on what one tool result may contribute to the next prompt.
#
# The execution layer already bounds a stream at ``DEFAULT_MAX_BYTES`` (50KB),
# which is the right size for an artifact on disk and far too large for a
# retained observation: every tool result stays in the prompt for the rest of
# the episode, so the episode has to satisfy
#
#     base_prompt + max_turns * max_tokens + max_tool_calls * observation <= max_model_len
#
# Measured against a ``max_model_len`` of 65,536 on the two AIME 2026 runs where
# the solver is told it holds a tool and actually calls one (``runs/viz_cli_tools``
# and ``runs/viz_py_tools``, six cells, 34 role-steps, 17 calls): first-turn
# prompts 387-2,449 tokens (2,500 reserved here) and largest solver completion
# 3,754. The number that sizes this constant is the last one -- real
# ``<tool_response>`` payloads tokenize at 1.72 to 5.48 bytes per token on those
# runs (1.64 at the densest ever recorded, which is the floor used below), so the
# dense end of the range is what the bound has to survive, not the average.
# A 50KB stream is therefore worth up to 50 * 1024 / 1.64 ~= 31,000 tokens: one
# command that prints an array overruns the context on its own and the server
# rejects the request.
#
# The cap is one number for both tool paths on purpose. ``bash`` and the internal
# ``<python_code>`` entry differ sharply in how often a *call* fails -- 4 of 9
# versus 0 of 8 on those runs, because a ``<python_code>`` block cannot be
# mis-sent as raw Python to a shell -- but that is a property of how the action
# is projected, not of how large the result is. Result sizes are near-identical
# (174-538 bytes for bash, 174-306 for python), and both paths are retained
# through the same ``tool_response_payload`` below, so one cap is correct. The
# failure asymmetry argues for the raised ``max_tool_calls`` instead: the shell
# path needs retries the python path does not.
#
# Solving the bound for the shipped budgets in ``configs/environment/default.yaml``
# and the AIME presets (max_turns 8, max_tokens 6,000, max_tool_calls 6):
#
#     2,500 + 8 * 6,000 + 6 * observation <= 65,536  =>  observation <= 2,506 tokens
#
# An observation carries both streams plus the JSON envelope, so with ~150
# tokens of envelope each stream may contribute ~1,178 tokens, i.e. about
# 1,932 bytes at the 1.64 B/token worst case. 1,024 rounds that down to a power
# of two: both streams then cost ceil(2 * 1024 / 1.64) + 150 = 1,399 tokens, so
# six calls cost 8,394 and the episode lands at 2,500 + 48,000 + 8,394 = 58,894
# of 65,536 -- 6,642 tokens, 10.1%, of headroom. For reference the largest whole
# observation this workload ever produced was 538 bytes, so the cap sits roughly
# 4x above real results.
#
# ``truncate_tail`` keeps the *end* of the stream, which is where a computation
# prints its answer, and ``format_truncated_output`` appends a marker naming the
# bytes dropped -- a model that cannot tell it got a partial result would trust
# a partial answer, so the truncation is always visible rather than silent.
MAX_OBSERVATION_STREAM_BYTES = 1024

_PYTHON_CODE_BLOCK_RE = re.compile(
    r"<python_code\s*>(?P<code>.*?)</python_code\s*>",
    re.IGNORECASE | re.DOTALL,
)
_PYTHON_CODE_OPEN_RE = re.compile(r"<python_code\s*>", re.IGNORECASE)
_PYTHON_CODE_CLOSE_RE = re.compile(r"</python_code\s*>", re.IGNORECASE)

ToolNormalizationResult = ToolRequest | ToolError | None
ToolPreflightResult = ToolRequest | ToolError | None


def normalize(payload: Any) -> ToolNormalizationResult:
    """Normalize exactly one tool request, or bypass ordinary non-tool output.

    Supported structured shapes are an OpenAI-compatible response, message,
    ``tool_calls`` envelope, or one direct function call. SDK response objects
    are accepted through a provider-neutral ``model_dump()`` boundary, so
    provider packages are deliberately not imported.
    """
    if isinstance(payload, str):
        return _normalize_python_code(payload)
    raw = _as_mapping(payload)
    if raw is None:
        return _parse_error(
            "unsupported_payload",
            "tool input must be text, a mapping, or expose model_dump() returning a mapping",
        )

    message_or_error = _extract_message(raw)
    if isinstance(message_or_error, ToolError):
        return message_or_error
    message = message_or_error

    calls_or_error = _extract_tool_calls(message)
    if isinstance(calls_or_error, ToolError):
        return calls_or_error
    calls = calls_or_error

    content = message.get("content")
    content_result: ToolNormalizationResult = None
    if isinstance(content, str):
        content_result = _normalize_python_code(content)

    if calls and content_result is not None:
        return _parse_error(
            "multiple_tool_calls",
            "structured tool_calls and <python_code> cannot appear in the same response",
        )
    if not calls:
        return content_result
    if len(calls) != 1:
        return _parse_error(
            "multiple_tool_calls",
            "single-turn execution accepts exactly one tool call",
        )
    return _normalize_function_call(calls[0])


def prepare_tool_request(
    payload: Any,
    context: ExecutionContext,
    *,
    catalog: ToolCatalog | None = None,
    policy: ExecutionPolicy | None = None,
) -> ToolPreflightResult:
    """Project, resolve, and authorize one model action before execution."""

    normalized = normalize(payload)
    if normalized is None or isinstance(normalized, ToolError):
        return normalized
    active_catalog = catalog or ToolCatalog()
    spec = active_catalog.resolve(normalized)
    if isinstance(spec, ToolError):
        return spec
    return (policy or ExecutionPolicy()).authorize(normalized, context, spec) or normalized


def _extract_message(payload: Mapping[str, Any]) -> Mapping[str, Any] | ToolError:
    choices = payload.get("choices")
    if choices is not None:
        if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
            return _parse_error("invalid_choices", "choices must be a sequence")
        if len(choices) != 1:
            return _parse_error(
                "multiple_choices",
                "single-turn execution accepts exactly one model choice",
            )
        choice = choices[0]
        if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
            return _parse_error("invalid_message", "choice.message must be a mapping")
        return choice["message"]

    message = payload.get("message")
    if message is not None:
        if not isinstance(message, Mapping):
            return _parse_error("invalid_message", "message must be a mapping")
        return message
    return payload


def _extract_tool_calls(message: Mapping[str, Any]) -> list[Mapping[str, Any]] | ToolError:
    if "function" in message:
        return [message]
    calls = message.get("tool_calls")
    if calls is None:
        return []
    if isinstance(calls, (str, bytes)) or not isinstance(calls, Sequence):
        return _parse_error("invalid_tool_calls", "tool_calls must be a sequence")
    normalized: list[Mapping[str, Any]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            return _parse_error("invalid_tool_call", "each tool call must be a mapping")
        normalized.append(call)
    return normalized


def _normalize_function_call(call: Mapping[str, Any]) -> ToolRequest | ToolError:
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        return _parse_error("missing_call_id", "provider tool call id must be non-empty")
    call_type = call.get("type", "function")
    if call_type != "function":
        return _parse_error(
            "unsupported_tool_call_type",
            "only function tool calls are supported",
            call_id=call_id,
        )

    function = call.get("function")
    if not isinstance(function, Mapping):
        return _parse_error(
            "invalid_function", "tool call function must be a mapping", call_id=call_id
        )
    tool_id = function.get("name")
    if not isinstance(tool_id, str) or not tool_id.strip():
        return _parse_error("missing_tool_id", "function name must be non-empty", call_id=call_id)

    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError):
            return _parse_error(
                "invalid_arguments_json",
                "function arguments must be strict JSON",
                call_id=call_id,
                tool_id=tool_id,
            )
    if not isinstance(arguments, dict):
        return _parse_error(
            "invalid_arguments",
            "function arguments must decode to an object",
            call_id=call_id,
            tool_id=tool_id,
        )
    try:
        require_json_value(arguments, path="$.arguments")
    except ValueError as exc:
        return _parse_error(
            "invalid_arguments_json_value",
            str(exc),
            call_id=call_id,
            tool_id=tool_id,
        )
    return ToolRequest(
        call_id=call_id,
        tool_id=tool_id,
        arguments=arguments,
        source="openai_tool_call",
    )


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def _as_mapping(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, Mapping):
        return payload
    model_dump = getattr(payload, "model_dump", None)
    if not callable(model_dump):
        return None
    try:
        dumped = model_dump()
    except (TypeError, ValueError):
        return None
    return dumped if isinstance(dumped, Mapping) else None


def _normalize_python_code(text: str) -> ToolNormalizationResult:
    matches = list(_PYTHON_CODE_BLOCK_RE.finditer(text))
    open_count = len(_PYTHON_CODE_OPEN_RE.findall(text))
    close_count = len(_PYTHON_CODE_CLOSE_RE.findall(text))

    if not matches and open_count == 0 and close_count == 0:
        return None
    if open_count != close_count or len(matches) != open_count:
        return _parse_error(
            "malformed_python_code",
            "<python_code> must have exactly one matching closing tag",
        )
    if len(matches) != 1:
        return _parse_error(
            "multiple_tool_calls",
            "single-turn execution accepts exactly one <python_code> block",
        )

    code = matches[0].group("code").strip()
    if not code:
        return _parse_error("empty_python_code", "<python_code> must contain source code")
    raw_block = matches[0].group(0)
    digest = hashlib.sha256(raw_block.encode("utf-8")).hexdigest()[:16]
    return ToolRequest(
        call_id=f"python-code-{digest}",
        tool_id=INTERNAL_PYTHON_TOOL_ID,
        arguments={"code": code},
        source="python_code",
    )


def _parse_error(
    code: str,
    message: str,
    *,
    call_id: str | None = None,
    tool_id: str | None = None,
) -> ToolError:
    return ToolError(
        stage="parse",
        code=code,
        message=message,
        call_id=call_id,
        tool_id=tool_id,
    )


# Descriptive compatibility name for callers that prefer an explicit verb phrase.
normalize_tool_input = normalize


FeedbackMode = Literal["differentiated", "generic", "missing"]
FEEDBACK_MODES = frozenset({"differentiated", "generic", "missing"})


def _validate_feedback_mode(feedback_mode: str) -> FeedbackMode:
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(
            f"unknown feedback_mode {feedback_mode!r}; expected one of {sorted(FEEDBACK_MODES)}"
        )
    return feedback_mode  # type: ignore[return-value]


def _mode_for_tool(feedback_mode: FeedbackMode, tool_id: str | None) -> FeedbackMode:
    """Apply the #170 ablation only to the internal Python tool.

    A workflow may grant Python alongside public tools such as Bash or read. Those
    tools keep their production differentiated failures in every ablation arm;
    otherwise a Bash failure could be mislabeled as ``Python execution failed``.
    """
    if tool_id == INTERNAL_PYTHON_TOOL_ID:
        return feedback_mode
    return "differentiated"


def execution_failure(exit_code: int) -> tuple[str, str] | None:
    """Classify an attempted process result for the model-facing protocol."""
    if exit_code == 0:
        return None
    if exit_code == 124:
        return "timeout", "execution_timeout"
    if exit_code == 130:
        return "execute", "execution_cancelled"
    return "execute", "nonzero_exit"


def bound_observation_stream(
    stream: str,
    text: str,
    *,
    max_bytes: int = MAX_OBSERVATION_STREAM_BYTES,
) -> str:
    """Bound one retained tool stream, leaving a marker when anything is dropped.

    The line budget is set to ``max_bytes`` so it can never bind first: a stream
    of N bytes holds at most N lines, so bytes -- the thing that costs prompt
    tokens -- is always the reason a result is cut.
    """
    if not isinstance(text, str):
        raise TypeError("observation stream must be text")
    result = truncate_tail(text, max_lines=max_bytes, max_bytes=max_bytes)
    if not result.truncated:
        return text
    return format_truncated_output(stream, result, None)


def tool_response_payload(
    result: ToolResponse | ToolError,
    *,
    request: ToolRequest | None = None,
    feedback_mode: FeedbackMode = "differentiated",
) -> dict[str, Any]:
    """Return the governed JSON payload placed inside ``<tool_response>``.

    ``differentiated`` is the production contract. For Python failures only, the
    ``generic`` and ``missing`` modes intentionally redact details for the #170
    ablation, where the tool and retry budget must remain identical across arms.
    Failures from every other tool always retain the differentiated contract.
    """
    feedback_mode = _validate_feedback_mode(feedback_mode)
    if isinstance(result, ToolResponse):
        feedback_mode = _mode_for_tool(feedback_mode, result.tool_id)
        payload: dict[str, Any] = {
            "ok": result.exit_code == 0,
            "call_id": result.call_id,
            "tool_id": result.tool_id,
            "artifacts": [ref.model_dump(mode="json") for ref in result.artifacts],
        }
        failure = execution_failure(result.exit_code)
        if failure is None or feedback_mode == "differentiated":
            payload.update(
                {
                    "stdout": bound_observation_stream("stdout", result.stdout),
                    "stderr": bound_observation_stream("stderr", result.stderr),
                    "exit_code": result.exit_code,
                }
            )
        if failure is not None:
            stage, code = failure
            if feedback_mode == "differentiated":
                payload.update(
                    {
                        "status": "timeout" if stage == "timeout" else "failed",
                        "error": {
                            "stage": stage,
                            "code": code,
                            "attempted": True,
                        },
                    }
                )
            elif feedback_mode == "generic":
                payload.update(
                    {
                        "status": "failed",
                        "error": {
                            "stage": "execute",
                            "code": "generic_failure",
                            "message": "Python execution failed",
                            "attempted": True,
                        },
                    }
                )
            else:
                # Keep the envelope and identity so the model can continue the
                # episode, while withholding any actionable failure signal.
                payload["attempted"] = True
        return payload
    if not isinstance(result, ToolError):
        raise TypeError("result must be ToolResponse or ToolError")

    call_id = result.call_id or (request.call_id if request is not None else None)
    tool_id = result.tool_id or (request.tool_id if request is not None else None)
    feedback_mode = _mode_for_tool(feedback_mode, tool_id)
    if feedback_mode == "missing":
        return {"ok": False, "call_id": call_id, "tool_id": tool_id}
    if feedback_mode == "generic":
        return {
            "ok": False,
            "call_id": call_id,
            "tool_id": tool_id,
            "status": "failed",
            "error": {
                "stage": "execute",
                "code": "generic_failure",
                "message": "Python execution failed",
                "attempted": False,
            },
        }
    return {
        "ok": False,
        "call_id": call_id,
        "tool_id": tool_id,
        "status": "rejected",
        "error": {
            "stage": result.stage,
            "code": result.code,
            "message": result.message,
            "attempted": False,
        },
    }


def format_tool_response_payload(payload: dict[str, Any]) -> str:
    """Encode one JSON payload with the v2 model–Environment response tags."""
    if not isinstance(payload, dict):
        raise TypeError("tool response payload must be a dict")
    encoded = json.dumps(payload, sort_keys=True)
    # Tool output is untrusted. Escaping tag delimiters prevents stdout/stderr
    # containing ``</tool_response>`` from terminating the observation early;
    # JSON decoders restore the original text.
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"<tool_response>\n{encoded}\n</tool_response>"


def format_tool_response(
    result: ToolResponse | ToolError,
    *,
    request: ToolRequest | None = None,
    feedback_mode: FeedbackMode = "differentiated",
) -> str:
    """Format a canonical response or pre-execution error for the next model turn."""
    return format_tool_response_payload(
        tool_response_payload(result, request=request, feedback_mode=feedback_mode)
    )
