from __future__ import annotations

import json

import pytest

from alphaapollo.common.environment.default.projection import (
    MAX_OBSERVATION_STREAM_BYTES,
    format_tool_response,
    format_tool_response_payload,
    tool_response_payload,
)
from alphaapollo.common.execution import ToolError, ToolRequest, ToolResponse


def _decode_observation(observation: str) -> dict:
    assert observation.startswith("<tool_response>\n")
    assert observation.endswith("\n</tool_response>")
    body = observation.removeprefix("<tool_response>\n").removesuffix("\n</tool_response>")
    return json.loads(body)


def test_success_response_is_a_v2_tagged_deterministic_json_observation() -> None:
    response = ToolResponse(
        call_id="python-call",
        tool_id="python",
        stdout="42\n",
        stderr="",
        exit_code=0,
    )

    first = format_tool_response(response)
    second = format_tool_response(response)

    assert first == second
    assert _decode_observation(first) == {
        "ok": True,
        "call_id": "python-call",
        "tool_id": "python",
        "stdout": "42\n",
        "stderr": "",
        "exit_code": 0,
        "artifacts": [],
    }


@pytest.mark.parametrize(
    ("exit_code", "stderr"),
    [
        (1, "RuntimeError: fixture failure"),
        (124, "command timed out after 0.1s"),
        (130, "command cancelled"),
    ],
)
def test_attempted_failures_remain_responses_with_exit_status(
    exit_code: int,
    stderr: str,
) -> None:
    payload = tool_response_payload(
        ToolResponse(
            call_id="python-call",
            tool_id="python",
            stderr=stderr,
            exit_code=exit_code,
        )
    )

    assert payload["ok"] is False
    assert payload["exit_code"] == exit_code
    assert payload["stderr"] == stderr
    assert payload["status"] == ("timeout" if exit_code == 124 else "failed")
    assert payload["error"]["attempted"] is True
    assert payload["error"]["code"] in {
        "nonzero_exit",
        "execution_timeout",
        "execution_cancelled",
    }


def test_pre_execution_error_uses_request_identity_without_claiming_an_attempt() -> None:
    request = ToolRequest(
        call_id="python-call",
        tool_id="python",
        arguments={"code": "print(1)"},
        source="python_code",
    )
    error = ToolError(
        stage="acquire",
        code="sandbox_unavailable",
        message="unable to acquire isolated sandbox",
    )

    payload = tool_response_payload(error, request=request)

    assert payload == {
        "ok": False,
        "call_id": "python-call",
        "tool_id": "python",
        "status": "rejected",
        "error": {
            "stage": "acquire",
            "code": "sandbox_unavailable",
            "message": "unable to acquire isolated sandbox",
            "attempted": False,
        },
    }


def test_formatter_escapes_model_visible_stdout_and_stderr_as_json() -> None:
    observation = format_tool_response(
        ToolResponse(
            call_id="python-call",
            tool_id="python",
            stdout='line 1\n"</tool_response>"\n',
            stderr="错误\n",
            exit_code=7,
        )
    )

    payload = _decode_observation(observation)
    assert payload["stdout"] == 'line 1\n"</tool_response>"\n'
    assert payload["stderr"] == "错误\n"
    assert observation.count("</tool_response>") == 1
    assert "\\u003c/tool_response\\u003e" in observation


def test_feedback_ablation_redacts_attempted_failure_details() -> None:
    response = ToolResponse(
        call_id="python-call",
        tool_id="python",
        stdout="partial output",
        stderr="Traceback: secret host path",
        exit_code=1,
    )

    generic = tool_response_payload(response, feedback_mode="generic")
    assert generic == {
        "ok": False,
        "call_id": "python-call",
        "tool_id": "python",
        "artifacts": [],
        "status": "failed",
        "error": {
            "stage": "execute",
            "code": "generic_failure",
            "message": "Python execution failed",
            "attempted": True,
        },
    }

    missing = tool_response_payload(response, feedback_mode="missing")
    assert missing == {
        "ok": False,
        "call_id": "python-call",
        "tool_id": "python",
        "artifacts": [],
        "attempted": True,
    }
    assert "secret host path" not in json.dumps(generic)
    assert "secret host path" not in json.dumps(missing)


@pytest.mark.parametrize("feedback_mode", ["generic", "missing"])
def test_python_feedback_ablation_does_not_redact_other_tool_failures(
    feedback_mode: str,
) -> None:
    response = ToolResponse(
        call_id="bash-call",
        tool_id="bash",
        stdout="partial output",
        stderr="bash: command not found",
        exit_code=127,
    )

    payload = tool_response_payload(response, feedback_mode=feedback_mode)  # type: ignore[arg-type]

    assert payload == {
        "ok": False,
        "call_id": "bash-call",
        "tool_id": "bash",
        "stdout": "partial output",
        "stderr": "bash: command not found",
        "exit_code": 127,
        "artifacts": [],
        "status": "failed",
        "error": {
            "stage": "execute",
            "code": "nonzero_exit",
            "attempted": True,
        },
    }


def test_payload_formatter_rejects_non_mapping_input() -> None:
    with pytest.raises(TypeError, match="must be a dict"):
        format_tool_response_payload([])  # type: ignore[arg-type]


def test_small_streams_reach_the_model_byte_for_byte() -> None:
    """The cap must not touch the results this workload actually produces.

    Real `<tool_response>` payloads here run 174-538 bytes; the cap sits at
    1,024 per stream precisely so ordinary output is never reshaped.
    """
    response = ToolResponse(
        call_id="bash-call",
        tool_id="bash",
        stdout="Ratio: 0.58\nFraction: 29/50\n",
        stderr="",
        exit_code=0,
    )

    payload = tool_response_payload(response)

    assert payload["stdout"] == "Ratio: 0.58\nFraction: 29/50\n"
    assert payload["stderr"] == ""


def test_oversized_streams_are_capped_per_stream_with_a_visible_marker() -> None:
    """An unbounded result would overrun the context; a silent one would mislead.

    The execution layer bounds a stream at 50KB, which is sized for an artifact
    on disk, not for text retained in every later prompt. The Environment caps
    what it retains, keeps the *tail* -- where a computation prints its answer --
    and says so, because a model that cannot tell it got a partial result will
    trust a partial answer.
    """
    stdout = "".join(f"row {index}\n" for index in range(20_000))
    stderr = "E" * 40_000
    response = ToolResponse(
        call_id="bash-call",
        tool_id="bash",
        stdout=stdout,
        stderr=stderr,
        exit_code=0,
    )

    payload = tool_response_payload(response)

    for stream in ("stdout", "stderr"):
        rendered = payload[stream]
        assert f"[{stream} truncated:" in rendered
        assert len(rendered.encode("utf-8")) < MAX_OBSERVATION_STREAM_BYTES + 200

    # The tail survives: the last line printed is the one the model needs.
    assert "row 19999" in payload["stdout"]
    assert "row 0\n" not in payload["stdout"]


def test_the_cap_bounds_the_whole_observation_not_just_one_stream() -> None:
    """Both streams are capped, so the retained observation is bounded either way.

    A command that writes 50KB to each stream used to contribute both in full.
    """
    response = ToolResponse(
        call_id="bash-call",
        tool_id="bash",
        stdout="O" * (50 * 1024),
        stderr="E" * (50 * 1024),
        exit_code=1,
    )

    observation = format_tool_response(response)

    assert len(observation.encode("utf-8")) < 4 * MAX_OBSERVATION_STREAM_BYTES
    decoded = _decode_observation(observation)
    assert "[stdout truncated:" in decoded["stdout"]
    assert "[stderr truncated:" in decoded["stderr"]


def test_a_refused_or_redacted_result_carries_no_stream_to_cap() -> None:
    """The cap applies where streams are surfaced and nowhere else."""
    error = ToolError(
        stage="policy",
        code="tool_not_granted",
        message="tool 'bash' is not granted to this role",
        call_id="bash-call",
        tool_id="bash",
    )

    payload = tool_response_payload(error)

    assert "stdout" not in payload
    assert "stderr" not in payload
