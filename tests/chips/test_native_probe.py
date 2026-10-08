"""Native validation probe boundaries; constructed HTTP responses, no live model."""

import json
import runpy
from pathlib import Path

import httpx
import pytest


def probe():
    return runpy.run_path(str(Path(__file__).parent / "probes/native_vabench.py"))["run_native"]


@pytest.mark.parametrize("mode", ["submit", "oversized", "http_error", "turn_limit"])
def test_native_probe_records_requests_and_enforces_wire_budgets(tmp_path, monkeypatch, mode):
    observed = []
    tool_calls = []

    def handle(request):
        body = json.loads(request.content)
        observed.append(body)
        if mode == "http_error":
            return httpx.Response(500, json={"error": {"message": "fixture error"}})
        name = "vabench_submit" if len(observed) == 2 else "vabench_read"
        args = "{}" if name.endswith("submit") else '{"path":""}'
        return httpx.Response(
            200,
            json={
                "id": f"fixture-{len(observed)}",
                "model": "fixture",
                "object": "chat.completion",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "visible reasoning",
                            "tool_calls": [
                                {
                                    "id": f"call-{len(observed)}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": args},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        )

    class Transport:
        def call(self, name, args, **kwargs):
            tool_calls.append((name, args))
            return {
                "ok": True,
                "result": {"status": "submitted"} if name.endswith("submit") else {"files": []},
            }

    config = {
        "transport": "local",
        "python": "/usr/bin/python3",
        "bundle": "/unused/chips.pyz",
        "session": "/unused/session",
        "pi_cli": "/unused/pi",
        "model": "fixture",
        "base_url": "http://127.0.0.1:1/v1",
        "policy_kind": "scripted_http_fixture",
        "thinking": "low",
        "max_model_calls": 1 if mode == "turn_limit" else 2,
        "max_request_bytes": 1 if mode == "oversized" else 64000,
        "max_output_tokens": 8192,
        "episode_timeout_s": 60,
    }
    monkeypatch.setenv("CHIPS_MODEL_KEY", "CONSTRUCTED_SECRET")
    evidence = tmp_path / "evidence"
    result = probe()(
        config,
        evidence,
        http_transport=httpx.MockTransport(handle),
        transport_factory=lambda *_: Transport(),
    )
    if mode == "submit":
        assert result["termination_reason"] == "submitted"
        assert len(tool_calls) == 2
        assert observed[1]["messages"][-1]["tool_call_id"] == "call-1"
        assert observed[1]["messages"][-2]["reasoning_content"] == "visible reasoning"
        assert observed[0]["max_tokens"] == 8192
        assert observed[0]["reasoning_effort"] == "low"
        assert observed[0]["thinking"] == {"type": "enabled", "clear_thinking": False}
    elif mode == "oversized":
        assert observed == [] and tool_calls == []
        assert result["state"] == "failed"
    elif mode == "http_error":
        assert len(observed) == 1 and tool_calls == []
        assert result["state"] == "failed"
    else:
        assert len(observed) == 1 and len(tool_calls) == 1
        assert result["termination_reason"] == "max_turns"
    assert "CONSTRUCTED_SECRET" not in "".join(
        p.read_text() for p in evidence.rglob("*") if p.is_file()
    )
    assert (evidence / "native-outcome.json").is_file()
