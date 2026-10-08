"""Shared Pi launch fixtures; no provider or simulator requests."""

import json
import os

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from alphaapollo.common.execution.chips.journal import atomic_json
from alphaapollo.common.generation.base import ToolCall
from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalEvent, ExternalRunOutcome
from alphaapollo.workflows import chips_analog_agent, chips_vabench_agent
from alphaapollo.workflows.chips_experiment_settings import experiment_settings_snapshot


@pytest.mark.parametrize("module", [chips_analog_agent, chips_vabench_agent])
def test_tasks_share_explicit_larger_per_request_output_budget(module):
    from alphaapollo.workflows.chips_experiment_settings import pi_zai_model

    config = {
        "transport": "local",
        "python": "/private/python",
        "bundle": "/private/chips.pyz",
        "session": "/private/session",
        "pi_cli": "/private/pi",
        "model": "glm-5.3-flash",
        "base_url": "https://example.invalid/v1",
        "policy_kind": "remote_model",
        "thinking": "low",
        "max_output_tokens": 16384,
    }
    if module is chips_vabench_agent:
        config.update(
            task_id="v4-001",
            job_root="/private/jobs",
            job_id="budget-001",
            archive_root="/private/archives",
        )
    module.validate_pilot(config)
    assert pi_zai_model(config)["maxTokens"] == 16384
    assert experiment_settings_snapshot(config, "pi")["budgets"]["max_output_tokens"] == 16384
    assert experiment_settings_snapshot(config, "pi")["budgets"]["max_model_calls"] == 12
    for value in (True, 0, 32769):
        with pytest.raises(ValueError, match="max_output_tokens"):
            module.validate_pilot({**config, "max_output_tokens": value})


@pytest.mark.parametrize(
    "kind,module", [("vabench", chips_vabench_agent), ("analog", chips_analog_agent)]
)
def test_pi_shared_runtime_uses_environment_and_retains_raw_and_normalized_evidence(
    tmp_path, monkeypatch, kind, module
):
    config = {
        "transport": "local",
        "python": "/usr/bin/python3",
        "bundle": "/private/chips.pyz",
        "session": "/private/session",
        "pi_cli": "/private/pi",
        "model": "fixture",
        "base_url": "http://127.0.0.1:1/v1",
        "policy_kind": "scripted_http_fixture",
        "runtime": "external",
        "http_retry_limit": 0,
    }
    if kind == "analog":
        session = tmp_path / "session"
        session.mkdir()
        atomic_json(
            session / "session.json",
            {
                "task_id": "rlc-rf-bandpass-100mhz",
                "max_actions": 24,
                "max_simulations": 4,
            },
        )
        config["session"] = str(session)
    captured = {}
    seen = []

    class Transport:
        def call(self, tool, arguments, **kwargs):
            seen.append((tool, arguments))
            field = "status" if kind == "vabench" else "state"
            return {"ok": True, "result": {field: "submitted"}}

    class Session:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, task, *, workspace):
            declaration = captured["mcp_servers"]["alphaapollo"]
            assert "CHIPS_MODEL_KEY" not in json.dumps(declaration)
            assert task.task_payload == {}
            events = []

            async def run():
                params = StdioServerParameters(**declaration, cwd=str(workspace))
                async with stdio_client(params) as (reader, writer):
                    async with ClientSession(reader, writer) as client:
                        await client.initialize()
                        listed = await client.list_tools()
                        assert [t.name for t in listed.tools] == [
                            t["function"]["name"] for t in module.tool_schemas()
                        ]
                        name = kind + "_submit"
                        response = await client.call_tool(name, {})
                        assert not response.isError
                        events.extend(
                            (
                                ExternalEvent(
                                    kind="tool_call",
                                    tool_call=ToolCall(id="one", name=name, arguments="{}"),
                                ),
                                ExternalEvent(
                                    kind="tool_result",
                                    call_id="one",
                                    tool_id=name,
                                    content=response.content[0].text,
                                ),
                            )
                        )

            anyio.run(run)
            captured["event_sink"]("RAW_FIXTURE_MODEL_EVENT CONSTRUCTED_SECRET\n")
            return ExternalRunOutcome(final_text="Done", events=tuple(events))

        def close(self):
            captured["closed"] = True

    monkeypatch.setenv("CHIPS_MODEL_KEY", "CONSTRUCTED_SECRET")
    monkeypatch.setattr(module, "transport", lambda *args: Transport())
    monkeypatch.setattr("alphaapollo.reasoning.runtime.external.agents.pi.PiSession", Session)
    evidence = tmp_path / "evidence"
    result = module.run_pi(config, evidence)
    assert result["final_text"] == "Done"
    assert seen == [(kind + "_submit", {})]
    assert captured["closed"] is True
    assert (evidence / "pi-outcome.json").is_file()
    normalized = json.loads((evidence / "runtime-result.json").read_text())
    assert normalized["termination_reason"] == "submitted"
    assert len(normalized["turns"]) == 1
    assert normalized["turns"][0]["environment_transition"]["success"] is None
    assert "RAW_FIXTURE_MODEL_EVENT [REDACTED]" in (evidence / "pi-events.jsonl").read_text()
    assert not list(evidence.rglob(".alphaapollo-environment.json"))
    assert "CONSTRUCTED_SECRET" not in "".join(
        p.read_text() for p in evidence.rglob("*") if p.is_file()
    )
    assert os.stat(evidence / "runtime-result.json").st_mode & 0o077 == 0
    assert experiment_settings_snapshot(config, "pi")["runtime"] == "external"
    settings = json.loads((evidence / "pi-home/.pi/agent/settings.json").read_text())
    assert settings["retry"] == {"enabled": False, "maxRetries": 0, "provider": {"maxRetries": 0}}


@pytest.mark.parametrize("module", [chips_analog_agent, chips_vabench_agent])
def test_invalid_runtime_fails_before_evidence(module, tmp_path):
    config = {
        "transport": "local",
        "python": "/usr/bin/python3",
        "bundle": "/private/chips.pyz",
        "session": "/private/session",
        "pi_cli": "/private/pi",
        "model": "fixture",
        "base_url": "http://127.0.0.1:1/v1",
        "policy_kind": "scripted_http_fixture",
        "runtime": "typo",
    }
    with pytest.raises(ValueError, match="runtime"):
        module.run_pi(config, tmp_path / "evidence")
    assert not (tmp_path / "evidence").exists()


@pytest.mark.skipif(not os.environ.get("CHIPS_TEST_PI_CLI"), reason="explicit real Pi CLI required")
@pytest.mark.parametrize("runtime", ["direct", "external"])
def test_real_pi_zero_retries_sends_only_one_http_attempt(tmp_path, monkeypatch, runtime):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    attempts = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            attempts.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = b'{"error":{"message":"constructed temporary failure","type":"server_error"}}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = {
        "transport": "local",
        "python": "/usr/bin/python3",
        "bundle": "/unused/chips.pyz",
        "session": "/unused/session",
        "pi_cli": os.environ["CHIPS_TEST_PI_CLI"],
        "model": "fixture",
        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "policy_kind": "scripted_http_fixture",
        "runtime": runtime,
        "http_retry_limit": 0,
        "episode_timeout_s": 30,
    }
    monkeypatch.setenv("CHIPS_MODEL_KEY", "CONSTRUCTED_SECRET")
    evidence = tmp_path / "evidence"
    try:
        result = chips_vabench_agent.run_pi(config, evidence)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result["termination_reason"] != "final"
    assert len(attempts) == 1
    events = [
        json.loads(line) for line in (evidence / "model-budget.jsonl").read_text().splitlines()
    ]
    assert sum(event["event"] == "request" for event in events) == 1
    assert not list((evidence / "tools").glob("*/request.json"))


@pytest.mark.parametrize(
    "raw,metadata,expected",
    [
        ("timeout", {}, "deadline"),
        ("external_error", {"returncode": 73}, "runtime_error"),
        ("external_error", {"returncode": 1}, "runtime_error"),
    ],
)
def test_recorded_stop_does_not_infer_budget_exhaustion_from_exit_code(
    tmp_path, raw, metadata, expected
):
    from alphaapollo.workflows.chips_pi_runtime import _RecordedSession

    class Session:
        def run(self, task, *, workspace):
            return ExternalRunOutcome(
                final_text="", termination_reason=raw, provider_metadata=metadata
            )

    recorded = _RecordedSession(Session(), tmp_path, None)
    recorded.run(None, workspace=tmp_path)
    saved = json.loads((tmp_path / "pi-outcome.json").read_text())
    assert saved["termination_reason"] == raw
    assert saved["harness_termination_reason"] == expected
