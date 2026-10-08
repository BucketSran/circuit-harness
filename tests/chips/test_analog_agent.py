"""Constructed Pi session and credential-boundary checks for Analog tools."""

import json
import os
import stat
import sys

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from alphaapollo.common.execution.chips.bundle import build_cli
from alphaapollo.common.execution.chips.journal import file_digest


@pytest.fixture
def config(tmp_path):
    session = tmp_path / "public-session"
    session.mkdir()
    (session / "session.json").write_text(
        json.dumps(
            {
                "task_id": "rlc-rf-bandpass-100mhz",
                "max_actions": 24,
                "max_simulations": 4,
            }
        )
    )
    return {
        "transport": "local",
        "python": "/usr/bin/python3",
        "bundle": "/private/chips.pyz",
        "session": str(session),
        "pi_cli": "/private/pi",
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "model": "glm-5.3-flash",
        "thinking": "low",
        "policy_kind": "remote_model",
    }


@pytest.mark.parametrize("model_id", ("glm-5.3", "glm-5.3-flash"))
@pytest.mark.parametrize("runtime", ("direct", "external"))
@pytest.mark.parametrize("task_id", ("rlc-rf-bandpass-100mhz", "rlc-broadband-50-to-200-match"))
def test_pi_receives_public_analog_tools_without_key_in_mcp_child(
    config, tmp_path, monkeypatch, model_id, runtime, task_id
):
    from alphaapollo.reasoning.runtime.external import agents
    from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalRunOutcome
    from alphaapollo.workflows import chips_analog_agent as module

    captured = {}
    from pathlib import Path

    from alphaapollo.common.execution.chips.analog_design_bench import rlc_contract

    config = {**config, "task_id": task_id}
    (Path(config["session"]) / "session.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "task_id": task_id,
                "task_contract_sha256": rlc_contract(task_id).sha256,
                "max_actions": 24,
                "max_simulations": 4,
            }
        )
    )

    class Session:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, task, *, workspace):
            captured["task"] = task
            captured["event_sink"]('{"type":"message_end","secret":"CONSTRUCTED_SECRET"}\n')
            return ExternalRunOutcome(final_text="constructed fixture")

        def close(self):
            pass

    monkeypatch.setattr(agents.pi, "PiSession", Session)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "CONSTRUCTED_SECRET")
    outcome = module.run_pi({**config, "model": model_id, "runtime": runtime}, tmp_path / "attempt")
    assert outcome["final_text"] == "constructed fixture"
    assert captured["tool_mode"] == "no_builtin"
    assert captured["thinking"] == "low"
    model = json.loads((tmp_path / "attempt/pi-home/.pi/agent/models.json").read_text())[
        "providers"
    ]["chips-glm"]["models"][0]
    assert model["compat"]["supportsReasoningEffort"] is True
    assert model["thinkingLevelMap"] == {"low": "low", "high": "high", "xhigh": "max"}
    assert captured["env"]["CHIPS_MODEL_KEY"] == "CONSTRUCTED_SECRET"
    child = captured["mcp_servers"]["analog" if runtime == "direct" else "alphaapollo"]
    assert child["command"] == "/usr/bin/env" and child["args"][0] == "-i"
    expected_module = (
        "alphaapollo.workflows.chips_analog_agent"
        if runtime == "direct"
        else "alphaapollo.reasoning.runtime.external.bridge.mcp_server"
    )
    assert expected_module in child["args"]
    assert "CONSTRUCTED_SECRET" not in json.dumps(child)
    task = captured["task"]
    # Check the delivered/recorded input boundary, not whether a model obeys it.
    role, protocol_and_budget = task.system.split("\n\nAnalog public-session protocol:\n")
    protocol, budget = protocol_and_budget.split("\n\nHarness limits: ")
    assert role.startswith("Experiment rules:\n")
    names = [tool["function"]["name"] for tool in module.tool_schemas()]
    assert all(name not in role for name in names)
    assert all(name in protocol for name in names)
    assert task.prompt == "Complete the public Analog RLC task. Start by listing public files."
    assert "100 MHz" not in task.system and "-3.2" not in task.system
    recorded = json.loads((tmp_path / "attempt/agent-input.json").read_text())
    assert recorded["task_id"] == task.task_id == task_id
    assert recorded["system"] == task.system
    assert recorded["prompt"] == task.prompt
    assert recorded["tools"] == module.tool_schemas()
    assert tuple(names) == task.tools
    assert recorded["experiment_settings"]["runtime"] == runtime
    assert budget.startswith(json.dumps(recorded["experiment_settings"]["budgets"]))
    assert "[REDACTED]" in (tmp_path / "attempt/pi-events.jsonl").read_text()
    assert stat.S_IMODE((tmp_path / "attempt/pi-events.jsonl").stat().st_mode) == 0o600
    for path in (tmp_path / "attempt").rglob("*"):
        if path.is_file():
            assert "CONSTRUCTED_SECRET" not in path.read_text()


def test_analog_rejects_invalid_thinking_level(config):
    from alphaapollo.workflows.chips_analog_agent import validate_pilot

    with pytest.raises(ValueError, match="thinking"):
        validate_pilot({**config, "thinking": "unbounded"})


def test_glm53_flash_rejects_unsupported_thinking_off_before_evidence(config, tmp_path):
    from alphaapollo.workflows.chips_analog_agent import run_pi

    target = tmp_path / "attempt"
    with pytest.raises(ValueError, match="cannot disable thinking"):
        run_pi({**config, "thinking": "off"}, target)
    assert not target.exists()


def test_real_analog_model_requires_an_explicit_thinking_level(config, tmp_path, monkeypatch):
    from alphaapollo.workflows.chips_analog_agent import run_pi

    monkeypatch.setenv("CHIPS_MODEL_KEY", "CONSTRUCTED_SECRET")
    target = tmp_path / "attempt"
    with pytest.raises(ValueError, match="explicit thinking level"):
        run_pi({key: value for key, value in config.items() if key != "thinking"}, target)
    assert not target.exists()


def test_pi_rejects_secret_config_and_bad_budget_before_evidence(config, tmp_path):
    from alphaapollo.workflows.chips_analog_agent import run_pi

    for extra in ({"api_key": "DO_NOT_WRITE"}, {"max_model_calls": 0}):
        target = tmp_path / "attempt"
        with pytest.raises(ValueError):
            run_pi({**config, **extra}, target)
        assert not target.exists()


def test_only_acknowledged_analog_submit_is_frozen(tmp_path):
    from alphaapollo.workflows.chips_analog_agent import submission_confirmed

    action = tmp_path / "tools/one"
    action.mkdir(parents=True)
    (action / "request.json").write_text(
        json.dumps({"id": "one", "tool": "analog_submit", "arguments": {}})
    )
    assert not submission_confirmed(tmp_path)
    (action / "response.json").write_text(
        json.dumps({"ok": True, "result": {"state": "submitted"}})
    )
    assert submission_confirmed(tmp_path)


def test_detach_starts_private_pi_process_without_persisting_key(config, tmp_path, monkeypatch):
    from alphaapollo.workflows.chips_analog_agent import main

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    captured = {}

    def popen(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return type("Process", (), {"pid": 1234})()

    monkeypatch.setenv("CHIPS_MODEL_KEY", "CONSTRUCTED_SECRET")
    monkeypatch.setattr("alphaapollo.workflows.chips_analog_agent.subprocess.Popen", popen)
    evidence = tmp_path / "evidence"
    assert main(["detach", "--config", str(config_path), "--evidence", str(evidence)]) == 0
    assert captured["start_new_session"]
    assert json.loads((evidence / "launcher.json").read_text())["pid"] == 1234
    assert "CONSTRUCTED_SECRET" not in (evidence / "launcher.json").read_text()


def test_pi_reads_private_key_file_without_persisting_it(config, tmp_path, monkeypatch):
    from alphaapollo.workflows.chips_analog_agent import main

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    secrets = tmp_path / "secrets"
    secrets.mkdir(mode=0o700)
    key_file = secrets / "glm.key"
    key_file.write_text("CONSTRUCTED_SECRET\n")
    key_file.chmod(0o600)
    monkeypatch.delenv("CHIPS_MODEL_KEY", raising=False)

    def run_pi(_config, _evidence):
        assert os.environ["CHIPS_MODEL_KEY"] == "CONSTRUCTED_SECRET"
        _evidence.mkdir()
        return {"termination_reason": "final"}

    monkeypatch.setattr("alphaapollo.workflows.chips_analog_agent.run_pi", run_pi)
    evidence = tmp_path / "evidence"
    assert (
        main(
            [
                "pi",
                "--config",
                str(config_path),
                "--evidence",
                str(evidence),
                "--key-file",
                str(key_file),
            ]
        )
        == 1
    )
    assert "CONSTRUCTED_SECRET" not in (evidence / "report.json").read_text()


def test_pi_rejects_key_file_with_unsafe_permissions(config, tmp_path, monkeypatch):
    from alphaapollo.workflows.chips_analog_agent import main

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    secrets = tmp_path / "secrets"
    secrets.mkdir(mode=0o700)
    key_file = secrets / "glm.key"
    key_file.write_text("CONSTRUCTED_SECRET")
    key_file.chmod(0o644)
    monkeypatch.delenv("CHIPS_MODEL_KEY", raising=False)
    evidence = tmp_path / "evidence"
    with pytest.raises(ValueError, match="private key file"):
        main(
            [
                "pi",
                "--config",
                str(config_path),
                "--evidence",
                str(evidence),
                "--key-file",
                str(key_file),
            ]
        )
    assert not evidence.exists()


def test_real_mcp_process_lists_only_public_analog_tools(config, tmp_path):
    session = tmp_path / "session"
    public = session / "public"
    public.mkdir(parents=True)
    (public / "instruction.md").write_text("PUBLIC TASK")
    (session / "private.txt").write_text("HIDDEN_SENTINEL")
    (session / "session.json").write_text(
        json.dumps(
            {
                "max_actions": 4,
                "max_simulations": 1,
                "public_files": {"instruction.md": file_digest(public / "instruction.md")},
            }
        )
    )
    bundle = tmp_path / "chips.pyz"
    build_cli(bundle)
    config = {**config, "python": sys.executable, "bundle": str(bundle), "session": str(session)}
    config_path = tmp_path / "operator.json"
    config_path.write_text(json.dumps(config))

    async def probe():
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "alphaapollo.workflows.chips_analog_agent",
                "serve",
                "--config",
                str(config_path),
                "--evidence",
                str(tmp_path / "tools"),
            ],
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                listed = await client.list_tools()
                assert [tool.name for tool in listed.tools] == [
                    "analog_read",
                    "analog_write",
                    "analog_simulate",
                    "analog_history",
                    "analog_restore",
                    "analog_submit",
                ]
                reply = await client.call_tool("analog_read", {"path": "instruction.md"})
                assert not reply.isError
                assert "PUBLIC TASK" in reply.content[0].text
                assert "HIDDEN_SENTINEL" not in reply.content[0].text

    anyio.run(probe)


def test_agent_refuses_a_different_session_task_before_model_launch(config, tmp_path, monkeypatch):
    from alphaapollo.workflows.chips_analog_agent import run_pi

    monkeypatch.setenv("CHIPS_MODEL_KEY", "CONSTRUCTED_SECRET")
    evidence = tmp_path / "attempt"
    with pytest.raises(ValueError, match="session task_id"):
        run_pi({**config, "task_id": "rlc-broadband-50-to-200-match"}, evidence)
    assert not evidence.exists()


def test_model_budget_stop_collects_without_another_model_request(config, tmp_path, monkeypatch):
    from alphaapollo.workflows import chips_analog_agent as module

    calls = []

    class Transport:
        def cli(self, *args, **kwargs):
            calls.append(args)
            return {
                "state": "collected",
                "agent_submitted": False,
                "collection_source": "episode_end",
                "candidate_sha256": "a" * 64,
            }

    monkeypatch.setattr(module, "transport", lambda *args: Transport())
    receipt = module.finish_episode(
        config,
        tmp_path,
        {
            "termination_reason": "external_error",
            "harness_termination_reason": "model_request_limit",
        },
    )
    assert receipt["state"] == "collected" and receipt["agent_submitted"] is False
    assert calls == [
        ("analog-close", "--session", config["session"], "--reason", "model_request_limit")
    ]
    assert json.loads((tmp_path / "collection.json").read_text()) == receipt
