"""Test-first native Codex VABench slice; CLI and server are local fixtures."""

import json
import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from alphaapollo.workflows import chips_vabench_agent


@pytest.mark.parametrize("login_stream", ("stdout", "stderr"))
def test_local_codex_records_an_unsubmitted_ssh_episode_without_a_glm_key(
    tmp_path, monkeypatch, login_stream
):
    """A Codex turn must retain evidence and never grade an unfrozen candidate."""

    invocation = tmp_path / "codex-invocation.json"
    codex = tmp_path / "fake-codex"
    codex.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "if sys.argv[1:3] == ['login', 'status']:\n"
        f"    print('Logged in using ChatGPT', file=sys.{login_stream})\n"
        "    raise SystemExit(0)\n"
        f"with open({str(invocation)!r}, 'w') as stream:\n"
        "    json.dump({'argv': sys.argv[1:], 'prompt': sys.stdin.read()}, stream)\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': 'fixture-thread'}))\n"
        "print(json.dumps({'type': 'item.completed', 'item': "
        "{'id': 'answer', 'type': 'agent_message', 'text': 'No submission.'}}))\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': "
        "{'input_tokens': 3, 'output_tokens': 2}}))\n",
        encoding="utf-8",
    )
    codex.chmod(0o700)
    config = {
        "transport": "ssh",
        "host": "fixture-server",
        "python": "/usr/bin/python3",
        "bundle": "/private/chips.pyz",
        "session": "/private/vabench-session",
        "task_id": "v4-001",
        "job_root": "/private/jobs",
        "job_id": "codex-fixture-001",
        "archive_root": "/private/archives",
        "policy_kind": "remote_model",
        "auth_kind": "chatgpt",
        "codex_cli": str(codex),
        "model": "gpt-fixture",
        "reasoning_effort": "medium",
        "episode_timeout_s": 30,
    }
    config_path = tmp_path / "operator.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    evidence = tmp_path / "evidence"
    server_calls = []

    class FixtureServer:
        def cli(self, command, *args, **kwargs):
            server_calls.append(command)
            assert command == "vabench-preflight", "final grading requires acknowledged submit"
            return {"state": "ready"}

    monkeypatch.setattr(chips_vabench_agent, "transport", lambda *_: FixtureServer())
    monkeypatch.delenv("CHIPS_MODEL_KEY", raising=False)

    assert (
        chips_vabench_agent.main(
            ["codex", "--config", str(config_path), "--evidence", str(evidence)]
        )
        == 1
    )
    assert server_calls == ["vabench-preflight"]
    run = json.loads(invocation.read_text(encoding="utf-8"))
    assert run["argv"][run["argv"].index("-m") + 1] == "gpt-fixture"
    assert 'model_reasoning_effort="medium"' in run["argv"]
    for tool in ("vabench_read", "vabench_write", "vabench_simulate", "vabench_submit"):
        assert tool in " ".join(run["argv"])
    assert "vabench" in run["prompt"].lower()
    assert "turn.completed" in (evidence / "codex-events.jsonl").read_text(encoding="utf-8")
    assert (
        json.loads((evidence / "codex-outcome.json").read_text(encoding="utf-8"))[
            "termination_reason"
        ]
        == "final"
    )
    assert (
        json.loads((evidence / "report.json").read_text(encoding="utf-8"))["state"] == "unsubmitted"
    )
    budget = json.loads((evidence / "episode-budget.json").read_text(encoding="utf-8"))
    assert budget["episode_timeout_s"] == 30
    assert budget["model_settings"] == {"model": "gpt-fixture", "reasoning_effort": "medium"}
    assert budget["experiment_settings"] == {
        "schema_version": 1,
        "agent": "codex",
        "runtime": "direct",
        "model": "gpt-fixture",
        "reasoning": {"parameter": "reasoning_effort", "value": "medium"},
        "budgets": {
            "max_model_calls": None,
            "max_request_bytes": None,
            "max_output_tokens": None,
            "episode_timeout_s": 30,
        },
    }
    assert budget["model_requests"]["enforced"] is False
    assert budget["model_requests"]["limit"] is None
    manifest = json.loads((evidence / "task_manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"] == "vabench"
    assert manifest["selected"] == [{"task_id": "v4-001", "repeat": 1}]
    resolved = json.loads((evidence / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["operator"]["model"] == "gpt-fixture"
    assert resolved["agent_kind"] == "codex"
    agent_input = json.loads((evidence / "agent-input.json").read_text(encoding="utf-8"))
    assert agent_input["task_id"] == "v4-001"
    assert "vabench_submit" in {tool["function"]["name"] for tool in agent_input["tools"]}
    assert agent_input["native_tools"] == "available"
    rows = [json.loads(line) for line in (evidence / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "unsubmitted"
    assert rows[0]["benchmark_success"] is None
    assert rows[0]["trajectory_refs"]["raw_agent_events"] == "codex-events.jsonl"
    summary = json.loads((evidence / "summary.json").read_text(encoding="utf-8"))
    assert summary["expected_attempts"] == 1
    assert summary["completed_attempts"] == 0
    assert summary["success_rate"] is None


def test_preflight_failure_keeps_one_blocked_result_without_starting_agent(tmp_path, monkeypatch):
    config = {
        "transport": "ssh",
        "host": "fixture-server",
        "python": "/usr/bin/python3",
        "bundle": "/private/chips.pyz",
        "session": "/private/vabench-session",
        "task_id": "v4-001",
        "job_root": "/private/jobs",
        "job_id": "blocked-fixture-001",
        "archive_root": "/private/archives",
        "policy_kind": "remote_model",
        "auth_kind": "chatgpt",
        "codex_cli": "/unused/codex",
        "model": "gpt-fixture",
        "reasoning_effort": "medium",
    }
    config_path = tmp_path / "operator.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    evidence = tmp_path / "evidence"

    class FixtureServer:
        def cli(self, command, *args, **kwargs):
            assert command == "vabench-preflight"
            return {"state": "unavailable"}

    monkeypatch.setattr(chips_vabench_agent, "transport", lambda *_: FixtureServer())
    monkeypatch.setattr(
        chips_vabench_agent, "run_codex", lambda *_: pytest.fail("agent must not start")
    )
    with pytest.raises(ValueError, match="preflight did not pass"):
        chips_vabench_agent.main(
            ["codex", "--config", str(config_path), "--evidence", str(evidence)]
        )
    row = json.loads((evidence / "results.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "blocked"
    assert row["error_type"] == "preflight_not_ready"
    assert row["benchmark_success"] is None
    assert json.loads((evidence / "summary.json").read_text())["completed_attempts"] == 0


def test_preflight_rejects_a_different_server_task_before_model(tmp_path, monkeypatch):
    config = {
        "transport": "ssh",
        "host": "fixture-server",
        "python": "/usr/bin/python3",
        "bundle": "/private/chips.pyz",
        "session": "/private/vabench-session",
        "task_id": "v4-001",
        "job_root": "/private/jobs",
        "job_id": "mismatch-fixture-001",
        "archive_root": "/private/archives",
        "policy_kind": "remote_model",
        "auth_kind": "chatgpt",
        "codex_cli": "/unused/codex",
        "model": "gpt-fixture",
        "reasoning_effort": "medium",
    }
    config_path = tmp_path / "operator.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    evidence = tmp_path / "evidence"

    class FixtureServer:
        def cli(self, command, *args, **kwargs):
            assert command == "vabench-preflight"
            return {"state": "ready", "task_id": "v4-002", "session_sha256": "a" * 64}

    monkeypatch.setattr(chips_vabench_agent, "transport", lambda *_: FixtureServer())
    monkeypatch.setattr(
        chips_vabench_agent, "run_codex", lambda *_: pytest.fail("agent must not start")
    )
    with pytest.raises(ValueError, match="task ID does not match"):
        chips_vabench_agent.main(
            ["codex", "--config", str(config_path), "--evidence", str(evidence)]
        )
    row = json.loads((evidence / "results.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "blocked"
    assert row["error_type"] == "task_identity_mismatch"
    assert row["task_identity_status"] == "mismatch"


@pytest.mark.parametrize("auth_kind, uses_openai_key", [("chatgpt", False), ("api_key", True)])
def test_codex_auth_mode_passes_only_selected_provider_key(
    tmp_path, monkeypatch, auth_kind, uses_openai_key
):
    codex = tmp_path / "fake-codex"
    observed = tmp_path / "environment.json"
    codex.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:3] == ['login', 'status']:\n"
        "    print('Logged in using ChatGPT')\n"
        "    raise SystemExit(0)\n"
        f"open({str(observed)!r}, 'w').write(json.dumps(dict(os.environ)))\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {}}))\n",
        encoding="utf-8",
    )
    codex.chmod(0o700)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "GLM_SECRET_SENTINEL")
    monkeypatch.setenv("OPENAI_API_KEY", "OPENAI_SECRET_SENTINEL")
    config = {
        "transport": "ssh",
        "policy_kind": "remote_model",
        "host": "fixture-server",
        "python": "/usr/bin/python3",
        "bundle": "/private/chips.pyz",
        "session": "/private/vabench-session",
        "codex_cli": str(codex),
        "model": "gpt-fixture",
        "reasoning_effort": "medium",
        "auth_kind": auth_kind,
    }
    chips_vabench_agent.run_codex(config, tmp_path / "evidence")
    environment = json.loads(observed.read_text(encoding="utf-8"))
    assert "CHIPS_MODEL_KEY" not in environment
    assert ("OPENAI_API_KEY" in environment) is uses_openai_key
    assert (
        json.loads((tmp_path / "evidence" / "auth-preflight.json").read_text())["kind"] == auth_kind
    )
    assert os.stat(tmp_path / "evidence" / "codex-events.jsonl").st_mode & 0o077 == 0


def test_real_vabench_codex_requires_an_explicit_reasoning_effort(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="explicit reasoning_effort"):
        chips_vabench_agent.validate_codex(
            {
                "transport": "ssh",
                "policy_kind": "remote_model",
                "auth_kind": "chatgpt",
                "codex_cli": "/private/codex",
                "model": "gpt-fixture",
            }
        )


def test_real_stdio_mcp_records_a_constructed_remote_action(tmp_path):
    """Exercise the real MCP child and journal; SSH replies are constructed."""

    config = {
        "host": "fixture-server",
        "transport": "ssh",
        "python": "/usr/bin/python3",
        "bundle": "/private/chips.pyz",
        "session": "/private/vabench-session",
    }
    config_path = tmp_path / "operator.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    evidence = tmp_path / "tools"
    repo = str(Path(chips_vabench_agent.__file__).resolve().parents[2])
    script = """
import json, sys
from alphaapollo.common.execution.chips.vabench_remote import RemoteVabench
from alphaapollo.workflows import chips_vabench_agent as agent
class FixtureRemote(RemoteVabench):
    def cli(self, command, *args, **kwargs):
        if command == 'vabench-request':
            return {'state': 'accepted'}
        return {'ok': True, 'result': {'files': []}}
agent.transport = lambda config, evidence: FixtureRemote(config, evidence)
agent.serve(json.load(open(sys.argv[1])), sys.argv[2])
"""

    async def probe():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-c", script, str(config_path), str(evidence)],
            env={"PYTHONPATH": repo},
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                listed = await client.list_tools()
                assert {tool.name for tool in listed.tools} == {
                    "vabench_read",
                    "vabench_write",
                    "vabench_simulate",
                    "vabench_submit",
                }
                reply = await client.call_tool("vabench_read", {"path": ""})
                assert not reply.isError
                assert json.loads(reply.content[0].text)["result"] == {"files": []}

    anyio.run(probe)
    actions = list(evidence.glob("*/request.json"))
    assert len(actions) == 1
    assert json.loads(actions[0].read_text(encoding="utf-8"))["tool"] == "vabench_read"
    assert actions[0].with_name("response.json").exists()


def test_chatgpt_mode_requires_local_codex_login_before_agent_runs(tmp_path):
    marker = tmp_path / "agent-started"
    codex = tmp_path / "fake-codex"
    codex.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1:3] == ['login', 'status']:\n"
        "    print('Not logged in')\n"
        "    raise SystemExit(1)\n"
        f"open({str(marker)!r}, 'w').write('started')\n",
        encoding="utf-8",
    )
    codex.chmod(0o700)
    config = {
        "transport": "ssh",
        "policy_kind": "remote_model",
        "auth_kind": "chatgpt",
        "host": "fixture-server",
        "python": "/usr/bin/python3",
        "bundle": "/private/chips.pyz",
        "session": "/private/vabench-session",
        "codex_cli": str(codex),
        "model": "gpt-fixture",
        "reasoning_effort": "medium",
    }
    with pytest.raises(ValueError, match="ChatGPT login"):
        chips_vabench_agent.run_codex(config, tmp_path / "evidence")
    assert not marker.exists()
