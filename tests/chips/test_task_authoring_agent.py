"""Phase boundary tests; no provider or lab connection is made here."""

import json

import pytest

from alphaapollo.workflows.chips_task_authoring_agent import RemoteGain, run_phase


def test_gain_transport_uses_its_own_commands_after_shared_transport_split(tmp_path, monkeypatch):
    remote = RemoteGain(
        {
            "host": "lab-host",
            "python": "/usr/bin/python3",
            "bundle": "/private/chips.pyz",
            "session": "/private/gain",
        },
        tmp_path / "tools",
    )
    commands = []

    def cli(command, *args, payload=None, **kwargs):
        commands.append(command)
        return {"state": "accepted"} if command == "gain-request" else {"ok": True}

    monkeypatch.setattr(remote, "cli", cli)
    assert remote.call("gain_simulate", {}, action_id="one")["ok"]
    assert commands == ["gain-request", "gain-response"]


def test_build_rejects_missing_confirmation_before_model_start(tmp_path):
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps({"schema_version": 1}))
    config = {"draft": str(draft), "materials": str(tmp_path), "confirmation": None}
    with pytest.raises(ValueError, match="confirmation"):
        run_phase("build", config, tmp_path / "run")
    assert not (tmp_path / "run/codex-events.jsonl").exists()


def test_extract_records_actual_image_and_preserves_unconfirmed_model_draft(tmp_path, monkeypatch):
    from alphaapollo.common.execution.chips.journal import file_digest
    from alphaapollo.reasoning.runtime.external.agents.codex import CodexSession
    from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalRunOutcome

    (tmp_path / "notes.md").write_text("Supply is unknown.")
    image = tmp_path / "image.png"
    image.write_bytes(b"constructed image fixture; no model invoked")
    draft = {
        "schema_version": 1,
        "task_id": "d0",
        "sources": [
            {"path": p, "sha256": file_digest(tmp_path / p)} for p in ("notes.md", "image.png")
        ],
        "fields": {"supply_v": {"value": None, "unit": "V", "status": "missing", "evidence": []}},
    }

    def fake_run(self, task, *, workspace):
        assert str(image) in self.argv()
        assert "--image" in self.argv()
        assert "Supply is unknown" in task.prompt
        assert 'model_reasoning_effort="medium"' in self.argv()
        return ExternalRunOutcome(final_text=json.dumps(draft))

    monkeypatch.setattr(CodexSession, "run", fake_run)
    config = {
        "materials": str(tmp_path),
        "notes": "notes.md",
        "image": "image.png",
        "task_id": "d0",
        "model": "constructed-model",
        "codex_cli": "/constructed",
        "reasoning_effort": "medium",
    }
    result = run_phase("extract", config, tmp_path / "run")
    assert result["status"] == "needs_clarification"
    assert json.loads((tmp_path / "run/draft.json").read_text()) == draft
    assert not (tmp_path / "run/confirmation.json").exists()


def test_real_mcp_subprocess_exposes_only_two_gain_tools(tmp_path):
    import sys
    from pathlib import Path

    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "remote": {
                    "host": "constructed-no-network",
                    "python": "/usr/bin/python3",
                    "bundle": "/not-used.pyz",
                    "session": "/not-used",
                }
            }
        )
    )

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "alphaapollo.workflows.chips_task_authoring_agent",
                "serve",
                "--config",
                str(config),
                "--evidence",
                str(tmp_path / "tools"),
            ],
            env={"PYTHONPATH": str(Path(__file__).resolve().parents[2])},
            cwd=str(tmp_path),
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                tools = (await client.list_tools()).tools
                assert [tool.name for tool in tools] == ["gain_simulate", "gain_submit"]
                assert all(set(tool.inputSchema["properties"]) == {"candidate"} for tool in tools)

    anyio.run(scenario)
