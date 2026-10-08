"""Chips grants use the existing external runtime and MCP transport."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from alphaapollo.common.execution.tools.chips import EMX_TOOL_SPEC
from alphaapollo.workflows._resources.runtime import _external_session_options
from alphaapollo.workflows.chips import main
from alphaapollo.workflows.config import ConfigError, ResourceConfig, load_run_config
from alphaapollo.workflows.resources import validate_composition_config

ROOT = Path(__file__).resolve().parents[2]


def test_workflow_and_mcp_settings(monkeypatch, tmp_path):
    config = load_run_config(ROOT / "examples/chips/emx/config.yaml")
    validate_composition_config(config)
    path = str(tmp_path / "operator.json")
    monkeypatch.setenv("ALPHAAPOLLO_CHIPS_CONFIG", path)
    options = _external_session_options(
        config.runtimes["solver"], ["emx_simulate"], where="solver", environment_backed=False
    )
    bridge = options["mcp_servers"]["alphaapollo"]
    assert bridge["env"]["ALPHAAPOLLO_CHIPS_CONFIG"] == path
    assert options["mcp_auto_approve_tools"]["alphaapollo"] == ["emx_simulate"]
    with pytest.raises(ConfigError, match="emx_simulate requires"):
        validate_composition_config(replace(config, environment=ResourceConfig(type="default")))


def test_real_stdio_tool_discovery_and_missing_config(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPHAAPOLLO_CHIPS_CONFIG", raising=False)

    async def scenario():
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "alphaapollo.reasoning.runtime.external.bridge.mcp_server",
                "--tools",
                "emx_simulate",
            ],
            env={"PYTHONPATH": str(ROOT)},
            cwd=str(tmp_path),
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                listed = await client.list_tools()
                assert [tool.name for tool in listed.tools] == [EMX_TOOL_SPEC.tool_id]
                reply = await client.call_tool("emx_simulate", {"layout_json": "{}"})
                encoded = json.dumps(reply.model_dump())
                assert "ALPHAAPOLLO_CHIPS_CONFIG" in encoded
                assert '"exit_code": 2' in encoded or '\\"exit_code\\": 2' in encoded

    anyio.run(scenario)


def test_cli_report_escapes_logs_and_status(tmp_path):
    from alphaapollo.common.execution.chips.journal import Journal

    Journal(tmp_path, call_id="test").emit("start", text="<script>alert(1)</script>")
    assert main(["report", str(tmp_path)]) == 0
    report = (tmp_path / "report.html").read_text()
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert main(["status", str(tmp_path)]) == 0
