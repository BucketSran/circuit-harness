"""Real stdio MCP over the public Unix socket, with no model or simulator."""

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from test_current_evas_session import make_session

from circuit_harness.execution.native_sandbox import NativeSandbox
from circuit_harness.public_mcp import PublicMCPBroker


def test_broker_start_failure_removes_its_socket_directory(tmp_path, monkeypatch):
    directory = make_session(tmp_path)

    async def unavailable(*args, **kwargs):
        raise OSError("synthetic socket failure")

    monkeypatch.setattr(asyncio, "start_unix_server", unavailable)

    async def run():
        broker = PublicMCPBroker(directory, tmp_path / "broker", Path(sys.executable))
        with pytest.raises(OSError, match="synthetic socket failure"):
            await broker.start()
        assert broker.socket_path is not None
        assert not broker.socket_path.parent.exists()
        await broker.close()

    asyncio.run(run())


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("experiments", [False, True])
def test_mcp_reads_edits_freezes_and_replays_an_action(tmp_path, isolated, experiments):
    if isolated and shutil.which("codex") is None:
        pytest.skip("Codex CLI not installed")
    declaration = (
        {"version": "v1", "files": ["probe.py"], "analyses": ["python_measurement"]}
        if experiments
        else None
    )
    directory = make_session(tmp_path, experiments=declaration)

    async def run():
        broker = await PublicMCPBroker(
            directory,
            tmp_path / "broker",
            Path(sys.executable),
        ).start()
        try:
            command = broker.client_command()
            env = None
            if isolated:
                public = tmp_path / "agent"
                public.mkdir()
                sandbox = NativeSandbox(
                    codex=Path(shutil.which("codex")),
                    directory=tmp_path / "policy",
                    workspace=public,
                    protected_paths=(directory,),
                    readonly_paths=(broker.client_path, Path(sys.base_prefix)),
                    socket_paths=(broker.socket_path,),
                )
                command[0] = str(Path(sys.executable).resolve())
                command, env = sandbox.command(command)
            params = StdioServerParameters(command=command[0], args=command[1:], env=env)
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as client:
                    await client.initialize()
                    tools = await client.list_tools()
                    expected = {
                        "evas_read",
                        "evas_write",
                        "evas_simulate",
                        "evas_submit",
                        "evas_observe",
                        "evas_read_artifact",
                    }
                    if experiments:
                        expected.add("evas_experiment")
                    assert {tool.name for tool in tools.tools} == expected
                    artifact = await client.call_tool(
                        "evas_read_artifact", {"artifact_id": "manifest", "offset": 0, "limit": 128}
                    )
                    assert not artifact.isError, artifact.content
                    assert json.loads(artifact.content[0].text)["result"]["next_offset"] == 128
                    public = await client.call_tool("evas_read", {"path": "instruction.md"})
                    assert "synthetic protocol fixture" in public.content[0].text
                    request = {"path": "dut.va", "content": "bad syntax", "action_id": "edit-one"}
                    first = await client.call_tool("evas_write", request)
                    assert not first.isError, first.content
                    repeated = await client.call_tool("evas_write", request)
                    assert first.content == repeated.content
                    if experiments:
                        script = await client.call_tool(
                            "evas_write", {"path": "probe.py", "content": "print('{}')"}
                        )
                        assert not script.isError
                    frozen = await client.call_tool("evas_submit", {})
                    assert not frozen.isError
                    changed = await client.call_tool(
                        "evas_write", {"path": "dut.va", "content": "late"}
                    )
                    assert json.loads(changed.content[0].text)["error"] == "submission_frozen"
            assert (directory / "candidate/files/dut.va").read_text() == "bad syntax"
            assert not (directory / "candidate/files/probe.py").exists()
        finally:
            await broker.close()
        assert not broker.socket_path.exists()

    asyncio.run(asyncio.wait_for(run(), timeout=20))
