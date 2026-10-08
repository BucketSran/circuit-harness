from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import anyio
import pytest

from alphaapollo.common.environment.base import EnvironmentTransition
from alphaapollo.common.environment.default.environment import (
    FakeToolBridge,
    GatewayToolBridge,
    ToolBridgeResult,
)
from alphaapollo.common.execution import (
    ExecutionContext,
    ExecutionRuntime,
    ToolError,
    ToolGateway,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.execution.tools import ToolCallRecord
from alphaapollo.common.execution.tools.base import get_tool_spec
from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
from alphaapollo.reasoning.runtime.external.agents.claude_code import ClaudeCodeSession
from alphaapollo.reasoning.runtime.external.agents.codex import CodexSession
from alphaapollo.reasoning.runtime.external.agents.pi import EXTENSION_PATH, SERVERS_ENV, PiSession
from alphaapollo.reasoning.runtime.external.bridge.environment_socket import (
    ENVIRONMENT_SOCKET_NAME,
    EnvironmentSocketToolBridge,
    RuntimeEnvironmentSocket,
)
from alphaapollo.reasoning.runtime.external.bridge.mcp_config import (
    BRIDGE_MODULE,
    alphaapollo_bridge,
    normalize_mcp_servers,
)
from alphaapollo.reasoning.runtime.external.cli import CliRun

tool_bridge = pytest.importorskip(
    "alphaapollo.reasoning.runtime.external.bridge.mcp_server",
    reason="the MCP tool bridge needs the optional 'mcp' extra",
)
memory = pytest.importorskip("mcp.shared.memory")

CONTEXT = ExecutionContext(session_id="tool-bridge-test")
TASK = AgentTask(task_id="t1", system="be brief", prompt="solve")


class _Backend:
    """A sandbox that never runs anything but answers like one that did."""

    def __init__(self, record: ToolCallRecord) -> None:
        self.record = record

    def exec(self, command: str) -> ToolCallRecord:
        return self.record

    def copy_out(self, container_path: str, host_dest: str) -> None:
        return None

    def release(self) -> None:
        return None


class _Manager:
    def __init__(self, record: ToolCallRecord) -> None:
        self.backend = _Backend(record)

    def acquire(self, kind: str, **kwargs: Any) -> _Backend:
        return self.backend


def _gateway_bridge(*, allowed: Iterable[str] | None = None, **kwargs: Any) -> GatewayToolBridge:
    """A bridge with the real catalog, policy, and adapters over a fake sandbox."""

    runtime = ExecutionRuntime(sandbox_manager=_Manager(ToolCallRecord(tool_id="bash")))  # type: ignore[arg-type]
    return GatewayToolBridge(ToolGateway(runtime=runtime), allowed_tool_ids=allowed, **kwargs)


def _ok_result(*, exit_code: int = 0) -> ToolBridgeResult:
    return ToolBridgeResult(
        request=ToolRequest(call_id="c1", tool_id="bash", arguments={"command": "echo hi"}),
        response=ToolResponse(call_id="c1", tool_id="bash", stdout="hi\n", exit_code=exit_code),
        record=ToolCallRecord(tool_id="bash", exit_code=exit_code),
    )


def _drive(bridge: Any, calls: Iterable[tuple[str, dict[str, Any]]], **kwargs: Any) -> list[Any]:
    """Run real MCP round trips against an in-memory client."""

    server = tool_bridge.build_server(bridge, CONTEXT, **kwargs)

    async def _run() -> list[Any]:
        async with memory.create_connected_server_and_client_session(
            server, raise_exceptions=True
        ) as client:
            return [await client.call_tool(name, arguments) for name, arguments in calls]

    return anyio.run(_run)


def _call(bridge: Any, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
    return _drive(bridge, [(name, arguments)], **kwargs)[0]


# --- the server: what the agent sees -----------------------------------------


def test_listed_tools_mirror_the_canonical_catalog() -> None:
    def plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        return [plain(item) for item in value] if isinstance(value, (list, tuple)) else value

    server = tool_bridge.build_server(FakeToolBridge(()), CONTEXT, tool_ids=("bash", "read"))

    async def _run() -> Any:
        async with memory.create_connected_server_and_client_session(
            server, raise_exceptions=True
        ) as client:
            return await client.list_tools()

    listed = anyio.run(_run)

    assert [tool.name for tool in listed.tools] == ["bash", "read"]
    for tool in listed.tools:
        spec = get_tool_spec(tool.name)
        assert tool.description == spec.description
        assert tool.inputSchema == plain(spec.parameters)


def test_successful_call_returns_the_canonical_observation_payload() -> None:
    result = _call(
        FakeToolBridge([_ok_result()]), "bash", {"command": "echo hi"}, tool_ids=("bash",)
    )

    assert result.isError is False
    assert result.structuredContent == {
        "ok": True,
        "call_id": "c1",
        "tool_id": "bash",
        "artifacts": [],
        "stdout": "hi\n",
        "stderr": "",
        "exit_code": 0,
    }
    # The text block is the same payload, so a client that ignores structured
    # content still reads exactly what the native path puts in <tool_response>.
    assert json.loads(result.content[0].text) == result.structuredContent


def test_a_failed_command_and_a_refusal_stay_distinguishable() -> None:
    failed = _call(
        FakeToolBridge([_ok_result(exit_code=2)]), "bash", {"command": "false"}, tool_ids=("bash",)
    )
    refused = _call(
        FakeToolBridge(
            [
                ToolBridgeResult(
                    request=ToolRequest(call_id="c1", tool_id="bash", arguments={"command": "x"}),
                    error=ToolError(
                        stage="policy",
                        code="path_forbidden",
                        message="tool path must stay within /workspace",
                        call_id="c1",
                        tool_id="bash",
                    ),
                )
            ]
        ),
        "bash",
        {"command": "x"},
        tool_ids=("bash",),
    )

    assert failed.isError is True
    assert failed.structuredContent["status"] == "failed"
    assert failed.structuredContent["error"]["code"] == "nonzero_exit"
    assert refused.isError is True
    assert refused.structuredContent["status"] == "rejected"
    assert refused.structuredContent["error"]["code"] == "path_forbidden"


def test_failed_call_includes_only_explicitly_configured_recovery_guidance() -> None:
    failed = _call(
        FakeToolBridge([_ok_result(exit_code=2)]),
        "bash",
        {"command": "false"},
        tool_ids=("bash",),
        failure_guidance={"bash": "Check the command and retry after correcting stderr."},
    )
    succeeded = _call(
        FakeToolBridge([_ok_result()]),
        "bash",
        {"command": "true"},
        tool_ids=("bash",),
        failure_guidance={"bash": "unused on success"},
    )

    assert failed.structuredContent["guidance"] == (
        "Check the command and retry after correcting stderr."
    )
    assert "guidance" not in succeeded.structuredContent


def test_the_catalog_decides_what_is_a_valid_call_not_the_protocol() -> None:
    # MCP-side schema validation is off on purpose, so a wrongly typed argument
    # and an unknown name both have to arrive and be rejected by our gateway.
    mistyped, unknown = _drive(
        _gateway_bridge(), [("bash", {"command": 17}), ("not_a_tool", {})], tool_ids=("bash",)
    )

    assert mistyped.structuredContent["error"] == {
        "stage": "catalog",
        "code": "invalid_arguments",
        "message": "$.arguments.command must be a string",
        "attempted": False,
    }
    assert unknown.structuredContent["error"]["code"] == "unknown_tool"


def test_an_exhausted_tool_budget_reaches_the_agent() -> None:
    calls = [("bash", {"command": "echo hi"})] * 2

    second = _drive(
        _gateway_bridge(max_tool_calls=1),
        calls,
        tool_ids=("bash",),
        failure_guidance={"bash": "retry after changing the command"},
    )[1]

    assert second.isError is True
    assert second.structuredContent["error"]["code"] == "tool_budget_exhausted"
    assert "do not retry" in second.structuredContent["guidance"]
    assert "changing the command" not in second.structuredContent["guidance"]


def test_parallel_calls_never_share_the_one_sandbox() -> None:
    # Agents do issue tool calls in parallel -- pi does it by default -- but one
    # gateway session owns one container, so overlapping dispatches would race
    # for it and come back `sandbox_unavailable`.
    class _Probe:
        def __init__(self) -> None:
            self.max_active = 0
            self._active = 0
            self._lock = threading.Lock()

        def dispatch(self, action: Any, context: ExecutionContext) -> ToolBridgeResult:
            with self._lock:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
            time.sleep(0.05)
            with self._lock:
                self._active -= 1
            return _ok_result()

    probe = _Probe()
    server = tool_bridge.build_server(probe, CONTEXT, tool_ids=("bash",))

    async def _run() -> None:
        async with memory.create_connected_server_and_client_session(
            server, raise_exceptions=True
        ) as client:
            async with anyio.create_task_group() as tasks:
                for _ in range(4):
                    tasks.start_soon(client.call_tool, "bash", {"command": "echo hi"})

    anyio.run(_run)

    assert probe.max_active == 1


def test_mcp_call_steps_the_runtime_owned_environment(tmp_path: Path) -> None:
    class _Environment:
        def __init__(self) -> None:
            self.actions: list[object] = []

        def step(self, action: object) -> EnvironmentTransition:
            self.actions.append(action)
            return EnvironmentTransition(
                observation='<tool_response>\n{"ok": true, "content": "hi"}\n</tool_response>',
                reward=0.5,
                done=False,
                metadata={"tool_request": {"tool_id": "bash"}},
            )

    environment = _Environment()
    endpoint = tmp_path / ENVIRONMENT_SOCKET_NAME
    owner = RuntimeEnvironmentSocket(environment, endpoint)
    assert endpoint.stat().st_mode & 0o777 == 0o600
    try:
        result = _call(
            EnvironmentSocketToolBridge(endpoint),
            "bash",
            {"command": "echo hi"},
            tool_ids=("bash",),
        )
    finally:
        owner.close()

    assert result.structuredContent == {"ok": True, "content": "hi"}
    assert len(owner.transitions) == 1
    action = environment.actions[0]
    assert isinstance(action, dict)
    assert action["function"] == {"name": "bash", "arguments": {"command": "echo hi"}}


def test_terminal_environment_rejects_later_calls_without_stepping(tmp_path: Path) -> None:
    class _Environment:
        def __init__(self) -> None:
            self.actions: list[object] = []

        def step(self, action: object) -> EnvironmentTransition:
            self.actions.append(action)
            return EnvironmentTransition(
                observation="limit reached",
                reward=0.0,
                done=True,
                success=False,
                termination_reason="max_tool_calls",
                metadata={"tool_request": {"tool_id": "bash"}},
            )

    environment = _Environment()
    endpoint = tmp_path / ENVIRONMENT_SOCKET_NAME
    owner = RuntimeEnvironmentSocket(environment, endpoint)
    bridge = EnvironmentSocketToolBridge(endpoint)
    try:
        first = bridge.dispatch(
            {"id": "first", "function": {"name": "bash", "arguments": {}}}, None
        ).observation_payload()
        second = bridge.dispatch(
            {"id": "second", "function": {"name": "bash", "arguments": {}}}, None
        ).observation_payload()
    finally:
        owner.close()

    assert first["ok"] is True
    assert second["error"]["code"] == "environment_terminated"
    assert second["error"]["termination_reason"] == "max_tool_calls"
    assert len(environment.actions) == 1
    assert len(owner.transitions) == 1
    assert owner.rejected_post_termination_calls == 1
    assert owner.errors == ()


def test_rejected_local_peers_do_not_poison_the_owned_session(tmp_path: Path) -> None:
    class _Environment:
        def step(self, _action: object) -> EnvironmentTransition:
            return EnvironmentTransition(observation="ok", reward=0.0, done=False)

    endpoint_path = tmp_path / ENVIRONMENT_SOCKET_NAME
    owner = RuntimeEnvironmentSocket(_Environment(), endpoint_path)
    endpoint = json.loads(endpoint_path.read_text())
    address = (endpoint["host"], endpoint["port"])
    try:
        with socket.create_connection(address) as connection:
            connection.sendall(b'{"token":"wrong","action":{}}\n')
            assert json.loads(connection.makefile().readline())["error"]["code"] == (
                "invalid_environment_bridge_token"
            )
        with socket.create_connection(address) as connection:
            connection.sendall(b"{")
        payload = (
            EnvironmentSocketToolBridge(endpoint_path)
            .dispatch({"id": "owned", "function": {"name": "bash", "arguments": {}}}, None)
            .observation_payload()
        )
    finally:
        owner.close()

    assert payload == {"ok": True, "content": "ok"}
    assert len(owner.transitions) == 1
    assert owner.errors == ()


def test_close_fails_loudly_while_an_environment_step_is_in_flight(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    class _Environment:
        def step(self, _action: object) -> EnvironmentTransition:
            started.set()
            assert release.wait(timeout=10.0)
            return EnvironmentTransition(observation="done", reward=0.0, done=False)

    endpoint = tmp_path / ENVIRONMENT_SOCKET_NAME
    owner = RuntimeEnvironmentSocket(_Environment(), endpoint)
    results: list[dict[str, Any]] = []

    def dispatch() -> None:
        results.append(
            EnvironmentSocketToolBridge(endpoint)
            .dispatch({"id": "slow", "function": {"name": "bash", "arguments": {}}}, None)
            .observation_payload()
        )

    worker = threading.Thread(target=dispatch)
    worker.start()
    assert started.wait(timeout=2.0)

    with pytest.raises(RuntimeError, match="in-flight step"):
        owner.close()
    assert owner.is_alive is True

    release.set()
    worker.join(timeout=2.0)
    owner.close()

    assert worker.is_alive() is False
    assert owner.is_alive is False
    assert results == [{"ok": True, "content": "done"}]
    assert len(owner.transitions) == 1


def test_build_server_rejects_a_selection_it_cannot_serve() -> None:
    for tool_ids, expected in (
        (("not_a_tool",), KeyError),
        (("python",), ValueError),  # internal, never model-visible
        ((), ValueError),
        ("bash", ValueError),  # a bare string would iterate into characters
    ):
        with pytest.raises(expected):
            tool_bridge.build_server(FakeToolBridge(()), CONTEXT, tool_ids=tool_ids)
    with pytest.raises(ValueError, match="unselected tool ids"):
        tool_bridge.build_server(
            FakeToolBridge(()),
            CONTEXT,
            tool_ids=("bash",),
            failure_guidance={"read": "not selected"},
        )
    with pytest.raises(ValueError, match="non-empty tool ids and messages"):
        tool_bridge.build_server(
            FakeToolBridge(()),
            CONTEXT,
            tool_ids=("bash",),
            failure_guidance={1: "invalid key", "read": "also unselected"},
        )


@pytest.mark.skipif(
    os.environ.get("ALPHAAPOLLO_LIVE_MATH_TOOLS") != "1",
    reason="needs ALPHAAPOLLO_LIVE_MATH_TOOLS=1 and rootless Podman",
)
def test_live_entrypoint_executes_public_python_in_the_shared_sandbox() -> None:
    stdio = pytest.importorskip("mcp.client.stdio")
    session = pytest.importorskip("mcp.client.session")
    params = stdio.StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            BRIDGE_MODULE,
            "--tools",
            "python_execute",
            "--timeout-s",
            "30",
        ],
    )

    async def _run() -> Any:
        async with stdio.stdio_client(params) as (read_stream, write_stream):
            async with session.ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                return await client.call_tool("python_execute", {"code": "6 * 7"})

    result = anyio.run(_run)

    assert result.isError is False
    assert result.structuredContent["tool_id"] == "python_execute"
    assert result.structuredContent["stdout"] == "42\n"


def test_the_environment_entrypoint_serves_tools_over_real_stdio(tmp_path: Path) -> None:
    stdio = pytest.importorskip("mcp.client.stdio")
    session = pytest.importorskip("mcp.client.session")
    endpoint = tmp_path / ENVIRONMENT_SOCKET_NAME

    class Environment:
        def step(self, _action: object) -> EnvironmentTransition:
            return EnvironmentTransition(observation="unused", reward=0, done=False)

    owner = RuntimeEnvironmentSocket(Environment(), endpoint)
    params = stdio.StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            BRIDGE_MODULE,
            "--tools",
            "bash",
            "--environment-socket",
            str(endpoint),
        ],
    )

    async def _run() -> list[str]:
        async with stdio.stdio_client(params) as (read_stream, write_stream):
            async with session.ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                return [tool.name for tool in (await client.list_tools()).tools]

    try:
        assert anyio.run(_run) == ["bash"]
    finally:
        owner.close()


# --- the wiring: how each CLI is told about it -------------------------------


def test_the_bridge_declaration_names_this_interpreter_and_its_limits() -> None:
    assert alphaapollo_bridge(tool_ids=("read", "bash"), timeout_s=30, max_tool_calls=4) == {
        # sys.executable, not `python`: the CLI inherits the operator's PATH.
        "command": sys.executable,
        "args": [
            "-m",
            BRIDGE_MODULE,
            "--tools",
            "read,bash",
            "--timeout-s",
            "30.0",
            "--max-tool-calls",
            "4",
        ],
        "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[3])},
    }


def test_a_malformed_declaration_fails_before_any_agent_starts() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        normalize_mcp_servers({"s": {"command": "python", "transport": "stdio"}})
    for spec in ({}, {"command": "python", "url": "https://x.invalid"}):
        with pytest.raises(ValueError, match="exactly one"):
            normalize_mcp_servers({"s": spec})
    with pytest.raises(TypeError, match="must be a string"):
        normalize_mcp_servers({"s": {"command": "python", "args": [7]}})


def test_an_http_server_survives_normalization_for_the_clients_that_can_use_one() -> None:
    assert normalize_mcp_servers({"s": {"url": "https://x.invalid", "headers": {"A": "b"}}}) == {
        "s": {"url": "https://x.invalid", "headers": {"A": "b"}}
    }


def test_claude_argv_pins_the_run_to_the_declared_servers() -> None:
    argv = ClaudeCodeSession(
        safe_mode=False, mcp_servers={"alphaapollo": alphaapollo_bridge(tool_ids=("bash",))}
    ).argv(TASK)

    document = json.loads(argv[argv.index("--mcp-config") + 1])
    assert document["mcpServers"]["alphaapollo"]["command"] == sys.executable
    assert "--strict-mcp-config" in argv
    # Bridging forfeits --safe-mode, so isolation is bought back where it can be.
    assert argv[argv.index("--setting-sources") + 1] == ""


def test_codex_argv_appends_bridge_overrides_after_explicit_ones() -> None:
    argv = CodexSession(
        sandbox="workspace-write",
        config_overrides=("model_provider=oss",),
        mcp_servers={"alphaapollo": {"command": "python", "args": ["-m", BRIDGE_MODULE]}},
        mcp_auto_approve_tools={"alphaapollo": ("bash",)},
    ).argv()

    assert argv[argv.index("model_provider=oss") + 1] == "-c"
    assert 'mcp_servers.alphaapollo.command="python"' in argv
    assert f'mcp_servers.alphaapollo.args=["-m", "{BRIDGE_MODULE}"]' in argv
    assert 'mcp_servers.alphaapollo.tools.bash.approval_mode="approve"' in argv
    assert argv[argv.index("-s") + 1] == "workspace-write"
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv


def test_pi_loads_the_bundled_extension_and_keeps_its_credentials() -> None:
    session = PiSession(
        tool_mode="no_builtin",
        env={"PI_API_KEY": "secret"},
        mcp_servers={"alphaapollo": {"command": "python", "args": ["-m", BRIDGE_MODULE]}},
    )

    assert session.argv(TASK)[session.argv(TASK).index("-e") + 1] == str(EXTENSION_PATH)
    assert EXTENSION_PATH.is_file()
    # pi cannot pass arguments to an extension, so the declaration travels in
    # the environment -- which must be added to, not replaced.
    assert session._settings.env["PI_API_KEY"] == "secret"
    assert json.loads(session._settings.env[SERVERS_ENV]) == {
        "alphaapollo": {"command": "python", "args": ["-m", BRIDGE_MODULE]}
    }


def test_pi_delivers_raw_event_stream_before_parsing_failure(tmp_path: Path) -> None:
    raw = '{"type": "unfinished"\n'
    captured = []

    def runner(*_args, **_kwargs):
        return CliRun(raw, "", 0, False)

    session = PiSession(runner=runner, event_sink=captured.append)
    with pytest.raises(ValueError, match="not valid JSON"):
        session.run(TASK, workspace=tmp_path)
    assert captured == [raw]


def test_each_agent_refuses_the_setting_that_would_silently_drop_the_bridge() -> None:
    stdio = {"s": {"command": "python"}}

    # --safe-mode disables MCP servers even when named on the command line.
    with pytest.raises(ValueError, match="safe_mode"):
        ClaudeCodeSession(safe_mode=True, mcp_servers=stdio)
    # A headless run must name the exact tools it is allowed to auto-approve.
    with pytest.raises(ValueError, match="explicit auto-approved"):
        CodexSession(sandbox="read-only", mcp_servers=stdio)
    with pytest.raises(ValueError, match="sandbox must be one of"):
        CodexSession(sandbox=None)
    with pytest.raises(ValueError, match="undeclared servers"):
        CodexSession(mcp_auto_approve_tools={"s": ("bash",)})
    # --no-tools would disable extension tools too; the default mode would leave
    # pi's own tools beside ours.
    for mode in ("default", "none"):
        with pytest.raises(ValueError, match="bridged run must use"):
            PiSession(tool_mode=mode, mcp_servers=stdio)
    with pytest.raises(ValueError, match="stdio only"):
        PiSession(tool_mode="no_builtin", mcp_servers={"s": {"url": "https://x.invalid"}})


def test_an_unbridged_session_is_untouched() -> None:
    claude = ClaudeCodeSession().argv(TASK)
    assert "--mcp-config" not in claude and "--safe-mode" in claude

    codex = CodexSession().argv()
    assert "-c" not in codex and codex[codex.index("-s") + 1] == "read-only"

    pi_session = PiSession()
    assert "-e" not in pi_session.argv(TASK)
    assert pi_session._settings.env is None


# --- live -------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("ALPHAAPOLLO_LIVE_BRIDGED_PI") != "1" or shutil.which("pi") is None,
    reason="needs ALPHAAPOLLO_LIVE_BRIDGED_PI=1, the pi CLI, and podman",
)
def test_live_pi_calls_the_bridged_tools_under_their_native_names(tmp_path: Path) -> None:
    from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalAgentRuntime

    runtime = ExternalAgentRuntime(
        lambda: PiSession(
            tool_mode="no_builtin",
            mcp_servers={"alphaapollo": alphaapollo_bridge(tool_ids=("bash",), timeout_s=60)},
            timeout_s=300.0,
        ),
        agent="pi",
        model="default",
        workspace_root=tmp_path,
    )
    result = runtime.run(
        AgentTask(
            task_id="bridged-pi",
            system="End with exactly: Final answer: <number>",
            prompt="Run `python -c 'print(9//3*60 + 24)'` in the shell and report what it printed.",
        )
    )

    assert result.termination_reason == "final"
    assert "204" in result.final_text
    # A pi extension registers plain names, so the bridged surface is the same
    # one the AlphaApollo runtime shows its own model. No other agent can do it.
    called = [call.name for turn in result.turns for call in turn.generation_response.tool_calls]
    assert called == ["bash"]
    assert all(turn.generation_response.is_trainable is False for turn in result.turns)


@pytest.mark.skipif(
    os.environ.get("ALPHAAPOLLO_LIVE_BRIDGED_CLAUDE") != "1" or shutil.which("claude") is None,
    reason="needs ALPHAAPOLLO_LIVE_BRIDGED_CLAUDE=1, the claude CLI, and podman",
)
def test_live_claude_code_solves_a_task_with_only_bridged_tools(tmp_path: Path) -> None:
    from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalAgentRuntime

    runtime = ExternalAgentRuntime(
        lambda: ClaudeCodeSession(
            model="sonnet",
            safe_mode=False,
            tools=[],
            permission_mode="bypassPermissions",
            mcp_servers={"alphaapollo": alphaapollo_bridge(tool_ids=("bash",), timeout_s=60)},
            timeout_s=300.0,
        ),
        agent="claude_code",
        model="sonnet",
        workspace_root=tmp_path,
    )
    result = runtime.run(
        AgentTask(
            task_id="bridged-claude",
            system="End with exactly: Final answer: <number>",
            prompt="Run `python -c 'print(9//3*60 + 24)'` in the shell and report what it printed.",
        )
    )

    assert result.termination_reason == "final"
    assert "204" in result.final_text
    # Built-ins are off and MCP config is strict, so this was the only tool the
    # model had: the answer came out of AlphaApollo's sandbox, not Claude's.
    called = [call.name for turn in result.turns for call in turn.generation_response.tool_calls]
    assert called == ["mcp__alphaapollo__bash"]
    assert all(turn.generation_response.is_trainable is False for turn in result.turns)
