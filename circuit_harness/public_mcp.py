"""Public-only MCP transport outside the native Agent's filesystem sandbox."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import signal
import tempfile
import uuid
from pathlib import Path

_CLIENT = r"""
import socket, sys, threading
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(sys.argv[1])
def upload():
    try:
        while True:
            data = sys.stdin.buffer.read1(65536)
            if not data:
                sock.shutdown(socket.SHUT_WR)
                return
            sock.sendall(data)
    except OSError:
        pass
threading.Thread(target=upload, daemon=True).start()
try:
    while True:
        data = sock.recv(65536)
        if not data:
            break
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
finally:
    sock.close()
"""


class PublicMCPBroker:
    """Give an isolated Agent a socket, never the private session directory.

    A disconnected worker may still finish its bounded public action. Reconnects
    use the same session's durable action records and budget. This object owns
    transport workers only, not session creation, freezing or evaluation.
    """

    def __init__(self, session_directory: Path, directory: Path, python: Path):
        self.session_directory = Path(session_directory).resolve()
        self.directory = Path(directory).resolve()
        self.python = Path(python).absolute()
        self.client_path = self.directory / "public_mcp_client.py"
        self.socket_path: Path | None = None
        self._socket_directory: Path | None = None
        self._server = None
        self._workers: set[asyncio.subprocess.Process] = set()
        self._connections: set[asyncio.StreamWriter] = set()
        self._tasks: set[asyncio.Task] = set()
        self._closing = False
        self._drain_timeout = 315.0

    async def start(self):
        if self._server is not None:
            raise RuntimeError("MCP broker already started")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.client_path.write_text(_CLIENT)
        self.client_path.chmod(0o400)
        # macOS limits Unix socket addresses to 104 bytes. This temporary path
        # holds only a private transport socket; logs stay in the trial directory.
        self._socket_directory = Path(tempfile.mkdtemp(prefix="chips-mcp-", dir="/tmp"))
        self.socket_path = self._socket_directory / "public.sock"
        try:
            config = json.loads((self.session_directory / "session.json").read_text())
            self._drain_timeout = float(config["timeout_s"]) + 15
            self._server = await asyncio.start_unix_server(
                self._connect,
                path=self.socket_path,
                limit=16 * 1024 * 1024 * 6 + 8192,
            )
            self.socket_path.chmod(0o600)
        except BaseException:
            await self.close()
            raise
        return self

    def client_command(self) -> list[str]:
        if self.socket_path is None:
            raise RuntimeError("start the MCP broker first")
        return [str(self.python), "-I", "-S", str(self.client_path), str(self.socket_path)]

    async def _connect(self, reader, writer):
        if self._closing or self._connections:
            writer.close()
            return
        task = asyncio.current_task()
        self._tasks.add(task)
        self._connections.add(writer)
        process = None
        pumps = []
        try:
            log = self.directory / ("worker-" + uuid.uuid4().hex + ".stderr.log")
            root = Path(__file__).resolve().parents[1]
            with log.open("wb") as stderr:
                process = await asyncio.create_subprocess_exec(
                    str(self.python),
                    "-m",
                    "circuit_harness.public_mcp",
                    "serve",
                    str(self.session_directory),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=stderr,
                    start_new_session=True,
                    env={
                        "PATH": os.environ.get("PATH", os.defpath),
                        "PYTHONPATH": str(root),
                        "PYTHONUNBUFFERED": "1",
                    },
                )
            self._workers.add(process)

            async def upload():
                try:
                    while data := await reader.readline():
                        process.stdin.write(data)
                        await process.stdin.drain()
                finally:
                    process.stdin.close()

            async def download():
                while data := await process.stdout.read(65536):
                    writer.write(data)
                    await writer.drain()

            pumps = [asyncio.create_task(upload()), asyncio.create_task(download())]
            done, _ = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                finished.result()
            if pumps[0] in done:
                await pumps[1]
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            for pump in pumps:
                pump.cancel()
            if pumps:
                await asyncio.gather(*pumps, return_exceptions=True)
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
            self._connections.discard(writer)
            if process is not None:
                # Public executions already own bounded deadlines and immutable
                # inputs. Do not kill them merely because the client went away.
                await process.wait()
                self._workers.discard(process)
            self._tasks.discard(task)

    async def close(self):
        self._closing = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for writer in tuple(self._connections):
            writer.close()
        for process in tuple(self._workers):
            process.stdin.close()
        cleanup_failed = False
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(p.wait() for p in tuple(self._workers))),
                    timeout=self._drain_timeout,
                )
            except TimeoutError:
                cleanup_failed = True
                for process in tuple(self._workers):
                    if process.returncode is None:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                await asyncio.gather(*(p.wait() for p in tuple(self._workers)))
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        if self._socket_directory is not None:
            shutil.rmtree(self._socket_directory)
            self._socket_directory = None
        if cleanup_failed:
            raise RuntimeError("public MCP worker exceeded its bounded cleanup deadline")


def serve(directory: Path):
    """Expose exactly the session's public tools, with no final evaluator."""
    import resource

    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    from circuit_harness.execution.current_evas_session import (
        session_action,
        session_info,
    )

    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))

    server = Server("chips-current-evas-public")
    public_tools = session_info(directory)["tools"]

    @server.list_tools()
    async def list_tools():
        result = []
        for row in public_tools:
            function = row["function"]
            parameters = function["parameters"]
            result.append(
                types.Tool(
                    name=function["name"],
                    description=function["description"],
                    inputSchema={
                        **parameters,
                        "properties": {
                            **parameters["properties"],
                            "action_id": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 80,
                                "description": "Reuse this ID only to recover the same action.",
                            },
                        },
                    },
                )
            )
        return result

    @server.call_tool()
    async def call_tool(name, arguments):
        # The caller may supply a stable action ID for reconnects. A new call
        # without one gets a fresh ID; this never resets the session budget.
        arguments = dict(arguments or {})
        action_id = arguments.pop("action_id", None) or uuid.uuid4().hex
        response = await anyio.to_thread.run_sync(
            session_action,
            directory,
            {"action_id": action_id, "tool": name, "arguments": arguments},
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(response))],
            isError=not response.get("ok", False),
        )

    async def run():
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    anyio.run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["serve"])
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    serve(args.session)
