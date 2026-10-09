"""Authenticated public session transport owned by one Harbor trial.

The environment creates/freezes the session. This gateway owns HTTP admissions
and drains accepted actions before the environment may collect the candidate.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import secrets
import socket
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from circuit_harness.execution.sessions.current_evas_session import session_action, session_info


class _Server(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self):
        # Harbor owns the surrounding event loop and process signals.
        yield


class PublicSessionGateway:
    def __init__(
        self,
        session_directory: Path,
        *,
        bind_host: str,
        advertised_host: str,
        max_request_bytes: int = 1024 * 1024,
        max_connections: int = 8,
        request_timeout_s: float = 10,
    ):
        if (
            not bind_host
            or not advertised_host
            or any(character in advertised_host for character in "/@?#\r\n")
        ):
            raise ValueError("explicit bind and advertised hosts are required")
        if (
            max_request_bytes < 1
            or max_connections < 1
            or not math.isfinite(request_timeout_s)
            or request_timeout_s <= 0
        ):
            raise ValueError("gateway limits must be positive")
        self.session_directory = Path(session_directory).resolve()
        self.bind_host = bind_host
        self.advertised_host = advertised_host
        self.max_request_bytes = max_request_bytes
        self.max_connections = max_connections
        self.request_timeout_s = request_timeout_s
        self.token = secrets.token_urlsafe(32)
        self.url = ""
        self._closing = False
        self._active = 0
        self._actions: set[asyncio.Task] = set()
        self._server = None
        self._serve_task = None
        self._close_task = None
        self._socket = None
        self._app = Starlette(
            routes=[Route("/info", self._info), Route("/action", self._action, methods=["POST"])]
        )

    async def start(self) -> PublicSessionGateway:
        if self._serve_task is not None or self._closing:
            raise RuntimeError("gateway cannot be started again")
        config = json.loads((self.session_directory / "session.json").read_text())
        self._drain_timeout = float(config["timeout_s"]) + 15
        addresses = socket.getaddrinfo(self.bind_host, 0, type=socket.SOCK_STREAM)
        family, socktype, protocol, _, address = addresses[0]
        self._socket = socket.socket(family, socktype, protocol)
        try:
            self._socket.bind(address)
            self._socket.listen(self.max_connections)
            self._socket.setblocking(False)
            host = self.advertised_host
            if ":" in host and not host.startswith("["):
                host = "[" + host + "]"
            self.url = f"http://{host}:{self._socket.getsockname()[1]}"
            self._server = _Server(
                uvicorn.Config(
                    self._dispatch,
                    interface="asgi3",
                    lifespan="off",
                    ws="none",
                    http="h11",
                    log_config=None,
                    log_level="critical",
                    access_log=False,
                    server_header=False,
                    timeout_keep_alive=2,
                    timeout_graceful_shutdown=self.request_timeout_s + 2,
                    limit_concurrency=self.max_connections * 4,
                )
            )
            self._serve_task = asyncio.create_task(self._server.serve(sockets=[self._socket]))
            async with asyncio.timeout(5):
                while not self._server.started:
                    if self._serve_task.done():
                        await self._serve_task
                        raise RuntimeError("public gateway failed to start")
                    await asyncio.sleep(0.01)
            return self
        except BaseException:
            await self.close()
            raise

    async def _dispatch(self, scope, receive, send):
        request = Request(scope, receive)
        authorization = request.headers.get("authorization", "")
        if not secrets.compare_digest(authorization.encode(), ("Bearer " + self.token).encode()):
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
        elif self._closing:
            response = JSONResponse({"error": "gateway_closed"}, status_code=503)
        elif self._active >= self.max_connections:
            response = JSONResponse({"error": "gateway_busy"}, status_code=503)
        else:
            self._active += 1
            try:
                await self._app(scope, receive, send)
            finally:
                self._active -= 1
            return
        await response(scope, receive, send)

    async def _info(self, request):
        try:
            info = session_info(self.session_directory)
            info.pop("public_workspace", None)
            info.pop("candidate_workspace", None)
            return JSONResponse(info)
        except Exception:
            return JSONResponse({"error": "session_unavailable"}, status_code=503)

    async def _action(self, request):
        try:
            declared_length = request.headers.get("content-length")
            if declared_length is not None and int(declared_length) > self.max_request_bytes:
                return JSONResponse({"error": "request_too_large"}, status_code=413)
            data = bytearray()
            async with asyncio.timeout(self.request_timeout_s):
                async for chunk in request.stream():
                    if len(data) + len(chunk) > self.max_request_bytes:
                        return JSONResponse({"error": "request_too_large"}, status_code=413)
                    data.extend(chunk)
            payload = json.loads(data)
        except TimeoutError:
            return JSONResponse({"error": "request_timeout"}, status_code=408)
        except (ValueError, UnicodeError, RecursionError, ClientDisconnect):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        if self._closing:
            return JSONResponse({"error": "gateway_closed"}, status_code=503)
        worker = asyncio.create_task(asyncio.to_thread(self._execute, payload))
        self._actions.add(worker)
        worker.add_done_callback(self._actions.discard)
        # A disconnected/cancelled HTTP handler cannot cancel durable session work.
        return await asyncio.shield(worker)

    def _execute(self, payload):
        try:
            reply = session_action(self.session_directory, payload)
            reply.pop("detail", None)
            if isinstance(reply.get("result"), dict):
                reply["result"].pop("candidate_directory", None)
            return JSONResponse(reply)
        except ValueError:
            return JSONResponse({"error": "invalid_action"}, status_code=400)
        except Exception:
            return JSONResponse(
                {"error": "session_unavailable", "retry_safe": False}, status_code=503
            )

    async def close(self):
        """Stop admissions immediately, then drain actions, including disconnected clients.

        A drain timeout raises instead of declaring the session safe to freeze.
        Cancellation waits for shutdown; accepted thread work is never cancelled.
        """
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._shutdown())
        cancelled = False
        while True:
            try:
                await asyncio.shield(self._close_task)
                break
            except asyncio.CancelledError:
                if self._close_task.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _shutdown(self):
        if self._server is not None:
            self._server.should_exit = True
        try:
            if self._actions:
                _, pending = await asyncio.wait(self._actions, timeout=self._drain_timeout)
                if pending:
                    raise RuntimeError(
                        "public actions remain unresolved; retain session for recovery"
                    )
            if self._serve_task is not None:
                await self._serve_task
        finally:
            if self._socket is not None:
                self._socket.close()
