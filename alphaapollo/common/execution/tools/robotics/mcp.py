# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MCP-backed adapters for provider-neutral robotics contracts.

The module depends on a narrow client protocol instead of FastMCP. Client
creation and transport ownership stay at the composition root, while these
adapters normalize ``call_tool`` results into the core provider records.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol, runtime_checkable

from alphaapollo.common.execution.robotics import RobotAction
from alphaapollo.common.execution.tools.robotics.perception import (
    PerceptionProvider,
    PerceptionRequest,
    PerceptionResult,
)
from alphaapollo.common.execution.tools.robotics.vla import (
    VLAProvider,
    VLARequest,
    VLAResult,
)


class MCPProviderError(RuntimeError):
    """Raised when an MCP provider returns no usable structured result."""


class MCPClientError(RuntimeError):
    """Raised when the optional Streamable HTTP MCP client cannot run."""


@runtime_checkable
class MCPToolClient(Protocol):
    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> object: ...


def _load_mcp_sdk() -> tuple[Any, Any]:
    """Load the optional MCP SDK only when a configured client is built."""

    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:  # pragma: no cover - exercised in optional envs
        raise MCPClientError(
            "MCP providers require the optional 'mcp' dependency; "
            "install AlphaApollo with the mcp extra"
        ) from exc
    return ClientSession, streamable_http_client


def _positive_timeout(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


# How long the constructor waits for the transport and the MCP handshake before
# it gives up and retires the loop thread.
_STARTUP_TIMEOUT_S = 30.0
# One episode's teardown budget. Every wait on the loop thread is bounded by
# it, because the alternative to a bounded wait here is a parked thread and an
# open HTTP session that outlive the episode that opened them.
_SHUTDOWN_TIMEOUT_S = 10.0


class StreamableHTTPMCPClient:
    """Synchronous ``MCPToolClient`` facade with one owned async session.

    Workflows call providers synchronously, while the official MCP SDK exposes
    an async Streamable HTTP client.  A private event-loop thread owns the
    transport and session for the entire provider lifetime; calls are submitted
    to that loop and ``close`` tears down both contexts exactly once.
    """

    def __init__(self, endpoint: str, *, default_timeout_s: float = 30.0) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be non-empty")
        timeout = _positive_timeout(default_timeout_s, name="default_timeout_s")
        assert timeout is not None
        self.endpoint = endpoint.strip()
        self.default_timeout_s = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._state_lock = threading.Lock()
        self._session: Any = None
        self._session_task: asyncio.Task[Any] | None = None
        self._startup_event: asyncio.Event | None = None
        self._stop_event: asyncio.Event | None = None
        self._startup_error: BaseException | None = None
        self._closed = False

        thread = threading.Thread(
            target=self._thread_main,
            name="alphaapollo-mcp-client",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        if not self._ready.wait(timeout=_STARTUP_TIMEOUT_S):
            try:
                self.close()
            except BaseException:  # noqa: BLE001 - the timeout is the real error
                pass
            raise MCPClientError("timed out while initializing the MCP session")
        if self._startup_error is not None:
            error = self._startup_error
            with self._state_lock:
                self._closed = True
                thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=_SHUTDOWN_TIMEOUT_S)
            if isinstance(error, MCPClientError):
                raise error
            raise MCPClientError(
                f"failed to initialize MCP endpoint {self.endpoint!r}: {error}"
            ) from error

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> object:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be non-empty")
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be a mapping")
        timeout = (
            self.default_timeout_s
            if timeout_s is None
            else _positive_timeout(
                timeout_s,
                name="timeout_s",
            )
        )
        assert timeout is not None
        with self._state_lock:
            if self._closed:
                raise MCPClientError("MCP client is closed")
            loop = self._loop
            thread = self._thread
        if loop is None or thread is None or not thread.is_alive():
            raise MCPClientError("MCP client session is not running")
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(name, dict(arguments), timeout_s=timeout),
            loop,
        )
        try:
            return future.result(timeout=timeout + 5.0)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise MCPClientError(f"MCP tool {name!r} timed out after {timeout:g}s") from exc

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread
        if loop is None or thread is None or not thread.is_alive():
            return
        stop_event = self._stop_event
        session_task = self._session_task
        if stop_event is None or session_task is None:
            return
        loop.call_soon_threadsafe(stop_event.set)
        future = asyncio.run_coroutine_threadsafe(
            self._await_session_task(session_task),
            loop,
        )
        error: BaseException | None = None
        try:
            future.result(timeout=_SHUTDOWN_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            # Only a session parked on the stop signal can answer it. One wedged
            # in connect or in a stalled read has to be cancelled, or this
            # thread and its transport outlive the episode that created them.
            loop.call_soon_threadsafe(session_task.cancel)
        except BaseException as exc:  # noqa: BLE001 - preserve cleanup failure
            error = exc
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=_SHUTDOWN_TIMEOUT_S)
        if error is not None:
            raise error

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._startup_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        session_task = loop.create_task(self._session_main())
        self._session_task = session_task
        try:
            loop.run_until_complete(self._wait_for_startup(session_task))
        except BaseException as exc:  # noqa: BLE001 - report startup to constructor
            self._startup_error = self._startup_error or exc
            self._retire_loop(loop, session_task)
            self._ready.set()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            self._retire_loop(loop, session_task)

    def _retire_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        session_task: asyncio.Task[Any],
    ) -> None:
        """Close the loop once its session task is finished, whatever it did.

        Both exits from ``_thread_main`` end here and neither may raise: this
        frame owns ``loop.close()``, so an escaping exception leaks the loop,
        its selector, and the open transport for the process lifetime. A
        cancelled task is the case that used to escape, because
        ``Task.exception()`` re-raises ``CancelledError`` instead of returning
        it. The session's own failure is already recorded in ``_startup_error``,
        so nothing is lost by suppressing it here.
        """

        if not session_task.done():
            assert self._stop_event is not None
            self._stop_event.set()
            with contextlib.suppress(BaseException):
                # Bounded, because the stop signal only reaches a session parked
                # on it; ``wait_for`` cancels a wedged one instead of waiting out
                # a transport that will never answer.
                loop.run_until_complete(asyncio.wait_for(session_task, _SHUTDOWN_TIMEOUT_S))
        if not session_task.cancelled():
            # Retrieve the outcome so a failed session is not reported again as
            # an unretrieved task exception at interpreter exit.
            session_task.exception()
        loop.close()

    async def _session_main(self) -> None:
        ClientSession, streamable_http_client = _load_mcp_sdk()
        try:
            async with streamable_http_client(self.endpoint) as (read_stream, write_stream, _):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.default_timeout_s),
                ) as session:
                    self._session = session
                    await session.initialize()
                    assert self._startup_event is not None
                    # ``_ready`` belongs to ``_thread_main``: signalling it here
                    # would release the constructor while this thread is still
                    # inside ``run_until_complete``, where a concurrent close
                    # stops a loop that has not reached ``run_forever`` yet.
                    self._startup_event.set()
                    assert self._stop_event is not None
                    await self._stop_event.wait()
        except BaseException as exc:  # noqa: BLE001 - report startup/serve failure
            self._startup_error = exc
            if self._startup_event is not None:
                self._startup_event.set()
            raise
        finally:
            self._session = None

    async def _wait_for_startup(self, session_task: asyncio.Task[Any]) -> None:
        assert self._startup_event is not None
        await self._startup_event.wait()
        if session_task.done():
            # Re-raise, don't just retrieve: returning normally here would
            # send the thread into run_forever() after a failed connect,
            # leaving the constructor to hit its 30s join timeout and leak a
            # parked event-loop thread forever.
            exc = session_task.exception()
            if exc is not None:
                raise exc

    async def _call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> object:
        session = self._session
        if session is None:
            # The session is cleared when ``_session_main`` unwinds, which also
            # records why. Reporting only "not initialized" would hide a session
            # that started fine and then died.
            cause = self._startup_error
            if cause is not None:
                raise MCPClientError(f"MCP session for {self.endpoint!r} ended: {cause}") from cause
            raise MCPClientError("MCP client session is not initialized")
        return await asyncio.wait_for(
            session.call_tool(
                name,
                dict(arguments),
                read_timeout_seconds=timedelta(seconds=timeout_s),
            ),
            timeout=timeout_s,
        )

    async def _await_session_task(self, session_task: asyncio.Task[Any]) -> None:
        await session_task


def _content_texts(value: object) -> list[str]:
    content = getattr(value, "content", None)
    texts: list[str] = []
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        for block in content:
            text = block.get("text") if isinstance(block, Mapping) else getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
    return texts


def _error_detail(texts: Sequence[str]) -> str:
    return " ".join(" ".join(text.split()) for text in texts if text.strip())[:300]


def _mapping_result(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    # A server-side tool exception arrives as isError=True whose text block may
    # itself be JSON; it must never parse as a successful structured result.
    if bool(getattr(value, "isError", False)) or bool(getattr(value, "is_error", False)):
        detail = _error_detail(_content_texts(value))
        raise MCPProviderError("MCP tool reported an error" + (f": {detail}" if detail else ""))
    for attribute in ("structured_content", "structuredContent"):
        structured = getattr(value, attribute, None)
        if isinstance(structured, Mapping):
            return dict(structured)
    content = getattr(value, "content", None)
    texts: list[str] = []
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        for block in content:
            text = block.get("text") if isinstance(block, Mapping) else getattr(block, "text", None)
            if not isinstance(text, str):
                continue
            texts.append(text)
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, Mapping):
                return dict(decoded)
    # A tool exception on the server side arrives as a plain error text block;
    # surface it, or the caller only ever sees this generic complaint.
    detail = _error_detail(texts)
    raise MCPProviderError(
        "MCP tool returned no structured mapping" + (f"; server said: {detail}" if detail else "")
    )


def _provider_fields(
    payload: Mapping[str, Any],
    *,
    default_provider: str,
    default_model: str | None,
) -> tuple[str, str | None, Mapping[str, Any]]:
    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise MCPProviderError("metadata must be a mapping")
    metadata = dict(raw_metadata)
    reported_provider = payload.get("provider")
    reported_model = payload.get("model")
    if reported_provider is not None:
        metadata["reported_provider"] = reported_provider
    if reported_model is not None:
        metadata["reported_model"] = reported_model
    return default_provider, default_model, metadata


@dataclass(slots=True)
class MCPVLAProvider(VLAProvider):
    client: MCPToolClient
    provider: str
    model: str | None = None
    tool_name: str = "vla_act"

    def __post_init__(self) -> None:
        if not isinstance(self.client, MCPToolClient):
            raise TypeError("client must implement MCPToolClient")
        for name, value in (("provider", self.provider), ("tool_name", self.tool_name)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("model must be non-empty when provided")

    def predict(self, request: VLARequest) -> VLAResult:
        payload = _mapping_result(
            self.client.call_tool(
                self.tool_name,
                request.to_payload(),
                timeout_s=request.timeout_s,
            )
        )
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, Sequence) or isinstance(
            raw_actions,
            (str, bytes, bytearray),
        ):
            raise MCPProviderError("VLA MCP result must contain an actions sequence")
        provider, model, metadata = _provider_fields(
            payload,
            default_provider=self.provider,
            default_model=self.model,
        )
        actions: list[RobotAction] = []
        for index, raw_action in enumerate(raw_actions):
            if not isinstance(raw_action, Mapping):
                raise MCPProviderError(f"actions[{index}] must be a mapping")
            kind = raw_action.get("kind")
            arguments = raw_action.get("arguments", {})
            provenance = raw_action.get("provenance", {})
            if not isinstance(arguments, Mapping):
                raise MCPProviderError(f"actions[{index}].arguments must be a mapping")
            if not isinstance(provenance, Mapping):
                raise MCPProviderError(f"actions[{index}].provenance must be a mapping")
            normalized_provenance = dict(provenance)
            normalized_provenance.update({"provider": provider, "transport": "mcp"})
            if model is not None:
                normalized_provenance["model"] = model
            try:
                action = RobotAction(
                    kind=kind,
                    arguments=arguments,
                    provenance=normalized_provenance,
                )
            except (TypeError, ValueError) as exc:
                raise MCPProviderError(f"invalid actions[{index}]: {exc}") from exc
            actions.append(action)
        return VLAResult(
            actions=tuple(actions),
            provider=provider,
            model=model,
            metadata=metadata,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


@dataclass(slots=True)
class MCPPerceptionProvider(PerceptionProvider):
    client: MCPToolClient
    provider: str
    model: str | None = None
    segment_tool_name: str = "segment"
    detect_tool_name: str = "detect_objects"

    def __post_init__(self) -> None:
        if not isinstance(self.client, MCPToolClient):
            raise TypeError("client must implement MCPToolClient")
        for name, value in (
            ("provider", self.provider),
            ("segment_tool_name", self.segment_tool_name),
            ("detect_tool_name", self.detect_tool_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("model must be non-empty when provided")

    def inspect(self, request: PerceptionRequest) -> PerceptionResult:
        tool_name = (
            self.segment_tool_name if request.operation == "segment" else self.detect_tool_name
        )
        payload = _mapping_result(
            self.client.call_tool(
                tool_name,
                request.to_payload(),
                timeout_s=request.timeout_s,
            )
        )
        raw_items = payload.get("items")
        if not isinstance(raw_items, Sequence) or isinstance(
            raw_items,
            (str, bytes, bytearray),
        ):
            raise MCPProviderError("perception MCP result must contain an items sequence")
        provider, model, metadata = _provider_fields(
            payload,
            default_provider=self.provider,
            default_model=self.model,
        )
        return PerceptionResult(
            items=tuple(raw_items),
            provider=provider,
            model=model,
            metadata=metadata,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


__all__ = [
    "MCPPerceptionProvider",
    "MCPClientError",
    "MCPProviderError",
    "MCPToolClient",
    "MCPVLAProvider",
    "StreamableHTTPMCPClient",
]
