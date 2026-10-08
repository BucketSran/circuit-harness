# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Forward external-agent tool calls to a Runtime-owned Environment.

The external CLI launches its MCP server in a child process.  The Environment
must stay in the Runtime process so that its lifecycle, limits, and exact
transitions remain authoritative.  A mode-0600 endpoint file gives that child a
random bearer token and loopback port for this task.  This module adds no tool
semantics of its own.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alphaapollo.common.environment.base import EnvironmentTransition

__all__ = [
    "ENVIRONMENT_SOCKET_NAME",
    "EnvironmentSocketToolBridge",
    "RuntimeEnvironmentSocket",
    "normalize_environment_transition",
]

ENVIRONMENT_SOCKET_NAME = ".alphaapollo-environment.json"
_MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_CLOSE_TIMEOUT_S = 2.0


class RuntimeEnvironmentSocket:
    """Serve one initialized Environment and retain its real transitions."""

    def __init__(self, environment: Any, path: Path) -> None:
        if not callable(getattr(environment, "step", None)):
            raise TypeError("Environment must expose step()")
        self._environment = environment
        self._path = path
        self._token = secrets.token_urlsafe(32)
        self._transitions: list[EnvironmentTransition] = []
        self._errors: list[BaseException] = []
        self._state_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = threading.Event()
        self._terminal_reason: str | None = None
        self._rejected_post_termination_calls = 0
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        try:
            self._server.bind(("127.0.0.1", 0))
        except BaseException:
            self._server.close()
            raise
        self._server.listen()
        self._server.settimeout(0.1)
        host, port = self._server.getsockname()
        try:
            descriptor = json.dumps({"host": host, "port": port, "token": self._token})
            file_descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as endpoint_file:
                endpoint_file.write(descriptor)
            self._thread = threading.Thread(
                target=self._serve,
                name="alphaapollo-environment-socket",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            self._server.close()
            path.unlink(missing_ok=True)
            raise

    @property
    def transitions(self) -> tuple[EnvironmentTransition, ...]:
        with self._state_lock:
            return tuple(self._transitions)

    @property
    def errors(self) -> tuple[BaseException, ...]:
        with self._state_lock:
            return tuple(self._errors)

    @property
    def rejected_post_termination_calls(self) -> int:
        with self._state_lock:
            return self._rejected_post_termination_calls

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def close(self) -> None:
        with self._close_lock:
            self._closed.set()
            self._server.close()
            try:
                # A remote job environment can stop its client-side wait without
                # cancelling accepted work. Other environments retain the bounded
                # join and failure behavior; close() is not called prematurely.
                interrupt = getattr(self._environment, "interrupt_wait", None)
                if callable(interrupt):
                    interrupt()
            finally:
                self._thread.join(timeout=_CLOSE_TIMEOUT_S)
                self._path.unlink(missing_ok=True)
            if self._thread.is_alive():
                raise RuntimeError(
                    "Environment bridge did not finish its in-flight step before close"
                )

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                connection, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with connection:
                # A peer that has not authenticated must not be able to hold the
                # single Environment server past its close deadline.
                connection.settimeout(1.0)
                self._serve_one(connection)

    def _serve_one(self, connection: socket.socket) -> None:
        try:
            request = _receive_json(connection)
        except BaseException:  # noqa: BLE001 - an untrusted peer is not the owned session
            self._send_response(
                connection,
                _error_response(
                    "invalid_environment_bridge_request",
                    "Environment bridge request is malformed",
                ),
            )
            return
        token = request.get("token")
        if not isinstance(token, str) or not secrets.compare_digest(token, self._token):
            self._send_response(
                connection,
                _error_response(
                    "invalid_environment_bridge_token",
                    "Environment bridge authentication failed",
                ),
            )
            return
        try:
            action = request.get("action")
            if not isinstance(action, Mapping):
                raise TypeError("Environment bridge action must be a JSON object")
            with self._state_lock:
                terminal_reason = self._terminal_reason
                if terminal_reason is not None:
                    self._rejected_post_termination_calls += 1
            if terminal_reason is not None:
                self._send_response(
                    connection,
                    _error_response(
                        "environment_terminated",
                        "Runtime-owned Environment has already terminated",
                        termination_reason=terminal_reason,
                    ),
                )
                return
            transition = normalize_environment_transition(self._environment.step(action))
            with self._state_lock:
                self._transitions.append(transition)
                if transition.done:
                    self._terminal_reason = transition.termination_reason
            response = {"payload": _tool_payload(transition)}
        except BaseException as exc:  # noqa: BLE001 - surfaced by the owning Runtime
            with self._state_lock:
                self._errors.append(exc)
            response = _error_response(
                "environment_bridge_failed",
                "Runtime-owned Environment failed to step the tool call",
            )
        self._send_response(connection, response)

    def _send_response(self, connection: socket.socket, response: Mapping[str, Any]) -> None:
        try:
            _send_json(connection, response)
        except OSError:
            return
        except BaseException as exc:  # noqa: BLE001 - the owning Runtime must see truncation
            with self._state_lock:
                self._errors.append(exc)


class EnvironmentSocketToolBridge:
    """Child-process ToolBridge that carries an action over the task socket."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def dispatch(self, action: Any, _context: Any) -> _RemoteToolResult:
        endpoint = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(endpoint, Mapping):
            raise TypeError("Environment bridge endpoint must be a JSON object")
        host = endpoint.get("host")
        port = endpoint.get("port")
        token = endpoint.get("token")
        if host != "127.0.0.1" or isinstance(port, bool) or not isinstance(port, int):
            raise ValueError("Environment bridge endpoint is invalid")
        if not isinstance(token, str) or not token:
            raise ValueError("Environment bridge endpoint has no token")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.connect((host, port))
            _send_json(connection, {"token": token, "action": action})
            response = _receive_json(connection)
        error = response.get("error")
        if isinstance(error, Mapping):
            return _RemoteToolResult(
                {
                    "ok": False,
                    "status": error.get("status", "failed"),
                    "error": dict(error),
                }
            )
        payload = response.get("payload")
        if not isinstance(payload, Mapping):
            raise RuntimeError("Environment bridge returned no observation payload")
        return _RemoteToolResult(dict(payload))

    def close(self) -> None:
        return None


class _RemoteToolResult:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def observation_payload(self) -> dict[str, Any]:
        return dict(self._payload)


def _error_response(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, **details}}


def _tool_payload(transition: EnvironmentTransition) -> dict[str, Any]:
    observation = transition.observation
    if isinstance(observation, str):
        prefix = "<tool_response>\n"
        suffix = "\n</tool_response>"
        if observation.startswith(prefix) and observation.endswith(suffix):
            decoded = json.loads(observation[len(prefix) : -len(suffix)])
            if isinstance(decoded, dict):
                return decoded
    observation_record = transition.metadata.get("observation")
    error_code = (
        observation_record.get("error_code") if isinstance(observation_record, Mapping) else None
    )
    payload: dict[str, Any] = {
        "ok": transition.response_format_valid
        and transition.env_action_valid
        and error_code is None,
        "content": str(observation),
    }
    if error_code is not None:
        payload["error"] = {"code": error_code}
    return payload


def normalize_environment_transition(value: Any) -> EnvironmentTransition:
    """Copy a structural Environment result into the public canonical record."""

    if isinstance(value, EnvironmentTransition):
        return value
    if isinstance(value, Mapping):
        observation = value.get("observations")
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise TypeError("Environment transition metadata must be a mapping")
        done = value.get("done")
        reason = metadata.get("termination_reason") if done else None
        return EnvironmentTransition(
            observation=observation,
            reward=value.get("reward"),
            done=done,
            success=metadata.get("success"),
            response_format_valid=metadata.get("response_format_valid", True),
            env_action_valid=metadata.get("env_action_valid", True),
            termination_reason=reason or ("completed" if done else None),
            metadata=dict(metadata),
        )
    required = (
        "observation",
        "reward",
        "done",
        "success",
        "response_format_valid",
        "env_action_valid",
        "termination_reason",
        "metadata",
    )
    if any(not hasattr(value, name) for name in required):
        raise TypeError("Environment step must return a typed transition record")
    return EnvironmentTransition(
        observation=value.observation,
        reward=value.reward,
        done=value.done,
        success=value.success,
        response_format_valid=value.response_format_valid,
        env_action_valid=value.env_action_valid,
        termination_reason=value.termination_reason,
        metadata=dict(value.metadata),
    )


def _send_json(connection: socket.socket, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError("Environment bridge message exceeds the size limit")
    connection.sendall(payload)


def _receive_json(connection: socket.socket) -> dict[str, Any]:
    payload = bytearray()
    while b"\n" not in payload:
        chunk = connection.recv(min(65536, _MAX_MESSAGE_BYTES + 1 - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > _MAX_MESSAGE_BYTES:
            raise ValueError("Environment bridge message exceeds the size limit")
    line, separator, trailing = bytes(payload).partition(b"\n")
    if not separator or trailing:
        raise ValueError("Environment bridge requires exactly one JSON line")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise TypeError("Environment bridge message must be a JSON object")
    return value
