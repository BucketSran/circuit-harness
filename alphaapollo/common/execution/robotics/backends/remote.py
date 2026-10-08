# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical ``RobotBackend`` transported to an external robot service."""

from __future__ import annotations

import base64
import http.client
import json
import math
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from alphaapollo.common.execution.robotics.schemas import (
    RobotAction,
    RobotObservation,
    RobotResetResult,
    RobotTask,
    RobotTransition,
)


@runtime_checkable
class RobotRpcTransport(Protocol):
    """Transport for AlphaApollo's canonical robot RPC contract."""

    def call(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any: ...

    def close(self) -> None: ...


class RobotRpcError(RuntimeError):
    """Transport or remote-dispatch failure with stable method identity."""

    def __init__(self, method: str, message: str, *, traceback: str | None = None) -> None:
        super().__init__(f"{method}: {message}")
        self.method = method
        self.server_traceback = traceback


def _decode_rpc_json(value: Any) -> Any:
    if isinstance(value, dict):
        if "__ndarray__" in value and set(value) <= {"__ndarray__", "dtype", "shape"}:
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError(
                    "decoding canonical robot RPC ndarray responses requires the optional "
                    "robotics dependencies"
                ) from exc
            raw = base64.b64decode(value["__ndarray__"], validate=True)
            # Return plain nested lists: every decoded value feeds records
            # whose validation rejects ndarray, so an array here would only
            # guarantee a downstream ValueError.
            return (
                np.frombuffer(raw, dtype=value.get("dtype"))
                .reshape(value.get("shape", (-1,)))
                .tolist()
            )
        return {key: _decode_rpc_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_rpc_json(item) for item in value]
    return value


class _RpcJsonEncoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        try:
            import numpy as np
        except ImportError:
            return super().default(value)
        if isinstance(value, np.ndarray):
            return {
                "__ndarray__": base64.b64encode(value.tobytes()).decode("ascii"),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return super().default(value)


class HttpRobotRpcTransport:
    """HTTP transport for AlphaApollo's canonical robot RPC contract."""

    def __init__(self, endpoint: str, *, default_timeout_s: float = 30.0) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be non-empty")
        normalized = endpoint.strip()
        if not normalized.startswith(("http://", "https://")):
            normalized = f"http://{normalized}"
        if (
            isinstance(default_timeout_s, bool)
            or not isinstance(default_timeout_s, (int, float))
            or not math.isfinite(default_timeout_s)
            or default_timeout_s <= 0
        ):
            raise ValueError("default_timeout_s must be finite and positive")
        self.endpoint = normalized.rstrip("/")
        self.default_timeout_s = float(default_timeout_s)
        self._closed = False

    def call(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("HTTP robot RPC transport is closed")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be non-empty")
        timeout = self.default_timeout_s if timeout_s is None else timeout_s
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout_s must be finite and positive")
        body = json.dumps(
            {"method": method, "args": list(args), "kwargs": dict(kwargs or {})},
            cls=_RpcJsonEncoder,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/call",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status: int | None = None
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                raw = exc.read()
            except (OSError, http.client.HTTPException) as read_exc:
                raise RobotRpcError(
                    method, f"HTTP {exc.code} error body unreadable: {read_exc}"
                ) from read_exc
        # BadStatusLine/IncompleteRead subclass HTTPException, not OSError;
        # unwrapped they would bypass every RobotRpcError handler.
        except (OSError, http.client.HTTPException) as exc:
            raise RobotRpcError(method, f"HTTP request failed: {exc}") from exc
        try:
            envelope = _decode_rpc_json(json.loads(raw))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            # A 404 with an HTML body must say "404", not just "invalid JSON".
            prefix = f"HTTP {status} with " if status is not None else ""
            raise RobotRpcError(method, f"{prefix}invalid JSON response: {exc}") from exc
        if not isinstance(envelope, Mapping):
            raise RobotRpcError(method, f"bad response type: {type(envelope).__name__}")
        if not envelope.get("ok"):
            prefix = f"HTTP {status}: " if status is not None else ""
            raise RobotRpcError(
                method,
                prefix + str(envelope.get("error", "remote call failed")),
                traceback=envelope.get("traceback"),
            )
        return envelope.get("result")

    def close(self) -> None:
        self._closed = True


def _mapping(value: Any, *, operation: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{operation} must return a mapping")
    return dict(value)


def _keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    operation: str,
) -> None:
    missing = sorted(required - set(value))
    unexpected = sorted(set(value) - required - optional)
    if missing or unexpected:
        raise ValueError(
            f"{operation} response schema mismatch: missing={missing}, unexpected={unexpected}"
        )


def _observation(value: Any, *, operation: str) -> RobotObservation:
    payload = _mapping(value, operation=operation)
    _keys(
        payload,
        required={"state", "artifact_refs", "timestamp", "backend_metadata"},
        operation=operation,
    )
    refs = payload["artifact_refs"]
    if isinstance(refs, (str, bytes, bytearray)) or not isinstance(refs, Sequence):
        raise TypeError(f"{operation}.artifact_refs must be a sequence")
    return RobotObservation(
        state=payload["state"],
        artifact_refs=tuple(refs),
        timestamp=payload["timestamp"],
        backend_metadata=payload["backend_metadata"],
    )


def _reset_result(value: Any) -> RobotResetResult:
    payload = _mapping(value, operation="robot.reset")
    _keys(payload, required={"observation", "info"}, operation="robot.reset")
    return RobotResetResult(
        observation=_observation(payload["observation"], operation="robot.reset.observation"),
        info=payload["info"],
    )


def _validate_reset_identity(task: RobotTask, result: RobotResetResult) -> None:
    """Require the remote worker to identify the episode it actually opened."""

    environment = result.info.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("robot.reset response is missing info.environment identity metadata")
    for field in ("task_id", "benchmark", "environment_version"):
        expected = getattr(task, field)
        actual = environment.get(field)
        if actual != expected:
            raise ValueError(
                f"robot.reset environment identity mismatch for {field}: "
                f"requested {expected!r}, remote returned {actual!r}"
            )
    remote_metadata = environment.get("backend_metadata")
    if not isinstance(remote_metadata, Mapping):
        raise ValueError("robot.reset response environment identity is missing backend_metadata")
    expected_metadata = dict(task.backend_metadata)
    mismatches = {
        key: (expected, remote_metadata.get(key))
        for key, expected in expected_metadata.items()
        if remote_metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"robot.reset environment backend_metadata mismatch: {mismatches!r}")


def _transition(value: Any) -> RobotTransition:
    payload = _mapping(value, operation="robot.execute")
    _keys(
        payload,
        required={
            "observation",
            "steps_used",
            "terminated",
            "truncated",
            "success",
            "termination_reason",
            "info",
        },
        operation="robot.execute",
    )
    return RobotTransition(
        observation=_observation(payload["observation"], operation="robot.execute.observation"),
        steps_used=payload["steps_used"],
        terminated=payload["terminated"],
        truncated=payload["truncated"],
        success=payload["success"],
        termination_reason=payload["termination_reason"],
        info=payload["info"],
    )


class RemoteBackend:
    """Forward canonical records without exposing a transport to tools or agents.

    Action semantics and controller dimensions remain the remote worker's
    responsibility; this client validates only the canonical wire records.
    """

    def __init__(
        self,
        transport: RobotRpcTransport,
        *,
        timeout_s: float | None = None,
    ) -> None:
        if not isinstance(transport, RobotRpcTransport):
            raise TypeError("transport must implement RobotRpcTransport")
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be finite and positive when provided")
        self._transport = transport
        self._timeout_s = None if timeout_s is None else float(timeout_s)
        self._task: RobotTask | None = None
        self._observation: RobotObservation | None = None
        self._terminated = False
        self._truncated = False
        self._success: bool | None = None
        self._termination_reason: str | None = None
        self._steps_used = 0
        self._episode_unknown = False
        self._closed = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def success(self) -> bool | None:
        return self._success

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    @property
    def steps_used(self) -> int:
        return self._steps_used

    def reset(self, task: RobotTask) -> RobotResetResult:
        self._require_open()
        # Clear the cached episode before issuing or parsing reset.  A
        # timeout or schema error must not make the old episode runnable.
        self._invalidate_episode()
        if not isinstance(task, RobotTask):
            raise TypeError("task must be a RobotTask")
        result = _reset_result(
            self._transport.call(
                "robot.reset",
                args=(task.to_dict(),),
                timeout_s=self._timeout_s,
            )
        )
        try:
            _validate_reset_identity(task, result)
        except Exception:
            try:
                self._transport.call("robot.close", timeout_s=self._timeout_s)
            except Exception:
                pass
            raise
        self._task = task
        self._observation = result.observation
        self._terminated = False
        self._truncated = False
        self._success = None
        self._termination_reason = None
        self._steps_used = 0
        return result

    def observe(self) -> RobotObservation:
        self._require_active("observe")
        self._require_synchronized()
        # Same cost model as the in-process backends: reset/execute already
        # returned the current observation, so observe() serves the cache
        # instead of paying an RPC round trip per call (the #254 primitives
        # observe 2-3x per simulator step).
        cached = self._observation
        if cached is None:
            raise RuntimeError("reset must be called before observe")
        return cached

    def execute(self, action: RobotAction) -> RobotTransition:
        self._require_active("execute")
        self._require_synchronized()
        if not isinstance(action, RobotAction):
            raise TypeError("action must be a RobotAction")
        if self._terminated or self._truncated:
            raise RuntimeError("cannot execute an action after the remote episode is terminal")
        try:
            transition = _transition(
                self._transport.call(
                    "robot.execute",
                    args=(action.to_dict(),),
                    timeout_s=self._timeout_s,
                )
            )
        except BaseException:
            # The worker may have stepped before the response was lost or
            # rejected.  Refuse observe/execute until reset re-establishes a
            # known episode instead of under-counting the remote step budget.
            self._episode_unknown = True
            raise
        self._observation = transition.observation
        self._steps_used += transition.steps_used
        self._terminated = transition.terminated
        self._truncated = transition.truncated
        self._success = transition.success
        self._termination_reason = transition.termination_reason
        return transition

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error: Exception | None = None
        try:
            self._transport.call("robot.close", timeout_s=self._timeout_s)
        except Exception as exc:  # noqa: BLE001 - transport cleanup must still run
            error = exc
        try:
            self._transport.close()
        except Exception:
            if error is None:
                raise
        if error is not None:
            raise error

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("remote backend is closed")

    def _invalidate_episode(self) -> None:
        """Clear all locally cached episode state before a new reset."""

        self._task = None
        self._observation = None
        self._terminated = False
        self._truncated = False
        self._success = None
        self._termination_reason = None
        self._steps_used = 0
        self._episode_unknown = False

    def _require_active(self, operation: str) -> None:
        self._require_open()
        if self._task is None:
            raise RuntimeError(f"reset must be called before {operation}")

    def _require_synchronized(self) -> None:
        if self._episode_unknown:
            raise RuntimeError("remote episode state is unknown; reset is required")


__all__ = [
    "HttpRobotRpcTransport",
    "RemoteBackend",
    "RobotRpcError",
    "RobotRpcTransport",
]
