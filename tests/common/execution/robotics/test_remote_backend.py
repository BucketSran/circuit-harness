"""Remote robot backend transport and schema-boundary tests."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from contextlib import AbstractContextManager
from email.message import Message
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from alphaapollo.common.execution.robotics import (
    RobotAction,
    RobotBackend,
    RobotObservation,
    RobotTask,
    RobotTransition,
)
from alphaapollo.common.execution.robotics.backends import (
    HttpRobotRpcTransport,
    RemoteBackend,
    RobotRpcError,
)


def _observation(frame: int) -> RobotObservation:
    return RobotObservation(
        state={"frame": frame},
        timestamp=f"2026-08-16T00:00:0{frame}Z",
        backend_metadata={"backend": "remote"},
    )


def _task() -> RobotTask:
    return RobotTask(
        task_id="remote-task-1",
        benchmark="libero",
        instruction="pick up the cup",
        environment_version="libero-0.1",
        backend_metadata={"suite": "libero_spatial", "task": 0, "seed": 7},
    )


def _identity(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "backend": "remote",
        "task_id": task["task_id"],
        "benchmark": task["benchmark"],
        "environment_version": task["environment_version"],
        "backend_metadata": task["backend_metadata"],
    }


def _transition(frame: int, *, terminal: bool) -> RobotTransition:
    return RobotTransition(
        observation=_observation(frame),
        steps_used=3,
        terminated=terminal,
        truncated=False,
        success=True if terminal else None,
        termination_reason="task_success" if terminal else None,
        info={"server_step": frame},
    )


class _Transport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], Mapping[str, Any] | None, float | None]] = []
        self.closed = False
        self.fail_reset = False
        self.transitions = [_transition(1, terminal=False), _transition(2, terminal=True)]

    def call(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        self.calls.append((method, args, kwargs, timeout_s))
        if method == "robot.reset":
            if self.fail_reset:
                raise TimeoutError("remote reset timed out")
            task = args[0]
            return {
                "observation": _observation(0).to_dict(),
                "info": {"worker": "a", "environment": _identity(task)},
            }
        if method == "robot.observe":
            return _observation(0).to_dict()
        if method == "robot.execute":
            return self.transitions.pop(0).to_dict()
        if method == "robot.close":
            return {"closed": True}
        raise AssertionError(method)

    def close(self) -> None:
        self.closed = True


def test_remote_backend_round_trips_canonical_records_and_accumulates_steps() -> None:
    transport = _Transport()
    backend = RemoteBackend(transport, timeout_s=12)

    reset = backend.reset(_task())
    observed = backend.observe()
    first = backend.execute(RobotAction(kind="continuous", arguments={"values": [0.1]}))
    second = backend.execute(RobotAction(kind="continuous", arguments={"values": [0.2]}))

    assert isinstance(backend, RobotBackend)
    assert reset.info["worker"] == "a"
    assert observed.state["frame"] == 0
    assert first.done is False
    assert second.success is True
    assert backend.steps_used == 6
    assert backend.terminated is True
    assert backend.success is True
    assert [call[0] for call in transport.calls] == [
        "robot.reset",
        "robot.execute",
        "robot.execute",
    ]
    assert transport.calls[0][1][0]["task_id"] == "remote-task-1"
    assert transport.calls[2][1][0]["kind"] == "continuous"
    assert all(call[3] == 12 for call in transport.calls)


def test_remote_backend_rejects_response_schema_drift_before_publishing_state() -> None:
    class _DriftedTransport(_Transport):
        def call(self, method: str, **kwargs: Any) -> Any:
            if method == "robot.reset":
                payload = {"observation": _observation(0).to_dict(), "info": {}}
                payload["invented_success"] = True
                return payload
            return super().call(method, **kwargs)

    backend = RemoteBackend(_DriftedTransport())

    with pytest.raises(ValueError, match="unexpected=.*invented_success"):
        backend.reset(_task())
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()


def test_remote_backend_rejects_missing_reset_identity() -> None:
    class _MissingIdentityTransport(_Transport):
        def call(self, method: str, **kwargs: Any) -> Any:
            if method == "robot.reset":
                return {"observation": _observation(0).to_dict(), "info": {}}
            return super().call(method, **kwargs)

    backend = RemoteBackend(_MissingIdentityTransport())

    with pytest.raises(ValueError, match="missing info.environment identity"):
        backend.reset(_task())


def test_remote_backend_rejects_reset_identity_drift() -> None:
    class _DriftedIdentityTransport(_Transport):
        def call(self, method: str, **kwargs: Any) -> Any:
            if method == "robot.reset":
                task = kwargs["args"][0]
                identity = _identity(task)
                identity["environment_version"] = "libero-stale"
                return {
                    "observation": _observation(0).to_dict(),
                    "info": {"environment": identity},
                }
            return super().call(method, **kwargs)

    backend = RemoteBackend(_DriftedIdentityTransport())

    with pytest.raises(ValueError, match="identity mismatch for environment_version"):
        backend.reset(_task())


def test_remote_failed_rereset_invalidates_the_previous_episode() -> None:
    transport = _Transport()
    backend = RemoteBackend(transport)
    backend.reset(_task())
    transport.fail_reset = True

    with pytest.raises(TimeoutError, match="timed out"):
        backend.reset(_task())

    assert backend.terminated is False
    assert backend.truncated is False
    assert backend.success is None
    assert backend.steps_used == 0
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.execute(RobotAction(kind="continuous", arguments={"values": [0.1]}))


def test_remote_backend_closes_transport_even_when_remote_cleanup_fails() -> None:
    class _FailingCloseTransport(_Transport):
        def call(self, method: str, **kwargs: Any) -> Any:
            if method == "robot.close":
                raise TimeoutError("remote close timed out")
            return super().call(method, **kwargs)

    transport = _FailingCloseTransport()
    backend = RemoteBackend(transport)
    backend.reset(_task())

    with pytest.raises(TimeoutError, match="timed out"):
        backend.close()

    assert transport.closed is True
    backend.close()


def test_remote_backend_rejects_post_terminal_execution() -> None:
    transport = _Transport()
    transport.transitions = [_transition(1, terminal=True)]
    backend = RemoteBackend(transport)
    backend.reset(_task())
    backend.execute(RobotAction(kind="continuous", arguments={"values": [0.1]}))

    with pytest.raises(RuntimeError, match="terminal"):
        backend.execute(RobotAction(kind="continuous", arguments={"values": [0.2]}))


def test_remote_execute_failure_latches_unknown_state_until_reset() -> None:
    class _LostResponseTransport(_Transport):
        fail_execute = True

        def call(self, method: str, **kwargs: Any) -> Any:
            if method == "robot.execute" and self.fail_execute:
                self.fail_execute = False
                raise TimeoutError("response lost after remote step")
            return super().call(method, **kwargs)

    transport = _LostResponseTransport()
    backend = RemoteBackend(transport)
    backend.reset(_task())

    with pytest.raises(TimeoutError, match="response lost"):
        backend.execute(RobotAction(kind="continuous", arguments={"values": [0.1]}))

    assert backend.steps_used == 0
    with pytest.raises(RuntimeError, match="state is unknown"):
        backend.observe()
    with pytest.raises(RuntimeError, match="state is unknown"):
        backend.execute(RobotAction(kind="continuous", arguments={"values": [0.2]}))

    backend.reset(_task())
    transition = backend.execute(
        RobotAction(kind="backend-specific", arguments={"opaque": {"controller": "remote"}})
    )
    assert transition.steps_used == 3


class _HttpResponse(AbstractContextManager["_HttpResponse"]):
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self.payload

    def __exit__(self, *_args: object) -> None:
        return None


def test_http_transport_matches_canonical_request_and_response_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[Request, float]] = []

    def urlopen(request: Request, *, timeout: float) -> _HttpResponse:
        requests.append((request, timeout))
        return _HttpResponse({"ok": True, "result": {"worker": "a"}})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    transport = HttpRobotRpcTransport("127.0.0.1:19072", default_timeout_s=20)

    result = transport.call(
        "robot.execute",
        args=([0.1, -0.2],),
        kwargs={"render": False},
        timeout_s=12,
    )

    request, timeout = requests[0]
    assert request.full_url == "http://127.0.0.1:19072/call"
    assert request.method == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data or b"") == {
        "method": "robot.execute",
        "args": [[0.1, -0.2]],
        "kwargs": {"render": False},
    }
    assert timeout == 12
    assert result == {"worker": "a"}


def test_http_transport_round_trips_numpy_wire_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    requests: list[Request] = []
    response_array = np.asarray([[5, 6]], dtype=np.int16)

    def urlopen(request: Request, *, timeout: float) -> _HttpResponse:
        del timeout
        requests.append(request)
        return _HttpResponse(
            {
                "ok": True,
                "result": {
                    "array": {
                        "__ndarray__": base64.b64encode(response_array.tobytes()).decode(),
                        "dtype": str(response_array.dtype),
                        "shape": list(response_array.shape),
                    }
                },
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    transport = HttpRobotRpcTransport("http://127.0.0.1:19072")
    request_array = np.asarray([[1.0, 2.0]], dtype=np.float32)

    result = transport.call(
        "robot.execute",
        args=(request_array, np.int64(3), np.float32(4.5), np.bool_(True)),
    )

    payload = json.loads(requests[0].data or b"")
    assert payload["args"][0]["dtype"] == "float32"
    assert payload["args"][0]["shape"] == [1, 2]
    assert payload["args"][1:] == [3, 4.5, True]
    assert result["array"] == [[5, 6]]


def test_http_transport_preserves_remote_error_identity_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps(
        {"ok": False, "error": "the GPU fell over", "traceback": "server trace"}
    ).encode()

    def urlopen(request: Request, *, timeout: float) -> _HttpResponse:
        del timeout
        raise HTTPError(request.full_url, 502, "failure", Message(), _BytesReader(body))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    transport = HttpRobotRpcTransport("http://127.0.0.1:19072")

    with pytest.raises(RobotRpcError, match="robot.reset: HTTP 502: the GPU fell over") as captured:
        transport.call("robot.reset")

    assert captured.value.method == "robot.reset"
    assert captured.value.server_traceback == "server trace"


class _BytesReader:
    def __init__(self, value: bytes) -> None:
        self._value = value

    def read(self) -> bytes:
        return self._value

    def close(self) -> None:
        return None


def test_http_transport_rejects_calls_after_close() -> None:
    transport = HttpRobotRpcTransport("http://127.0.0.1:19072")

    transport.close()

    with pytest.raises(RuntimeError, match="closed"):
        transport.call("healthz")
