from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from alphaapollo.common.environment.default.projection import prepare_tool_request
from alphaapollo.common.execution import ExecutionContext, ExecutionRuntime, ToolError, ToolRequest
from alphaapollo.common.execution.sandbox.base import PODMAN_DEFAULT
from alphaapollo.common.execution.tools.schemas import CostProgressRecord, ToolCallRecord


class RecordingBackend:
    def __init__(
        self,
        record: ToolCallRecord | None = None,
        *,
        execute_error: Exception | None = None,
        release_error: Exception | None = None,
    ) -> None:
        self.record = record or ToolCallRecord(tool_id="podman", stdout="done")
        self.execute_error = execute_error
        self.release_error = release_error
        self.commands: list[str] = []
        self.timeouts: list[float | None] = []
        self.released = False
        self.release_count = 0

    def exec(self, command: str) -> ToolCallRecord:
        self.commands.append(command)
        if self.execute_error is not None:
            raise self.execute_error
        return self.record

    def exec_with_timeout(
        self,
        command: str,
        *,
        timeout_seconds: float | None,
    ) -> ToolCallRecord:
        self.timeouts.append(timeout_seconds)
        return self.exec(command)

    def release(self) -> None:
        self.release_count += 1
        self.released = True
        if self.release_error is not None:
            raise self.release_error


class RecordingManager:
    def __init__(
        self,
        backend: RecordingBackend | None = None,
        *,
        acquire_error: Exception | None = None,
    ) -> None:
        self.backend = backend or RecordingBackend()
        self.acquire_error = acquire_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def acquire(self, kind: str, **kwargs: Any) -> RecordingBackend:
        self.calls.append((kind, kwargs))
        if self.acquire_error is not None:
            raise self.acquire_error
        return self.backend


def _runtime(manager: RecordingManager) -> ExecutionRuntime:
    return ExecutionRuntime(sandbox_manager=manager)  # type: ignore[arg-type]


def _bash_request(*, timeout: float | None = None) -> ToolRequest:
    arguments: dict[str, object] = {"command": "python -c 'print(23 * 17)'"}
    if timeout is not None:
        arguments["timeout"] = timeout
    return ToolRequest(call_id="call-bash", tool_id="bash", arguments=arguments)


def _python_request(
    *,
    call_id: str = "call-python",
    source: str = "python_code",
) -> ToolRequest:
    return ToolRequest(
        call_id=call_id,
        tool_id="python",
        arguments={"code": "6 * 7"},
        source=source,
    )


def _context(*, timeout: float | None = None, **kwargs: object) -> ExecutionContext:
    return ExecutionContext(session_id="session", timeout_s=timeout, **kwargs)  # type: ignore[arg-type]


def _assert_error(result: object, stage: str, code: str) -> ToolError:
    assert isinstance(result, ToolError)
    assert result.stage == stage
    assert result.code == code
    return result


def test_runtime_runs_bash_in_rootless_podman_and_canonicalizes_record() -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stdout="391\n", fs_diff={"x": 1}))
    manager = RecordingManager(backend)

    result = _runtime(manager).invoke(_bash_request(), _context())

    assert isinstance(result, ToolCallRecord)
    assert result.tool_id == "bash"
    assert result.args == {"command": "python -c 'print(23 * 17)'"}
    assert result.stdout == "391\n"
    assert result.fs_diff == {"x": 1}
    assert result.sandbox == {
        "kind": "podman",
        "workspace_root": "/workspace",
        "network": False,
        "profile": "podman_default",
        "image": "python:3.11-slim",
        "resource_limits": {
            "cpu_seconds": 30,
            "memory_bytes": 2 * 1024 * 1024 * 1024,
            "max_processes": 64,
            "max_open_files": 4096,
            "cpus": None,
        },
        "effective_timeout_s": 30,
        "session_id": "session",
        "branch_id": "main",
        "actor": "solver",
        "mode": "default",
    }
    assert result.cost.tool_calls == 1
    assert backend.commands == ["python -c 'print(23 * 17)'"]
    assert backend.released is True
    assert manager.calls[0][0] == "podman"
    assert manager.calls[0][1]["network"] is False
    assert manager.calls[0][1]["profile"].timeout_seconds == 30


def test_runtime_honours_an_injected_no_network_podman_profile() -> None:
    manager = RecordingManager()
    profile = PODMAN_DEFAULT.with_overrides(
        name="scientific_task",
        image="scientific-task:test",
        timeout_seconds=45,
    )
    runtime = ExecutionRuntime(
        sandbox_manager=manager,  # type: ignore[arg-type]
        sandbox_profile=profile,
    )

    result = runtime.invoke(_python_request(), _context())

    assert isinstance(result, ToolCallRecord)
    applied = manager.calls[0][1]["profile"]
    assert applied.image == "scientific-task:test"
    assert applied.timeout_seconds == 45
    assert manager.calls[0][1]["network"] is False
    assert result.sandbox is not None
    assert result.sandbox["profile"] == "scientific_task"
    assert result.sandbox["image"] == "scientific-task:test"
    assert result.sandbox["resource_limits"] == {
        "cpu_seconds": 30,
        "memory_bytes": 2 * 1024 * 1024 * 1024,
        "max_processes": 64,
        "max_open_files": 4096,
        "cpus": None,
    }


@pytest.mark.parametrize(
    "profile",
    [
        PODMAN_DEFAULT.with_overrides(name="networked", network=True),
        PODMAN_DEFAULT.with_overrides(name="mounted", allow_host_mounts=True),
        PODMAN_DEFAULT.with_overrides(name="unbounded", timeout_seconds=None),
    ],
)
def test_runtime_refuses_profiles_that_weaken_isolation(profile: object) -> None:
    with pytest.raises(ValueError, match="must disable|must enforce a finite timeout"):
        ExecutionRuntime(sandbox_profile=profile)  # type: ignore[arg-type]


def test_runtime_dispatches_tagged_python_through_python_tool() -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stdout="42\n"))
    manager = RecordingManager(backend)

    result = _runtime(manager).invoke(_python_request(), _context())

    assert isinstance(result, ToolCallRecord)
    assert result.tool_id == "python"
    assert result.args == {"code": "6 * 7"}
    assert result.stdout == "42\n"
    assert len(backend.commands) == 1
    assert backend.commands[0].startswith("python -c ")
    assert manager.calls[0][1]["tool_id"] == "python"
    assert backend.released is True


@pytest.mark.parametrize(
    ("tool_id", "arguments"),
    [
        ("read", {"path": "README.md"}),
        ("write", {"path": "notes.txt", "content": "hello"}),
        (
            "edit",
            {
                "path": "notes.txt",
                "edits": [{"oldText": "hello", "newText": "goodbye"}],
            },
        ),
        ("grep", {"pattern": "ToolGateway", "path": "."}),
        ("find", {"pattern": "**/*.py", "path": "."}),
        ("ls", {"path": "."}),
    ],
)
def test_runtime_dispatches_public_workspace_adapters(
    tool_id: str,
    arguments: dict[str, object],
) -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stdout="done"))
    manager = RecordingManager(backend)
    request = ToolRequest(call_id=f"call-{tool_id}", tool_id=tool_id, arguments=arguments)

    result = _runtime(manager).invoke(request, _context())

    assert isinstance(result, ToolCallRecord)
    assert result.tool_id == tool_id
    assert result.args == arguments
    assert len(backend.commands) == 1
    assert backend.commands[0].startswith("python -c ")
    assert manager.calls[0][1]["tool_id"] == tool_id
    assert backend.released is True


@pytest.mark.parametrize("source", ["direct", "openai_tool_call"])
def test_runtime_rejects_non_tag_internal_python_before_acquire(source: str) -> None:
    manager = RecordingManager()

    result = _runtime(manager).invoke(_python_request(source=source), _context())

    _assert_error(result, "policy", "internal_tool_forbidden")
    assert manager.calls == []


def test_runtime_uses_smaller_system_or_tool_timeout_in_podman_profile() -> None:
    manager = RecordingManager()

    result = _runtime(manager).invoke(_bash_request(timeout=30), _context(timeout=5))

    assert isinstance(result, ToolCallRecord)
    assert result.sandbox is not None
    assert result.sandbox["effective_timeout_s"] == 5
    assert manager.calls[0][1]["profile"].timeout_seconds == 5


def test_nonzero_exit_is_an_attempted_tool_record_not_a_pre_execution_error() -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stderr="failed", exit_code=7))

    result = _runtime(RecordingManager(backend)).invoke(_bash_request(), _context())

    assert isinstance(result, ToolCallRecord)
    assert result.exit_code == 7
    assert result.stderr == "failed"
    assert backend.released is True


def test_timeout_record_is_preserved_and_sandbox_is_released() -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stderr="timed out", exit_code=124))

    result = _runtime(RecordingManager(backend)).invoke(_bash_request(timeout=1), _context())

    assert isinstance(result, ToolCallRecord)
    assert result.exit_code == 124
    assert backend.released is True


def test_runtime_session_reuses_one_backend_until_idempotent_close() -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stdout="42\n"))
    manager = RecordingManager(backend)
    context = _context(
        branch_id="candidate/solver",
        actor="solver",
        mode="isolated",
        workspace_snapshot_ref="workspace-solver",
    )
    session = _runtime(manager).open_session(context)

    first = session.invoke(_python_request(call_id="python-1"))
    second = session.invoke(_python_request(call_id="python-2"), context)

    assert isinstance(first, ToolCallRecord)
    assert isinstance(second, ToolCallRecord)
    assert len(manager.calls) == 1
    assert len(backend.commands) == 2
    assert backend.release_count == 0
    assert first.sandbox is not None
    assert second.sandbox is not None
    assert first.sandbox["workspace_snapshot_ref"] == "workspace-solver"
    assert second.sandbox["workspace_snapshot_ref"] == "workspace-solver"

    session.close()
    session.close()

    assert session.closed is True
    assert backend.release_count == 1


def test_runtime_session_applies_per_call_timeouts_without_reacquiring() -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stdout="42\n"))
    manager = RecordingManager(backend)
    context = _context(timeout=10)
    session = _runtime(manager).open_session(context)

    first = session.invoke(_bash_request(timeout=1))
    second = session.invoke(_python_request(call_id="python-after-bash"))

    assert isinstance(first, ToolCallRecord)
    assert isinstance(second, ToolCallRecord)
    assert len(manager.calls) == 1
    assert manager.calls[0][1]["profile"].timeout_seconds == 10
    assert backend.timeouts == [1, 10]
    session.close()


def test_runtime_session_rejects_cross_actor_context_before_acquire() -> None:
    manager = RecordingManager()
    solver = _context(
        branch_id="candidate/solver",
        actor="solver",
        workspace_snapshot_ref="workspace-solver",
    )
    verifier = _context(
        branch_id="candidate/verifier",
        actor="verifier",
        workspace_snapshot_ref="workspace-verifier",
    )
    session = _runtime(manager).open_session(solver)

    result = session.invoke(_python_request(), verifier)

    _assert_error(result, "policy", "session_context_mismatch")
    assert manager.calls == []
    session.close()


def test_runtime_session_cleanup_failure_is_retryable() -> None:
    backend = RecordingBackend(release_error=RuntimeError("cleanup failed"))
    session = _runtime(RecordingManager(backend)).open_session(_context())
    result = session.invoke(_python_request())
    assert isinstance(result, ToolCallRecord)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        session.close()

    assert session.closed is False
    backend.release_error = None
    session.close()

    assert session.closed is True
    assert backend.release_count == 2


def test_runtime_session_rejects_use_after_close() -> None:
    session = _runtime(RecordingManager()).open_session(_context())
    session.close()

    with pytest.raises(RuntimeError, match="closed"):
        session.invoke(_python_request())


def test_acquisition_failure_is_a_typed_pre_execution_error() -> None:
    manager = RecordingManager(acquire_error=RuntimeError("podman unavailable"))

    error = _assert_error(
        _runtime(manager).invoke(_bash_request(), _context()),
        "acquire",
        "sandbox_unavailable",
    )

    assert error.call_id == "call-bash"
    assert manager.calls[0][0] == "podman"


def test_execution_and_cleanup_failures_remain_tool_records() -> None:
    execution = RecordingBackend(execute_error=RuntimeError("backend error"))
    execution_result = _runtime(RecordingManager(execution)).invoke(_bash_request(), _context())

    assert isinstance(execution_result, ToolCallRecord)
    assert execution_result.exit_code == -1
    assert execution.released is True

    cleanup = RecordingBackend(release_error=RuntimeError("cleanup error"))
    cleanup_result = _runtime(RecordingManager(cleanup)).invoke(_bash_request(), _context())

    assert isinstance(cleanup_result, ToolCallRecord)
    assert cleanup_result.exit_code == -1
    assert "sandbox cleanup failed" in cleanup_result.stderr


def test_runtime_owns_persistent_args_and_does_not_double_count_backend_cost() -> None:
    request = ToolRequest(
        call_id="call-owned",
        tool_id="bash",
        arguments={
            "command": "true",
            "metadata": {"nested": ["before"]},
        },
    )
    backend = RecordingBackend(
        ToolCallRecord(
            tool_id="podman",
            fs_diff={"modified": ["before.txt"]},
            cost=CostProgressRecord(tool_calls=1, cpu_s=0.5),
        )
    )

    result = _runtime(RecordingManager(backend)).invoke(request, _context())
    request.arguments["metadata"]["nested"][0] = "after"
    backend.record.fs_diff["modified"][0] = "after.txt"

    assert isinstance(result, ToolCallRecord)
    assert result.args["metadata"]["nested"] == ["before"]
    assert result.fs_diff == {"modified": ["before.txt"]}
    assert result.cost.tool_calls == 1
    assert result.cost.cpu_s == 0.5


def test_runtime_logs_safe_exception_types_without_leaking_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = RecordingManager(
        RecordingBackend(release_error=RuntimeError("secret cleanup details"))
    )

    with caplog.at_level(logging.WARNING):
        result = _runtime(manager).invoke(_bash_request(), _context())

    assert isinstance(result, ToolCallRecord)
    assert "sandbox release failed with RuntimeError" in caplog.text
    assert "secret cleanup details" not in caplog.text


@pytest.mark.parametrize(
    ("tool_request", "context", "stage", "code"),
    [
        (
            ToolRequest(call_id="unknown", tool_id="missing", arguments={}),
            _context(),
            "catalog",
            "unknown_tool",
        ),
        (
            ToolRequest(
                call_id="read",
                tool_id="read",
                arguments={"path": "README.md", "offset": 0},
            ),
            _context(),
            "catalog",
            "invalid_arguments",
        ),
        (
            ToolRequest(call_id="blank", tool_id="bash", arguments={"command": " \t"}),
            _context(),
            "catalog",
            "invalid_arguments",
        ),
        (_bash_request(), _context(actor="guest"), "policy", "actor_forbidden"),
    ],
)
def test_runtime_rechecks_m1_boundaries_before_acquire(
    tool_request: ToolRequest,
    context: ExecutionContext,
    stage: str,
    code: str,
) -> None:
    manager = RecordingManager()

    _assert_error(_runtime(manager).invoke(tool_request, context), stage, code)

    assert manager.calls == []


@dataclass
class RecordedLLM:
    response: dict[str, Any]

    def complete(self) -> dict[str, Any]:
        return self.response


def test_recorded_llm_runs_complete_single_turn_m2_vertical_slice() -> None:
    llm = RecordedLLM(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-recorded",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command":"python -c \\"print(391)\\""}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    request = prepare_tool_request(llm.complete(), _context())
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stdout="391\n"))

    assert isinstance(request, ToolRequest)
    result = _runtime(RecordingManager(backend)).invoke(request, _context())

    assert isinstance(result, ToolCallRecord)
    assert result.tool_id == "bash"
    assert result.stdout == "391\n"
