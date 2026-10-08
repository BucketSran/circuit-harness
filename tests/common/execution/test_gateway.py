from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from alphaapollo.common.artifacts.schemas import ArtifactRef
from alphaapollo.common.environment.default.projection import format_tool_response, normalize
from alphaapollo.common.execution import (
    CancellationToken,
    ExecutionContext,
    ExecutionRuntime,
    ToolError,
    ToolGateway,
    ToolGatewayResult,
    ToolGatewaySession,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.execution.sandbox.base import OutputChunk
from alphaapollo.common.execution.tools.schemas import ToolCallRecord


class RecordingBackend:
    def __init__(self, record: ToolCallRecord) -> None:
        self.record = record
        self.commands: list[str] = []
        self.release_count = 0

    def exec(self, command: str) -> ToolCallRecord:
        self.commands.append(command)
        return self.record

    def release(self) -> None:
        self.release_count += 1

    def copy_out(self, container_path: str, host_dest: str) -> None:
        return None


class StreamingBackend(RecordingBackend):
    def __init__(self, record: ToolCallRecord) -> None:
        super().__init__(record)
        self.output_sink: Callable[[OutputChunk], None] | None = None
        self.cancellation: CancellationToken | None = None

    def exec_stream(
        self,
        command: str,
        *,
        on_output: Callable[[OutputChunk], None] | None = None,
        cancellation: CancellationToken | None = None,
        artifact_store: object | None = None,
    ) -> ToolCallRecord:
        self.commands.append(command)
        self.output_sink = on_output
        self.cancellation = cancellation
        return self.record


class RecordingManager:
    def __init__(
        self,
        backend: RecordingBackend,
        *,
        acquire_error: Exception | None = None,
    ) -> None:
        self.backend = backend
        self.acquire_error = acquire_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def acquire(self, kind: str, **kwargs: Any) -> RecordingBackend:
        self.calls.append((kind, kwargs))
        if self.acquire_error is not None:
            raise self.acquire_error
        return self.backend


def _gateway(
    backend: RecordingBackend,
    *,
    acquire_error: Exception | None = None,
) -> tuple[ToolGateway, RecordingManager]:
    manager = RecordingManager(backend, acquire_error=acquire_error)
    runtime = ExecutionRuntime(sandbox_manager=manager)  # type: ignore[arg-type]
    return ToolGateway(runtime=runtime), manager


def _python_request(
    *,
    call_id: str = "python-call",
    source: str = "python_code",
) -> ToolRequest:
    return ToolRequest(
        call_id=call_id,
        tool_id="python",
        arguments={"code": "6 * 7"},
        source=source,
    )


def _bash_request() -> ToolRequest:
    return ToolRequest(
        call_id="bash-call",
        tool_id="bash",
        arguments={"command": "printf 'unchanged\\n'"},
    )


def _context(**kwargs: object) -> ExecutionContext:
    return ExecutionContext(session_id="session", **kwargs)  # type: ignore[arg-type]


def _environment_step(
    payload: object,
    gateway: ToolGateway,
    context: ExecutionContext,
) -> ToolGatewayResult | None:
    normalized = normalize(payload)
    if normalized is None:
        return None
    if isinstance(normalized, ToolError):
        return ToolGatewayResult(outcome=normalized, record=None)
    return gateway.invoke(normalized, context)


def test_gateway_converts_attempted_record_to_response_and_retains_audit_data() -> None:
    artifact = ArtifactRef(id="sha256:artifact")
    backend = RecordingBackend(
        ToolCallRecord(
            tool_id="podman",
            stdout="42\n",
            artifacts=[artifact],
        )
    )
    gateway, _ = _gateway(backend)

    result = gateway.invoke(_python_request(), _context())

    assert result.outcome == ToolResponse(
        call_id="python-call",
        tool_id="python",
        stdout="42\n",
        artifacts=(artifact,),
    )
    assert result.record is not None
    assert result.record.tool_id == "python"
    assert result.record.args == {"code": "6 * 7"}
    assert result.record.artifacts == [artifact]


def test_gateway_keeps_existing_bash_runtime_path() -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stdout="unchanged\n"))
    gateway, manager = _gateway(backend)

    result = gateway.invoke(_bash_request(), _context())

    assert result.outcome == ToolResponse(
        call_id="bash-call",
        tool_id="bash",
        stdout="unchanged\n",
    )
    assert result.record is not None
    assert result.record.tool_id == "bash"
    assert backend.commands == ["printf 'unchanged\\n'"]
    assert manager.calls[0][1]["tool_id"] == "bash"


@pytest.mark.parametrize(
    ("exit_code", "stderr"),
    [
        (1, "SyntaxError: invalid syntax"),
        (7, "RuntimeError: failed"),
        (124, "command timed out"),
        (130, "command cancelled"),
    ],
)
def test_gateway_preserves_attempted_python_failures(
    exit_code: int,
    stderr: str,
) -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stderr=stderr, exit_code=exit_code))
    gateway, _ = _gateway(backend)

    result = gateway.invoke(_python_request(), _context())

    assert isinstance(result.outcome, ToolResponse)
    assert result.outcome.exit_code == exit_code
    assert result.outcome.stderr == stderr
    assert result.record is not None
    assert result.record.exit_code == exit_code
    assert backend.release_count == 1


def test_gateway_forwards_streaming_and_cancellation_to_python_execution() -> None:
    backend = StreamingBackend(
        ToolCallRecord(tool_id="podman", stderr="command cancelled", exit_code=130)
    )
    gateway, _ = _gateway(backend)
    token = CancellationToken()
    chunks: list[OutputChunk] = []

    result = gateway.invoke(
        _python_request(),
        _context(),
        on_output=chunks.append,
        cancellation=token,
    )

    assert isinstance(result.outcome, ToolResponse)
    assert result.outcome.exit_code == 130
    assert backend.output_sink == chunks.append
    assert backend.cancellation is token


def test_gateway_session_preserves_results_and_releases_only_on_close() -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stdout="42\n"))
    gateway, manager = _gateway(backend)
    context = _context(
        branch_id="candidate/solver",
        actor="solver",
        mode="isolated",
        workspace_snapshot_ref="workspace-solver",
    )

    with gateway.open_session(context) as session:
        assert isinstance(session, ToolGatewaySession)
        first = session.invoke(_python_request(call_id="solver-1"))
        second = session.invoke(_python_request(call_id="solver-2"), context)

        assert isinstance(first.outcome, ToolResponse)
        assert isinstance(second.outcome, ToolResponse)
        assert first.record is not None
        assert second.record is not None
        assert len(manager.calls) == 1
        assert backend.release_count == 0
        assert first.record.sandbox is not None
        assert first.record.sandbox["workspace_snapshot_ref"] == "workspace-solver"

    assert session.closed is True
    assert backend.release_count == 1


def test_gateway_session_returns_typed_error_for_context_mismatch() -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stdout="42\n"))
    gateway, manager = _gateway(backend)
    solver = _context(branch_id="solver", actor="solver")
    verifier = _context(branch_id="verifier", actor="verifier")

    with gateway.open_session(solver) as session:
        result = session.invoke(_python_request(), verifier)

    assert isinstance(result.outcome, ToolError)
    assert result.outcome.stage == "policy"
    assert result.outcome.code == "session_context_mismatch"
    assert result.record is None
    assert manager.calls == []


@pytest.mark.parametrize(
    ("tool_request", "context", "stage", "code"),
    [
        (
            ToolRequest(call_id="missing", tool_id="missing"),
            _context(),
            "catalog",
            "unknown_tool",
        ),
        (
            ToolRequest(
                call_id="read",
                tool_id="read",
                arguments={"path": "README.md", "limit": 0},
            ),
            _context(),
            "catalog",
            "invalid_arguments",
        ),
        (
            _python_request(source="direct"),
            _context(),
            "policy",
            "internal_tool_forbidden",
        ),
    ],
)
def test_gateway_pre_execution_failures_have_no_record(
    tool_request: ToolRequest,
    context: ExecutionContext,
    stage: str,
    code: str,
) -> None:
    gateway, manager = _gateway(RecordingBackend(ToolCallRecord(tool_id="podman")))

    result = gateway.invoke(tool_request, context)

    assert isinstance(result.outcome, ToolError)
    assert result.outcome.stage == stage
    assert result.outcome.code == code
    assert result.record is None
    assert manager.calls == []


def test_gateway_acquire_failure_has_no_record() -> None:
    gateway, manager = _gateway(
        RecordingBackend(ToolCallRecord(tool_id="podman")),
        acquire_error=RuntimeError("podman unavailable"),
    )

    result = gateway.invoke(_python_request(), _context())

    assert isinstance(result.outcome, ToolError)
    assert result.outcome.stage == "acquire"
    assert result.outcome.code == "sandbox_unavailable"
    assert result.record is None
    assert len(manager.calls) == 1


def test_parse_failure_bypasses_gateway_and_has_no_record() -> None:
    gateway, manager = _gateway(RecordingBackend(ToolCallRecord(tool_id="podman")))

    result = _environment_step("<python_code>broken", gateway, _context())

    assert result is not None
    assert isinstance(result.outcome, ToolError)
    assert result.outcome.stage == "parse"
    assert result.record is None
    assert manager.calls == []


def test_tagged_python_round_trip_reaches_model_response_format() -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stdout="42\n"))
    gateway, _ = _gateway(backend)

    result = _environment_step("<python_code>6 * 7</python_code>", gateway, _context())

    assert result is not None
    assert isinstance(result.outcome, ToolResponse)
    observation = format_tool_response(result.outcome)
    payload = json.loads(
        observation.removeprefix("<tool_response>\n").removesuffix("\n</tool_response>")
    )
    assert payload == {
        "ok": True,
        "call_id": result.outcome.call_id,
        "tool_id": "python",
        "stdout": "42\n",
        "stderr": "",
        "exit_code": 0,
        "artifacts": [],
    }
    assert len(backend.commands) == 1
    assert backend.commands[0].startswith("python -c ")


def test_solver_and_verifier_share_gateway_but_keep_separate_context_records() -> None:
    backend = RecordingBackend(ToolCallRecord(tool_id="podman", stdout="42\n"))
    gateway, manager = _gateway(backend)
    workflow_branch = "candidate-7"

    solver = gateway.invoke(
        _python_request(call_id="solver-call"),
        _context(
            branch_id=f"{workflow_branch}/solver",
            actor="solver",
            mode="isolated",
        ),
    )
    verifier = gateway.invoke(
        _python_request(call_id="verifier-call"),
        _context(
            branch_id=f"{workflow_branch}/verifier",
            actor="verifier",
            mode="isolated",
        ),
    )

    assert solver.record is not None
    assert verifier.record is not None
    assert solver.record.sandbox is not None
    assert verifier.record.sandbox is not None
    assert solver.record.sandbox["actor"] == "solver"
    assert solver.record.sandbox["branch_id"] == "candidate-7/solver"
    assert verifier.record.sandbox["actor"] == "verifier"
    assert verifier.record.sandbox["branch_id"] == "candidate-7/verifier"
    assert len(manager.calls) == 2
    assert backend.release_count == 2
