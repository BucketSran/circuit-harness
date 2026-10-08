from __future__ import annotations

from typing import Any

from alphaapollo.common.execution import ExecutionContext, ExecutionRuntime, ToolRequest
from alphaapollo.common.execution.tools import (
    ToolCallRecord,
    ToolCatalog,
    get_tool_spec,
    list_tool_specs,
)
from alphaapollo.common.execution.tools.gateway import ExecutionPolicy
from alphaapollo.common.execution.tools.python import (
    PYTHON_EXECUTE_SPEC,
    PYTHON_EXECUTE_TOOL_ID,
    PythonExecuteTool,
)


class _Backend:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def exec(self, command: str) -> ToolCallRecord:
        self.commands.append(command)
        return ToolCallRecord(tool_id="bash", stdout="42\n")

    def copy_out(self, container_path: str, host_dest: str) -> None:
        return None

    def release(self) -> None:
        return None


class _Manager:
    def __init__(self) -> None:
        self.backend = _Backend()
        self.acquisitions: list[dict[str, Any]] = []

    def acquire(self, kind: str, **kwargs: Any) -> _Backend:
        self.acquisitions.append({"kind": kind, **kwargs})
        return self.backend


def _request(*, timeout: float | None = None) -> ToolRequest:
    arguments: dict[str, Any] = {"code": "6 * 7"}
    if timeout is not None:
        arguments["timeout"] = timeout
    return ToolRequest(
        call_id="python-public",
        tool_id=PYTHON_EXECUTE_TOOL_ID,
        arguments=arguments,
        source="openai_tool_call",
    )


def test_public_python_schema_is_closed_and_legacy_python_stays_internal() -> None:
    catalog = ToolCatalog((*list_tool_specs(include_internal=True), PYTHON_EXECUTE_SPEC))

    assert catalog.resolve(_request()) is PYTHON_EXECUTE_SPEC
    assert get_tool_spec("python").model_visible is False
    assert PYTHON_EXECUTE_SPEC.parameters["additionalProperties"] is False


def test_public_python_adapter_accepts_function_call_ingress() -> None:
    backend = _Backend()

    result = PythonExecuteTool().execute(backend, _request())

    assert result.tool_id == PYTHON_EXECUTE_TOOL_ID
    assert result.stdout == "42\n"
    assert len(backend.commands) == 1
    assert backend.commands[0].startswith("python -c ")


def test_execution_runtime_can_add_public_python_without_changing_legacy_defaults() -> None:
    manager = _Manager()
    runtime = ExecutionRuntime(
        sandbox_manager=manager,  # type: ignore[arg-type]
        catalog=ToolCatalog((*list_tool_specs(include_internal=True), PYTHON_EXECUTE_SPEC)),
        additional_tools=(PythonExecuteTool(),),
    )

    result = runtime.invoke(_request(timeout=2.5), ExecutionContext(session_id="public-python"))

    assert isinstance(result, ToolCallRecord)
    assert result.tool_id == PYTHON_EXECUTE_TOOL_ID
    assert result.args == {"code": "6 * 7", "timeout": 2.5}
    assert result.stdout == "42\n"
    assert result.sandbox["network"] is False
    assert manager.acquisitions[0]["profile"].timeout_seconds == 2.5


def test_public_python_invalid_timeout_is_rejected_before_sandbox_acquisition() -> None:
    manager = _Manager()
    runtime = ExecutionRuntime(
        sandbox_manager=manager,  # type: ignore[arg-type]
        catalog=ToolCatalog((*list_tool_specs(include_internal=True), PYTHON_EXECUTE_SPEC)),
        additional_tools=(PythonExecuteTool(),),
    )
    request = ToolRequest(
        call_id="bad-timeout",
        tool_id=PYTHON_EXECUTE_TOOL_ID,
        arguments={"code": "1 + 1", "timeout": -1},
    )

    result = runtime.invoke(request, ExecutionContext(session_id="public-python"))

    assert result.code == "invalid_arguments"
    assert manager.acquisitions == []


def test_strict_policy_accepts_the_public_python_requested_timeout() -> None:
    request = _request(timeout=2.5)

    denial = ExecutionPolicy(require_timeout=True).authorize(
        request,
        ExecutionContext(session_id="strict-public-python"),
        PYTHON_EXECUTE_SPEC,
    )

    assert denial is None
