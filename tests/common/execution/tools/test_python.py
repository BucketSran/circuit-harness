from __future__ import annotations

import shlex
import subprocess
import sys

import pytest

from alphaapollo.common.execution import ToolRequest
from alphaapollo.common.execution.sandbox.base import SandboxBackend
from alphaapollo.common.execution.tools import (
    PythonCompatibilityTool,
    build_python_command,
    wrap_v2_python_code,
)
from alphaapollo.common.execution.tools.schemas import ToolCallRecord


class RecordingBackend:
    def __init__(self, result: ToolCallRecord | None = None) -> None:
        self.commands: list[str] = []
        self.result = result or ToolCallRecord(tool_id="backend", stdout="ok\n")

    def exec(self, command: str) -> ToolCallRecord:
        self.commands.append(command)
        return self.result

    def release(self) -> None:
        return None

    def copy_out(self, container_path: str, host_dest: str) -> None:
        return None


class PythonFixtureBackend(RecordingBackend):
    """Test-only backend that evaluates the generated argv with this test Python."""

    def exec(self, command: str) -> ToolCallRecord:
        self.commands.append(command)
        argv = shlex.split(command)
        assert argv[:2] == ["python", "-c"]
        completed = subprocess.run(
            [sys.executable, *argv[1:]],
            capture_output=True,
            text=True,
            check=False,
        )
        return ToolCallRecord(
            tool_id="backend",
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )


def _request(code: str, *, source: str = "python_code") -> ToolRequest:
    return ToolRequest(
        call_id="python-call",
        tool_id="python",
        arguments={"code": code},
        source=source,
    )


def test_v2_wrapper_preimports_math_and_prints_a_final_expression() -> None:
    wrapped = wrap_v2_python_code("sqrt(81)")

    assert "from math import *" in wrapped
    assert wrapped.endswith("print(sqrt(81))\n")


def test_v2_wrapper_prints_the_last_simple_assignment() -> None:
    wrapped = wrap_v2_python_code("first = 1\nanswer = first + 41")

    assert wrapped.endswith("answer = first + 41\nprint(answer)\n")


def test_v2_wrapper_ignores_assignments_in_unexecuted_nested_blocks() -> None:
    result = PythonCompatibilityTool().execute(
        PythonFixtureBackend(),
        _request("answer = 42\nif False:\n    never_bound = 0"),
    )

    assert result.exit_code == 0
    assert result.stdout == "42\n"
    assert result.stderr == ""


def test_v2_wrapper_does_not_double_print_an_explicit_print_call() -> None:
    wrapped = wrap_v2_python_code("print('already visible')")

    assert wrapped.count("print('already visible')") == 1


def test_v2_wrapper_leaves_syntax_error_for_sandbox_execution() -> None:
    wrapped = wrap_v2_python_code("if True print('broken')")

    assert wrapped.endswith("if True print('broken')\n")


@pytest.mark.parametrize("code", ["", "   "])
def test_v2_wrapper_rejects_empty_code(code: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        wrap_v2_python_code(code)


def test_python_command_contains_no_raw_model_source() -> None:
    source = "print('quote: \\' and shell: $HOME; rm -rf /not-real')"

    command = build_python_command(source)

    assert source not in command
    assert command.startswith("python -c ")
    assert shlex.split(command)[:2] == ["python", "-c"]


def test_python_adapter_passes_only_wrapped_command_to_isolated_backend() -> None:
    backend = RecordingBackend()

    result = PythonCompatibilityTool().execute(backend, _request("print(6 * 7)"))

    assert isinstance(backend, SandboxBackend)
    assert len(backend.commands) == 1
    assert backend.commands[0].startswith("python -c ")
    assert result.tool_id == "python"
    assert result.stdout == "ok\n"


def test_python_adapter_success_fixture_executes_implicit_final_print() -> None:
    result = PythonCompatibilityTool().execute(PythonFixtureBackend(), _request("6 * 7"))

    assert result.exit_code == 0
    assert result.stdout == "42\n"
    assert result.stderr == ""


def test_python_adapter_runtime_failure_fixture_is_an_attempted_record() -> None:
    result = PythonCompatibilityTool().execute(
        PythonFixtureBackend(),
        _request("raise RuntimeError('fixture failure')"),
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "RuntimeError: fixture failure" in result.stderr
    assert "<python_code>" in result.stderr


def test_python_adapter_syntax_failure_fixture_is_an_attempted_record() -> None:
    result = PythonCompatibilityTool().execute(
        PythonFixtureBackend(),
        _request("if True print('broken')"),
    )

    assert result.exit_code != 0
    assert "SyntaxError" in result.stderr
    assert "<python_code>" in result.stderr


def test_python_adapter_preserves_timeout_and_cancellation_records() -> None:
    for exit_code, message in ((124, "command timed out"), (130, "command cancelled")):
        result = PythonCompatibilityTool().execute(
            RecordingBackend(
                ToolCallRecord(tool_id="backend", stderr=message, exit_code=exit_code)
            ),
            _request("while True: pass"),
        )

        assert result.tool_id == "python"
        assert result.exit_code == exit_code
        assert result.stderr == message


def test_python_adapter_rejects_non_tag_ingress_and_other_tool_ids() -> None:
    tool = PythonCompatibilityTool()
    with pytest.raises(ValueError, match="<python_code> ingress"):
        tool.execute(RecordingBackend(), _request("print(1)", source="direct"))
    with pytest.raises(ValueError, match="cannot execute"):
        tool.execute(
            RecordingBackend(),
            ToolRequest(
                call_id="bash-call",
                tool_id="bash",
                arguments={"command": "echo wrong"},
            ),
        )
