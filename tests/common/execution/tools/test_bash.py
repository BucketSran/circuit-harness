from __future__ import annotations

import pytest

from alphaapollo.common.execution import ArtifactStore, ToolRequest
from alphaapollo.common.execution.sandbox.base import SandboxBackend
from alphaapollo.common.execution.tools import BashTool
from alphaapollo.common.execution.tools.schemas import ToolCallRecord


class RecordingBackend:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def exec(self, command: str) -> ToolCallRecord:
        self.commands.append(command)
        return ToolCallRecord(tool_id="backend", stdout="ok")

    def release(self) -> None:
        return None

    def copy_out(self, container_path: str, host_dest: str) -> None:
        return None


def test_bash_adapter_passes_model_command_only_to_isolated_backend() -> None:
    backend = RecordingBackend()
    request = ToolRequest(
        call_id="call-bash",
        tool_id="bash",
        arguments={"command": "python -c 'print(23 * 17)'"},
    )

    result = BashTool().execute(backend, request)

    assert isinstance(backend, SandboxBackend)
    assert backend.commands == ["python -c 'print(23 * 17)'"]
    assert result.stdout == "ok"


def test_bash_adapter_rejects_a_different_tool_id() -> None:
    with pytest.raises(ValueError, match="cannot execute"):
        BashTool().execute(
            RecordingBackend(),
            ToolRequest(call_id="call-read", tool_id="read", arguments={"path": "README.md"}),
        )


def test_bash_adapter_rejects_blank_command() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        BashTool().execute(
            RecordingBackend(),
            ToolRequest(call_id="call-empty", tool_id="bash", arguments={"command": "  "}),
        )


def test_bash_adapter_keeps_pi_style_tail_and_persists_pre_truncation_output(tmp_path) -> None:
    original = "".join(f"line-{index}\n" for index in range(2_001))
    backend = RecordingBackend()
    backend.exec = lambda command: ToolCallRecord(tool_id="backend", stdout=original)  # type: ignore[method-assign]
    store = ArtifactStore(tmp_path / "artifacts")

    result = BashTool().execute(
        backend,
        ToolRequest(call_id="call-long", tool_id="bash", arguments={"command": "long-output"}),
        artifact_store=store,
    )

    assert "line-0\n" not in result.stdout
    assert result.stdout.startswith("line-1\n")
    assert "line-2000" in result.stdout
    assert "[stdout truncated: showing lines 2-2001 of 2001" in result.stdout
    assert result.stderr == ""
    assert len(result.artifacts) == 1
    assert store.get(result.artifacts[0]).decode("utf-8") == original
    assert result.cost.artifacts_produced == 1
    assert result.sandbox is not None
    truncation = result.sandbox["output_truncation"]["stdout"]
    assert truncation["truncated_by"] == "lines"
    assert truncation["total_lines"] == 2_001
    assert truncation["output_lines"] == 2_000


def test_bash_adapter_truncates_safely_without_an_artifact_store() -> None:
    backend = RecordingBackend()
    backend.exec = lambda command: ToolCallRecord(  # type: ignore[method-assign]
        tool_id="backend",
        stderr="x" * (51 * 1024),
        exit_code=7,
    )

    result = BashTool().execute(
        backend,
        ToolRequest(call_id="call-long", tool_id="bash", arguments={"command": "long-error"}),
    )

    assert result.exit_code == 7
    assert result.stderr.startswith("x")
    assert "[stderr truncated:" in result.stderr
    assert result.artifacts == []
    assert result.cost.artifacts_produced == 0
