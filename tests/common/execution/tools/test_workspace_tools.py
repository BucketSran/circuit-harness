from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from alphaapollo.common.execution.output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES
from alphaapollo.common.execution.sandbox.base import CancellationToken
from alphaapollo.common.execution.tools import (
    EditTool,
    FindTool,
    GrepTool,
    LsTool,
    ReadTool,
    WriteTool,
)
from alphaapollo.common.execution.tools.base import ToolRequest
from alphaapollo.common.execution.tools.builtins._files.worker import _WORKSPACE_WORKER
from alphaapollo.common.execution.tools.schemas import ToolCallRecord
from alphaapollo.common.execution.workspace import ArtifactStore


def test_embedded_workspace_worker_is_valid_python() -> None:
    compile(_WORKSPACE_WORKER, "<workspace_tool>", "exec")


def _worker_constant(name: str) -> int:
    """Read one module-level integer constant out of the in-sandbox worker source.

    The worker cannot be imported here: it resolves ``WORKSPACE_ROOT`` at module
    level inside the sandbox. Its constant assignments are evaluated in order
    instead, so a value defined in terms of an earlier one still resolves.
    """

    namespace: dict[str, Any] = {}
    for node in ast.parse(_WORKSPACE_WORKER).body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        try:
            value = eval(  # noqa: S307 - repository-owned worker source
                compile(ast.Expression(node.value), "<workspace_tool>", "eval"), {}, namespace
            )
        except Exception:  # noqa: BLE001 - a non-constant assignment is not a constant
            continue
        for target in targets:
            namespace[target] = value
        if name in targets:
            assert isinstance(value, int), f"{name} must stay an integer cap"
            return value
    raise AssertionError(f"the workspace worker no longer defines {name}")


def test_worker_output_caps_match_the_host_side_accumulator() -> None:
    """The worker's head-oriented caps must fit inside the host's tail truncation.

    The worker runs inside the sandbox and cannot import
    ``execution.output``, so it restates the caps. Its own comment states the
    relation this pins: the worker reserves room below the caps for an
    actionable notice, so that a host-side ``truncate_tail`` at the same limits
    never tail-truncates a head-oriented workspace result.
    """

    assert _worker_constant("MAX_LINES") == DEFAULT_MAX_LINES
    assert _worker_constant("MAX_BYTES") == DEFAULT_MAX_BYTES
    assert _worker_constant("CONTENT_MAX_LINES") < DEFAULT_MAX_LINES
    assert _worker_constant("CONTENT_MAX_BYTES") < DEFAULT_MAX_BYTES


class LocalWorkspaceBackend:
    """Test-only command backend for the encoded in-sandbox worker."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def exec(self, command: str) -> ToolCallRecord:
        self.commands.append(command)
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            text=True,
            capture_output=True,
        )
        return ToolCallRecord(
            tool_id="local-test",
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )

    def copy_out(self, container_path: str, host_dest: str) -> None:
        shutil.copyfile(container_path, host_dest)

    def release(self) -> None:
        pass


class StaticOutputBackend:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout

    def exec(self, command: str) -> ToolCallRecord:
        return ToolCallRecord(tool_id="static", stdout=self.stdout)

    def copy_out(self, container_path: str, host_dest: str) -> None:
        raise AssertionError("workspace tools should not copy host files directly")

    def release(self) -> None:
        pass


class RecordingStreamingBackend:
    def __init__(self) -> None:
        self.cancellation: CancellationToken | None = None

    def exec(self, command: str) -> ToolCallRecord:
        raise AssertionError("streaming backend should use exec_stream")

    def exec_stream(
        self,
        command: str,
        *,
        on_output: object = None,
        cancellation: CancellationToken | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> ToolCallRecord:
        self.cancellation = cancellation
        return ToolCallRecord(tool_id="streaming", stderr="command cancelled", exit_code=130)

    def copy_out(self, container_path: str, host_dest: str) -> None:
        raise AssertionError("workspace tools should not copy host files directly")

    def release(self) -> None:
        pass


def _request(tool_id: str, **arguments: Any) -> ToolRequest:
    return ToolRequest(
        call_id=f"call-{tool_id}",
        tool_id=tool_id,
        arguments=arguments,
        source="openai_tool_call",
    )


def _execute(tool: Any, request: ToolRequest) -> ToolCallRecord:
    return tool.execute(LocalWorkspaceBackend(), request)


def _assert_tool_error(record: ToolCallRecord, code: str) -> None:
    assert record.exit_code == 2
    assert record.sandbox is not None
    assert record.sandbox["tool_error"]["code"] == code
    assert record.sandbox["tool_error"]["message"] in record.stderr
    assert "__ALPHAAPOLLO_TOOL_ERROR__" not in record.stderr


def test_write_read_and_parent_creation_round_trip(tmp_path: Path) -> None:
    write = WriteTool(_workspace_root=str(tmp_path))
    read = ReadTool(_workspace_root=str(tmp_path))

    created = _execute(
        write,
        _request("write", path="src/nested/example.txt", content="alpha\nbeta\ngamma"),
    )
    observed = _execute(
        read,
        _request("read", path="src/nested/example.txt", offset=2, limit=1),
    )
    overwritten = _execute(
        write,
        _request("write", path="src/nested/example.txt", content="replacement"),
    )

    assert created.exit_code == 0
    assert created.stdout == "Successfully wrote 16 bytes to src/nested/example.txt\n"
    assert observed.exit_code == 0
    assert observed.stdout == "beta\n\n[1 more lines in file. Use offset=3 to continue]"
    assert overwritten.exit_code == 0
    assert (tmp_path / "src/nested/example.txt").read_text() == "replacement"


def test_read_reports_head_truncation_and_offset_bounds(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("\n".join(f"line-{index}" for index in range(1, 2003)))
    read = ReadTool(_workspace_root=str(tmp_path))

    truncated = _execute(read, _request("read", path="large.txt"))
    out_of_range = _execute(read, _request("read", path="large.txt", offset=3000))

    assert truncated.exit_code == 0
    assert truncated.stdout.startswith("line-1\nline-2")
    assert "Showing lines 1-1998 of 2002" in truncated.stdout
    assert "Use offset=1999 to continue" in truncated.stdout
    _assert_tool_error(out_of_range, "offset_out_of_range")


def test_read_truncation_persists_full_selected_content(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    original = "\n".join(f"line-{index}" for index in range(1, 2003))
    path.write_text(original)
    store = ArtifactStore(tmp_path / "artifacts")
    backend = LocalWorkspaceBackend()

    record = ReadTool(_workspace_root=str(tmp_path)).execute(
        backend,
        _request("read", path="large.txt"),
        artifact_store=store,
    )

    assert record.exit_code == 0
    assert len(record.artifacts) == 1
    assert store.get(record.artifacts[0]).decode() == original
    assert record.artifacts[0].id in record.stdout
    assert "__ALPHAAPOLLO_TOOL_ARTIFACT_ID__" not in record.stdout
    assert record.cost.artifacts_produced == 1
    assert len(backend.commands) == 2


def test_read_rejects_non_utf8_files(tmp_path: Path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe\x00")

    record = _execute(
        ReadTool(_workspace_root=str(tmp_path)),
        _request("read", path="binary.bin"),
    )

    _assert_tool_error(record, "not_text_file")


def test_edit_applies_unique_disjoint_hunks_atomically(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("first = 1\nmiddle = 2\nlast = 3\n")
    tool = EditTool(_workspace_root=str(tmp_path))

    record = _execute(
        tool,
        _request(
            "edit",
            path="module.py",
            edits=[
                {"oldText": "first = 1", "newText": "first = 10"},
                {"oldText": "last = 3", "newText": "last = 30"},
            ],
        ),
    )

    assert record.exit_code == 0
    assert record.stdout == "Successfully replaced 2 block(s) in module.py.\n"
    assert path.read_text() == "first = 10\nmiddle = 2\nlast = 30\n"


def test_edit_matches_lf_input_against_bom_crlf_file(tmp_path: Path) -> None:
    path = tmp_path / "windows.txt"
    path.write_bytes("\ufefffirst\r\nsecond\r\n".encode())

    record = _execute(
        EditTool(_workspace_root=str(tmp_path)),
        _request(
            "edit",
            path="windows.txt",
            edits=[{"oldText": "first\nsecond", "newText": "one\ntwo"}],
        ),
    )

    assert record.exit_code == 0
    assert path.read_bytes() == "\ufeffone\r\ntwo\r\n".encode()


def test_edit_rejects_overlapping_occurrences_as_non_unique(tmp_path: Path) -> None:
    path = tmp_path / "overlapping.txt"
    path.write_text("aaa")

    record = _execute(
        EditTool(_workspace_root=str(tmp_path)),
        _request(
            "edit",
            path="overlapping.txt",
            edits=[{"oldText": "aa", "newText": "x"}],
        ),
    )

    _assert_tool_error(record, "edit_not_unique")
    assert path.read_text() == "aaa"


@pytest.mark.parametrize(
    ("edits", "code"),
    [
        ([{"oldText": "missing", "newText": "x"}], "edit_no_match"),
        ([{"oldText": "same", "newText": "x"}], "edit_not_unique"),
        (
            [
                {"oldText": "same target", "newText": "x"},
                {"oldText": "target", "newText": "y"},
            ],
            "edit_overlap",
        ),
    ],
)
def test_edit_failures_leave_original_file_unchanged(
    tmp_path: Path,
    edits: list[dict[str, str]],
    code: str,
) -> None:
    path = tmp_path / "edit.txt"
    original = "same target\nsame other\n"
    path.write_text(original)

    record = _execute(
        EditTool(_workspace_root=str(tmp_path)),
        _request("edit", path="edit.txt", edits=edits),
    )

    _assert_tool_error(record, code)
    assert path.read_text() == original


def test_grep_and_find_respect_gitignore_and_globs(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored/\n*.secret\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.py").write_text("before\nNeedle value\nafter\n")
    (tmp_path / ".hidden.py").write_text("Needle hidden but allowed\n")
    (tmp_path / "src/data.secret").write_text("Needle hidden\n")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored/skip.py").write_text("Needle ignored\n")

    grep = _execute(
        GrepTool(_workspace_root=str(tmp_path)),
        _request(
            "grep",
            pattern="needle",
            path=".",
            glob="**/*.py",
            ignoreCase=True,
            literal=True,
            context=1,
        ),
    )
    find = _execute(
        FindTool(_workspace_root=str(tmp_path)),
        _request("find", pattern="**/*.py", path="."),
    )

    assert grep.exit_code == 0
    assert "src/main.py-1- before" in grep.stdout
    assert "src/main.py:2: Needle value" in grep.stdout
    assert "src/main.py-3- after" in grep.stdout
    assert ".hidden.py:1: Needle hidden but allowed" in grep.stdout
    assert "ignored" not in grep.stdout
    assert "data.secret" not in grep.stdout
    assert find.exit_code == 0
    assert find.stdout == ".hidden.py\nsrc/main.py"


def test_grep_reports_invalid_regex_as_typed_attempt_failure(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("content")

    record = _execute(
        GrepTool(_workspace_root=str(tmp_path)),
        _request("grep", pattern="[", path="."),
    )

    _assert_tool_error(record, "invalid_pattern")


def test_find_uses_parent_gitignore_when_searching_nested_directory(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("src/generated/\n")
    (tmp_path / "src/generated").mkdir(parents=True)
    (tmp_path / "src/generated/ignored.py").write_text("ignored")
    (tmp_path / "src/kept.py").write_text("kept")

    record = _execute(
        FindTool(_workspace_root=str(tmp_path)),
        _request("find", pattern="**/*.py", path="src"),
    )

    assert record.exit_code == 0
    assert record.stdout == "kept.py"


def test_gitignore_anchored_rule_does_not_hide_nested_basename(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("/root-only.py\n")
    (tmp_path / "root-only.py").write_text("ignored")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/root-only.py").write_text("kept")

    record = _execute(
        FindTool(_workspace_root=str(tmp_path)),
        _request("find", pattern="**/*.py", path="."),
    )

    assert record.exit_code == 0
    assert record.stdout == "nested/root-only.py"


def test_ls_is_sorted_includes_dotfiles_and_marks_directories(tmp_path: Path) -> None:
    (tmp_path / "zeta.txt").write_text("z")
    (tmp_path / "Alpha.txt").write_text("a")
    (tmp_path / ".hidden").write_text("h")
    (tmp_path / "folder").mkdir()

    record = _execute(
        LsTool(_workspace_root=str(tmp_path)),
        _request("ls", path="."),
    )

    assert record.exit_code == 0
    assert record.stdout == ".hidden\nAlpha.txt\nfolder/\nzeta.txt"


def test_tools_reject_resolved_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)

    read = _execute(
        ReadTool(_workspace_root=str(workspace)),
        _request("read", path="escape/secret.txt"),
    )
    write = _execute(
        WriteTool(_workspace_root=str(workspace)),
        _request("write", path="escape/new.txt", content="blocked"),
    )

    _assert_tool_error(read, "path_forbidden")
    _assert_tool_error(write, "path_forbidden")
    assert not (outside / "new.txt").exists()


def test_tools_reject_resolved_symlink_escape_when_target_is_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "missing-outside"
    workspace.mkdir()
    (workspace / "escape").symlink_to(outside, target_is_directory=True)

    read = _execute(
        ReadTool(_workspace_root=str(workspace)),
        _request("read", path="escape/secret.txt"),
    )
    write = _execute(
        WriteTool(_workspace_root=str(workspace)),
        _request("write", path="escape/new.txt", content="blocked"),
    )

    _assert_tool_error(read, "path_forbidden")
    _assert_tool_error(write, "path_forbidden")
    assert not outside.exists()


def test_ls_skips_entries_whose_symlink_target_escapes_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret")
    (tmp_path / "safe.txt").write_text("safe")
    (tmp_path / "escape.txt").symlink_to(outside)

    record = _execute(
        LsTool(_workspace_root=str(tmp_path)),
        _request("ls", path="."),
    )

    assert record.exit_code == 0
    assert record.stdout == "safe.txt"


def test_workspace_adapter_preserves_large_output_as_artifact(tmp_path: Path) -> None:
    original = "".join(f"line-{index}\n" for index in range(5000))
    store = ArtifactStore(tmp_path / "artifacts")
    record = ReadTool(_workspace_root=str(tmp_path)).execute(
        StaticOutputBackend(original),  # type: ignore[arg-type]
        _request("read", path="large.txt"),
        artifact_store=store,
    )

    assert "[stdout truncated:" in record.stdout
    assert len(record.artifacts) == 1
    assert store.get(record.artifacts[0]) == original.encode()


def test_workspace_adapter_forwards_cancellation_to_streaming_backend(tmp_path: Path) -> None:
    backend = RecordingStreamingBackend()
    cancellation = CancellationToken()

    record = ReadTool(_workspace_root=str(tmp_path)).execute(
        backend,  # type: ignore[arg-type]
        _request("read", path="file.txt"),
        cancellation=cancellation,
    )

    assert record.exit_code == 130
    assert backend.cancellation is cancellation
