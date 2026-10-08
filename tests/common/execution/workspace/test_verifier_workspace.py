"""GPU/Podman-free tests for VerifierWorkspace (isolated, disposable verifier dir).

Pure local filesystem + temp dirs: verifies isolation, file materialization,
path-escape rejection, snapshot/restore delegation, idempotent cleanup, and context
management — with NO Podman.
"""

from __future__ import annotations

import pathlib

import pytest

from alphaapollo.common.execution.sandbox.base import VERIFIER_DEFAULT
from alphaapollo.common.execution.workspace import (
    VerifierWorkspace,
    VerifierWorkspaceError,
)


class _FakeSnapshotter:
    def __init__(self) -> None:
        self.snapshots: list = []
        self.restores: list = []

    def snapshot(self, root) -> str:
        self.snapshots.append(pathlib.Path(root))
        return f"snap-{len(self.snapshots) - 1}"

    def restore(self, root, ref: str) -> None:
        self.restores.append((pathlib.Path(root), ref))


# --- isolation + materialize -------------------------------------------------
def test_workspace_has_its_own_dir_and_uses_verifier_profile() -> None:
    ws = VerifierWorkspace()
    try:
        assert ws.path.is_dir()
        assert ws.profile is VERIFIER_DEFAULT  # strictest: no net, no host mounts
    finally:
        ws.cleanup()


def test_two_workspaces_are_isolated() -> None:
    a = VerifierWorkspace()
    b = VerifierWorkspace()
    try:
        assert a.path != b.path
    finally:
        a.cleanup()
        b.cleanup()


def test_materialize_writes_files_into_isolated_dir() -> None:
    with VerifierWorkspace() as ws:
        ws.materialize({"solution.py": "print(391)\n", "tests/test_it.py": "assert True\n"})
        assert (ws.path / "solution.py").read_text() == "print(391)\n"
        assert (ws.path / "tests" / "test_it.py").read_text() == "assert True\n"


def test_materialize_rejects_path_escape() -> None:
    with VerifierWorkspace() as ws:
        with pytest.raises(VerifierWorkspaceError, match="outside"):
            ws.materialize({"../evil.py": "x"})


def test_materialize_rejects_bad_name() -> None:
    with VerifierWorkspace() as ws:
        with pytest.raises(VerifierWorkspaceError, match="invalid file name"):
            ws.materialize({"": "x"})


# --- snapshot / restore ------------------------------------------------------
def test_snapshot_restore_delegate_to_snapshotter() -> None:
    snap = _FakeSnapshotter()
    with VerifierWorkspace(snapshotter=snap) as ws:
        ref = ws.snapshot()
        assert ref == "snap-0"
        assert snap.snapshots == [ws.path]
        ws.restore(ref)
        assert snap.restores == [(ws.path, "snap-0")]


def test_snapshot_without_snapshotter_fails_loud() -> None:
    with VerifierWorkspace() as ws:
        with pytest.raises(VerifierWorkspaceError, match="snapshotter"):
            ws.snapshot()


# --- cleanup / context manager -----------------------------------------------
def test_cleanup_removes_owned_dir_and_is_idempotent() -> None:
    ws = VerifierWorkspace()
    root = ws.path
    ws.cleanup()
    ws.cleanup()  # no-op
    assert not root.exists()


def test_context_manager_cleans_on_exit() -> None:
    with VerifierWorkspace() as ws:
        root = ws.path
        assert root.exists()
    assert not root.exists()


def test_use_after_cleanup_raises() -> None:
    ws = VerifierWorkspace()
    ws.cleanup()
    with pytest.raises(VerifierWorkspaceError, match="cleaned up"):
        _ = ws.path


def test_caller_provided_root_is_not_deleted(tmp_path: pathlib.Path) -> None:
    """A caller-provided root is used but NOT removed on cleanup (only owned dirs are)."""
    ws = VerifierWorkspace(root=tmp_path)
    ws.materialize({"a.py": "1"})
    ws.cleanup()
    assert tmp_path.exists()  # caller owns it; we don't delete it
    assert (tmp_path / "a.py").exists()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
