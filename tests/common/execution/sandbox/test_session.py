"""GPU/Podman-free tests for SandboxSession (orchestration over fake backend/snapshotter).

Fakes stand in for the backend and snapshotter, so the session's forwarding, state
continuity, snapshot/restore delegation, idempotent close, context management, and
use-after-close guards are all verified with NO Podman.
"""

from __future__ import annotations

import pytest

from alphaapollo.common.execution.session import SandboxSession, SandboxSessionError
from alphaapollo.common.execution.tools.schemas import ToolCallRecord


class _FakeBackend:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.released = 0

    def exec(self, cmd: str) -> ToolCallRecord:
        self.commands.append(cmd)
        return ToolCallRecord(tool_id="fake", stdout=f"ran:{cmd}", exit_code=0)

    def release(self) -> None:
        self.released += 1


class _FakeSnapshotter:
    def __init__(self) -> None:
        self.snapshots: list[str] = []
        self.restores: list[tuple[str, str]] = []

    def snapshot(self, workspace: str) -> str:
        ref = f"snap-{len(self.snapshots)}"
        self.snapshots.append(workspace)
        return ref

    def restore(self, workspace: str, ref: str) -> None:
        self.restores.append((workspace, ref))


# --- run / state continuity --------------------------------------------------
def test_run_forwards_to_backend_and_returns_record() -> None:
    backend = _FakeBackend()
    session = SandboxSession(backend)
    rec = session.run("echo hi")
    assert isinstance(rec, ToolCallRecord)
    assert rec.stdout == "ran:echo hi"
    assert backend.commands == ["echo hi"]


def test_multiple_runs_use_the_same_backend_state_persists() -> None:
    backend = _FakeBackend()
    session = SandboxSession(backend)
    session.run("pip install numpy")
    session.run("python -c 'import numpy'")
    assert backend.commands == ["pip install numpy", "python -c 'import numpy'"]
    assert session.command_count == 2  # same session, continuous state


# --- snapshot / restore ------------------------------------------------------
def test_snapshot_and_restore_delegate_to_snapshotter() -> None:
    backend = _FakeBackend()
    snap = _FakeSnapshotter()
    session = SandboxSession(backend, snapshotter=snap, workspace="/ws")
    ref = session.snapshot()
    assert ref == "snap-0"
    assert snap.snapshots == ["/ws"]
    session.restore(ref)
    assert snap.restores == [("/ws", "snap-0")]


def test_snapshot_without_snapshotter_fails_loud() -> None:
    session = SandboxSession(_FakeBackend())
    with pytest.raises(SandboxSessionError, match="snapshotter"):
        session.snapshot()


def test_restore_without_workspace_fails_loud() -> None:
    session = SandboxSession(_FakeBackend(), snapshotter=_FakeSnapshotter())  # no workspace
    with pytest.raises(SandboxSessionError, match="snapshotter"):
        session.restore("snap-0")


# --- fork placeholder --------------------------------------------------------
def test_fork_is_placeholder_and_fails_loud() -> None:
    session = SandboxSession(_FakeBackend())
    with pytest.raises(NotImplementedError, match="placeholder"):
        session.fork()


# --- close / context manager -------------------------------------------------
def test_close_releases_backend_and_is_idempotent() -> None:
    backend = _FakeBackend()
    session = SandboxSession(backend)
    session.close()
    session.close()  # no-op
    assert backend.released == 1


def test_context_manager_closes_on_exit() -> None:
    backend = _FakeBackend()
    with SandboxSession(backend) as session:
        session.run("echo hi")
    assert backend.released == 1
    assert session.closed is True


def test_run_after_close_raises() -> None:
    session = SandboxSession(_FakeBackend())
    session.close()
    with pytest.raises(SandboxSessionError, match="closed"):
        session.run("echo hi")


def test_snapshot_after_close_raises() -> None:
    session = SandboxSession(_FakeBackend(), snapshotter=_FakeSnapshotter(), workspace="/ws")
    session.close()
    with pytest.raises(SandboxSessionError, match="closed"):
        session.snapshot()


def test_requires_a_backend() -> None:
    with pytest.raises(SandboxSessionError, match="backend"):
        SandboxSession(None)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
