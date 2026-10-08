"""Unit tests for the WorkspaceSnapshotter (02-02, EXEC-04).

The tar/copy path runs in the GPU/Docker-free quick suite: snapshot a directory,
mutate + delete files, restore, and assert the filesystem byte-matches the
snapshot. The git path is exercised against a throwaway local repo (git is
present in this env) and marked ``docker`` only where it needs the sandbox; here
it runs against a plain temp git repo so it stays in the quick suite.

No Docker/Ray imports at module top level (acceptance criterion).
"""

from __future__ import annotations

import io
import os
import pathlib
import subprocess
import tarfile

import pytest

from alphaapollo.common.execution.workspace import SnapshotError, WorkspaceSnapshotter


def _read_tree(root: pathlib.Path) -> dict[str, bytes]:
    """Return {relative-posix-path: bytes} for every regular file under root."""
    out: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = path.read_bytes()
    return out


# ---------------------------------------------------------------------------
# tar/copy path round-trip (the quick-suite acceptance test)
# ---------------------------------------------------------------------------
def test_snapshot_restore_round_trip(tmp_path: pathlib.Path) -> None:
    """snapshot -> add a file + delete an existing file -> restore -> byte-match."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "keep.txt").write_text("keep-me")
    (ws / "delete_me.txt").write_text("original-content")
    sub = ws / "nested"
    sub.mkdir()
    (sub / "deep.txt").write_text("deep-original")

    snapshotter = WorkspaceSnapshotter()
    before = _read_tree(ws)
    ref = snapshotter.snapshot(ws)
    assert isinstance(ref, str) and ref

    # Mutate the workspace: add a new file, delete one, modify another.
    (ws / "newfile.txt").write_text("should-disappear-on-restore")
    (ws / "delete_me.txt").unlink()
    (sub / "deep.txt").write_text("MUTATED")

    assert _read_tree(ws) != before  # sanity: we actually changed it

    snapshotter.restore(ws, ref)
    after = _read_tree(ws)
    assert after == before


def test_snapshot_ref_is_deterministic(tmp_path: pathlib.Path) -> None:
    """Identical content with DIFFERENT mtimes yields an identical ref (XCUT-04).

    Regression for #29-A1: the previous version built two same-second trees, so
    it stayed green even when ``info.mtime = 0`` was removed (mtimes already
    matched). This version snapshots, then rewinds every file's mtime far into
    the past and snapshots again — the refs match ONLY because the snapshotter
    zeroes mtime. Drop ``info.mtime = 0`` and this fails.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "x.txt").write_text("same-bytes")
    (ws / "y.txt").write_text("more")

    snapshotter = WorkspaceSnapshotter()
    ref_now = snapshotter.snapshot(ws)

    # Force a wildly different mtime on every file (2001-09-09 vs ~now).
    for p in ws.rglob("*"):
        os.utime(p, (1_000_000_000, 1_000_000_000))
    ref_past = snapshotter.snapshot(ws)

    assert ref_now == ref_past, "ref must be mtime-independent (XCUT-04)"


def test_ref_is_a_nonempty_str_snapshot_token(tmp_path: pathlib.Path) -> None:
    """The ref is a non-empty str, usable as a str | None workspace-snapshot field.

    (The former Checkpoint schema type is now Reasoning-owned runtime state and no
    longer imported here; the snapshotter's contract is just a deterministic str ref.)
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "f.txt").write_text("data")
    ref = WorkspaceSnapshotter().snapshot(ws)

    assert isinstance(ref, str) and ref


def test_restore_unknown_ref_fails_loud(tmp_path: pathlib.Path) -> None:
    """Restoring an unknown ref raises SnapshotError (fail-loud)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(SnapshotError):
        WorkspaceSnapshotter().restore(ws, "deadbeef")


def test_snapshot_nonexistent_workspace_fails_loud(tmp_path: pathlib.Path) -> None:
    """Snapshotting a non-directory raises SnapshotError."""
    with pytest.raises(SnapshotError):
        WorkspaceSnapshotter().snapshot(tmp_path / "does-not-exist")


def test_persisted_store_round_trip(tmp_path: pathlib.Path) -> None:
    """A snapshotter with a store_root restores via a fresh instance (on-disk)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("persisted")
    store = tmp_path / "snap_store"

    ref = WorkspaceSnapshotter(store_root=store).snapshot(ws)
    before = _read_tree(ws)
    (ws / "a.txt").write_text("changed")

    # A brand-new snapshotter (empty memory) must restore from the on-disk store.
    WorkspaceSnapshotter(store_root=store).restore(ws, ref)
    assert _read_tree(ws) == before


def test_python310_fallback_restores_snapshot(monkeypatch, tmp_path: pathlib.Path) -> None:
    """The validated fallback preserves restore support without PEP 706 APIs."""

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "file.txt").write_text("original")
    snapshotter = WorkspaceSnapshotter()
    ref = snapshotter.snapshot(ws)
    (ws / "file.txt").write_text("mutated")

    snapshotter.restore(ws, ref)

    assert (ws / "file.txt").read_text() == "original"


def test_python310_fallback_rejects_path_traversal(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    """The compatibility path must not weaken PEP 706 traversal protection."""

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    buffer = io.BytesIO()
    payload = b"escaped"
    with tarfile.open(fileobj=buffer, mode="w:") as tar:
        member = tarfile.TarInfo("../escaped.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))

    ws = tmp_path / "ws"
    ws.mkdir()
    snapshotter = WorkspaceSnapshotter()
    snapshotter._mem_store["malicious"] = buffer.getvalue()

    with pytest.raises(SnapshotError, match="escapes the workspace"):
        snapshotter.restore(ws, "malicious")

    assert not (tmp_path / "escaped.txt").exists()


def _link_tar(name: str, link_type: bytes, linkname: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as tar:
        member = tarfile.TarInfo(name)
        member.type = link_type
        member.linkname = linkname
        tar.addfile(member)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("link_type", "linkname", "problem"),
    [
        pytest.param(tarfile.SYMTYPE, "/etc/passwd", "absolute target", id="symlink-absolute"),
        pytest.param(tarfile.SYMTYPE, "../../outside", "escapes the workspace", id="symlink-up"),
        pytest.param(tarfile.LNKTYPE, "../outside.txt", "escapes the workspace", id="hardlink-out"),
    ],
)
def test_python310_fallback_rejects_escaping_links(
    monkeypatch,
    tmp_path: pathlib.Path,
    link_type: bytes,
    linkname: str,
    problem: str,
) -> None:
    """PEP 706's raison d'être is the link members, not just plain-file traversal."""

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    snapshotter = WorkspaceSnapshotter()
    snapshotter._mem_store["malicious"] = _link_tar("nested/link", link_type, linkname)

    with pytest.raises(SnapshotError, match=problem):
        snapshotter.restore(ws, "malicious")

    assert not (ws / "nested" / "link").exists()
    assert not (tmp_path / "outside.txt").exists()


def test_python310_fallback_restores_safe_symlinks(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    """An in-tree relative symlink survives the fallback round trip."""

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    ws = tmp_path / "ws"
    (ws / "sub").mkdir(parents=True)
    (ws / "sub" / "real.txt").write_text("linked content")
    (ws / "alias.txt").symlink_to(os.path.join("sub", "real.txt"))
    snapshotter = WorkspaceSnapshotter()
    ref = snapshotter.snapshot(ws)
    (ws / "alias.txt").unlink()
    (ws / "sub" / "real.txt").write_text("mutated")

    snapshotter.restore(ws, ref)

    assert (ws / "alias.txt").is_symlink()
    assert (ws / "alias.txt").read_text() == "linked content"


# ---------------------------------------------------------------------------
# git path round-trip (runs against a throwaway local repo — git is present)
# ---------------------------------------------------------------------------
def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def test_git_snapshot_restore_round_trip(tmp_path: pathlib.Path) -> None:
    """git path: snapshot HEAD, mutate tracked + add untracked, restore -> clean."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "tracked.txt").write_text("committed")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "init")

    snapshotter = WorkspaceSnapshotter()
    ref = snapshotter.snapshot_git(repo)
    assert isinstance(ref, str) and len(ref) >= 7

    # Mutate a tracked file and add an untracked one.
    (repo / "tracked.txt").write_text("DIRTY")
    (repo / "untracked.txt").write_text("remove-me")

    snapshotter.restore_git(repo, ref)

    assert (repo / "tracked.txt").read_text() == "committed"
    assert not (repo / "untracked.txt").exists()


# ---------------------------------------------------------------------------
# #29-C: untested fail-loud / defensive branches
# ---------------------------------------------------------------------------
def test_git_snapshot_on_non_git_dir_fails_loud(tmp_path: pathlib.Path) -> None:
    """snapshot_git on a plain (non-git) directory raises SnapshotError (#29-C)."""
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    with pytest.raises(SnapshotError):
        WorkspaceSnapshotter().snapshot_git(plain)


def test_git_restore_on_non_git_dir_fails_loud(tmp_path: pathlib.Path) -> None:
    """restore_git on a plain (non-git) directory raises SnapshotError (#29-C)."""
    plain = tmp_path / "not_a_repo2"
    plain.mkdir()
    with pytest.raises(SnapshotError):
        WorkspaceSnapshotter().restore_git(plain, "deadbeef")


def test_restore_wipes_symlink_in_subdir(tmp_path: pathlib.Path) -> None:
    """restore handles a symlink nested in a subdir on the wipe path (#29-C, L183).

    The snapshot predates the symlink, so restore must remove it. The symlink
    lives inside a subdir, so the removal goes through ``_rmtree`` — exercising
    its ``is_symlink()`` branch (unlink, never follow/recurse).
    """
    ws = tmp_path / "ws"
    (ws / "d").mkdir(parents=True)
    (ws / "d" / "f.txt").write_text("keep")
    snapshotter = WorkspaceSnapshotter()
    ref = snapshotter.snapshot(ws)

    target = tmp_path / "outside"
    target.mkdir()
    (ws / "d" / "link").symlink_to(target, target_is_directory=True)

    snapshotter.restore(ws, ref)
    assert not (ws / "d" / "link").exists()  # symlink removed, not followed
    assert (ws / "d" / "f.txt").read_text() == "keep"
    assert target.exists()  # the symlink's target was NOT recursed into / deleted


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
