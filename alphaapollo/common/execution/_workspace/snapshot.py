# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deterministic workspace snapshot and restore support."""

from __future__ import annotations

import copy
import hashlib
import io
import logging
import os
import pathlib
import subprocess
import tarfile

logger = logging.getLogger(__name__)


class SnapshotError(Exception):
    """Raised when a workspace snapshot/restore operation fails (fail-loud)."""


class WorkspaceSnapshotter:
    """Snapshot/restore a workspace directory by tar/copy (local) or git (Docker).

    The tar/copy path keeps an in-memory (or on-disk, under ``store_root``) tar of
    the workspace keyed by a deterministic content hash; ``restore`` wipes the live
    tree and re-extracts the stored tar so deletions and mutations are both undone.
    """

    def __init__(self, store_root: str | pathlib.Path | None = None) -> None:
        """Args:
        store_root: optional directory to persist snapshot tarballs. When
            omitted, snapshots are held in memory (keyed by their content ref) —
            adequate for in-process round-trips and unit tests.
        """
        self._mem_store: dict[str, bytes] = {}
        self._store_root: pathlib.Path | None = (
            pathlib.Path(store_root) if store_root is not None else None
        )
        if self._store_root is not None:
            self._store_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # tar/copy path (local-subprocess workspaces)
    # ------------------------------------------------------------------
    def snapshot(self, workspace: str | pathlib.Path) -> str:
        """Snapshot ``workspace`` (a host directory) → a deterministic ref string.

        The ref is ``sha256`` over canonical, timestamp-free tar bytes, so an
        identical tree always produces an identical ref.
        """
        root = pathlib.Path(workspace)
        if not root.is_dir():
            raise SnapshotError(f"workspace {root} is not a directory")

        tar_bytes = self._make_canonical_tar(root)
        ref = hashlib.sha256(tar_bytes).hexdigest()
        self._persist(ref, tar_bytes)
        logger.debug("Snapshotted %s -> %s", root, ref)
        return ref

    def restore(self, workspace: str | pathlib.Path, ref: str) -> None:
        """Restore ``workspace`` to the state captured by ``ref``.

        Wipes the current contents of ``workspace`` and re-extracts the stored
        snapshot tar — undoing both mutations and deletions.
        """
        root = pathlib.Path(workspace)
        tar_bytes = self._load(ref)
        if tar_bytes is None:
            raise SnapshotError(f"unknown snapshot ref {ref!r}")

        root.mkdir(parents=True, exist_ok=True)
        # Clear the live tree (children only — keep the workspace dir itself).
        for child in root.iterdir():
            if child.is_dir() and not child.is_symlink():
                self._rmtree(child)
            else:
                child.unlink()

        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
            self._extract_data_tar(tar, root)
        logger.debug("Restored %s from %s", root, ref)

    # ------------------------------------------------------------------
    # git path (Docker workspaces) — reuses the CLISandboxEnv reset idiom
    # ------------------------------------------------------------------
    def snapshot_git(self, workspace: str | pathlib.Path) -> str:
        """Snapshot a git workspace → the current ``HEAD`` commit SHA (the ref).

        Mirrors the ``CLISandboxEnv`` git idiom; the returned SHA is what restore
        resets to. Uncommitted changes are NOT captured by the ref itself — the
        ref is the committed baseline, exactly like ``git reset --hard HEAD``.
        """
        root = pathlib.Path(workspace)
        try:
            out = subprocess.run(  # noqa: S603,S607 — fixed git argv, trusted host repo
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            raise SnapshotError(f"git snapshot failed for {root}: {exc}") from exc
        return out.stdout.strip()

    def restore_git(self, workspace: str | pathlib.Path, ref: str) -> None:
        """Restore a git workspace to ``ref`` via ``git reset --hard && git clean -fd``."""
        root = pathlib.Path(workspace)
        try:
            subprocess.run(  # noqa: S603,S607 — fixed git argv, trusted host repo
                ["git", "reset", "--hard", ref],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(  # noqa: S603,S607
                ["git", "clean", "-fd"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            raise SnapshotError(f"git restore failed for {root}: {exc}") from exc

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_data_tar(tar: tarfile.TarFile, root: pathlib.Path) -> None:
        """Apply PEP 706 data-filter semantics, including on Python 3.10."""

        try:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(root, filter="data")  # noqa: S202
                return

            destination = os.path.realpath(root)
            for original in tar:
                member = copy.copy(original)
                WorkspaceSnapshotter._validate_legacy_tar_member(member, destination)
                # Python 3.10 has no extraction filter argument. Every member and
                # link target has been checked against the real destination above.
                tar.extract(member, root, set_attrs=True, numeric_owner=False)  # noqa: S202
        except SnapshotError:
            raise
        except (OSError, tarfile.TarError, ValueError) as exc:
            raise SnapshotError(f"unsafe or invalid workspace snapshot: {exc}") from exc

    @staticmethod
    def _validate_legacy_tar_member(member: tarfile.TarInfo, destination: str) -> None:
        """Validate and restrict one member before Python 3.10 extracts it."""

        if not member.name or os.path.isabs(member.name):
            raise SnapshotError(f"unsafe absolute snapshot member {member.name!r}")
        target = os.path.realpath(os.path.join(destination, member.name))
        if os.path.commonpath((target, destination)) != destination:
            raise SnapshotError(f"snapshot member {member.name!r} escapes the workspace")

        if not (member.isreg() or member.isdir() or member.issym() or member.islnk()):
            raise SnapshotError(f"snapshot member {member.name!r} is a special file")

        member.mode &= 0o755
        if member.isreg() or member.islnk():
            if not member.mode & 0o100:
                member.mode &= ~0o111
            member.mode |= 0o600
        member.uid = 0
        member.gid = 0
        member.uname = ""
        member.gname = ""

        if not (member.issym() or member.islnk()):
            return
        if os.path.isabs(member.linkname):
            raise SnapshotError(f"snapshot link {member.name!r} has an absolute target")
        member.linkname = os.path.normpath(member.linkname)
        link_base = os.path.dirname(target) if member.issym() else destination
        link_target = os.path.realpath(os.path.join(link_base, member.linkname))
        if os.path.commonpath((link_target, destination)) != destination:
            raise SnapshotError(f"snapshot link {member.name!r} escapes the workspace")

    @staticmethod
    def _make_canonical_tar(root: pathlib.Path) -> bytes:
        """Build a deterministic, timestamp-free tar of ``root``'s contents.

        Members are added in sorted relative-path order with mtime/uid/gid zeroed
        so the byte output (and therefore its sha256 ref) is reproducible.
        """
        buffer = io.BytesIO()
        files = sorted(
            (p for p in root.rglob("*")),
            key=lambda p: p.relative_to(root).as_posix(),
        )
        with tarfile.open(fileobj=buffer, mode="w:") as tar:
            for path in files:
                arcname = path.relative_to(root).as_posix()
                info = tar.gettarinfo(str(path), arcname=arcname)
                # Zero out non-content metadata for determinism.
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                if info.isreg():
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
                else:
                    tar.addfile(info)
        return buffer.getvalue()

    @classmethod
    def _rmtree(cls, path: pathlib.Path) -> None:
        """Recursively remove a directory subtree (children of the workspace)."""
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                cls._rmtree(child)
            else:
                child.unlink()
        path.rmdir()

    def _persist(self, ref: str, tar_bytes: bytes) -> None:
        """Store snapshot bytes in memory and (if configured) on disk."""
        self._mem_store[ref] = tar_bytes
        if self._store_root is not None:
            (self._store_root / (f"{ref}.tar")).write_bytes(tar_bytes)

    def _load(self, ref: str) -> bytes | None:
        """Load snapshot bytes from memory or the on-disk store."""
        if ref in self._mem_store:
            return self._mem_store[ref]
        if self._store_root is not None:
            path = self._store_root / (f"{ref}.tar")
            if path.exists():
                return path.read_bytes()
        return None
