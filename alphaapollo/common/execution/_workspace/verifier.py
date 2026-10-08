# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Disposable verifier workspace staging."""

from __future__ import annotations

import logging
import pathlib
import shutil
import tempfile
from typing import Any

from alphaapollo.common.execution.sandbox.base import VERIFIER_DEFAULT, SandboxProfile

logger = logging.getLogger(__name__)


class VerifierWorkspaceError(Exception):
    """Raised on an invalid verifier-workspace operation (fail-loud)."""


class VerifierWorkspace:
    """A disposable, isolated host directory for one verification run.

    Each instance owns a fresh ``mkdtemp`` root (isolation); ``materialize`` writes
    the files-to-verify into it; ``cleanup`` (also the ``with`` exit) removes the
    directory. Snapshot/restore delegate to an optional injected snapshotter.
    """

    def __init__(
        self,
        *,
        profile: SandboxProfile = VERIFIER_DEFAULT,
        snapshotter: Any | None = None,
        root: str | pathlib.Path | None = None,
    ) -> None:
        self._profile = profile
        self._snapshotter = snapshotter
        self._cleaned = False
        if root is not None:
            self._root = pathlib.Path(root)
            self._root.mkdir(parents=True, exist_ok=True)
            self._owns_root = False  # caller-provided root: do not mkdtemp
        else:
            self._root = pathlib.Path(tempfile.mkdtemp(prefix="apollo_verify_"))
            self._owns_root = True
        logger.debug("VerifierWorkspace created at %s (profile=%s)", self._root, profile.name)

    @property
    def path(self) -> pathlib.Path:
        """The isolated workspace directory the verifier runs in."""
        self._ensure_live()
        return self._root

    @property
    def profile(self) -> SandboxProfile:
        """The (strict) profile this verifier workspace runs under."""
        return self._profile

    # ------------------------------------------------------------------
    # materialize files to verify
    # ------------------------------------------------------------------
    def materialize(self, files: dict[str, str]) -> None:
        """Write ``{relative_path: content}`` into the isolated workspace.

        Relative paths are resolved under the workspace root; a path escaping the
        root (``..`` / absolute) is rejected (defensive — the verifier workspace is
        a closed box). Parent directories are created as needed.
        """
        self._ensure_live()
        for rel, content in files.items():
            if not rel or not isinstance(rel, str):
                raise VerifierWorkspaceError(f"invalid file name {rel!r}")
            target = (self._root / rel).resolve()
            root_resolved = self._root.resolve()
            if root_resolved != target and root_resolved not in target.parents:
                raise VerifierWorkspaceError(
                    f"refusing to write outside the verifier workspace: {rel!r}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        logger.debug("materialized %d file(s) into %s", len(files), self._root)

    # ------------------------------------------------------------------
    # snapshot / restore (delegated)
    # ------------------------------------------------------------------
    def snapshot(self) -> str:
        """Snapshot the workspace → a ref. Requires an injected snapshotter."""
        self._ensure_live()
        self._require_snapshotter()
        return self._snapshotter.snapshot(self._root)

    def restore(self, ref: str) -> None:
        """Restore the workspace to ``ref``. Requires an injected snapshotter."""
        self._ensure_live()
        self._require_snapshotter()
        self._snapshotter.restore(self._root, ref)

    # ------------------------------------------------------------------
    # teardown + context manager
    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """Remove the workspace directory. Idempotent; best-effort (never raises)."""
        if self._cleaned:
            return
        self._cleaned = True
        if self._owns_root:
            shutil.rmtree(self._root, ignore_errors=True)
            logger.debug("VerifierWorkspace cleaned %s", self._root)

    def __enter__(self) -> VerifierWorkspace:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cleanup()

    # ------------------------------------------------------------------
    # internal guards
    # ------------------------------------------------------------------
    def _ensure_live(self) -> None:
        if self._cleaned:
            raise VerifierWorkspaceError("verifier workspace has been cleaned up")

    def _require_snapshotter(self) -> None:
        if self._snapshotter is None:
            raise VerifierWorkspaceError(
                "snapshot/restore requires the workspace to be constructed with a snapshotter"
            )
