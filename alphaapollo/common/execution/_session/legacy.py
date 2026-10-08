# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Legacy stateful sandbox-session compatibility surface."""

from __future__ import annotations

from typing import Any

from alphaapollo.common.execution.tools.schemas import ToolCallRecord


class SandboxSessionError(Exception):
    """Raised on an invalid session operation (fail-loud, e.g. use after close)."""


class SandboxSession:
    """A stateful session over a sandbox backend (+ optional snapshotter).

    ``backend`` must satisfy the Backend contract (``exec(cmd) -> ToolCallRecord``,
    ``release() -> None``). ``snapshotter`` (optional) must expose
    ``snapshot(workspace) -> str`` and ``restore(workspace, ref) -> None`` (the
    ``WorkspaceSnapshotter`` shape); ``workspace`` is the host path this session's
    filesystem maps to for snapshot/restore.
    """

    def __init__(
        self,
        backend: Any,
        *,
        snapshotter: Any | None = None,
        workspace: str | None = None,
    ) -> None:
        if backend is None:
            raise SandboxSessionError("SandboxSession requires a backend")
        self._backend = backend
        self._snapshotter = snapshotter
        self._workspace = workspace
        self._closed = False
        self._command_count = 0

    # ------------------------------------------------------------------
    # command execution
    # ------------------------------------------------------------------
    def run(self, cmd: str) -> ToolCallRecord:
        """Run ``cmd`` in the session's container; state persists across calls."""
        self._ensure_open()
        self._command_count += 1
        return self._backend.exec(cmd)

    @property
    def command_count(self) -> int:
        """Number of commands run in this session (state-continuity observability)."""
        return self._command_count

    @property
    def closed(self) -> bool:
        """True once ``close`` has been called."""
        return self._closed

    # ------------------------------------------------------------------
    # snapshot / restore
    # ------------------------------------------------------------------
    def snapshot(self) -> str:
        """Snapshot the session workspace → a ref. Requires a snapshotter + workspace."""
        self._ensure_open()
        self._require_snapshotter()
        return self._snapshotter.snapshot(self._workspace)

    def restore(self, ref: str) -> None:
        """Restore the session workspace to ``ref``. Requires a snapshotter + workspace."""
        self._ensure_open()
        self._require_snapshotter()
        self._snapshotter.restore(self._workspace, ref)

    # ------------------------------------------------------------------
    # fork (placeholder — real container-state clone wired later)
    # ------------------------------------------------------------------
    def fork(self, *args: Any, **kwargs: Any) -> SandboxSession:
        """PLACEHOLDER: clone this session's container state into a child session.

        A byte-for-byte container clone is a real Podman capability (checkpoint /
        commit-based) wired during the real-container step. Deferred on purpose;
        calling it now fails loud rather than silently returning a shallow copy.
        """
        raise NotImplementedError(
            "SandboxSession.fork is a placeholder; the container-state clone is wired "
            "in the real-container step (Podman checkpoint/commit)."
        )

    # ------------------------------------------------------------------
    # teardown + context manager
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the backend. A failed release remains retryable."""
        if self._closed:
            return
        self._backend.release()
        self._closed = True

    def __enter__(self) -> SandboxSession:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # internal guards
    # ------------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed:
            raise SandboxSessionError("session is closed")

    def _require_snapshotter(self) -> None:
        if self._snapshotter is None or self._workspace is None:
            raise SandboxSessionError(
                "snapshot/restore requires the session to be constructed with a "
                "snapshotter and a workspace path"
            )
