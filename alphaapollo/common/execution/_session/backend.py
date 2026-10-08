# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Backend borrowing for episode-scoped execution sessions."""

from __future__ import annotations

from typing import Any, Protocol

from alphaapollo.common.execution.sandbox.base import (
    CancellationToken,
    OutputSink,
    SandboxBackend,
    SandboxProfile,
    StreamingSandboxBackend,
)
from alphaapollo.common.execution.tools.base import ToolRequest
from alphaapollo.common.execution.tools.schemas import ToolCallRecord
from alphaapollo.common.execution.workspace import ArtifactStore


class _SandboxAcquirer(Protocol):
    def acquire(
        self,
        kind: str,
        *,
        network: bool | None = None,
        **kwargs: Any,
    ) -> SandboxBackend: ...


class _ToolAdapter(Protocol):
    tool_id: str

    def validate_request(self, request: ToolRequest) -> str | None: ...

    def requested_timeout_s(self, request: ToolRequest) -> float | None: ...

    def execute(
        self,
        backend: SandboxBackend,
        request: ToolRequest,
        *,
        artifact_store: ArtifactStore | None = None,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ToolCallRecord: ...


class _BorrowedSandboxBackend:
    """Forward backend operations while leaving final release to the session."""

    def __init__(
        self,
        backend: SandboxBackend,
        *,
        timeout_seconds: float | None,
    ) -> None:
        self._backend = backend
        self._timeout_seconds = timeout_seconds

    def exec(self, cmd: str) -> ToolCallRecord:
        exec_with_timeout = getattr(self._backend, "exec_with_timeout", None)
        if callable(exec_with_timeout):
            return exec_with_timeout(cmd, timeout_seconds=self._timeout_seconds)
        return self._backend.exec(cmd)

    def copy_out(self, container_path: str, host_dest: str) -> None:
        self._backend.copy_out(container_path, host_dest)

    def release(self) -> None:
        """Return a per-call borrow; the session owns the real backend lifecycle."""


class _BorrowedStreamingSandboxBackend(_BorrowedSandboxBackend):
    def __init__(
        self,
        backend: StreamingSandboxBackend,
        *,
        timeout_seconds: float | None,
    ) -> None:
        self._streaming_backend = backend
        super().__init__(backend, timeout_seconds=timeout_seconds)  # type: ignore[arg-type]

    def exec_stream(
        self,
        cmd: str,
        *,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> ToolCallRecord:
        exec_stream_with_timeout = getattr(
            self._streaming_backend,
            "exec_stream_with_timeout",
            None,
        )
        if callable(exec_stream_with_timeout):
            return exec_stream_with_timeout(
                cmd,
                timeout_seconds=self._timeout_seconds,
                on_output=on_output,
                cancellation=cancellation,
                artifact_store=artifact_store,
            )
        return self._streaming_backend.exec_stream(
            cmd,
            on_output=on_output,
            cancellation=cancellation,
            artifact_store=artifact_store,
        )


class _SessionSandboxManager:
    """Lazily acquire one backend and lend it to every call in a session."""

    def __init__(
        self,
        delegate: _SandboxAcquirer,
        *,
        timeout_seconds: float | None,
    ) -> None:
        self._delegate = delegate
        self._timeout_seconds = timeout_seconds
        self._backend: SandboxBackend | None = None
        self._configuration: tuple[str, dict[str, Any]] | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def acquire(
        self,
        kind: str,
        *,
        network: bool | None = None,
        **kwargs: Any,
    ) -> SandboxBackend:
        if self._closed:
            raise RuntimeError("execution session is closed")

        requested_profile = kwargs.get("profile")
        execution_timeout = (
            requested_profile.timeout_seconds
            if isinstance(requested_profile, SandboxProfile)
            else None
        )
        stable_kwargs = dict(kwargs)
        if isinstance(requested_profile, SandboxProfile):
            stable_kwargs["profile"] = requested_profile.with_overrides(
                timeout_seconds=self._timeout_seconds
            )

        # ``tool_id`` and the per-call execution timeout do not alter the sandbox.
        # Runtime canonicalization restores the request tool id, while the borrowed
        # backend applies the effective timeout to only the current command.
        configuration = (
            kind,
            {
                "network": network,
                **{key: value for key, value in stable_kwargs.items() if key != "tool_id"},
            },
        )
        if self._backend is None:
            self._backend = self._delegate.acquire(
                kind,
                network=network,
                **stable_kwargs,
            )
            self._configuration = configuration
        elif configuration != self._configuration:
            raise RuntimeError("sandbox configuration changed within execution session")

        if isinstance(self._backend, StreamingSandboxBackend):
            return _BorrowedStreamingSandboxBackend(
                self._backend,
                timeout_seconds=execution_timeout,
            )
        return _BorrowedSandboxBackend(
            self._backend,
            timeout_seconds=execution_timeout,
        )

    def close(self) -> None:
        """Release the owned backend once; failed cleanup remains retryable."""
        if self._closed:
            return
        if self._backend is not None:
            self._backend.release()
        self._closed = True
