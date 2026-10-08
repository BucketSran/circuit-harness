# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Rootless Podman backend adapter and streamed-output materialization."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from alphaapollo.common.artifacts.schemas import ArtifactRef
from alphaapollo.common.execution.output import format_truncated_output
from alphaapollo.common.execution.sandbox._podman.cli import (
    copy_in,
    copy_out,
    exec_argv,
    exec_in_streaming,
    inspect_container_image_digest,
    kill_container,
    remove_container,
    run_container,
)
from alphaapollo.common.execution.sandbox._podman.runner import (
    PodmanCliError,
    PodmanResult,
    Runner,
    default_runner,
)
from alphaapollo.common.execution.sandbox.base import (
    CancellationToken,
    OutputSink,
    SandboxProfile,
)
from alphaapollo.common.execution.tools.schemas import CostProgressRecord, ToolCallRecord
from alphaapollo.common.execution.workspace import ArtifactStore

logger = logging.getLogger(__name__)


def _unlink_capture(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def _materialize_streams(
    result: PodmanResult,
    artifact_store: ArtifactStore | None,
) -> tuple[str, str, list, dict[str, object], int]:
    rendered: dict[str, str] = {}
    artifacts: list[ArtifactRef] = []
    metadata: dict[str, object] = {}
    produced = 0
    for stream, content, truncation, capture_path in (
        (
            "stdout",
            result.stdout,
            result.stdout_truncation,
            result.stdout_capture_path,
        ),
        (
            "stderr",
            result.stderr,
            result.stderr_truncation,
            result.stderr_capture_path,
        ),
    ):
        artifact_id: str | None = None
        try:
            if truncation is not None and artifact_store is not None and capture_path is not None:
                artifact = artifact_store.put_file(
                    capture_path,
                    type_=f"bash_{stream}",
                    created_by="bash",
                )
                ref = artifact_store.ref(artifact)
                artifacts.append(ref)
                artifact_id = ref.id
                produced += 1
        except Exception as exc:  # noqa: BLE001 - preserve the attempted call
            logger.warning(
                "bash %s streaming artifact persistence failed with %s",
                stream,
                type(exc).__name__,
            )
        finally:
            _unlink_capture(capture_path)

        if truncation is None:
            rendered[stream] = content
            continue
        stream_metadata = asdict(truncation)
        stream_metadata.pop("content")
        stream_metadata["artifact_id"] = artifact_id
        metadata[stream] = stream_metadata
        rendered[stream] = format_truncated_output(stream, truncation, artifact_id)
    return rendered["stdout"], rendered["stderr"], artifacts, metadata, produced


class PodmanBackendError(Exception):
    """Raised when the Podman container cannot be started/managed (fail-loud)."""


class PodmanBackend:
    """Long-lived rootless-Podman sandbox backend (mirrors DockerBackend's contract).

    Starts a container from ``profile.image`` at construction and runs commands in it
    until ``release``. Distinct ``instance_id``s give distinct container names so two
    backends never share a container (isolation).

    ``profile`` is required and must declare ``kind="podman"``: this backend is the
    only one that reads a profile, so it is where a profile targeting another
    family would otherwise be honoured as if it described this one. Both checks run
    before the container starts, so a mismatch is refused without Podman installed.
    """

    def __init__(
        self,
        *,
        profile: SandboxProfile,
        instance_id: str | None = None,
        tool_id: str = "podman",
        runner: Runner = default_runner,
    ) -> None:
        profile.require_kind("podman")
        if profile.image is None or not str(profile.image).strip():
            raise PodmanBackendError(
                f"PodmanBackend requires a profile with a container image; "
                f"profile {profile.name!r} has image=None"
            )
        self._profile = profile
        self._tool_id = tool_id
        self._runner = runner
        self._instance_id = instance_id or uuid.uuid4().hex[:12]
        self._name = f"apollo_pod_{self._instance_id}"
        self._released = False
        self._poisoned = False

        result = run_container(
            self._name,
            profile.image,
            network=profile.network,
            memory_bytes=profile.memory_bytes,
            cpus=profile.cpus,
            cpu_seconds=profile.cpu_seconds,
            max_open_files=profile.max_open_files,
            pids_limit=profile.max_processes,
            gpus=profile.gpus,
            read_only_root=profile.read_only_root,
            drop_all_capabilities=profile.drop_all_capabilities,
            no_new_privileges=profile.no_new_privileges,
            run_as_user=profile.run_as_user,
            workspace_tmpfs_bytes=profile.workspace_tmpfs_bytes,
            tmp_tmpfs_bytes=profile.tmp_tmpfs_bytes,
            timeout_seconds=profile.timeout_seconds,
            runner=runner,
        )
        if result.exit_code != 0:
            # ``podman run`` can create the named container before crun fails to
            # start it. Remove that partial resource here because construction
            # never returns an object whose ordinary ``release`` can own cleanup.
            try:
                cleanup = remove_container(self._name, runner=self._runner)
                if cleanup.exit_code != 0:
                    logger.warning(
                        "failed to clean up Podman container %s after startup failure (exit=%s)",
                        self._name,
                        cleanup.exit_code,
                    )
            except Exception as exc:  # noqa: BLE001 — preserve the startup failure
                logger.warning(
                    "error cleaning up Podman container %s after startup failure with %s",
                    self._name,
                    type(exc).__name__,
                )
            raise PodmanBackendError(
                f"failed to start Podman container {self._name!r} from image "
                f"{profile.image!r}: exit={result.exit_code} stderr={result.stderr!r}"
            )
        logger.debug("PodmanBackend started container %s (profile=%s)", self._name, profile.name)

    @property
    def instance_id(self) -> str:
        """The distinct container instance id (container = ``apollo_pod_{id}``)."""
        return self._instance_id

    @property
    def container_name(self) -> str:
        """The container name this backend manages."""
        return self._name

    @property
    def image_digest(self) -> str:
        """Resolve the immutable digest of the image this container actually uses."""

        if self._released:
            raise PodmanBackendError("backend has been released")
        result = inspect_container_image_digest(self._name, runner=self._runner)
        digest = result.stdout.strip() if result.exit_code == 0 else ""
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise PodmanBackendError(
                f"failed to resolve image digest for container {self._name!r}: "
                f"exit={result.exit_code} stderr={result.stderr!r}"
            )
        return digest

    def exec(self, cmd: str) -> ToolCallRecord:
        """Compatibility entry point; execution itself is always streaming."""
        return self.exec_stream(cmd)

    def exec_argv(self, argv: Sequence[str]) -> ToolCallRecord:
        """Run an argument vector in the container without an intermediate shell.

        Maps the ``exec_argv`` cli result through the same ``_materialize_streams``
        contract as ``exec``, so a fixed-entrypoint tool gets the same bounded
        stdout/stderr + timeout handling, but never builds a shell command.
        """
        if self._released:
            raise PodmanBackendError("backend has been released")
        if self._poisoned:
            return ToolCallRecord(
                tool_id=self._tool_id,
                exit_code=-1,
                stderr="sandbox is unavailable after a timed-out command; release it",
            )
        result = exec_argv(
            self._name,
            argv,
            timeout_seconds=self._profile.timeout_seconds,
            runner=self._runner,
        )
        stdout, stderr, artifacts, truncation, produced = _materialize_streams(
            result,
            None,
        )
        if result.timed_out or result.cancelled:
            self._poisoned = True
            try:
                kill_container(self._name, runner=self._runner)
            except Exception as exc:  # noqa: BLE001
                logger.warning("error killing timed-out Podman container %s: %s", self._name, exc)
            status = "command timed out" if result.timed_out else "command cancelled"
            return ToolCallRecord(
                tool_id=self._tool_id,
                stdout=stdout,
                stderr=f"{stderr}\n{status}" if stderr else status,
                exit_code=124 if result.timed_out else 130,
                fs_diff={},
                sandbox={"output_truncation": truncation} if truncation else None,
                artifacts=artifacts,
                cost=CostProgressRecord(artifacts_produced=produced),
            )
        return ToolCallRecord(
            tool_id=self._tool_id,
            stdout=stdout,
            stderr=stderr,
            exit_code=result.exit_code,
            fs_diff={},
            sandbox={"output_truncation": truncation} if truncation else None,
            artifacts=artifacts,
            cost=CostProgressRecord(artifacts_produced=produced),
        )

    def exec_with_timeout(
        self,
        cmd: str,
        *,
        timeout_seconds: float | None,
    ) -> ToolCallRecord:
        """Run one command with a per-call timeout in a persistent container."""
        return self.exec_stream_with_timeout(cmd, timeout_seconds=timeout_seconds)

    def exec_stream(
        self,
        cmd: str,
        *,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> ToolCallRecord:
        return self._exec_stream(
            cmd,
            timeout_seconds=self._profile.timeout_seconds,
            on_output=on_output,
            cancellation=cancellation,
            artifact_store=artifact_store,
        )

    def exec_stream_with_timeout(
        self,
        cmd: str,
        *,
        timeout_seconds: float | None,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> ToolCallRecord:
        """Stream one command with a timeout that does not mutate the profile."""
        return self._exec_stream(
            cmd,
            timeout_seconds=timeout_seconds,
            on_output=on_output,
            cancellation=cancellation,
            artifact_store=artifact_store,
        )

    def _exec_stream(
        self,
        cmd: str,
        *,
        timeout_seconds: float | None,
        on_output: OutputSink | None,
        cancellation: CancellationToken | None,
        artifact_store: ArtifactStore | None,
    ) -> ToolCallRecord:
        """Run ``cmd`` in the container; map the PodmanResult onto a ToolCallRecord.

        A tool command failure is captured into the record's ``exit_code``/``stderr``
        (never raised) — the never-crash contract. Only calling after ``release``
        raises (a programming error).
        """
        if self._released:
            raise PodmanBackendError("backend has been released")
        if self._poisoned:
            return ToolCallRecord(
                tool_id=self._tool_id,
                exit_code=-1,
                stderr="sandbox is unavailable after a timed-out command; release it",
            )
        try:
            result = exec_in_streaming(
                self._name,
                cmd,
                timeout_seconds=timeout_seconds,
                on_output=on_output,
                cancellation=cancellation,
                runner=self._runner,
            )
        except PodmanCliError:
            logger.warning("Podman exec failed with PodmanCliError")
            return ToolCallRecord(
                tool_id=self._tool_id,
                stderr="podman exec failed",
                exit_code=-1,
            )
        except Exception as exc:  # noqa: BLE001 — never-crash execution boundary
            logger.warning(
                "unexpected Podman exec failure in %s with %s",
                self._name,
                type(exc).__name__,
            )
            return ToolCallRecord(
                tool_id=self._tool_id,
                stderr="podman exec failed",
                exit_code=-1,
            )
        stdout, stderr, artifacts, truncation, produced = _materialize_streams(
            result,
            artifact_store,
        )
        if result.timed_out or result.cancelled:
            self._poisoned = True
            try:
                killed = kill_container(self._name, runner=self._runner)
                if killed.exit_code != 0:
                    logger.error(
                        "failed to kill timed-out Podman container %s: exit=%s stderr=%r",
                        self._name,
                        killed.exit_code,
                        killed.stderr,
                    )
            except Exception as exc:  # noqa: BLE001 — preserve the timeout record
                logger.warning(
                    "error killing timed-out Podman container %s with %s",
                    self._name,
                    type(exc).__name__,
                )
                message = "failed to kill sandbox"
                stderr = f"{stderr}\n{message}" if stderr else message
            if result.timed_out:
                status = f"command timed out after {timeout_seconds}s"
                exit_code = 124
            else:
                status = "command cancelled"
                exit_code = 130
            stderr = f"{stderr}\n{status}" if stderr else status
            return ToolCallRecord(
                tool_id=self._tool_id,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                fs_diff={},
                sandbox={"output_truncation": truncation} if truncation else None,
                artifacts=artifacts,
                cost=CostProgressRecord(artifacts_produced=produced),
            )
        return ToolCallRecord(
            tool_id=self._tool_id,
            stdout=stdout,
            stderr=stderr,
            exit_code=result.exit_code,
            fs_diff={},  # podman backend does not compute a git fs_diff here
            sandbox={"output_truncation": truncation} if truncation else None,
            artifacts=artifacts,
            cost=CostProgressRecord(artifacts_produced=produced),
        )

    def copy_out(self, container_path: str, host_dest: str) -> None:
        """Copy ``container_path`` from this backend's container to ``host_dest``.

        The workspace-export primitive (``podman cp``): after a problem finishes, the
        caller copies the final ``/workspace`` state OUT to the host so it can be
        persisted. Fails loud (``PodmanBackendError``) on a non-zero ``podman cp`` — an
        export must never silently produce an empty/partial directory.
        """
        if self._released:
            raise PodmanBackendError("backend has been released")
        result = copy_out(self._name, container_path, host_dest, runner=self._runner)
        if result.exit_code != 0:
            raise PodmanBackendError(
                f"failed to copy {container_path!r} out of container {self._name!r}: "
                f"exit={result.exit_code} stderr={result.stderr!r}"
            )

    def copy_in(self, host_src: str, container_path: str) -> None:
        """Copy ``host_src`` into this backend's container at ``container_path``.

        The staging primitive (``podman cp`` host -> container). The caller owns
        path validation and digest verification; this fails loud on a non-zero
        ``podman cp``, mirroring ``copy_out``.
        """
        if self._released:
            raise PodmanBackendError("backend has been released")
        result = copy_in(self._name, host_src, container_path, runner=self._runner)
        if result.exit_code != 0:
            raise PodmanBackendError(
                f"failed to copy {host_src!r} into container {self._name!r}: "
                f"exit={result.exit_code} stderr={result.stderr!r}"
            )

    def release(self) -> None:
        """Remove the container. Failed cleanup remains retryable and is observable."""
        if self._released:
            return
        try:
            # Podman 3.x waits roughly ten seconds when ``rm -f`` stops a running
            # container. Killing first keeps per-invocation teardown fast; removal
            # remains the authoritative, observable cleanup step.
            try:
                killed = kill_container(self._name, runner=self._runner)
                if killed.exit_code != 0:
                    logger.debug(
                        "Podman container %s was already stopped before removal (exit=%s)",
                        self._name,
                        killed.exit_code,
                    )
            except Exception as exc:  # noqa: BLE001 — still attempt authoritative removal
                logger.warning(
                    "error stopping Podman container %s with %s",
                    self._name,
                    type(exc).__name__,
                )
            result = remove_container(self._name, runner=self._runner)
            if result.exit_code != 0:
                raise PodmanBackendError(
                    f"failed to remove Podman container {self._name!r}: "
                    f"exit={result.exit_code} stderr={result.stderr!r}"
                )
        except Exception as exc:  # noqa: BLE001 — teardown is best-effort
            logger.warning(
                "error releasing Podman container %s with %s",
                self._name,
                type(exc).__name__,
            )
            raise
        self._released = True
