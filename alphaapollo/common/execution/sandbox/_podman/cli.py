# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Podman container command construction and lifecycle operations."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from alphaapollo.common.execution.sandbox._podman.runner import (
    PODMAN,
    WORKSPACE,
    PodmanCliError,
    PodmanResult,
    Runner,
    _safe_emit,
    _streaming_podman_exec,
    default_runner,
)
from alphaapollo.common.execution.sandbox.base import CancellationToken, OutputSink

logger = logging.getLogger(__name__)


def _build_run_argv(
    *,
    image: str,
    name: str,
    network: bool,
    memory_bytes: int | None,
    cpus: float | None,
    cpu_seconds: int | None,
    max_open_files: int | None,
    pids_limit: int | None,
    gpus: tuple[str, ...] = (),
    read_only_root: bool = False,
    drop_all_capabilities: bool = False,
    no_new_privileges: bool = False,
    run_as_user: str | None = None,
    workspace_tmpfs_bytes: int | None = None,
    tmp_tmpfs_bytes: int | None = None,
) -> list[str]:
    """Build the ``podman run`` argv for a long-lived container.

    Emits ``-d`` (detached), a fixed ``--name``, ``--network none`` when
    ``network`` is False, and resource caps when provided. It NEVER emits a
    ``-v``/``--volume`` (no host mounts — the security boundary). The container's
    command is ``sleep infinity`` so it stays alive for subsequent ``podman exec``.
    """
    if not image or not image.strip():
        raise PodmanCliError("run_container requires a non-empty image")
    if not name or not name.strip():
        raise PodmanCliError("run_container requires a non-empty container name")
    argv: list[str] = [PODMAN, "run", "-d", "--name", name]
    if read_only_root:
        argv.append("--read-only")
    if drop_all_capabilities:
        argv += ["--cap-drop", "all"]
    if no_new_privileges:
        argv += ["--security-opt", "no-new-privileges"]
    if run_as_user is not None:
        if not run_as_user.strip():
            raise PodmanCliError("run_as_user must be non-empty when provided")
        argv += ["--user", run_as_user]
    for destination, size, mode, label in (
        (WORKSPACE, workspace_tmpfs_bytes, "0777", "workspace_tmpfs_bytes"),
        ("/tmp", tmp_tmpfs_bytes, "1777", "tmp_tmpfs_bytes"),
    ):
        if size is None:
            continue
        if size <= 0:
            raise PodmanCliError(f"{label} must be positive when provided")
        argv += [
            "--tmpfs",
            f"{destination}:rw,size={int(size)},mode={mode}",
        ]
    if not network:
        argv += ["--network", "none"]
    if memory_bytes is not None:
        if memory_bytes <= 0:
            raise PodmanCliError("memory_bytes must be positive when provided")
        argv += ["--memory", f"{int(memory_bytes)}b"]
    if cpus is not None:
        if cpus <= 0:
            raise PodmanCliError("cpus must be positive when provided")
        argv += ["--cpus", str(cpus)]
    if cpu_seconds is not None:
        if cpu_seconds <= 0:
            raise PodmanCliError("cpu_seconds must be positive when provided")
        # --ulimit cpu= is RLIMIT_CPU (total CPU seconds) — the same mechanism the
        # local backend uses; it SIGKILLs a runaway command, unlike --cpus (rate).
        argv += ["--ulimit", f"cpu={int(cpu_seconds)}"]
    if max_open_files is not None:
        if max_open_files <= 0:
            raise PodmanCliError("max_open_files must be positive when provided")
        argv += ["--ulimit", f"nofile={int(max_open_files)}"]
    if pids_limit is not None:
        if pids_limit <= 0:
            raise PodmanCliError("pids_limit must be positive when provided")
        argv += ["--pids-limit", str(int(pids_limit))]
    # GPU passthrough via CDI. The CDI spec's per-device selector (``nvidia.com/gpu=5``)
    # is not honoured by this podman/CTK combination, so we pass through ALL devices
    # and select the actual card in-container via ``CUDA_VISIBLE_DEVICES``. A non-empty
    # ``gpus`` means "enable GPU"; the values are operator-resolved selectors that the
    # predictor threads into the container environment, never a model argument.
    if gpus:
        if any(not g or not g.strip() for g in gpus):
            raise PodmanCliError("gpus must contain non-empty device selectors")
        argv += ["--device", "nvidia.com/gpu=all"]
        # Select the actual card at container start (fixed env, not per-exec),
        # so the predictor never threads CUDA_VISIBLE_DEVICES into a command.
        argv += ["--env", "CUDA_VISIBLE_DEVICES=" + ",".join(gpus)]
    # No -v / --volume: host mounts are never allowed (security boundary).  The
    # bootstrap command creates the common workspace path before keeping the
    # container alive.  ``sh`` is the POSIX baseline required by OCI images used
    # by this substrate; tool commands themselves still run through bash.
    argv += [image, "sh", "-c", f"mkdir -p {WORKSPACE} && exec sleep infinity"]
    return argv


def run_container(
    name: str,
    image: str,
    *,
    network: bool = False,
    memory_bytes: int | None = None,
    cpus: float | None = None,
    cpu_seconds: int | None = None,
    max_open_files: int | None = None,
    pids_limit: int | None = None,
    gpus: tuple[str, ...] = (),
    read_only_root: bool = False,
    drop_all_capabilities: bool = False,
    no_new_privileges: bool = False,
    run_as_user: str | None = None,
    workspace_tmpfs_bytes: int | None = None,
    tmp_tmpfs_bytes: int | None = None,
    timeout_seconds: float | None = None,
    runner: Runner = default_runner,
) -> PodmanResult:
    """Start a long-lived detached container named ``name`` from ``image``.

    No network unless ``network=True``; never mounts a host path. Returns the run
    result (its stdout is the container id on success). A non-zero exit is returned,
    not raised — the caller decides.
    """
    argv = _build_run_argv(
        image=image,
        name=name,
        network=network,
        memory_bytes=memory_bytes,
        cpus=cpus,
        cpu_seconds=cpu_seconds,
        max_open_files=max_open_files,
        pids_limit=pids_limit,
        gpus=gpus,
        read_only_root=read_only_root,
        drop_all_capabilities=drop_all_capabilities,
        no_new_privileges=no_new_privileges,
        run_as_user=run_as_user,
        workspace_tmpfs_bytes=workspace_tmpfs_bytes,
        tmp_tmpfs_bytes=tmp_tmpfs_bytes,
    )
    logger.debug("podman run: %s", argv)
    return _invoke_runner(runner, argv, timeout_seconds=timeout_seconds)


def _invoke_runner(
    runner: Runner,
    argv: Sequence[str],
    *,
    timeout_seconds: float | None = None,
) -> PodmanResult:
    """Invoke an injectable runner while preserving the one-argument test seam.

    Production uses ``default_runner`` and receives the wall-clock timeout.  Test
    doubles historically accept only ``argv``; keeping that shape avoids forcing
    every caller to change just to exercise argument construction.
    """
    if timeout_seconds is not None and runner is default_runner:
        return default_runner(argv, timeout_seconds=timeout_seconds)
    return runner(argv)


def exec_in(
    name: str,
    command: str,
    *,
    timeout_seconds: float | None = None,
    runner: Runner = default_runner,
) -> PodmanResult:
    """Run ``command`` inside container ``name`` via ``podman exec ... bash -c``.

    The command is passed as a SINGLE argv element to ``bash -c`` (no shell on the
    host side), so shell metacharacters in ``command`` run inside the container but
    cannot inject into the host podman invocation.
    """
    if not name or not name.strip():
        raise PodmanCliError("exec_in requires a non-empty container name")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise PodmanCliError("timeout_seconds must be positive when provided")
    argv = [PODMAN, "exec", "--workdir", WORKSPACE, name, "bash", "-c", command]
    logger.debug("podman exec: %s ...", name)
    if runner is default_runner:
        return _streaming_podman_exec(
            name,
            command,
            timeout_seconds=timeout_seconds,
            on_output=None,
            cancellation=None,
        )
    return runner(argv)


def exec_argv(
    name: str,
    argv: Sequence[str],
    *,
    timeout_seconds: float | None = None,
    runner: Runner = default_runner,
) -> PodmanResult:
    """Run an argument vector inside ``name`` without an intermediate shell.

    ``podman exec name -- argv...`` is passed straight to the container's entrypoint
    machinery; no ``bash -c`` wrapper is involved, so the command is never a shell
    string and no argument can be interpreted as shell metacharacters. This is the
    fixed-entrypoint primitive for tools that must not build shell commands.
    """
    if not name or not name.strip():
        raise PodmanCliError("exec_argv requires a non-empty container name")
    if not argv:
        raise PodmanCliError("exec_argv requires a non-empty argv")
    if any(not isinstance(a, str) for a in argv):
        raise PodmanCliError("exec_argv requires string arguments")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise PodmanCliError("timeout_seconds must be positive when provided")
    podman_argv = [PODMAN, "exec", "--workdir", WORKSPACE, name, *argv]
    logger.debug("podman exec argv: %s ...", name)
    return _invoke_runner(runner, podman_argv, timeout_seconds=timeout_seconds)


def exec_in_streaming(
    name: str,
    command: str,
    *,
    timeout_seconds: float | None = None,
    on_output: OutputSink | None = None,
    cancellation: CancellationToken | None = None,
    runner: Runner = default_runner,
) -> PodmanResult:
    """Run a command through the one canonical streaming Podman exec path."""
    if not name or not name.strip():
        raise PodmanCliError("exec_in_streaming requires a non-empty container name")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise PodmanCliError("timeout_seconds must be positive when provided")
    argv = [PODMAN, "exec", "--workdir", WORKSPACE, name, "bash", "-c", command]
    logger.debug("streaming podman exec: %s ...", name)
    if runner is default_runner:
        return _streaming_podman_exec(
            name,
            command,
            timeout_seconds=timeout_seconds,
            on_output=on_output,
            cancellation=cancellation,
        )
    result = runner(argv)
    _safe_emit(on_output, "stdout", result.stdout)
    _safe_emit(on_output, "stderr", result.stderr)
    return result


def remove_container(name: str, *, runner: Runner = default_runner) -> PodmanResult:
    """Force-remove container ``name`` (``podman rm -f``). Idempotent-friendly.

    ``rm -f`` on a missing container returns non-zero rather than raising; the
    result is returned so the caller can treat teardown as best-effort.
    """
    if not name or not name.strip():
        raise PodmanCliError("remove_container requires a non-empty container name")
    argv = [PODMAN, "rm", "-f", name]
    logger.debug("podman rm -f %s", name)
    return runner(argv)


def kill_container(name: str, *, runner: Runner = default_runner) -> PodmanResult:
    """Kill a container after an exec timeout so the timed-out command cannot run on."""
    if not name or not name.strip():
        raise PodmanCliError("kill_container requires a non-empty container name")
    argv = [PODMAN, "kill", name]
    logger.debug("podman kill %s", name)
    return runner(argv)


def container_exists(name: str, *, runner: Runner = default_runner) -> bool:
    """Return True iff a container named ``name`` exists (``podman container exists``)."""
    if not name or not name.strip():
        raise PodmanCliError("container_exists requires a non-empty container name")
    argv = [PODMAN, "container", "exists", name]
    return runner(argv).exit_code == 0


def inspect_container_image_digest(
    name: str,
    *,
    runner: Runner = default_runner,
) -> PodmanResult:
    """Return the immutable image digest attached to a running container."""

    if not name or not name.strip():
        raise PodmanCliError("inspect_container_image_digest requires a non-empty name")
    argv = [PODMAN, "container", "inspect", name, "--format", "{{.ImageDigest}}"]
    return runner(argv)


def copy_out(
    name: str,
    container_path: str,
    host_dest: str,
    *,
    runner: Runner = default_runner,
) -> PodmanResult:
    """Copy ``container_path`` from container ``name`` to ``host_dest`` (``podman cp``).

    This is the workspace-export primitive: after a problem finishes, the final
    ``/workspace`` state is copied OUT of the container to the host filesystem so it
    can be persisted, replayed, or fed to training. It mirrors ``docker cp``.

    The paths are passed as fixed argv elements (no shell), so this shares the
    injection-safe posture of the other podman_cli calls. A non-zero exit (e.g. a
    missing source path) is returned, not raised — the caller decides.
    """
    if not name or not name.strip():
        raise PodmanCliError("copy_out requires a non-empty container name")
    if not container_path or not container_path.strip():
        raise PodmanCliError("copy_out requires a non-empty container path")
    if not host_dest or not str(host_dest).strip():
        raise PodmanCliError("copy_out requires a non-empty host destination")
    argv = [PODMAN, "cp", f"{name}:{container_path}", str(host_dest)]
    logger.debug("podman cp %s:%s -> %s", name, container_path, host_dest)
    return runner(argv)


def copy_in(
    name: str,
    host_src: str,
    container_path: str,
    *,
    runner: Runner = default_runner,
) -> PodmanResult:
    """Copy ``host_src`` into container ``name`` at ``container_path`` (``podman cp``).

    The staging primitive: the host materializes a digest-checked artifact and
    copies it into the sandbox at a fixed relative path. It is the inverse of
    :func:`copy_out`. The caller (not this function) owns path validation and
    digest verification — this stays a thin, injection-safe argv builder.
    """
    if not name or not name.strip():
        raise PodmanCliError("copy_in requires a non-empty container name")
    if not host_src or not str(host_src).strip():
        raise PodmanCliError("copy_in requires a non-empty host source")
    if not container_path or not container_path.strip():
        raise PodmanCliError("copy_in requires a non-empty container path")
    argv = [PODMAN, "cp", str(host_src), f"{name}:{container_path}"]
    logger.debug("podman cp %s -> %s:%s", host_src, name, container_path)
    return runner(argv)
