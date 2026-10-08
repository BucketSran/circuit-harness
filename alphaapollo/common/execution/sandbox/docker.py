# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Docker CLI environment and Docker execution backend.

Legacy imports of ``LocalSubprocessBackend`` and ``SandboxManager`` remain
available from this module while their implementations live in sibling modules.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any

from alphaapollo.common.execution.sandbox.base import SandboxBackend, SandboxError
from alphaapollo.common.execution.tools.schemas import ToolCallRecord

if TYPE_CHECKING:
    from docker.models.containers import Container

Backend = SandboxBackend

_COMPATIBILITY_EXPORTS = {
    "DEFAULT_ADDRESS_SPACE_BYTES": "alphaapollo.common.execution.sandbox.local",
    "DEFAULT_CPU_SECONDS": "alphaapollo.common.execution.sandbox.local",
    "DEFAULT_MAX_OPEN_FILES": "alphaapollo.common.execution.sandbox.local",
    "DEFAULT_MAX_PROCESSES": "alphaapollo.common.execution.sandbox.local",
    "DEFAULT_TIMEOUT_SECONDS": "alphaapollo.common.execution.sandbox.local",
    "LocalSubprocessBackend": "alphaapollo.common.execution.sandbox.local",
    "_DOCKER_KINDS": "alphaapollo.common.execution.sandbox.manager",
    "_LOCAL_KINDS": "alphaapollo.common.execution.sandbox.manager",
    "_PODMAN_KINDS": "alphaapollo.common.execution.sandbox.manager",
    "SandboxManager": "alphaapollo.common.execution.sandbox.manager",
}


def __getattr__(name: str) -> Any:
    module_name = _COMPATIBILITY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


@dataclass
class FsDiff:
    """Filesystem changes produced by executing a command.

    Attributes:
        created: List of file paths that were created.
        deleted: List of file paths that were deleted.
        modified: Map from file path to {"before": str, "after": str} content diff.
    """

    created: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    modified: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class EnvResult:
    """Structured result returned by ``CLISandboxEnv.execute``."""

    stdout: str
    stderr: str
    exit_code: int
    fs_diff: FsDiff = field(default_factory=FsDiff)


@dataclass
class StateSnapshot:
    """Snapshot of the current environment state.

    ``CLISandboxEnv.get_state()`` returns this compact workspace view.

    Attributes:
        cwd: Current working directory path inside the sandbox.
        working_dir_contents: Output of `ls -la {cwd}` or equivalent listing.
    """

    cwd: str
    working_dir_contents: str


logger = logging.getLogger(__name__)


class CLISandboxEnv:
    """General-purpose Docker bash sandbox for CLI benchmark tasks.

    One container is created at construction time and reused across all
    execute() calls within the task. The container is NOT automatically
    torn down — call close() explicitly or use as a context manager.

    Args:
        image_name: Any Docker image name (e.g., 'ubuntu:22.04', 'node:20-slim',
            'python:3.11-slim', or a custom benchmark image).
        instance_id: Unique string used to name the container (e.g., a task ID or
            any slug). Does not need to follow SWE-bench conventions.
        workdir: Initial working directory inside the container. Defaults to '/'.

    General-Purpose Usage
    ---------------------
    CLISandboxEnv is not limited to SWE-bench. Any Docker image and any task can
    be plugged in. Three common patterns:

    **1. Custom docker_image** — swap the image for any environment you need::

        env = CLISandboxEnv(image_name="ubuntu:22.04", instance_id="my-task-001")
        result = env.execute("apt-get install -y jq && echo 'ready'")

    **2. check_cmd evaluation** — pass a shell command string that exits 0 on success
    as a lightweight alternative to subclassing ``check_solved()``::

        # After running the agent, evaluate success with a one-liner:
        check_result = env.execute("grep -q PASS /workspace/solution.txt")
        solved = (check_result.exit_code == 0)

    **3. Minimal "MyBench" snippet** — wrapping CLISandboxEnv for a custom benchmark
    (no subclassing required)::

        from alphaapollo.common.execution.sandbox.docker import CLISandboxEnv

        task_description = "Install numpy and print its version."
        docker_image = "python:3.11-slim"
        check_cmd = "python3 -c 'import numpy; print(numpy.__version__)'"

        with CLISandboxEnv(image_name=docker_image, instance_id="mybench-task-001") as env:
            env.execute("pip install numpy -q")
            check = env.execute(check_cmd)
            solved = (check.exit_code == 0)
            print("solved:", solved, "| output:", check.stdout.strip())

    Out-of-scope
    ------------
    GUI / desktop environments (OSWorld-style VM or QEMU images) are **not supported**.
    CLISandboxEnv is a bash-only sandbox: it communicates via exec_run() stdin/stdout
    and has no display, VNC, or accessibility tree support.
    """

    def __init__(
        self,
        image_name: str,
        instance_id: str,
        workdir: str = "/",
        network_mode: str | None = None,
    ) -> None:
        try:
            import docker
        except ImportError as exc:  # pragma: no cover - installation concern
            raise ImportError("Docker execution requires the optional docker package") from exc
        self._image_name = image_name
        self._instance_id = instance_id
        self._cwd: str = workdir
        self._shell: str = "/bin/bash"
        # ``network_mode="none"`` enforces the no-network policy required by
        # formal and executable sandboxes. ``None`` leaves Docker's default
        # bridge; SandboxManager chooses the policy for each sandbox kind.
        self._network_mode = network_mode
        self._docker = docker
        self._client = docker.from_env()
        self._container: Container | None = None
        self._start_container()

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def _start_container(self) -> None:
        """Pull image (if needed) and start long-lived container."""
        container_name = f"openclaw_{self._instance_id}"

        # Remove any pre-existing container with the same name
        try:
            old = self._client.containers.get(container_name)
            old.remove(force=True)
            logger.debug("Removed pre-existing container %s", container_name)
        except self._docker.errors.NotFound:
            pass

        logger.info("Starting container %s from image %s", container_name, self._image_name)
        run_kwargs: dict[str, Any] = {
            "image": self._image_name,
            "command": "/bin/sh",
            "tty": True,
            "stdin_open": True,
            "detach": True,
            "remove": False,
            "name": container_name,
        }
        # Enforce the selected network policy at the Docker runtime.
        if self._network_mode is not None:
            run_kwargs["network_mode"] = self._network_mode
        self._container = self._client.containers.run(**run_kwargs)
        self._shell = self._detect_shell()

    def _detect_shell(self) -> str:
        """Prefer bash when available; fall back to POSIX sh for minimal images."""
        assert self._container is not None, "Container is not running"
        if not hasattr(self._container, "exec_run"):
            return self._shell
        for shell in ("/bin/bash", "bash", "/bin/sh", "sh"):
            result = self._container.exec_run(
                cmd=["/bin/sh", "-c", f"command -v {shell} >/dev/null 2>&1"],
                stdout=False,
                stderr=False,
            )
            if result.exit_code == 0:
                return shell
        return "/bin/sh"

    def close(self) -> None:
        """Stop and remove the container. Call when the task is finished."""
        if self._container is not None:
            try:
                self._container.stop(timeout=5)
                self._container.remove()
                logger.info("Container %s stopped and removed", self._container.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing container: %s", exc)
            finally:
                self._container = None

    def __enter__(self) -> CLISandboxEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def execute(self, cmd: str) -> EnvResult:
        """Execute a bash command and return structured results.

        Working directory persists across calls via explicit cwd tracking.
        FsDiff is computed using git status/diff if the cwd is inside a git repo.

        Args:
            cmd: Bash command string to execute.

        Returns:
            EnvResult with stdout, stderr, exit_code, and fs_diff.
        """
        assert self._container is not None, "Container is not running"

        # Capture pre-execution git state for fs_diff
        pre_status = self._git_status()

        # Run the command with cwd prefix for stateful session
        full_cmd = f"cd {self._cwd} && {cmd}"
        exec_result = self._container.exec_run(
            cmd=[self._shell, "-c", full_cmd],
            stdout=True,
            stderr=True,
            demux=True,
        )

        stdout_bytes, stderr_bytes = exec_result.output
        stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
        exit_code: int = exec_result.exit_code

        # Update tracked cwd if the command changed directory successfully
        if exit_code == 0:
            self._update_cwd(cmd)

        # Compute fs_diff from git state change
        fs_diff = self._compute_fs_diff(pre_status)

        return EnvResult(stdout=stdout, stderr=stderr, exit_code=exit_code, fs_diff=fs_diff)

    def reset(self) -> None:
        """Reset the environment by discarding all uncommitted file changes.

        Runs `git reset --hard HEAD && git clean -fd` inside the repo root.
        Does NOT restart the container. Working directory is reset to '/'.
        """
        assert self._container is not None, "Container is not running"

        reset_cmd = "git reset --hard HEAD && git clean -fd"
        self._container.exec_run(
            cmd=[self._shell, "-c", reset_cmd],
            stdout=True,
            stderr=True,
        )
        self._cwd = "/"
        logger.debug("Environment reset for instance %s", self._instance_id)

    def get_state(self) -> StateSnapshot:
        """Return current working directory and its file listing.

        Returns:
            StateSnapshot with cwd and working_dir_contents (output of ls -la).
        """
        assert self._container is not None, "Container is not running"

        ls_cmd = f"ls -la {self._cwd}"
        exec_result = self._container.exec_run(
            cmd=[self._shell, "-c", ls_cmd],
            stdout=True,
            stderr=True,
            demux=True,
        )
        stdout_bytes, _ = exec_result.output
        listing = (stdout_bytes or b"").decode("utf-8", errors="replace")

        return StateSnapshot(cwd=self._cwd, working_dir_contents=listing)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_cwd(self, cmd: str) -> None:
        """If cmd starts with 'cd', query container for new cwd."""
        stripped = cmd.strip()
        if stripped.startswith("cd ") or stripped == "cd":
            # Ask container for actual pwd after the cd
            full_cmd = f"cd {self._cwd} && {cmd} && pwd"
            exec_result = self._container.exec_run(
                cmd=[self._shell, "-c", full_cmd],
                stdout=True,
                stderr=True,
                demux=True,
            )
            stdout_bytes, _ = exec_result.output
            if exec_result.exit_code == 0 and stdout_bytes:
                new_cwd = stdout_bytes.decode("utf-8", errors="replace").strip().splitlines()[-1]
                if new_cwd:
                    self._cwd = new_cwd

    def _git_status(self) -> str:
        """Run git status --porcelain in the container; return output or empty string."""
        if self._container is None:
            return ""
        exec_result = self._container.exec_run(
            cmd=[self._shell, "-c", "git status --porcelain 2>/dev/null || true"],
            stdout=True,
            stderr=False,
            demux=True,
        )
        stdout_bytes, _ = exec_result.output
        return (stdout_bytes or b"").decode("utf-8", errors="replace")

    def _compute_fs_diff(self, pre_status: str) -> FsDiff:
        """Compute FsDiff by comparing pre/post git status.

        Uses git status --porcelain output format:
          'M ' = modified, '??' = untracked (new), ' D' = deleted
        """
        post_status = self._git_status()

        def parse_status(status_output: str) -> dict[str, str]:
            result: dict[str, str] = {}
            for line in status_output.splitlines():
                if len(line) >= 3:
                    code = line[:2]
                    path = line[3:].strip()
                    result[path] = code
            return result

        pre = parse_status(pre_status)
        post = parse_status(post_status)

        created: list[str] = []
        deleted: list[str] = []
        modified: dict[str, dict[str, str]] = {}

        all_paths = set(pre.keys()) | set(post.keys())
        for path in all_paths:
            pre_code = pre.get(path, "")
            post_code = post.get(path, "")

            if post_code.startswith("??") and not pre_code.startswith("??"):
                created.append(path)
            elif " D" in post_code and " D" not in pre_code:
                deleted.append(path)
            elif post_code and post_code != pre_code and not post_code.startswith("??"):
                # Modified file — get content diff
                diff_cmd = f"git diff HEAD -- {path} 2>/dev/null || true"
                exec_result = self._container.exec_run(
                    cmd=[self._shell, "-c", diff_cmd],
                    stdout=True,
                    stderr=False,
                    demux=True,
                )
                diff_bytes, _ = exec_result.output
                diff_text = (diff_bytes or b"").decode("utf-8", errors="replace")
                modified[path] = {"before": "", "after": diff_text}

        return FsDiff(created=created, deleted=deleted, modified=modified)


# ---------------------------------------------------------------------------
# EnvResult -> ToolCallRecord adapter
# ---------------------------------------------------------------------------
def _envresult_to_record(tool_id: str, env_result: EnvResult) -> ToolCallRecord:
    """Map a CLISandboxEnv ``EnvResult`` onto the frozen ``ToolCallRecord``.

    The ``FsDiff`` dataclass is converted to the ``fs_diff`` dict via
    ``dataclasses.asdict`` (the frozen contract takes a ``dict``, not the
    dataclass).
    """
    return ToolCallRecord(
        tool_id=tool_id,
        stdout=env_result.stdout,
        stderr=env_result.stderr,
        exit_code=env_result.exit_code,
        fs_diff=asdict(env_result.fs_diff),
    )


# ---------------------------------------------------------------------------
# Docker backend
# ---------------------------------------------------------------------------
class DockerBackend:
    """Docker sandbox backend wrapping ``CLISandboxEnv`` (CLI/SWE/Lean).

    Reuses the existing long-lived-container + git-``FsDiff`` machinery; this
    class only adapts ``EnvResult`` -> ``ToolCallRecord`` and supplies a distinct
    ``instance_id`` per acquire so concurrent backends never share a container.
    """

    def __init__(
        self,
        *,
        image: str,
        instance_id: str | None = None,
        workdir: str = "/",
        tool_id: str = "docker",
        network_mode: str | None = None,
    ) -> None:
        if not image:
            raise ValueError("DockerBackend requires a non-empty image name")
        self._instance_id = instance_id or uuid.uuid4().hex[:12]
        self._tool_id = tool_id
        # Thread the selected network policy through to the container runtime.
        self._network_mode = network_mode
        self._env = CLISandboxEnv(
            image_name=image,
            instance_id=self._instance_id,
            workdir=workdir,
            network_mode=network_mode,
        )
        self._released = False
        logger.debug(
            "DockerBackend acquired; image=%s instance_id=%s",
            image,
            self._instance_id,
        )

    @property
    def instance_id(self) -> str:
        """The distinct container instance id (container = ``openclaw_{id}``)."""
        return self._instance_id

    def exec(self, cmd: str) -> ToolCallRecord:
        """Run a bash ``cmd`` in the container; map EnvResult -> ToolCallRecord."""
        if self._released:
            raise SandboxError("backend has been released")
        env_result = self._env.execute(cmd)
        return _envresult_to_record(self._tool_id, env_result)

    def copy_out(self, container_path: str, host_dest: str) -> None:
        """Copy ``container_path`` out of the container to ``host_dest`` (``docker cp``).

        The workspace-export primitive, symmetric with ``PodmanBackend.copy_out``:
        after a problem finishes the final ``/workspace`` state is copied OUT to the
        host so it can be persisted / replayed / fed to training. Fails loud on a
        non-zero ``docker cp`` — an export must never silently produce an empty dir.
        """
        if self._released:
            raise SandboxError("backend has been released")
        container_name = f"openclaw_{self._instance_id}"
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell, injection-safe
            ["docker", "cp", f"{container_name}:{container_path}", str(host_dest)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise SandboxError(
                f"failed to copy {container_path!r} out of container "
                f"{container_name!r}: exit={completed.returncode} "
                f"stderr={completed.stderr!r}"
            )

    def release(self) -> None:
        """Stop + remove the container. Idempotent; teardown never raises."""
        if self._released:
            return
        self._released = True
        try:
            self._env.close()
        except Exception as exc:  # noqa: BLE001 — teardown is best-effort
            logger.warning("Error releasing DockerBackend: %s", exc)
