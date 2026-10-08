# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Hardened host-local subprocess execution backend."""

from __future__ import annotations

import contextlib
import logging
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile

from alphaapollo.common.execution.process_group import signal_process_group
from alphaapollo.common.execution.sandbox.base import SandboxError
from alphaapollo.common.execution.tools.schemas import ToolCallRecord

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_CPU_SECONDS = 30
DEFAULT_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
#: The default process cap for a sandbox *profile*. Only the Podman backend can
#: honour it, as ``--pids-limit`` (a per-container cgroup cap); the local backend
#: refuses the option outright rather than applying something else. See
#: :class:`LocalSubprocessBackend` for why ``RLIMIT_NPROC`` cannot express it.
DEFAULT_MAX_PROCESSES = 64
DEFAULT_MAX_OPEN_FILES = 64  # A low fd cap limits sockets without a network namespace.

# How long to wait for the pipes to close after the process group is killed.
# SIGKILL is uncatchable, so this is a backstop against an unkillable (D-state)
# member, not an expected wait.
_POST_KILL_DRAIN_SECONDS = 5.0

#: Caps this platform cannot enforce at any value a caller would ask for, so
#: requesting one is not a caller error and must not fail the run — but it must
#: not read as applied either.
#:
#: Measured on the development host (macOS 15, CPython 3.12; pinned by
#: ``test_local_rlimits.py``): Darwin aliases ``RLIMIT_AS`` to ``RLIMIT_RSS``
#: (both are resource id 5) and returns ``EINVAL`` for every value below roughly
#: 415 GiB — 64 MiB, 2 GiB and 16 GiB are all refused, while 1 TiB is accepted.
#: The accepted range begins far above installed memory (16 GiB here), so no
#: usable memory cap exists on macOS. ``getrlimit`` reports ``hard=RLIM_INFINITY``,
#: so this is invisible to a parent-side hard-limit check: it is a property of the
#: platform, not of the requested value. Linux enforces ``RLIMIT_AS`` normally and
#: is unaffected.
_UNENFORCEABLE_LIMITS: frozenset[int] = (
    frozenset({resource.RLIMIT_AS}) if sys.platform == "darwin" else frozenset()
)

#: Option names already reported as unenforceable, so the warning is emitted once
#: per process rather than once per sandbox acquisition.
_warned_unenforceable: set[str] = set()

logger = logging.getLogger(__name__)


def _build_clean_env() -> dict[str, str]:
    """Return the minimal clean environment instead of copying the host environment.

    Only ``PATH`` (so the interpreter resolves), deterministic hashing,
    user-site suppression, and a fixed locale are exported. No host secrets, no
    proxy/network env. ``LC_CTYPE`` is pinned explicitly because CPython would
    otherwise inject a host-derived ``LC_CTYPE`` at startup (locale coercion) —
    pinning it keeps the child env fully deterministic and host-independent.
    """
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "LC_CTYPE": "C.UTF-8",
    }


def _resolve_rlimits(
    cpu_seconds: int,
    address_space_bytes: int,
    max_open_files: int,
) -> tuple[tuple[int, int], ...]:
    """Return the ``(resource_id, value)`` caps to apply, or refuse the request.

    Resolving here rather than in the forked child is what makes a cap that
    cannot apply visible: the child cannot report anything useful, because
    CPython replaces a ``preexec_fn`` exception with the fixed message
    ``"Exception occurred in preexec_fn."`` and drops the original.

    Each requested cap is one of three things:

    * enforceable — returned, and applied in the child with no fallback;
    * above the host's hard limit — refused with :class:`SandboxError`, because
      the caller asked for an isolation property this host cannot provide and a
      silently uncapped sandbox is the worse answer. A child inherits the
      parent's hard limits, so the parent can decide this before forking;
    * unenforceable on this platform (:data:`_UNENFORCEABLE_LIMITS`) — dropped
      with a warning, since no value would work and failing every run on the
      platform is not a defensible response to a platform limitation.
    """
    applied: list[tuple[int, int]] = []
    for res_id, value, option in (
        (resource.RLIMIT_CPU, cpu_seconds, "cpu_seconds"),
        (resource.RLIMIT_AS, address_space_bytes, "address_space_bytes"),
        (resource.RLIMIT_NOFILE, max_open_files, "max_open_files"),
    ):
        if res_id in _UNENFORCEABLE_LIMITS:
            if option not in _warned_unenforceable:
                _warned_unenforceable.add(option)
                logger.warning(
                    "%s is not enforceable on %s: the sandbox runs without this cap. "
                    "Use a container backend if the limit must hold.",
                    option,
                    sys.platform,
                )
            continue
        try:
            _soft, hard = resource.getrlimit(res_id)
        except (ValueError, OSError) as exc:
            raise SandboxError(f"host does not support the {option} resource limit: {exc}") from exc
        if hard != resource.RLIM_INFINITY and value > hard:
            raise SandboxError(
                f"{option}={value} exceeds this host's hard limit of {hard} for the same "
                f"resource, so the cap would not be applied and the sandbox would run "
                f"uncapped. Lower {option} to {hard} or less, or raise the host limit."
            )
        applied.append((res_id, value))
    return tuple(applied)


def _make_rlimit_preexec(limits: tuple[tuple[int, int], ...]):
    """Build a ``preexec_fn`` applying the caps resolved by :func:`_resolve_rlimits`.

    Runs in the forked child before ``exec``. A failure here is deliberately not
    swallowed: the child dies, ``Popen`` raises, and ``exec`` reports a failed
    execution. Running sandboxed code with an isolation cap silently absent is
    the worse outcome, because nothing downstream can tell that it did not apply.
    Every cap the host is known to reject was already resolved away by the caller.
    """

    def _preexec() -> None:  # pragma: no cover — runs in the forked child
        for res_id, soft in limits:
            resource.setrlimit(res_id, (soft, soft))

    return _preexec


class LocalSubprocessBackend:
    """Hardened host-local Python execution backend.

    Each instance owns a fresh ``tempfile.mkdtemp()`` cwd and runs Python under
    a clean environment, resource limits, and a timeout. It is intended for
    trusted-shape code where the threat is accidental failure rather than an
    adversarial program; untrusted code belongs in a container backend.

    What this backend does and does not bound:

    * **Wall clock** — enforced. A timed-out command is killed as a process
      group, so it cannot leave a child running (see :meth:`exec`).
    * **CPU time** and **open files** — enforced, via ``RLIMIT_CPU`` and
      ``RLIMIT_NOFILE``.
    * **Address space** — enforced via ``RLIMIT_AS`` where the platform supports
      it. It does not on macOS; a warning names the gap once per process.
    * **Process count** — *not bounded, at any setting.* ``RLIMIT_NPROC`` counts
      every process of the real uid, not of this sandbox, so it is a threshold
      against whatever the host account already runs rather than an allowance for
      this command: measured on the development host (~420 processes owned by the
      uid), a cap of 64 or 256 blocked every ``fork`` in the sandbox with
      ``BlockingIOError: [Errno 35]``, while anything above ~420 never bound.
      No value means what ``max_processes`` says, so this backend does not set it
      and refuses the option instead of applying something else. The OS's per-uid
      limit is the only backstop. A caller that needs a real process cap wants the
      Podman backend, whose ``--pids-limit`` is a per-container cgroup limit.
    """

    def __init__(
        self,
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        cpu_seconds: int = DEFAULT_CPU_SECONDS,
        address_space_bytes: int = DEFAULT_ADDRESS_SPACE_BYTES,
        max_processes: int | None = None,
        max_open_files: int = DEFAULT_MAX_OPEN_FILES,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be a positive number of seconds")
        if max_processes is not None:
            # The default is ``None``, and no profile reaches this constructor
            # (``SandboxManager.acquire("python", **kwargs)`` forwards caller
            # kwargs verbatim), so a value here was always typed by a caller.
            # Refusing names where the capability lives; honouring it is
            # impossible and ignoring it would be the silent-config defect.
            raise SandboxError(
                f"max_processes={max_processes} cannot be honoured by the local subprocess "
                "backend: RLIMIT_NPROC is enforced per real uid, not per sandbox, so any "
                "value either blocks every fork in the sandbox or never binds. Acquire the "
                "podman backend, whose --pids-limit is a per-container cap, or drop the "
                "option to run without a process bound."
            )
        self._timeout = timeout
        # The resolved caps are the only surviving form of the requested ones:
        # they are applied in the child and named in the failure path below.
        self._rlimits = _resolve_rlimits(cpu_seconds, address_space_bytes, max_open_files)
        self._workdir = tempfile.mkdtemp(prefix="apollo_local_sbx_")
        self._released = False
        logger.debug("LocalSubprocessBackend acquired; cwd=%s", self._workdir)

    @property
    def workdir(self) -> str:
        """The backend's isolated temp working directory."""
        return self._workdir

    def copy_out(self, container_path: str, host_dest: str) -> None:
        """Copy the local workdir out to ``host_dest`` for workspace export parity."""
        del container_path
        if self._released:
            raise SandboxError("backend has been released")
        shutil.copytree(self._workdir, host_dest, dirs_exist_ok=True)

    def exec(self, cmd: str) -> ToolCallRecord:
        """Run Python source under resource limits and capture command failures.

        A timeout or crash becomes a non-zero ``ToolCallRecord``; no exception
        escapes the manager boundary.
        """
        if self._released:
            raise SandboxError("backend has been released")
        if not cmd or not cmd.strip():
            return ToolCallRecord(
                tool_id="local_python",
                exit_code=-1,
                stderr="No code provided.",
            )

        fd, temp_file = tempfile.mkstemp(suffix=".py", prefix="code_", dir=self._workdir, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(cmd)
            try:
                # ``start_new_session`` makes the child a process-group leader, so a
                # timeout below stops the whole subtree instead of orphaning anything
                # the code spawned. ``subprocess.run(timeout=)`` cannot do this: it
                # signals the direct child only, and a surviving grandchild both keeps
                # running and holds the inherited stdout/stderr pipes open.
                with subprocess.Popen(  # noqa: S603 — sandboxed, fixed argv
                    [sys.executable, temp_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self._workdir,
                    env=_build_clean_env(),
                    start_new_session=True,
                    preexec_fn=_make_rlimit_preexec(self._rlimits),  # noqa: PLW1509 — rlimits
                ) as process:
                    try:
                        stdout, stderr = process.communicate(timeout=self._timeout)
                        exit_code = process.returncode
                    except subprocess.TimeoutExpired:
                        logger.warning("Local subprocess timed out after %ss", self._timeout)
                        signal_process_group(process, signal.SIGKILL)
                        # Drain only after the group is dead. The reverse order waits
                        # on writers nothing has stopped yet, so a grandchild holding
                        # the pipes would hang the caller past its own timeout.
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            process.communicate(timeout=_POST_KILL_DRAIN_SECONDS)
                        return ToolCallRecord(
                            tool_id="local_python",
                            exit_code=-1,
                            stderr=f"Code execution timed out after {self._timeout}s",
                        )
                    finally:
                        # Any other exit with the child alive — a KeyboardInterrupt
                        # during ``communicate``, say — must not orphan the session we
                        # created, because nothing else owns it.
                        if process.poll() is None:
                            signal_process_group(process, signal.SIGKILL)
            except Exception as exc:  # noqa: BLE001 — never-crash boundary
                # CPython reports a ``preexec_fn`` failure with a fixed message, so
                # name the caps it was applying — they are all it does.
                logger.error(
                    "Local subprocess execution error: %s (resource caps %r)",
                    exc,
                    self._rlimits,
                )
                return ToolCallRecord(
                    tool_id="local_python",
                    exit_code=-1,
                    stderr=f"Exception during execution: {exc}",
                )
            return ToolCallRecord(
                tool_id="local_python",
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                fs_diff={},
            )
        finally:
            try:
                os.unlink(temp_file)
            except OSError as exc:
                logger.debug("Failed to remove temp code file %s: %s", temp_file, exc)

    def release(self) -> None:
        """Remove the temp cwd. Idempotent (safe to call twice)."""
        if self._released:
            return
        self._released = True
        shutil.rmtree(self._workdir, ignore_errors=True)
        logger.debug("LocalSubprocessBackend released; removed cwd=%s", self._workdir)
