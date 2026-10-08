# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"Execution backend contracts, isolation profiles, and lifecycle management."

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from alphaapollo.common.execution.tools.schemas import ToolCallRecord

# Resource-cap defaults — kept identical to local.DEFAULT_* so a profile
# is a pure rename/collection of the current values (verified by the profile tests).
_DEFAULT_TIMEOUT_SECONDS = 30
_DEFAULT_CPU_SECONDS = 30
_DEFAULT_MEMORY_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB (== DEFAULT_ADDRESS_SPACE_BYTES)
# Honoured by the Podman backend only, as ``--pids-limit``. The local subprocess
# backend cannot express a per-sandbox process cap and refuses the option instead
# of applying something else — see ``LocalSubprocessBackend`` for the measurement.
_DEFAULT_MAX_PROCESSES = 64
_DEFAULT_MAX_OPEN_FILES = 64

#: The backend families a profile may target. ``kind`` is what a reader consults to
#: tell a container sandbox from a host subprocess, so it is checked against this
#: set at construction and against the selected backend at acquisition
#: (:meth:`SandboxProfile.require_kind`). A kind outside this set names no backend
#: and could never be matched.
_PROFILE_KINDS = frozenset({"local_subprocess", "podman", "docker"})
_LOCAL_SUBPROCESS_KIND = "local_subprocess"


class SandboxProfileError(Exception):
    """Raised on a malformed profile (fail-loud, never a silent bad config)."""


class SandboxError(Exception):
    """Raised for sandbox acquisition/configuration failures (fail-loud)."""


@dataclass(frozen=True)
class SandboxProfile:
    """An immutable resource + security config for one sandbox acquisition.

    A profile is a pure-data descriptor: it holds the caps and policy a backend
    should enforce, but performs no execution itself. ``kind`` names the backend
    family the profile targets (``"local_subprocess"`` / ``"podman"`` / ``"docker"``);
    ``image`` is the OCI image for container kinds and is ``None`` for the local
    subprocess kind.

    ``kind`` is a contract, not a label: it is the field a reader consults to answer
    whether a sandbox is a container or a subprocess on the host, and the two have
    different isolation properties. It is therefore enforced twice — against the
    known families here, and against the backend that was actually selected, by
    :meth:`require_kind` at acquisition. The container-only fields follow from it:
    a ``local_subprocess`` profile carries no ``image`` and no ``cpus``, because
    nothing on that path could honour either.

    All fields are validated in ``__post_init__`` so an out-of-range profile fails
    at construction, not at run time.
    """

    name: str
    kind: str
    # Resource caps.
    # ``None`` is an explicit caller opt-out of a wall-clock command timeout.
    # Preset profiles retain their finite defaults. Podman uses its profile
    # default when neither the system nor the tool requests a shorter timeout.
    timeout_seconds: float | None = _DEFAULT_TIMEOUT_SECONDS
    cpu_seconds: int = _DEFAULT_CPU_SECONDS
    memory_bytes: int = _DEFAULT_MEMORY_BYTES
    max_processes: int = _DEFAULT_MAX_PROCESSES
    max_open_files: int = _DEFAULT_MAX_OPEN_FILES
    # Security policy (the execution-substrate boundary).
    network: bool = False
    allow_host_mounts: bool = False
    read_only_root: bool = False
    drop_all_capabilities: bool = False
    no_new_privileges: bool = False
    run_as_user: str | None = None
    workspace_tmpfs_bytes: int | None = None
    tmp_tmpfs_bytes: int | None = None
    # Container-only: the OCI image to run (None for local_subprocess).
    image: str | None = None
    # Container-only: --cpus rate cap (cores, e.g. 0.5). None = no rate limit.
    # Orthogonal to cpu_seconds: --cpus throttles rate, cpu_seconds (RLIMIT_CPU)
    # bounds total CPU time and kills a runaway command.
    cpus: float | None = None
    # Container-only: GPUs to pass through via CDI (``nvidia.com/gpu``).
    # Empty tuple means no GPU. Values are operator-resolved device selectors,
    # never model-authored; they name an accelerator grant, not an env override.
    gpus: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise SandboxProfileError("SandboxProfile.name must be a non-empty string")
        if not self.kind or not self.kind.strip():
            raise SandboxProfileError("SandboxProfile.kind must be a non-empty string")
        if self.kind not in _PROFILE_KINDS:
            raise SandboxProfileError(
                f"SandboxProfile.kind {self.kind!r} names no backend family; "
                f"known kinds: {sorted(_PROFILE_KINDS)}"
            )
        timeout = self.timeout_seconds
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise SandboxProfileError(
                "SandboxProfile.timeout_seconds must be None or a finite positive number"
            )
        for field_name in (
            "cpu_seconds",
            "memory_bytes",
            "max_processes",
            "max_open_files",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise SandboxProfileError(
                    f"SandboxProfile.{field_name} must be an int, got {type(value).__name__}"
                )
            if value <= 0:
                raise SandboxProfileError(
                    f"SandboxProfile.{field_name} must be positive, got {value}"
                )
        if not isinstance(self.network, bool):
            raise SandboxProfileError("SandboxProfile.network must be a bool")
        if not isinstance(self.allow_host_mounts, bool):
            raise SandboxProfileError("SandboxProfile.allow_host_mounts must be a bool")
        for field_name in ("read_only_root", "drop_all_capabilities", "no_new_privileges"):
            if not isinstance(getattr(self, field_name), bool):
                raise SandboxProfileError(f"SandboxProfile.{field_name} must be a bool")
        if self.run_as_user is not None and (
            not isinstance(self.run_as_user, str) or not self.run_as_user.strip()
        ):
            raise SandboxProfileError(
                "SandboxProfile.run_as_user must be None or a non-empty user[:group] string"
            )
        for field_name in ("workspace_tmpfs_bytes", "tmp_tmpfs_bytes"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise SandboxProfileError(
                    f"SandboxProfile.{field_name} must be None or a positive int"
                )
        if not self.read_only_root and (
            self.workspace_tmpfs_bytes is not None or self.tmp_tmpfs_bytes is not None
        ):
            raise SandboxProfileError(
                "writable tmpfs bounds require SandboxProfile.read_only_root=True"
            )
        if self.image is not None and (not isinstance(self.image, str) or not self.image.strip()):
            raise SandboxProfileError("SandboxProfile.image must be None or a non-empty string")
        if self.cpus is not None:
            if isinstance(self.cpus, bool) or not isinstance(self.cpus, (int, float)):
                raise SandboxProfileError("SandboxProfile.cpus must be None or a number")
            if not math.isfinite(self.cpus):
                raise SandboxProfileError(f"SandboxProfile.cpus must be finite, got {self.cpus}")
            if self.cpus <= 0:
                raise SandboxProfileError(f"SandboxProfile.cpus must be positive, got {self.cpus}")
        if not isinstance(self.gpus, tuple) or any(
            not isinstance(g, str) or not g.strip() for g in self.gpus
        ):
            raise SandboxProfileError(
                "SandboxProfile.gpus must be a tuple of non-empty device selectors"
            )
        if self.gpus and self.kind != "podman":
            raise SandboxProfileError(
                f"SandboxProfile {self.name!r} declares gpus for kind={self.kind!r}, "
                "but only the podman backend implements CDI GPU passthrough"
            )
        if self.kind == _LOCAL_SUBPROCESS_KIND:
            # Both fields are container-only. Carrying either on a local profile
            # would read as configuration while changing nothing: the local
            # subprocess backend runs the host interpreter (no image to pull) and
            # has no cgroup to rate-limit with. Refusing here is the only place the
            # conflict is visible, because no local profile reaches a backend that
            # could report it.
            if self.image is not None:
                raise SandboxProfileError(
                    f"SandboxProfile {self.name!r} declares kind={_LOCAL_SUBPROCESS_KIND!r} "
                    f"with image={self.image!r}: the local subprocess backend runs the host "
                    "interpreter and cannot run an image. Drop image, or declare a "
                    "container kind."
                )
            if self.cpus is not None:
                raise SandboxProfileError(
                    f"SandboxProfile {self.name!r} declares kind={_LOCAL_SUBPROCESS_KIND!r} "
                    f"with cpus={self.cpus}: a --cpus rate cap is a container property the "
                    "local subprocess backend cannot apply. Drop cpus, or declare a "
                    "container kind."
                )

    def require_kind(self, backend_kind: str) -> None:
        """Refuse this profile if it does not target the backend about to run it.

        Called where a profile and a concrete backend first meet, so a profile that
        declares one isolation substrate can never be honoured by another. A
        ``podman`` profile served by the local subprocess backend, or the reverse,
        would give the caller host-level execution while the configuration they
        read says container — the two bound different things, so the mismatch is
        an error rather than a preference to resolve.
        """
        if self.kind != backend_kind:
            raise SandboxProfileError(
                f"profile {self.name!r} declares kind={self.kind!r} but the "
                f"{backend_kind!r} backend was selected: a {self.kind!r} profile does "
                f"not describe the isolation a {backend_kind!r} sandbox provides. "
                f"Acquire a {backend_kind!r}-kind profile, or route this profile to "
                f"the {self.kind!r} backend."
            )

    def with_overrides(self, **overrides: Any) -> SandboxProfile:
        """Return a NEW profile with ``overrides`` applied; the original is unchanged.

        Only declared fields may be overridden (an unknown field fails loud). The
        result is re-validated, so an override that violates a bound is rejected.
        """
        allowed = set(self.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(overrides) - allowed
        if unknown:
            raise SandboxProfileError(
                f"unknown profile field(s) {sorted(unknown)}; allowed={sorted(allowed)}"
            )
        return replace(self, **overrides)


# ---------------------------------------------------------------------------
# Preset profiles — the named catalog. Resource values mirror the current
# local.DEFAULT_* so adopting a profile changes no existing behaviour.
# ---------------------------------------------------------------------------

#: Host-local Python execution. Mirrors LocalSubprocessBackend defaults.
PYTHON_DEFAULT = SandboxProfile(
    name="python_default",
    kind="local_subprocess",
    network=False,
    allow_host_mounts=False,
)

#: Running a pytest suite — a longer wall-clock than a one-shot compute.
PYTEST_DEFAULT = SandboxProfile(
    name="pytest_default",
    kind="local_subprocess",
    timeout_seconds=120,
    cpu_seconds=120,
    network=False,
    allow_host_mounts=False,
)

# Generic rootless Podman profile.
PODMAN_DEFAULT = SandboxProfile(
    name="podman_default",
    kind="podman",
    # Rootless crun opens host CPU sysfs descriptors before entering the
    # container. The local-subprocess default of 64 can therefore fail during
    # OCI startup on many-core hosts before a sandboxed command ever runs.
    max_open_files=4096,
    network=False,
    allow_host_mounts=False,
    image="python:3.11-slim",
)

#: Verifier-side isolated workspace — the strictest policy (no net, no host mounts).
VERIFIER_DEFAULT = SandboxProfile(
    name="verifier_default",
    kind="local_subprocess",
    timeout_seconds=120,
    cpu_seconds=120,
    network=False,
    allow_host_mounts=False,
)

_PRESETS: dict[str, SandboxProfile] = {
    profile.name: profile
    for profile in (
        PYTHON_DEFAULT,
        PYTEST_DEFAULT,
        VERIFIER_DEFAULT,
        PODMAN_DEFAULT,
    )
}


def get_profile(name: str) -> SandboxProfile:
    """Return the preset profile named ``name``; fail loud on an unknown name."""
    try:
        return _PRESETS[name]
    except KeyError:
        raise SandboxProfileError(
            f"unknown sandbox profile {name!r}; known profiles: {sorted(_PRESETS)}"
        ) from None


def list_profiles() -> list[str]:
    """Return the sorted names of all preset profiles."""
    return sorted(_PRESETS)


if TYPE_CHECKING:
    from alphaapollo.common.execution.workspace import ArtifactStore


@dataclass(frozen=True, slots=True)
class OutputChunk:
    """One decoded stdout/stderr update emitted while a command is running."""

    stream: Literal["stdout", "stderr"]
    text: str


@runtime_checkable
class OutputSink(Protocol):
    def __call__(self, chunk: OutputChunk) -> None: ...


class CancellationToken:
    """Thread-safe cooperative cancellation signal for one tool attempt."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


@runtime_checkable
class SandboxBackend(Protocol):
    """A sandbox backend: run a command, tear down, and export the workspace.

    Satisfied by the rootless-Podman, Docker and local-subprocess backends. All
    three return a persistent
    :class:`~alphaapollo.common.execution.tools.schemas.ToolCallRecord`
    (stdout/stderr/exit_code plus fs_diff/cost evidence); a tool-command failure is
    captured in the record, never raised (the never-crash boundary).
    """

    def exec(self, cmd: str) -> ToolCallRecord: ...

    def release(self) -> None: ...

    def copy_out(self, container_path: str, host_dest: str) -> None: ...


@runtime_checkable
class StreamingSandboxBackend(Protocol):
    """Optional backend capability for live output, cancellation, and artifacts."""

    def exec_stream(
        self,
        cmd: str,
        *,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> ToolCallRecord: ...


@runtime_checkable
class Snapshotter(Protocol):
    """Snapshot/restore a workspace directory to a content-addressed ref.

    Satisfied by ``WorkspaceSnapshotter``; injected into sandbox sessions and the
    verifier workspace so a filesystem can be checkpointed and rolled back.
    ``snapshot`` returns a deterministic ref (sha256 of a canonical tar); ``restore``
    round-trips the tree back to that ref.
    """

    def snapshot(self, workspace: str) -> str: ...

    def restore(self, workspace: str, ref: str) -> None: ...


class SandboxLifecycle(str, Enum):
    """How long a sandbox lives before a new one is built.

    The scheduler/loop chooses the policy; the lifecycle manager enforces it.
    """

    PER_PROBLEM = "per_problem"
    PER_BRANCH = "per_branch"
    PER_CALL = "per_call"


logger = logging.getLogger(__name__)


class SandboxLifecycleError(Exception):
    """Raised on an invalid lifecycle operation (fail-loud, never a silent no-op)."""


#: An injected callable that builds a backend for a given profile. A backend is any
#: object with ``release() -> None`` (the manager only ever tears down via release);
#: production passes ``SandboxManager.acquire``-shaped factories, tests pass a fake.
BackendFactory = Callable[[SandboxProfile], Any]
_ScopeKey = tuple[str, ...]


@dataclass
class _Handle:
    """A live sandbox handle tracked by the manager."""

    key: _ScopeKey
    backend: Any
    profile: SandboxProfile
    parent_key: _ScopeKey | None = None


@dataclass
class SandboxLifecycleManager:
    """Create / reuse / tear down sandboxes according to a lifecycle policy.

    The manager keeps a registry of live handles keyed by a lifecycle-dependent
    scope key and delegates the actual build to ``backend_factory`` and teardown to
    each backend's ``release()``. It never executes a command itself.
    """

    backend_factory: BackendFactory
    lifecycle: SandboxLifecycle = SandboxLifecycle.PER_PROBLEM
    default_profile: SandboxProfile = PYTHON_DEFAULT
    _live: dict[_ScopeKey, _Handle] = field(default_factory=dict, init=False, repr=False)
    _call_counter: int = field(default=0, init=False, repr=False)
    _fork_counter: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    # scope key
    # ------------------------------------------------------------------
    def _scope_key(self, *, problem_id: str, branch_id: str | None) -> _ScopeKey:
        """Compute the registry key for the current lifecycle policy.

        PER_PROBLEM keys by problem (branches/calls reuse one box); PER_BRANCH keys
        by (problem, branch); PER_CALL keys by a monotonically increasing counter so
        every acquire is a distinct box.
        """
        if self.lifecycle is SandboxLifecycle.PER_PROBLEM:
            return ("problem", problem_id)
        if self.lifecycle is SandboxLifecycle.PER_BRANCH:
            if branch_id is None:
                raise SandboxLifecycleError(
                    "PER_BRANCH lifecycle requires a branch_id at acquire time"
                )
            return ("branch", problem_id, branch_id)
        # PER_CALL — a distinct key per acquire.
        self._call_counter += 1
        return ("call", problem_id, branch_id or "", str(self._call_counter))

    @staticmethod
    def _branch_key(*, problem_id: str, branch_id: str) -> _ScopeKey:
        """Build an unambiguous branch key.

        IDs are user-controlled strings and may contain punctuation (including
        colons), so they must never be identified by concatenated strings.
        """
        return ("branch", problem_id, branch_id)

    @staticmethod
    def _display_key(key: _ScopeKey) -> str:
        """Return the historical human-readable key without using it for identity."""
        if key[0] == "problem":
            return f"problem:{key[1]}"
        if key[0] == "branch":
            return f"branch:{key[1]}:{key[2]}"
        return f"call:{key[1]}:{key[3]}"

    # ------------------------------------------------------------------
    # acquire / reuse
    # ------------------------------------------------------------------
    def acquire_for(
        self,
        *,
        problem_id: str,
        branch_id: str | None = None,
        profile: SandboxProfile | None = None,
    ) -> Any:
        """Return the backend for this scope, building a new one only when needed.

        Under PER_PROBLEM/PER_BRANCH an existing box for the same scope key is
        REUSED; under PER_CALL a fresh box is always built. The chosen ``profile``
        (or ``default_profile``) is passed to the factory when a build happens.
        """
        if not problem_id:
            raise SandboxLifecycleError("acquire_for requires a non-empty problem_id")
        key = self._scope_key(problem_id=problem_id, branch_id=branch_id)
        chosen = profile or self.default_profile
        existing = self._live.get(key)
        if existing is not None:
            if existing.profile != chosen:
                raise SandboxLifecycleError(
                    f"profile mismatch for {self._display_key(key)!r}: "
                    f"live={existing.profile.name!r}, requested={chosen.name!r}"
                )
            logger.debug("sandbox reuse: %s", key)
            return existing.backend
        backend = self.backend_factory(chosen)
        self._live[key] = _Handle(key=key, backend=backend, profile=chosen)
        logger.debug("sandbox build: %s (profile=%s)", key, chosen.name)
        return backend

    # ------------------------------------------------------------------
    # fork
    # ------------------------------------------------------------------
    def fork(
        self,
        *,
        problem_id: str,
        parent_branch_id: str,
        child_branch_id: str | None = None,
    ) -> tuple[str, Any]:
        """Derive a child sandbox from a live parent branch box.

        Returns ``(child_branch_id, backend)``. The parent must be a live
        PER_BRANCH box. The child is registered under its own branch key with the
        parent linkage recorded (``parent_key``) so a state-copying backend can
        honour it later. Fork is only meaningful for the PER_BRANCH lifecycle.
        """
        if self.lifecycle is not SandboxLifecycle.PER_BRANCH:
            raise SandboxLifecycleError(
                f"fork is only supported under PER_BRANCH lifecycle, not {self.lifecycle.value}"
            )
        parent_key = self._branch_key(problem_id=problem_id, branch_id=parent_branch_id)
        parent = self._live.get(parent_key)
        if parent is None:
            raise SandboxLifecycleError(
                f"cannot fork: no live parent sandbox for {self._display_key(parent_key)!r}"
            )
        if child_branch_id is None:
            self._fork_counter += 1
            child_branch_id = f"{parent_branch_id}.fork-{self._fork_counter}"
        child_key = self._branch_key(problem_id=problem_id, branch_id=child_branch_id)
        if child_key in self._live:
            raise SandboxLifecycleError(
                f"fork target already live: {self._display_key(child_key)!r}"
            )
        child_backend = self.backend_factory(parent.profile)
        self._live[child_key] = _Handle(
            key=child_key,
            backend=child_backend,
            profile=parent.profile,
            parent_key=parent_key,
        )
        logger.debug("sandbox fork: %s -> %s", parent_key, child_key)
        return child_branch_id, child_backend

    # ------------------------------------------------------------------
    # release
    # ------------------------------------------------------------------
    def release(self, *, problem_id: str, branch_id: str | None = None) -> None:
        """Release the box(es) for a scope. Idempotent (releasing twice is a no-op).

        PER_PROBLEM releases the problem box (and thus every branch sharing it);
        PER_BRANCH releases just that branch's box. PER_CALL boxes are keyed per
        call and are released via ``release_all`` (there is no stable per-call key
        to name here).
        """
        if self.lifecycle is SandboxLifecycle.PER_PROBLEM:
            self._release_key(("problem", problem_id))
            return
        if self.lifecycle is SandboxLifecycle.PER_BRANCH:
            if branch_id is None:
                raise SandboxLifecycleError("PER_BRANCH release requires a branch_id")
            self._release_key(self._branch_key(problem_id=problem_id, branch_id=branch_id))
            return
        # PER_CALL has no single stable key, so release every live call matching
        # the supplied scope.  This makes the public release API useful for the
        # normal acquire/execute/release path while retaining release_all() for
        # emergency cleanup.
        matching = [
            key
            for key in self._live
            if key[0] == "call"
            and key[1] == problem_id
            and (branch_id is None or key[2] == branch_id)
        ]
        for key in matching:
            self._release_key(key)

    def release_all(self) -> None:
        """Tear down every live sandbox, retaining failed handles for retry."""
        for key in list(self._live):
            self._release_key(key)

    def _release_key(self, key: _ScopeKey) -> None:
        handle = self._live.get(key)
        if handle is None:
            return  # idempotent
        try:
            handle.backend.release()
        except Exception as exc:  # noqa: BLE001 — teardown is best-effort
            logger.warning("error releasing sandbox %s: %s", self._display_key(key), exc)
            return
        self._live.pop(key, None)

    # ------------------------------------------------------------------
    # introspection (for tests / observability)
    # ------------------------------------------------------------------
    @property
    def live_count(self) -> int:
        """Number of currently-live sandboxes."""
        return len(self._live)

    def live_keys(self) -> list[str]:
        """Sorted keys of the currently-live sandboxes."""
        return sorted(self._display_key(key) for key in self._live)
