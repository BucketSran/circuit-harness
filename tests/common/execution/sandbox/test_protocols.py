"""Structural contract tests for the sandbox protocols (house style).

Assert the concrete backends and snapshotter satisfy the runtime_checkable
protocols, so an owner cannot drift the seam without breaking these.
"""

from __future__ import annotations

from alphaapollo.common.execution.sandbox.base import (
    SandboxBackend,
    SandboxLifecycle,
    Snapshotter,
)


def test_podman_backend_satisfies_sandbox_backend() -> None:
    from alphaapollo.common.execution.sandbox.podman import PodmanBackend

    assert issubclass(PodmanBackend, SandboxBackend)


def test_docker_and_local_backends_satisfy_sandbox_backend() -> None:
    from alphaapollo.common.execution.sandbox.docker import DockerBackend
    from alphaapollo.common.execution.sandbox.local import LocalSubprocessBackend

    assert issubclass(DockerBackend, SandboxBackend)
    assert issubclass(LocalSubprocessBackend, SandboxBackend)


def test_legacy_docker_backend_exports_remain_compatible() -> None:
    from alphaapollo.common.execution.sandbox.docker import (
        LocalSubprocessBackend as LegacyLocalSubprocessBackend,
    )
    from alphaapollo.common.execution.sandbox.docker import SandboxManager as LegacySandboxManager
    from alphaapollo.common.execution.sandbox.local import LocalSubprocessBackend
    from alphaapollo.common.execution.sandbox.manager import SandboxManager

    assert LegacyLocalSubprocessBackend is LocalSubprocessBackend
    assert LegacySandboxManager is SandboxManager


def test_snapshotter_protocol_satisfied() -> None:
    from alphaapollo.common.execution.workspace import WorkspaceSnapshotter

    assert issubclass(WorkspaceSnapshotter, Snapshotter)


def test_lifecycle_enum_values() -> None:
    assert {m.value for m in SandboxLifecycle} == {
        "per_problem",
        "per_branch",
        "per_call",
    }
