"""Compatibility tests for the former sandbox backend package paths."""

from __future__ import annotations

from importlib import import_module

import pytest

_MODULE_SYMBOLS = {
    "": (
        "DockerBackend",
        "LocalSubprocessBackend",
        "PodmanBackend",
        "SandboxError",
        "SandboxManager",
    ),
    "base": (
        "CancellationToken",
        "OutputChunk",
        "OutputSink",
        "SandboxBackend",
        "SandboxError",
        "SandboxLifecycle",
        "SandboxLifecycleManager",
        "SandboxProfile",
        "Snapshotter",
    ),
    "docker": (
        "CLISandboxEnv",
        "DockerBackend",
        "LocalSubprocessBackend",
        "SandboxError",
        "SandboxManager",
        "_DOCKER_KINDS",
    ),
    "local": ("LocalSubprocessBackend",),
    "manager": ("SandboxManager",),
    "podman": (
        "PodmanBackend",
        "PodmanBackendError",
        "PodmanCliError",
        "default_runner",
        "run_container",
    ),
    "_podman.backend": ("PodmanBackend", "PodmanBackendError"),
    "_podman.cli": ("exec_in", "run_container"),
    "_podman.runner": ("PodmanCliError", "PodmanResult", "default_runner"),
}


@pytest.mark.parametrize(("suffix", "symbols"), _MODULE_SYMBOLS.items())
def test_backend_compatibility_modules_reexport_canonical_symbols(
    suffix: str,
    symbols: tuple[str, ...],
) -> None:
    legacy_name = "alphaapollo.common.execution.backends"
    canonical_name = "alphaapollo.common.execution.sandbox"
    if suffix:
        legacy_name = f"{legacy_name}.{suffix}"
        canonical_name = f"{canonical_name}.{suffix}"

    legacy = import_module(legacy_name)
    canonical = import_module(canonical_name)

    if suffix:
        assert legacy is canonical
    for symbol in symbols:
        assert getattr(legacy, symbol) is getattr(canonical, symbol)


def test_backend_implementations_are_owned_by_the_sandbox_package() -> None:
    from alphaapollo.common.execution.backends import (
        DockerBackend,
        LocalSubprocessBackend,
        PodmanBackend,
        SandboxManager,
    )

    implementations = (
        DockerBackend,
        LocalSubprocessBackend,
        PodmanBackend,
        SandboxManager,
    )
    assert all(
        ".execution.sandbox." in implementation.__module__ for implementation in implementations
    )
