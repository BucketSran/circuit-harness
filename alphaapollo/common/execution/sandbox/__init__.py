"""Interchangeable local, Docker, and Podman execution backends."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DockerBackend": "alphaapollo.common.execution.sandbox.docker",
    "LocalSubprocessBackend": "alphaapollo.common.execution.sandbox.local",
    "PodmanBackend": "alphaapollo.common.execution.sandbox.podman",
    "SandboxError": "alphaapollo.common.execution.sandbox.base",
    "SandboxManager": "alphaapollo.common.execution.sandbox.manager",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
