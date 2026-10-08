"""Compatibility tests for the split Podman implementation."""

from alphaapollo.common.execution.sandbox._podman.backend import PodmanBackend as SplitBackend
from alphaapollo.common.execution.sandbox.podman import PodmanBackend


def test_podman_facade_reexports_split_backend() -> None:
    assert PodmanBackend is SplitBackend
