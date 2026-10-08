"""GPU/Podman-free tests for the podman kind wired into SandboxManager.acquire.

PodmanBackend is monkeypatched with a recording double so these tests verify the
ROUTING + profile/override plumbing without a Podman binary. The docker/local paths
are asserted unchanged (additive integration).
"""

from __future__ import annotations

import dataclasses

import pytest

from alphaapollo.common.execution.sandbox import docker as sandbox_manager
from alphaapollo.common.execution.sandbox.base import (
    PODMAN_DEFAULT,
    SandboxProfileError,
    get_profile,
)
from alphaapollo.common.execution.sandbox.docker import SandboxManager


class _RecordingPodmanBackend:
    """Stands in for PodmanBackend; records the profile/args it was built with."""

    last: dict = {}

    def __init__(self, *, profile, instance_id=None, tool_id="podman", runner=None):
        _RecordingPodmanBackend.last = {
            "profile": profile,
            "instance_id": instance_id,
            "tool_id": tool_id,
            "runner": runner,
        }
        self.profile = profile

    def exec(self, cmd):  # pragma: no cover - not exercised here
        raise NotImplementedError

    def release(self):  # pragma: no cover
        pass


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(
        "alphaapollo.common.execution.sandbox.podman.PodmanBackend", _RecordingPodmanBackend
    )
    _RecordingPodmanBackend.last = {}
    return _RecordingPodmanBackend


# --- routing -----------------------------------------------------------------
def test_acquire_podman_routes_to_podman_backend_with_default_profile(patched) -> None:
    backend = SandboxManager().acquire("podman")
    assert isinstance(backend, _RecordingPodmanBackend)
    # default is the dedicated Podman container shape (has an image, no network)
    assert patched.last["profile"].image == PODMAN_DEFAULT.image
    assert patched.last["profile"].network is False
    assert patched.last["tool_id"] == "podman"


def test_acquire_podman_named_profile(patched) -> None:
    """A named profile still routes; ``podman_default`` is the podman-kind preset."""
    SandboxManager().acquire("podman", profile_name="podman_default")
    assert patched.last["profile"] is PODMAN_DEFAULT


def test_acquire_podman_matching_profile_still_acquires(patched) -> None:
    """The regression that matters: a podman-kind profile reaches the backend intact."""
    profile = PODMAN_DEFAULT.with_overrides(timeout_seconds=7)
    backend = SandboxManager().acquire("podman", profile=profile)
    assert isinstance(backend, _RecordingPodmanBackend)
    assert patched.last["profile"] == profile


def test_acquire_podman_still_requires_an_image(patched) -> None:
    """A podman-kind profile without an image is refused for the image, not the kind."""
    with pytest.raises(ValueError, match="no container image"):
        SandboxManager().acquire(
            "podman",
            profile=dataclasses.replace(PODMAN_DEFAULT, image=None),
        )


def test_acquire_podman_named_profile_with_image(patched) -> None:
    """An image override applies to the named profile it was given."""
    SandboxManager().acquire(
        "podman",
        profile_name="podman_default",
        image="python:3.12-slim",
    )
    assert patched.last["profile"].image == "python:3.12-slim"
    assert patched.last["profile"].network is False


@pytest.mark.parametrize("profile_name", ["verifier_default"])
def test_acquire_podman_refuses_a_profile_for_another_backend(patched, profile_name) -> None:
    """A profile targeting another backend family is refused, naming both sides.

    ``verifier_default`` uses ``local_subprocess``;
    neither describes the isolation a Podman container provides. Before this check
    the docker profile acquired a Podman container from the Lean image, and the
    local profile did the same as soon as an ``image=`` override supplied what its
    kind says it never has. The refusal precedes overrides and construction, so the
    backend is never built.
    """
    expected_kind = get_profile(profile_name).kind
    for kwargs in ({}, {"image": "python:3.11-slim"}):
        with pytest.raises(SandboxProfileError) as excinfo:
            SandboxManager().acquire("podman", profile_name=profile_name, **kwargs)
        message = str(excinfo.value)
        assert f"kind={expected_kind!r}" in message
        assert "'podman' backend was selected" in message
        assert patched.last == {}  # never reached the backend


def test_acquire_podman_image_override(patched) -> None:
    SandboxManager().acquire("podman", image="python:3.11-slim")
    assert patched.last["profile"].image == "python:3.11-slim"
    # base is PODMAN_DEFAULT; override changes image but keeps no-network
    assert patched.last["profile"].network is False


def test_acquire_podman_network_override(patched) -> None:
    SandboxManager().acquire("podman", network=True, image="img")
    assert patched.last["profile"].network is True


def test_acquire_podman_rejects_bad_profile_type(patched) -> None:
    with pytest.raises(ValueError, match="SandboxProfile"):
        SandboxManager().acquire("podman", profile="not-a-profile")


def test_acquire_podman_forwards_runner_and_rejects_unknown_options(patched) -> None:
    runner = object()
    SandboxManager().acquire("podman", runner=runner, tool_id="bash")
    assert patched.last["tool_id"] == "bash"
    with pytest.raises(TypeError, match="unknown podman backend"):
        SandboxManager().acquire("podman", typo=True)


# --- docker/local unchanged (additive) --------------------------------------
def test_local_kind_still_routes_to_local_backend() -> None:
    backend = SandboxManager().acquire("python")
    assert type(backend).__name__ == "LocalSubprocessBackend"
    backend.release()


def test_unknown_kind_lists_podman_in_error() -> None:
    with pytest.raises(ValueError, match="podman"):
        SandboxManager().acquire("bogus")


def test_podman_in_kind_routing_set() -> None:
    assert "podman" in sandbox_manager._PODMAN_KINDS


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
