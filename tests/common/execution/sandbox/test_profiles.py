"""GPU/container-free tests for SandboxProfile (the named sandbox config catalog).

Covers:
  - preset profiles carry the expected values;
  - profiles are frozen (immutable) and validate their fields (fail-loud);
  - get_profile / list_profiles behave;
  - with_overrides returns a new profile without mutating the original;
  - CONSISTENCY: the local-subprocess preset resource caps equal the current
    sandbox_manager.DEFAULT_* constants, proving this collection changes no behaviour.

Pure Python; no Podman/Docker/GPU.
"""

from __future__ import annotations

import dataclasses

import pytest

from alphaapollo.common.execution.sandbox import docker as sandbox_manager
from alphaapollo.common.execution.sandbox.base import (
    PODMAN_DEFAULT,
    PYTHON_DEFAULT,
    VERIFIER_DEFAULT,
    SandboxProfile,
    SandboxProfileError,
    get_profile,
    list_profiles,
)

# --- presets -----------------------------------------------------------------

DOCKER_PROFILE = SandboxProfile(
    name="docker_test", kind="docker", image="python:3.11-slim", network=False
)


def test_python_default_is_local_no_network_no_mounts() -> None:
    assert PYTHON_DEFAULT.name == "python_default"
    assert PYTHON_DEFAULT.kind == "local_subprocess"
    assert PYTHON_DEFAULT.network is False
    assert PYTHON_DEFAULT.allow_host_mounts is False
    assert PYTHON_DEFAULT.image is None


def test_docker_profile_targets_a_container_without_network() -> None:
    assert DOCKER_PROFILE.kind == "docker"
    assert DOCKER_PROFILE.image == "python:3.11-slim"
    assert DOCKER_PROFILE.network is False


def test_podman_default_is_container_with_workspace_image() -> None:
    assert PODMAN_DEFAULT.kind == "podman"
    assert PODMAN_DEFAULT.image == "python:3.11-slim"
    assert PODMAN_DEFAULT.max_open_files == 4096
    assert PODMAN_DEFAULT.network is False


def test_verifier_default_is_strictest_isolation() -> None:
    assert VERIFIER_DEFAULT.network is False
    assert VERIFIER_DEFAULT.allow_host_mounts is False


# --- consistency with the current defaults (no behaviour change) -------------
def test_python_default_resource_caps_match_sandbox_manager_defaults() -> None:
    """The preset must equal the live DEFAULT_* so adopting it is a no-op change."""
    assert PYTHON_DEFAULT.timeout_seconds == sandbox_manager.DEFAULT_TIMEOUT_SECONDS
    assert PYTHON_DEFAULT.cpu_seconds == sandbox_manager.DEFAULT_CPU_SECONDS
    assert PYTHON_DEFAULT.memory_bytes == sandbox_manager.DEFAULT_ADDRESS_SPACE_BYTES
    assert PYTHON_DEFAULT.max_processes == sandbox_manager.DEFAULT_MAX_PROCESSES
    assert PYTHON_DEFAULT.max_open_files == sandbox_manager.DEFAULT_MAX_OPEN_FILES


# --- immutability + validation -----------------------------------------------
def test_profile_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PYTHON_DEFAULT.timeout_seconds = 5  # type: ignore[misc]


def test_negative_timeout_rejected() -> None:
    with pytest.raises(SandboxProfileError, match="timeout_seconds"):
        SandboxProfile(name="bad", kind="local_subprocess", timeout_seconds=-1)


def test_timeout_accepts_none_and_fractional_seconds() -> None:
    assert (
        SandboxProfile(name="unbounded", kind="podman", timeout_seconds=None).timeout_seconds
        is None
    )
    assert (
        SandboxProfile(name="fractional", kind="podman", timeout_seconds=0.5).timeout_seconds == 0.5
    )


def test_zero_memory_rejected() -> None:
    with pytest.raises(SandboxProfileError, match="memory_bytes"):
        SandboxProfile(name="bad", kind="local_subprocess", memory_bytes=0)


def test_bool_is_not_a_valid_int_cap() -> None:
    """A bool must not sneak in as an int cap (True == 1 footgun)."""
    with pytest.raises(SandboxProfileError, match="cpu_seconds"):
        SandboxProfile(name="bad", kind="local_subprocess", cpu_seconds=True)  # type: ignore[arg-type]


def test_empty_name_or_kind_rejected() -> None:
    with pytest.raises(SandboxProfileError, match="name"):
        SandboxProfile(name="", kind="local_subprocess")
    with pytest.raises(SandboxProfileError, match="kind"):
        SandboxProfile(name="x", kind="")


def test_blank_image_rejected() -> None:
    with pytest.raises(SandboxProfileError, match="image"):
        SandboxProfile(name="x", kind="docker", image="   ")


def test_kind_must_name_a_known_backend_family() -> None:
    """``kind`` is matched against the backend at acquisition, so it is a closed set.

    A kind outside ``local_subprocess`` / ``podman`` / ``docker`` names no backend
    and could therefore never match one; catching the typo at construction beats
    refusing it at acquire time with the same message.
    """
    with pytest.raises(SandboxProfileError, match="names no backend family"):
        SandboxProfile(name="x", kind="podmn")


def test_local_subprocess_profile_carries_no_container_only_field() -> None:
    """``image`` and ``cpus`` are container-only, and the rule is enforced, not documented.

    The local subprocess backend runs the host interpreter under rlimits: it has no
    image to pull and no cgroup to rate-limit with. A local profile carrying either
    would read as configuration and change nothing, so construction refuses it —
    the only place the conflict is visible, since no local profile is ever handed
    to a backend that could report it.
    """
    with pytest.raises(SandboxProfileError, match="cannot run an image"):
        SandboxProfile(name="x", kind="local_subprocess", image="python:3.11-slim")
    with pytest.raises(SandboxProfileError, match="cpus"):
        SandboxProfile(name="x", kind="local_subprocess", cpus=0.5)
    with pytest.raises(SandboxProfileError, match="CDI GPU"):
        SandboxProfile(name="x", kind="local_subprocess", gpus=("0",))
    with pytest.raises(SandboxProfileError, match="CDI GPU"):
        SandboxProfile(name="x", kind="docker", image="img", gpus=("0",))
    # Container kinds keep both, and every preset already satisfies the rule.
    assert SandboxProfile(name="x", kind="podman", image="img", cpus=0.5).cpus == 0.5


def test_require_kind_refuses_a_foreign_backend_and_passes_its_own() -> None:
    """``require_kind`` names both sides of the conflict, and stays silent on a match."""
    with pytest.raises(SandboxProfileError) as excinfo:
        PYTHON_DEFAULT.require_kind("podman")
    message = str(excinfo.value)
    assert "kind='local_subprocess'" in message
    assert "'podman' backend was selected" in message
    assert PODMAN_DEFAULT.require_kind("podman") is None  # a match is silent


# --- get_profile / list_profiles ---------------------------------------------
def test_get_profile_known_and_unknown() -> None:
    assert get_profile("python_default") is PYTHON_DEFAULT
    with pytest.raises(SandboxProfileError, match="unknown sandbox profile"):
        get_profile("does_not_exist")


def test_list_profiles_sorted_and_complete() -> None:
    names = list_profiles()
    assert names == sorted(names)
    assert set(names) == {
        "python_default",
        "pytest_default",
        "verifier_default",
        "podman_default",
    }


# --- with_overrides ----------------------------------------------------------
def test_with_overrides_returns_new_profile_without_mutating_original() -> None:
    tighter = PYTHON_DEFAULT.with_overrides(timeout_seconds=5)
    assert tighter.timeout_seconds == 5
    assert tighter is not PYTHON_DEFAULT
    # original unchanged
    assert PYTHON_DEFAULT.timeout_seconds == sandbox_manager.DEFAULT_TIMEOUT_SECONDS
    # other fields carried over
    assert tighter.kind == PYTHON_DEFAULT.kind
    assert tighter.network == PYTHON_DEFAULT.network


def test_with_overrides_unknown_field_fails_loud() -> None:
    with pytest.raises(SandboxProfileError, match="unknown profile field"):
        PYTHON_DEFAULT.with_overrides(bogus=1)


def test_with_overrides_revalidates() -> None:
    with pytest.raises(SandboxProfileError, match="timeout_seconds"):
        PYTHON_DEFAULT.with_overrides(timeout_seconds=-10)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))


def test_profile_accepts_and_rejects_cpus() -> None:
    import dataclasses

    from alphaapollo.common.execution.sandbox.base import (
        SandboxProfileError,
    )

    ok = dataclasses.replace(DOCKER_PROFILE, cpus=0.5)
    assert ok.cpus == 0.5
    with pytest.raises(SandboxProfileError):
        dataclasses.replace(DOCKER_PROFILE, cpus=0)
    with pytest.raises(SandboxProfileError):
        dataclasses.replace(DOCKER_PROFILE, cpus=-1)
    with pytest.raises(SandboxProfileError):
        dataclasses.replace(DOCKER_PROFILE, cpus=float("inf"))
