# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Select and configure the concrete backend for a sandbox kind."""

from __future__ import annotations

from typing import Any

from alphaapollo.common.execution.sandbox.base import SandboxBackend, SandboxError
from alphaapollo.common.execution.sandbox.local import LocalSubprocessBackend

_LOCAL_KINDS = frozenset({"python"})
_DOCKER_KINDS = frozenset({"cli", "swe"})
_PODMAN_KINDS = frozenset({"podman"})


def _network_mode(network: bool | None) -> str | None:
    """Map a network policy to Docker's no-network mode."""
    return "none" if network is False else None


class SandboxManager:
    """Dispatch a sandbox kind to an isolated concrete backend.

    ``python`` selects :class:`LocalSubprocessBackend`; ``cli`` and
    ``swe`` select the Docker backend; ``podman`` selects the rootless Podman
    backend. Every acquisition returns an isolated backend owned by the caller.
    """

    def acquire(self, kind: str, *, network: bool | None = None, **kwargs: Any) -> SandboxBackend:
        """Acquire an isolated backend for ``kind``.

        Args:
            kind: Registered local or container backend kind.
            network: Optional network policy. ``False`` disables container
                networking; ``None`` retains the backend-specific default. The
                local subprocess backend has no network namespace, so it accepts
                only ``True`` (which is what it already does) or ``None``.
            **kwargs: Backend-specific options such as resource limits, image,
                instance id, workspace, profile, or command runner.

        Raises:
            TypeError: If Podman receives an unknown backend option.
            ValueError: If ``kind`` or its backend configuration is invalid.
            SandboxProfileError: If a supplied profile targets a different backend
                family than the one ``kind`` selects.
            SandboxError: If the selected backend cannot provide the requested
                network policy, or two inputs declare it at once.
        """
        if kind in _LOCAL_KINDS:
            if network is False:
                # Note the polarity: it is the RESTRICTIVE request this backend
                # cannot provide. It runs the command as an ordinary host process,
                # which has the host's network whatever this option says, so
                # ``network=False`` would read as honoured while providing nothing;
                # ``network=True`` is exactly what it does and is accepted below.
                # ``acquire`` forwards caller kwargs verbatim here and never applies
                # a profile, so a value that arrives was typed by a caller.
                raise SandboxError(
                    "network=False cannot be honoured by the local subprocess backend: it "
                    "runs the command as a host process with no network namespace, so the "
                    "sandbox keeps the host's network at any setting. Note that it is the "
                    "restrictive value that is refused here, not the permissive one — "
                    "network=True describes what this backend already provides. Acquire the "
                    "podman backend, whose --network none is a real namespace, or drop the "
                    "option to accept host networking."
                )
            return LocalSubprocessBackend(**kwargs)
        if kind in _DOCKER_KINDS:
            from alphaapollo.common.execution.sandbox.docker import DockerBackend

            image = kwargs.pop("image", None)
            if image is None:
                image = "ubuntu:22.04"
            if network is not None and "network_mode" in kwargs:
                # Both declare the same property, and the caller means both. Docker's
                # own mode used to win by ``setdefault``, so network=False plus
                # network_mode="host" ran on the host network with the stricter
                # request silently discarded.
                raise SandboxError(
                    f"network={network!r} and network_mode={kwargs['network_mode']!r} both "
                    "declare this sandbox's network policy. Pass one: network is the "
                    "portable policy every kind understands, network_mode is Docker's own "
                    "setting."
                )
            policy = network
            mode = _network_mode(policy)
            if mode is not None:
                # ``setdefault`` now only ever fills in a derived default,
                # because the case where
                # it would have overridden a stated ``network`` was refused above.
                kwargs.setdefault("network_mode", mode)
            return DockerBackend(image=image, tool_id=kind, **kwargs)
        if kind in _PODMAN_KINDS:
            from alphaapollo.common.execution.sandbox.base import (
                PODMAN_DEFAULT,
                SandboxProfile,
                get_profile,
            )
            from alphaapollo.common.execution.sandbox.podman import PodmanBackend

            profile = kwargs.pop("profile", None)
            if profile is None:
                profile_name = kwargs.pop("profile_name", None)
                profile = get_profile(profile_name) if profile_name else PODMAN_DEFAULT
            elif not isinstance(profile, SandboxProfile):
                raise ValueError("acquire(kind='podman') profile must be a SandboxProfile")
            # Refuse before any override is applied, so the reported conflict is the
            # profile the caller named rather than a symptom of it (an imageless
            # local profile otherwise fails below for a missing image, which is not
            # the reason it may not run here).
            profile.require_kind("podman")

            image = kwargs.pop("image", None)
            overrides: dict[str, Any] = {}
            if image is not None:
                overrides["image"] = image
            if network is not None:
                overrides["network"] = bool(network)
            profile_fields = {
                "timeout_seconds",
                "cpu_seconds",
                "memory_bytes",
                "max_processes",
                "max_open_files",
                "cpus",
                "allow_host_mounts",
                "gpus",
                "read_only_root",
                "drop_all_capabilities",
                "no_new_privileges",
                "run_as_user",
                "workspace_tmpfs_bytes",
                "tmp_tmpfs_bytes",
            }
            for field_name in tuple(kwargs):
                if field_name in profile_fields:
                    overrides[field_name] = kwargs.pop(field_name)
            if overrides:
                profile = profile.with_overrides(**overrides)

            instance_id = kwargs.pop("instance_id", None)
            tool_id = kwargs.pop("tool_id", kind)
            runner = kwargs.pop("runner", None)
            if kwargs:
                raise TypeError(f"unknown podman backend option(s): {sorted(kwargs)}")
            if profile.image is None:
                raise ValueError(
                    f"profile {profile.name!r} has no container image; "
                    "provide image=... or choose a container profile"
                )
            podman_kwargs: dict[str, Any] = {
                "profile": profile,
                "instance_id": instance_id,
                "tool_id": tool_id,
            }
            if runner is not None:
                podman_kwargs["runner"] = runner
            return PodmanBackend(**podman_kwargs)
        raise ValueError(
            f"Unknown sandbox kind {kind!r} (expected one of "
            f"{sorted(_LOCAL_KINDS | _DOCKER_KINDS | _PODMAN_KINDS)})"
        )
