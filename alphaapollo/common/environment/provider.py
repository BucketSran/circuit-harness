"""Reusable pool ownership and per-episode Environment factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Any

from alphaapollo.common.environment.base import BaseEnvironment


class EnvironmentProvider:
    """Own reusable resources and vend fresh single-episode Environments."""

    def __init__(
        self,
        pool: Any,
        builder: Callable[[int, Mapping[str, Any]], BaseEnvironment[Any, str]],
        *,
        prepare_batch: Callable[[int], Sequence[int | None]] | None = None,
    ) -> None:
        self._pool = pool
        self._builder = builder
        self._prepare_batch = prepare_batch
        self._closed = False

    def episode_seeds(
        self,
        payloads: Sequence[Mapping[str, Any]],
    ) -> tuple[int | None, ...]:
        """Advance the Environment seed stream, preserving explicit task seeds."""

        if self._closed:
            raise RuntimeError("cannot prepare a batch from a closed provider")
        active_size = len(payloads)
        if active_size < 0 or active_size > int(self._pool.batch_size):
            raise ValueError(
                f"active_size must be in [0, {self._pool.batch_size}], got {active_size}"
            )
        defaults = (
            tuple(self._prepare_batch(active_size))
            if self._prepare_batch is not None
            else (None,) * active_size
        )
        if len(defaults) != active_size:
            raise RuntimeError(
                "Environment batch preparation returned "
                f"{len(defaults)} seeds for {active_size} episodes"
            )
        seeds: list[int | None] = []
        for payload, default in zip(payloads, defaults, strict=True):
            seed = payload.get("task_seed", payload.get("seed", default))
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
                raise TypeError("Environment task seed must be an integer when provided")
            seeds.append(None if seed is None else int(seed))
        return tuple(seeds)

    def environment_factory(self, task: Any) -> BaseEnvironment[Any, str]:
        if self._closed:
            raise RuntimeError("cannot create an Environment from a closed provider")
        slot = int(getattr(task, "environment_slot", -1))
        if slot < 0 or slot >= int(self._pool.batch_size):
            raise ValueError(f"invalid environment_slot {slot}")
        payload = getattr(task, "task_payload", {})
        if not isinstance(payload, Mapping):
            raise TypeError("task_payload must be a mapping")
        return self._builder(slot, payload)

    def close(self) -> None:
        if not self._closed:
            self._pool.close()
            self._closed = True


__all__ = ["EnvironmentProvider", "make_environment_provider", "register_environment_provider"]


_PROVIDER_FACTORIES: dict[str, Callable[..., EnvironmentProvider]] = {}


def register_environment_provider(
    name: str, factory: Callable[..., EnvironmentProvider], *, replace: bool = False
) -> None:
    """Register an application-owned pool builder; this branch bundles no task pools."""
    key = name.strip().casefold()
    if not key or not callable(factory):
        raise ValueError("provider name must be non-empty and factory must be callable")
    if key in _PROVIDER_FACTORIES and not replace:
        raise ValueError(f"Environment provider {key!r} is already registered")
    _PROVIDER_FACTORIES[key] = factory


def make_environment_provider(
    env_config: Any, *, is_train: bool, env_num: int, group_n: int
) -> EnvironmentProvider:
    """Compose an explicitly registered pool factory for the shared training runner."""
    name = str(env_config.env_name).strip().casefold()
    try:
        factory = _PROVIDER_FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"Environment provider {name!r} is not registered in this branch"
        ) from None
    provider = factory(env_config, is_train=is_train, env_num=env_num, group_n=group_n)
    if not isinstance(provider, EnvironmentProvider):
        raise TypeError(f"provider factory {name!r} did not return EnvironmentProvider")
    return provider
