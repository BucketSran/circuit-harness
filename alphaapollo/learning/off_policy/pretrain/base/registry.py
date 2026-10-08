"""Generic, fail-loud component registry for the pretrain package.

alphaapollo has no generic registry base, and verl's decorator registries
(``@register_adv_est``) are verl-coupled — unusable in this verl-free leaf. This module
provides a small, self-contained ``Registry[T]`` mirroring the in-house semantics:

- ``register(key)``: decorator; re-registering the SAME factory under ``key`` is a no-op,
  re-registering a DIFFERENT one raises ``ValueError`` (fail loud, like verl).
- ``get(key)``: returns ``factory()`` — registries store zero-arg factories (typically the
  component class); raises if unknown (no silent default; defaults live in callers, like
  the pre-existing ``archs/ARCH_REGISTRY``).
- ``get_or_custom(key)``: accepts ``"custom:<module>:<attr>"`` to import any zero-arg
  factory on the fly, else ``get(key)`` — the ``archs/`` escape hatch, generalized.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named, string-keyed registry of pluggable component factories."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._factories: dict[str, Callable[[], T]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, key: str) -> Callable[[Callable[[], T]], Callable[[], T]]:
        """Decorator registering a zero-arg factory (usually a component class) under ``key``."""

        def decorator(factory: Callable[[], T]) -> Callable[[], T]:
            existing = self._factories.get(key)
            if existing is not None and existing is not factory:
                raise ValueError(
                    f"{self._name} {key!r} already registered to {existing!r}; "
                    f"refusing to overwrite with {factory!r}."
                )
            self._factories[key] = factory
            return factory

        return decorator

    def get(self, key: str) -> T:
        """Return a fresh component instance for ``key`` (calls the stored factory)."""
        try:
            factory = self._factories[key]
        except KeyError:
            raise ValueError(
                f"Unknown {self._name} {key!r}. Known: {self.names()}. "
                f"Use a registered name or 'custom:<module>:<attr>'."
            ) from None
        return factory()

    def get_or_custom(self, key: str) -> T:
        """``"custom:<module>:<attr>"`` imports + calls a zero-arg factory; else ``get(key)``."""
        if isinstance(key, str) and key.startswith("custom:"):
            spec = key[len("custom:") :]
            mod_path, sep, attr = spec.rpartition(":")
            if not sep or not mod_path or not attr:
                raise ValueError(f"{self._name}: 'custom:' needs '<module>:<attr>', got {key!r}.")
            mod = importlib.import_module(mod_path)
            if not hasattr(mod, attr):
                raise AttributeError(
                    f"{self._name}: custom module {mod_path!r} has no attribute {attr!r}."
                )
            return getattr(mod, attr)()
        return self.get(key)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, key: object) -> bool:
        return key in self._factories
