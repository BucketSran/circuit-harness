"""Framework-neutral registry for persistent memory adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .interface import SharedMemory
from .persistent import Mem0MemoryAdapter

PersistentMemoryFactory = Callable[[Mapping[str, Any]], SharedMemory]


class MemoryBackendRegistry:
    """Resolve a configured provider to the common ``SharedMemory`` contract."""

    def __init__(self) -> None:
        self._factories: dict[str, PersistentMemoryFactory] = {}

    def register(self, name: str, factory: PersistentMemoryFactory) -> None:
        canonical = _backend_name(name)
        if canonical in self._factories:
            raise ValueError(f"memory backend {canonical!r} is already registered")
        if not callable(factory):
            raise TypeError("memory backend factory must be callable")
        self._factories[canonical] = factory

    def build(self, name: str, config: Mapping[str, Any]) -> SharedMemory:
        canonical = _backend_name(name)
        try:
            factory = self._factories[canonical]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(
                f"unknown memory backend {canonical!r}; registered backends: {available}"
            ) from exc
        backend = factory(dict(config))
        if not isinstance(backend, SharedMemory):
            raise TypeError(f"memory backend {canonical!r} does not implement SharedMemory")
        return backend

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _backend_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("memory backend name must be non-empty")
    return name.strip().casefold()


DEFAULT_MEMORY_BACKENDS = MemoryBackendRegistry()
DEFAULT_MEMORY_BACKENDS.register("mem0", Mem0MemoryAdapter.from_config)

__all__ = [
    "DEFAULT_MEMORY_BACKENDS",
    "MemoryBackendRegistry",
    "PersistentMemoryFactory",
]
