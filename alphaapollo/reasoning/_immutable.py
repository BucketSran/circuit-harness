"""Internal helpers for immutable, JSON-shaped Reasoning records.

Public Reasoning dataclasses are frozen value records.  Freezing only their
attributes is insufficient when an attribute contains mutable ``dict`` and
``list`` objects, so JSON-shaped audit payloads are normalized recursively at
construction time.  Mutable copies are produced only at explicit integration
boundaries through :func:`_thaw_json`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__: list[str] = []


class FrozenDict(dict[str, Any]):
    """A recursively frozen mapping that remains JSON and pickle compatible."""

    __slots__ = ("_initialized",)

    def __new__(cls, *_args: object, **_kwargs: object) -> FrozenDict:
        result = dict.__new__(cls)
        object.__setattr__(result, "_initialized", False)
        return result

    def __init__(self, *args: object, **kwargs: object) -> None:
        # ``dataclasses.asdict`` reconstructs mapping subclasses from an
        # iterable of pairs.  Accept that standard construction path, but
        # normalize it recursively and install values only through the base
        # implementation so no mutable phase is exposed.
        if self._initialized:
            raise TypeError("frozen JSON mappings do not support reinitialization")
        if not args and not kwargs:
            object.__setattr__(self, "_initialized", True)
            return
        source = dict(*args, **kwargs)  # type: ignore[arg-type]
        frozen = _freeze_json_mapping(source, where="FrozenDict")
        for key, value in frozen.items():
            dict.__setitem__(self, key, value)
        object.__setattr__(self, "_initialized", True)

    @classmethod
    def _from_items(cls, items: Sequence[tuple[str, Any]]) -> FrozenDict:
        result = cls()
        for key, value in items:
            dict.__setitem__(result, key, value)
        return result

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("frozen JSON mappings do not support mutation")

    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    __setattr__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    @classmethod
    def fromkeys(cls, iterable: object, value: object = None) -> FrozenDict:
        return _freeze_json_mapping(dict.fromkeys(iterable, value), where="FrozenDict")

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[object, tuple[tuple[tuple[str, Any], ...]]]:
        return (_restore_frozen_dict, (tuple(self.items()),))


def _restore_frozen_dict(items: tuple[tuple[str, Any], ...]) -> FrozenDict:
    """Rebuild a frozen mapping without exposing a mutable pickle phase."""

    return FrozenDict._from_items(items)


def _freeze_json_mapping(value: Mapping[str, Any], *, where: str) -> FrozenDict:
    """Validate, isolate, and recursively freeze one JSON object."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be a mapping")
    frozen = _freeze_json(value, where=where, active=set())
    if not isinstance(frozen, FrozenDict):  # defensive; Mapping normalized above
        raise TypeError(f"{where} must be a mapping")
    return frozen


def _freeze_json(value: Any, *, where: str, active: set[int]) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} must not contain NaN or infinity")
        return float(value)

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{where} must not contain recursive containers")
        active.add(identity)
        try:
            items: list[tuple[str, Any]] = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{where} keys must be strings")
                normalized_key = str(key)
                items.append(
                    (
                        normalized_key,
                        _freeze_json(
                            item,
                            where=f"{where}.{normalized_key}",
                            active=active,
                        ),
                    )
                )
            return FrozenDict._from_items(items)
        finally:
            active.remove(identity)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{where} must not contain recursive containers")
        active.add(identity)
        try:
            return tuple(_freeze_json(item, where=f"{where}[]", active=active) for item in value)
        finally:
            active.remove(identity)

    raise ValueError(f"{where} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    """Return mutable plain JSON containers at an integration boundary."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw_json(item) for item in value]
    return value
