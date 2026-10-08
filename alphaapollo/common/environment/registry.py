# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"Environment registration and configuration helpers."

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from numbers import Real
from typing import Any, TypeVar

from alphaapollo.common.environment.base import BaseEnvironment

ValueT = TypeVar("ValueT")


def value_or_default(configured: ValueT | None, default: ValueT) -> ValueT:
    """Use a default only when the configured value is actually absent."""

    return default if configured is None else configured


def boolean(value: Any, *, name: str) -> bool:
    """Require a real boolean instead of applying Python truthiness."""

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false; got {value!r}")
    return value


def finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    """Convert a finite float and enforce an optional lower bound."""

    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number; got {value!r}") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    if minimum is not None:
        violates_bound = converted <= minimum if exclusive_minimum else converted < minimum
        if violates_bound:
            relation = "greater than" if exclusive_minimum else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}; got {converted!r}")
    return converted


def integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    """Convert an integer without silently truncating real-valued inputs."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    if isinstance(value, Real) and not float(value).is_integer():
        raise ValueError(f"{name} must be an integer; got {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    if minimum is not None and converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {converted!r}")
    return converted


def nonempty_string(value: Any, *, name: str) -> str:
    """Convert a non-empty string, rejecting explicit empty values."""

    converted = str(value).strip()
    if not converted:
        raise ValueError(f"{name} must not be empty")
    return converted


def string_choice(value: Any, *, name: str, choices: set[str]) -> str:
    """Return a normalized non-empty string from an allowed set."""

    converted = nonempty_string(value, name=name).lower()
    if converted not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {allowed}; got {converted!r}")
    return converted


EnvironmentFactory = Callable[..., BaseEnvironment[Any, Any]]

_BUILTINS = {
    "default": "alphaapollo.common.environment.default:DefaultEnvironment",
    "robotics": "alphaapollo.common.environment.robotics:RobotEnvironment",
}
_FACTORIES: dict[str, EnvironmentFactory | str] = dict(_BUILTINS)


def register_environment(name: str, factory: EnvironmentFactory, *, replace: bool = False) -> None:
    """Register a task environment without importing Learning."""

    key = _normalize_name(name)
    if key in _FACTORIES and not replace:
        raise ValueError(f"environment {key!r} is already registered")
    if not callable(factory):
        raise TypeError("environment factory must be callable")
    _FACTORIES[key] = factory


def get_environment_factory(name: str) -> EnvironmentFactory:
    key = _normalize_name(name)
    try:
        factory = _FACTORIES[key]
    except KeyError as exc:
        available = available_environments()
        raise KeyError(f"unknown environment {key!r}; available: {available}") from exc
    if isinstance(factory, str):
        module_name, attribute = factory.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
        _FACTORIES[key] = factory
    return factory


def create_environment(name: str, /, **kwargs: Any) -> BaseEnvironment[Any, Any]:
    environment = get_environment_factory(name)(**kwargs)
    if not isinstance(environment, BaseEnvironment):
        raise TypeError(f"factory for {name!r} did not return BaseEnvironment")
    return environment


def available_environments() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("environment name must be non-empty")
    return name.strip().casefold()
