# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicitly registered Verifier factories selected by Workflow recipes.

Registration is an application decision: a run configuration may select a
known name, but it cannot import an arbitrary Python object.  Factories receive
only their JSON-safe ``options.config`` mapping and must return a fresh Verifier
for each composed run because Verifiers may retain round-local state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from alphaapollo.reasoning.verification.base import Verifier

__all__ = [
    "CustomVerifierFactory",
    "available_custom_verifiers",
    "create_custom_verifier",
    "get_custom_verifier_factory",
    "register_custom_verifier",
]

CustomVerifierFactory = Callable[[Mapping[str, Any]], Verifier]

_FACTORIES: dict[str, CustomVerifierFactory] = {}


def register_custom_verifier(
    name: str,
    factory: CustomVerifierFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one trusted factory, refusing silent replacement by default."""

    key = _name(name)
    if not callable(factory):
        raise TypeError("custom Verifier factory must be callable")
    if key in _FACTORIES and not replace:
        raise ValueError(f"custom Verifier {key!r} is already registered")
    _FACTORIES[key] = factory


def get_custom_verifier_factory(name: str) -> CustomVerifierFactory:
    """Resolve a registered factory without constructing its Verifier."""

    key = _name(name)
    try:
        return _FACTORIES[key]
    except KeyError:
        raise ValueError(
            f"unknown custom Verifier {key!r}; available: {sorted(_FACTORIES)}"
        ) from None


def create_custom_verifier(
    name: str,
    config: Mapping[str, Any] | None = None,
) -> Verifier:
    """Create a fresh registered Verifier from one JSON-safe config mapping."""

    if config is not None and not isinstance(config, Mapping):
        raise TypeError("custom Verifier config must be a mapping")
    verifier = get_custom_verifier_factory(name)(dict(config or {}))
    if not isinstance(verifier, Verifier):
        raise TypeError(
            f"factory for custom Verifier {_name(name)!r} returned "
            f"{type(verifier).__name__}; expected Verifier"
        )
    return verifier


def available_custom_verifiers() -> tuple[str, ...]:
    """Return registered names in deterministic order."""

    return tuple(sorted(_FACTORIES))


def _name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("custom Verifier name must be a non-empty string")
    return value.strip()
