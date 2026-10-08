"""Helpers for forwarding legacy backend modules to the sandbox package."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Any


def alias_module(legacy_name: str, canonical_name: str) -> ModuleType:
    """Make a legacy leaf-module import resolve to the canonical module object."""

    canonical = import_module(canonical_name)
    sys.modules[legacy_name] = canonical
    return canonical


def forward_module(namespace: dict[str, Any], canonical_name: str) -> None:
    """Expose one canonical sandbox module through a legacy module path."""

    canonical = import_module(canonical_name)
    exported = getattr(
        canonical,
        "__all__",
        [name for name in vars(canonical) if not name.startswith("_")],
    )

    def __getattr__(name: str) -> Any:
        return getattr(canonical, name)

    def __dir__() -> list[str]:
        return sorted(set(namespace) | set(dir(canonical)))

    namespace["__all__"] = list(exported)
    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__


# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
