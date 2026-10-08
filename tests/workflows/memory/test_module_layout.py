# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pin the workflows/memory package boundary introduced by #406.

Two directions, matching the acceptance criteria on the issue: the public
``alphaapollo.workflows.memory`` entry point keeps resolving, and every legacy
flat-module path fails loudly instead of quietly resurrecting a second import
path for the same code.
"""

from __future__ import annotations

import importlib

import pytest

_LEGACY_MODULES = (
    "alphaapollo.workflows.bio_memory",
    "alphaapollo.workflows.robotics_memory",
    "alphaapollo.workflows.memory_adapter",
    "alphaapollo.workflows.unified_memory",
    "alphaapollo.workflows.verification_memory",
    "alphaapollo.workflows._memory_journal",
)

_PACKAGE_MODULES = (
    "alphaapollo.workflows.memory",
    "alphaapollo.workflows.memory.adapter",
    "alphaapollo.workflows.memory.unified",
    "alphaapollo.workflows.memory.robotics",
    "alphaapollo.workflows.memory.verification",
    "alphaapollo.workflows.memory._journal",
)


@pytest.mark.parametrize("module", _PACKAGE_MODULES)
def test_every_package_module_imports(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize("module", _LEGACY_MODULES)
def test_every_legacy_path_is_gone(module: str) -> None:
    """The removal is intentional: no shim, no lingering module file.

    A resurrected legacy path would be one implementation importable under two
    names -- two class objects, broken isinstance checks, and a second way to
    write every import this package just unified.
    """

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_the_public_entry_point_is_the_package() -> None:
    memory = importlib.import_module("alphaapollo.workflows.memory")
    assert memory.__file__ is not None and memory.__file__.endswith("__init__.py")


def test_math_memory_has_no_parallel_flat_module_path() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("alphaapollo.workflows.math_memory")
