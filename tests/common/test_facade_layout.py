"""Pin the stable-facade shape for every module split into a private package.

`.agents/skills/write-clear-code/references/patterns.md` states the three rules
that make the pattern work: the facade holds only re-exports, the implementations
live in a sibling `_<facade>` package, and the split is recorded in the owning
package's README module map. The per-facade `test_*_layout.py` files assert that
individual public names are the split implementations; this file asserts the
property that makes those names trustworthy — nothing else can be defined in a
facade, so no public name can quietly become a second implementation.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_COMMON_ROOT = Path(__file__).resolve().parents[2] / "alphaapollo" / "common"
_README = _COMMON_ROOT / "README.md"

_FACADES = (
    "environment/default/environment.py",
    "execution/sandbox/podman.py",
    "execution/session.py",
    "execution/tools/builtins/files.py",
    "execution/tools/builtins/shell.py",
    "execution/workspace.py",
    "trajectory/recorder.py",
)


def _module_name(relative: str) -> str:
    return "alphaapollo.common." + relative.removesuffix(".py").replace("/", ".")


def _facade_body(relative: str) -> tuple[list[ast.stmt], list[str]]:
    """Return the statements that are not re-exports, and the exported names."""

    tree = ast.parse((_COMMON_ROOT / relative).read_text(encoding="utf-8"))
    other: list[ast.stmt] = []
    exported: list[str] = []
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.ImportFrom):
            continue
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.Assign) and [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ] == ["__all__"]:
            exported = [element.value for element in node.value.elts]  # type: ignore[attr-defined]
            continue
        other.append(node)
    return other, exported


@pytest.mark.parametrize("relative", _FACADES)
def test_facade_contains_only_reexports(relative: str) -> None:
    other, exported = _facade_body(relative)
    assert [f"{type(node).__name__} at line {node.lineno}" for node in other] == []
    assert exported, f"{relative} must declare __all__"


@pytest.mark.parametrize("relative", _FACADES)
def test_facade_exports_exactly_what_it_imports(relative: str) -> None:
    tree = ast.parse((_COMMON_ROOT / relative).read_text(encoding="utf-8"))
    imported = sorted(
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
    _, exported = _facade_body(relative)
    assert sorted(exported) == imported


@pytest.mark.parametrize("relative", _FACADES)
def test_facade_implementations_live_in_a_private_sibling_package(relative: str) -> None:
    module = importlib.import_module(_module_name(relative))
    private = "_" + Path(relative).stem
    owners = {
        getattr(getattr(module, name), "__module__", "")
        for name in module.__all__
        if hasattr(getattr(module, name), "__module__")
    }
    # A facade may also re-export a lower-level primitive that another module
    # canonically owns (``shell.py`` re-exports ``execution.output``); what must
    # not happen is a public name owned by the facade module itself.
    assert _module_name(relative) not in owners
    assert any(private in owner.split(".") for owner in owners), owners


@pytest.mark.parametrize("relative", _FACADES)
def test_facade_is_recorded_in_the_common_module_map(relative: str) -> None:
    assert relative in _README.read_text(encoding="utf-8")
