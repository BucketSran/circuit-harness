from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _PROJECT_ROOT / "alphaapollo"
_COMMON_ROOT = _PACKAGE_ROOT / "common"
_DATA_PREPROCESS_ROOT = _PACKAGE_ROOT / "data_preprocess"
_REASONING_ROOT = _PACKAGE_ROOT / "reasoning"
_WORKFLOWS_ROOT = _PACKAGE_ROOT / "workflows"
_PRODUCTS = ("reasoning", "learning", "evolving")
_PRODUCT_PREFIXES = {f"alphaapollo.{name}" for name in _PRODUCTS}
_COMMON_FORBIDDEN = _PRODUCT_PREFIXES | {"alphaapollo.workflows"}
# Chips retains the shared contracts and the Robotics reference adapter.
_RETAINED_COMMON_FILES = {
    "environment/__init__.py",
    "environment/base.py",
    "environment/robotics/environment.py",
    "environment/provider.py",
    "environment/registry.py",
    "environment/default/__init__.py",
    "environment/default/environment.py",
    "environment/default/projection.py",
    "execution/__init__.py",
    # Sandbox execution now has a domain-specific canonical owner. Thin modules
    # under `execution/backends/` preserve the established import paths.
    "execution/sandbox/__init__.py",
    "execution/sandbox/base.py",
    "execution/sandbox/docker.py",
    "execution/sandbox/local.py",
    "execution/sandbox/manager.py",
    "execution/sandbox/podman.py",
    "execution/session.py",
    "execution/workspace.py",
    "execution/tools/__init__.py",
    "execution/tools/base.py",
    "execution/tools/gateway.py",
    "execution/tools/registry.py",
    "execution/tools/schemas.py",
    "execution/tools/builtins/__init__.py",
    "execution/tools/builtins/shell.py",
    "execution/tools/builtins/files.py",
    "trajectory/__init__.py",
    "trajectory/episode.py",
    # Added after #186: reducing a recorded trajectory into token/tool metrics is
    # a trajectory capability, not a Workflow-reporting one, and depends only
    # on `trajectory/schemas`.
    "trajectory/metrics.py",
    "trajectory/recorder.py",
    "trajectory/schemas.py",
    "trajectory/store.py",
}


def _module_package(path: Path, package_root: Path) -> str:
    relative = path.relative_to(package_root).with_suffix("")
    parts = [package_root.name, *relative.parts]
    if parts[-1] == "__init__":
        parts.pop()
    else:
        parts.pop()
    return ".".join(parts)


def _imports(path: Path, *, package_root: Path = _PACKAGE_ROOT) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = _module_package(path, package_root).split(".")
                keep = len(package) - (node.level - 1)
                base_parts = package[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                names.append(base)
                names.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Call):
            function = node.func
            is_dynamic_import = (
                isinstance(function, ast.Name)
                and function.id == "__import__"
                or isinstance(function, ast.Attribute)
                and function.attr == "import_module"
            )
            if (
                is_dynamic_import
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                dynamic_name = node.args[0].value
                if dynamic_name.startswith("."):
                    package_name = None
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                        package_name = node.args[1].value
                    for keyword in node.keywords:
                        if keyword.arg == "package" and isinstance(keyword.value, ast.Constant):
                            package_name = keyword.value.value
                    if isinstance(package_name, str):
                        dynamic_name = importlib.util.resolve_name(dynamic_name, package_name)
                names.append(dynamic_name)
    return names


def _find_violations(
    root: Path,
    forbidden: set[str],
    *,
    package_root: Path = _PACKAGE_ROOT,
) -> list[str]:
    violations: list[str] = []
    for path in root.rglob("*.py"):
        for name in _imports(path, package_root=package_root):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
                violations.append(f"{path.relative_to(package_root)}: {name}")
    return violations


def test_target_packages_are_real_and_importable() -> None:
    targets = (
        "alphaapollo.reasoning",
        "alphaapollo.learning",
        "alphaapollo.workflows",
        "alphaapollo.common.environment.default",
        "alphaapollo.common.artifacts",
        "alphaapollo.common.execution.sandbox",
        "alphaapollo.common.execution.tools",
        "alphaapollo.common.execution.tools.builtins",
        "alphaapollo.common.trajectory",
    )
    for name in targets:
        module = importlib.import_module(name)
        assert Path(module.__file__).name == "__init__.py"


def test_common_tree_contains_the_retained_shared_contract() -> None:
    actual = {
        path.relative_to(_COMMON_ROOT).as_posix()
        for root in ("environment", "execution", "trajectory")
        for path in (_COMMON_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    }
    missing = sorted(_RETAINED_COMMON_FILES - actual)
    assert missing == []


def test_robotics_source_provenance_is_retained() -> None:
    assert (_DATA_PREPROCESS_ROOT / "robotics" / "UPSTREAM.md").is_file()


def test_common_has_no_product_line_imports() -> None:
    assert _find_violations(_COMMON_ROOT, _COMMON_FORBIDDEN) == []


def test_data_preparation_has_no_runtime_or_workflow_imports() -> None:
    forbidden = {"alphaapollo.reasoning", "alphaapollo.workflows"}

    assert _find_violations(_DATA_PREPROCESS_ROOT, forbidden) == []


def test_boundary_parser_catches_alias_relative_and_dynamic_imports(
    tmp_path: Path,
) -> None:
    root = tmp_path / "alphaapollo"
    source = root / "common" / "probe.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "from alphaapollo import learning as le",
                "from .. import evolving",
                "from ..reasoning import obsolete_component",
                "import importlib",
                'importlib.import_module("alphaapollo.learning.adapters")',
                'importlib.import_module(".adapters", package="alphaapollo.learning")',
                '__import__("alphaapollo.evolving.memory")',
            ]
        ),
        encoding="utf-8",
    )

    imports = set(_imports(source, package_root=root))

    assert "alphaapollo.learning" in imports
    assert "alphaapollo.evolving" in imports
    assert "alphaapollo.reasoning" in imports
    assert "alphaapollo.reasoning.obsolete_component" in imports
    assert "alphaapollo.learning.adapters" in imports
    assert "alphaapollo.evolving.memory" in imports


def test_global_schema_package_is_removed() -> None:
    assert not (_PACKAGE_ROOT / "schema").exists()
    assert _find_violations(_PACKAGE_ROOT, {"alphaapollo.schema"}) == []


def test_product_lines_do_not_import_sibling_policy_or_workflows() -> None:
    for product in _PRODUCTS:
        forbidden = {prefix for prefix in _PRODUCT_PREFIXES if prefix != f"alphaapollo.{product}"}
        # Evaluation is an application package and may compose top-level
        # Workflows; Runtime/Verification remain the lower Reasoning layers.
        if product != "reasoning":
            forbidden.add("alphaapollo.workflows")
        violations = _find_violations(_PACKAGE_ROOT / product, forbidden)
        if product == "learning":
            # #201 defines one explicit composition seam: Learning owns the
            # training rollout pipeline and may consume Reasoning's *public*
            # Runtime contract there. Keep every other Learning -> Reasoning
            # dependency, and every dependency on Reasoning internals, forbidden.
            allowed_path = "learning/on_policy/rollout/reasoning_consumer.py"
            allowed_module = "alphaapollo.reasoning.runtime"
            violations = [
                violation
                for violation in violations
                if not (
                    violation.startswith(f"{allowed_path}: {allowed_module}")
                    and (
                        violation == f"{allowed_path}: {allowed_module}"
                        or violation.startswith(f"{allowed_path}: {allowed_module}.")
                    )
                )
            ]
        assert violations == []


def test_lower_layers_do_not_import_workflows() -> None:
    for root in (
        _COMMON_ROOT,
        _REASONING_ROOT / "runtime",
        _REASONING_ROOT / "verification",
        _PACKAGE_ROOT / "learning",
        _PACKAGE_ROOT / "evolving",
    ):
        assert _find_violations(root, {"alphaapollo.workflows"}) == []


def test_reasoning_and_workflows_have_no_training_or_gpu_imports() -> None:
    forbidden = {"torch", "ray", "tensordict", "verl", "vllm", "sglang"}
    assert _find_violations(_REASONING_ROOT, forbidden) == []
    assert _find_violations(_WORKFLOWS_ROOT, forbidden) == []


def test_reasoning_and_workflows_import_without_gpu_or_provider_packages() -> None:
    script = """
import sys
for name in ('openai', 'torch', 'ray', 'tensordict', 'verl', 'vllm', 'sglang'):
    sys.modules[name] = None
import alphaapollo.reasoning
import alphaapollo.reasoning.runtime
import alphaapollo.reasoning.verification
import alphaapollo.workflows
import alphaapollo.workflows.main
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_common_imports_without_gpu_or_provider_packages() -> None:
    # Also blocks every package Common reaches only behind an extra
    # (`environments`, `math`, `sokoban`, `data`): importing Common, listing the
    # environment registry, and resolving a grader must all work on a base
    # install. A task environment that needs one of them raises when its factory
    # is resolved, not when the package is imported.
    script = """
import sys
for name in (
    'docker', 'openai', 'torch', 'ray', 'verl', 'vllm', 'sglang',
    'requests', 'sympy', 'scipy', 'matplotlib', 'gym', 'gym_sokoban',
    'pyarrow', 'datasets',
):
    sys.modules[name] = None
import alphaapollo.common
import alphaapollo.common.artifacts
import alphaapollo.common.environment
import alphaapollo.common.execution
import alphaapollo.common.generation
import alphaapollo.common.grader
import alphaapollo.common.prompts
import alphaapollo.common.trajectory
assert alphaapollo.common.environment.available_environments()
assert alphaapollo.common.grader.available_graders()
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_chips_workflow_and_schema_tools_import_without_bio() -> None:
    """Shared startup must not depend on the removed Bio application stack."""
    script = """
import sys
for name in (
    'alphaapollo.common.grader.bio',
    'alphaapollo.common.execution.bio',
    'alphaapollo.workflows.memory.bio',
    'alphaapollo.data_preprocess.bio',
):
    sys.modules[name] = None
import alphaapollo.workflows.main
import alphaapollo.workflows.memory.unified
from alphaapollo._schema_export import check_snapshots
assert check_snapshots() == []
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_chips_and_robotics_catalogs_work_without_removed_domains() -> None:
    script = r"""
import sys
for name in (
    'alphaapollo.common.grader.math',
    'alphaapollo.common.execution.tools.math',
    'alphaapollo.common.environment.circle_packing',
    'alphaapollo.common.environment.alfworld',
    'alphaapollo.common.environment.search',
    'alphaapollo.common.environment.sokoban',
    'alphaapollo.common.environment.webshop',
    'alphaapollo.reasoning.verification.lean4',
    'alphaapollo.workflows.memory.math',
):
    sys.modules[name] = None
import alphaapollo.workflows.main
from alphaapollo.common.environment import available_environments
from alphaapollo.common.grader import available_graders, declared_answers_equivalent
from alphaapollo.common.execution.tools import PythonExecuteTool
from alphaapollo.common.execution.tools.robotics import ROBOTICS_TOOL_SPECS
from alphaapollo.reasoning.runtime.external.bridge.mcp_server import _tool_specs
assert available_environments() == ('default', 'robotics')
assert available_graders() == ('environment_success', 'exact_match')
assert declared_answers_equivalent('204', r'\frac{408}{2}')
assert PythonExecuteTool().tool_id == 'python_execute'
assert ROBOTICS_TOOL_SPECS
tool_ids = ('python_execute', 'emx_simulate')
assert tuple(spec.tool_id for spec in _tool_specs(tool_ids)) == tool_ids
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
