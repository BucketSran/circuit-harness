"""Public composition and ownership contracts across the shared resource split."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from alphaapollo.workflows import resources
from alphaapollo.workflows._resources import lifecycle
from alphaapollo.workflows._resources import runtime as runtime_resources
from alphaapollo.workflows._resources import verifier as verifier_resources
from alphaapollo.workflows.config import ConfigError, ResourceConfig, RunConfig, parse_run_config
from alphaapollo.workflows.records import Workflow


@pytest.mark.parametrize(
    "module,name",
    (
        ("config", "ConfigError"),
        ("records", "Workflow"),
        ("executor", "WorkflowExecutor"),
        ("run", "run_workflow"),
    ),
)
def test_lazy_workflow_exports_keep_public_identity(module, name):
    import alphaapollo.workflows as workflows

    assert name in dir(workflows) and name in workflows.__all__
    assert getattr(workflows, name) is getattr(
        importlib.import_module("alphaapollo.workflows." + module), name
    )
    with pytest.raises(AttributeError):
        _ = workflows.nonexistent_public_contract


@pytest.fixture
def config(tmp_path: Path) -> RunConfig:
    return parse_run_config(
        {
            "version": 1,
            "workflow": {
                "version": 1,
                "name": "resource-contract",
                "roles": [
                    {"id": "solver", "target": "a", "system_prompt": "Solve."},
                    {
                        "id": "judge",
                        "target": "check",
                        "system_prompt": "Verify.",
                        "input_template": "{problem}\n{candidate}",
                        "output_format": "json",
                    },
                ],
                "steps": [
                    {"id": "solve", "kind": "agent", "role": "solver"},
                    {"id": "verify", "kind": "verifier", "role": "judge", "output": True},
                ],
                "entry_step": "solve",
                "transitions": [{"source": "solve", "target": "verify"}],
            },
            "dataset": {"path": "unused.jsonl", "input_key": "question"},
            "runtimes": {
                name: {
                    "type": "alphaapollo",
                    "options": {"backend": {"type": "fake"}, "model": "offline"},
                }
                for name in ("a", "b")
            },
            "verifiers": {"check": {"type": "agent", "options": {"runtime": "b"}}},
            "environment": {"type": "text_only"},
            "output": {"directory": str(tmp_path / "output")},
        },
        base_dir=tmp_path,
    )


def test_public_resources_use_the_canonical_lifecycle_type(config: RunConfig) -> None:
    assert resources.ExecutionResources is lifecycle.ExecutionResources
    with resources.compose_resources(config) as composed:
        assert type(composed) is resources.ExecutionResources
        assert callable(composed.environment_factory)


@pytest.mark.parametrize("name", ["runtime", "verifier", "environment", "lifecycle"])
def test_shared_resource_modules_do_not_import_the_public_composer(name: str) -> None:
    module = importlib.import_module(f"alphaapollo.workflows._resources.{name}")
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = importlib.util.resolve_name("." * node.level + base, module.__package__)
            imported.add(base)
            imported.update(f"{base}.{alias.name}" for alias in node.names)
    assert "alphaapollo.workflows.resources" not in imported


@pytest.mark.parametrize("invalid_part", ["runtime", "verifier", "cross_resource"])
def test_public_composition_validates_before_any_resource_construction(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch, invalid_part: str
) -> None:
    if invalid_part == "runtime":
        config = replace(
            config, runtimes={**config.runtimes, "b": ResourceConfig(type="unsupported")}
        )
        message = "runtime 'b'.type"
    elif invalid_part == "verifier":
        config = replace(
            config,
            verifiers={"check": ResourceConfig(type="agent", options={"runtime": "missing"})},
        )
        message = "references unknown runtime"
    else:
        config = replace(config, environment=ResourceConfig(type="none"))
        message = "these need an Environment"

    def must_not_construct(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid composition reached resource construction")

    monkeypatch.setattr(resources, "_environment_factory", must_not_construct)
    monkeypatch.setattr(runtime_resources, "_build_runtime", must_not_construct)
    monkeypatch.setattr(verifier_resources, "_build_verifier", must_not_construct)
    with pytest.raises(ConfigError, match=message):
        resources.compose_resources(config)


@pytest.mark.parametrize("failure_at", ["second_runtime", "verifier"])
def test_partial_composition_releases_dependents_first_and_preserves_the_original_error(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    events: list[str] = []
    construction_error = RuntimeError("resource construction failed")

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(self.name)
            if self.name == "runtime-a":
                raise OSError("cleanup also failed")

    def build_runtime(name: str, *args: object, **kwargs: object) -> tuple[object, object]:
        if name == "b" and failure_at == "second_runtime":
            raise construction_error
        return Resource(f"runtime-{name}"), Resource(f"backend-{name}")

    def build_verifier(*args: object, **kwargs: object) -> None:
        raise construction_error

    monkeypatch.setattr(runtime_resources, "_build_runtime", build_runtime)
    monkeypatch.setattr(verifier_resources, "_build_verifier", build_verifier)
    with pytest.raises(RuntimeError) as raised:
        resources.compose_resources(config)

    assert raised.value is construction_error
    expected = ["runtime-a", "backend-a"]
    if failure_at == "verifier":
        expected = ["runtime-b", "backend-b", *expected]
    assert events == expected


def test_owned_aliases_close_once_and_borrowed_resources_remain_open(config: RunConfig) -> None:
    events: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def terminate(self, reason: str) -> None:
            events.append(f"{self.name}:terminate:{reason}")

        def close(self) -> None:
            events.append(f"{self.name}:close")

    owned = Resource("owned")
    borrowed = Resource("borrowed")
    composed = resources.ExecutionResources(
        workflow=Workflow.from_config(config.workflow),
        runtimes={"borrowed": borrowed},
        verifiers={},
        _owned=(owned, owned),
    )
    composed.close()
    composed.close()

    assert events == ["owned:terminate:workflow_shutdown", "owned:close"]
