from __future__ import annotations

import asyncio
import gc
import importlib
import json
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from alphaapollo.reasoning.runtime import AgentResult
from alphaapollo.reasoning.verification import VerificationResult
from alphaapollo.workflows import data as workflow_data
from alphaapollo.workflows import executor as workflow_executor
from alphaapollo.workflows import resources as workflow_resources
from alphaapollo.workflows import run as workflow_run
from alphaapollo.workflows._resources import environment as environment_resources
from alphaapollo.workflows._resources import lifecycle as lifecycle_resources
from alphaapollo.workflows._resources import runtime as runtime_resources
from alphaapollo.workflows.config import ConfigError, ResourceConfig, RunConfig, parse_run_config
from alphaapollo.workflows.records import WorkflowInput

workflow_main = importlib.import_module("alphaapollo.workflows.main")


def test_default_environment_uses_stable_common_package_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    class TextOnlyToolBridge:
        pass

    class DefaultEnvironment:
        def __init__(self, **options: object) -> None:
            created.append(options)

        def close(self) -> None:
            return None

    module = ModuleType("alphaapollo.common.environment.default")
    module.DefaultEnvironment = DefaultEnvironment  # type: ignore[attr-defined]
    module.TextOnlyToolBridge = TextOnlyToolBridge  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    environment = workflow_resources._environment_factory(
        ResourceConfig(type="default", options={})
    )(object())
    environment.close()

    assert len(created) == 1
    assert isinstance(created[0]["tool_bridge"], TextOnlyToolBridge)  # type: ignore[index]


def test_execution_resources_preserves_the_pre_environment_factory_positional_api() -> None:
    owned = (object(),)

    resources = workflow_resources.ExecutionResources(
        SimpleNamespace(),
        {},
        {},
        owned,
        True,
    )

    assert resources._owned is owned
    assert resources._closed is True
    assert resources.environment_factory is None


def _run_mapping(tmp_path: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "workflow": {
            "version": 1,
            "name": "offline",
            "roles": [
                {
                    "id": "solver",
                    "target": "solver",
                    "system_prompt": "Solve the public problem.",
                }
            ],
            "steps": [
                {
                    "id": "solve",
                    "kind": "agent",
                    "role": "solver",
                    "output": True,
                }
            ],
            "entry_step": "solve",
            "transitions": [],
        },
        "dataset": {
            "path": "prepared.jsonl",
            "format": "jsonl",
            "input_key": "question",
            "id_key": "task_id",
            "metadata_keys": ["source"],
        },
        "runtimes": {
            "solver": {
                "type": "alphaapollo",
                "options": {
                    "backend": {
                        "type": "fake",
                        "options": {
                            "text": None,
                            "verdict": "pass",
                            "feedback": "accepted",
                        },
                    },
                    "model": "offline-model",
                    "sampling": {"temperature": 0.0, "max_tokens": 64},
                    "max_turns": 1,
                },
            }
        },
        "verifiers": {},
        "environment": {"type": "text_only", "options": {}},
        "output": {"directory": str(tmp_path / "output"), "format": "jsonl"},
    }


def _config(tmp_path: Path) -> RunConfig:
    return parse_run_config(_run_mapping(tmp_path), base_dir=tmp_path)


def test_run_workflow_reuses_memory_run_id_when_resuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _run_mapping(tmp_path)
    raw["memory"] = {"mode": "working", "profile": "full"}
    config = parse_run_config(raw, base_dir=tmp_path)
    run_ids: list[str] = []
    sessions: list[workflow_run.WorkflowMemorySession] = []
    memory_session = workflow_run.WorkflowMemorySession

    class RecordingMemorySession(memory_session):
        def __init__(self, *args: object, **kwargs: object) -> None:
            run_ids.append(str(kwargs["run_id"]))
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            sessions.append(self)

    monkeypatch.setattr(workflow_run, "WorkflowMemorySession", RecordingMemorySession)
    inputs = (WorkflowInput(input_id="aime-one", problem="Solve this problem."),)

    workflow_run.run_workflow(config, inputs=inputs)
    sessions[0].sync_scratchpad(
        input_id="aime-one",
        branch_index=0,
        step_id="solve",
        content="recover this note",
    )
    workflow_run.run_workflow(config, inputs=inputs)

    assert len(run_ids) == 2
    assert run_ids[0] == run_ids[1]
    assert sessions[1].scratchpad_content(input_id="aime-one", branch_index=0) == (
        "recover this note"
    )
    metadata = json.loads(
        (config.output.directory / "workflow-memory-run.json").read_text(encoding="utf-8")
    )
    assert metadata["run_id"] == run_ids[0]
    assert metadata["config_digest"] == workflow_run.config_digest(config)


def _solver_then_verifier(raw: dict[str, Any]) -> dict[str, Any]:
    """Reshape ``_run_mapping`` into ``solve -> verify`` and return the verifier role.

    The shortest fixture for verifier-role validation used to be a lone verifier
    step. ``WorkflowConfig`` refuses that topology because nothing in it produces
    the candidate a verifier judges, so the solver step added here is fixture
    scaffolding rather than part of any caller's subject.
    """

    workflow = raw["workflow"]
    role = workflow["roles"][0]
    role["id"] = "judge"
    workflow["roles"].append(
        {"id": "solver", "target": "solver", "system_prompt": "Solve the public problem."}
    )
    workflow["steps"][0].update(
        {"id": "verify", "kind": "verifier", "role": "judge", "candidate_from": "solve"}
    )
    workflow["steps"].insert(0, {"id": "solve", "kind": "agent", "role": "solver"})
    workflow["entry_step"] = "solve"
    workflow["transitions"] = [{"source": "solve", "target": "verify", "condition": "always"}]
    return role


def test_main_returns_two_for_argument_and_config_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert workflow_main.main([]) == 2
    argument_error = capsys.readouterr().err
    assert "--config" in argument_error
    assert "Traceback" not in argument_error

    raw = _run_mapping(tmp_path)
    raw["runtimes"]["solver"]["type"] = "dynamic_import"  # type: ignore[index]
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    created = False

    def must_not_build(*_args: object, **_kwargs: object) -> object:
        nonlocal created
        created = True
        raise AssertionError("resources must not be created for invalid config")

    monkeypatch.setattr(runtime_resources, "_build_runtime", must_not_build)
    assert workflow_main.main(["--config", str(config_path)]) == 2
    config_error = capsys.readouterr().err
    assert "configuration error" in config_error
    assert "Traceback" not in config_error
    assert created is False


def test_main_returns_two_for_prepared_dataset_errors_before_resource_creation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _run_mapping(tmp_path)
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "prepared.jsonl").write_text("not-json\n", encoding="utf-8")
    created = False

    def must_not_create(*_args: object, **_kwargs: object) -> object:
        nonlocal created
        created = True
        raise AssertionError("resources must not be created for invalid prepared input")

    monkeypatch.setattr(workflow_resources, "_environment_factory", must_not_create)

    assert workflow_main.main(["--config", str(config_path)]) == 2
    error = capsys.readouterr().err
    assert "configuration error" in error
    assert "invalid JSON on line 1" in error
    assert "Traceback" not in error
    assert created is False


@pytest.mark.parametrize(
    ("sampling_change", "message"),
    (
        ({"temperature": -0.1}, "temperature must be finite and non-negative"),
        ({"tool_choice": "auto"}, "tool_choice requires role tools"),
    ),
)
def test_sampling_config_errors_before_backend_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sampling_change: dict[str, object],
    message: str,
) -> None:
    raw = _run_mapping(tmp_path)
    sampling = raw["runtimes"]["solver"]["options"]["sampling"]  # type: ignore[index]
    sampling.update(sampling_change)
    config = parse_run_config(raw, base_dir=tmp_path)
    created = False

    def must_not_build(*_args: object, **_kwargs: object) -> object:
        nonlocal created
        created = True
        raise AssertionError("backend must not be created for invalid sampling config")

    monkeypatch.setattr(runtime_resources, "_build_backend", must_not_build)

    with pytest.raises(ConfigError, match=message):
        workflow_main.run_workflow(
            config,
            inputs=(WorkflowInput("one", "public problem", {"source": "safe"}),),
        )

    assert created is False


@pytest.mark.parametrize("step_kind", ["agent", "verifier"])
def test_internal_python_cannot_be_declared_as_a_native_role_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    step_kind: str,
) -> None:
    raw = _run_mapping(tmp_path)
    role = raw["workflow"]["roles"][0]  # type: ignore[index]
    if step_kind == "verifier":
        role = _solver_then_verifier(raw)
        role.update(
            {
                "target": "judge",
                "input_template": "{problem}\n{candidate}",
                "output_format": "json",
            }
        )
        raw["verifiers"] = {"judge": {"type": "agent", "options": {"runtime": "solver"}}}
    role["tools"] = ["python"]
    config = parse_run_config(raw, base_dir=tmp_path)
    created = False

    def must_not_create(*_args: object, **_kwargs: object) -> object:
        nonlocal created
        created = True
        raise AssertionError("resources must not be created for unresolved role tools")

    monkeypatch.setattr(workflow_resources, "_environment_factory", must_not_create)
    monkeypatch.setattr(runtime_resources, "_build_backend", must_not_create)

    with pytest.raises(ConfigError, match="tool 'python' is not a native model tool"):
        workflow_main.run_workflow(
            config,
            inputs=(WorkflowInput("one", "public problem", {"source": "safe"}),),
        )

    assert created is False


def test_native_role_tool_resolves_to_runtime_schema(tmp_path: Path) -> None:
    raw = _run_mapping(tmp_path)
    raw["workflow"]["roles"][0]["tools"] = ["bash"]  # type: ignore[index]
    raw["runtimes"]["solver"]["options"]["sampling"]["tool_choice"] = "auto"  # type: ignore[index]
    raw["environment"] = {"type": "default", "options": {"max_tool_calls": 2}}
    config = parse_run_config(raw, base_dir=tmp_path)

    verifier_specs = workflow_resources._validate_composition_config(config)
    schemas = workflow_resources._runtime_tool_schemas(config, verifier_specs, "solver")

    assert [schema["function"]["name"] for schema in schemas] == ["bash"]


def test_runtime_tool_grant_applies_only_to_roles_using_that_runtime(tmp_path: Path) -> None:
    raw = _run_mapping(tmp_path)
    raw["runtimes"]["solver"]["options"]["tools"] = ["bash"]  # type: ignore[index]
    raw["runtimes"]["solver"]["options"]["sampling"]["tool_choice"] = "auto"  # type: ignore[index]
    raw["environment"] = {"type": "default", "options": {"max_tool_calls": 2}}
    config = parse_run_config(raw, base_dir=tmp_path)

    verifier_specs = workflow_resources._validate_composition_config(config)
    workflow = runtime_resources._effective_workflow_config(config, verifier_specs)
    schemas = workflow_resources._runtime_tool_schemas(config, verifier_specs, "solver")

    assert config.workflow.roles[0].tools == ()
    assert workflow.roles[0].tools == ("bash",)
    assert [schema["function"]["name"] for schema in schemas] == ["bash"]


@pytest.mark.parametrize("tools", ["bash", ["bash", "bash"]])
def test_runtime_tool_grant_rejects_invalid_shape(
    tmp_path: Path,
    tools: object,
) -> None:
    raw = _run_mapping(tmp_path)
    raw["runtimes"]["solver"]["options"]["tools"] = tools  # type: ignore[index]
    config = parse_run_config(raw, base_dir=tmp_path)

    with pytest.raises(ConfigError, match="options.tools"):
        workflow_resources._validate_composition_config(config)


def test_native_role_tool_requires_tool_capable_environment(tmp_path: Path) -> None:
    raw = _run_mapping(tmp_path)
    raw["workflow"]["roles"][0]["tools"] = ["bash"]  # type: ignore[index]
    config = parse_run_config(raw, base_dir=tmp_path)

    with pytest.raises(ConfigError, match="environment.type='default'"):
        workflow_resources._validate_composition_config(config)


def test_default_environment_enables_explicit_python_code_compatibility() -> None:
    from alphaapollo.reasoning.runtime import AgentTask

    environment = workflow_resources._environment_factory(
        ResourceConfig(
            type="default",
            options={"enable_python_code": True, "max_tool_calls": 2},
        )
    )(AgentTask("one", "use Python", "1 + 1"))

    bridge = environment._environment._tool_bridge
    assert bridge._allowed_tool_ids == frozenset({"python"})
    assert bridge._max_tool_calls == 2
    environment.close()


def test_native_runtime_composes_public_python_with_builtin_tools(tmp_path: Path) -> None:
    from alphaapollo.reasoning.runtime import AgentTask

    raw = _run_mapping(tmp_path)
    raw["workflow"]["roles"][0]["tools"] = ["python_execute", "bash"]  # type: ignore[index]
    raw["environment"] = {"type": "default", "options": {"max_tool_calls": 3}}
    config = parse_run_config(raw, base_dir=tmp_path)

    verifier_specs = workflow_resources._validate_composition_config(config)
    schemas = workflow_resources._runtime_tool_schemas(config, verifier_specs, "solver")
    owned = workflow_resources._environment_factory(config.environment)(
        AgentTask(
            "one",
            "use Python",
            "compute",
            tools=("python_execute", "bash"),
        )
    )

    bridge = owned._environment._tool_bridge
    assert [schema["function"]["name"] for schema in schemas] == [
        "python_execute",
        "bash",
    ]
    assert bridge._allowed_tool_ids == frozenset({"python_execute", "bash"})
    assert "python_execute" in bridge._gateway.runtime._tools_by_id
    owned.close()


@pytest.mark.parametrize("tool_executor", ["auto", "gateway", 3])
def test_default_environment_rejects_unknown_tool_executor(tool_executor: object) -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        workflow_resources._validate_environment_resource(
            ResourceConfig(type="default", options={"tool_executor": tool_executor})
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("system_prompt", "", "system_prompt must be a non-empty string"),
        ("input_template", "", "input_template must be a non-empty string"),
        ("output_format", None, "output_format must be 'json' or 'verdict-line'"),
    ],
)
def test_agent_verifier_requires_consumable_role_fields_before_resource_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    raw = _run_mapping(tmp_path)
    role = _solver_then_verifier(raw)
    role.update(
        {
            "target": "judge",
            "system_prompt": "judge",
            "input_template": "{candidate}",
            "output_format": "json",
            field: value,
        }
    )
    raw["verifiers"] = {"judge": {"type": "agent", "options": {"runtime": "solver"}}}
    config = parse_run_config(raw, base_dir=tmp_path)
    created = False

    def must_not_create(*_args: object, **_kwargs: object) -> object:
        nonlocal created
        created = True
        raise AssertionError("resources must not be created for an invalid verifier role")

    monkeypatch.setattr(workflow_resources, "_environment_factory", must_not_create)
    monkeypatch.setattr(runtime_resources, "_build_backend", must_not_create)

    with pytest.raises(ConfigError, match=message):
        workflow_main.run_workflow(config, inputs=())

    assert created is False


def test_return_token_ids_must_be_bool_before_resource_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _run_mapping(tmp_path)
    raw["runtimes"]["solver"]["options"]["backend"] = {  # type: ignore[index]
        "type": "openai_compatible",
        "options": {
            "base_url": "http://localhost:8000/v1",
            "return_token_ids": "true",
        },
    }
    config = parse_run_config(raw, base_dir=tmp_path)
    created = False

    def must_not_create(*_args: object, **_kwargs: object) -> object:
        nonlocal created
        created = True
        raise AssertionError("resources must not be created for invalid backend options")

    monkeypatch.setattr(workflow_resources, "_environment_factory", must_not_create)
    monkeypatch.setattr(runtime_resources, "_build_backend", must_not_create)

    with pytest.raises(ConfigError, match="return_token_ids must be a bool"):
        workflow_main.run_workflow(
            config,
            inputs=(WorkflowInput("one", "public problem", {"source": "safe"}),),
        )

    assert created is False


def test_backend_max_concurrency_must_be_positive_int_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _run_mapping(tmp_path)
    raw["runtimes"]["solver"]["options"]["backend"] = {  # type: ignore[index]
        "type": "openai_compatible",
        "options": {
            "base_url": "http://localhost:8000/v1",
            "max_concurrency": True,
        },
    }
    config = parse_run_config(raw, base_dir=tmp_path)
    created = False

    def must_not_create(*_args: object, **_kwargs: object) -> object:
        nonlocal created
        created = True
        raise AssertionError("resources must not be created for invalid backend options")

    monkeypatch.setattr(workflow_resources, "_environment_factory", must_not_create)
    monkeypatch.setattr(runtime_resources, "_build_backend", must_not_create)

    with pytest.raises(ConfigError, match="max_concurrency must be a positive int"):
        workflow_main.run_workflow(
            config,
            inputs=(WorkflowInput("one", "public problem", {"source": "safe"}),),
        )
    assert created is False


def test_main_applies_repeated_safe_overrides(tmp_path: Path) -> None:
    raw = _run_mapping(tmp_path)
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "prepared.jsonl").write_text(
        json.dumps({"task_id": "one", "question": "1+1", "source": "unit"}) + "\n",
        encoding="utf-8",
    )

    status = workflow_main.main(
        [
            "--config",
            str(config_path),
            "--set",
            "runtimes.solver.options.backend.options.text=overridden",
            "--set",
            "output.format=json",
        ]
    )

    assert status == 0
    records = json.loads((tmp_path / "output" / "workflow_results.json").read_text())
    assert records[0]["output"]["final_text"] == "overridden"


def test_injected_fake_run_preserves_order_persists_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend_closes: list[object] = []
    environment_closes: list[object] = []
    wrapper_closes: list[object] = []
    original_backend_close = runtime_resources._FakeBackend.close
    original_environment_close = environment_resources._TextOnlyEnvironment.close
    original_wrapper_close = environment_resources._OwnedEnvironment.close

    def backend_close(resource: object) -> None:
        backend_closes.append(resource)
        original_backend_close(resource)

    def environment_close(resource: object) -> None:
        environment_closes.append(resource)
        original_environment_close(resource)

    def wrapper_close(resource: object) -> None:
        wrapper_closes.append(resource)
        original_wrapper_close(resource)

    monkeypatch.setattr(runtime_resources._FakeBackend, "close", backend_close)
    monkeypatch.setattr(environment_resources._TextOnlyEnvironment, "close", environment_close)
    monkeypatch.setattr(environment_resources._OwnedEnvironment, "close", wrapper_close)

    results = workflow_main.run_workflow(
        _config(tmp_path),
        inputs=(
            WorkflowInput("second", "second problem", {"source": "injected"}),
            WorkflowInput("first", "first problem", {"source": "injected"}),
        ),
    )

    assert [result.input_id for result in results] == ["second", "first"]
    assert [result.output.final_text for result in results] == [  # type: ignore[union-attr]
        "fake response: second problem",
        "fake response: first problem",
    ]
    result_lines = (tmp_path / "output" / "workflow_results.jsonl").read_text().splitlines()
    trajectory_lines = (tmp_path / "output" / "trajectories.jsonl").read_text().splitlines()
    persisted_results = [json.loads(line) for line in result_lines]
    assert [row["input_id"] for row in persisted_results] == ["second", "first"]
    assert all(row["workflow_result_schema_version"] == 1 for row in persisted_results)
    assert all("turns" not in row["output"] for row in persisted_results)
    assert all("turns" not in row["steps"][0]["output"] for row in persisted_results)
    assert all("final_text" not in row["steps"][0]["output"] for row in persisted_results)
    trajectories = [json.loads(line) for line in trajectory_lines]
    assert len(trajectories) == 2
    assert trajectories[0]["agent_trajectory_schema_version"] == 1
    assert persisted_results[0]["trajectory_refs"] == [trajectories[0]["task_id"]]
    assert trajectories[0]["turns"][0]["generation_request"]["messages"][1]["content"] == (
        "second problem"
    )
    assert "candidates" not in trajectories[0]["turns"][0]["generation_response"]
    assert len(backend_closes) == 1
    assert len({id(resource) for resource in environment_closes}) == 2
    assert len(wrapper_closes) == 2


def test_nested_metadata_and_provider_options_cross_runtime_boundary(tmp_path: Path) -> None:
    raw = _run_mapping(tmp_path)
    sampling = raw["runtimes"]["solver"]["options"]["sampling"]  # type: ignore[index]
    sampling["provider_options"] = {
        "response_format": {"type": "json_object"},
        "stop": ["END"],
    }
    config = parse_run_config(raw, base_dir=tmp_path)

    results = workflow_main.run_workflow(
        config,
        inputs=(
            WorkflowInput(
                "nested",
                "public problem",
                {"source": {"name": "prepared"}, "tags": ["math", {"level": 2}]},
            ),
        ),
    )

    assert results[0].status == "completed"
    persisted = json.loads((tmp_path / "output" / "workflow_results.jsonl").read_text())
    assert persisted["output"]["metadata"]["source"] == {"name": "prepared"}


def test_environment_task_payload_is_absent_from_persisted_results(tmp_path: Path) -> None:
    canary = "PRIVATE-PERSISTENCE-CANARY"

    results = workflow_main.run_workflow(
        _config(tmp_path),
        inputs=(
            WorkflowInput(
                "private-one",
                "public problem",
                {"source": "public"},
                task_payload={"answer": canary, "token": canary},
            ),
        ),
    )

    assert isinstance(results[0].output, AgentResult)
    assert canary not in repr(results[0])
    output = tmp_path / "output"
    durable = "".join(
        path.read_text(encoding="utf-8") for path in output.iterdir() if path.is_file()
    )
    assert canary not in durable
    assert "task_payload" not in durable


def test_generation_persistence_projection_does_not_reflect_unknown_attributes() -> None:
    response = SimpleNamespace(
        request_id="request-1",
        group_id="group-1",
        sample_id=0,
        content="public output",
        usage={"prompt_tokens": 0, "completion_tokens": 0},
        backend_metadata={},
        private_backend_handle="must-not-be-persisted",
    )

    record = workflow_data._generation_response_record(response)

    assert record["content"] == "public output"
    assert "private_backend_handle" not in record


def test_environment_observation_uses_explicit_to_dict_projection() -> None:
    class Observation:
        def to_dict(self) -> dict[str, object]:
            return {"kind": "model_output", "content": "public output"}

    transition = SimpleNamespace(
        observation=Observation(),
        reward=0.0,
        done=True,
        success=True,
        response_format_valid=True,
        env_action_valid=True,
        termination_reason="model_output",
        metadata={},
    )

    record = workflow_data._environment_transition_record(transition)

    assert record["observation"] == {"kind": "model_output", "content": "public output"}


def test_persistence_validates_all_records_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = workflow_main.run_workflow(
        _config(tmp_path),
        inputs=(WorkflowInput("one", "first problem"),),
    )
    output = tmp_path / "output"
    for path in output.iterdir():
        path.unlink()

    def invalid_trajectory(_result: object) -> dict[str, object]:
        raise TypeError("invalid trajectory")

    monkeypatch.setattr(workflow_data, "_agent_trajectory_record", invalid_trajectory)

    with pytest.raises(TypeError, match="invalid trajectory"):
        workflow_data.persist_results(_config(tmp_path), results)

    assert list(output.iterdir()) == []


def test_runtime_releases_environments_before_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references: list[weakref.ReferenceType[object]] = []
    original_init = environment_resources._OwnedEnvironment.__init__

    def track_environment(resource: object, environment: object) -> None:
        original_init(resource, environment)
        references.append(weakref.ref(resource))

    def assert_released(*_args: object, **_kwargs: object) -> None:
        gc.collect()
        assert references
        assert all(reference() is None for reference in references)

    monkeypatch.setattr(environment_resources._OwnedEnvironment, "__init__", track_environment)
    monkeypatch.setattr(workflow_run, "persist_results", assert_released)

    workflow_main.run_workflow(
        _config(tmp_path),
        inputs=(
            WorkflowInput("one", "first problem"),
            WorkflowInput("two", "second problem"),
        ),
    )


def test_partial_environment_construction_failure_closes_created_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = 0
    closed: list[object] = []
    original_init = environment_resources._TextOnlyEnvironment.__init__
    original_close = environment_resources._TextOnlyEnvironment.close

    def fail_second_init(resource: object) -> None:
        nonlocal created
        created += 1
        if created == 2:
            raise RuntimeError("environment construction failed")
        original_init(resource)

    def track_close(resource: object) -> None:
        closed.append(resource)
        original_close(resource)

    monkeypatch.setattr(environment_resources._TextOnlyEnvironment, "__init__", fail_second_init)
    monkeypatch.setattr(environment_resources._TextOnlyEnvironment, "close", track_close)

    with pytest.raises(RuntimeError, match="environment construction failed"):
        workflow_main.run_workflow(
            _config(tmp_path),
            inputs=(
                WorkflowInput("one", "first problem"),
                WorkflowInput("two", "second problem"),
            ),
        )

    assert created == 2
    assert len(closed) == 1


@pytest.mark.parametrize("preset_name", ["vanilla_reasoning.yaml", "vanilla_ensemble.yaml"])
def test_shipped_presets_inherit_run_config_runtime_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preset_name: str
) -> None:
    declared_answer = "Reasoning complete. Final answer: 42"
    raw = _run_mapping(tmp_path)
    preset = Path(workflow_main.__file__).parents[1] / "configs" / "preset" / preset_name
    raw["workflow"] = str(preset)
    solver_options = raw["runtimes"]["solver"]["options"]  # type: ignore[index]
    solver_options["model"] = "solver-from-run-config"
    if preset_name == "vanilla_ensemble.yaml":
        solver_options["backend"]["options"]["text"] = declared_answer
    raw["runtimes"]["verifier_runtime"] = {  # type: ignore[index]
        "type": "alphaapollo",
        "options": {
            "backend": {
                "type": "fake",
                "options": {"verdict": "pass", "feedback": "accepted"},
            },
            "model": "verifier-from-run-config",
            "sampling": {"temperature": 0.0, "max_tokens": 64},
            "max_turns": 1,
        },
    }
    raw["verifiers"] = {"verifier": {"type": "agent", "options": {"runtime": "verifier_runtime"}}}
    config = parse_run_config(raw, base_dir=tmp_path)
    captured: list[tuple[str, str]] = []
    original_generate_batch = runtime_resources._FakeBackend.generate_batch

    def capture_models(backend: object, requests: object) -> list[runtime_resources._FakeResponse]:
        batch = tuple(requests)  # type: ignore[arg-type]
        for request in batch:
            system = request.messages[0]["content"]
            captured.append((system, request.model))
        return original_generate_batch(backend, batch)

    monkeypatch.setattr(runtime_resources._FakeBackend, "generate_batch", capture_models)

    results = workflow_main.run_workflow(
        config,
        inputs=(WorkflowInput("one", "public problem"),),
    )

    expected_model_by_prompt = {
        role.system_prompt: (
            "solver-from-run-config" if role.target == "solver" else "verifier-from-run-config"
        )
        for role in config.workflow.roles
    }
    assert captured
    assert all(model == expected_model_by_prompt[prompt] for prompt, model in captured)
    if preset_name == "vanilla_ensemble.yaml":
        assert results[0].selected_branch_index == 0
        assert isinstance(results[0].output, AgentResult)
        assert results[0].output.final_text == declared_answer


def test_dataset_mapping_exposes_only_public_input_fields(tmp_path: Path) -> None:
    raw = _run_mapping(tmp_path)
    raw["dataset"] = {
        "path": "prepared.jsonl",
        "format": "jsonl",
        "input_key": "statement",
        "id_key": "task_uid",
        "metadata_keys": ["constraints", "domain", "prompt"],
    }
    (tmp_path / "prepared.jsonl").write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "task_uid": "a",
                        "statement": "public A",
                        "constraints": [],
                        "domain": "math",
                        "prompt": "",
                        "gold_answer": "secret A",
                        "source_uri": "https://example.invalid/answer-a",
                        "reference_solution": "private derivation A",
                    }
                ),
                json.dumps(
                    {
                        "task_uid": "b",
                        "statement": "public B",
                        "constraints": [],
                        "domain": "math",
                        "prompt": "",
                        "gold_answer": "secret B",
                        "source_uri": "https://example.invalid/answer-b",
                        "reference_solution": "private derivation B",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    config = parse_run_config(raw, base_dir=tmp_path)
    results = workflow_main.run_workflow(config)

    assert [result.input_id for result in results] == ["a", "b"]
    for result in results:
        output = result.steps[0].output
        assert isinstance(output, AgentResult)
        assert output.metadata["constraints"] == ()
        assert output.metadata["domain"] == "math"
        assert output.metadata["prompt"] == ""
        assert "gold_answer" not in output.metadata
        serialized = json.dumps(workflow_data._workflow_result_record(result))
        assert "secret" not in serialized
        assert "example.invalid" not in serialized
        assert "private derivation" not in serialized


def test_fake_agent_verifier_emits_valid_json_and_persists_trajectory(tmp_path: Path) -> None:
    # The solver step is what produces the ``{candidate}`` this verifier judges;
    # a verifier-only topology is refused by ``WorkflowConfig`` because it has no
    # such producer. The subject here is unchanged: the verifier's own turn is
    # located by task id below rather than assumed to be the only trajectory.
    raw = _run_mapping(tmp_path)
    raw["workflow"] = {
        "version": 1,
        "name": "verify",
        "roles": [
            {
                "id": "solver",
                "target": "solver",
                "system_prompt": "Solve the public problem.",
            },
            {
                "id": "judge",
                "target": "judge",
                "system_prompt": "Return a JSON verdict for this public candidate.",
                "model": "offline-model",
                "tools": [],
                "input_template": "Problem: {problem}\nCandidate: {candidate}",
                "output_format": "json",
            },
        ],
        "steps": [
            {"id": "solve", "kind": "agent", "role": "solver"},
            {
                "id": "verify",
                "kind": "verifier",
                "role": "judge",
                "candidate_from": "solve",
                "output": True,
            },
        ],
        "entry_step": "solve",
        "transitions": [{"source": "solve", "target": "verify", "condition": "always"}],
    }
    raw["runtimes"]["judge_runtime"] = raw["runtimes"]["solver"]  # type: ignore[index]
    raw["verifiers"] = {"judge": {"type": "agent", "options": {"runtime": "judge_runtime"}}}
    config = parse_run_config(raw, base_dir=tmp_path)

    result = workflow_main.run_workflow(
        config, inputs=(WorkflowInput("candidate", "public problem"),)
    )[0]

    assert isinstance(result.output, VerificationResult)
    assert result.output.verdict == "pass"
    assert result.output.agent_result is not None
    trajectories = [
        json.loads(line)
        for line in (tmp_path / "output" / "trajectories.jsonl").read_text().splitlines()
    ]
    verifier_trajectory = next(
        record for record in trajectories if record["task_id"] == result.output.agent_result.task_id
    )
    verifier_text = verifier_trajectory["turns"][0]["generation_response"]["content"]
    assert json.loads(verifier_text)["verdict"] == "pass"


def test_execution_failure_still_closes_owned_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[object] = []
    original_close = runtime_resources._FakeBackend.close

    def close(resource: object) -> None:
        closed.append(resource)
        original_close(resource)

    def fail(_self: object, _inputs: object) -> object:
        raise RuntimeError("executor failed")

    monkeypatch.setattr(runtime_resources._FakeBackend, "close", close)
    monkeypatch.setattr(workflow_executor.WorkflowExecutor, "run_batch", fail)

    with pytest.raises(RuntimeError, match="executor failed"):
        workflow_main.run_workflow(_config(tmp_path), inputs=(WorkflowInput("one", "problem"),))

    assert len(closed) == 1


@pytest.mark.parametrize("failure", [OSError("disk full"), asyncio.CancelledError()])
def test_persistence_failure_and_cancellation_close_all_owned_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    backend_closes: list[object] = []
    environment_closes: list[object] = []
    original_backend_close = runtime_resources._FakeBackend.close
    original_environment_close = environment_resources._TextOnlyEnvironment.close

    def backend_close(resource: object) -> None:
        backend_closes.append(resource)
        original_backend_close(resource)

    def environment_close(resource: object) -> None:
        environment_closes.append(resource)
        original_environment_close(resource)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(runtime_resources._FakeBackend, "close", backend_close)
    monkeypatch.setattr(environment_resources._TextOnlyEnvironment, "close", environment_close)
    if isinstance(failure, asyncio.CancelledError):
        monkeypatch.setattr(runtime_resources._FakeBackend, "generate_batch", fail)
    else:
        monkeypatch.setattr(workflow_run, "persist_results", fail)

    match = None if isinstance(failure, asyncio.CancelledError) else str(failure)
    with pytest.raises(type(failure), match=match):
        workflow_main.run_workflow(
            _config(tmp_path), inputs=(WorkflowInput("one", "public problem"),)
        )

    assert len(backend_closes) == 1
    assert len(environment_closes) == 1


def test_prepared_split_errors_before_resource_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _run_mapping(tmp_path)
    raw["dataset"]["split"] = "test"  # type: ignore[index]
    (tmp_path / "prepared.jsonl").write_text(
        json.dumps(
            {
                "task_id": "one",
                "question": "public problem",
                "source": "safe",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = parse_run_config(raw, base_dir=tmp_path)
    created = False

    def must_not_build(*_args: object, **_kwargs: object) -> object:
        nonlocal created
        created = True
        raise AssertionError("resources must not be created before input validation")

    monkeypatch.setattr(runtime_resources, "_build_runtime", must_not_build)

    with pytest.raises(ValueError, match="split field"):
        workflow_main.run_workflow(config)

    assert created is False


def test_json_top_level_split_does_not_require_redundant_record_labels(
    tmp_path: Path,
) -> None:
    raw = _run_mapping(tmp_path)
    raw["dataset"].update(  # type: ignore[union-attr]
        {"path": "prepared.json", "format": "json", "split": "test"}
    )
    (tmp_path / "prepared.json").write_text(
        json.dumps(
            {
                "train": [],
                "test": [
                    {
                        "task_id": "one",
                        "question": "public problem",
                        "source": "safe",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = parse_run_config(raw, base_dir=tmp_path)

    inputs = workflow_data.load_prepared_inputs(config)

    assert inputs == (WorkflowInput("one", "public problem", {"source": "safe"}),)


def test_task_payload_sidecar_uses_its_own_format_and_is_already_split_scoped(
    tmp_path: Path,
) -> None:
    raw = _run_mapping(tmp_path)
    raw["dataset"].update(  # type: ignore[union-attr]
        {
            "path": "public.json",
            "format": "json",
            "split": "test",
            "task_payload_path": "private.jsonl",
            "task_payload_format": "jsonl",
        }
    )
    (tmp_path / "public.json").write_text(
        json.dumps(
            {
                "train": [],
                "test": [
                    {
                        "task_id": "one",
                        "question": "pick up the cup",
                        "source": "libero",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # Prepared private split rows have no public ``split`` field.  The concrete
    # sidecar path selects the split, independently of the public JSON layout.
    (tmp_path / "private.jsonl").write_text(
        json.dumps(
            {
                "task_id": "one",
                "env_payload": json.dumps({"scene_seed": 7}),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = parse_run_config(raw, base_dir=tmp_path)

    inputs = workflow_data.load_prepared_inputs(config)

    assert inputs[0].task_payload == {"scene_seed": 7}


def test_fake_backend_enforces_batch_identity() -> None:
    request_type = type(
        "Request",
        (),
        {
            "request_id": "duplicate",
            "group_id": "group",
            "sample_id": 0,
            "messages": ({"role": "user", "content": "public"},),
        },
    )
    backend = runtime_resources._FakeBackend(
        verifier_formats={}, text="result", verdict="pass", feedback="accepted"
    )

    with pytest.raises(RuntimeError, match="duplicate generation request_id"):
        backend.generate_batch((request_type(), request_type()))


def test_build_backend_prefers_canonical_common_backend_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[object] = []

    class Client:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class CanonicalBackend:
        capabilities = SimpleNamespace(token_native=False)

        def __init__(self, **options: object) -> None:
            self.options = options
            self._client = Client()
            constructed.append(self)

        def generate_batch(self, requests: object) -> object:
            return requests

    monkeypatch.setattr(
        runtime_resources,
        "_canonical_openai_backend_type",
        lambda: CanonicalBackend,
    )

    backend = runtime_resources._build_backend(
        {
            "type": "openai_compatible",
            "options": {
                "base_url": "http://localhost:8000/v1",
                "api_key": "test-key",
                "max_retries": 2,
                "timeout": 12,
                "return_token_ids": True,
                "max_concurrency": 7,
            },
        },
        verifier_formats={},
    )

    assert isinstance(backend, runtime_resources._OwnedGenerationBackend)
    assert backend._backend is constructed[0]
    assert constructed[0].options == {  # type: ignore[attr-defined]
        "base_url": "http://localhost:8000/v1",
        "api_key": "test-key",
        "max_retries": 2,
        "timeout": 12.0,
        "return_token_ids": True,
        "max_concurrency": 7,
    }
    assert backend.generate_batch(("request",)) == ("request",)

    assert lifecycle_resources._shutdown_resources((backend,)) is None
    assert constructed[0]._client.close_calls == 1  # type: ignore[attr-defined]
    assert lifecycle_resources._shutdown_resources((backend,)) is None
    assert constructed[0]._client.close_calls == 1  # type: ignore[attr-defined]


def test_default_environment_projection_preserves_structured_tool_call_identity() -> None:
    response = SimpleNamespace(
        request_id="generation-1",
        content="",
        reasoning_content="inspect with tool",
        finish_reason="tool_calls",
        tool_calls=(
            SimpleNamespace(
                id="call-123",
                name="python",
                arguments='{"code":"print(1)"}',
            ),
        ),
        usage={"prompt_tokens": 4, "completion_tokens": 2},
        backend_metadata={"backend": "fake"},
    )

    action = environment_resources._provider_response_action(response)
    call = action["choices"][0]["message"]["tool_calls"][0]
    assert call["id"] == "call-123"
    assert call["function"]["arguments"] == '{"code":"print(1)"}'

    transition = SimpleNamespace(
        done=False,
        observation='{"ok":true}',
        metadata={"tool_request": {"call_id": "call-123"}},
    )
    continuation = environment_resources._default_continuation_messages(response, transition)
    assert continuation[0]["tool_calls"][0]["id"] == "call-123"
    assert continuation[1] == {
        "role": "tool",
        "tool_call_id": "call-123",
        "content": '{"ok":true}',
    }


def test_conflicting_roles_cannot_share_agent_verifier_target(tmp_path: Path) -> None:
    raw = _run_mapping(tmp_path)
    raw["workflow"] = {
        "version": 1,
        "name": "conflict",
        "roles": [
            # The solver is here only to satisfy the topology rule that some step
            # must produce the candidate; the subject is the two verifier roles
            # sharing one target with conflicting prompts.
            {
                "id": "solver",
                "target": "solver",
                "system_prompt": "Solve the public problem.",
            },
            {
                "id": "judge_one",
                "target": "judge",
                "system_prompt": "First prompt",
                "model": "m",
                "input_template": "{problem} {candidate}",
                "output_format": "json",
            },
            {
                "id": "judge_two",
                "target": "judge",
                "system_prompt": "Conflicting prompt",
                "model": "m",
                "input_template": "{problem} {candidate}",
                "output_format": "json",
            },
        ],
        "steps": [
            {"id": "solve", "kind": "agent", "role": "solver"},
            {"id": "one", "kind": "verifier", "role": "judge_one", "candidate_from": "solve"},
            {
                "id": "two",
                "kind": "verifier",
                "role": "judge_two",
                "candidate_from": "solve",
                "output": True,
            },
        ],
        "entry_step": "solve",
        "transitions": [
            {"source": "solve", "target": "one", "condition": "always"},
            {"source": "one", "target": "two", "condition": "always"},
        ],
    }
    raw["runtimes"]["judge_runtime"] = raw["runtimes"]["solver"]  # type: ignore[index]
    raw["verifiers"] = {"judge": {"type": "agent", "options": {"runtime": "judge_runtime"}}}
    config = parse_run_config(raw, base_dir=tmp_path)

    with pytest.raises(ConfigError, match="conflicting role"):
        workflow_main.run_workflow(config, inputs=())


def test_runtime_validation_accepts_and_bounds_image_history_messages(tmp_path: Path) -> None:
    options = _run_mapping(tmp_path)["runtimes"]["solver"]["options"]
    options["image_history_messages"] = 4
    runtime_resources._validate_runtime_resource(
        "solver", ResourceConfig(type="alphaapollo", options=options)
    )

    options["image_history_messages"] = 0
    with pytest.raises(ConfigError, match="image_history_messages"):
        runtime_resources._validate_runtime_resource(
            "solver", ResourceConfig(type="alphaapollo", options=options)
        )


def test_loader_joins_environment_task_payload_without_exposing_it_as_metadata(
    tmp_path: Path,
) -> None:
    raw = _run_mapping(tmp_path)
    raw["dataset"].update(  # type: ignore[union-attr]
        {
            "path": "public.jsonl",
            "format": "jsonl",
            "task_payload_path": "private.jsonl",
            "task_payload_key": "env_payload",
        }
    )
    (tmp_path / "public.jsonl").write_text(
        json.dumps(
            {
                "task_id": "one",
                "question": "pick up the cup",
                "source": "libero",
                "source_id": "libero",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "private.jsonl").write_text(
        json.dumps(
            {
                "task_id": "one",
                "env_payload": json.dumps(
                    {
                        "benchmark": "libero",
                        "environment_version": "libero-v1.0.1",
                        "suite": "libero_spatial",
                        "task_index": 3,
                        "initial_state_index": 1,
                        "answer": "PRIVATE-LOADER-CANARY",
                    }
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = parse_run_config(raw, base_dir=tmp_path)

    inputs = workflow_data.load_prepared_inputs(config)

    assert inputs[0].metadata == {"source": "libero"}
    # The direct envelope passes the private payload through verbatim; only
    # the whitelisted metadata keys are ever renderable into prompts.
    assert inputs[0].task_payload == {
        "benchmark": "libero",
        "environment_version": "libero-v1.0.1",
        "suite": "libero_spatial",
        "task_index": 3,
        "initial_state_index": 1,
        "answer": "PRIVATE-LOADER-CANARY",
    }
    assert "PRIVATE-LOADER-CANARY" not in repr(inputs[0])


def test_loader_shapes_the_robot_task_envelope_without_exposing_it_as_metadata(
    tmp_path: Path,
) -> None:
    raw = _run_mapping(tmp_path)
    raw["dataset"].update(  # type: ignore[union-attr]
        {
            "path": "public.jsonl",
            "format": "jsonl",
            "task_payload_path": "private.jsonl",
            "task_payload_key": "env_payload",
            "task_payload_envelope": "robot_task",
        }
    )
    (tmp_path / "public.jsonl").write_text(
        json.dumps(
            {
                "task_id": "one",
                "question": "pick up the cup",
                "source": "libero",
                "source_id": "libero",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "private.jsonl").write_text(
        json.dumps(
            {
                "task_id": "one",
                "env_payload": json.dumps(
                    {
                        "benchmark": "libero",
                        "environment_version": "libero-v1.0.1",
                        "suite": "libero_spatial",
                        "task_index": 3,
                        "initial_state_index": 1,
                    }
                ),
                "answer": "must not cross",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = parse_run_config(raw, base_dir=tmp_path)

    inputs = workflow_data.load_prepared_inputs(config)

    assert inputs[0].metadata == {"source": "libero"}
    assert inputs[0].task_payload == {
        "instruction": "pick up the cup",
        "benchmark": "libero",
        "environment_version": "libero-v1.0.1",
        "backend_metadata": {
            "suite": "libero_spatial",
            "task_index": 3,
            "initial_state_index": 1,
        },
    }
    assert "answer" not in repr(inputs[0])
