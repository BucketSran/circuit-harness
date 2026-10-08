"""Robotics-specific workflow composition tests (split out of test_main)."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alphaapollo.common.execution.workspace import ArtifactStore
from alphaapollo.reasoning.runtime import AgentResult
from alphaapollo.workflows import data as workflow_data
from alphaapollo.workflows import resources as workflow_resources
from alphaapollo.workflows._resources import runtime as runtime_resources
from alphaapollo.workflows.config import ConfigError, parse_run_config
from alphaapollo.workflows.records import WorkflowInput

workflow_main = importlib.import_module("alphaapollo.workflows.main")


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


def _robotics_options() -> dict[str, Any]:
    return {
        "backend": {
            "type": "fake",
            "options": {
                "initial_observation": {"state": {"frame": 0}},
                "transitions": [
                    {
                        "observation": {"state": {"frame": 1}},
                        "steps_used": 1,
                        "terminated": True,
                        "truncated": False,
                        "success": True,
                        "termination_reason": "task_success",
                        "info": {"oracle": "fake-environment"},
                    }
                ],
            },
        },
        "providers": {
            "vla": {
                "type": "fake",
                "options": {
                    "actions": [
                        {
                            "kind": "continuous",
                            "arguments": {"values": [0.1, -0.2]},
                            "provenance": {"provider": "fake-vla"},
                        }
                    ]
                },
            },
            "perception": {"type": "fake", "options": {"items": []}},
        },
        "max_turns": 2,
        "max_episode_steps": 4,
    }


def test_robotics_resources_run_closed_loop_through_generic_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _run_mapping(tmp_path)
    raw["workflow"]["roles"][0]["tools"] = ["vla_act", "finish"]
    raw["environment"] = {"type": "robotics", "options": _robotics_options()}
    config = parse_run_config(raw, base_dir=tmp_path)
    original = runtime_resources._FakeBackend._generate_one

    def vla_call(backend: object, request: object) -> object:
        response = original(backend, request)  # type: ignore[arg-type]
        return replace(
            response,
            content="",
            finish_reason="tool_calls",
            tool_calls=(
                SimpleNamespace(
                    id="vla-call-1",
                    name="vla_act",
                    arguments=json.dumps({"instruction": "pick up the cup"}),
                ),
            ),
        )

    monkeypatch.setattr(runtime_resources._FakeBackend, "_generate_one", vla_call)

    results = workflow_main.run_workflow(
        config,
        inputs=(
            WorkflowInput(
                input_id="libero-one",
                problem="pick up the cup",
                task_payload={
                    "instruction": "pick up the cup",
                    "benchmark": "libero",
                    "environment_version": "libero-v1.0.1",
                    "backend_metadata": {
                        "suite": "libero_spatial",
                        "task": 3,
                        "seed": 7,
                    },
                },
            ),
        ),
    )

    assert len(results) == 1
    assert isinstance(results[0].output, AgentResult)
    turn = results[0].output.turns[0]
    transition = turn.environment_transition
    payload = json.loads(transition.observation)
    assert transition.done is True
    assert transition.success is True
    assert transition.termination_reason == "task_success"
    assert payload["environment_success"] is True
    assert payload["observation"]["state"]["frame"] == 1
    assert payload["tool"]["ok"] is True
    assert payload["tool"]["tool_id"] == "vla_act"
    assert payload["tool"]["exit_code"] == 0
    assert turn.generation_response.tool_calls[0].name == "vla_act"


def test_robotics_resource_construction_failure_closes_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []
    original = workflow_resources._fake_robot_backend

    def tracked_backend(value: object) -> object:
        backend = original(value)
        created.append(backend)
        return backend

    def fail_vla(_value: object) -> object:
        raise RuntimeError("VLA provider construction failed")

    monkeypatch.setattr(workflow_resources, "_fake_robot_backend", tracked_backend)
    monkeypatch.setattr(workflow_resources, "_fake_vla_provider", fail_vla)

    with pytest.raises(RuntimeError, match="VLA provider construction failed"):
        workflow_resources._robotics_environment(
            SimpleNamespace(tools=()),
            _robotics_options(),
        )

    assert len(created) == 1
    with pytest.raises(RuntimeError, match="fake backend is closed"):
        created[0].reset(object())  # type: ignore[attr-defined]


def test_libero_resource_composition_wires_artifact_store_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _robotics_options()
    options["artifact_root"] = str(tmp_path / "artifacts")
    options["backend"] = {
        "type": "libero",
        "options": {"camera_height": 128, "camera_width": 128},
    }
    workflow_resources._validate_robotics_environment_options(options)

    captured: dict[str, object] = {}

    def build_backend(value: object, *, artifact_store: object | None = None) -> object:
        captured["artifact_store"] = artifact_store
        return workflow_resources._fake_robot_backend(_robotics_options()["backend"])

    monkeypatch.setattr(workflow_resources, "_libero_robot_backend", build_backend)
    environment = workflow_resources._robotics_environment(
        SimpleNamespace(tools=()),
        options,
    )
    try:
        store = captured["artifact_store"]
        artifact = store.put(  # type: ignore[union-attr]
            b"frame",
            type_="image/png",
            created_by="test",
        )
        assert environment._load_artifact(store.ref(artifact)) == b"frame"  # type: ignore[union-attr]
    finally:
        environment.close()


def test_libero_resource_validation_requires_artifact_root() -> None:
    options = _robotics_options()
    options["backend"] = {
        "type": "libero",
        "options": {"camera_height": 128},
    }

    with pytest.raises(ConfigError, match="artifact_root is required"):
        workflow_resources._validate_robotics_environment_options(options)


def test_libero_resource_validation_rejects_transport_options() -> None:
    options = _robotics_options()
    options["artifact_root"] = "runs/test/artifacts"
    options["backend"] = {
        "type": "libero",
        "options": {"endpoint": "http://libero.test"},
    }

    with pytest.raises(ConfigError, match="endpoint"):
        workflow_resources._validate_robotics_environment_options(options)


def test_libero_resource_validation_checks_recording_options() -> None:
    options = _robotics_options()
    options["artifact_root"] = "runs/test/artifacts"
    options["backend"] = {
        "type": "libero",
        "options": {"record_dir": "runs/test/rollout", "record_fps": 20},
    }
    workflow_resources._validate_robotics_environment_options(options)

    options["backend"]["options"] = {"record_dir": " "}
    with pytest.raises(ConfigError, match="record_dir"):
        workflow_resources._validate_robotics_environment_options(options)

    options["backend"]["options"] = {"record_fps": 0}
    with pytest.raises(ConfigError, match="record_fps"):
        workflow_resources._validate_robotics_environment_options(options)


@pytest.mark.parametrize("backend_type", ["libero", "robocasa"])
def test_enable_depth_reaches_the_backend_that_owns_the_geometry_tools(
    backend_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`back_project` is grantable, so a config must be able to turn depth on.

    Without this the geometry tools land inert: the model can be granted
    `back_project` and every call answers "metric depth is disabled".
    """

    options = _robotics_options()
    options["artifact_root"] = "runs/test/artifacts"
    options["backend"] = {"type": backend_type, "options": {"enable_depth": True}}
    workflow_resources._validate_robotics_environment_options(options)

    captured: dict[str, object] = {}

    class _Recorder:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    module = "alphaapollo.common.execution.robotics.backends"
    name = "LiberoBackend" if backend_type == "libero" else "RoboCasaBackend"
    monkeypatch.setattr(f"{module}.{name}", _Recorder)
    builder = getattr(workflow_resources, f"_{backend_type}_robot_backend")
    builder(options["backend"], artifact_store=ArtifactStore(tmp_path / "artifacts"))

    assert captured["enable_depth"] is True


@pytest.mark.parametrize("backend_type", ["libero", "robocasa"])
def test_enable_depth_refuses_a_value_that_is_not_a_boolean(backend_type: str) -> None:
    options = _robotics_options()
    options["artifact_root"] = "runs/test/artifacts"
    options["backend"] = {"type": backend_type, "options": {"enable_depth": "yes"}}

    with pytest.raises(ConfigError, match="enable_depth must be a boolean"):
        workflow_resources._validate_robotics_environment_options(options)


def test_mcp_provider_resources_are_created_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alphaapollo.common.execution.tools.robotics import mcp as mcp_adapters

    options = _robotics_options()
    clients: list[object] = []

    class FakeClient:
        def __init__(self, endpoint: str, *, default_timeout_s: float) -> None:
            self.endpoint = endpoint
            self.default_timeout_s = default_timeout_s
            self.closed = False
            clients.append(self)

        def call_tool(
            self,
            _name: str,
            _arguments: object,
            *,
            timeout_s: float | None = None,
        ) -> object:
            return {"timeout_s": timeout_s}

        def close(self) -> None:
            self.closed = True

    options["providers"] = {
        "vla": {
            "type": "mcp",
            "options": {
                "endpoint": "http://vla.test/mcp",
                "provider": "vla-service",
                "timeout_s": 11,
            },
        },
        "perception": {
            "type": "mcp",
            "options": {
                "endpoint": "http://perception.test/mcp",
                "provider": "perception-service",
            },
        },
    }
    workflow_resources._validate_robotics_environment_options(options)
    monkeypatch.setattr(mcp_adapters, "StreamableHTTPMCPClient", FakeClient)

    environment = workflow_resources._robotics_environment(SimpleNamespace(tools=()), options)
    environment.close()

    assert [client.endpoint for client in clients] == [
        "http://vla.test/mcp",
        "http://perception.test/mcp",
    ]
    assert [client.closed for client in clients] == [True, True]


def test_robotics_episode_recovers_from_a_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-episode provider fault is episode data, not an episode-ending error."""

    raw = _run_mapping(tmp_path)
    raw["workflow"]["roles"][0]["tools"] = ["vla_act", "finish"]
    raw["runtimes"]["solver"]["options"]["max_turns"] = 3
    raw["environment"] = {"type": "robotics", "options": _robotics_options()}
    config = parse_run_config(raw, base_dir=tmp_path)

    original_generate = runtime_resources._FakeBackend._generate_one
    counter = {"calls": 0}

    def vla_call(backend: object, request: object) -> object:
        response = original_generate(backend, request)  # type: ignore[arg-type]
        counter["calls"] += 1
        return replace(
            response,
            content="",
            finish_reason="tool_calls",
            tool_calls=(
                SimpleNamespace(
                    id=f"vla-call-{counter['calls']}",
                    name="vla_act",
                    arguments=json.dumps({"instruction": "pick up the cup"}),
                ),
            ),
        )

    monkeypatch.setattr(runtime_resources._FakeBackend, "_generate_one", vla_call)

    class _FlakyVLAProvider:
        def __init__(self, inner: object) -> None:
            self._inner = inner
            self._calls = 0

        def predict(self, request: object) -> object:
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("VLA service unreachable")
            return self._inner.predict(request)  # type: ignore[attr-defined]

        def close(self) -> None:
            close = getattr(self._inner, "close", None)
            if callable(close):
                close()

    original_provider = workflow_resources._fake_vla_provider
    monkeypatch.setattr(
        workflow_resources,
        "_fake_vla_provider",
        lambda value: _FlakyVLAProvider(original_provider(value)),
    )

    results = workflow_main.run_workflow(
        config,
        inputs=(
            WorkflowInput(
                input_id="libero-recovery",
                problem="pick up the cup",
                task_payload={
                    "instruction": "pick up the cup",
                    "benchmark": "libero",
                    "environment_version": "libero-v1.0.1",
                    "backend_metadata": {"suite": "libero_spatial", "task": 3, "seed": 7},
                },
            ),
        ),
    )

    turns = results[0].output.turns
    assert len(turns) == 2

    failed = json.loads(turns[0].environment_transition.observation)
    assert failed["tool"]["ok"] is False
    assert "VLA service unreachable" in json.dumps(failed["tool"])
    assert turns[0].environment_transition.done is False
    assert failed["environment_terminated"] is False
    assert failed["next_action_required"] is True

    recovered = turns[1].environment_transition
    assert recovered.done is True
    assert recovered.success is True
    assert recovered.termination_reason == "task_success"
    recovered_payload = json.loads(recovered.observation)
    assert recovered_payload["tool"]["ok"] is True
    assert recovered_payload["environment_success"] is True


def test_provider_validation_rejects_unknown_transport_types(tmp_path: Path) -> None:
    options = _robotics_options()
    options["artifact_root"] = str(tmp_path / "artifacts")
    options["providers"] = {
        "vla": {
            "type": "grpc",
            "options": {"endpoint": "http://vla.test", "provider": "some-vla"},
        },
        "perception": {
            "type": "mcp",
            "options": {"endpoint": "http://sam3.test/mcp", "provider": "sam3"},
        },
    }

    with pytest.raises(ConfigError, match="must be 'fake' or 'mcp'"):
        workflow_resources._validate_robotics_environment_options(options)


@pytest.mark.parametrize(
    ("backend_type", "helper_name"),
    (("remote", "_remote_robot_backend"), ("robocasa", "_robocasa_robot_backend")),
)
def test_other_production_backends_compose_with_artifact_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_type: str,
    helper_name: str,
) -> None:
    options = _robotics_options()
    options["artifact_root"] = str(tmp_path / "artifacts")
    backend_options = {"endpoint": "http://robot.test"} if backend_type == "remote" else {}
    options["backend"] = {"type": backend_type, "options": backend_options}
    workflow_resources._validate_robotics_environment_options(options)

    captured: dict[str, object] = {}

    def build_backend(value: object, *, artifact_store: object | None = None) -> object:
        captured["artifact_store"] = artifact_store
        return workflow_resources._fake_robot_backend(_robotics_options()["backend"])

    monkeypatch.setattr(workflow_resources, helper_name, build_backend)
    environment = workflow_resources._robotics_environment(SimpleNamespace(tools=()), options)
    try:
        if backend_type == "remote":
            assert callable(environment._load_artifact)
        else:
            assert captured["artifact_store"].root == tmp_path / "artifacts"  # type: ignore[union-attr]
    finally:
        environment.close()


def test_a_prepared_robot_task_dataset_drives_the_environment_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Join the loader's envelope to the environment's task contract.

    The loader shapes ``robot_task`` rows and ``RobotEnvironment`` validates the
    shape it receives, but each is tested against its own fixture. Only running
    the two together proves the field names, the required identity fields, and
    the instruction cross-check still agree.
    """

    raw = _run_mapping(tmp_path)
    raw["workflow"]["roles"][0]["tools"] = ["vla_act", "finish"]
    raw["environment"] = {"type": "robotics", "options": _robotics_options()}
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
        json.dumps({"task_id": "one", "question": "pick up the cup", "source": "libero"}) + "\n",
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
                    }
                ),
                "answer": "PRIVATE-EPISODE-CANARY",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = parse_run_config(raw, base_dir=tmp_path)
    inputs = workflow_data.load_prepared_inputs(config)
    original = runtime_resources._FakeBackend._generate_one

    def vla_call(backend: object, request: object) -> object:
        response = original(backend, request)  # type: ignore[arg-type]
        return replace(
            response,
            content="",
            finish_reason="tool_calls",
            tool_calls=(
                SimpleNamespace(
                    id="vla-call-1",
                    name="vla_act",
                    arguments=json.dumps({"instruction": "pick up the cup"}),
                ),
            ),
        )

    monkeypatch.setattr(runtime_resources._FakeBackend, "_generate_one", vla_call)

    results = workflow_main.run_workflow(config, inputs=inputs)

    assert len(results) == 1
    assert isinstance(results[0].output, AgentResult)
    transition = results[0].output.turns[0].environment_transition
    assert transition.done is True
    assert transition.success is True
    assert json.loads(transition.observation)["tool"]["tool_id"] == "vla_act"
    assert "PRIVATE-EPISODE-CANARY" not in repr(results)
