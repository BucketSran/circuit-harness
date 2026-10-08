# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""End-to-end composition coverage for ``verifier.type: custom``."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from alphaapollo.reasoning.verification import (
    VerificationRequest,
    VerificationResult,
    Verifier,
    register_custom_verifier,
)
from alphaapollo.workflows.config import ConfigError, parse_run_config
from alphaapollo.workflows.resources import validate_composition_config
from alphaapollo.workflows.run import run


class _PassingVerifier(Verifier):
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.closed = False
        self.requests: list[VerificationRequest] = []

    def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
        self.requests.extend(requests)
        return [
            VerificationResult(
                request_id=request.request_id,
                verdict="pass",
                candidate=request.candidate,
                candidate_ref=request.candidate_ref,
                feedback=str(self.config["feedback"]),
            )
            for request in requests
        ]

    def close(self) -> None:
        self.closed = True


class _FailingVerifier(Verifier):
    def __init__(self, config: Mapping[str, Any]) -> None:
        del config

    def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
        assert requests
        raise RuntimeError("official evaluator timed out")

    def close(self) -> None:
        return None


def _mapping(tmp_path: Path, *, verifier_name: str) -> dict[str, Any]:
    (tmp_path / "prepared.jsonl").write_text(
        json.dumps({"task_id": "task-1", "question": "public problem"}) + "\n",
        encoding="utf-8",
    )
    return {
        "version": 1,
        "workflow": {
            "version": 1,
            "name": "custom-verifier-test",
            "roles": [
                {"id": "solver", "target": "solver", "system_prompt": "Solve."},
                {"id": "verifier", "target": "verifier"},
            ],
            "steps": [
                {"id": "solve", "kind": "agent", "role": "solver"},
                {
                    "id": "verify",
                    "kind": "verifier",
                    "role": "verifier",
                    "output": True,
                },
            ],
            "entry_step": "solve",
            "transitions": [{"source": "solve", "target": "verify"}],
        },
        "dataset": {
            "path": "prepared.jsonl",
            "format": "jsonl",
            "input_key": "question",
            "id_key": "task_id",
        },
        "runtimes": {
            "solver": {
                "type": "external",
                "options": {
                    "agent": "fake",
                    "model": "fake-model",
                    "session": {"answer": "candidate"},
                },
            }
        },
        "verifiers": {
            "verifier": {
                "type": "custom",
                "options": {
                    "name": verifier_name,
                    "config": {"feedback": "measured feedback"},
                },
            }
        },
        "environment": {"type": "none"},
        "output": {"directory": "output", "format": "jsonl"},
    }


def test_custom_verifier_runs_from_full_config_and_standard_results_are_saved(
    tmp_path: Path,
) -> None:
    name = "tests.workflow.passing"
    created: list[_PassingVerifier] = []

    def factory(config: Mapping[str, Any]) -> _PassingVerifier:
        verifier = _PassingVerifier(config)
        created.append(verifier)
        return verifier

    register_custom_verifier(name, factory, replace=True)
    config = parse_run_config(_mapping(tmp_path, verifier_name=name), base_dir=tmp_path)

    results = run(config)

    assert len(results) == 1
    assert isinstance(results[0].output, VerificationResult)
    assert results[0].output.feedback == "measured feedback"
    assert created[0].config == {"feedback": "measured feedback"}
    assert created[0].requests[0].metadata["candidate_runtime_seconds"] >= 0.0
    assert created[0].closed is True
    assert (tmp_path / "output" / "workflow_results.jsonl").is_file()
    assert (tmp_path / "output" / "trajectories.jsonl").is_file()


def test_agent_output_and_trajectory_survive_a_later_verifier_failure(tmp_path: Path) -> None:
    name = "tests.workflow.failing"
    register_custom_verifier(name, _FailingVerifier, replace=True)
    config = parse_run_config(_mapping(tmp_path, verifier_name=name), base_dir=tmp_path)

    with pytest.raises(RuntimeError, match="official evaluator timed out"):
        run(config)

    checkpoints = sorted((tmp_path / "output").glob("workflow-step-*.json"))
    assert len(checkpoints) == 1
    payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert payload["step"]["step_id"] == "solve"
    assert payload["step"]["output"]["final_text"] == "candidate"
    assert payload["agent_trajectory"]["task_id"] == payload["step"]["output"]["task_id"]


def test_existing_native_output_is_rejected_before_model_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalAgentRuntime

    name = "tests.workflow.no-overwrite"
    register_custom_verifier(name, _PassingVerifier, replace=True)
    config = parse_run_config(_mapping(tmp_path, verifier_name=name), base_dir=tmp_path)
    run(config)
    previous = (tmp_path / "output" / "trajectories.jsonl").read_bytes()

    def must_not_run(_self: object, _tasks: object) -> object:
        raise AssertionError("model runtime should not start")

    monkeypatch.setattr(ExternalAgentRuntime, "run_batch", must_not_run)

    with pytest.raises(ConfigError, match="already contains evidence"):
        run(config)

    assert (tmp_path / "output" / "trajectories.jsonl").read_bytes() == previous


def test_unknown_custom_verifier_is_rejected_before_resources_are_built(
    tmp_path: Path,
) -> None:
    config = parse_run_config(
        _mapping(tmp_path, verifier_name="tests.workflow.missing"),
        base_dir=tmp_path,
    )

    with pytest.raises(ConfigError, match="unknown custom Verifier"):
        validate_composition_config(config)


def test_custom_verifier_role_rejects_unused_agent_fields(tmp_path: Path) -> None:
    name = "tests.workflow.role-fields"
    register_custom_verifier(name, _PassingVerifier, replace=True)
    mapping = _mapping(tmp_path, verifier_name=name)
    mapping["workflow"]["roles"][1]["system_prompt"] = "unused"
    config = parse_run_config(mapping, base_dir=tmp_path)

    with pytest.raises(ConfigError, match="custom verifier role.*AgentVerifier-only"):
        validate_composition_config(config)


def test_custom_verifier_config_must_be_a_mapping(tmp_path: Path) -> None:
    name = "tests.workflow.bad-config"
    register_custom_verifier(name, _PassingVerifier, replace=True)
    mapping = _mapping(tmp_path, verifier_name=name)
    mapping["verifiers"]["verifier"]["options"]["config"] = ["not", "a", "mapping"]
    config = parse_run_config(mapping, base_dir=tmp_path)

    with pytest.raises(ConfigError, match="options.config must be a mapping"):
        validate_composition_config(config)
