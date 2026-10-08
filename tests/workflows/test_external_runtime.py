# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Composition-root coverage for `runtime.type: external` and `environment.type: none`."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from alphaapollo.workflows import resources as workflow_resources
from alphaapollo.workflows._resources import runtime as runtime_resources
from alphaapollo.workflows.config import (
    ConfigError,
    RunConfig,
    parse_run_config,
)

workflow_main = importlib.import_module("alphaapollo.workflows.main")


def _external_runtime(**session_options: Any) -> dict[str, Any]:
    return {
        "type": "external",
        "options": {
            "agent": "fake",
            "model": "scripted-external",
            "session": {"answer": "scripted answer", **session_options},
        },
    }


def _run_mapping(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "prepared.jsonl").write_text(
        json.dumps({"task_id": "t-1", "question": "public problem", "source": "unit"}) + "\n"
    )
    return {
        "version": 1,
        "workflow": {
            "version": 1,
            "name": "offline-external",
            "roles": [{"id": "solver", "target": "solver", "system_prompt": "Solve the problem."}],
            "steps": [{"id": "solve", "kind": "agent", "role": "solver", "output": True}],
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
        "runtimes": {"solver": _external_runtime()},
        "verifiers": {},
        "environment": {"type": "none", "options": {}},
        "output": {"directory": str(tmp_path / "output"), "format": "jsonl"},
    }


def _config(tmp_path: Path, **overrides: Any) -> RunConfig:
    mapping = _run_mapping(tmp_path)
    mapping.update(overrides)
    return parse_run_config(mapping, base_dir=tmp_path)


def _alphaapollo_runtime() -> dict[str, Any]:
    return {
        "type": "alphaapollo",
        "options": {
            "backend": {"type": "fake", "options": {"text": "offline"}},
            "model": "offline-model",
            "sampling": {"temperature": 0.0, "max_tokens": 32},
            "max_turns": 1,
        },
    }


class TestRuntimeResourceValidation:
    def test_external_runtime_is_accepted(self, tmp_path: Path) -> None:
        workflow_resources._validate_composition_config(_config(tmp_path))

    def test_unknown_runtime_type_names_both_supported_kinds(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"]["type"] = "codex_direct"
        with pytest.raises(ConfigError, match="'alphaapollo' or 'external'"):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_unknown_agent_is_rejected_against_the_closed_registry(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"]["options"]["agent"] = "gpt-cli"
        with pytest.raises(ConfigError, match="options.agent must be one of"):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_unknown_option_is_rejected(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"]["options"]["sandbox"] = "workspace-write"
        with pytest.raises(ConfigError, match="sandbox"):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_missing_model_is_rejected(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        del mapping["runtimes"]["solver"]["options"]["model"]
        with pytest.raises(ConfigError, match="model"):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_bad_session_option_is_a_config_error_not_a_late_type_error(
        self, tmp_path: Path
    ) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"]["options"]["session"] = {"nonexistent": True}
        with pytest.raises(ConfigError, match="cannot configure the 'fake' session"):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_invalid_session_value_is_a_config_error(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"]["options"]["session"] = {"termination_reason": "gave_up"}
        with pytest.raises(ConfigError, match="cannot configure the 'fake' session"):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_non_positive_concurrency_is_rejected(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"]["options"]["concurrency"] = 0
        with pytest.raises(ConfigError, match="concurrency"):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_reuse_workspace_is_accepted_and_composed(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"]["options"]["reuse_workspace"] = True
        config = parse_run_config(mapping, base_dir=tmp_path)

        with workflow_resources.compose_resources(config) as resources:
            runtime = resources.runtimes["solver"]
            assert runtime._reuse_workspace is True  # type: ignore[attr-defined]

    def test_non_boolean_reuse_workspace_is_rejected(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"]["options"]["reuse_workspace"] = "yes"

        with pytest.raises(ConfigError, match="reuse_workspace must be a bool"):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_reuse_session_is_composed_for_codex_with_a_shared_workspace(
        self, tmp_path: Path
    ) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"]["options"].update(
            agent="codex",
            model="gpt-5.6-sol",
            reuse_workspace=True,
            reuse_session=True,
            session={},
        )
        config = parse_run_config(mapping, base_dir=tmp_path)

        with workflow_resources.compose_resources(config) as resources:
            runtime = resources.runtimes["solver"]
            assert runtime._reuse_session is True  # type: ignore[attr-defined]

    def test_reuse_session_requires_workspace_and_a_verified_provider(self, tmp_path: Path) -> None:
        missing_workspace = _run_mapping(tmp_path)
        missing_workspace["runtimes"]["solver"]["options"]["reuse_session"] = True
        with pytest.raises(ConfigError, match="requires reuse_workspace"):
            workflow_resources._validate_composition_config(
                parse_run_config(missing_workspace, base_dir=tmp_path)
            )

        unverified = _run_mapping(tmp_path)
        unverified["runtimes"]["solver"]["options"].update(
            reuse_workspace=True,
            reuse_session=True,
        )
        with pytest.raises(ConfigError, match="verified only for agent 'codex'"):
            workflow_resources._validate_composition_config(
                parse_run_config(unverified, base_dir=tmp_path)
            )


class TestEnvironmentRequirement:
    def test_none_is_rejected_when_a_runtime_needs_an_environment(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"] = _alphaapollo_runtime()
        with pytest.raises(ConfigError, match="requires every referenced runtime to own its loop"):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_external_runtime_accepts_and_receives_a_configured_environment(
        self, tmp_path: Path
    ) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["environment"] = {"type": "text_only", "options": {}}
        config = parse_run_config(mapping, base_dir=tmp_path)

        workflow_resources._validate_composition_config(config)
        with workflow_resources.compose_resources(config) as resources:
            runtime = resources.runtimes["solver"]
            assert runtime._environment_factory is not None  # type: ignore[attr-defined]

    def test_native_codex_refuses_an_environment_it_cannot_exclusively_own(
        self, tmp_path: Path
    ) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["environment"] = {"type": "text_only", "options": {}}
        mapping["runtimes"]["solver"]["options"].update(
            agent="codex", session={"sandbox": "read-only"}
        )

        with pytest.raises(ConfigError, match="Codex cannot disable all native execution tools"):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_codex_via_pi_uses_the_runtime_model_and_exclusive_tool_mode(
        self, tmp_path: Path
    ) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["environment"] = {"type": "default", "options": {}}
        mapping["runtimes"]["solver"]["options"].update(
            agent="codex_via_pi",
            model="gpt-5.4-mini",
            tools=["bash"],
            session={},
        )
        config = parse_run_config(mapping, base_dir=tmp_path)

        workflow_resources._validate_composition_config(config)
        session_options = runtime_resources._external_session_options(
            config.runtimes["solver"],
            ("bash",),
            where="runtimes.solver",
            environment_backed=True,
        )

        assert session_options["model"] == "gpt-5.4-mini"
        assert session_options["tool_mode"] == "no_builtin"
        assert "alphaapollo" in session_options["mcp_servers"]

    def test_codex_runtime_model_reaches_the_provider_command(self, tmp_path: Path) -> None:
        from alphaapollo.reasoning.runtime.external import session_factory

        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"]["options"].update(
            agent="codex",
            model="gpt-5.6-sol",
            session={"sandbox": "workspace-write"},
        )
        config = parse_run_config(mapping, base_dir=tmp_path)
        options = runtime_resources._external_session_options(
            config.runtimes["solver"],
            (),
            where="runtimes.solver",
        )
        session = session_factory("codex", options)()

        argv = session.argv()  # type: ignore[attr-defined]

        assert argv[argv.index("-m") + 1] == "gpt-5.6-sol"

    def test_codex_via_pi_rejects_conflicting_model_authorities(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["environment"] = {"type": "text_only", "options": {}}
        mapping["runtimes"]["solver"]["options"].update(
            agent="codex_via_pi",
            model="gpt-5.4-mini",
            session={"model": "gpt-5.6-sol"},
        )

        with pytest.raises(ConfigError, match="options.model and .*session.model must match"):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_unused_alphaapollo_runtime_does_not_force_an_environment(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["unused"] = _alphaapollo_runtime()
        workflow_resources._validate_composition_config(
            parse_run_config(mapping, base_dir=tmp_path)
        )

    def test_workflow_without_any_runtime_is_refused_before_the_environment_question(
        self, tmp_path: Path
    ) -> None:
        # This case used to compose: a deterministic verifier referencing no
        # runtime at all was accepted so that its Environment would not be judged
        # unusable. The topology itself is the defect -- with no agent step
        # nothing produced the candidate, so the verifier judged the problem
        # statement and the report died on an empty agent-target list. The
        # refusal now lands in WorkflowConfig, which is why the runtime set
        # reaching _validate_environment_requirement is never empty.
        mapping = _run_mapping(tmp_path)
        mapping["workflow"]["roles"][0] = {"id": "solver", "target": "judge"}
        mapping["workflow"]["steps"][0]["kind"] = "verifier"
        mapping["runtimes"] = {}
        mapping["verifiers"] = {"judge": {"type": "custom", "options": {"name": "test-only"}}}
        mapping["environment"] = {"type": "text_only", "options": {}}
        with pytest.raises(ConfigError, match="at least one agent step"):
            parse_run_config(mapping, base_dir=tmp_path)

    def test_unknown_environment_type_names_the_supported_kinds(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["environment"] = {"type": "sandboxed", "options": {}}
        with pytest.raises(
            ConfigError,
            match="'default', 'robotics', 'text_only', or 'none'",
        ):
            workflow_resources._validate_composition_config(
                parse_run_config(mapping, base_dir=tmp_path)
            )

    def test_none_environment_factory_fails_loudly_if_a_runtime_asks_for_one(self) -> None:
        from alphaapollo.workflows.config import ResourceConfig

        create = workflow_resources._environment_factory(ResourceConfig(type="none", options={}))
        with pytest.raises(ConfigError, match="a runtime asked for an Environment"):
            create(object())


class TestEndToEnd:
    def test_external_run_persists_real_environment_transition(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["environment"] = {"type": "text_only", "options": {}}

        workflow_main.run_workflow(parse_run_config(mapping, base_dir=tmp_path))

        trajectory = json.loads(
            (tmp_path / "output" / "trajectories.jsonl").read_text().splitlines()[0]
        )
        assert trajectory["metadata"]["environment"]["stepped"] is True
        transition = trajectory["turns"][0]["environment_transition"]
        assert transition["metadata"]["external_environment"] == {
            "environment_stepped": True,
            "synthesized": False,
        }
        assert transition["termination_reason"] == "model_output"

    def test_external_run_persists_results_and_labelled_trajectories(self, tmp_path: Path) -> None:
        mapping = _run_mapping(tmp_path)
        mapping["runtimes"]["solver"]["options"]["session"] = {
            "answer": "Final answer: 204",
            "tool_id": "bash",
            "tool_arguments": {"command": "python3 -c 'print(204)'"},
            "tool_result": "204",
        }
        config = parse_run_config(mapping, base_dir=tmp_path)

        results = workflow_main.run_workflow(config)

        assert len(results) == 1
        assert results[0].output.final_text == "Final answer: 204"

        output_dir = tmp_path / "output"
        summaries = [
            json.loads(line)
            for line in (output_dir / "workflow_results.jsonl").read_text().splitlines()
        ]
        trajectories = [
            json.loads(line)
            for line in (output_dir / "trajectories.jsonl").read_text().splitlines()
        ]
        assert len(summaries) == 1
        assert len(trajectories) == 1

        trajectory = trajectories[0]
        assert trajectory["metadata"]["policy_source"] == "external_agent"
        assert trajectory["metadata"]["trainable"] is False
        assert [turn["index"] for turn in trajectory["turns"]] == [0, 1]

        tool_turn, final_turn = trajectory["turns"]
        assert tool_turn["generation_response"]["tool_calls"][0]["function"]["name"] == "bash"
        assert tool_turn["environment_transition"]["observation"]["content"] == "204"
        assert tool_turn["environment_transition"]["done"] is False
        assert final_turn["environment_transition"]["done"] is True
        assert final_turn["environment_transition"]["termination_reason"] == "final"
        for turn in trajectory["turns"]:
            response = turn["generation_response"]
            assert response["provenance"] is None
            assert response["response_token_ids"] is None
            assert response["response_logprobs"] is None

    def test_no_host_path_leaks_into_persisted_records(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        workflow_main.run_workflow(config)
        trajectory = (tmp_path / "output" / "trajectories.jsonl").read_text()
        assert "alphaapollo-external-" not in trajectory
