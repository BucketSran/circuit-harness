from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.resources
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import alphaapollo.workflows.config as workflow_config
from alphaapollo.workflows.config import (
    ConfigError,
    DatasetConfig,
    EnsembleConfig,
    MemoryConfig,
    ResourceConfig,
    RoleConfig,
    RunConfig,
    _thaw_json,
    compose_config_mapping,
    config_digest,
    load_run_config,
    load_workflow_config,
    parse_run_config,
    parse_workflow_config,
)
from alphaapollo.workflows.records import Workflow, WorkflowInput
from alphaapollo.workflows.resources import validate_composition_config

REPOSITORY_ROOT = Path(__file__).parents[3]
PRESETS = REPOSITORY_ROOT / "alphaapollo" / "configs" / "preset"


def _linear_config() -> dict[str, object]:
    return {
        "version": 1,
        "name": "linear",
        "roles": [
            {
                "id": "solver",
                "target": "primary",
                "system_prompt": "Solve the public problem.",
            }
        ],
        "steps": [{"id": "solve", "kind": "agent", "role": "solver", "output": True}],
        "entry_step": "solve",
        "transitions": [],
    }


def _verifier_route_config(
    transitions: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "version": 1,
        "name": "verifier_routes",
        "roles": [
            {
                "id": "judge",
                "target": "judge",
                "system_prompt": "judge",
                "input_template": "{candidate}",
                "output_format": "json",
            },
            {"id": "solver", "target": "runtime", "system_prompt": "finish"},
        ],
        "steps": [
            {"id": "judge", "kind": "verifier", "role": "judge"},
            {"id": "out", "kind": "agent", "role": "solver", "output": True},
        ],
        "entry_step": "judge",
        "transitions": transitions,
    }


@pytest.mark.parametrize(
    "name",
    [
        "vanilla_reasoning.yaml",
        "vanilla_ensemble.yaml",
        "bash_reasoning.yaml",
        "python_code_reasoning.yaml",
    ],
)
def test_shipped_presets_load_and_contain_only_topology(name: str) -> None:
    config = load_workflow_config(PRESETS / name)

    assert config.version == 1
    # The single-branch presets end on a terminal verifier so the recorded
    # verdict is bound to the answer that gets scored; the ensemble preset
    # cannot (see its own comment) and still ends on the finalizer.
    expected_output_step = "final" if name == "vanilla_ensemble.yaml" else "certify"
    assert config.steps[-1].id == expected_output_step
    assert config.steps[-1].output is True
    assert all(role.model is None for role in config.roles)
    text = (PRESETS / name).read_text(encoding="utf-8")
    for forbidden in ("dataset:", "runtimes:", "environment:", "output_dir:"):
        assert forbidden not in text
    if name == "vanilla_ensemble.yaml":
        finalizer = next(role for role in config.roles if role.id == "finalizer")
        assert "Final answer: <answer>" in finalizer.system_prompt


@pytest.mark.parametrize(
    "name",
    [
        "vanilla_reasoning.yaml",
        "vanilla_ensemble.yaml",
        "bash_reasoning.yaml",
        "python_code_reasoning.yaml",
    ],
)
def test_every_verifier_role_asks_its_model_for_the_format_its_parser_reads(name: str) -> None:
    """A prompt and a parser that disagree discard 100% of judgments.

    `output_format` selects the parser; the prompt selects what the model
    writes. Nothing else connects them -- the fake backend answers in whatever
    the role declares, so every offline test would keep passing while a live
    run threw away every verdict it was given.
    """

    config = load_workflow_config(PRESETS / name)
    verifier_steps = {step.role for step in config.steps if step.kind == "verifier"}

    assert verifier_steps
    for role in (role for role in config.roles if role.id in verifier_steps):
        prompt = role.system_prompt or ""
        if role.output_format == "verdict-line":
            assert "VERDICT:" in prompt
            assert "FEEDBACK:" in prompt
            assert "JSON" not in prompt
        else:
            assert role.output_format == "json"
            assert "JSON" in prompt


def test_only_the_ensemble_preset_still_scores_an_unjudged_final_answer() -> None:
    """The topology is what makes a verdict bind, so pin which presets have it.

    `scoring._final_output_verification` returns a judgment only when it is
    bound to the selected output, and an agent finalizer rewrites the text the
    verifier saw. A single-branch preset that ends on `final` therefore records
    `inconclusive` no matter what its verifier said. `vanilla_ensemble` still
    does, deliberately: see the comment in that file.
    """

    for name in ("vanilla_reasoning.yaml", "bash_reasoning.yaml", "python_code_reasoning.yaml"):
        config = load_workflow_config(PRESETS / name)
        output_step = next(step for step in config.steps if step.output)
        assert output_step.kind == "verifier"
        assert [t.target for t in config.transitions if t.source == "final"] == [output_step.id]

    ensemble = load_workflow_config(PRESETS / "vanilla_ensemble.yaml")
    assert next(step for step in ensemble.steps if step.output).kind == "agent"


def test_the_packaged_dataset_group_declares_the_dataclass_defaults() -> None:
    """One semantic unit, two places that state it. Pin that they agree.

    `alphaapollo/configs/dataset/default.yaml` is the discoverable form an
    operator copies; `DatasetConfig` is what actually applies when a recipe omits
    a key. They agree today and nothing made them, so a future edit to either
    would resolve by whichever path a caller took rather than failing.

    `path` is deliberately absent from the comparison: the group marks it with
    Hydra's mandatory-value token and the dataclass gives it no default, which is
    the same statement in each dialect.
    """

    packaged = yaml.safe_load(
        (REPOSITORY_ROOT / "alphaapollo" / "configs" / "dataset" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )
    fields = {field.name: field for field in dataclasses.fields(DatasetConfig)}

    assert packaged["path"] == "???"
    assert fields["path"].default is dataclasses.MISSING
    assert set(packaged) == set(fields)
    declared = DatasetConfig(path=Path("unused.jsonl"))
    for name in sorted(set(packaged) - {"path"}):
        value = packaged[name]
        if isinstance(value, list):
            value = tuple(value)
        assert value == getattr(declared, name), name


def test_task_payload_format_is_optional_but_validated_when_set() -> None:
    inherited = DatasetConfig(
        path=Path("public.json"),
        format="json",
        task_payload_path=Path("private.json"),
    )
    explicit = DatasetConfig(
        path=Path("public.json"),
        format="json",
        task_payload_path=Path("private.jsonl"),
        task_payload_format="jsonl",
    )

    assert inherited.task_payload_format is None
    assert explicit.task_payload_format == "jsonl"
    with pytest.raises(ConfigError, match="task_payload_format must be one of"):
        DatasetConfig(
            path=Path("public.json"),
            task_payload_path=Path("private.csv"),
            task_payload_format="csv",
        )
    with pytest.raises(ConfigError, match="task_payload_format requires dataset.task_payload_path"):
        DatasetConfig(path=Path("public.json"), task_payload_format="json")
    with pytest.raises(ConfigError, match="task_payload_envelope must be one of.*direct"):
        DatasetConfig(path=Path("public.json"), task_payload_envelope="inferred")


def test_a_non_direct_task_payload_envelope_requires_the_private_split() -> None:
    """``robot_task`` reshapes private rows, so it is meaningless without them.

    ``direct`` stays valid on its own because it is the default: a public-only
    dataset never names an envelope.
    """

    with pytest.raises(ConfigError, match="requires dataset.task_payload_path"):
        DatasetConfig(path=Path("public.json"), task_payload_envelope="robot_task")

    configured = DatasetConfig(
        path=Path("public.json"),
        task_payload_path=Path("private.jsonl"),
        task_payload_envelope="robot_task",
    )

    assert configured.task_payload_envelope == "robot_task"


def test_ensemble_solver_roles_state_the_answer_block_contract() -> None:
    # The ensemble preset votes on `declared_answer`, and the reducer reads the
    # branches these three roles produce. If they do not ask for the same output
    # contract the scorer reads, voting groups branches on text scoring excludes.
    config = load_workflow_config(PRESETS / "vanilla_ensemble.yaml")
    solver_roles = {role.id: role for role in config.roles if role.target == "solver"}

    assert set(solver_roles) == {"problem_analyst", "proposer", "reviser", "finalizer"}
    for role_id in ("proposer", "reviser", "finalizer"):
        prompt = solver_roles[role_id].system_prompt
        assert "give the final answer only inside <answer>...</answer>" in prompt
        assert "formatted in LaTeX as \\boxed{...}" in prompt
        assert "Do not put code, reasoning, or any other format inside <answer>." in prompt
        # Kept alongside the block, not replaced: a branch truncated before it
        # closes <answer> must still leave the reducer a marker to group on.
        assert "Final answer: <answer>" in prompt


def test_memory_options_preserve_existing_positional_constructor_order() -> None:
    config = MemoryConfig(
        "verification",
        "persistent",
        5,
        "full",
        True,
        "mem0",
        {"version": "v1.1"},
    )

    assert config.persistent_backend == "mem0"
    assert config.persistent_config == {"version": "v1.1"}
    assert config.options == {}


def test_packaged_hydra_groups_are_installed_as_resources() -> None:
    root = importlib.resources.files("alphaapollo.configs")

    assert root.joinpath("workflow.yaml").is_file()
    assert root.joinpath("runtime", "alphaapollo.yaml").is_file()
    assert root.joinpath("runtime", "alphaapollo_verifier.yaml").is_file()
    assert root.joinpath("runtime", "external.yaml").is_file()
    assert root.joinpath("preset", "vanilla_reasoning.yaml").is_file()
    assert root.joinpath("preset", "vanilla_ensemble.yaml").is_file()
    assert root.joinpath("preset", "bash_reasoning.yaml").is_file()
    assert root.joinpath("preset", "python_code_reasoning.yaml").is_file()
    assert root.joinpath("environment", "none.yaml").is_file()


def test_ensemble_preset_is_selectable_from_the_hydra_root(tmp_path: Path) -> None:
    primary = tmp_path / "ensemble.yaml"
    primary.write_text(
        """\
defaults:
  - workflow
  - override preset@workflow: vanilla_ensemble
  - _self_

dataset:
  path: prepared.jsonl

output:
  directory: output
""",
        encoding="utf-8",
    )

    config = load_run_config(primary)

    assert config.workflow.name == "vanilla_ensemble"
    assert config.workflow.ensemble is not None
    assert config.workflow.ensemble.replicas == 5


def test_hydra_primary_paths_still_resolve_relative_to_the_leaf(tmp_path: Path) -> None:
    primary = tmp_path / "task.yaml"
    primary.write_text(
        """\
defaults:
  - workflow
  - _self_

dataset:
  path: prepared.jsonl

output:
  directory: output
""",
        encoding="utf-8",
    )

    config = load_run_config(primary)

    assert config.dataset.path == tmp_path / "prepared.jsonl"
    assert config.output.directory == tmp_path / "output"
    assert config.scoring is None


def _scored_primary(tmp_path: Path, *, scoring: bool, grader_id: bool) -> Path:
    scoring_block = (
        """
scoring:
  dataset_root: data
  source_id: aime2026_30
"""
        if scoring
        else ""
    )
    environment_block = (
        """
environment:
  options:
    grader_id: exact_match
"""
        if grader_id
        else """
environment:
  options:
    max_steps: 4
"""
    )
    primary = tmp_path / "task.yaml"
    primary.write_text(
        "defaults:\n  - workflow\n  - _self_\n\n"
        "dataset:\n  path: prepared.jsonl\n\n"
        "output:\n  directory: output\n"
        f"{environment_block}{scoring_block}",
        encoding="utf-8",
    )
    return primary


def test_scored_run_refuses_an_environment_that_grades_in_loop(tmp_path: Path) -> None:
    """Gold must never reach the process running the model in a scored run.

    Accepting the option and quietly withholding gold would be worse: the config
    would read as honoured while doing nothing.
    """

    primary = _scored_primary(tmp_path, scoring=True, grader_id=True)

    with pytest.raises(ConfigError, match="grades inside the episode"):
        load_run_config(primary)


def test_scored_run_accepts_an_environment_without_in_loop_grading(tmp_path: Path) -> None:
    config = load_run_config(_scored_primary(tmp_path, scoring=True, grader_id=False))

    assert config.scoring is not None
    assert config.environment.options["max_steps"] == 4
    assert "grader_id" not in config.environment.options


def test_an_unscored_run_is_refused_in_loop_grading_too(tmp_path: Path) -> None:
    """No Workflow config composes in-loop grading, with or without `scoring`.

    ``parse_run_config`` keeps the key because ``ResourceConfig.options`` is a
    free-form JSON mapping; the Environment schema is where the accepted keys are
    named, and it does not name this one. The parse-time refusal above exists only
    to give a scored recipe an error that names the boundary it crossed, so both
    halves are pinned here: the key survives parsing, and composition rejects it.
    """

    config = load_run_config(_scored_primary(tmp_path, scoring=False, grader_id=True))

    assert config.scoring is None
    assert config.environment.options["grader_id"] == "exact_match"
    with pytest.raises(ConfigError, match="environment.options contains unknown keys"):
        validate_composition_config(config)


def test_hydra_composition_does_not_bypass_strict_run_config_keys(tmp_path: Path) -> None:
    primary = tmp_path / "task.yaml"
    primary.write_text(
        """\
defaults:
  - workflow
  - _self_

dataset:
  path: prepared.jsonl

output:
  directory: output

unexpected: rejected
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unknown run config keys.*unexpected"):
        load_run_config(primary)


def test_general_workflow_config_rejects_unbound_mandatory_values() -> None:
    root = importlib.resources.files("alphaapollo.configs")

    with pytest.raises(ConfigError, match="could not compose config.*Missing mandatory value"):
        compose_config_mapping(Path(str(root.joinpath("workflow.yaml"))))


def test_config_is_strict_and_rejects_unknown_keys() -> None:
    raw = _linear_config()
    raw["python"] = "do_not_execute"

    with pytest.raises(ConfigError, match="unknown workflow config keys"):
        parse_workflow_config(raw)


def test_role_prompt_ref_resolves_shared_prompt_content() -> None:
    raw = _linear_config()
    raw["roles"] = [{"id": "solver", "target": "primary", "prompt_ref": "roles.model_default"}]

    role = parse_workflow_config(raw).roles[0]

    assert role.prompt_ref == "roles.model_default"
    assert role.system_prompt == ""
    assert role.input_template == "{problem}"
    assert replace(role, tools=("bash",)).prompt_ref == "roles.model_default"


def test_role_prompt_ref_preserves_the_existing_positional_constructor() -> None:
    role = RoleConfig("solver", "primary", "system", "model")

    assert role.system_prompt == "system"
    assert role.model == "model"
    assert role.prompt_ref is None


def test_role_rejects_ambiguous_inline_and_referenced_prompts() -> None:
    raw = _linear_config()
    raw["roles"][0]["prompt_ref"] = "roles.proposer"  # type: ignore[index]

    with pytest.raises(ConfigError, match="cannot combine prompt_ref"):
        parse_workflow_config(raw)


def test_role_rejects_unknown_prompt_ref_during_config_parsing() -> None:
    raw = _linear_config()
    raw["roles"] = [{"id": "solver", "target": "primary", "prompt_ref": "roles.missing"}]

    with pytest.raises(ConfigError, match="unknown prompt_ref"):
        parse_workflow_config(raw)


def test_step_cannot_override_a_role_prompt_ref_user_template() -> None:
    raw = _linear_config()
    raw["roles"] = [{"id": "solver", "target": "primary", "prompt_ref": "roles.model_default"}]
    raw["steps"][0]["input_template"] = "Step override: {problem}"  # type: ignore[index]

    with pytest.raises(ConfigError, match="input_template cannot override prompt_ref"):
        parse_workflow_config(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("depends_on", ["prepare"]),
        ("retry", {"steps": ["prepare"], "max_iterations": 1}),
    ],
)
def test_config_rejects_static_dag_step_fields(key: str, value: object) -> None:
    raw = _linear_config()
    raw["steps"][0][key] = value  # type: ignore[index]

    with pytest.raises(ConfigError, match=rf"unknown workflow\.steps\[0\] keys.*{key}"):
        parse_workflow_config(raw)


def test_config_rejects_dynamic_template_fields() -> None:
    raw = _linear_config()
    raw["steps"][0]["input_template"] = "{problem.__class__}"  # type: ignore[index]

    with pytest.raises(ConfigError, match="unsupported placeholder"):
        parse_workflow_config(raw)


@pytest.mark.parametrize("field", ["upstream_outputs", "upstream_text"])
def test_config_rejects_static_dag_template_fields(field: str) -> None:
    raw = _linear_config()
    raw["steps"][0]["input_template"] = "{" + field + "}"  # type: ignore[index]

    with pytest.raises(ConfigError, match=rf"unsupported placeholder '{field}'"):
        parse_workflow_config(raw)


@pytest.mark.parametrize("candidate_from", ["consume", "future"])
def test_config_rejects_candidate_not_available_on_first_step_arrival(
    candidate_from: str,
) -> None:
    raw = {
        "version": 1,
        "name": "candidate_order",
        "roles": [
            {"id": "solver", "target": "runtime", "system_prompt": "solve"},
        ],
        "steps": [
            {"id": "start", "kind": "agent", "role": "solver"},
            {
                "id": "consume",
                "kind": "agent",
                "role": "solver",
                "candidate_from": candidate_from,
            },
            {"id": "future", "kind": "agent", "role": "solver", "output": True},
        ],
        "entry_step": "start",
        "transitions": [
            {"source": "start", "target": "consume"},
            {"source": "consume", "target": "future"},
        ],
    }

    with pytest.raises(ConfigError, match="guaranteed to run before"):
        parse_workflow_config(raw)


def test_config_rejects_branch_local_candidate_after_merge() -> None:
    raw = {
        "version": 1,
        "name": "candidate_merge",
        "roles": [
            {"id": "solver", "target": "runtime", "system_prompt": "solve"},
            {
                "id": "judge",
                "target": "judge",
                "system_prompt": "judge",
                "input_template": "{candidate}",
                "output_format": "json",
            },
        ],
        "steps": [
            {"id": "start", "kind": "agent", "role": "solver"},
            {"id": "branch", "kind": "verifier", "role": "judge"},
            {"id": "candidate", "kind": "agent", "role": "solver"},
            {
                "id": "merge",
                "kind": "agent",
                "role": "solver",
                "candidate_from": "candidate",
                "output": True,
            },
        ],
        "entry_step": "start",
        "transitions": [
            {"source": "start", "target": "branch"},
            {"source": "branch", "target": "candidate", "condition": "passed"},
            {"source": "branch", "target": "merge", "condition": "not_passed"},
            {"source": "candidate", "target": "merge"},
        ],
    }

    with pytest.raises(ConfigError, match="guaranteed to run before"):
        parse_workflow_config(raw)


@pytest.mark.parametrize("condition", ["failed", "inconclusive"])
def test_config_rejects_routes_shadowed_by_unbounded_not_passed(condition: str) -> None:
    raw = _verifier_route_config(
        [
            {"source": "judge", "target": "out", "condition": "not_passed"},
            {"source": "judge", "target": "out", "condition": condition},
            {"source": "judge", "target": "out", "condition": "passed"},
        ]
    )

    with pytest.raises(ConfigError, match="shadowed by earlier unbounded routes"):
        parse_workflow_config(raw)


def test_config_rejects_finite_route_that_cannot_exhaust_before_fallback() -> None:
    raw = _verifier_route_config(
        [
            {
                "source": "judge",
                "target": "out",
                "condition": "not_passed",
                "max_iterations": 1,
            },
            {"source": "judge", "target": "out", "condition": "failed"},
            {"source": "judge", "target": "out", "condition": "inconclusive"},
            {"source": "judge", "target": "out", "condition": "passed"},
        ]
    )

    with pytest.raises(ConfigError, match="semantically unreachable"):
        parse_workflow_config(raw)


def test_config_accepts_finite_route_when_target_returns_to_source() -> None:
    raw = {
        "version": 1,
        "name": "bounded_retry",
        "roles": [
            {
                "id": "judge",
                "target": "judge",
                "system_prompt": "judge",
                "input_template": "{candidate}",
                "output_format": "json",
            },
            {"id": "solver", "target": "runtime", "system_prompt": "revise"},
        ],
        "steps": [
            {"id": "judge", "kind": "verifier", "role": "judge"},
            {"id": "revise", "kind": "agent", "role": "solver"},
            {"id": "out", "kind": "agent", "role": "solver", "output": True},
        ],
        "entry_step": "judge",
        "transitions": [
            {"source": "judge", "target": "out", "condition": "passed"},
            {
                "source": "judge",
                "target": "revise",
                "condition": "not_passed",
                "max_iterations": 1,
            },
            {"source": "judge", "target": "out", "condition": "not_passed"},
            {"source": "revise", "target": "judge"},
        ],
    }

    config = parse_workflow_config(raw)

    assert config.transitions[1].target == "revise"


def test_config_accounts_for_return_edge_budget_when_finding_fallback() -> None:
    raw = {
        "version": 1,
        "name": "insufficient_return_budget",
        "roles": [
            {"id": "solver", "target": "runtime", "system_prompt": "continue"},
        ],
        "steps": [
            {"id": "start", "kind": "agent", "role": "solver"},
            {"id": "detour", "kind": "agent", "role": "solver"},
            {"id": "out", "kind": "agent", "role": "solver", "output": True},
        ],
        "entry_step": "start",
        "transitions": [
            {"source": "start", "target": "detour", "max_iterations": 2},
            {"source": "start", "target": "out"},
            {"source": "detour", "target": "start", "max_iterations": 1},
            {"source": "detour", "target": "out"},
        ],
    }

    with pytest.raises(ConfigError, match="transition 1 from 'start'.*semantically unreachable"):
        parse_workflow_config(raw)


def test_agent_role_rejects_unconsumed_output_format() -> None:
    raw = _linear_config()
    raw["roles"][0]["output_format"] = "json"  # type: ignore[index]

    with pytest.raises(ConfigError, match="agent role 'solver'.output_format"):
        parse_workflow_config(raw)


def test_agent_role_requires_system_prompt() -> None:
    raw = _linear_config()
    del raw["roles"][0]["system_prompt"]  # type: ignore[index]

    with pytest.raises(ConfigError, match="agent role 'solver'.system_prompt is required"):
        parse_workflow_config(raw)


def test_excessive_edge_budget_fails_before_semantic_state_exploration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _linear_config()
    raw["steps"] = [
        {"id": "start", "kind": "agent", "role": "solver"},
        {"id": "out", "kind": "agent", "role": "solver", "output": True},
    ]
    raw["entry_step"] = "start"
    raw["transitions"] = [
        {"source": "start", "target": "start", "max_iterations": 1_000_000_000},
        {"source": "start", "target": "out"},
    ]

    def must_not_explore(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized edge budgets must fail before state exploration")

    monkeypatch.setattr(
        workflow_config,
        "_validate_semantic_route_reachability",
        must_not_explore,
    )

    with pytest.raises(ConfigError, match="max_iterations must be between 1 and 100"):
        parse_workflow_config(raw)


def test_semantic_route_state_exploration_has_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _linear_config()
    raw["steps"] = [
        {"id": "start", "kind": "agent", "role": "solver"},
        {"id": "out", "kind": "agent", "role": "solver", "output": True},
    ]
    raw["entry_step"] = "start"
    raw["transitions"] = [
        {"source": "start", "target": "start", "max_iterations": 3},
        {"source": "start", "target": "out"},
    ]
    monkeypatch.setattr(workflow_config, "_MAX_SEMANTIC_ROUTE_STATES", 2)

    with pytest.raises(ConfigError, match="semantic validation limit of 2 states"):
        parse_workflow_config(raw)


def test_config_accepts_many_low_cost_bounded_transitions() -> None:
    raw = _linear_config()
    raw["steps"] = [
        {"id": "start", "kind": "agent", "role": "solver"},
        {"id": "out", "kind": "agent", "role": "solver", "output": True},
    ]
    raw["entry_step"] = "start"
    raw["transitions"] = [
        *({"source": "start", "target": "start", "max_iterations": 1} for _ in range(33)),
        {"source": "start", "target": "out"},
    ]

    config = parse_workflow_config(raw)

    assert len(config.transitions) == 34


def test_config_rejects_role_without_any_step_consumer() -> None:
    raw = _linear_config()
    raw["roles"].append(  # type: ignore[union-attr]
        {"id": "unused", "target": "other", "system_prompt": "never consumed"}
    )

    with pytest.raises(ConfigError, match="unused roles.*unused"):
        parse_workflow_config(raw)


def test_role_cannot_be_shared_across_step_kinds() -> None:
    raw = _linear_config()
    raw["steps"] = [
        {"id": "start", "kind": "agent", "role": "solver"},
        {"id": "out", "kind": "verifier", "role": "solver", "output": True},
    ]
    raw["entry_step"] = "start"
    raw["transitions"] = [{"source": "start", "target": "out"}]

    with pytest.raises(ConfigError, match="cannot be shared by agent and verifier"):
        parse_workflow_config(raw)


def test_verifier_output_format_has_closed_vocabulary() -> None:
    raw = _verifier_route_config([{"source": "judge", "target": "out", "condition": "always"}])
    raw["roles"][0]["output_format"] = "definitely-not-a-format"  # type: ignore[index]

    with pytest.raises(ConfigError, match="must be one of.*json.*verdict-line"):
        parse_workflow_config(raw)


@pytest.mark.parametrize(
    ("configured", "canonical"),
    [(" JSON ", "json"), ("verdict_line", "verdict-line")],
)
def test_verifier_output_format_is_normalized(
    configured: str,
    canonical: str,
) -> None:
    raw = _verifier_route_config([{"source": "judge", "target": "out", "condition": "always"}])
    raw["roles"][0]["output_format"] = configured  # type: ignore[index]

    config = parse_workflow_config(raw)

    assert config.roles[0].output_format == canonical


def test_config_rejects_unbounded_cycle() -> None:
    raw = {
        "version": 1,
        "name": "cycle",
        "roles": [
            {"id": "solver", "target": "runtime", "system_prompt": "solve"},
            {"id": "judge", "target": "judge", "system_prompt": "judge"},
        ],
        "steps": [
            {"id": "a", "kind": "agent", "role": "solver"},
            {"id": "b", "kind": "verifier", "role": "judge"},
            {"id": "out", "kind": "agent", "role": "solver", "output": True},
        ],
        "entry_step": "a",
        "transitions": [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "out", "condition": "passed"},
            {"source": "b", "target": "a", "condition": "not_passed"},
        ],
    }

    with pytest.raises(ConfigError, match="cycle without an explicit"):
        parse_workflow_config(raw)


def test_config_refuses_a_topology_with_no_agent_step() -> None:
    """A verifier-only topology is a ``ConfigError``, not a run that reports ``solver=none``.

    Refusal rather than reporting, because such a run cannot state a result. Only
    an agent step produces the candidate a verifier judges: ``WorkflowInput``
    carries the problem and nothing else, so ``WorkflowExecutor`` falls back to
    the problem statement as the candidate and the ledger records the problem as
    the run's own ``final_answer``. Nothing shipped, documented, or exported
    declares such a topology, and ``report.resource_summary`` has no solver
    runtime to name for it. The information is complete the moment the graph is
    parsed, so the error lands here rather than after the verifier is paid.
    """

    raw = _verifier_route_config([{"source": "judge", "target": None, "condition": "always"}])
    raw["roles"] = [raw["roles"][0]]  # type: ignore[index]
    raw["steps"] = [{"id": "judge", "kind": "verifier", "role": "judge", "output": True}]
    raw["transitions"] = []

    with pytest.raises(ConfigError, match="at least one agent step"):
        parse_workflow_config(raw)


def test_config_requires_one_reachable_terminal_output() -> None:
    raw = _linear_config()
    raw["steps"][0]["output"] = False  # type: ignore[index]

    with pytest.raises(ConfigError, match="exactly one output"):
        parse_workflow_config(raw)


def test_workflow_plan_is_immutable_and_indexed() -> None:
    config = parse_workflow_config(_linear_config())
    workflow = Workflow.from_config(config)

    assert workflow.get_step("solve") is config.steps[0]
    assert workflow.output_step_id == "solve"
    with pytest.raises((AttributeError, TypeError)):
        workflow._steps["other"] = config.steps[0]  # type: ignore[index]


def test_run_config_resolves_paths_and_validates_targets(tmp_path: Path) -> None:
    preset = tmp_path / "preset.yaml"
    preset.write_text(
        """\
version: 1
name: run_target
roles:
  - id: solver
    target: primary
    system_prompt: Solve.
steps:
  - id: solve
    kind: agent
    role: solver
    output: true
entry_step: solve
transitions: []
""",
        encoding="utf-8",
    )
    raw = {
        "version": 1,
        "workflow": "preset.yaml",
        "dataset": {
            "path": "data/tasks.jsonl",
            "format": "jsonl",
            "input_key": "question",
            "id_key": "task_id",
            "metadata_keys": ["source", "difficulty"],
        },
        "runtimes": {"primary": {"type": "fake", "options": {"seed": 7}}},
        "verifiers": {},
        "environment": {"type": "default", "options": {"max_steps": 4}},
        "output": {"directory": "runs/test", "format": "jsonl"},
    }

    config = parse_run_config(raw, base_dir=tmp_path)

    assert isinstance(config, RunConfig)
    assert config.workflow.name == "run_target"
    assert config.dataset.path == (tmp_path / "data" / "tasks.jsonl").resolve()
    assert config.dataset.metadata_keys == ("source", "difficulty")
    assert config.output.directory == (tmp_path / "runs" / "test").resolve()
    assert config.runtimes["primary"].options["seed"] == 7
    assert config.memory is None
    with pytest.raises(TypeError):
        config.runtimes["primary"].options["seed"] = 9  # type: ignore[index]


def test_run_config_accepts_opt_in_working_memory(tmp_path: Path) -> None:
    raw = {
        "version": 1,
        "workflow": _linear_config(),
        "dataset": {"path": "tasks.jsonl"},
        "runtimes": {"primary": {"type": "fake"}},
        "verifiers": {},
        "environment": {"type": "default"},
        "output": {"directory": "runs"},
        "memory": {"mode": "working", "top_k": 4},
    }

    config = parse_run_config(raw, base_dir=tmp_path)

    assert config.memory is not None
    assert config.memory.mode == "working"
    assert config.memory.top_k == 4


def test_run_config_accepts_independent_full_memory_profile(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "workflow": _linear_config(),
        "dataset": {"path": "tasks.jsonl"},
        "runtimes": {"primary": {"type": "fake"}},
        "verifiers": {},
        "environment": {"type": "default"},
        "output": {"directory": "runs"},
    }
    payload["memory"] = {"mode": "working", "profile": "full", "top_k": 5}

    config = parse_run_config(payload, base_dir=tmp_path)

    assert config.memory is not None
    assert config.memory.profile == "full"


def test_run_config_rejects_memory_off_profile_inside_enabled_block(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "workflow": _linear_config(),
        "dataset": {"path": "tasks.jsonl"},
        "runtimes": {"primary": {"type": "fake"}},
        "verifiers": {},
        "environment": {"type": "default"},
        "output": {"directory": "runs"},
    }
    payload["memory"] = {"profile": "off"}

    with pytest.raises(ConfigError, match="memory.profile"):
        parse_run_config(payload, base_dir=tmp_path)


def test_memory_off_keeps_the_pre_memory_config_digest_shape(tmp_path: Path) -> None:
    raw = {
        "version": 1,
        "workflow": _linear_config(),
        "dataset": {"path": "tasks.jsonl"},
        "runtimes": {"primary": {"type": "fake"}},
        "verifiers": {},
        "environment": {"type": "default"},
        "output": {"directory": "runs"},
    }
    config = parse_run_config(raw, base_dir=tmp_path)
    payload = workflow_config._jsonable(config)
    payload.pop("memory")
    payload.pop("output")
    payload["dataset"].pop("path")
    payload["dataset"].pop("task_payload_path")
    for key in ("concurrency", "limit", "resume"):
        payload["execution"].pop(key)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    assert (
        workflow_config.config_digest(config)
        == hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    )


def test_run_config_accepts_persistent_workflow_memory(tmp_path: Path) -> None:
    raw = {
        "version": 1,
        "workflow": _linear_config(),
        "dataset": {"path": "tasks.jsonl"},
        "runtimes": {"primary": {"type": "fake"}},
        "verifiers": {},
        "environment": {"type": "default"},
        "output": {"directory": "runs"},
        "memory": {
            "adapter": "robotics",
            "mode": "persistent",
            "persistent_backend": "mem0",
            "persistent_config": {
                "vector_store": {"provider": "chroma", "config": {"path": "memory"}}
            },
        },
    }

    config = parse_run_config(raw, base_dir=tmp_path)

    assert config.memory is not None
    assert config.memory.adapter == "robotics"
    assert config.memory.mode == "persistent"
    assert config.memory.persistent_backend == "mem0"
    assert config.memory.persistent_config == {
        "vector_store": {"provider": "chroma", "config": {"path": "memory"}}
    }
    with pytest.raises(TypeError):
        config.memory.persistent_config["vector_store"] = {}  # type: ignore[index]


def test_run_config_rejects_unknown_workflow_memory_adapter(tmp_path: Path) -> None:
    raw = {
        "version": 1,
        "workflow": _linear_config(),
        "dataset": {"path": "tasks.jsonl"},
        "runtimes": {"primary": {"type": "fake"}},
        "verifiers": {},
        "environment": {"type": "default"},
        "output": {"directory": "runs"},
        "memory": {"adapter": "unknown"},
    }

    with pytest.raises(ConfigError, match="memory.adapter"):
        parse_run_config(raw, base_dir=tmp_path)


@pytest.mark.parametrize(
    ("memory", "message"),
    [
        ({"mode": "persistent"}, "persistent_backend"),
        (
            {"mode": "persistent", "persistent_backend": "mem0"},
            "persistent_config",
        ),
        (
            {"mode": "working", "persistent_backend": "mem0", "persistent_config": {}},
            "requires memory.mode='persistent'",
        ),
    ],
)
def test_run_config_rejects_incomplete_persistent_workflow_memory(
    tmp_path: Path, memory: dict[str, object], message: str
) -> None:
    raw = {
        "version": 1,
        "workflow": _linear_config(),
        "dataset": {"path": "tasks.jsonl"},
        "runtimes": {"primary": {"type": "fake"}},
        "verifiers": {},
        "environment": {"type": "default"},
        "output": {"directory": "runs"},
        "memory": memory,
    }

    with pytest.raises(ConfigError, match=message):
        parse_run_config(raw, base_dir=tmp_path)


def test_run_config_does_not_accept_resource_fields_in_topology() -> None:
    raw = _linear_config()
    raw["dataset"] = {"path": "answers.jsonl"}

    with pytest.raises(ConfigError, match="unknown workflow config keys"):
        parse_workflow_config(raw)


def test_run_config_rejects_missing_role_target(tmp_path: Path) -> None:
    raw = {
        "version": 1,
        "workflow": _linear_config(),
        "dataset": {"path": "tasks.jsonl"},
        "runtimes": {},
        "verifiers": {},
        "environment": {"type": "default"},
        "output": {"directory": "runs"},
    }

    with pytest.raises(ConfigError, match="unknown runtime 'primary'"):
        parse_run_config(raw, base_dir=tmp_path)


def test_workflow_input_nested_metadata_is_immutable_and_json_serializable() -> None:
    metadata = {"source": {"name": "a"}, "tags": ["math", {"level": 2}]}

    workflow_input = WorkflowInput(input_id="one", problem="problem", metadata=metadata)

    assert json.loads(json.dumps(workflow_input.metadata)) == metadata
    with pytest.raises(TypeError, match="do not support mutation"):
        workflow_input.metadata["source"]["name"] = "changed"  # type: ignore[index]


def test_workflow_input_task_payload_is_immutable_and_separate_from_metadata() -> None:
    canary = "PRIVATE-WORKFLOW-INPUT-CANARY"
    payload = {"scene": {"seed": 7}, "objects": ["cup"], "answer": canary}

    workflow_input = WorkflowInput(
        input_id="robot-one",
        problem="pick up the cup",
        metadata={"source": "public"},
        task_payload=payload,
    )

    assert json.loads(json.dumps(workflow_input.task_payload)) == payload
    assert "scene" not in workflow_input.metadata
    assert canary not in repr(workflow_input)
    with pytest.raises(TypeError, match="do not support mutation"):
        workflow_input.task_payload["scene"]["seed"] = 9  # type: ignore[index]


def test_resource_options_support_deepcopy_and_explicit_mutable_thaw() -> None:
    resource = ResourceConfig(
        type="openai",
        options={
            "sampling": {
                "provider_options": {
                    "response_format": {"type": "json_schema"},
                    "stop": ["END"],
                }
            }
        },
    )

    copied = copy.deepcopy(dict(resource.options))
    assert copied == resource.options

    thawed = _thaw_json(resource.options)
    assert thawed == {
        "sampling": {
            "provider_options": {
                "response_format": {"type": "json_schema"},
                "stop": ["END"],
            }
        }
    }
    assert isinstance(thawed["sampling"]["provider_options"]["stop"], list)
    thawed["sampling"]["provider_options"]["response_format"]["type"] = "text"
    assert resource.options["sampling"]["provider_options"]["response_format"]["type"] == (
        "json_schema"
    )


def test_ensemble_requires_an_explicit_valid_replica_count() -> None:
    with pytest.raises(TypeError, match="replicas"):
        EnsembleConfig()  # type: ignore[call-arg]

    assert EnsembleConfig(replicas=2).replicas == 2

    declared = EnsembleConfig(replicas=3, selection_key="declared_answer")
    assert declared.selection_key == "declared_answer"
    with pytest.raises(ConfigError, match="selection_key"):
        EnsembleConfig(replicas=3, selection_key="python:callable")


def test_packaged_groups_compose_without_a_domain_example(tmp_path: Path) -> None:
    leaf = tmp_path / "fixture.yaml"
    leaf.write_text(
        """defaults:
  - /workflow
  - _self_
dataset:
  path: public.jsonl
output:
  directory: runs
""",
        encoding="utf-8",
    )
    raw = compose_config_mapping(leaf)
    config = load_run_config(leaf)
    assert config.workflow.version == 1
    assert "grading" not in raw
    assert config.environment.type == "default"


def test_cross_run_memory_changes_the_identity_without_a_domain_example(tmp_path: Path) -> None:
    leaf = tmp_path / "fixture.yaml"
    leaf.write_text(
        "defaults: [/workflow, _self_]\ndataset:\n  path: public.jsonl\n"
        "output:\n  directory: runs\n",
        encoding="utf-8",
    )
    config = dataclasses.replace(
        load_run_config(leaf),
        memory=MemoryConfig(),
    )
    enabled = dataclasses.replace(
        config,
        memory=MemoryConfig(
            mode="persistent",
            cross_run=True,
            persistent_backend="mem0",
            persistent_config={"version": "v1.1"},
        ),
    )
    assert config_digest(config) != config_digest(enabled)


@pytest.mark.parametrize("adapter", ["math", "bio"])
def test_removed_memory_adapter_is_rejected(adapter: str) -> None:
    with pytest.raises(ConfigError, match="built-in adapters"):
        MemoryConfig(adapter=adapter)
