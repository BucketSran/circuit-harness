"""Canonical resumable-run and aggregate-report contracts."""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from alphaapollo.common.trajectory.metrics import (
    METRICS_EVIDENCE_COMPLETE,
    METRICS_EVIDENCE_PARTIAL,
    METRICS_EVIDENCE_UNAVAILABLE,
    METRICS_REDUCER_VERSION,
    trajectory_digest,
)
from alphaapollo.common.trajectory.schemas import TRAJECTORY_SCHEMA_VERSION
from alphaapollo.workflows import scoring as scoring_module
from alphaapollo.workflows._resources import runtime as runtime_resources
from alphaapollo.workflows.config import ConfigError
from alphaapollo.workflows.main import main as run_cli
from alphaapollo.workflows.records import RunRecord, ScoredTask
from alphaapollo.workflows.report import (
    build_report,
    every_run_failed,
    format_report_markdown,
    mcnemar_exact_p,
)
from alphaapollo.workflows.run import run_batch


def _config(k: int, limit: int) -> dict:
    return {
        "version": 1,
        "workflow": {
            "version": 1,
            "name": "eval_test",
            "roles": [
                {
                    "id": "solver",
                    "target": "solver",
                    "system_prompt": "Solve using public data only.",
                }
            ],
            "steps": [{"id": "solve", "kind": "agent", "role": "solver", "output": True}],
            "entry_step": "solve",
            "transitions": [],
        },
        "dataset": {"path": "unused.jsonl", "format": "jsonl"},
        "runtimes": {
            "solver": {
                "type": "alphaapollo",
                "options": {
                    "backend": {"type": "fake", "options": {"text": "\\boxed{2}"}},
                    "model": "fake",
                    "sampling": {"temperature": 0.0, "max_tokens": 32},
                    "max_turns": 1,
                },
            }
        },
        "verifiers": {},
        "environment": {"type": "text_only", "options": {}},
        "output": {"directory": "unused-output", "format": "jsonl"},
        "execution": {
            "samples": k,
            "seed": 0,
            "limit": limit,
            "concurrency": 1,
        },
    }


def _problems() -> list[ScoredTask]:
    return [
        ScoredTask(id="a", problem="1+1?", gold_answer="2", metadata={"year": "2024"}),
        ScoredTask(id="b", problem="2+2?", gold_answer="4", metadata={"year": "2024"}),
    ]


def test_matrix_size_resume_and_no_resume(tmp_path: Path) -> None:
    config = _config(k=2, limit=2)
    records = run_batch(config, workdir=tmp_path, problems=_problems())
    assert len(records) == 4
    assert all(not record.resumed for record in records)

    resumed = run_batch(config, workdir=tmp_path, problems=_problems())
    assert all(record.resumed for record in resumed)

    rerun = run_batch(config, workdir=tmp_path, problems=_problems(), resume=False)
    assert all(not record.resumed for record in rerun)


def test_changed_task_payload_invalidates_resume(tmp_path: Path) -> None:
    config = _config(k=1, limit=1)
    public_path = tmp_path / "public.jsonl"
    private_path = tmp_path / "private.jsonl"
    public_path.write_text(
        json.dumps({"id": "one", "problem": "pick up the cup"}) + "\n",
        encoding="utf-8",
    )
    config["dataset"] = {
        "path": str(public_path),
        "format": "jsonl",
        "task_payload_path": str(private_path),
    }

    private_path.write_text(
        json.dumps({"id": "one", "env_payload": {"scene_seed": 1}}) + "\n",
        encoding="utf-8",
    )
    first = run_batch(config, workdir=tmp_path / "run")[0]
    assert first.resumed is False
    result_path = tmp_path / "run" / "no_tool" / "p0000" / "sample-000" / "result.json"
    first_fingerprint = json.loads(result_path.read_text(encoding="utf-8"))["problem_fingerprint"]

    private_path.write_text(
        json.dumps({"id": "one", "env_payload": {"scene_seed": 2}}) + "\n",
        encoding="utf-8",
    )
    changed = run_batch(config, workdir=tmp_path / "run")[0]

    assert changed.resumed is False
    changed_fingerprint = json.loads(result_path.read_text(encoding="utf-8"))["problem_fingerprint"]
    assert changed_fingerprint != first_fingerprint
    archived = result_path.parent / "attempts" / "attempt-000"
    archived_result = json.loads((archived / "result.json").read_text(encoding="utf-8"))
    assert archived_result["problem_fingerprint"] == first_fingerprint
    assert archived_result["trajectory_location"] == str((archived / "traj.jsonl").resolve())
    assert (archived / "canonical" / "workflow_results.jsonl").is_file()
    assert run_batch(config, workdir=tmp_path / "run")[0].resumed is True


def test_failed_cell_retry_gets_fresh_canonical_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(k=1, limit=1)
    original = scoring_module.run_workflow
    attempts = 0

    def fail_once(cell_config, *, inputs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            output = cell_config.output.directory
            output.mkdir(parents=True)
            (output / ".alphaapollo-workflow-run").write_text("1\n", encoding="utf-8")
            raise RuntimeError("scripted execution failure")
        return original(cell_config, inputs=inputs)

    monkeypatch.setattr(scoring_module, "run_workflow", fail_once)
    workdir = tmp_path / "run"

    failed = run_batch(config, workdir=workdir, problems=_problems()[:1])[0]
    retried = run_batch(config, workdir=workdir, problems=_problems()[:1])[0]

    run_dir = workdir / "no_tool" / "p0000" / "sample-000"
    archived = run_dir / "attempts" / "attempt-000"
    archived_result = json.loads((archived / "result.json").read_text(encoding="utf-8"))
    assert failed.termination_reason == "error"
    assert retried.termination_reason != "error"
    assert archived_result["error"] == "RuntimeError: scripted execution failure"
    assert (archived / "canonical" / ".alphaapollo-workflow-run").is_file()
    assert (run_dir / "canonical" / "workflow_results.jsonl").is_file()


def test_result_binds_versioned_trajectory_metrics(tmp_path: Path) -> None:
    run_batch(_config(k=1, limit=1), workdir=tmp_path, problems=_problems()[:1])
    run_dir = tmp_path / "no_tool" / "p0000" / "sample-000"
    result = json.loads((run_dir / "result.json").read_text())
    events = [json.loads(line) for line in (run_dir / "traj.jsonl").read_text().splitlines()]

    assert result["trajectory_schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert result["metrics_reducer_version"] == METRICS_REDUCER_VERSION
    assert len(result["trajectory_digest"]) == 64
    assert events
    assert all(event["trajectory_schema_version"] == TRAJECTORY_SCHEMA_VERSION for event in events)


def test_stale_or_tampered_metrics_refresh_without_model_rerun(tmp_path: Path) -> None:
    config = _config(k=1, limit=1)
    run_batch(config, workdir=tmp_path, problems=_problems()[:1])
    run_dir = tmp_path / "no_tool" / "p0000" / "sample-000"
    result_path = run_dir / "result.json"
    trajectory_before = (run_dir / "traj.jsonl").read_bytes()
    stale = json.loads(result_path.read_text())
    stale["metrics_reducer_version"] = METRICS_REDUCER_VERSION - 1
    stale["prompt_tokens"] = 999
    result_path.write_text(json.dumps(stale))

    record = run_batch(config, workdir=tmp_path, problems=_problems()[:1])[0]

    assert record.resumed is True
    assert record.prompt_tokens != 999
    assert (run_dir / "traj.jsonl").read_bytes() == trajectory_before
    assert json.loads(result_path.read_text())["prompt_tokens"] != 999


def test_invalid_or_changed_cache_is_rerun(tmp_path: Path) -> None:
    config = _config(k=1, limit=1)
    run_batch(config, workdir=tmp_path, problems=_problems()[:1])
    result_path = tmp_path / "no_tool" / "p0000" / "sample-000" / "result.json"
    legacy = json.loads(result_path.read_text())
    legacy.pop("trajectory_schema_version")
    result_path.write_text(json.dumps(legacy))
    assert run_batch(config, workdir=tmp_path, problems=_problems()[:1])[0].resumed is False

    changed = copy.deepcopy(config)
    changed["runtimes"]["solver"]["options"]["sampling"]["max_tokens"] = 64
    assert run_batch(changed, workdir=tmp_path, problems=_problems()[:1])[0].resumed is False


def test_shards_and_reverse_cover_the_same_matrix(tmp_path: Path) -> None:
    config = _config(k=2, limit=2)

    def cells(records: list[RunRecord]) -> set[tuple[str, str, int]]:
        return {(record.problem_id, record.condition, record.seed) for record in records}

    shards = [
        run_batch(config, workdir=tmp_path / "shards", problems=_problems(), shard=(i, 3))
        for i in range(3)
    ]
    union: set[tuple[str, str, int]] = set()
    for shard in shards:
        selected = cells(shard)
        assert union.isdisjoint(selected)
        union |= selected
    forward = cells(run_batch(config, workdir=tmp_path / "forward", problems=_problems()))
    reverse = cells(
        run_batch(config, workdir=tmp_path / "reverse", problems=_problems(), reverse=True)
    )
    assert union == forward == reverse


def test_legacy_tool_and_backend_injections_fail_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="legacy execution injection"):
        run_batch(
            _config(k=1, limit=1),
            workdir=tmp_path,
            problems=_problems()[:1],
            tool_executor_factory=object,
        )
    assert list(tmp_path.iterdir()) == []


def _rec(
    pid: str,
    condition: str,
    seed: int,
    correct: bool | None,
    *,
    tool_failures: int = 0,
    python_failures: int = 0,
    python_repairs: int = 0,
) -> RunRecord:
    return RunRecord(
        problem_id=pid,
        problem_index=0,
        condition=condition,
        seed=seed,
        final_answer="x",
        correct=correct,
        verdict="pass",
        rounds=1,
        termination_reason="verifier_pass",
        solver_tool_calls=1 if condition.startswith("python_code") else 0,
        verifier_tool_calls=0,
        solver_tool_failures=tool_failures,
        verifier_tool_failures=0,
        solver_python_tool_failures=python_failures,
        verifier_python_tool_failures=0,
        solver_python_tool_repairs=python_repairs,
        verifier_python_tool_repairs=0,
        prompt_tokens=10,
        completion_tokens=5,
        wall_seconds=0.1,
        trajectory_location="loc",
        resumed=False,
    )


def test_report_pass_rates_and_paired_delta() -> None:
    records = [
        _rec("1", "python_code", 0, True),
        _rec("1", "python_code", 1, True),
        _rec("1", "no_tool", 0, False),
        _rec("1", "no_tool", 1, False),
        _rec("2", "python_code", 0, True),
        _rec("2", "python_code", 1, False),
        _rec("2", "no_tool", 0, True),
        _rec("2", "no_tool", 1, False),
    ]
    report = build_report(records, model="fake", config_digest="d", seed_base=0, k=2)

    assert report["conditions"]["python_code"]["pass_at_1"] == pytest.approx(3 / 4)
    assert report["conditions"]["no_tool"]["pass_at_1"] == pytest.approx(1 / 4)
    assert report["paired"]["mcnemar"]["b_python_only_correct"] == 2
    assert report["paired"]["mcnemar"]["c_no_tool_only_correct"] == 0
    assert report["paired"]["aggregate_pass_at_1_delta"] == pytest.approx(0.5)


def test_report_labels_incomplete_token_counts_as_lower_bounds() -> None:
    complete = _rec("1", "no_tool", 0, True)
    partial = replace(
        complete,
        seed=1,
        prompt_tokens=0,
        completion_tokens=0,
        metrics_evidence_status=METRICS_EVIDENCE_PARTIAL,
        metrics_evidence_gaps=("model_usage.prompt_tokens: missing",),
    )
    unavailable = replace(
        complete,
        seed=2,
        prompt_tokens=0,
        completion_tokens=0,
        metrics_evidence_status=METRICS_EVIDENCE_UNAVAILABLE,
    )

    report = build_report(
        [complete, partial, unavailable],
        model="fake",
        config_digest="d",
        seed_base=0,
        k=3,
    )
    tokens = report["conditions"]["no_tool"]["tokens"]

    assert tokens["prompt"] == 10
    assert tokens["completion"] == 5
    assert tokens["evidence_status"] == METRICS_EVIDENCE_PARTIAL
    assert tokens["evidence_counts"] == {
        METRICS_EVIDENCE_COMPLETE: 1,
        METRICS_EVIDENCE_PARTIAL: 1,
        METRICS_EVIDENCE_UNAVAILABLE: 1,
    }
    assert tokens["counts_are_lower_bounds"] is True
    assert "lower bound" in format_report_markdown(report)


def _failed(pid: str, seed: int) -> RunRecord:
    """A cell that never produced an answer, as `_failure_record` writes it."""

    return replace(
        _rec(pid, "no_tool", seed, None),
        final_answer=None,
        verdict="error",
        rounds=0,
        termination_reason="error",
        metrics_evidence_status=METRICS_EVIDENCE_UNAVAILABLE,
    )


def test_a_report_in_which_nothing_ran_says_so_before_its_rates() -> None:
    """0.000 by absence and 0.000 by measurement must not look the same.

    An unreachable server or a model name the server does not serve fails every
    cell this way, and the pass@1 that follows is not a score.
    """

    report = build_report(
        [_failed("1", 0), _failed("2", 0)],
        model="fake",
        config_digest="d",
        seed_base=0,
        k=1,
    )
    markdown = format_report_markdown(report)

    assert every_run_failed(report) is True
    assert report["conditions"]["no_tool"]["pass_at_1"] == 0.0
    assert report["conditions"]["no_tool"]["n_error_runs"] == 2
    assert "NO RESULT: all 2 run(s) ended in a typed error" in markdown
    # Ahead of the number it is warning about.
    assert markdown.index("NO RESULT") < markdown.index("pass@1")


def test_the_pass_at_k_column_names_the_run_s_own_k() -> None:
    """The header must not claim a sample count the run did not draw.

    ``pass_at_k`` is "solved by at least one of this problem's runs", so its k is
    ``execution.samples``. The column was labelled ``pass@32`` unconditionally,
    which read as a claim about the run rather than as a column name. k=1 is in
    the sweep because that is where a naive substitution prints two columns both
    headed ``pass@1``.
    """

    for k in (1, 2, 32):
        report = build_report(
            [_rec("1", "no_tool", seed, True) for seed in range(k)],
            model="fake",
            config_digest="d",
            seed_base=0,
            k=k,
        )
        header = format_report_markdown(report).split("\n")

        assert f"| condition | pass@1 | pass@k (k={k}) |" in "\n".join(header)
        columns = next(line for line in header if line.startswith("| condition |")).split("|")
        assert len(columns) == len(set(column.strip() for column in columns)) + 1


def test_the_retries_exhausted_section_has_a_reader_but_no_producer() -> None:
    """``RunRecord.retries_exhausted`` is reserved, not live. Pin both halves.

    #160's DAG engine wrote it from its ``plan_audit``; retiring that engine left
    the reader in ``report`` without a producer, so no run can currently reach
    this section. Keeping the reader honest means proving it still renders what
    it promises, and proving that nothing on the canonical path fills the field --
    otherwise the next sweep of unread fields deletes the reader and the signal
    with it.
    """

    record = replace(
        _rec("1", "no_tool", 0, True),
        plan_mode="ensemble",
        n_branches=2,
        retries_exhausted=("checker",),
    )

    report = build_report([record], model="fake", config_digest="d", seed_base=0, k=1)
    markdown = format_report_markdown(report)

    assert report["plan_mode"]["retries_exhausted"]["n_runs"] == 1
    assert report["plan_mode"]["retries_exhausted"]["runs"][0]["verifiers"] == ["checker"]
    assert "retries exhausted: 1 run(s)" in markdown
    assert "not clean solves" in markdown
    # No canonical producer: `scoring.run_cell` builds every RunRecord field it
    # knows about by name, and this one is not among them.
    assert "retries_exhausted" not in inspect.getsource(scoring_module)


def test_a_partly_failed_run_is_an_ordinary_result() -> None:
    """One failed cell among many is what the typed-fail column is for."""

    report = build_report(
        [_failed("1", 0), _rec("2", "no_tool", 0, True)],
        model="fake",
        config_digest="d",
        seed_base=0,
        k=1,
    )

    assert every_run_failed(report) is False
    assert "NO RESULT" not in format_report_markdown(report)


def test_unscoreable_counts_against_pass_at_1() -> None:
    records = [_rec("1", "no_tool", 0, None), _rec("1", "no_tool", 1, True)]
    summary = build_report(records, model="fake", config_digest="d", seed_base=0, k=2)[
        "conditions"
    ]["no_tool"]
    assert summary["n_unscoreable"] == 1
    assert summary["pass_at_1"] == pytest.approx(0.5)


def test_feedback_ablation_preserves_tool_id_and_repair_semantics() -> None:
    records = [
        _rec("1", "python_code", 0, True, python_failures=1, python_repairs=1),
        _rec("2", "python_code", 0, False, python_failures=1),
        _rec("1", "python_code_generic_feedback", 0, False, python_failures=1),
        _rec("2", "python_code_generic_feedback", 0, False, python_failures=1),
        _rec("1", "python_code_missing_feedback", 0, False, python_failures=1),
        _rec("2", "python_code_missing_feedback", 0, None, python_failures=1),
    ]
    ablation = build_report(records, model="fake", config_digest="d", seed_base=0, k=1)[
        "feedback_ablation"
    ]

    assert ablation["primary_metric"] == "repair_success_rate"
    assert ablation["differentiated"] == pytest.approx(0.5)
    assert ablation["generic"] == pytest.approx(0.0)
    assert ablation["missing"] == pytest.approx(0.0)


def test_feedback_ablation_distinguishes_no_opportunity_from_no_repair() -> None:
    records = [
        _rec("1", "python_code", 0, True),
        _rec("1", "python_code_generic_feedback", 0, False, python_failures=10),
        _rec(
            "1",
            "python_code_missing_feedback",
            0,
            False,
            python_failures=1,
            python_repairs=1,
        ),
    ]
    report = build_report(records, model="fake", config_digest="d", seed_base=0, k=1)
    ablation = report["feedback_ablation"]
    assert ablation["differentiated"] is None
    assert ablation["generic"] == pytest.approx(0.0)
    assert ablation["missing"] == pytest.approx(1.0)


def test_feedback_ablation_ignores_non_python_failures() -> None:
    records = [
        _rec("1", condition, 0, False, tool_failures=1)
        for condition in (
            "python_code",
            "python_code_generic_feedback",
            "python_code_missing_feedback",
        )
    ]
    report = build_report(records, model="fake", config_digest="d", seed_base=0, k=1)
    for condition in (
        "python_code",
        "python_code_generic_feedback",
        "python_code_missing_feedback",
    ):
        tool = report["conditions"][condition]["tool"]
        assert tool["solver_failures"] == 1
        assert tool["solver_python_failures"] == 0
        assert tool["repair_opportunities"] == 0


def _verifier_only(config: dict) -> dict:
    """Strip ``_config``'s solver step, leaving a workflow that declares only a verifier."""

    config["workflow"]["roles"] = [
        {
            "id": "judge",
            "target": "judge",
            "system_prompt": "Return a JSON verdict for this public candidate.",
            "input_template": "Problem: {problem}\nCandidate: {candidate}",
            "output_format": "json",
        }
    ]
    config["workflow"]["steps"] = [
        {"id": "verify", "kind": "verifier", "role": "judge", "output": True}
    ]
    config["workflow"]["entry_step"] = "verify"
    config["runtimes"] = {"judge_runtime": config["runtimes"]["solver"]}
    config["verifiers"] = {"judge": {"type": "agent", "options": {"runtime": "judge_runtime"}}}
    return config


def _write_run(tmp_path: Path, config: dict) -> Path:
    dataset = tmp_path / "public.jsonl"
    dataset.write_text(
        json.dumps({"id": "a", "problem": "What is one plus one?"}) + "\n",
        encoding="utf-8",
    )
    config["dataset"]["path"] = str(dataset)
    config["output"]["directory"] = str(tmp_path / "out")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_verifier_only_run_is_refused_before_any_backend_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The defined outcome for an empty agent-target list is a refusal, not a report.

    ``report.resource_summary`` reads ``agent_targets[0]`` for the solver runtime
    it must name, and a verifier-only workflow leaves that list empty. Reporting
    such a run honestly is not available: with no agent step nothing produces the
    candidate, so the executor verifies the problem statement against itself and
    the ledger records the problem text as the run's ``final_answer``. The
    topology is therefore refused, and refused early enough that the operator
    pays for nothing -- asserted here by a backend builder that fails if reached,
    rather than by reading the message.
    """

    built: list[object] = []

    def must_not_build(*args: object, **_kwargs: object) -> object:
        # Recorded as well as raised: ``main`` reports a failed run without a
        # traceback, so a bare raise here would only change the exit code.
        built.append(args)
        raise AssertionError("backend must not be built for a refused topology")

    monkeypatch.setattr(runtime_resources, "_build_backend", must_not_build)
    config_path = _write_run(tmp_path, _verifier_only(_config(k=2, limit=1)))

    exit_code = run_cli(["--config", str(config_path), "--quiet"])

    assert built == []
    assert exit_code == 2
    error = capsys.readouterr().err
    assert "at least one agent step" in error
    assert not list((tmp_path / "out").rglob("result.json"))
    assert not (tmp_path / "out" / "report.md").exists()


def test_solver_bearing_run_still_renders_its_report(tmp_path: Path) -> None:
    """Regression: the refusal above leaves an ordinary solver workflow untouched."""

    config_path = _write_run(tmp_path, _config(k=2, limit=1))

    assert run_cli(["--config", str(config_path), "--quiet"]) == 0

    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["provenance"]["model"] == "solver=fake,verifier=none"
    assert format_report_markdown(report) == (tmp_path / "out" / "report.md").read_text(
        encoding="utf-8"
    )


def test_mcnemar_exact_values() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(3, 0) == pytest.approx(0.25)
    assert mcnemar_exact_p(5, 0) == pytest.approx(0.0625)
    assert mcnemar_exact_p(2, 2) == 1.0


@pytest.mark.parametrize("stored_status", [METRICS_EVIDENCE_COMPLETE, None])
def test_cache_over_reported_event_gap_is_rerun(
    tmp_path: Path,
    stored_status: str | None,
) -> None:
    config = _config(k=1, limit=1)
    run_batch(config, workdir=tmp_path, problems=_problems()[:1])
    run_dir = tmp_path / "no_tool" / "p0000" / "sample-000"
    result_path = run_dir / "result.json"
    traj_path = run_dir / "traj.jsonl"
    event = json.loads(traj_path.read_text().splitlines()[0])
    event.update(event_id="lost-event-report", sequence=None, type="EvidenceAdded")
    event["payload"] = {
        "environment_event": "environment_closed",
        "actor": "solver",
        "cleanup_errors": [],
        "event_errors": ["tool_execution_completed: OSError: event recording failed"],
    }
    traj_path.write_text(traj_path.read_text() + json.dumps(event) + "\n")
    result = json.loads(result_path.read_text())
    result["trajectory_digest"] = trajectory_digest(traj_path)
    if stored_status is None:
        result.pop("metrics_evidence_status", None)
    else:
        result["metrics_evidence_status"] = stored_status
    result_path.write_text(json.dumps(result))

    assert run_batch(config, workdir=tmp_path, problems=_problems()[:1])[0].resumed is False
