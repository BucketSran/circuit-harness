from __future__ import annotations

import ast
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import alphaapollo.workflows.data as projection_module
import alphaapollo.workflows.resources as workflow_resources
import alphaapollo.workflows.scoring as cell_module
from alphaapollo.common.grader import get_grader
from alphaapollo.common.trajectory.metrics import (
    METRICS_EVIDENCE_COMPLETE,
    METRICS_REDUCER_VERSION,
    materialize_trajectory_metrics,
)
from alphaapollo.common.trajectory.schemas import TRAJECTORY_SCHEMA_VERSION
from alphaapollo.data_preprocess import read_records
from alphaapollo.data_preprocess.prepare_custom_data import prepare as prepare_fixture_dataset
from alphaapollo.reasoning.runtime import AgentResult, AgentTurn
from alphaapollo.reasoning.verification import VerificationResult
from alphaapollo.workflows._resources import runtime as runtime_resources
from alphaapollo.workflows.config import (
    ConfigError,
    RunConfig,
    parse_run_config,
)
from alphaapollo.workflows.main import main as run_cli
from alphaapollo.workflows.records import (
    RunRecord,
    ScoredTask,
    StepResult,
    WorkflowInput,
    WorkflowResult,
)
from alphaapollo.workflows.report import _max_rounds, build_report, format_report_markdown
from alphaapollo.workflows.run import run, run_batch
from alphaapollo.workflows.selection import select_result


def test_repeated_run_modules_do_not_import_the_legacy_stack() -> None:
    package = Path(cell_module.__file__).parent
    forbidden = tuple(
        f"alphaapollo.reasoning.{name}"
        for name in ("workflows", "runtimes", "loop", "session", "solver", "verifier")
    )
    violations: list[str] = []
    modules = ("config.py", "data.py", "main.py", "records.py", "run.py", "scoring.py")
    for name in modules:
        tree = ast.parse((package / name).read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        violations.extend(
            f"{name}: {module}"
            for module in imports
            if any(module == root or module.startswith(f"{root}.") for root in forbidden)
        )
    assert violations == []


def _problem(problem_id: str = "p", *, gold: str = "917") -> ScoredTask:
    return ScoredTask(
        id=problem_id,
        problem=f"Public problem {problem_id}: compute one plus one.",
        gold_answer=gold,
        metadata={"year": "2024"},
    )


def _mapping(tmp_path: Path, *, k: int = 1, concurrency: int = 1) -> dict:
    return {
        "version": 1,
        "workflow": {
            "version": 1,
            "name": "eval_test",
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
        "dataset": {"path": str(tmp_path / "unused.jsonl")},
        "runtimes": {
            "solver": {
                "type": "alphaapollo",
                "options": {
                    "backend": {
                        "type": "fake",
                        "options": {"text": "The final answer is 2"},
                    },
                    "model": "offline",
                    "sampling": {"temperature": 0.0, "max_tokens": 32},
                    "max_turns": 1,
                },
            }
        },
        "verifiers": {},
        "environment": {"type": "text_only", "options": {}},
        "output": {"directory": str(tmp_path / "configured")},
        "execution": {
            "samples": k,
            "seed": 0,
            "concurrency": concurrency,
        },
    }


def test_repeated_run_does_not_require_private_scoring_data(tmp_path: Path) -> None:
    public = tmp_path / "public.jsonl"
    public.write_text(
        json.dumps({"id": "p", "problem": "What is one plus one?"}) + "\n",
        encoding="utf-8",
    )
    raw = _mapping(tmp_path, k=2)
    raw["dataset"]["path"] = str(public)
    raw["output"]["directory"] = str(tmp_path / "run")

    report = run(parse_run_config(raw))

    summary = report["conditions"]["no_tool"]
    assert summary["n_runs"] == 2
    assert summary["n_unscoreable"] == 2
    assert summary["n_correct"] == 0


def test_scoring_contract_fails_before_execution_for_unknown_grader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _mapping(tmp_path)
    raw["scoring"] = {
        "dataset_root": str(tmp_path),
        "source_id": "private_data",
        "version": "v1",
    }
    task = SimpleNamespace(
        task_uid="p",
        statement="problem",
        answer="2",
        extra={},
        source_uri="local",
        grader_id="missing_grader",
    )
    monkeypatch.setattr(projection_module, "read_prepared", lambda *_args: [task])

    with pytest.raises(ConfigError, match="unknown grader_id"):
        projection_module.align_public_inputs_with_private_gold(
            parse_run_config(raw),
            [WorkflowInput("p", "problem")],
        )


def test_scoring_contract_fails_before_execution_for_an_environment_graded_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``environment_success`` resolves, so the old ``get_grader`` probe passed
    # and the refusal surfaced once per cell inside the scorer's per-cell
    # ``except Exception`` -- an hour of matrix for a config error (#256).
    raw = _mapping(tmp_path)
    raw["scoring"] = {
        "dataset_root": str(tmp_path),
        "source_id": "private_data",
        "version": "v1",
    }
    task = SimpleNamespace(
        task_uid="p",
        statement="problem",
        answer="environment",
        extra={},
        source_uri="local",
        grader_id="environment_success",
    )
    monkeypatch.setattr(projection_module, "read_prepared", lambda *_args: [task])

    with pytest.raises(ConfigError, match="execution backend's success"):
        projection_module.align_public_inputs_with_private_gold(
            parse_run_config(raw),
            [WorkflowInput("p", "problem")],
        )


def test_canonical_batch_preserves_result_and_trajectory_metrics_contract(
    tmp_path: Path,
) -> None:
    config = parse_run_config(_mapping(tmp_path))
    records = run_batch(config, workdir=tmp_path / "run", problems=[_problem()])

    assert len(records) == 1
    record = records[0]
    assert record.resumed is False
    assert record.prompt_tokens > 0
    assert record.completion_tokens > 0
    result_path = tmp_path / "run/no_tool/p0000/sample-000/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["result_schema_version"] == 4
    assert result["trajectory_schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert result["metrics_reducer_version"] == METRICS_REDUCER_VERSION
    assert result["metrics_evidence_status"] == "complete"
    assert result["prompt_tokens"] == record.prompt_tokens

    trajectory = Path(record.trajectory_location).read_text(encoding="utf-8")
    canonical = (result_path.parent / "canonical/workflow_results.jsonl").read_text(
        encoding="utf-8"
    )
    # The gold answer must not leak into recorded text — but only as a value
    # standing on its own: timestamps, durations, token counts, and hex
    # digests routinely contain the digits as a substring, which made a plain
    # substring check flaky across runs.
    gold_leak = re.compile(r"(?<![0-9a-fA-F.])917(?![0-9a-fA-F])")
    assert not gold_leak.search(trajectory)
    assert not gold_leak.search(canonical)
    assert result["gold_answer"] == "917"

    resumed = run_batch(config, workdir=tmp_path / "run", problems=[_problem()])
    assert resumed[0].resumed is True


def test_fresh_batch_returns_partial_usage_evidence_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runtime_resources._FakeBackend._generate_one

    def without_usage(self, request):  # noqa: ANN001, ANN202 - private test seam
        return replace(original(self, request), usage={})

    monkeypatch.setattr(runtime_resources._FakeBackend, "_generate_one", without_usage)
    config = parse_run_config(_mapping(tmp_path))

    record = run_batch(config, workdir=tmp_path / "run", problems=[_problem()])[0]
    persisted = json.loads(
        (tmp_path / "run/no_tool/p0000/sample-000/result.json").read_text(encoding="utf-8")
    )

    assert record.prompt_tokens == record.completion_tokens == 0
    assert record.metrics_evidence_status == "partial"
    assert record.metrics_evidence_gaps
    assert persisted["metrics_evidence_status"] == record.metrics_evidence_status
    assert persisted["metrics_evidence_gaps"] == list(record.metrics_evidence_gaps)


@pytest.mark.parametrize(
    ("usage", "expected_status", "expected_prompt", "has_gap"),
    [
        (
            {"prompt_tokens": 0, "completion_tokens": 0},
            "complete",
            0,
            False,
        ),
        ({}, "partial", 0, True),
        (
            {"prompt_tokens": "unknown", "completion_tokens": 2},
            "partial",
            0,
            True,
        ),
    ],
)
def test_canonical_usage_projection_distinguishes_zero_from_missing_or_invalid(
    tmp_path: Path,
    usage: dict[str, object],
    expected_status: str,
    expected_prompt: int,
    has_gap: bool,
) -> None:
    transition = SimpleNamespace(
        observation="",
        reward=0.0,
        done=True,
        termination_reason="model_output",
        metadata={},
    )
    turn = AgentTurn(
        index=0,
        generation_request=object(),
        generation_response=SimpleNamespace(usage=usage),
        environment_transition=transition,
    )
    agent = AgentResult(
        task_id="usage:solve:1:0",
        final_text="The final answer is 2",
        turns=(turn,),
    )
    outcome = WorkflowResult(
        input_id="usage",
        workflow_name="usage-evidence",
        steps=(
            StepResult(
                input_id="usage",
                step_id="solve",
                role="solver",
                iteration=1,
                branch_index=0,
                output=agent,
            ),
        ),
        output=agent,
        selected_step_id="solve",
        selected_branch_index=0,
        status="completed",
    )
    trajectory = tmp_path / "usage.jsonl"

    projection_module.write_trajectory_projection(outcome, trajectory, session_id="usage")
    materialized = materialize_trajectory_metrics({}, trajectory, status=METRICS_EVIDENCE_COMPLETE)

    assert materialized["prompt_tokens"] == expected_prompt
    assert materialized["metrics_evidence_status"] == expected_status
    assert ("metrics_evidence_gaps" in materialized) is has_gap


@pytest.mark.parametrize(
    "agent_result",
    [
        None,
        AgentResult(
            task_id="verifier-agent",
            final_text='{"verdict":"pass","feedback":"accepted"}',
        ),
    ],
)
def test_eval_scores_bound_candidate_from_any_terminal_verifier(
    agent_result: AgentResult | None,
) -> None:
    result = VerificationResult(
        request_id="verify",
        verdict="pass",
        candidate="Reasoning. The final answer is 42",
        agent_result=agent_result,
    )

    assert cell_module._output_text(result) == result.candidate


def test_batch_propagates_only_verdict_bound_to_final_agent_output(tmp_path: Path) -> None:
    config = parse_run_config(_mapping(tmp_path))
    verified = AgentResult(task_id="draft", final_text="The final answer is 42")
    verification = VerificationResult(
        request_id="verify",
        verdict="pass",
        candidate=verified.final_text,
        candidate_ref=verified.task_id,
    )
    finalizer = AgentResult(task_id="final", final_text="The final answer is 917")
    copied_finalizer = AgentResult(task_id="final-copy", final_text=verified.final_text)
    rewritten_source = AgentResult(task_id=verified.task_id, final_text=finalizer.final_text)

    def outcome(output: AgentResult) -> WorkflowResult:
        return WorkflowResult(
            input_id="p0000",
            workflow_name="binding-test",
            steps=(
                StepResult("p0000", "draft", "solver", 1, 0, verified),
                StepResult("p0000", "verify", "verifier", 1, 0, verification),
                StepResult("p0000", "final", "finalizer", 1, 0, output),
            ),
            output=output,
            selected_step_id="final",
            selected_branch_index=0,
            status="completed",
        )

    rewritten = cell_module._metrics(outcome(finalizer), config)
    copied = cell_module._metrics(outcome(copied_finalizer), config)
    changed_candidate = cell_module._metrics(outcome(rewritten_source), config)
    bound = cell_module._metrics(
        WorkflowResult(
            input_id="p0000",
            workflow_name="binding-test",
            steps=(
                StepResult("p0000", "draft", "solver", 1, 0, verified),
                StepResult("p0000", "verify", "verifier", 1, 0, verification),
            ),
            output=verified,
            selected_step_id="draft",
            selected_branch_index=0,
            status="completed",
        ),
        config,
    )

    assert rewritten.final_answer == "917"
    assert rewritten.verdict == "inconclusive"
    assert copied.final_answer == "42"
    assert copied.verdict == "inconclusive"
    assert changed_candidate.final_answer == "917"
    assert changed_candidate.verdict == "inconclusive"
    assert bound.final_answer == "42"
    assert bound.verdict == "pass"


@pytest.mark.parametrize(
    ("producer_reason", "expected"),
    [("final", "verified"), ("max_turns", "max_turns")],
)
def test_a_terminal_verifier_does_not_relabel_how_the_solution_ended(
    tmp_path: Path, producer_reason: str, expected: str
) -> None:
    """`verified` describes the verifier, and the report asks about the solver.

    With a terminal verifier the last thing that runs is always a clean verifier
    call, so reading its status alone would retire the report's `max_turns` and
    `error` counters: a finalizer that exhausted its turn budget and one that
    finished cleanly would both read as `verified`.
    """

    config = parse_run_config(_mapping(tmp_path))
    finalizer = AgentResult(
        task_id="final",
        final_text="The final answer is 42",
        termination_reason=producer_reason,
    )
    certification = VerificationResult(
        request_id="certify",
        verdict="pass",
        candidate=finalizer.final_text,
        candidate_ref=finalizer.task_id,
    )

    metrics = cell_module._metrics(
        WorkflowResult(
            input_id="p0000",
            workflow_name="terminal-verifier",
            steps=(
                StepResult("p0000", "final", "finalizer", 1, 0, finalizer),
                StepResult("p0000", "certify", "verifier", 1, 0, certification),
            ),
            output=certification,
            selected_step_id="certify",
            selected_branch_index=0,
            status="completed",
        ),
        config,
    )

    assert metrics.verdict == "pass"
    assert metrics.final_answer == "42"
    assert metrics.termination_reason == expected


# A chain of thought cut off by ``max_tokens``: no ``\boxed{}``, no "the answer
# is", no trailing standalone integer. Captured shape of the live 8000-token run
# that motivated the contract below; shortened here, since only the absence of an
# answer matters, not the length.
_NO_ANSWER_OUTPUT = (
    "Let me work through this carefully. If n is even the parity argument gives a "
    "contradiction, so n must be odd. Write n = 2k + 1 and substitute back into the "
    "original relation; that yields a quadratic in k whose discriminant I still have "
    "to evaluate, and before that I should redo the case analysis, because the second "
    "branch may have been dropped prematurely. Let me redo the substitution"
)


def _clipped_turn(finish_reason: str) -> AgentTurn:
    return AgentTurn(
        index=0,
        generation_request=object(),
        generation_response=SimpleNamespace(finish_reason=finish_reason, usage={}),
        environment_transition=SimpleNamespace(
            observation="",
            reward=0.0,
            done=True,
            termination_reason="model_output",
            metadata={},
        ),
    )


def test_a_terminal_verifier_does_not_hide_that_the_scored_answer_was_cut_off(
    tmp_path: Path,
) -> None:
    """The fact belongs to the answer's producer, not to whatever ran last.

    With a terminal verifier the last thing that runs is a short, clean verifier
    reply, so reading truncation off the run's output would report ``False`` for
    a run whose finalizer was clipped mid-sentence. That is the same resolution
    ``termination_reason`` already needed, which is why both facts are read from
    one ``_scored_agent_result``.
    """

    config = parse_run_config(_mapping(tmp_path))
    clipped_finalizer = AgentResult(
        task_id="final",
        final_text="<answer>Final answer: 42</answer>, and the remaining case to check is",
        turns=(_clipped_turn("length"),),
    )
    certification = VerificationResult(
        request_id="certify",
        verdict="pass",
        candidate=clipped_finalizer.final_text,
        candidate_ref=clipped_finalizer.task_id,
        agent_result=AgentResult(
            task_id="certify-agent",
            final_text="VERDICT: PASS",
            turns=(_clipped_turn("stop"),),
        ),
    )

    metrics = cell_module._metrics(
        WorkflowResult(
            input_id="p0000",
            workflow_name="terminal-verifier",
            steps=(
                StepResult("p0000", "final", "finalizer", 1, 0, clipped_finalizer),
                StepResult("p0000", "certify", "verifier", 1, 0, certification),
            ),
            output=certification,
            selected_step_id="certify",
            selected_branch_index=0,
            status="completed",
        ),
        config,
    )

    assert metrics.output_truncated is True
    # The Environment's own verdict is untouched: it still describes a finalizer
    # that ended cleanly as far as the Environment could tell.
    assert metrics.termination_reason == "verified"
    # And the answer stated before the cut is still the scored one.
    assert metrics.final_answer == "42"


def _shipped_preset_mapping(tmp_path: Path, preset: str, *, verdict: str) -> dict:
    """Run a packaged solver preset end to end against the offline fake backend."""

    raw = _mapping(tmp_path)
    raw["workflow"] = str(
        Path(cell_module.__file__).parents[1] / "configs" / "preset" / f"{preset}.yaml"
    )
    raw["runtimes"]["solver"]["options"]["backend"]["options"]["text"] = (
        "Reasoning complete. Final answer: <answer>\\boxed{917}</answer>"
    )
    raw["runtimes"]["verifier_runtime"] = {
        "type": "alphaapollo",
        "options": {
            "backend": {
                "type": "fake",
                "options": {"verdict": verdict, "feedback": "checked $\\sqrt{2} \\le 2$"},
            },
            "model": "offline-verifier",
            "sampling": {"temperature": 0.0, "max_tokens": 64},
            "max_turns": 1,
        },
    }
    raw["verifiers"] = {"verifier": {"type": "agent", "options": {"runtime": "verifier_runtime"}}}
    return raw


@pytest.mark.parametrize("preset", ["vanilla_reasoning", "bash_reasoning", "python_code_reasoning"])
@pytest.mark.parametrize("verdict", ["pass", "fail"])
def test_the_recorded_verdict_is_the_one_the_verifier_gave(
    tmp_path: Path, preset: str, verdict: str
) -> None:
    """Measured on `main`: 11 verifier judgments, 0 of them reached `result.json`.

    Every shipped solver preset ended on an agent finalizer, and
    `_final_output_verification` only returns a judgment bound to the selected
    output -- so the finalizer's rewrite made the binding unreachable and every
    row recorded `inconclusive`. The terminal certification step judges the
    finalizer's own text, so the binding holds by construction.

    `certified` stays `False` on purpose: `VerificationResult` only accepts a
    certificate with `trust_level >= 2`, zero false-positive risk, and a
    witness, and `AgentVerifier` supplies none of them. Passing is not
    certifying, and that invariant is what keeps the difference.
    """

    raw = _shipped_preset_mapping(tmp_path, preset, verdict=verdict)

    record = run_batch(
        parse_run_config(raw),
        workdir=tmp_path / "run",
        problems=[_problem(gold="917")],
    )[0]

    assert record.verdict == verdict
    assert record.certified is False
    assert record.unreadable_verifier_outputs == 0
    assert record.termination_reason == "verified"
    # The judged text is still the scoreable one: scoring reads the bound
    # candidate, not the verifier's own prose.
    assert record.final_answer == "917"
    assert record.correct is True
    # `pass` goes straight to the finalizer; `fail` spends both revision edges,
    # re-running `revise` once and `verify` twice.
    assert record.repeated_steps == (0 if verdict == "pass" else 3)
    assert record.rounds == (2 if verdict == "pass" else 4)


def test_an_unreadable_verifier_reply_is_recorded_apart_from_an_unsure_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`inconclusive` must not mean both "unsure" and "we could not read it".

    Six of the eleven measured judgments were discarded at parse. The record was
    honest at the data layer -- `feedback` and `details.parse_error` said so --
    but nothing a reader sees distinguished them from a model that deliberated
    and declined to decide.
    """

    raw = _shipped_preset_mapping(tmp_path, "vanilla_reasoning", verdict="pass")
    unreadable = "The verifier's considered opinion is that $\\sqrt{2} \\le 2$."
    # Answer the verifier prompt with prose instead of the format it demanded.
    original_generate_one = runtime_resources._FakeBackend._generate_one

    def unparseable_verifier_reply(backend: object, request: object):  # noqa: ANN001, ANN202
        response = original_generate_one(backend, request)
        if "VERDICT:" not in response.content:
            return response
        return replace(response, content=unreadable)

    monkeypatch.setattr(runtime_resources._FakeBackend, "_generate_one", unparseable_verifier_reply)
    record = run_batch(
        parse_run_config(raw),
        workdir=tmp_path / "run",
        problems=[_problem(gold="917")],
    )[0]

    assert record.verdict == "inconclusive"
    assert record.unreadable_verifier_outputs == record.rounds > 0

    report = build_report(
        [record],
        model="offline",
        config_digest="digest",
        seed_base=0,
        k=1,
    )
    verifier_summary = report["conditions"][record.condition]["verifier"]

    assert verifier_summary["verdict_counts"] == {"INCONCLUSIVE": 1}
    assert verifier_summary["n_unreadable_verifier_outputs"] == record.rounds
    assert verifier_summary["n_runs_with_unreadable_verifier_output"] == 1
    assert "unreadable verifier outputs" in format_report_markdown(report)


def _no_answer_config(tmp_path: Path) -> RunConfig:
    raw = _mapping(tmp_path)
    raw["runtimes"]["solver"]["options"]["backend"]["options"]["text"] = _NO_ANSWER_OUTPUT
    return parse_run_config(raw)


@pytest.mark.parametrize(
    ("grader_id", "gold"),
    [
        ("exact_match", "204"),
        ("exact_match", "204"),
    ],
)
def test_output_with_no_extractable_answer_is_unscoreable_under_every_grader(
    tmp_path: Path,
    grader_id: str,
    gold: str,
) -> None:
    """ "No answer" must not be recorded as "wrong answer" under any grader.

    ``final_answer`` used to fall back to the entire model output when extraction
    found nothing, and the graders then split on it: ``exact_match`` and
    ``math_expression`` returned ``False`` for a whole transcript while
    ``integer_answer`` returned ``None``. One failure was therefore classified two
    ways according to the dataset's ``grader_id``, which does not carry that
    distinction. ``None`` reaches every shipped grader's unscoreable branch.
    """

    config = _no_answer_config(tmp_path)
    task = ScoredTask(id="p", problem="Find n.", gold_answer=gold, grader_id=grader_id)

    record = run_batch(config, workdir=tmp_path / "run", problems=[task])[0]
    result_path = tmp_path / "run/no_tool/p0000/sample-000/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert record.final_answer is None
    assert record.correct is None
    assert result["final_answer"] is None
    assert result["correct"] is None
    # The transcript is the evidence for why extraction found nothing, so removing
    # it from the graded field must not remove it from the run. ``traj.jsonl`` is a
    # token/tool ledger and has never carried the text; the canonical projection
    # written beside ``result.json`` does, and stays the one place it lives.
    assert _NO_ANSWER_OUTPUT not in Path(record.trajectory_location).read_text(encoding="utf-8")
    canonical = result_path.parent / "canonical/workflow_results.jsonl"
    assert _NO_ANSWER_OUTPUT in canonical.read_text(encoding="utf-8")


def test_report_counts_an_unextractable_answer_as_unscoreable_not_incorrect(
    tmp_path: Path,
) -> None:
    # ``exact_match`` is one of the two graders that used to score the transcript
    # ``False``, so this cell landed in neither the unscoreable column nor the
    # typed-failure column -- it silently depressed pass@1 as a wrong answer.
    config = _no_answer_config(tmp_path)
    task = ScoredTask(id="p", problem="Find n.", gold_answer="204", grader_id="exact_match")

    records = run_batch(config, workdir=tmp_path / "run", problems=[task])
    report = build_report(records, model="offline", config_digest="d", seed_base=0, k=1)
    summary = report["conditions"]["no_tool"]

    assert summary["n_runs"] == 1
    assert summary["n_correct"] == 0
    assert summary["n_unscoreable"] == 1
    # Not a typed failure either: the cell ran to completion and produced output.
    assert summary["n_error_runs"] == 0
    # First ``| no_tool`` row is the per-condition table:
    # condition | pass@1 | pass@k | problems | correct | unscoreable | ...
    row = next(
        line
        for line in format_report_markdown(report).splitlines()
        if line.startswith("| no_tool ")
    ).split("|")
    assert row[5].strip() == "0"
    assert row[6].strip() == "1"


def _clip_every_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the offline fake backend answer with ``finish_reason: length``.

    The fake backend has no knob for this and must not grow one: a decoding
    detail that only tests set would read as configuration while changing
    nothing in any shipped run. Patching the one response it builds keeps the
    rest of the run -- content, usage, identity checks -- exactly as it is.
    """

    generate_one = runtime_resources._FakeBackend._generate_one

    def clipped(self: object, request: object) -> object:
        return replace(generate_one(self, request), finish_reason="length")

    monkeypatch.setattr(runtime_resources._FakeBackend, "_generate_one", clipped)


def test_a_cut_off_reply_is_recorded_as_cut_off_without_changing_its_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record the truncation; do not let it touch extraction, grading, or the verdict.

    A model may state its answer and then ramble until the sampler clips it. That
    answer is valid, so the clipped run must differ from the clean one in exactly
    one field. ``termination_reason`` in particular keeps reporting what the
    Environment decided, which never saw the ``finish_reason`` at all.
    """

    config = parse_run_config(_mapping(tmp_path))
    task = ScoredTask(id="p", problem="1+1?", gold_answer="2")

    clean = run_batch(config, workdir=tmp_path / "clean", problems=[task])[0]

    _clip_every_reply(monkeypatch)
    clipped = run_batch(config, workdir=tmp_path / "clipped", problems=[task])[0]
    ledger = json.loads(
        (tmp_path / "clipped/no_tool/p0000/sample-000/result.json").read_text(encoding="utf-8")
    )

    assert clean.output_truncated is False
    assert clipped.output_truncated is True
    assert ledger["output_truncated"] is True
    assert (clipped.final_answer, clipped.correct) == (clean.final_answer, clean.correct)
    assert (clipped.final_answer, clipped.correct) == ("2", True)
    assert clipped.termination_reason == clean.termination_reason
    assert clipped.verdict == clean.verdict


def test_the_report_counts_a_cut_off_reply_apart_from_every_other_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clipped reply is the one column a reader fixes by raising a number."""

    _clip_every_reply(monkeypatch)
    config = parse_run_config(_mapping(tmp_path))
    task = ScoredTask(id="p", problem="1+1?", gold_answer="2")

    records = run_batch(config, workdir=tmp_path / "run", problems=[task])
    report = build_report(records, model="offline", config_digest="d", seed_base=0, k=1)
    summary = report["conditions"]["no_tool"]

    assert summary["n_truncated_runs"] == 1
    # No existing column absorbed it: this cell ran, answered, and scored right.
    assert summary["n_error_runs"] == 0
    assert summary["n_max_turns_runs"] == 0
    assert summary["n_unscoreable"] == 0
    assert summary["n_correct"] == 1
    # condition | pass@1 | pass@k | problems | correct | unscoreable |
    #   typed-fail (err) | max-turns | cut-off | revised | ...
    row = next(
        line
        for line in format_report_markdown(report).splitlines()
        if line.startswith("| no_tool ")
    ).split("|")
    assert row[9].strip() == "1"


def test_an_output_that_states_an_answer_is_scored_from_that_answer(tmp_path: Path) -> None:
    raw = _mapping(tmp_path)
    raw["runtimes"]["solver"]["options"]["backend"]["options"]["text"] = (
        f"{_NO_ANSWER_OUTPUT}\nFinal answer: 204"
    )
    task = ScoredTask(id="p", problem="Find n.", gold_answer="204")

    record = run_batch(parse_run_config(raw), workdir=tmp_path / "run", problems=[task])[0]

    assert record.final_answer == "204"
    assert record.correct is True


@pytest.mark.parametrize("stale_version", [1, 2, 3])
def test_a_pre_bump_result_ledger_is_rerun_rather_than_resumed(
    tmp_path: Path, stale_version: int
) -> None:
    """A pre-bump ledger states a claim the current shape forbids.

    A v1 ``result.json`` could hold the whole model output in ``final_answer``
    with a ``correct`` graded from it. Resuming one would keep publishing
    "answered wrongly" for a run that answered nothing. A v2 ledger predates the
    terminal certification step, so its ``rounds`` counts a different set of
    steps and it carries neither ``unreadable_verifier_outputs`` nor
    ``revisions``; reading it back would publish 0 for measurements it never took.
    A v3 ledger carries no ``output_truncated``, and defaulting it would claim
    "this reply was not cut off" about a run that never looked -- the one thing
    that separates a model which stated no answer from one the sampler clipped.
    """

    config = _no_answer_config(tmp_path)
    task = ScoredTask(id="p", problem="Find n.", gold_answer="204", grader_id="exact_match")
    run_batch(config, workdir=tmp_path / "run", problems=[task])
    result_path = tmp_path / "run/no_tool/p0000/sample-000/result.json"
    stale = json.loads(result_path.read_text(encoding="utf-8"))
    stale.update(result_schema_version=stale_version, final_answer=_NO_ANSWER_OUTPUT, correct=False)
    if stale_version == 2:
        stale.pop("unreadable_verifier_outputs", None)
        stale.pop("revisions", None)
    if stale_version == 3:
        stale.pop("output_truncated", None)
    result_path.write_text(json.dumps(stale), encoding="utf-8")

    resumed = run_batch(config, workdir=tmp_path / "run", problems=[task])[0]

    assert resumed.resumed is False
    assert resumed.correct is None
    assert json.loads(result_path.read_text(encoding="utf-8"))["correct"] is None


def test_exact_text_vote_keeps_grouping_branches_with_no_extractable_answer() -> None:
    """The vote audit mirrors ``selection._output_key``, not scoring's rule.

    Under ``exact_text`` the stripped output text *is* the declared selection key,
    so branches with no extractable answer still vote and still elect a winner.
    Dropping the fallback here as ``_metrics`` does would file those branches under
    ``abstained`` and make the report's ``n_no_winner`` claim every branch declined
    -- describing a different election from the one that chose ``outcome.output``.
    """

    outputs = (
        AgentResult(task_id="branch-0", final_text=_NO_ANSWER_OUTPUT),
        AgentResult(task_id="branch-1", final_text=_NO_ANSWER_OUTPUT),
        AgentResult(task_id="branch-2", final_text="  "),
    )
    selected = select_result(outputs, strategy="majority_vote", selection_key="exact_text")
    outcome = WorkflowResult(
        input_id="vote",
        workflow_name="ensemble",
        steps=tuple(
            StepResult(
                input_id="vote",
                step_id="solve",
                role="solver",
                iteration=1,
                branch_index=index,
                output=output,
            )
            for index, output in enumerate(outputs)
        ),
        output=selected,
        selected_step_id="solve",
        selected_branch_index=outputs.index(selected),
        status="completed",
    )

    vote = cell_module._ensemble_vote(
        outcome,
        selection_key="exact_text",
        extract=get_grader("exact_match").extract,
    )

    assert selected is outputs[0]
    assert vote["winner"] == _NO_ANSWER_OUTPUT
    assert vote["groups"] == [
        {"answer": _NO_ANSWER_OUTPUT, "count": 2, "branches": [0, 1]},
    ]
    assert vote["abstained"] == [2]


def test_eval_declared_answer_vote_groups_equivalent_answers_and_abstains_empty() -> None:
    outputs = (
        AgentResult(task_id="branch-0", final_text="Reasoning A. The final answer is 42"),
        AgentResult(task_id="branch-1", final_text=r"Reasoning B. \boxed{\frac{84}{2}}"),
        AgentResult(task_id="branch-2", final_text="  "),
    )
    outcome = WorkflowResult(
        input_id="vote",
        workflow_name="ensemble",
        steps=tuple(
            StepResult(
                input_id="vote",
                step_id="solve",
                role="solver",
                iteration=1,
                branch_index=index,
                output=output,
            )
            for index, output in enumerate(outputs)
        ),
        output=outputs[0],
        selected_step_id="solve",
        selected_branch_index=0,
        status="completed",
    )

    vote = cell_module._ensemble_vote(
        outcome,
        selection_key="declared_answer",
        extract=get_grader("exact_match").extract,
    )

    assert vote["winner"] == "42"
    assert vote["groups"] == [{"answer": "42", "count": 2, "branches": [0, 1]}]
    assert vote["abstained"] == [2]


def test_terminal_verifier_vote_audit_matches_candidate_selection() -> None:
    outputs = (
        VerificationResult("verify-0", "pass", "Final answer: A"),
        VerificationResult("verify-1", "pass", "Reasoning B1. Final answer: B"),
        VerificationResult("verify-2", "fail", "Reasoning B2. Final answer: B"),
    )
    selected = select_result(
        outputs,
        strategy="majority_vote",
        selection_key="declared_answer",
    )
    outcome = WorkflowResult(
        input_id="verifier-vote",
        workflow_name="ensemble",
        steps=tuple(
            StepResult(
                input_id="verifier-vote",
                step_id="verify",
                role="verifier",
                iteration=1,
                branch_index=index,
                output=output,
            )
            for index, output in enumerate(outputs)
        ),
        output=selected,
        selected_step_id="verify",
        selected_branch_index=outputs.index(selected),
        status="completed",
    )

    vote = cell_module._ensemble_vote(
        outcome,
        selection_key="declared_answer",
        extract=get_grader("exact_match").extract,
    )

    assert selected is outputs[0]
    assert vote == {
        "winner": "A",
        "groups": [
            {"answer": "A", "count": 1, "branches": [0]},
            {"answer": "B", "count": 1, "branches": [1]},
        ],
        "abstained": [2],
    }


def test_eval_uses_declared_public_dataset_and_binds_private_gold(tmp_path: Path) -> None:
    source = tmp_path / "fixture.jsonl"
    source.write_text(
        json.dumps({"problem_idx": 6, "problem": "What is 1 + 1?", "answer": 2}) + "\n",
        encoding="utf-8",
    )
    dataset_root = tmp_path / "prepared"
    prepared = prepare_fixture_dataset(
        question_key="problem",
        answer_key="answer",
        id_key="problem_idx",
        data_source=str(source),
        output_root=dataset_root,
        dataset_name="fixture_test",
        revision=None,
        splits=("test",),
        output_format="jsonl",
    )
    public_path = prepared.public_splits["test"]
    private = read_records(prepared.private_splits["test"])[0]
    task_uid = private["task_uid"]

    raw = _mapping(tmp_path)
    raw["dataset"] = {
        "path": str(public_path),
        "format": "jsonl",
        "split": "test",
        "input_key": "statement",
        "id_key": "task_uid",
        "metadata_keys": [],
    }
    raw["scoring"] = {
        "dataset_root": str(dataset_root),
        "source_id": "fixture_test",
        "version": "v1",
    }
    config = parse_run_config(raw)

    records = run_batch(config, workdir=tmp_path / "declared")

    assert len(records) == 1
    assert records[0].problem_id == task_uid
    canonical = (
        tmp_path / "declared/no_tool/p0000/sample-000/canonical/workflow_results.jsonl"
    ).read_text(encoding="utf-8")
    assert task_uid in canonical

    # A public file that keeps the task_uid but carries a different statement must
    # never be scored against the prepared gold. Tampering with the prepared file
    # itself is caught earlier by the manifest digest, so the substituted file is
    # written outside the prepared build.
    substituted = tmp_path / "substituted.jsonl"
    substituted.write_text(
        json.dumps({"task_uid": task_uid, "split": "test", "statement": "a different problem"})
        + "\n",
        encoding="utf-8",
    )
    raw["dataset"]["path"] = str(substituted)
    with pytest.raises(ConfigError, match="does not match private scoring statement"):
        run_batch(parse_run_config(raw), workdir=tmp_path / "mismatch")
    assert not (tmp_path / "mismatch").exists()


def test_resume_repairs_tampered_metrics_without_model_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = parse_run_config(_mapping(tmp_path))
    run_batch(config, workdir=tmp_path / "run", problems=[_problem()])
    result_path = tmp_path / "run/no_tool/p0000/sample-000/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected = result["prompt_tokens"]
    result["prompt_tokens"] = 999_999
    result_path.write_text(json.dumps(result), encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("resume unexpectedly reran the model")

    monkeypatch.setattr(cell_module, "run_workflow", forbidden)
    resumed = run_batch(config, workdir=tmp_path / "run", problems=[_problem()])
    assert resumed[0].resumed is True
    assert resumed[0].prompt_tokens == expected
    assert json.loads(result_path.read_text(encoding="utf-8"))["prompt_tokens"] == expected


@pytest.mark.parametrize(
    "legacy",
    [
        {"version": 1, "workflow": {"mode": "custom"}},
        {"version": 1, "workflow": {"preset": "ensemble", "ensemble": {"n": 5}}},
        {"version": 1, "tools": {"solver": ["python_code"]}},
        {"version": 1, "verifier": {"kind": "lean4", "lean4": {}}},
    ],
)
def test_legacy_run_recipes_fail_loud(legacy: dict) -> None:
    with pytest.raises(ConfigError):
        parse_run_config(legacy)


def test_legacy_mapping_is_rejected_before_workdir_or_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = {
        "version": 1,
        "solver": {"backend": "fake"},
        "verifier": {"kind": "llm_as_judge", "backend": "fake"},
        "matrix": {"conditions": ["no_tool"]},
    }
    workdir = tmp_path / "must-not-exist"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy config reached canonical resource composition")

    monkeypatch.setattr(cell_module, "run_workflow", forbidden)
    with pytest.raises(ConfigError):
        run_batch(legacy, workdir=workdir, problems=[_problem()])
    assert not workdir.exists()


def test_execution_does_not_accept_a_redundant_condition_label(tmp_path: Path) -> None:
    raw = _mapping(tmp_path)
    raw["execution"]["conditions"] = ["python_code"]

    with pytest.raises(ConfigError, match="unknown run.execution keys"):
        parse_run_config(raw)


def test_shards_and_reverse_preserve_the_same_matrix(tmp_path: Path) -> None:
    config = parse_run_config(_mapping(tmp_path, k=2, concurrency=2))
    problems = [_problem("a"), _problem("b")]

    def cells(records):
        return {(record.problem_index, record.condition, record.seed) for record in records}

    forward = run_batch(config, workdir=tmp_path / "forward", problems=problems)
    reverse = run_batch(config, workdir=tmp_path / "reverse", problems=problems, reverse=True)
    shard_zero = run_batch(config, workdir=tmp_path / "shards", problems=problems, shard=(0, 2))
    shard_one = run_batch(config, workdir=tmp_path / "shards", problems=problems, shard=(1, 2))
    assert cells(forward) == cells(reverse)
    assert cells(shard_zero).isdisjoint(cells(shard_one))
    assert cells(shard_zero) | cells(shard_one) == cells(forward)


def test_failure_after_provider_work_is_not_materialized_as_complete_zero_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = parse_run_config(_mapping(tmp_path))

    class FailedAfterOneTurn(RuntimeError):
        turns_completed = 1

    def fail_after_one_turn(*_args, **_kwargs):
        raise FailedAfterOneTurn("provider failed after one completed turn")

    monkeypatch.setattr(cell_module, "run_workflow", fail_after_one_turn)
    records = run_batch(config, workdir=tmp_path / "partial", problems=[_problem()])

    assert records[0].termination_reason == "error"
    result = json.loads(
        (tmp_path / "partial/no_tool/p0000/sample-000/result.json").read_text(encoding="utf-8")
    )
    assert result["metrics_evidence_status"] == "unavailable"
    assert result["prompt_tokens"] == result["completion_tokens"] == 0
    assert result["metrics_evidence_status"] != "complete"
    assert "FailedAfterOneTurn" in result["error"]


def test_max_rounds_uses_transition_iteration_bound(tmp_path: Path) -> None:
    raw = _mapping(tmp_path)
    raw["workflow"] = {
        "version": 1,
        "name": "bounded_transition",
        "roles": [
            {"id": "solver", "target": "solver", "system_prompt": "Solve."},
            {
                "id": "judge",
                "target": "judge",
                "system_prompt": "Check.",
            },
        ],
        "steps": [
            {"id": "draft", "kind": "agent", "role": "solver"},
            {
                "id": "verify",
                "kind": "verifier",
                "role": "judge",
            },
            {
                "id": "revise",
                "kind": "agent",
                "role": "solver",
            },
            {
                "id": "final",
                "kind": "agent",
                "role": "solver",
                "output": True,
            },
        ],
        "entry_step": "draft",
        "transitions": [
            {"source": "draft", "target": "verify"},
            {"source": "verify", "target": "final", "condition": "passed"},
            {
                "source": "verify",
                "target": "revise",
                "condition": "not_passed",
                "max_iterations": 2,
            },
            {"source": "verify", "target": "final", "condition": "not_passed"},
            {"source": "revise", "target": "verify"},
        ],
    }
    raw["verifiers"] = {"judge": {"type": "lean4", "options": {}}}
    config = parse_run_config(raw)
    assert _max_rounds(config) == 3


def test_unified_cli_reports_missing_public_dataset_as_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = _mapping(tmp_path)
    config_path = tmp_path / "eval.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    assert run_cli(["--config", str(config_path), "--quiet"]) == 2
    captured = capsys.readouterr()
    assert "configuration error:" in captured.err
    assert "could not read prepared JSONL dataset" in captured.err
    assert "Traceback" not in captured.err


def test_scoring_rejects_execution_fields(tmp_path: Path) -> None:
    raw = _mapping(tmp_path)
    raw["scoring"] = {"conditions": ["no_tool"], "samples": 2}

    with pytest.raises(ConfigError, match="unknown run.scoring keys"):
        parse_run_config(raw)


def test_certification_is_recorded_separately_from_the_verdict(tmp_path: Path) -> None:
    """A passing verdict is weaker than a certification.

    Only certification claims the answer was proven, so the two are recorded as
    distinct fields; collapsing them would hide verifier false positives.
    """

    config = parse_run_config(_mapping(tmp_path))
    answer = AgentResult(task_id="draft", final_text="The final answer is 42")

    def outcome(certified: bool) -> WorkflowResult:
        # Certification is contract-guarded: it needs a passing verdict, a
        # trusted checker, zero false-positive risk, and a witness.
        verification = VerificationResult(
            request_id="verify",
            verdict="pass",
            candidate=answer.final_text,
            candidate_ref=answer.task_id,
            trust_level=2 if certified else 0,
            false_positive_risk=0.0 if certified else 1.0,
            witness=object() if certified else None,
            certified=certified,
        )
        return WorkflowResult(
            input_id="p0000",
            workflow_name="certification-test",
            steps=(
                StepResult("p0000", "draft", "solver", 1, 0, answer),
                StepResult("p0000", "verify", "verifier", 1, 0, verification),
            ),
            output=answer,
            selected_step_id="draft",
            selected_branch_index=0,
            status="completed",
        )

    uncertified = cell_module._metrics(outcome(False), config)
    certified = cell_module._metrics(outcome(True), config)

    assert uncertified.verdict == certified.verdict == "pass"
    assert uncertified.certified is False
    assert certified.certified is True


def test_report_counts_certified_answers_that_score_wrong() -> None:
    """A verifier that certifies a wrong answer must be visible in the report.

    This is the failure the benchmark most needs to surface; without its own
    counter it reads as ordinary model error.
    """

    def record(problem_id: str, *, certified: bool, correct: bool | None) -> RunRecord:
        return RunRecord(
            problem_id=problem_id,
            problem_index=0,
            condition="no_tool",
            seed=0,
            final_answer="42",
            correct=correct,
            verdict="pass",
            certified=certified,
            rounds=1,
            termination_reason="model_output",
            solver_tool_calls=0,
            verifier_tool_calls=0,
            solver_tool_failures=0,
            verifier_tool_failures=0,
            prompt_tokens=1,
            completion_tokens=1,
            wall_seconds=0.0,
            trajectory_location="traj.jsonl",
            resumed=False,
        )

    report = build_report(
        [
            record("p0", certified=True, correct=True),
            record("p1", certified=True, correct=False),
            record("p2", certified=True, correct=None),
            record("p3", certified=False, correct=False),
        ],
        model="m",
        config_digest="d",
        seed_base=0,
        k=1,
        code_sha="sha",
        decoding={},
    )
    summary = report["conditions"]["no_tool"]

    assert summary["n_certified"] == 3
    assert summary["n_certified_correct"] == 1
    # Unscoreable-but-certified counts too: it was claimed proven and was not.
    assert summary["n_false_positive_certifications"] == 2


def test_scored_run_refuses_an_environment_that_grades_in_loop(tmp_path: Path) -> None:
    """Gold must never reach the process running the model during scoring.

    Accepting the option and quietly withholding gold would be worse: the config
    would read as honoured while doing nothing.
    """

    raw = _mapping(tmp_path)
    raw["environment"] = {"type": "default", "options": {"grader_id": "exact_match"}}

    with pytest.raises(ConfigError, match="environment.options contains unknown keys"):
        workflow_resources.validate_composition_config(parse_run_config(raw))


def test_scored_run_accepts_an_environment_without_in_loop_grading(tmp_path: Path) -> None:
    raw = _mapping(tmp_path)
    raw["environment"] = {"type": "default", "options": {"max_steps": 4}}

    config = parse_run_config(raw)
    workflow_resources.validate_composition_config(config)
    assert config.environment.options["max_steps"] == 4
