# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run one sample cell, then score it out of band after Workflow execution.

This module is the repeated-run cell boundary that invokes ``run_workflow``.
Gold is applied strictly after execution and never enters ``WorkflowInput`` or
canonical persistence, so the public/private separation established by the
prepared dataset survives all the way to the report.

Completed ``WorkflowResult`` children are projected into the historical
``traj.jsonl`` envelope so the reducer stays the sole source of persisted
token/tool metrics. Partial failure evidence is deliberately not claimed: a
Workflow exception is marked ``unavailable`` and forced to rerun rather than
recorded as a zero-cost ledger.

``final_answer`` records only what the dataset's grader could extract. A run
that stated no answer records ``None`` and grades to ``None`` (unscoreable), not
``False``. The evidence for *why* extraction found nothing is the transcript,
which lives beside ``result.json`` in ``canonical/workflow_results.jsonl`` and
``canonical/trajectories.jsonl``; ``traj.jsonl`` is a token/tool ledger and has
never carried the text.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from alphaapollo.common.grader import get_grader
from alphaapollo.common.trajectory.metrics import (
    METRICS_EVIDENCE_COMPLETE,
    METRICS_EVIDENCE_PARTIAL,
    METRICS_EVIDENCE_UNAVAILABLE,
    TrajectoryEvidenceError,
    reduce_trajectory_evidence,
)
from alphaapollo.reasoning.runtime import AgentResult
from alphaapollo.reasoning.verification import VerificationResult
from alphaapollo.workflows.config import (
    ConfigError,
    RunConfig,
    coerce_config,
    run_config_for_cell,
)
from alphaapollo.workflows.data import Cell, write_trajectory_projection
from alphaapollo.workflows.records import (
    RunRecord,
    StepResult,
    WorkflowInput,
    WorkflowResult,
    _failure_record,
    _load_record,
    _problem_fingerprint,
    _write_result,
)
from alphaapollo.workflows.run import run_workflow
from alphaapollo.workflows.selection import (
    declared_answers_equivalent,
    extract_declared_answer,
)

__all__ = ["run_cell", "supported_plan_modes"]


def supported_plan_modes() -> frozenset[str]:
    """Plan labels representable by the current canonical Workflow DSL."""

    return frozenset({"single", "ensemble"})


def _workflow_selection(config: RunConfig) -> str:
    if config.workflow.ensemble is not None:
        return "ensemble"
    return "single"


def run_cell(
    config: object,
    cell: Cell,
    *,
    root: Path,
    tool_executor_factory: object | None,
    resume: bool,
    expected_digest: str,
    solver_backend: object | None = None,
    verifier_backend: object | None = None,
) -> RunRecord:
    """Execute one cell; extra legacy injection arguments fail before execution."""

    normalized = coerce_config(config)
    if any(
        value is not None for value in (tool_executor_factory, solver_backend, verifier_backend)
    ):
        raise ConfigError(
            "eval cells no longer accept legacy backend/tool injections; use canonical "
            "RunConfig resources"
        )
    run_dir = root / cell.condition / f"p{cell.problem_index:04d}" / f"sample-{cell.seed:03d}"
    result_path = run_dir / "result.json"
    traj_path = run_dir / "traj.jsonl"
    fingerprint = _problem_fingerprint(cell.problem, cell.public_input)
    selection = _workflow_selection(normalized)
    expected_plan_mode = None if selection == "single" else selection

    if resume and result_path.exists():
        cached = _load_record(
            result_path,
            cell,
            expected_digest,
            fingerprint,
            traj_path,
            expected_plan_mode,
        )
        if cached is not None:
            return cached

    _archive_previous_attempt(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    if traj_path.exists():
        traj_path.write_text("", encoding="utf-8")
    session_id = f"p{cell.problem_index:04d}-{cell.condition}-s{cell.seed:03d}"
    # ``cell.problem`` is a ``ScoredTask``: it holds the gold answer. Everything
    # in a ``WorkflowInput`` reaches the model, so this reads the task's own
    # public projection rather than picking fields off it here. The rule for what
    # may cross lives on ``ScoredTask.public_view``, next to the gold field, where
    # someone adding a field to that class will see it.
    workflow_input = (
        cell.public_input
        if cell.public_input is not None
        else WorkflowInput(
            input_id=session_id,
            problem=cell.problem.problem,
            metadata=cell.problem.public_view(),
        )
    )
    cell_config = run_config_for_cell(
        normalized,
        run_dir=run_dir,
        condition=cell.condition,
        seed=cell.seed,
    )

    start = time.perf_counter()
    try:
        outcomes = run_workflow(cell_config, inputs=(workflow_input,))
        if len(outcomes) != 1:
            raise RuntimeError(
                f"canonical run_workflow returned {len(outcomes)} rows for one eval cell"
            )
        outcome = outcomes[0]
        write_trajectory_projection(outcome, traj_path, session_id=session_id)
    except Exception as exc:  # noqa: BLE001 - a live cell failure is durable and retryable
        wall = time.perf_counter() - start
        # ``run_workflow`` currently returns only complete WorkflowResult objects;
        # it has no injected recorder/on-step callback that could expose turns from
        # a failed execution.  Do not fabricate a complete zero-cost ledger for a
        # failure that may have happened after provider work.  The row is explicitly
        # unavailable and is never resumable.  A true partial lower bound is held for
        # #186's canonical recorder contract.
        record = _failure_record(
            cell,
            wall,
            traj_path,
            expected_plan_mode,
            metrics_evidence_status=(
                METRICS_EVIDENCE_PARTIAL
                if traj_path.exists() and traj_path.stat().st_size
                else METRICS_EVIDENCE_UNAVAILABLE
            ),
        )
        _write_result(
            result_path,
            record,
            cell,
            expected_digest,
            fingerprint,
            error=f"{type(exc).__name__}: {exc}",
        )
        return record
    wall = time.perf_counter() - start

    scoring_enabled = normalized.scoring is not None or cell.injected
    metrics = _metrics(
        outcome,
        normalized,
        grader_id=cell.problem.grader_id if scoring_enabled else None,
    )
    try:
        trajectory_metrics, evidence_gaps = reduce_trajectory_evidence(traj_path)
        metrics = replace(metrics, **asdict(trajectory_metrics))
        metrics_evidence_status = (
            METRICS_EVIDENCE_PARTIAL if evidence_gaps else METRICS_EVIDENCE_COMPLETE
        )
    except TrajectoryEvidenceError as exc:
        record = _failure_record(
            cell,
            wall,
            traj_path,
            expected_plan_mode,
            metrics_evidence_status=METRICS_EVIDENCE_UNAVAILABLE,
        )
        _write_result(
            result_path,
            record,
            cell,
            expected_digest,
            fingerprint,
            error=f"trajectory metrics: {exc}",
        )
        return record
    correct: bool | None = None
    if scoring_enabled:
        try:
            correct = get_grader(cell.problem.grader_id).grade(
                metrics.final_answer,
                cell.problem.gold_answer,
            )
        except Exception as exc:  # noqa: BLE001 - scoring failures remain cell-local
            record = _failure_record(
                cell,
                wall,
                traj_path,
                expected_plan_mode,
                metrics_evidence_status=metrics_evidence_status,
            )
            _write_result(
                result_path,
                record,
                cell,
                expected_digest,
                fingerprint,
                error=f"scoring: {type(exc).__name__}: {exc}",
            )
            return record

    record = RunRecord(
        problem_id=cell.problem.id,
        problem_index=cell.problem_index,
        condition=cell.condition,
        seed=cell.seed,
        final_answer=metrics.final_answer,
        correct=correct,
        verdict=metrics.verdict,
        certified=metrics.certified,
        rounds=metrics.rounds,
        unreadable_verifier_outputs=metrics.unreadable_verifier_outputs,
        repeated_steps=metrics.repeated_steps,
        termination_reason=metrics.termination_reason,
        output_truncated=metrics.output_truncated,
        solver_tool_calls=metrics.solver_tool_calls,
        verifier_tool_calls=metrics.verifier_tool_calls,
        solver_tool_failures=metrics.solver_tool_failures,
        verifier_tool_failures=metrics.verifier_tool_failures,
        solver_python_tool_failures=metrics.solver_python_tool_failures,
        verifier_python_tool_failures=metrics.verifier_python_tool_failures,
        solver_python_tool_repairs=metrics.solver_python_tool_repairs,
        verifier_python_tool_repairs=metrics.verifier_python_tool_repairs,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        wall_seconds=wall,
        trajectory_location=str(traj_path.resolve()),
        resumed=False,
        plan_mode=metrics.plan_mode,
        n_branches=metrics.n_branches,
        vote=metrics.vote,
        metrics_evidence_status=metrics_evidence_status,
        metrics_evidence_gaps=evidence_gaps,
    )
    _write_result(
        result_path,
        record,
        cell,
        expected_digest,
        fingerprint,
        error=None,
    )
    return record


def _archive_previous_attempt(run_dir: Path) -> Path | None:
    """Rotate one invalidated or failed cell attempt without losing evidence.

    ``run_workflow`` refuses to write into a non-empty canonical directory.
    Repeated scoring intentionally revisits a cell after cache invalidation or
    an execution failure, so move that attempt aside before creating the next
    one.  The stable cell paths continue to name the latest attempt while every
    earlier result, trajectory, artifact, and incremental checkpoint remains
    available under ``attempts/``.
    """

    if not run_dir.exists():
        return None
    owned_names = ("result.json", "traj.jsonl", "artifacts", "canonical")
    existing = tuple(run_dir / name for name in owned_names if (run_dir / name).exists())
    if not existing:
        return None

    attempts = run_dir / "attempts"
    attempts.mkdir(exist_ok=True)
    index = 0
    while (attempts / f"attempt-{index:03d}").exists():
        index += 1
    archived = attempts / f"attempt-{index:03d}"
    archived.mkdir()
    for source in existing:
        source.replace(archived / source.name)

    archived_result = archived / "result.json"
    archived_trajectory = archived / "traj.jsonl"
    if archived_result.is_file() and archived_trajectory.is_file():
        try:
            payload = json.loads(archived_result.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        else:
            if isinstance(payload, dict):
                payload["trajectory_location"] = str(archived_trajectory.resolve())
                temporary = archived / ".result.json.tmp"
                temporary.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(archived_result)
    return archived


@dataclass(frozen=True, slots=True)
class _Metrics:
    final_answer: str | None
    verdict: str
    # Distinct from ``verdict``: a verifier can return a passing judgment without
    # certifying it. Only certification claims the answer was proven, so a
    # certified-but-wrong row is a verifier false positive worth counting.
    #
    # No packaged configuration can set this today, and the terminal
    # certification step in the shipped presets does not change that.
    # ``VerificationResult`` only accepts ``certified`` alongside
    # ``trust_level >= 2``, zero false-positive risk, and a witness;
    # ``AgentVerifier`` sets none of the three, and every packaged verifier
    # resource selects ``AgentVerifier``. That invariant is what stops a chatty
    # "looks good to me" from being recorded as a checked result, so it is left
    # alone: this column reads ``false`` because nothing has proven anything,
    # which is the honest value. Binding the *verdict* to the scored text is
    # what the terminal step buys.
    certified: bool
    rounds: int
    # Verifier steps whose reply could not be read at all. ``verdict`` for such a
    # step is ``inconclusive`` because that is the only honest value the record
    # can hold, but "the verifier was unsure" and "we could not read the
    # verifier" are not the same claim, and only one of them is a defect in this
    # repository. Counted here so the report can keep them apart.
    unreadable_verifier_outputs: int
    # Step executions after the first for the same step. Exactly what its name
    # says and nothing more: one turn of a revision loop that re-runs both
    # ``revise`` and ``verify`` contributes 2. Its consumer asks only whether it
    # is non-zero, which is the question "did this run loop at all". Recorded
    # rather than inferred from ``rounds > 1``, because ``rounds`` now also
    # counts the terminal certification step that every run performs.
    repeated_steps: int
    termination_reason: str
    # Whether the reply that produced the scored answer was cut off at
    # ``max_tokens``. Beside ``termination_reason``, never folded into it: the
    # Environment's reason and the sampler's reason are different facts that can
    # both be true, and ``termination_reason``'s values already have consumers
    # (the report's ``max_turns`` and ``error`` counters) that a widened value
    # domain would silently retire.
    output_truncated: bool
    solver_tool_calls: int = 0
    verifier_tool_calls: int = 0
    solver_tool_failures: int = 0
    verifier_tool_failures: int = 0
    solver_python_tool_failures: int = 0
    verifier_python_tool_failures: int = 0
    solver_python_tool_repairs: int = 0
    verifier_python_tool_repairs: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    plan_mode: str | None = None
    n_branches: int | None = None
    vote: dict[str, Any] | None = None


def _metrics(
    outcome: WorkflowResult,
    config: RunConfig,
    *,
    grader_id: str | None = "exact_match",
) -> _Metrics:
    selected_branch = outcome.selected_branch_index
    output_text = _output_text(outcome.output)
    extract = get_grader(grader_id).extract if grader_id is not None else extract_declared_answer
    # No fallback to the raw output: stating no answer is not the same claim as
    # stating a wrong one. Handing the whole transcript to a grader made which
    # claim got recorded depend on the dataset's ``grader_id`` -- a truncated
    # chain of thought scored ``False`` under ``exact_match`` and
    # ``math_expression`` but ``None`` under ``integer_answer``. ``None`` reaches
    # every shipped grader's unscoreable branch instead. See the module docstring
    # for where the transcript that explains the absence is kept.
    final_answer = extract(output_text)
    selected_steps = [
        step
        for step in outcome.steps
        if selected_branch is None or step.branch_index == selected_branch
    ]
    final_verification = _final_output_verification(outcome)
    verifications = [
        step.output for step in selected_steps if isinstance(step.output, VerificationResult)
    ]
    # Counts every verifier step on the selected branch, which under the shipped
    # presets is the bounded ``verify`` loop *plus* the terminal ``certify``
    # step: a run that passed first try now reports 2, not 1. The report renders
    # this as "total verifier rounds", so that column's baseline moved with the
    # topology; ``repeated_steps`` below is what still isolates the loop.
    verifier_rounds = len(verifications)
    ensemble = config.workflow.ensemble
    selection = _workflow_selection(config)
    scored_producer = _scored_agent_result(outcome, selected_steps)
    return _Metrics(
        final_answer=final_answer,
        verdict=(
            str(final_verification.verdict) if final_verification is not None else "inconclusive"
        ),
        certified=final_verification is not None and bool(final_verification.certified),
        rounds=verifier_rounds,
        unreadable_verifier_outputs=sum(
            1 for verification in verifications if _verifier_output_unreadable(verification)
        ),
        repeated_steps=sum(1 for step in selected_steps if step.iteration > 1),
        termination_reason=_termination_reason(outcome, scored_producer),
        # Deliberately does not touch ``final_answer`` or ``correct`` above. A
        # model may state ``\boxed{79}`` and then ramble until it is cut off;
        # that answer is valid and is graded exactly as an untruncated one. This
        # records the fact so a reader can tell "the model stated no answer"
        # (capability limit) from "the model was cut off" (configuration bug),
        # which are the same unscoreable row without it.
        output_truncated=scored_producer is not None and scored_producer.output_truncated,
        plan_mode=None if selection == "single" else selection,
        n_branches=ensemble.replicas if ensemble is not None else None,
        vote=(
            _ensemble_vote(outcome, selection_key=ensemble.selection_key, extract=extract)
            if ensemble is not None
            else None
        ),
    )


def _verifier_output_unreadable(verification: VerificationResult) -> bool:
    """True when no judgment was read from this verifier step.

    ``AgentVerifier`` records ``inconclusive`` both when the model said it was
    unsure and when the model's reply could not be parsed or was never
    evaluated, because ``inconclusive`` is the only verdict the record may hold.
    The two are told apart by the audit keys the adapter writes beside the
    verdict, which is why they are read here rather than re-derived: a reader
    who cannot separate them cannot tell a cautious verifier from a broken one.
    """

    details = verification.details
    return "parse_error" in details or bool(details.get("parse_skipped"))


def _scored_agent_result(
    outcome: WorkflowResult,
    selected_steps: Sequence[StepResult],
) -> AgentResult | None:
    """The agent result whose text became the scored answer, if there is one.

    With a terminal verifier as the output step, the run's output is a
    ``VerificationResult`` whose ``candidate_ref`` names the agent result it
    judged. Every per-run fact about *how the answer was produced* -- how the
    episode ended, whether the reply was cut off -- has to be read from that one
    record, so it is resolved once here rather than once per fact. Two resolvers
    would eventually disagree about which step they were describing.

    ``None`` when the run has no agent-produced output at all (a failed or
    verifier-only outcome), which is not the same as an agent result that ended
    cleanly and must not be reported as one.
    """

    output = outcome.output
    if isinstance(output, AgentResult):
        return output
    if not isinstance(output, VerificationResult):
        return None
    return next(
        (
            step.output
            for step in selected_steps
            if isinstance(step.output, AgentResult) and step.output.task_id == output.candidate_ref
        ),
        None,
    )


def _termination_reason(outcome: WorkflowResult, producer: AgentResult | None) -> str:
    """How the run ended, from the perspective of the answer that was scored.

    With a terminal verifier as the output step the last thing that ran is
    always a clean verifier call, so reporting its status alone would retire
    the ``max_turns`` and ``error`` counters the report keeps: a finalizer that
    exhausted its turn budget and one that finished cleanly would both read as
    ``verified``. The producing agent result is named by ``candidate_ref``, so
    its own termination reason is recoverable and outranks ``verified``
    whenever it was not a clean finish.

    This value domain is closed on purpose. A reply cut off at ``max_tokens`` is
    reported by ``_Metrics.output_truncated`` instead of by a new reason here,
    because the Environment's verdict and the sampler's are different facts that
    can both be true, and the counters above match on exact values.
    """

    output = outcome.output
    if isinstance(output, AgentResult):
        return output.termination_reason
    if not isinstance(output, VerificationResult):
        return outcome.status
    if producer is not None and producer.termination_reason != "final":
        return producer.termination_reason
    return "verified"


def _ensemble_vote(
    outcome: WorkflowResult,
    *,
    selection_key: str,
    extract: Callable[[str | None], str | None],
) -> dict[str, Any]:
    """Audit the election that ``selection.select_result`` already decided.

    This must group branches exactly as :func:`selection._output_key` grouped
    them, or the reported vote describes a different election than the one that
    chose ``outcome.output``. That is why the ``exact_text`` branch below falls
    back to the stripped output text where :func:`_metrics` no longer does: under
    ``exact_text`` the raw text *is* the declared selection key, not a stand-in
    for a missing answer. Dropping the fallback here would move every branch of
    an ``exact_text`` ensemble into ``abstained``, which claims the branches
    declined to vote when they in fact voted and elected a winner -- a new
    dishonesty in place of the one :func:`_metrics` removes.
    """

    final_by_branch: dict[int, AgentResult | VerificationResult] = {}
    for step in outcome.steps:
        if step.step_id == outcome.selected_step_id:
            final_by_branch[step.branch_index] = step.output
    groups: list[dict[str, Any]] = []
    abstained: list[int] = []
    for branch, output in sorted(final_by_branch.items()):
        if isinstance(output, VerificationResult) and output.verdict != "pass":
            abstained.append(branch)
            continue
        text = _output_text(output)
        stripped = text.strip()
        if selection_key == "exact_text":
            # ``group_key`` is the selection key itself (see ``_output_key``);
            # ``answer`` is only its label in the report, so it prefers the
            # extracted answer and names the text when there is none.
            answer = extract(text) or (stripped or None)
            group_key = stripped or None
        elif selection_key == "declared_answer":
            answer = extract_declared_answer(text)
            group_key = answer
        else:  # pragma: no cover - protected by EnsembleConfig validation
            raise ValueError(f"unsupported ensemble selection key {selection_key!r}")
        if answer is None or group_key is None:
            abstained.append(branch)
            continue
        matching = next(
            (
                group
                for group in groups
                if (
                    group["key"] == group_key
                    if selection_key == "exact_text"
                    else declared_answers_equivalent(group["answer"], answer)
                )
            ),
            None,
        )
        if matching is None:
            groups.append({"key": group_key, "answer": answer, "branches": [branch]})
        else:
            matching["branches"].append(branch)
    selected_text = _output_text(outcome.output)
    selected = (
        extract_declared_answer(selected_text)
        if selection_key == "declared_answer"
        # Same reason as the group label above: the winner names the group that
        # won, and under ``exact_text`` that group is identified by its text. A
        # ``None`` winner beside non-empty groups would read in the report's
        # ``n_no_winner`` count as "every branch abstained", which is false.
        else extract(selected_text) or (selected_text.strip() or None)
    )
    return {
        "winner": selected,
        "groups": [
            {
                "answer": group["answer"],
                "count": len(group["branches"]),
                "branches": group["branches"],
            }
            for group in groups
        ],
        "abstained": abstained,
    }


def _output_text(output: object) -> str:
    if isinstance(output, AgentResult):
        return output.final_text
    if isinstance(output, VerificationResult):
        # The agent result, when present, is the verifier's judgment JSON and
        # trajectory.  The scoreable solution is the candidate to which that
        # judgment is cryptographically/identically bound.
        return output.candidate
    return ""


def _final_output_verification(outcome: WorkflowResult) -> VerificationResult | None:
    """Return only a verifier judgment bound to the selected final output.

    A terminal ``VerificationResult`` is already the selected, candidate-bound
    output.  An ``AgentResult`` can inherit a prior judgment only when it is the
    exact producing result named by ``candidate_ref`` and its text and digest
    still match.  In particular, a finalizer has a different task identity, so
    its output cannot inherit a verifier verdict even if it copies or rewrites
    an earlier answer.
    """

    output = outcome.output
    if output is None:
        return None
    selected_steps = [
        step
        for step in outcome.steps
        if (outcome.selected_branch_index is None)
        or step.branch_index == outcome.selected_branch_index
    ]
    if not any(
        step.step_id == outcome.selected_step_id and step.output is output
        for step in selected_steps
    ):
        return None
    if isinstance(output, VerificationResult):
        return output
    if not isinstance(output, AgentResult):  # pragma: no cover - WorkflowResult validates this
        return None

    candidate_sha256 = hashlib.sha256(output.final_text.encode("utf-8")).hexdigest()
    for step in reversed(selected_steps):
        verification = step.output
        if not isinstance(verification, VerificationResult):
            continue
        if (
            verification.candidate_ref == output.task_id
            and verification.candidate == output.final_text
            and verification.candidate_sha256 == candidate_sha256
        ):
            return verification
    return None
