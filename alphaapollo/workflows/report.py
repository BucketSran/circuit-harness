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

"""Aggregate repeated Workflow records and render JSON/Markdown reports.

Unscoreable samples remain visible and count conservatively in accuracy
denominators. Token totals retain evidence status whenever provider usage is
partial or unavailable.
"""

from __future__ import annotations

import math
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from alphaapollo.common.trajectory.metrics import (
    METRICS_EVIDENCE_COMPLETE,
    METRICS_EVIDENCE_PARTIAL,
    METRICS_EVIDENCE_UNAVAILABLE,
)
from alphaapollo.workflows.records import RunRecord

__all__ = [
    "build_report",
    "decoding_summary",
    "every_run_failed",
    "format_report_markdown",
    "git_sha",
    "mcnemar_exact_p",
    "model_label",
    "resource_summary",
]


def every_run_failed(report: Mapping[str, Any]) -> bool:
    """True when every run in this report ended in a typed error.

    A run that never produced an answer and a run that answered wrongly both
    land on ``pass_at_1: 0.000``, and the two mean opposite things: the first is
    usually a broken configuration -- a model name the server 404s, an
    unreachable base_url -- and the second is a result. The distinction is
    already in the records (``termination_reason == "error"``); this only reads
    it back, so callers can say which one happened instead of publishing a zero.

    Deliberately all-or-nothing. A run where some cells failed is an ordinary
    partial result that the ``typed-fail (err)`` column already reports.
    """

    conditions = report.get("conditions") or {}
    total = sum(int(summary.get("n_runs", 0)) for summary in conditions.values())
    errors = sum(int(summary.get("n_error_runs", 0)) for summary in conditions.values())
    return total > 0 and errors == total


def build_report(
    records: Sequence[RunRecord],
    *,
    model: str,
    config_digest: str,
    seed_base: int,
    k: int,
    code_sha: str | None = None,
    decoding: dict[str, Any] | None = None,
    trajectory_note: str | None = None,
) -> dict[str, Any]:
    conditions = sorted({r.condition for r in records})
    per_condition = {
        c: _condition_summary([r for r in records if r.condition == c]) for c in conditions
    }

    report: dict[str, Any] = {
        "provenance": {
            "model": model,
            "config_digest": config_digest,
            "seed_base": seed_base,
            "k": k,
            "code_sha": code_sha,
            "n_records": len(records),
            # #158 report requires model, prompt/config version, decoding settings,
            # seeds, code SHA, and trajectory locations. config_digest pins the exact
            # config (prompts + max_rounds + tool grants); decoding carries the
            # sampling knobs; trajectory locations are per-record (see trajectory_note).
            "decoding": decoding or {},
            "trajectory_note": trajectory_note
            or "each run record carries its own trajectory_location (traj.jsonl path)",
        },
        "conditions": per_condition,
    }

    # Plan modes (#160): a purely ADDITIVE block, present only when some run actually
    # executed under a plan mode. A vanilla report keeps exactly its previous keys.
    plan_mode = _plan_mode_summary(records)
    if plan_mode is not None:
        report["plan_mode"] = plan_mode

    if "python_code" in per_condition and "no_tool" in per_condition:
        report["paired"] = _paired(records)
    feedback_conditions = {
        "differentiated": "python_code",
        "generic": "python_code_generic_feedback",
        "missing": "python_code_missing_feedback",
    }
    if all(condition in per_condition for condition in feedback_conditions.values()):
        feedback_ablation = {
            "primary_metric": "repair_success_rate",
            "definition": "failed Python calls whose next Python attempt succeeds",
            # An arm with no failures has no repair rate: it reports null, never 0.0,
            # so it can never be read as "repaired nothing". The counts ship alongside
            # because a bare ratio hides its own sample size, and any significance test
            # over these arms needs the denominator anyway.
            "null_rate_means": "no Python failures in this arm, so there was nothing to repair",
        }
        feedback_ablation.update(
            {
                mode: per_condition[condition]["tool"]["repair_success_rate"]
                for mode, condition in feedback_conditions.items()
            }
        )
        feedback_ablation["counts"] = {
            mode: {
                "repair_opportunities": per_condition[condition]["tool"]["repair_opportunities"],
                "repair_successes": per_condition[condition]["tool"]["repair_successes"],
            }
            for mode, condition in feedback_conditions.items()
        }
        report["feedback_ablation"] = feedback_ablation
    return report


def _plan_mode_summary(records: Sequence[RunRecord]) -> dict[str, Any] | None:
    """Which plan mode(s) produced these runs, and the per-run consensus votes.

    Returns ``None`` when every record is vanilla, so the vanilla report shape is
    untouched. Vote rows are the audit trail the honesty rule needs for a consensus
    experiment: a reported answer must be traceable to the branch counts that chose
    it, including the case where every branch abstained and there is no winner.
    """

    tagged = [r for r in records if r.plan_mode]
    if not tagged:
        return None

    modes: dict[str, int] = defaultdict(int)
    for record in tagged:
        modes[record.plan_mode] += 1
    branch_counts = sorted({r.n_branches for r in tagged if r.n_branches is not None})

    summary: dict[str, Any] = {
        "modes": dict(sorted(modes.items())),
        "n_plan_mode_runs": len(tagged),
        "n_vanilla_runs": len(records) - len(tagged),
        "branches_per_run": branch_counts,
        "total_branches": sum(r.n_branches or 0 for r in tagged),
    }

    voted = [r for r in tagged if r.vote]
    if voted:
        rows = [_vote_row(r) for r in voted]
        rows.sort(key=lambda row: (row["condition"], row["problem_id"], row["seed"]))
        summary["consensus"] = {
            "n_votes": len(rows),
            # A vote with no winner means every branch abstained: reported, never
            # hidden, because it is an unscoreable run with a specific cause.
            "n_no_winner": sum(1 for row in rows if row["winner"] is None),
            "n_unanimous": sum(
                1 for row in rows if len(row["groups"]) == 1 and not row["abstained"]
            ),
            "n_split": sum(1 for row in rows if len(row["groups"]) > 1),
            "n_with_abstentions": sum(1 for row in rows if row["abstained"]),
            "runs": rows,
        }

    # A run whose verifier spent every retry and still FAILed produces an answer -- the
    # documented degrade -- but it is NOT a clean solve. Reported separately so a
    # solve-rate over static graphs can be read honestly: without this, four failed
    # attempts and a first-time pass are the same row.
    exhausted = [r for r in tagged if r.retries_exhausted]
    if exhausted:
        summary["retries_exhausted"] = {
            "n_runs": len(exhausted),
            "n_correct": sum(1 for r in exhausted if r.correct is True),
            "runs": sorted(
                (
                    {
                        "problem_id": r.problem_id,
                        "condition": r.condition,
                        "seed": r.seed,
                        "verifiers": list(r.retries_exhausted or ()),
                        "final_answer": r.final_answer,
                        "correct": r.correct,
                    }
                    for r in exhausted
                ),
                key=lambda row: (row["condition"], row["problem_id"], row["seed"]),
            ),
        }
    return summary


def _vote_row(record: RunRecord) -> dict[str, Any]:
    vote = record.vote or {}
    return {
        "problem_id": record.problem_id,
        "condition": record.condition,
        "seed": record.seed,
        "winner": vote.get("winner"),
        "groups": [
            {
                "answer": group.get("answer"),
                "count": group.get("count"),
                "branches": list(group.get("branches", [])),
            }
            for group in vote.get("groups", [])
        ],
        "abstained": list(vote.get("abstained", [])),
        "n_branches": record.n_branches,
        "correct": record.correct,
    }


def _condition_summary(records: Sequence[RunRecord]) -> dict[str, Any]:
    by_problem: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_problem[record.problem_id].append(record)

    n = len(records)
    n_correct = sum(1 for r in records if r.correct is True)
    n_unscoreable = sum(1 for r in records if r.correct is None)

    # Run-level typed-failure accounting (#158 Gate 2: "32 completed or explicitly
    # typed-failure runs" per pair). A typed failure is a run that ended in a typed
    # error rather than a scored final answer.
    n_error_runs = sum(1 for r in records if r.termination_reason == "error")
    n_max_turns_runs = sum(1 for r in records if r.termination_reason == "max_turns")
    # Counted apart from every column above, because it is the only one a reader
    # fixes by raising a number. A run whose reply was cut off at ``max_tokens``
    # is not a typed failure and not a turn-budget exhaustion -- those are read
    # from ``termination_reason``, which the Environment decides without ever
    # seeing the sampler's ``finish_reason``. It also does not imply an
    # unscoreable row: a truncated reply that still stated its answer scores
    # normally and is counted here as well, so this is "how many replies were
    # clipped", not "how many runs were spoiled".
    n_truncated_runs = sum(1 for r in records if r.output_truncated)
    # A revised run is one where the Workflow looped, i.e. executed some step
    # more than once. Read from the record rather than from ``rounds > 1``: the
    # shipped solver presets end on a terminal certification verifier, so every
    # run now reports at least two verifier rounds and that test would call
    # every run revised.
    n_revised_runs = sum(1 for r in records if r.repeated_steps > 0)

    # Verifier verdict distribution (#158: "Verifier and revision counts"). One
    # verifier verdict is produced per round; total_rounds counts verifier
    # invocations, which under the shipped solver presets is the bounded
    # revision loop plus one terminal certification step per run.
    #
    # ``INCONCLUSIVE`` means the verifier deliberated and declined to decide.
    # A reply that could not be read is recorded as ``inconclusive`` too, because
    # that is the only verdict the contract allows, so it is counted separately
    # here: one of the two is a model being careful and the other is this
    # repository failing to read its own verifier, and a reader who sees one
    # number cannot act on either.
    # Verifier certification accounting. ``certified`` is stronger than a passing
    # verdict: it claims the answer was proven. A certified row that scores wrong
    # is a verifier false positive, which is the failure this benchmark most needs
    # to surface -- a silent one would read as ordinary model error.
    n_certified = sum(1 for r in records if r.certified)
    n_certified_correct = sum(1 for r in records if r.certified and r.correct is True)
    n_false_positive_certifications = sum(
        1 for r in records if r.certified and r.correct is not True
    )

    verdict_counts: dict[str, int] = defaultdict(int)
    for r in records:
        verdict_counts[(r.verdict or "unknown").upper()] += 1
    total_rounds = sum(r.rounds for r in records)
    n_unreadable_verifier_outputs = sum(r.unreadable_verifier_outputs for r in records)
    n_runs_with_unreadable_verifier_output = sum(
        1 for r in records if r.unreadable_verifier_outputs
    )

    per_problem = {}
    n_pass_at_k = 0
    for pid, runs in sorted(by_problem.items()):
        correct = sum(1 for r in runs if r.correct is True)
        solved = correct > 0
        n_pass_at_k += 1 if solved else 0
        per_problem[pid] = {
            "n": len(runs),
            "n_correct": correct,
            "pass_at_1": correct / len(runs) if runs else 0.0,
            "solved_any": solved,
        }

    solver_calls = sum(r.solver_tool_calls for r in records)
    verifier_calls = sum(r.verifier_tool_calls for r in records)
    solver_fail = sum(r.solver_tool_failures for r in records)
    verifier_fail = sum(r.verifier_tool_failures for r in records)
    solver_python_fail = sum(r.solver_python_tool_failures for r in records)
    verifier_python_fail = sum(r.verifier_python_tool_failures for r in records)
    solver_repairs = sum(r.solver_python_tool_repairs for r in records)
    verifier_repairs = sum(r.verifier_python_tool_repairs for r in records)
    solver_runs_with_call = sum(1 for r in records if r.solver_tool_calls > 0)
    verifier_runs_with_call = sum(1 for r in records if r.verifier_tool_calls > 0)
    repair_opportunities = solver_python_fail + verifier_python_fail
    repair_successes = solver_repairs + verifier_repairs
    runs_with_failure = [
        r for r in records if (r.solver_python_tool_failures + r.verifier_python_tool_failures) > 0
    ]
    correct_after_failure = sum(1 for r in runs_with_failure if r.correct is True)
    evidence_counts = {
        status: sum(1 for record in records if record.metrics_evidence_status == status)
        for status in (
            METRICS_EVIDENCE_COMPLETE,
            METRICS_EVIDENCE_PARTIAL,
            METRICS_EVIDENCE_UNAVAILABLE,
        )
    }
    if evidence_counts[METRICS_EVIDENCE_COMPLETE] == n:
        aggregate_evidence_status = METRICS_EVIDENCE_COMPLETE
    elif evidence_counts[METRICS_EVIDENCE_UNAVAILABLE] == n:
        aggregate_evidence_status = METRICS_EVIDENCE_UNAVAILABLE
    else:
        aggregate_evidence_status = METRICS_EVIDENCE_PARTIAL

    return {
        "n_runs": n,
        # pass@1 is plain sample accuracy: correct over ALL runs (unscoreable runs
        # count against it, never dropped). Same denominator as per-problem pass@1.
        "pass_at_1": _rate(n_correct, n),
        "pass_at_k": n_pass_at_k / len(by_problem) if by_problem else 0.0,
        "n_problems": len(by_problem),
        "n_correct": n_correct,
        "n_unscoreable": n_unscoreable,
        "n_certified": n_certified,
        "n_certified_correct": n_certified_correct,
        "n_false_positive_certifications": n_false_positive_certifications,
        "n_error_runs": n_error_runs,
        "n_max_turns_runs": n_max_turns_runs,
        "n_truncated_runs": n_truncated_runs,
        "n_revised_runs": n_revised_runs,
        "verifier": {
            "verdict_counts": dict(verdict_counts),
            "total_verifier_rounds": total_rounds,
            "n_unreadable_verifier_outputs": n_unreadable_verifier_outputs,
            "n_runs_with_unreadable_verifier_output": n_runs_with_unreadable_verifier_output,
        },
        "tool": {
            "solver_calls": solver_calls,
            "verifier_calls": verifier_calls,
            "solver_failures": solver_fail,
            "verifier_failures": verifier_fail,
            "solver_python_failures": solver_python_fail,
            "verifier_python_failures": verifier_python_fail,
            "solver_success_rate": _rate(solver_calls - solver_fail, solver_calls),
            "verifier_success_rate": _rate(verifier_calls - verifier_fail, verifier_calls),
            # Python-call rates (#158): calls per run and the fraction of runs in
            # which the actor invoked Python at least once.
            "solver_calls_per_run": _rate(solver_calls, n),
            "verifier_calls_per_run": _rate(verifier_calls, n),
            "solver_runs_with_call_frac": _rate(solver_runs_with_call, n),
            "verifier_runs_with_call_frac": _rate(verifier_runs_with_call, n),
            "repair_opportunities": repair_opportunities,
            "repair_successes": repair_successes,
            # ``null`` when nothing ever failed: an arm with no repair opportunities has
            # no repair rate, and reporting 0.0 would score the best possible outcome
            # (Python never failed) exactly like the worst (nothing was ever repaired).
            "repair_success_rate": _optional_rate(repair_successes, repair_opportunities),
            "runs_with_failure": len(runs_with_failure),
            "correct_after_failure": correct_after_failure,
            # ``null`` for the same reason: with no run that hit a Python failure there
            # is no recovery rate to report, and 0.0 would read as "never recovered".
            "correct_after_failure_rate": _optional_rate(
                correct_after_failure, len(runs_with_failure)
            ),
        },
        "tokens": {
            "prompt": sum(r.prompt_tokens for r in records),
            "completion": sum(r.completion_tokens for r in records),
            # Existing integer totals remain additive and backward compatible.
            # They are authoritative only when every contributing ledger is
            # complete; otherwise they are explicitly labelled lower bounds.
            "evidence_status": aggregate_evidence_status,
            "evidence_counts": evidence_counts,
            "counts_are_lower_bounds": (aggregate_evidence_status != METRICS_EVIDENCE_COMPLETE),
        },
        "wall_seconds": sum(r.wall_seconds for r in records),
        "per_problem": per_problem,
    }


def _paired(records: Sequence[RunRecord]) -> dict[str, Any]:
    # Pair by (problem, seed): the same fixed seed schedule runs in both conditions.
    keyed: dict[tuple[str, int], dict[str, RunRecord]] = defaultdict(dict)
    for record in records:
        if record.condition in {"python_code", "no_tool"}:
            keyed[(record.problem_id, record.seed)][record.condition] = record

    both = 0
    b = 0  # python_code correct, no_tool wrong
    c = 0  # no_tool correct, python_code wrong
    concordant = 0
    dropped_unscoreable = 0  # complete pairs where a side was unscoreable
    for pair in keyed.values():
        if "python_code" not in pair or "no_tool" not in pair:
            continue
        py = pair["python_code"].correct
        no = pair["no_tool"].correct
        if py is None or no is None:
            # McNemar is defined over scoreable discordant pairs; report how many
            # complete pairs were dropped so the exclusion is never silent.
            dropped_unscoreable += 1
            continue
        both += 1
        if py and not no:
            b += 1
        elif no and not py:
            c += 1
        else:
            concordant += 1

    per_problem_delta = {}
    problems = sorted({pid for (pid, _seed) in keyed})
    for pid in problems:
        py_runs = [
            r for (p, _s), v in keyed.items() if p == pid for r in [v.get("python_code")] if r
        ]
        no_runs = [r for (p, _s), v in keyed.items() if p == pid for r in [v.get("no_tool")] if r]
        py_rate = _rate(sum(1 for r in py_runs if r.correct is True), len(py_runs))
        no_rate = _rate(sum(1 for r in no_runs if r.correct is True), len(no_runs))
        per_problem_delta[pid] = {
            "python_code_pass_at_1": py_rate,
            "no_tool_pass_at_1": no_rate,
            "delta": py_rate - no_rate,
        }

    agg_delta = _mean(d["delta"] for d in per_problem_delta.values())
    return {
        "n_paired_trials": both,
        "mcnemar": {
            "b_python_only_correct": b,
            "c_no_tool_only_correct": c,
            "concordant": concordant,
            "dropped_unscoreable_pairs": dropped_unscoreable,
            "p_value_two_sided": mcnemar_exact_p(b, c),
        },
        "aggregate_pass_at_1_delta": agg_delta,
        "per_problem_delta": per_problem_delta,
        "hypothesis": "python_code improves accuracy over no_tool",
    }


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant counts ``b`` and ``c``.

    Under H0 each discordant pair is a fair coin. p = 2 * P(X <= min(b, c)) with
    X ~ Binomial(n=b+c, 0.5), clamped to 1. Returns 1.0 when there are no
    discordant pairs (no evidence of a difference).
    """

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


# Votes are rendered per run; a full matrix has thousands, so the table is capped and
# points at report.json (which always carries every row) instead of being truncated
# silently.
_MAX_VOTE_ROWS = 20


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def _optional_rate(num: int, den: int) -> float | None:
    """A rate that refuses to invent a value for an empty denominator.

    ``_rate``'s 0.0 fallback is fine for a descriptive column, but not for a headline
    ratio: it makes "there was nothing to measure" indistinguishable from "measured,
    and the outcome was the worst possible". Only used where that confusion would
    change a conclusion; the markdown-rendered columns keep ``_rate`` because they are
    formatted with ``:.3f`` and have no null-safe rendering.
    """

    return num / den if den else None


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def resource_summary(config) -> tuple[dict[str, Any], dict[str, Any], str]:
    roles = {role.id: role for role in config.workflow.roles}
    agent_targets = [
        roles[step.role].target for step in config.workflow.steps if step.kind == "agent"
    ]
    # Non-empty by construction: ``WorkflowConfig`` refuses a topology with no
    # agent step, so the report never has to invent a solver for a run that had
    # none. The first agent step is the solver whose sampling settings the
    # decoding summary reports.
    solver_target = agent_targets[0]
    solver_options = dict(config.runtimes[solver_target].options)

    verifier_steps = [step for step in config.workflow.steps if step.kind == "verifier"]
    if not verifier_steps:
        return solver_options, {}, "none"
    verifier_target = roles[verifier_steps[0].role].target
    verifier_resource = config.verifiers[verifier_target]
    if verifier_resource.type == "lean4":
        image = dict(verifier_resource.options).get("image", "alphaapollo-lean:v4.24.0")
        return solver_options, {}, f"lean4:{image}"
    runtime_target = dict(verifier_resource.options)["runtime"]
    verifier_options = dict(config.runtimes[runtime_target].options)
    return solver_options, verifier_options, str(verifier_options.get("model", runtime_target))


def _verifier_kind(config) -> str:
    roles = {role.id: role for role in config.workflow.roles}
    for step in config.workflow.steps:
        if step.kind == "verifier":
            return config.verifiers[roles[step.role].target].type
    return "none"


def _max_rounds(config) -> int:
    transition_retries = [
        transition.max_iterations
        for transition in config.workflow.transitions
        if transition.condition == "not_passed" and transition.max_iterations is not None
    ]
    return 1 + max(transition_retries, default=0)


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def model_label(solver_options: dict[str, Any], verifier_label: str) -> str:
    """The one-line "what ran" string stored on the report."""

    return f"solver={solver_options.get('model', 'unknown')},verifier={verifier_label}"


def decoding_summary(
    config: Any,
    solver_options: dict[str, Any],
    verifier_options: dict[str, Any],
) -> dict[str, Any]:
    """Sampling settings that must match for two runs to be comparable."""

    solver_sampling = solver_options.get("sampling", {})
    verifier_sampling = verifier_options.get("sampling", {})
    return {
        "temperature": solver_sampling.get("temperature"),
        "solver_max_tokens": solver_sampling.get("max_tokens"),
        "verifier_max_tokens": verifier_sampling.get("max_tokens"),
        "verifier_kind": _verifier_kind(config),
        "max_rounds": _max_rounds(config),
        "solver_max_turns": solver_options.get("max_turns"),
        "verifier_max_turns": verifier_options.get("max_turns"),
    }


def format_report_markdown(report: dict[str, Any]) -> str:
    prov = report["provenance"]
    lines = [
        "# AlphaApollo Workflow run report",
        "",
        f"- model: `{prov['model']}`  | config: `{prov['config_digest']}`  "
        f"| code: `{prov.get('code_sha') or 'n/a'}`",
        f"- seeds: {prov['seed_base']}..{prov['seed_base'] + prov['k'] - 1} (k={prov['k']})  "
        f"| runs: {prov['n_records']}",
        f"- decoding: `{prov.get('decoding') or 'n/a'}`",
        f"- trajectories: {prov.get('trajectory_note', 'per-record traj.jsonl')}",
        "",
    ]
    if every_run_failed(report):
        # First thing the reader sees, because the number they came for is the
        # one they must not trust.
        lines += [
            f"> **NO RESULT: all {prov['n_records']} run(s) ended in a typed error.**",
            "> Nothing was answered, so every rate below is 0.000 by absence, not by measurement.",
            "> Check the per-cell `result.json` `error` field first -- an unreachable server or a",
            "> model name the server does not serve fails this way.",
            "",
        ]
    lines += [
        "## Per condition",
        "",
        # The pass@k column is "solved by at least one of this problem's runs", so
        # its k is the run's own sample count. It was a hardcoded 32 and read as a
        # claim about how many samples were drawn whatever k actually was. The
        # parameter is spelled out rather than substituted into the name, so a k=1
        # report does not print two columns both headed `pass@1`.
        # ``cut-off`` sits beside ``typed-fail`` and ``max-turns`` and is none of
        # them: it is the count of runs whose scored reply hit the sampling
        # ``max_tokens`` ceiling, which is the one column here that a reader
        # answers by editing a number rather than by changing model or prompt.
        f"| condition | pass@1 | pass@k (k={prov['k']}) | problems | correct | unscoreable |"
        " typed-fail (err) | max-turns | cut-off | revised | tokens (c) | token evidence"
        " | wall (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, summary in report["conditions"].items():
        token_evidence = summary["tokens"]
        evidence_counts = token_evidence.get(
            "evidence_counts",
            {
                METRICS_EVIDENCE_COMPLETE: summary.get("n_runs", 0),
                METRICS_EVIDENCE_PARTIAL: 0,
                METRICS_EVIDENCE_UNAVAILABLE: 0,
            },
        )
        evidence_label = token_evidence.get("evidence_status", METRICS_EVIDENCE_COMPLETE)
        if token_evidence.get("counts_are_lower_bounds", False):
            evidence_label += (
                " (lower bound; "
                f"complete={evidence_counts[METRICS_EVIDENCE_COMPLETE]}, "
                f"partial={evidence_counts[METRICS_EVIDENCE_PARTIAL]}, "
                f"unavailable={evidence_counts[METRICS_EVIDENCE_UNAVAILABLE]})"
            )
        lines.append(
            f"| {name} | {summary['pass_at_1']:.3f} | {summary['pass_at_k']:.3f} | "
            f"{summary['n_problems']} | {summary['n_correct']} | {summary['n_unscoreable']} | "
            f"{summary['n_error_runs']} | {summary['n_max_turns_runs']} | "
            f"{summary['n_truncated_runs']} | {summary['n_revised_runs']} | "
            f"{summary['tokens']['completion']} | {evidence_label} | "
            f"{summary['wall_seconds']:.1f} |"
        )

    lines += [
        "",
        "## Verifier verdicts and revisions (per condition)",
        "",
        "| condition | verdict counts | unreadable verifier outputs "
        "| total verifier rounds | revised runs |",
        "|---|---|---|---|---|",
    ]
    for name, summary in report["conditions"].items():
        v = summary["verifier"]
        vc = ", ".join(f"{k}={n}" for k, n in sorted(v["verdict_counts"].items()))
        # Beside the verdicts, never folded into them: an INCONCLUSIVE the model
        # chose and one this repository could not read look identical in the
        # verdict column, and only the second is ours to fix.
        unreadable = (
            f"{v['n_unreadable_verifier_outputs']} "
            f"(in {v['n_runs_with_unreadable_verifier_output']} runs)"
        )
        lines.append(
            f"| {name} | {vc or 'n/a'} | {unreadable} | {v['total_verifier_rounds']} | "
            f"{summary['n_revised_runs']} |"
        )

    lines += [
        "",
        "## Tool use (per role, per condition)",
        "",
        "| condition | solver calls/run | solver runs-with-call | solver success |"
        " verifier calls/run | verifier runs-with-call | verifier success |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, summary in report["conditions"].items():
        t = summary["tool"]
        lines.append(
            f"| {name} | {t['solver_calls_per_run']:.3f} | {t['solver_runs_with_call_frac']:.3f} |"
            f" {t['solver_success_rate']:.3f} | {t['verifier_calls_per_run']:.3f} |"
            f" {t['verifier_runs_with_call_frac']:.3f} | {t['verifier_success_rate']:.3f} |"
        )

    lines += _plan_mode_markdown(report.get("plan_mode"))

    if "paired" in report:
        paired = report["paired"]
        mc = paired["mcnemar"]
        lines += [
            "",
            "## Paired: python_code vs no_tool",
            "",
            f"- aggregate pass@1 delta: **{paired['aggregate_pass_at_1_delta']:+.3f}**",
            f"- paired trials: {paired['n_paired_trials']}  |"
            f" discordant b(py only)={mc['b_python_only_correct']},"
            f" c(no_tool only)={mc['c_no_tool_only_correct']}"
            f"  | dropped unscoreable pairs: {mc['dropped_unscoreable_pairs']}",
            f"- McNemar exact two-sided p = {mc['p_value_two_sided']:.4g}",
            f"- hypothesis: {paired['hypothesis']}",
            "",
            "### Per-problem correct counts and delta",
            "",
            "| problem | no_tool correct | python_code correct | pass@1 delta |",
            "|---|---|---|---|",
        ]
        no_pp = report["conditions"]["no_tool"]["per_problem"]
        py_pp = report["conditions"]["python_code"]["per_problem"]
        deltas = paired["per_problem_delta"]
        for pid in sorted(deltas, key=lambda p: no_pp.get(p, {}).get("n_correct", 0)):
            no_c = no_pp.get(pid, {}).get("n_correct", 0)
            py_c = py_pp.get(pid, {}).get("n_correct", 0)
            lines.append(f"| {pid} | {no_c} | {py_c} | {deltas[pid]['delta']:+.3f} |")
    return "\n".join(lines) + "\n"


def _plan_mode_markdown(plan_mode: dict[str, Any] | None) -> list[str]:
    """Render the plan-mode section; empty for a vanilla report (key absent)."""

    if not plan_mode:
        return []
    modes = ", ".join(f"{name}={n}" for name, n in plan_mode["modes"].items())
    branches = ", ".join(str(b) for b in plan_mode["branches_per_run"]) or "n/a"
    lines = [
        "",
        "## Plan mode",
        "",
        f"- runs by mode: {modes}  | vanilla runs: {plan_mode['n_vanilla_runs']}",
        f"- branches per run: {branches}  | total branch runs: {plan_mode['total_branches']}",
    ]
    exhausted = plan_mode.get("retries_exhausted")
    if exhausted:
        # Stated in the summary, not buried in the JSON: a reader scanning the report
        # has to see that some answers came out of a verifier that never passed.
        lines.append(
            f"- **retries exhausted: {exhausted['n_runs']} run(s)** answered without ever "
            f"passing their verifier ({exhausted['n_correct']} still correct) — these are "
            "not clean solves"
        )
    consensus = plan_mode.get("consensus")
    if not consensus:
        return lines
    lines += [
        f"- votes: {consensus['n_votes']}  | unanimous: {consensus['n_unanimous']}"
        f"  | split: {consensus['n_split']}  | no winner: {consensus['n_no_winner']}"
        f"  | with abstentions: {consensus['n_with_abstentions']}",
        "",
        "### Consensus votes (per run)",
        "",
        "| condition | problem | seed | winner | groups (answer x votes) | abstained |",
        "|---|---|---|---|---|---|",
    ]
    for row in consensus["runs"][:_MAX_VOTE_ROWS]:
        groups = (
            ", ".join(f"{group['answer']} x{group['count']}" for group in row["groups"]) or "none"
        )
        abstained = ",".join(str(i) for i in row["abstained"]) or "-"
        lines.append(
            f"| {row['condition']} | {row['problem_id']} | {row['seed']} | "
            f"{row['winner'] if row['winner'] is not None else 'none'} | {groups} | {abstained} |"
        )
    if len(consensus["runs"]) > _MAX_VOTE_ROWS:
        lines.append(f"\n_{len(consensus['runs']) - _MAX_VOTE_ROWS} further votes in report.json_")
    return lines
