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

"""Immutable Workflow plans plus execution, scoring, and resume records."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from alphaapollo.common.trajectory.metrics import (
    METRICS_EVIDENCE_COMPLETE,
    METRICS_EVIDENCE_PARTIAL,
    METRICS_EVIDENCE_UNAVAILABLE,
    TrajectoryEvidenceError,
    materialize_trajectory_metrics,
    reduce_trajectory_evidence,
    refresh_trajectory_metrics,
)
from alphaapollo.reasoning.runtime import AgentResult
from alphaapollo.reasoning.verification import VerificationResult
from alphaapollo.workflows.config import (
    RoleConfig,
    StepConfig,
    TransitionConfig,
    WorkflowConfig,
    _freeze_json_mapping,
)

if TYPE_CHECKING:
    from alphaapollo.workflows.data import Cell

__all__ = [
    "StepResult",
    "WorkflowResumeStep",
    "RESULT_SCHEMA_VERSION",
    "RunRecord",
    "ScoredTask",
    "Workflow",
    "WorkflowInput",
    "WorkflowResult",
]

WorkflowStepOutput = AgentResult | VerificationResult


@dataclass(frozen=True, slots=True)
class Workflow:
    """A validated topology with immutable, indexed lookup tables."""

    config: WorkflowConfig
    _steps: Mapping[str, StepConfig] = field(init=False, repr=False, compare=False)
    _roles: Mapping[str, RoleConfig] = field(init=False, repr=False, compare=False)
    _transitions: Mapping[str, tuple[tuple[int, TransitionConfig], ...]] = field(
        init=False, repr=False, compare=False
    )
    _output_step_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, WorkflowConfig):
            raise TypeError("config must be a WorkflowConfig")
        steps = MappingProxyType({step.id: step for step in self.config.steps})
        roles = MappingProxyType({role.id: role for role in self.config.roles})
        outgoing: dict[str, list[tuple[int, TransitionConfig]]] = {
            step.id: [] for step in self.config.steps
        }
        for index, transition in enumerate(self.config.transitions):
            outgoing[transition.source].append((index, transition))
        transitions = MappingProxyType(
            {step_id: tuple(values) for step_id, values in outgoing.items()}
        )
        output_step_id = next(step.id for step in self.config.steps if step.output)
        object.__setattr__(self, "_steps", steps)
        object.__setattr__(self, "_roles", roles)
        object.__setattr__(self, "_transitions", transitions)
        object.__setattr__(self, "_output_step_id", output_step_id)

    @classmethod
    def from_config(cls, config: WorkflowConfig) -> Workflow:
        """Create an immutable execution plan from an already validated config."""

        return cls(config=config)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def entry_step(self) -> str:
        return self.config.entry_step

    @property
    def output_step_id(self) -> str:
        return self._output_step_id

    @property
    def ensemble_replicas(self) -> int:
        return self.config.ensemble.replicas if self.config.ensemble is not None else 1

    @property
    def execution_limit(self) -> int:
        """A defensive cap implied solely by the finite declarative graph."""

        loop_budget = sum(transition.max_iterations or 0 for transition in self.config.transitions)
        return len(self.config.steps) * (loop_budget + 1)

    def get_step(self, step_id: str) -> StepConfig:
        """Return a step or raise a useful error for an unknown id."""

        try:
            return self._steps[step_id]
        except KeyError as exc:
            raise KeyError(f"unknown workflow step {step_id!r}") from exc

    def get_role(self, role_id: str) -> RoleConfig:
        """Return the immutable role record referenced by a step."""

        try:
            return self._roles[role_id]
        except KeyError as exc:
            raise KeyError(f"unknown workflow role {role_id!r}") from exc

    def transitions_from(self, step_id: str) -> tuple[tuple[int, TransitionConfig], ...]:
        """Return ordered ``(config index, transition)`` pairs."""

        try:
            return self._transitions[step_id]
        except KeyError as exc:
            raise KeyError(f"unknown workflow step {step_id!r}") from exc


@dataclass(frozen=True, slots=True)
class WorkflowInput:
    """Public workflow input plus an Environment-only task payload.

    ``metadata`` may be rendered into prompts. ``task_payload`` is carried to
    the Runtime-owned Environment and is deliberately excluded from template
    values and invocation attribution.
    """

    input_id: str
    problem: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_payload: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.input_id, str) or not self.input_id.strip():
            raise ValueError("input_id must be a non-empty string")
        if not isinstance(self.problem, str) or not self.problem.strip():
            raise ValueError("problem must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if not isinstance(self.task_payload, Mapping):
            raise TypeError("task_payload must be a mapping")
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_mapping(self.metadata, "workflow input.metadata"),
        )
        object.__setattr__(
            self,
            "task_payload",
            _freeze_json_mapping(self.task_payload, "workflow input.task_payload"),
        )


@dataclass(frozen=True, slots=True)
class StepResult:
    """One Workflow node outcome without a duplicate output wrapper."""

    input_id: str
    step_id: str
    role: str
    iteration: int
    branch_index: int
    output: WorkflowStepOutput

    def __post_init__(self) -> None:
        for name in ("input_id", "step_id", "role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int):
            raise TypeError("iteration must be an int")
        if self.iteration < 1:
            raise ValueError("iteration must be at least one")
        if isinstance(self.branch_index, bool) or not isinstance(self.branch_index, int):
            raise TypeError("branch_index must be an int")
        if self.branch_index < 0:
            raise ValueError("branch_index must be non-negative")
        if not isinstance(self.output, (AgentResult, VerificationResult)):
            raise TypeError("output must be an AgentResult or VerificationResult")


@dataclass(frozen=True, slots=True)
class WorkflowResumeStep:
    """One previously completed agent step offered for strict replay.

    The executor, rather than memory, supplies input, branch, and role authority
    after proving that these coordinates follow the configured Workflow graph.
    """

    step_id: str
    iteration: int
    output: AgentResult

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("step_id must be a non-empty string")
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int):
            raise TypeError("iteration must be an int")
        if self.iteration < 1:
            raise ValueError("iteration must be at least one")
        if not isinstance(self.output, AgentResult):
            raise TypeError("output must be an AgentResult")


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Stable orchestration output plus all child trajectory-bearing results.

    ``steps`` directly retains every :class:`AgentResult`; each AgentResult owns
    its complete trajectory.  ``trajectory_refs`` contains the corresponding
    stable task ids until Common defines a persistent canonical reference type.
    """

    input_id: str
    workflow_name: str
    steps: tuple[StepResult, ...]
    output: WorkflowStepOutput | None
    selected_step_id: str | None
    selected_branch_index: int | None
    status: str
    trajectory_refs: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.input_id, str) or not self.input_id.strip():
            raise ValueError("input_id must be a non-empty string")
        if not isinstance(self.workflow_name, str) or not self.workflow_name.strip():
            raise ValueError("workflow_name must be a non-empty string")
        object.__setattr__(self, "steps", tuple(self.steps))
        if not all(isinstance(step, StepResult) for step in self.steps):
            raise TypeError("steps must contain only StepResult records")
        if self.output is not None and not isinstance(
            self.output, (AgentResult, VerificationResult)
        ):
            raise TypeError("output must be an AgentResult, VerificationResult, or None")
        if self.status not in {"completed", "incomplete"}:
            raise ValueError("status must be 'completed' or 'incomplete'")
        if self.status == "completed" and self.output is None:
            raise ValueError("a completed WorkflowResult requires output")
        if self.output is None and (
            self.selected_step_id is not None or self.selected_branch_index is not None
        ):
            raise ValueError("an output-less result cannot declare a selected step or branch")
        object.__setattr__(self, "trajectory_refs", _child_trajectory_refs(self.steps))


def _child_trajectory_refs(steps: tuple[StepResult, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()
    for step in steps:
        result = step.output if isinstance(step.output, AgentResult) else step.output.agent_result
        if result is not None and result.task_id not in seen:
            refs.append(result.task_id)
            seen.add(result.task_id)
    return tuple(refs)


@dataclass(frozen=True, slots=True)
class ScoredTask:
    """Private gold and grader metadata bound after model execution is planned.

    ``metadata`` is the dataset-shaped descriptive tail (``year``, and
    whatever an equivalent key is for another benchmark). It is deliberately a
    mapping rather than named fields so that adding a benchmark does not add a
    column here that every other benchmark leaves empty. Whoever populates it
    owns the boundary decision, because :meth:`public_view` hands it to the model.
    """

    id: str
    problem: str
    gold_answer: str  # gold; kept out of the loop, used only for scoring
    metadata: Mapping[str, str] = field(default_factory=dict)
    grader_id: str = "exact_match"

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.problem.strip() or not self.grader_id.strip():
            raise ValueError("scored task id, problem, and grader_id must be non-empty")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("scored task metadata must be a mapping")
        if any(not isinstance(value, str) for value in self.metadata.values()):
            raise TypeError("scored task metadata values must be strings")
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_mapping(self.metadata, "scored task.metadata"),
        )

    def public_view(self) -> dict[str, str]:
        """Return exactly the fields allowed to cross into a ``WorkflowInput``.

        This is the public projection of a record that also holds the gold
        answer. Everything returned here reaches the process that runs the
        model, so adding a key is adding it to the prompt side of the
        public/private boundary -- and reading ``gold_answer`` from here would
        put the answer in the prompt. Widen it only with that in mind, and only
        for a value whose producer has already restricted it to public content.
        """

        return {"problem_id": self.id, **self.metadata}


# v2: ``final_answer`` is the grader-extracted answer only. A v1 ledger could
# hold the entire model output there whenever extraction found nothing, and its
# ``correct`` was graded from that text -- ``False`` under ``exact_match`` and
# ``math_expression``, ``None`` under ``integer_answer``. Resuming such a row
# would keep publishing "answered wrongly" for a run that answered nothing, so
# the bump forces those cells to re-run rather than be read under the new rule.
#
# v3: the shipped solver presets end on a terminal certification verifier, so
# ``rounds`` counts one more step than it did and ``verdict`` now describes the
# answer the run reports rather than an intermediate proposal nothing scored.
# ``unreadable_verifier_outputs`` and ``revisions`` are new and a v2 cell cannot
# supply them -- defaulting them to 0 would publish "nothing was unreadable"
# and "nothing was revised" for a run that measured neither. The bump makes
# those cells re-run instead.
#
# v4: ``output_truncated`` is new and a v3 cell cannot supply it. Defaulting it
# to ``false`` would publish "this reply was not cut off" for a run that never
# looked, and the whole point of the field is to separate a model that stated no
# answer from one the sampler cut off at ``max_tokens`` -- opposite diagnoses
# that were previously the same ``correct: null`` row. The evidence is in the
# cell's ``canonical/trajectories.jsonl``, but scoring reads the ledger, so the
# bump makes those cells re-run rather than be read under the new rule.
RESULT_SCHEMA_VERSION = 4
_PLAN_MODE_FIELDS = ("plan_mode", "n_branches", "vote", "retries_exhausted")


def _problem_fingerprint(
    problem: ScoredTask,
    public_input: WorkflowInput | None = None,
) -> str:
    identity = {
        "statement": problem.problem,
        "gold": problem.gold_answer,
        "grader_id": problem.grader_id,
        "public_input_id": None if public_input is None else public_input.input_id,
        "public_metadata": None if public_input is None else dict(public_input.metadata),
    }
    if public_input is not None and public_input.task_payload:
        # The private payload can change the Environment trajectory and reward
        # while every public field stays identical.  Hash it into the cell
        # identity without persisting the payload itself.  Omitting an empty
        # payload preserves the fingerprints of existing public-only runs.
        identity["task_payload"] = dict(public_input.task_payload)
    payload = json.dumps(
        identity,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One completed (or resumed) run's stable scoring/report row."""

    problem_id: str
    problem_index: int
    condition: str
    seed: int
    # Only what the dataset's grader could extract; ``None`` means the run stated
    # no answer, never "the answer was empty prose". ``correct`` is then ``None``
    # (unscoreable) under every shipped grader, and the transcript that explains
    # the absence stays in this cell's sibling ``canonical/`` files.
    final_answer: str | None
    correct: bool | None
    verdict: str
    certified: bool = field(default=False, kw_only=True)
    # Verifier steps on the selected branch. The shipped solver presets end on a
    # terminal certification step, so a run that passed first try records 2.
    rounds: int
    # Verifier steps whose reply could not be read. Their ``verdict`` is
    # ``inconclusive`` -- the only value the contract allows -- so without this
    # count a verifier that was never read is indistinguishable from one that
    # deliberated and declined to decide.
    unreadable_verifier_outputs: int = field(default=0, kw_only=True)
    # Step executions after the first for the same step. Non-zero exactly when
    # the Workflow looped, which is what the report's "revised runs" column
    # asks. Recorded rather than inferred from ``rounds > 1``, because
    # ``rounds`` now includes a certification step every run performs.
    repeated_steps: int = field(default=0, kw_only=True)
    termination_reason: str
    # Whether the reply that produced the scored answer was cut off at the
    # sampler's ``max_tokens``. Read beside ``correct``, never instead of it:
    # ``correct: None`` says the run is unscoreable and this says why it might
    # be. "The model stated no answer" is a capability limit answered by a
    # different model or prompt; "the model was cut off" is a configuration bug
    # answered by raising a number, and the two used to be the same row.
    #
    # Scoring is deliberately unaffected. A truncated reply that still carries an
    # extractable answer is graded exactly like an untruncated one -- the report
    # counts it here instead of failing the cell.
    output_truncated: bool = field(default=False, kw_only=True)
    solver_tool_calls: int
    verifier_tool_calls: int
    solver_tool_failures: int
    verifier_tool_failures: int
    solver_python_tool_failures: int = field(default=0, kw_only=True)
    verifier_python_tool_failures: int = field(default=0, kw_only=True)
    solver_python_tool_repairs: int = field(default=0, kw_only=True)
    verifier_python_tool_repairs: int = field(default=0, kw_only=True)
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float
    trajectory_location: str
    resumed: bool
    plan_mode: str | None = None
    n_branches: int | None = None
    vote: dict[str, Any] | None = None
    # RESERVED, currently never written: the verifiers whose bounded retry edge
    # was spent without ever passing. #160's DAG engine filled it from its
    # ``plan_audit``; that engine was retired and the canonical WorkflowExecutor
    # has not been given a producer, so ``report._plan_mode_summary``'s
    # "retries exhausted" section cannot fire today and no result.json carries
    # the key. The signal is still available -- ``WorkflowExecutor``
    # ``_select_transition`` is exactly where a ``not_passed`` budget runs out
    # and the terminal fallback is taken -- and is worth restoring, because
    # without it an answer produced after four failed verifications and a
    # first-time clean pass are the same row. Kept rather than deleted so the
    # reader in ``report`` keeps its contract; see ``tests/workflows/
    # test_report.py::test_the_retries_exhausted_section_has_a_reader_but_no_producer``.
    retries_exhausted: tuple[str, ...] | None = None
    metrics_evidence_status: str = field(default=METRICS_EVIDENCE_COMPLETE, kw_only=True)
    metrics_evidence_gaps: tuple[str, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        if self.metrics_evidence_status not in {
            METRICS_EVIDENCE_COMPLETE,
            METRICS_EVIDENCE_PARTIAL,
            METRICS_EVIDENCE_UNAVAILABLE,
        }:
            raise ValueError(f"invalid metrics evidence status {self.metrics_evidence_status!r}")
        if isinstance(self.metrics_evidence_gaps, (str, bytes)):
            raise TypeError("metrics_evidence_gaps must be a sequence of strings")
        gaps = tuple(self.metrics_evidence_gaps)
        if any(not isinstance(gap, str) or not gap for gap in gaps):
            raise ValueError("metrics_evidence_gaps must contain non-empty strings")
        if self.metrics_evidence_status == METRICS_EVIDENCE_COMPLETE and gaps:
            raise ValueError("complete metrics evidence cannot declare gaps")
        object.__setattr__(self, "metrics_evidence_gaps", gaps)

    def to_json(self) -> dict[str, Any]:
        data = {key: getattr(self, key) for key in self.__slots__ if key != "resumed"}
        for key in _PLAN_MODE_FIELDS:
            if data.get(key) is None:
                data.pop(key, None)
        if not self.metrics_evidence_gaps:
            data.pop("metrics_evidence_gaps", None)
        return data


def _failure_record(
    cell: Cell,
    wall: float,
    traj_path: Path,
    plan_mode: str | None = None,
    *,
    metrics_evidence_status: str = METRICS_EVIDENCE_UNAVAILABLE,
) -> RunRecord:
    record = RunRecord(
        problem_id=cell.problem.id,
        problem_index=cell.problem_index,
        condition=cell.condition,
        seed=cell.seed,
        final_answer=None,
        correct=None,
        verdict="error",
        rounds=0,
        termination_reason="error",
        solver_tool_calls=0,
        verifier_tool_calls=0,
        solver_tool_failures=0,
        verifier_tool_failures=0,
        solver_python_tool_failures=0,
        verifier_python_tool_failures=0,
        solver_python_tool_repairs=0,
        verifier_python_tool_repairs=0,
        prompt_tokens=0,
        completion_tokens=0,
        wall_seconds=wall,
        trajectory_location=str(traj_path.resolve()),
        resumed=False,
        plan_mode=plan_mode,
        metrics_evidence_status=metrics_evidence_status,
    )
    if metrics_evidence_status == METRICS_EVIDENCE_UNAVAILABLE:
        return record
    try:
        metrics, gaps = reduce_trajectory_evidence(traj_path)
    except TrajectoryEvidenceError:
        return replace(record, metrics_evidence_status=METRICS_EVIDENCE_UNAVAILABLE)
    resolved_status = (
        METRICS_EVIDENCE_PARTIAL
        if metrics_evidence_status == METRICS_EVIDENCE_PARTIAL or gaps
        else METRICS_EVIDENCE_COMPLETE
    )
    return replace(
        record,
        **asdict(metrics),
        metrics_evidence_status=resolved_status,
        metrics_evidence_gaps=gaps,
    )


def _write_result(
    path: Path,
    record: RunRecord,
    cell: Cell,
    expected_digest: str,
    fingerprint: str,
    *,
    error: str | None,
) -> None:
    payload = {
        **record.to_json(),
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "gold_answer": cell.problem.gold_answer,
        "config_digest": expected_digest,
        "problem_fingerprint": fingerprint,
    }
    if error is not None:
        payload["error"] = error
    trajectory = Path(record.trajectory_location)
    if record.metrics_evidence_status == METRICS_EVIDENCE_UNAVAILABLE:
        payload["metrics_evidence_status"] = METRICS_EVIDENCE_UNAVAILABLE
    else:
        try:
            payload = materialize_trajectory_metrics(
                payload,
                trajectory,
                status=record.metrics_evidence_status,
            )
        except TrajectoryEvidenceError as exc:
            if error is None:
                raise
            payload["metrics_evidence_status"] = METRICS_EVIDENCE_UNAVAILABLE
            payload["metrics_evidence_error"] = str(exc)
    _write_json(path, payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_record(
    path: Path,
    cell: Cell,
    expected_digest: str,
    fingerprint: str,
    traj_path: Path,
    expected_plan_mode: str | None = None,
) -> RunRecord | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("problem_id") != cell.problem.id:
        return None
    if data.get("problem_index") != cell.problem_index:
        return None
    if data.get("config_digest") != expected_digest:
        return None
    if data.get("problem_fingerprint") != fingerprint:
        return None
    if data.get("plan_mode") != expected_plan_mode:
        return None
    if data.get("termination_reason") == "error":
        return None
    if data.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        return None
    evidence_status = data.get("metrics_evidence_status")
    if evidence_status not in {None, METRICS_EVIDENCE_COMPLETE}:
        return None
    try:
        data, refreshed = refresh_trajectory_metrics(data, traj_path)
    except TrajectoryEvidenceError:
        return None
    if evidence_status is None:
        data["metrics_evidence_status"] = (
            METRICS_EVIDENCE_PARTIAL
            if data.get("metrics_evidence_gaps")
            else METRICS_EVIDENCE_COMPLETE
        )
        refreshed = True
    if data.get("metrics_evidence_status") != METRICS_EVIDENCE_COMPLETE:
        return None
    if refreshed:
        _write_json(path, data)
    try:
        return RunRecord(
            problem_id=data["problem_id"],
            problem_index=data["problem_index"],
            condition=data["condition"],
            seed=data["seed"],
            final_answer=data["final_answer"],
            correct=data["correct"],
            verdict=data["verdict"],
            certified=data.get("certified", False),
            rounds=data["rounds"],
            unreadable_verifier_outputs=data["unreadable_verifier_outputs"],
            repeated_steps=data["repeated_steps"],
            termination_reason=data["termination_reason"],
            output_truncated=data["output_truncated"],
            solver_tool_calls=data["solver_tool_calls"],
            verifier_tool_calls=data["verifier_tool_calls"],
            solver_tool_failures=data["solver_tool_failures"],
            verifier_tool_failures=data["verifier_tool_failures"],
            solver_python_tool_failures=data["solver_python_tool_failures"],
            verifier_python_tool_failures=data["verifier_python_tool_failures"],
            solver_python_tool_repairs=data["solver_python_tool_repairs"],
            verifier_python_tool_repairs=data["verifier_python_tool_repairs"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            wall_seconds=data["wall_seconds"],
            trajectory_location=data["trajectory_location"],
            resumed=True,
            plan_mode=data.get("plan_mode"),
            n_branches=data.get("n_branches"),
            vote=data.get("vote"),
            retries_exhausted=(
                tuple(data["retries_exhausted"]) if data.get("retries_exhausted") else None
            ),
            metrics_evidence_status=data.get("metrics_evidence_status", METRICS_EVIDENCE_COMPLETE),
            metrics_evidence_gaps=tuple(data.get("metrics_evidence_gaps", ())),
        )
    except (KeyError, TypeError, ValueError):
        return None
