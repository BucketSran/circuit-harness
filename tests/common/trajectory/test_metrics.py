"""Trajectory metric reduction and evidence-materialization contracts."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alphaapollo.common.environment import DefaultEnvironment, TrajectoryEnvironmentEventSink
from alphaapollo.common.execution import (
    ArtifactStore,
    ExecutionContext,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.execution.tools.base import INTERNAL_PYTHON_TOOL_ID
from alphaapollo.common.trajectory import TrajectoryStore
from alphaapollo.common.trajectory import metrics as metrics_module
from alphaapollo.common.trajectory.metrics import (
    METRICS_EVIDENCE_COMPLETE,
    METRICS_EVIDENCE_PARTIAL,
    METRICS_REDUCER_VERSION,
    TrajectoryEvidenceError,
    materialize_trajectory_metrics,
    reduce_trajectory,
    reduce_trajectory_evidence,
    refresh_trajectory_metrics,
    trajectory_digest,
)
from alphaapollo.common.trajectory.schemas import TrajectoryEvent, TrajectoryEventType
from tests.common.environment.fakes import CanonicalTestToolBridge


def test_reducer_reads_the_same_python_tool_id_the_catalog_writes() -> None:
    """The reducer restates the tool id instead of importing the catalog.

    `metrics.py` deliberately keeps its own copy so reducing a trajectory does
    not pull the platform-specific execution aggregate in at evaluation import
    time. That is only safe while the two spellings agree: renaming the catalog
    id alone would silently zero the python-tool failure and repair counters
    rather than fail.
    """

    assert metrics_module.INTERNAL_PYTHON_TOOL_ID == INTERNAL_PYTHON_TOOL_ID


def build_store(jsonl_path: Path, artifact_root: Path) -> TrajectoryStore:
    return TrajectoryStore(
        jsonl_path=jsonl_path,
        artifacts=ArtifactStore(artifact_root),
    )


class FakeToolExecutor:
    def execute(self, request: ToolRequest, _context: ExecutionContext) -> ToolResponse:
        return ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stdout="(fake tool output)",
            stderr="",
            exit_code=0,
        )


def _append(store, event_type, branch_id: str, payload: dict) -> None:
    store.append(
        TrajectoryEvent(
            type=event_type,
            event_id=uuid.uuid4().hex,
            session_id="session",
            branch_id=branch_id,
            timestamp=datetime.now(timezone.utc),
            payload=payload,
        )
    )


def _model_turn(store, branch_id: str, prompt: int, completion: int) -> None:
    _append(
        store,
        TrajectoryEventType.EVIDENCE_ADDED,
        branch_id,
        {
            "capture_kind": "model_interaction",
            "role": "solver",
            "round": 0,
            "turn": 0,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        },
    )


def _python_call(store, branch_id: str, *, ok: bool, round_index: int = 0) -> None:
    _append(
        store,
        TrajectoryEventType.TOOL_CALLED if ok else TrajectoryEventType.TOOL_FAILED,
        branch_id,
        {
            "role": "solver",
            "round": round_index,
            "request": {"tool_id": "python", "call_id": uuid.uuid4().hex},
        },
    )


def test_reducer_counts_every_replay_without_cross_attempt_repair(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "node-0", 10, 2)
    _python_call(store, "node-0", ok=False)
    _model_turn(store, "node-0~1", 20, 3)
    _python_call(store, "node-0~1", ok=True)
    store.close()

    metrics = reduce_trajectory(path)

    assert metrics.solver_tool_calls == 2
    assert metrics.solver_tool_failures == 1
    assert metrics.solver_python_tool_failures == 1
    assert metrics.solver_python_tool_repairs == 0
    assert metrics.prompt_tokens == 30
    assert metrics.completion_tokens == 5


def test_reducer_counts_next_same_branch_python_success_as_repair(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _python_call(store, "solver", ok=False)
    _python_call(store, "solver", ok=True)
    store.close()

    assert reduce_trajectory(path).solver_python_tool_repairs == 1


def test_reducer_does_not_join_reused_branch_across_graph_invocations(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "node-0~1", 1, 1)
    _python_call(store, "node-0~1", ok=False)
    _model_turn(store, "node-0~1", 1, 1)
    _python_call(store, "node-0~1", ok=True)
    store.close()

    metrics = reduce_trajectory(path)

    assert metrics.solver_tool_calls == 2
    assert metrics.solver_python_tool_failures == 1
    assert metrics.solver_python_tool_repairs == 0


def test_reducer_does_not_join_repairs_across_runtime_rounds(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _python_call(store, "solver", ok=False, round_index=0)
    _python_call(store, "solver", ok=True, round_index=1)
    store.close()

    assert reduce_trajectory(path).solver_python_tool_repairs == 0


def test_reducer_counts_rejected_parse_call_as_generic_failure(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _append(
        store,
        TrajectoryEventType.TOOL_FAILED,
        "solver",
        {
            "role": "solver",
            "round": 0,
            "request": {
                "source": "python_code",
                "stage": "parse",
                "code": "multiple_blocks",
                "attempted": False,
                "canonicalization_status": "rejected_before_resolution",
            },
        },
    )
    store.close()

    metrics = reduce_trajectory(path)

    assert metrics.solver_tool_calls == 1
    assert metrics.solver_tool_failures == 1
    assert metrics.solver_python_tool_failures == 0
    assert metrics.solver_python_tool_repairs == 0


def test_missing_tool_id_execute_failure_is_not_accepted_as_parse_rejection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _append(
        store,
        TrajectoryEventType.TOOL_FAILED,
        "solver",
        {
            "role": "solver",
            "round_index": 0,
            "tool_id": None,
            "stage": "execute",
            "code": "nonzero_exit",
            "attempted": True,
        },
    )
    store.close()

    with pytest.raises(TrajectoryEvidenceError, match="no canonical tool_id"):
        reduce_trajectory(path)


def test_production_environment_malformed_call_reduces_as_generic_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(FakeToolExecutor()),
        event_sink=TrajectoryEnvironmentEventSink(store),
    )
    env.init(session_id="production-malformed", actor="solver", branch_id="solver")
    env.step("<python_code></python_code>")
    env.close()
    store.close()

    metrics = reduce_trajectory(path)

    assert metrics.solver_tool_calls == 1
    assert metrics.solver_tool_failures == 1
    assert metrics.solver_python_tool_failures == 0
    assert metrics.solver_python_tool_repairs == 0

    failed = next(
        event
        for event in map(json.loads, path.read_text().splitlines())
        if event["type"] == TrajectoryEventType.TOOL_FAILED.value
    )
    assert failed["payload"]["attempted"] is False
    assert failed["payload"]["canonicalization_status"] == "rejected_before_resolution"


def test_unversioned_historical_events_are_readable_but_not_reducible(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traj.jsonl"
    event = TrajectoryEvent(
        type=TrajectoryEventType.EVIDENCE_ADDED,
        event_id="legacy",
        session_id="session",
        branch_id="solver",
        timestamp=datetime.now(timezone.utc),
        payload={
            "capture_kind": "model_interaction",
            "prompt_tokens": 1,
            "completion_tokens": 1,
        },
    )
    path.write_text(json.dumps(event.model_dump(mode="json")) + "\n", encoding="utf-8")

    with pytest.raises(TrajectoryEvidenceError, match="evidence schema None"):
        reduce_trajectory(path)


def test_stale_reducer_view_refreshes_from_digest_bound_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "solver", 7, 3)
    _python_call(store, "solver", ok=False)
    store.close()
    current = materialize_trajectory_metrics({"solver_tool_calls": 999}, path)
    stale = {
        **current,
        "metrics_reducer_version": METRICS_REDUCER_VERSION - 1,
        "solver_tool_calls": 999,
        "prompt_tokens": 999,
    }

    refreshed, changed = refresh_trajectory_metrics(stale, path)

    assert changed is True
    assert refreshed["metrics_reducer_version"] == METRICS_REDUCER_VERSION
    assert refreshed["solver_tool_calls"] == 1
    assert refreshed["prompt_tokens"] == 7


def test_current_reducer_view_repairs_tampered_materialized_metrics(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "solver", 7, 3)
    _python_call(store, "solver", ok=False)
    store.close()
    tampered = {
        **materialize_trajectory_metrics({}, path),
        "solver_tool_calls": 999,
        "prompt_tokens": 999,
    }

    refreshed, changed = refresh_trajectory_metrics(tampered, path)

    assert changed is True
    assert refreshed["metrics_reducer_version"] == METRICS_REDUCER_VERSION
    assert refreshed["solver_tool_calls"] == 1
    assert refreshed["prompt_tokens"] == 7


def test_current_reducer_validates_digest_bound_event_schema(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "solver", 1, 1)
    store.close()
    payload = materialize_trajectory_metrics({}, path)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        event.pop("trajectory_schema_version")
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    payload["trajectory_digest"] = trajectory_digest(path)

    with pytest.raises(TrajectoryEvidenceError, match="evidence schema None"):
        refresh_trajectory_metrics(payload, path)


def test_current_reducer_rejects_digest_bound_empty_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "solver", 1, 1)
    store.close()
    payload = materialize_trajectory_metrics({}, path)
    path.write_text("")
    payload["trajectory_digest"] = trajectory_digest(path)

    with pytest.raises(TrajectoryEvidenceError, match="no events"):
        refresh_trajectory_metrics(payload, path)


def test_digest_mismatch_is_not_silently_reduced(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "solver", 1, 1)
    store.close()
    payload = materialize_trajectory_metrics({}, path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(TrajectoryEvidenceError, match="digest"):
        refresh_trajectory_metrics(payload, path)


def test_future_reducer_version_is_not_silently_downgraded(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "solver", 1, 1)
    store.close()
    payload = {
        **materialize_trajectory_metrics({}, path),
        "metrics_reducer_version": METRICS_REDUCER_VERSION + 1,
    }

    with pytest.raises(TrajectoryEvidenceError, match="newer than supported"):
        refresh_trajectory_metrics(payload, path)


@pytest.mark.parametrize("field", ("trajectory_schema_version", "metrics_reducer_version"))
def test_boolean_versions_are_rejected(field: str, tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "solver", 1, 1)
    store.close()
    payload = {**materialize_trajectory_metrics({}, path), field: True}

    with pytest.raises(TrajectoryEvidenceError):
        refresh_trajectory_metrics(payload, path)


def _environment_closed(store, branch_id: str, event_errors: list[str]) -> None:
    _append(
        store,
        TrajectoryEventType.EVIDENCE_ADDED,
        branch_id,
        {
            "environment_event": "environment_closed",
            "actor": "solver",
            "cleanup_errors": [],
            "event_errors": event_errors,
        },
    )


def test_reported_lost_event_downgrades_complete_to_partial(tmp_path: Path) -> None:
    """A run can finish cleanly and still only prove a lower bound.

    ``DefaultEnvironment._emit`` never lets a trajectory write failure abort a live run,
    so a dropped ToolCalled leaves a short ledger behind a successful run. Publishing
    that as ``complete`` is the under-count this module exists to prevent.
    """

    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "main", 10, 2)
    _environment_closed(store, "main", ["tool_execution_completed: OSError: recording failed"])
    store.close()

    metrics, gaps = reduce_trajectory_evidence(path)

    assert metrics.prompt_tokens == 10  # what survived is still real
    assert gaps == ("tool_execution_completed: OSError: recording failed",)

    payload = materialize_trajectory_metrics({}, path, status=METRICS_EVIDENCE_COMPLETE)

    assert payload["metrics_evidence_status"] == METRICS_EVIDENCE_PARTIAL
    assert payload["metrics_evidence_gaps"] == list(gaps)


def test_intact_ledger_keeps_the_status_it_was_given(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "main", 10, 2)
    _environment_closed(store, "main", [])
    store.close()

    assert reduce_trajectory_evidence(path)[1] == ()
    payload = materialize_trajectory_metrics({}, path, status=METRICS_EVIDENCE_COMPLETE)
    assert payload["metrics_evidence_status"] == METRICS_EVIDENCE_COMPLETE
    assert "metrics_evidence_gaps" not in payload


def test_a_typed_error_stays_partial_over_an_intact_ledger(tmp_path: Path) -> None:
    """A whole trajectory is not evidence that the run itself finished."""

    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "main", 10, 2)
    store.close()

    payload = materialize_trajectory_metrics({}, path, status=METRICS_EVIDENCE_PARTIAL)

    assert payload["metrics_evidence_status"] == METRICS_EVIDENCE_PARTIAL


def test_refresh_cannot_keep_asserting_complete_over_a_gapped_ledger(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "main", 10, 2)
    _environment_closed(store, "main", ["tool_execution_completed: OSError: recording failed"])
    store.close()
    # A hand-edited (or pre-fix) row that claims whole evidence, with a correct digest.
    payload = {
        **materialize_trajectory_metrics({}, path),
        "metrics_evidence_status": METRICS_EVIDENCE_COMPLETE,
    }

    refreshed, changed = refresh_trajectory_metrics(payload, path)

    assert changed is True
    assert refreshed["metrics_evidence_status"] == METRICS_EVIDENCE_PARTIAL
    assert refreshed["metrics_evidence_gaps"]


def test_malformed_event_errors_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    store = build_store(path, tmp_path / "artifacts")
    _model_turn(store, "main", 10, 2)
    _append(
        store,
        TrajectoryEventType.EVIDENCE_ADDED,
        "main",
        {"environment_event": "environment_closed", "event_errors": "disk full"},
    )
    store.close()

    with pytest.raises(TrajectoryEvidenceError, match="event_errors"):
        reduce_trajectory(path)
