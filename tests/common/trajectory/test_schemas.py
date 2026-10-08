from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from alphaapollo.common.artifacts.schemas import ProvenanceRef
from alphaapollo.common.trajectory.schemas import (
    TRAJECTORY_EVENT_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryEvent,
    TrajectoryEventType,
    require_supported_trajectory_event_schema,
)


def test_historical_trajectory_json_remains_readable() -> None:
    historical = {
        "type": "HypothesisCreated",
        "event_id": "event-1",
        "session_id": "session-1",
        "branch_id": "main",
        "timestamp": "2026-07-17T00:00:00Z",
        "payload": {"candidate_id": "candidate-1", "answer": "42"},
        "cost_delta": {"tokens": 7},
        "provenance": {"id": "model-a"},
    }

    event = TrajectoryEvent.model_validate(historical)

    assert event.type is TrajectoryEventType.HYPOTHESIS_CREATED
    assert event.schema_version == TRAJECTORY_EVENT_SCHEMA_VERSION
    assert event.model_dump(mode="json", exclude_defaults=True) == historical
    # ``cost_delta`` left the model in event schema 1.1; a historical record keeps
    # it verbatim as an extra rather than losing it on the round trip.
    assert event.model_extra == {"cost_delta": {"tokens": 7}}


def test_event_schema_1_0_records_still_load_with_cost_delta() -> None:
    """A 1.0 event is why the 1.1 bump is minor rather than major."""

    recorded = {
        "type": "ToolCalled",
        "event_id": "event-2",
        "session_id": "session-1",
        "timestamp": "2026-07-17T00:00:00Z",
        "payload": {"tool": "python"},
        "cost_delta": {
            "tokens": 7,
            "reasoning_tokens": 3,
            "wall_clock_s": 0.5,
            "gpu_s": 0.0,
            "cpu_s": 0.0,
            "tool_calls": 1,
            "verifier_calls": 0,
            "sandbox_time_s": 0.0,
            "cache_hits": 0,
            "artifacts_produced": 0,
            "trust_level_improvement": 0,
            "branch_progress_score": 0.25,
            "budget_consumed_fraction": 0.125,
        },
        "schema_version": "1.0",
        "trajectory_schema_version": 1,
        "sequence": 0,
    }

    require_supported_trajectory_event_schema("1.0")
    event = TrajectoryEvent.model_validate(recorded)

    assert event.schema_version == "1.0"
    assert "cost_delta" not in TrajectoryEvent.model_fields
    # The whole record must survive ``_persistent_values_are_json_exact``, which
    # validates ``model_extra`` and would reject a non-JSON value there.
    assert event.model_extra == {"cost_delta": recorded["cost_delta"]}
    assert event.model_dump(mode="json")["cost_delta"] == recorded["cost_delta"]


def test_new_event_stamps_1_1_and_omits_cost_delta() -> None:
    event = TrajectoryEvent(
        type=TrajectoryEventType.TOOL_CALLED,
        event_id="event-3",
        session_id="session-1",
        timestamp=datetime(2026, 7, 17, tzinfo=timezone.utc),
    )

    assert TRAJECTORY_EVENT_SCHEMA_VERSION == "1.1"
    assert event.schema_version == "1.1"
    assert not event.model_extra
    assert "cost_delta" not in event.model_dump(mode="json")
    # Trajectory files are validated for an exact framing version, so removing an
    # event field must leave it alone.
    assert TRAJECTORY_SCHEMA_VERSION == 1


def test_trajectory_reader_accepts_additive_fields() -> None:
    event = TrajectoryEvent.model_validate(
        {
            "type": "ProblemUnderstood",
            "event_id": "future-event",
            "session_id": "session",
            "timestamp": "2026-07-17T00:00:00Z",
            "payload": {},
            "future_optional_field": {"value": 1},
        }
    )

    assert event.model_extra == {"future_optional_field": {"value": 1}}


def test_trajectory_event_rejects_naive_timestamp_and_non_json_payload() -> None:
    with pytest.raises(ValidationError):
        TrajectoryEvent(
            type=TrajectoryEventType.PROBLEM_UNDERSTOOD,
            event_id="naive",
            session_id="session",
            timestamp=datetime.now(),
            provenance=ProvenanceRef(id="test"),
        )
    with pytest.raises(ValidationError, match="JSON"):
        TrajectoryEvent(
            type=TrajectoryEventType.EVIDENCE_ADDED,
            event_id="unsafe",
            session_id="session",
            timestamp=datetime.now(timezone.utc),
            payload={"unsafe": ("tuple",)},
        )


def test_trajectory_schema_rejects_unknown_major() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        require_supported_trajectory_event_schema("2.0")
