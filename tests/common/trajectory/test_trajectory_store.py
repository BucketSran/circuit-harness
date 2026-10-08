from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alphaapollo.common.artifacts.schemas import ProvenanceRef
from alphaapollo.common.execution import ArtifactStore
from alphaapollo.common.trajectory import (
    TrajectoryQuery,
    TrajectoryReader,
    TrajectorySink,
    TrajectoryStore,
    TrajectoryStoreError,
)
from alphaapollo.common.trajectory.metrics import reduce_trajectory
from alphaapollo.common.trajectory.schemas import (
    TRAJECTORY_EVENT_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryEvent,
    TrajectoryEventType,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _event(
    event_id: str,
    event_type: TrajectoryEventType,
    *,
    payload: dict | None = None,
    session_id: str = "session-1",
) -> TrajectoryEvent:
    return TrajectoryEvent(
        type=event_type,
        event_id=event_id,
        session_id=session_id,
        branch_id="main",
        timestamp=datetime.now(timezone.utc),
        payload=payload or {},
        provenance=ProvenanceRef(id="test"),
    )


def _store(tmp_path: Path) -> TrajectoryStore:
    return TrajectoryStore(
        jsonl_path=tmp_path / "trajectory.jsonl",
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )


def test_store_satisfies_protocols_and_assigns_replay_sequence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert isinstance(store, TrajectorySink)
    assert isinstance(store, TrajectoryReader)

    unstamped = _event("e1", TrajectoryEventType.PROBLEM_UNDERSTOOD)
    assert unstamped.trajectory_schema_version is None
    first = store.append(unstamped)
    second = store.append(_event("e2", TrajectoryEventType.HYPOTHESIS_CREATED))

    assert first.sequence == 0
    assert second.sequence == 1
    assert first.trajectory_schema_version == TRAJECTORY_SCHEMA_VERSION
    assert second.trajectory_schema_version == TRAJECTORY_SCHEMA_VERSION
    assert store.read_events() == [first, second]
    store.close()
    reopened = TrajectoryStore(
        jsonl_path=store.jsonl_path,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    assert reopened.read_events() == [first, second]
    reopened.close()


def test_new_trajectory_uses_owner_only_file_permissions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append(_event("private", TrajectoryEventType.PROBLEM_UNDERSTOOD))

    assert store.jsonl_path.stat().st_mode & 0o777 == 0o600


def test_existing_trajectory_permissions_are_tightened_on_open(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    first = TrajectoryStore(jsonl_path=path, artifacts=ArtifactStore(tmp_path / "artifacts"))
    first.append(_event("one", TrajectoryEventType.PROBLEM_UNDERSTOOD))
    first.close()
    path.chmod(0o644)

    reopened = TrajectoryStore(jsonl_path=path, artifacts=ArtifactStore(tmp_path / "artifacts"))
    try:
        assert path.stat().st_mode & 0o777 == 0o600
    finally:
        reopened.close()


def test_second_writer_is_rejected_until_first_closes(tmp_path: Path) -> None:
    first = _store(tmp_path)
    with pytest.raises(TrajectoryStoreError, match="already has a writer"):
        _store(tmp_path)
    first.close()
    reopened = _store(tmp_path)
    reopened.close()


def test_single_store_serializes_threaded_appends(tmp_path: Path) -> None:
    store = _store(tmp_path)

    def append(index: int) -> int:
        return store.append(_event(f"thread-{index}", TrajectoryEventType.EVIDENCE_ADDED)).sequence

    with ThreadPoolExecutor(max_workers=8) as pool:
        sequences = list(pool.map(append, range(40)))

    assert sorted(sequences) == list(range(40))
    assert [event.sequence for event in store.read_events()] == list(range(40))


def test_forked_child_cannot_reuse_inherited_writer(tmp_path: Path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method unavailable")
    store = _store(tmp_path)
    parent, child = multiprocessing.get_context("fork").Pipe(duplex=False)

    def attempt_append() -> None:
        try:
            store.append(_event("child", TrajectoryEventType.EVIDENCE_ADDED))
        except Exception as exc:
            child.send((type(exc).__name__, str(exc)))
        finally:
            child.close()

    process = multiprocessing.get_context("fork").Process(target=attempt_append)
    process.start()
    process.join(timeout=5)

    assert process.exitcode == 0
    assert parent.recv() == (
        "TrajectoryStoreError",
        "trajectory writer cannot be used after fork; create a new store in the child",
    )
    assert store.read_events() == []


def test_duplicate_id_and_out_of_order_sequence_fail_loudly(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append(_event("e1", TrajectoryEventType.PROBLEM_UNDERSTOOD))

    with pytest.raises(TrajectoryStoreError, match="duplicate"):
        store.append(_event("e1", TrajectoryEventType.EVIDENCE_ADDED))
    with pytest.raises(TrajectoryStoreError, match="sequence must be 1"):
        store.append(
            _event("e2", TrajectoryEventType.EVIDENCE_ADDED).model_copy(update={"sequence": 7})
        )


def test_training_query_returns_clean_eligible_semantic_events(tmp_path: Path) -> None:
    store = _store(tmp_path)
    eligible = TrajectoryStore.stamp_payload(
        {
            "prompt": "question",
            "response": "answer",
            "training_eligibility": True,
        },
        source_flags=[],
    )
    contaminated = TrajectoryStore.stamp_payload(
        {
            "prompt": "seen",
            "response": "bad",
            "training_eligibility": True,
        },
        source_flags=["benchmark_seen"],
    )
    store.append(_event("clean", TrajectoryEventType.SOLUTION_PACKAGED, payload=eligible))
    store.append(_event("dirty", TrajectoryEventType.SOLUTION_PACKAGED, payload=contaminated))
    store.append(
        _event(
            "audit-only",
            TrajectoryEventType.TOOL_CALLED,
            payload={"training_eligibility": False},
        )
    )
    store.append(
        _event(
            "string-false",
            TrajectoryEventType.SOLUTION_PACKAGED,
            payload={"training_eligibility": "false"},
        )
    )

    all_eligible = store.export(TrajectoryQuery(training_eligible_only=True))
    exported = store.export(
        TrajectoryQuery(
            training_eligible_only=True,
            exclude_contaminated=True,
        )
    )

    assert [event.event_id for event in all_eligible] == ["clean", "dirty"]
    assert [event.event_id for event in exported] == ["clean"]
    assert (
        store.export_json(
            TrajectoryQuery(
                training_eligible_only=True,
                exclude_contaminated=True,
            )
        )[0]["payload"]["response"]
        == "answer"
    )


def test_stamp_payload_redacts_only_explicit_paths_and_records_them() -> None:
    source = {
        "response": "model answer",
        "tool_args": {"expected": "legitimate tool argument"},
        "evaluation": {"ground_truth": "secret", "kept": 1},
    }

    stamped = TrajectoryStore.stamp_payload(
        source,
        source_flags=[],
        redact_paths=(("evaluation", "ground_truth"),),
    )

    assert stamped["evaluation"] == {"kept": 1}
    assert stamped["tool_args"]["expected"] == "legitimate tool argument"
    assert stamped["_redactions"] == ["evaluation.ground_truth"]
    assert source["evaluation"]["ground_truth"] == "secret"
    assert "training_eligibility" not in stamped


def test_blob_reference_recovers_exact_large_payload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ref = store.put_blob(
        b"long completion",
        type_="model-response",
        created_by="generation",
        provenance=ProvenanceRef(id="model-a"),
    )
    event = _event(
        "response",
        TrajectoryEventType.HYPOTHESIS_CREATED,
        payload={"response_ref": ref.model_dump(mode="json")},
    ).model_copy(update={"artifact_refs": [ref]})
    stored = store.append(event)

    assert store.get_blob(stored.artifact_refs[0]) == b"long completion"


def test_contaminated_artifact_reference_is_excluded_from_clean_query(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ref = store.put_blob(
        b"seen benchmark output",
        type_="model-response",
        created_by="generation",
        contamination_flags=["benchmark_seen"],
    )
    store.append(
        _event(
            "contaminated-ref",
            TrajectoryEventType.HYPOTHESIS_CREATED,
            payload={"training_eligibility": True},
        ).model_copy(update={"artifact_refs": [ref]})
    )

    assert ref.contamination_flags == ["benchmark_seen"]
    assert store.export(TrajectoryQuery(exclude_contaminated=True)) == []


def test_malformed_line_names_its_line_number(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append(_event("ok", TrajectoryEventType.PROBLEM_UNDERSTOOD))
    with store.jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")

    with pytest.raises(TrajectoryStoreError, match="line 2"):
        store.read_events()


def test_reattach_rejects_sequence_that_disagrees_with_file_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    event = _event("bad-sequence", TrajectoryEventType.PROBLEM_UNDERSTOOD).model_copy(
        update={"sequence": 9}
    )
    store.jsonl_path.write_text(event.model_dump_json() + "\n", encoding="utf-8")
    store.close()

    with pytest.raises(TrajectoryStoreError, match="expected 0"):
        _store(tmp_path)


def test_reattach_repairs_one_incomplete_trailing_line(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.append(_event("e1", TrajectoryEventType.PROBLEM_UNDERSTOOD))
    with store.jsonl_path.open("ab") as handle:
        handle.write(b'{"type":"Hypothesis')

    store.close()
    recovered = _store(tmp_path)
    second = recovered.append(_event("e2", TrajectoryEventType.HYPOTHESIS_CREATED))

    assert recovered.read_events() == [first, second]
    assert second.sequence == 1


def test_reattach_terminates_complete_json_without_newline_before_append(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _event("e1", TrajectoryEventType.PROBLEM_UNDERSTOOD).model_copy(update={"sequence": 0})
    store.jsonl_path.write_text(first.model_dump_json(), encoding="utf-8")
    store.close()

    recovered = _store(tmp_path)
    second = recovered.append(_event("e2", TrajectoryEventType.EVIDENCE_ADDED))

    assert recovered.jsonl_path.read_bytes().count(b"\n") == 2
    assert recovered.read_events() == [first, second]


def test_reattach_does_not_truncate_complete_unsupported_tail(tmp_path: Path) -> None:
    store = _store(tmp_path)
    event = _event("future", TrajectoryEventType.PROBLEM_UNDERSTOOD).model_copy(
        update={"schema_version": "2.0", "sequence": 0}
    )
    raw = event.model_dump_json().encode("utf-8")
    store.jsonl_path.write_bytes(raw)
    store.close()

    with pytest.raises(TrajectoryStoreError, match="unsupported trajectory schema"):
        _store(tmp_path)

    assert store.jsonl_path.read_bytes() == raw


def test_append_oserror_resynchronizes_when_line_reached_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)

    def fail_fsync(_fd: int) -> None:
        raise OSError("simulated fsync failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fsync", fail_fsync)
        with pytest.raises(TrajectoryStoreError, match="resynchronized"):
            store.append(_event("uncertain", TrajectoryEventType.PROBLEM_UNDERSTOOD))

    second = store.append(_event("confirmed", TrajectoryEventType.EVIDENCE_ADDED))
    assert second.sequence == 1
    assert [event.event_id for event in store.read_events()] == [
        "uncertain",
        "confirmed",
    ]


def test_reader_accepts_unknown_additive_field_but_rejects_new_major(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    additive = _event("future", TrajectoryEventType.PROBLEM_UNDERSTOOD)
    raw = additive.model_dump(mode="json")
    raw["future_field"] = "preserved"
    store.jsonl_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    restored = store.read_events()[0]
    assert restored.model_extra == {"future_field": "preserved"}

    raw["schema_version"] = "2.0"
    store.jsonl_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    store.close()
    with pytest.raises(TrajectoryStoreError, match="unsupported trajectory schema"):
        TrajectoryStore(
            jsonl_path=store.jsonl_path,
            artifacts=ArtifactStore(tmp_path / "artifacts"),
        )


def test_trajectory_written_before_event_schema_1_1_still_opens(tmp_path: Path) -> None:
    """Round trip a real pre-1.1 ledger, not a hand-built dict.

    ``fixtures/trajectory_event_schema_1_0.jsonl`` was written by ``TrajectoryStore``
    at commit 839efcc4, before ``TrajectoryEvent.cost_delta`` was removed, so every
    line carries the field. ``TRAJECTORY_SCHEMA_VERSION`` is validated for exact
    equality by both readers below; this file is the evidence that removing an event
    field left it at 1.
    """

    jsonl_path = tmp_path / "trajectory.jsonl"
    shutil.copyfile(_FIXTURES / "trajectory_event_schema_1_0.jsonl", jsonl_path)
    store = TrajectoryStore(jsonl_path=jsonl_path, artifacts=ArtifactStore(tmp_path / "art"))

    recorded = store.read_events()
    assert [event.event_id for event in recorded] == [
        "event-1",
        "event-2",
        "event-3",
        "event-4",
    ]
    assert [event.schema_version for event in recorded] == ["1.0"] * 4
    assert [event.trajectory_schema_version for event in recorded] == [
        TRAJECTORY_SCHEMA_VERSION
    ] * 4
    # The removed field survives as an extra rather than failing validation.
    assert [sorted(event.model_extra or {}) for event in recorded] == [["cost_delta"]] * 4
    assert recorded[2].model_extra["cost_delta"]["tokens"] == 7

    appended = store.append(_event("event-5", TrajectoryEventType.CLAIM_VERIFIED))
    assert appended.schema_version == TRAJECTORY_EVENT_SCHEMA_VERSION
    assert appended.sequence == 4
    store.close()

    reopened = TrajectoryStore(jsonl_path=jsonl_path, artifacts=ArtifactStore(tmp_path / "art"))
    assert [event.schema_version for event in reopened.read_events()] == ["1.0"] * 4 + ["1.1"]
    reopened.close()

    # metrics.py is the second exact-version reader of the same file.
    assert reduce_trajectory(jsonl_path).prompt_tokens == 11


def test_query_filters_session_branch_and_event_type(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append(_event("s1", TrajectoryEventType.TOOL_CALLED, session_id="one"))
    store.append(_event("s2", TrajectoryEventType.VERIFIER_PASSED, session_id="two"))

    query = TrajectoryQuery(
        session_id="two",
        branch_id="main",
        event_types=(TrajectoryEventType.VERIFIER_PASSED,),
        exclude_contaminated=False,
    )
    assert [event.event_id for event in store.read_events(query)] == ["s2"]
