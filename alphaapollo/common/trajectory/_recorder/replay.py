# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Typed validation and read-back of persisted environment captures."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from alphaapollo.common.artifacts.schemas import ArtifactRef, require_json_value
from alphaapollo.common.trajectory._recorder.contracts import (
    EnvironmentCaptureKind,
    TrajectoryCaptureReader,
)
from alphaapollo.common.trajectory.episode import TrajectoryQuery


class EnvironmentTrajectoryError(ValueError):
    """Raised when an Environment capture cannot be safely replayed."""


@dataclass(frozen=True, slots=True)
class EnvironmentCaptureRecord:
    kind: EnvironmentCaptureKind
    session_id: str
    branch_id: str
    actor: str
    round_index: int
    step_index: int
    content: Any
    redactions: tuple[str, ...]
    contamination_flags: tuple[str, ...]
    event_id: str
    sequence: int
    artifact_ref: ArtifactRef

    def __post_init__(self) -> None:
        content = copy.deepcopy(self.content)
        require_json_value(content, path="$.content")
        object.__setattr__(self, "content", content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "session_id": self.session_id,
            "branch_id": self.branch_id,
            "actor": self.actor,
            "round_index": self.round_index,
            "step_index": self.step_index,
            "content": copy.deepcopy(self.content),
            "redactions": list(self.redactions),
            "contamination_flags": list(self.contamination_flags),
            "event_id": self.event_id,
            "sequence": self.sequence,
            "artifact_ref": self.artifact_ref.model_dump(mode="json"),
        }


@dataclass(frozen=True, slots=True)
class EnvironmentReplayStep:
    round_index: int
    step_index: int
    model_outputs: tuple[EnvironmentCaptureRecord, ...]
    tool_requests: tuple[EnvironmentCaptureRecord, ...]
    tool_results: tuple[EnvironmentCaptureRecord, ...]
    observations: tuple[EnvironmentCaptureRecord, ...]
    final_outputs: tuple[EnvironmentCaptureRecord, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentTrajectoryView:
    session_id: str
    branch_id: str | None
    records: tuple[EnvironmentCaptureRecord, ...]

    def records_of(
        self,
        kind: EnvironmentCaptureKind,
    ) -> tuple[EnvironmentCaptureRecord, ...]:
        return tuple(record for record in self.records if record.kind is kind)

    def export_json(self) -> list[dict[str, Any]]:
        """Return a detached JSON-compatible view for Learning or inspection."""
        return [record.to_dict() for record in self.records]

    def replay_steps(self) -> tuple[EnvironmentReplayStep, ...]:
        by_step: dict[tuple[int, int], list[EnvironmentCaptureRecord]] = {}
        for record in self.records:
            if record.kind is EnvironmentCaptureKind.MODEL_INPUT:
                continue
            by_step.setdefault(
                (record.round_index, record.step_index),
                [],
            ).append(record)
        steps: list[EnvironmentReplayStep] = []
        for round_index, step_index in sorted(by_step):
            records = by_step[(round_index, step_index)]
            steps.append(
                EnvironmentReplayStep(
                    round_index=round_index,
                    step_index=step_index,
                    model_outputs=_select(records, EnvironmentCaptureKind.MODEL_OUTPUT),
                    tool_requests=_select(records, EnvironmentCaptureKind.TOOL_REQUEST),
                    tool_results=tuple(
                        record
                        for record in records
                        if record.kind
                        in {
                            EnvironmentCaptureKind.TOOL_RESPONSE,
                            EnvironmentCaptureKind.TOOL_ERROR,
                        }
                    ),
                    observations=_select(
                        records,
                        EnvironmentCaptureKind.ENVIRONMENT_OBSERVATION,
                    ),
                    final_outputs=_select(
                        records,
                        EnvironmentCaptureKind.FINAL_OUTPUT,
                    ),
                )
            )
        return tuple(steps)


class EnvironmentTrajectoryReader:
    """Resolve and validate Environment capture artifacts from Common records."""

    def __init__(self, store: TrajectoryCaptureReader) -> None:
        if not isinstance(store, TrajectoryCaptureReader):
            raise TypeError("EnvironmentTrajectoryReader requires read_events/get_blob")
        self._store = store

    def read_session(
        self,
        session_id: str,
        *,
        branch_id: str | None = None,
        actor: str | None = None,
        exclude_contaminated: bool = False,
    ) -> EnvironmentTrajectoryView:
        if not session_id.strip():
            raise ValueError("session_id must be non-empty")
        events = self._store.read_events(
            TrajectoryQuery(
                session_id=session_id,
                branch_id=branch_id,
                exclude_contaminated=exclude_contaminated,
            )
        )
        records: list[EnvironmentCaptureRecord] = []
        for event in events:
            capture_ref_value = event.payload.get("capture_ref")
            if capture_ref_value is None:
                continue
            try:
                ref = ArtifactRef.model_validate(capture_ref_value)
            except Exception as exc:
                raise EnvironmentTrajectoryError(
                    f"event {event.event_id} has invalid capture_ref"
                ) from exc
            if not any(
                candidate.id == ref.id and candidate.hash == ref.hash
                for candidate in event.artifact_refs
            ):
                raise EnvironmentTrajectoryError(
                    f"event {event.event_id} capture_ref is not attached to the event"
                )
            try:
                envelope = json.loads(self._store.get_blob(ref).decode("utf-8"))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EnvironmentTrajectoryError(
                    f"event {event.event_id} capture artifact is unreadable"
                ) from exc
            record = _record_from_envelope(event, ref, envelope)
            if actor is None or record.actor == actor:
                records.append(record)
        records.sort(key=lambda record: record.sequence)
        return EnvironmentTrajectoryView(
            session_id=session_id,
            branch_id=branch_id,
            records=tuple(records),
        )


def _record_from_envelope(
    event: Any,
    ref: ArtifactRef,
    envelope: Any,
) -> EnvironmentCaptureRecord:
    if not isinstance(envelope, dict) or envelope.get("capture_version") != 1:
        raise EnvironmentTrajectoryError(
            f"event {event.event_id} uses an unsupported capture envelope"
        )
    try:
        kind = EnvironmentCaptureKind(envelope["kind"])
        session_id = envelope["session_id"]
        branch_id = envelope["branch_id"]
        actor = envelope["actor"]
        round_index = envelope["round_index"]
        step_index = envelope["step_index"]
        content = envelope["content"]
        redactions = tuple(envelope.get("redactions", ()))
        source_flags = tuple(envelope.get("source_contamination_flags", ()))
    except (KeyError, TypeError, ValueError) as exc:
        raise EnvironmentTrajectoryError(
            f"event {event.event_id} has a malformed capture envelope"
        ) from exc
    if (
        session_id != event.session_id
        or branch_id != event.branch_id
        or actor != event.payload.get("actor")
        or round_index != event.payload.get("round_index")
        or step_index != event.payload.get("step_index")
        or kind.value != event.payload.get("capture_kind")
    ):
        raise EnvironmentTrajectoryError(
            f"event {event.event_id} capture metadata does not match its envelope"
        )
    expected_flags = tuple(sorted(ref.contamination_flags))
    payload_flags = tuple(sorted(event.payload.get("capture_contamination_flags", ())))
    expected_status = "redacted" if expected_flags else "captured"
    if (
        payload_flags != expected_flags
        or event.payload.get("capture_status") != expected_status
        or (expected_flags and not redactions and not source_flags)
    ):
        raise EnvironmentTrajectoryError(
            f"event {event.event_id} capture redaction metadata is inconsistent"
        )
    if event.sequence is None:
        raise EnvironmentTrajectoryError(f"event {event.event_id} has no replay sequence")
    if (
        not isinstance(session_id, str)
        or not isinstance(branch_id, str)
        or not isinstance(actor, str)
        or isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index < 0
        or isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or any(not isinstance(path, str) for path in redactions)
        or any(not isinstance(flag, str) for flag in source_flags)
    ):
        raise EnvironmentTrajectoryError(f"event {event.event_id} has invalid capture field types")
    return EnvironmentCaptureRecord(
        kind=kind,
        session_id=session_id,
        branch_id=branch_id,
        actor=actor,
        round_index=round_index,
        step_index=step_index,
        content=content,
        redactions=redactions,
        contamination_flags=tuple(ref.contamination_flags),
        event_id=event.event_id,
        sequence=event.sequence,
        artifact_ref=ref,
    )


def _select(
    records: list[EnvironmentCaptureRecord],
    kind: EnvironmentCaptureKind,
) -> tuple[EnvironmentCaptureRecord, ...]:
    return tuple(record for record in records if record.kind is kind)
