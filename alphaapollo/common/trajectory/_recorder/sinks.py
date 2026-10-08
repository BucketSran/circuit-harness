# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""In-memory and persistent sinks for environment capture events."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from alphaapollo.common.artifacts.schemas import ArtifactRef, ProvenanceRef
from alphaapollo.common.trajectory._recorder.contracts import (
    EnvironmentCapture,
    EnvironmentEventKind,
    TrajectoryCaptureStore,
)
from alphaapollo.common.trajectory._recorder.sanitize import (
    _CAPTURE_MEDIA_TYPE,
    _CAPTURE_PROVENANCE,
    _canonical_json,
    build_capture_envelope,
    sanitize_capture_value,
)
from alphaapollo.common.trajectory.schemas import TrajectoryEvent, TrajectoryEventType

if TYPE_CHECKING:
    from alphaapollo.common.environment.base import EnvironmentSession


class NullEnvironmentEventSink:
    def emit(
        self,
        kind: EnvironmentEventKind,
        session: EnvironmentSession,
        payload: dict[str, Any] | None = None,
        capture: EnvironmentCapture | None = None,
    ) -> None:
        return None


@dataclass(frozen=True, slots=True)
class RecordedEnvironmentEvent:
    kind: EnvironmentEventKind
    session: dict[str, Any]
    payload: dict[str, Any]
    capture: dict[str, Any] | None = None


class RecordingEnvironmentEventSink:
    """In-memory sink that applies the same redaction policy as persistence."""

    def __init__(self) -> None:
        self.events: list[RecordedEnvironmentEvent] = []

    def emit(
        self,
        kind: EnvironmentEventKind,
        session: EnvironmentSession,
        payload: dict[str, Any] | None = None,
        capture: EnvironmentCapture | None = None,
    ) -> None:
        copied, _, _ = sanitize_capture_value(payload or {})
        if not isinstance(copied, dict):
            raise TypeError("environment event payload must remain an object")
        envelope = None
        if capture is not None:
            envelope, _ = build_capture_envelope(session, capture)
        self.events.append(
            RecordedEnvironmentEvent(
                kind=kind,
                session=session.snapshot(),
                payload=copied,
                capture=envelope,
            )
        )


class TrajectoryEnvironmentEventSink:
    """Persist ordered audit metadata plus hash-bound full interaction artifacts."""

    def __init__(self, store: TrajectoryCaptureStore) -> None:
        if not isinstance(store, TrajectoryCaptureStore):
            raise TypeError("TrajectoryEnvironmentEventSink requires append/put_blob/get_blob")
        self._store = store

    def emit(
        self,
        kind: EnvironmentEventKind,
        session: EnvironmentSession,
        payload: dict[str, Any] | None = None,
        capture: EnvironmentCapture | None = None,
    ) -> None:
        sanitized_payload, payload_flags, payload_redactions = sanitize_capture_value(payload or {})
        if not isinstance(sanitized_payload, dict):
            raise TypeError("environment event payload must remain an object")
        body = sanitized_payload
        body["environment_event"] = kind.value
        body["actor"] = session.actor
        body["round_index"] = session.round_index
        body["step_index"] = session.step_index

        event_flags = set(payload_flags)
        if payload_redactions:
            body["payload_redactions"] = list(payload_redactions)
        artifact_refs: list[ArtifactRef] = []
        if capture is not None:
            envelope, flags = build_capture_envelope(session, capture)
            content = _canonical_json(envelope)
            ref = self._store.put_blob(
                content,
                type_=_CAPTURE_MEDIA_TYPE,
                created_by="common.environment.trajectory_capture",
                provenance=_CAPTURE_PROVENANCE,
                contamination_flags=list(flags),
            )
            artifact_refs.append(ref)
            known_refs = {(ref.id, ref.hash)}
            artifact_refs.extend(
                candidate
                for candidate in capture.artifact_refs
                if (candidate.id, candidate.hash) not in known_refs
            )
            body["capture_kind"] = capture.kind.value
            body["capture_ref"] = ref.model_dump(mode="json")
            body["capture_status"] = "redacted" if flags else "captured"
            if flags:
                body["capture_contamination_flags"] = list(flags)
                event_flags.update(flags)
        if event_flags:
            body["contamination_flags"] = sorted(event_flags)

        event_type = TrajectoryEventType.EVIDENCE_ADDED
        if kind is EnvironmentEventKind.EXECUTION_COMPLETED:
            event_type = (
                TrajectoryEventType.TOOL_CALLED
                if body.get("ok") is True
                else TrajectoryEventType.TOOL_FAILED
            )
        elif kind is EnvironmentEventKind.PREFLIGHT_FAILED:
            event_type = TrajectoryEventType.TOOL_FAILED
        self._store.append(
            TrajectoryEvent(
                type=event_type,
                event_id=uuid.uuid4().hex,
                session_id=session.session_id,
                branch_id=session.branch_id,
                timestamp=datetime.now(timezone.utc),
                payload=body,
                provenance=ProvenanceRef(
                    id="common.environment.default_environment",
                    version="1",
                ),
                artifact_refs=artifact_refs,
            )
        )
