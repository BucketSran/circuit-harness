# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Environment capture value objects and persistence protocols."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from alphaapollo.common.artifacts.schemas import ArtifactRef, ProvenanceRef
from alphaapollo.common.trajectory._recorder.sanitize import _bounded_capture_copy
from alphaapollo.common.trajectory.episode import TrajectoryQuery
from alphaapollo.common.trajectory.schemas import TrajectoryEvent

if TYPE_CHECKING:
    from alphaapollo.common.environment.base import EnvironmentSession


class EnvironmentEventKind(str, Enum):
    INITIALIZED = "environment_initialized"
    ACTION_RECEIVED = "agent_action_received"
    TOOL_REQUESTED = "tool_request_normalized"
    TOOL_REFUSED = "tool_refused"
    PREFLIGHT_FAILED = "preflight_failed"
    EXECUTION_ATTEMPTED = "tool_execution_attempted"
    EXECUTION_COMPLETED = "tool_execution_completed"
    OBSERVATION_EMITTED = "observation_emitted"
    TERMINATED = "environment_terminated"
    FAILED = "environment_failed"
    CLOSED = "environment_closed"


class EnvironmentCaptureKind(str, Enum):
    MODEL_INPUT = "model_input"
    MODEL_OUTPUT = "model_output"
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"
    TOOL_ERROR = "tool_error"
    ENVIRONMENT_OBSERVATION = "environment_observation"
    FINAL_OUTPUT = "final_output"


@dataclass(frozen=True, slots=True)
class EnvironmentCapture:
    """Full interaction content to offload into a governed artifact."""

    kind: EnvironmentCaptureKind
    content: Any
    artifact_refs: tuple[ArtifactRef, ...] = ()
    contamination_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        copied, traversal_flags = _bounded_capture_copy(self.content)
        object.__setattr__(self, "content", copied)
        if any(not isinstance(ref, ArtifactRef) for ref in self.artifact_refs):
            raise TypeError("capture artifact_refs must contain ArtifactRef values")
        object.__setattr__(self, "artifact_refs", tuple(self.artifact_refs))
        if any(not isinstance(flag, str) or not flag.strip() for flag in self.contamination_flags):
            raise ValueError("capture contamination_flags must be non-empty strings")
        object.__setattr__(
            self,
            "contamination_flags",
            tuple(sorted(set(self.contamination_flags).union(traversal_flags))),
        )


@runtime_checkable
class EnvironmentEventSink(Protocol):
    def emit(
        self,
        kind: EnvironmentEventKind,
        session: EnvironmentSession,
        payload: dict[str, Any] | None = None,
        capture: EnvironmentCapture | None = None,
    ) -> None: ...


@runtime_checkable
class TrajectoryCaptureReader(Protocol):
    def get_blob(self, ref: ArtifactRef) -> bytes: ...

    def read_events(
        self,
        query: TrajectoryQuery | None = None,
    ) -> list[TrajectoryEvent]: ...


@runtime_checkable
class TrajectoryCaptureStore(TrajectoryCaptureReader, Protocol):
    def append(self, event: TrajectoryEvent) -> TrajectoryEvent: ...

    def put_blob(
        self,
        content: bytes,
        *,
        type_: str,
        created_by: str,
        provenance: ProvenanceRef | None = None,
        contamination_flags: list[str] | None = None,
    ) -> ArtifactRef: ...
