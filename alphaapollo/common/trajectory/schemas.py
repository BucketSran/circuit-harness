"""Persistent trajectory records and compatibility versions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from alphaapollo.common.artifacts.schemas import (
    ArtifactRef,
    ProvenanceRef,
    Ref,
    require_json_value,
)

# 1.1 dropped the unread ``TrajectoryEvent.cost_delta`` field. A minor bump is
# enough because the event version is validated by major only
# (``require_supported_trajectory_event_schema``) and ``TrajectoryEvent`` allows
# extras, so a 1.0 event still validates and keeps its ``cost_delta`` in
# ``model_extra``. Raise the major instead whenever a change would stop an
# earlier event from loading.
TRAJECTORY_EVENT_SCHEMA_VERSION = "1.1"
SUPPORTED_TRAJECTORY_EVENT_SCHEMA_MAJOR = 1

# Validated for exact equality by ``store.py`` and ``metrics.py``, so every bump
# refuses every trajectory file already on disk. Removing an event field does not
# change how a trajectory file is framed and must not bump this.
TRAJECTORY_SCHEMA_VERSION = 1


def trajectory_event_schema_major(version: str) -> int:
    try:
        return int(version.split(".", maxsplit=1)[0])
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid trajectory event schema version {version!r}") from exc


def require_supported_trajectory_event_schema(version: str) -> None:
    major = trajectory_event_schema_major(version)
    if major != SUPPORTED_TRAJECTORY_EVENT_SCHEMA_MAJOR:
        raise ValueError(
            "unsupported trajectory event schema major "
            f"{major}; reader supports {SUPPORTED_TRAJECTORY_EVENT_SCHEMA_MAJOR}"
        )


class TrajectoryEventType(str, Enum):
    PROBLEM_UNDERSTOOD = "ProblemUnderstood"
    PROBLEM_DECOMPOSED = "ProblemDecomposed"
    HYPOTHESIS_CREATED = "HypothesisCreated"
    EVIDENCE_ADDED = "EvidenceAdded"
    CLAIM_VERIFIED = "ClaimVerified"
    TOOL_CALLED = "ToolCalled"
    TOOL_FAILED = "ToolFailed"
    BRANCH_FORKED = "BranchForked"
    BRANCH_PRUNED = "BranchPruned"
    VERIFIER_PASSED = "VerifierPassed"
    VERIFIER_FAILED = "VerifierFailed"
    ASSUMPTION_CHANGED = "AssumptionChanged"
    SKILL_APPLIED = "SkillApplied"
    ARTIFACT_GENERATED = "ArtifactGenerated"
    HUMAN_INTERVENED = "HumanIntervened"
    SOLUTION_PACKAGED = "SolutionPackaged"


class TrajectoryRef(Ref):
    pass


class TrajectoryEvent(BaseModel):
    """Append-only audit envelope compatible with historical v3 records."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    type: TrajectoryEventType
    event_id: str
    session_id: str
    branch_id: str | None = None
    timestamp: AwareDatetime
    payload: dict[str, Any] = Field(default_factory=dict)
    provenance: ProvenanceRef | None = None
    schema_version: str = TRAJECTORY_EVENT_SCHEMA_VERSION
    trajectory_schema_version: int | None = Field(default=None, ge=1)
    sequence: int | None = Field(default=None, ge=0)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _persistent_values_are_json_exact(self) -> TrajectoryEvent:
        require_json_value(self.payload, path="$.payload")
        require_json_value(self.model_extra or {}, path="$.extra")
        return self


SCHEMA_MODELS: dict[str, type[BaseModel]] = {"trajectory_event": TrajectoryEvent}

__all__ = [
    "SCHEMA_MODELS",
    "SUPPORTED_TRAJECTORY_EVENT_SCHEMA_MAJOR",
    "TRAJECTORY_EVENT_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "TrajectoryEvent",
    "TrajectoryEventType",
    "TrajectoryRef",
    "require_supported_trajectory_event_schema",
    "trajectory_event_schema_major",
]
