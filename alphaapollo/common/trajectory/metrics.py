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

"""Rebuild cost-shaped evaluation metrics from durable trajectory evidence.

``traj.jsonl`` is the append-only execution ledger; ``result.json`` is a
materialized view. Every cache/report read validates and re-reduces the digest-bound
ledger. Reducer-version changes and damaged materialized metrics can therefore be
repaired without another model invocation, provided every event was written under
the current evidence contract.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alphaapollo.common.trajectory.schemas import (
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryEventType,
)

# This reducer reads the serialized, versioned tool id. Importing the Common
# execution aggregate would allocate platform-specific sandbox dependencies at
# evaluation import time; the on-disk value is deliberately the stable string.
INTERNAL_PYTHON_TOOL_ID = "python"

__all__ = [
    "METRICS_EVIDENCE_COMPLETE",
    "METRICS_EVIDENCE_PARTIAL",
    "METRICS_EVIDENCE_UNAVAILABLE",
    "METRICS_REDUCER_VERSION",
    "TrajectoryEvidenceError",
    "TrajectoryMetrics",
    "materialize_trajectory_metrics",
    "reduce_trajectory",
    "reduce_trajectory_evidence",
    "refresh_trajectory_metrics",
    "trajectory_digest",
]


METRICS_REDUCER_VERSION = 2
# How much of a row's real cost the ledger can prove. ``complete`` means every event
# the run produced is present; ``partial`` means the counts are real but are a lower
# bound; ``unavailable`` means the ledger cannot be reduced at all. These live here,
# not in the batch runner, because the reducer is what decides which one a given
# trajectory can support.
METRICS_EVIDENCE_COMPLETE = "complete"
METRICS_EVIDENCE_PARTIAL = "partial"
METRICS_EVIDENCE_UNAVAILABLE = "unavailable"
_REJECTED_TOOL_ID = "__rejected_tool_call__"
_REJECTED_CANONICALIZATION_STATUS = "rejected_before_resolution"


class TrajectoryEvidenceError(ValueError):
    """The ledger cannot safely support the current metrics reducer."""


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
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


def trajectory_digest(path: Path) -> str:
    """Content identity used to bind a materialized result to its source ledger."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise TrajectoryEvidenceError(f"cannot read trajectory {path}: {exc}") from exc
    return hashlib.sha256(content).hexdigest()


def reduce_trajectory(path: Path) -> TrajectoryMetrics:
    """Reduce one current-evidence JSONL ledger into cost metrics.

    Callers that publish a metrics_evidence_status must use
    :func:`reduce_trajectory_evidence` instead: the counts alone cannot say whether
    the ledger is whole.
    """

    return reduce_trajectory_evidence(path)[0]


def reduce_trajectory_evidence(path: Path) -> tuple[TrajectoryMetrics, tuple[str, ...]]:
    """Reduce a ledger into cost metrics plus any producer-reported evidence gaps.

    Repairs are deliberately scoped to one runtime invocation and
    ``(branch, role, round, tool)`` stream. Thus a Python success in a later graph
    replay does not retroactively repair an earlier invocation, even when main's
    execution semantics reuse the same branch id across independent retry policies.

    The second element is non-empty when the Environment told us it failed to record
    something. ``DefaultEnvironment._emit`` deliberately never lets a trajectory write
    failure abort a live run -- the model work is real and must not be thrown away --
    so it swallows the error and reports it on the CLOSED event instead. That makes
    the surviving counts a lower bound, not a complete measurement, and the reducer is
    the only place that can tell the two apart. Ignoring the signal would republish a
    short ledger as authoritative, which is the exact under-count this module exists
    to prevent.
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TrajectoryEvidenceError(f"cannot read trajectory {path}: {exc}") from exc

    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise TrajectoryEvidenceError(
                f"malformed trajectory JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise TrajectoryEvidenceError(f"trajectory line {line_number} must be a JSON object")
        version = event.get("trajectory_schema_version")
        if not _exact_version(version, TRAJECTORY_SCHEMA_VERSION):
            raise TrajectoryEvidenceError(
                f"trajectory line {line_number} has evidence schema {version!r}; "
                f"expected {TRAJECTORY_SCHEMA_VERSION}"
            )
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in seen_ids:
            raise TrajectoryEvidenceError(
                f"trajectory line {line_number} has a missing or duplicate event_id"
            )
        seen_ids.add(event_id)
        events.append(event)

    if not events:
        raise TrajectoryEvidenceError("trajectory contains no events")

    counts = defaultdict(int)
    gaps: list[str] = []
    invocation_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    interactions: dict[tuple[str, str, int, int, str], list[bool]] = defaultdict(list)
    for event in events:
        event_type = event.get("type")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise TrajectoryEvidenceError(f"event {event['event_id']} has a non-object payload")

        # Producers may discover an evidence gap at any event boundary.  In
        # particular, a completed model interaction with omitted/invalid provider
        # usage is still a valid lower-bound event, but it cannot support a
        # ``complete`` token-cost claim.
        gaps.extend(_evidence_errors(payload, event["event_id"]))

        if (
            event_type == TrajectoryEventType.EVIDENCE_ADDED.value
            and payload.get("capture_kind") == "model_interaction"
        ):
            counts["prompt_tokens"] += _non_negative_int(
                payload, "prompt_tokens", event["event_id"]
            )
            counts["completion_tokens"] += _non_negative_int(
                payload, "completion_tokens", event["event_id"]
            )
            # Graph main semantics may reuse ``node-N~1`` when separate verifier
            # policies each perform their first retry. Turn zero is the durable start
            # of an AgentResult invocation, so track an observation-only epoch without
            # changing branch ids, seeds, prompts, or the calls made to either model.
            model_role = payload.get("role") or payload.get("actor")
            model_branch = event.get("branch_id")
            model_round = payload.get("round", payload.get("round_index"))
            if (
                model_role in {"solver", "verifier"}
                and isinstance(model_branch, str)
                and model_branch
                and isinstance(model_round, int)
                and not isinstance(model_round, bool)
                and model_round >= 0
                and payload.get("turn") == 0
            ):
                invocation_counts[(model_branch, model_role, model_round)] += 1

        if event_type not in {
            TrajectoryEventType.TOOL_CALLED.value,
            TrajectoryEventType.TOOL_FAILED.value,
        }:
            continue
        role = payload.get("role") or payload.get("actor")
        if role not in {"solver", "verifier"}:
            raise TrajectoryEvidenceError(
                f"tool event {event['event_id']} has unsupported role {role!r}"
            )
        branch_id = event.get("branch_id")
        if not isinstance(branch_id, str) or not branch_id:
            raise TrajectoryEvidenceError(f"tool event {event['event_id']} has no branch_id")
        failed = event_type == TrajectoryEventType.TOOL_FAILED.value
        request = payload.get("request")
        request = request if isinstance(request, dict) else {}
        tool_id = payload.get("tool_id") or request.get("tool_id")
        if not isinstance(tool_id, str) or not tool_id:
            # A malformed special-token call is still an attempted generic tool
            # failure, but ingress could not mint a canonical tool id. Runtime
            # accounting deliberately excludes it from Python-specific failures, so
            # preserve the call/failure with a private reducer key instead of either
            # rejecting the whole ledger or misclassifying it as Python execution.
            error_stage = payload.get("stage") or request.get("stage")
            error_code = payload.get("code") or request.get("code")
            attempted = payload.get("attempted", request.get("attempted"))
            canonicalization_status = payload.get(
                "canonicalization_status", request.get("canonicalization_status")
            )
            rejected_parse_call = (
                failed
                and error_stage == "parse"
                and isinstance(error_code, str)
                and bool(error_code)
                and attempted is False
                and canonicalization_status == _REJECTED_CANONICALIZATION_STATUS
            )
            if not rejected_parse_call:
                raise TrajectoryEvidenceError(
                    f"tool event {event['event_id']} has no canonical tool_id"
                )
            tool_id = _REJECTED_TOOL_ID
        round_index = payload.get("round", payload.get("round_index"))
        if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 0:
            raise TrajectoryEvidenceError(
                f"tool event {event['event_id']} has invalid round {round_index!r}"
            )

        counts[f"{role}_tool_calls"] += 1
        if failed:
            counts[f"{role}_tool_failures"] += 1
            if tool_id == INTERNAL_PYTHON_TOOL_ID:
                counts[f"{role}_python_tool_failures"] += 1
        invocation = invocation_counts[(branch_id, role, round_index)]
        interactions[(branch_id, role, round_index, invocation, tool_id)].append(not failed)

    for (_branch_id, role, _round_index, _invocation, tool_id), outcomes in interactions.items():
        if tool_id != INTERNAL_PYTHON_TOOL_ID:
            continue
        counts[f"{role}_python_tool_repairs"] += sum(
            1
            for current, following in zip(outcomes, outcomes[1:], strict=False)
            if not current and following
        )

    metrics = TrajectoryMetrics(**{field: counts[field] for field in TrajectoryMetrics.__slots__})
    return metrics, tuple(gaps)


def materialize_trajectory_metrics(
    payload: Mapping[str, Any],
    path: Path,
    *,
    status: str = METRICS_EVIDENCE_COMPLETE,
) -> dict[str, Any]:
    """Return a current materialized view of ``path``, with its provable status.

    ``status`` is what the caller believes the run achieved; the returned view carries
    what the ledger can actually support, which is never stronger.
    """

    metrics, gaps = reduce_trajectory_evidence(path)
    return {
        **payload,
        **asdict(metrics),
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "metrics_reducer_version": METRICS_REDUCER_VERSION,
        "trajectory_digest": trajectory_digest(path),
        **_evidence_status(status, gaps),
    }


def _evidence_status(status: str, gaps: tuple[str, ...]) -> dict[str, Any]:
    """Weaken ``complete`` to ``partial`` when the producer reported a lost event.

    A run that finished cleanly still yields only a lower bound if its Environment
    could not persist part of the ledger, so the gaps are recorded alongside the
    downgrade rather than being dropped on the floor.
    """

    if not gaps:
        return {"metrics_evidence_status": status}
    return {
        "metrics_evidence_status": (
            METRICS_EVIDENCE_PARTIAL if status == METRICS_EVIDENCE_COMPLETE else status
        ),
        "metrics_evidence_gaps": list(gaps),
    }


def _evidence_errors(payload: Mapping[str, Any], event_id: str) -> list[str]:
    errors = payload.get("event_errors")
    if errors is None:
        return []
    if not isinstance(errors, list) or any(not isinstance(item, str) for item in errors):
        raise TrajectoryEvidenceError(f"event {event_id} has a malformed event_errors list")
    return list(errors)


def refresh_trajectory_metrics(
    payload: Mapping[str, Any], path: Path
) -> tuple[dict[str, Any], bool]:
    """Validate and refresh a materialized metric view without running the model.

    A digest mismatch is not silently accepted: it could be a truncated or replaced
    ledger, so the batch runner must recreate the cell. Even a current reducer version
    is recomputed: the version identifies the algorithm, not proof that result.json's
    cost fields still match the digest-bound trajectory. A ``complete`` claim is
    likewise re-derived, so a row can never keep asserting whole evidence over a
    ledger whose producer reported a lost event.
    """

    version = payload.get("trajectory_schema_version")
    if not _exact_version(version, TRAJECTORY_SCHEMA_VERSION):
        raise TrajectoryEvidenceError(
            f"trajectory evidence schema {version!r} is incompatible; "
            f"expected {TRAJECTORY_SCHEMA_VERSION}"
        )
    digest = trajectory_digest(path)
    if payload.get("trajectory_digest") != digest:
        raise TrajectoryEvidenceError("trajectory digest does not match result.json")
    reducer_version = payload.get("metrics_reducer_version")
    current_reducer = _exact_version(reducer_version, METRICS_REDUCER_VERSION)
    if (
        not current_reducer
        and reducer_version is not None
        and (
            not isinstance(reducer_version, int)
            or isinstance(reducer_version, bool)
            or reducer_version < 0
        )
    ):
        raise TrajectoryEvidenceError(f"invalid metrics reducer version {reducer_version!r}")
    if isinstance(reducer_version, int) and reducer_version > METRICS_REDUCER_VERSION:
        raise TrajectoryEvidenceError(
            f"metrics reducer version {reducer_version} is newer than supported "
            f"version {METRICS_REDUCER_VERSION}"
        )
    metrics, gaps = reduce_trajectory_evidence(path)
    reduced = asdict(metrics)
    # Only a ``complete`` claim can be weakened here. ``partial``/``unavailable`` are
    # the caller's own execution verdict (a typed error, an unreducible ledger) and a
    # whole trajectory is not evidence that the run itself finished.
    status: dict[str, Any] = {}
    if gaps:
        status["metrics_evidence_gaps"] = list(gaps)
        if payload.get("metrics_evidence_status") == METRICS_EVIDENCE_COMPLETE:
            status["metrics_evidence_status"] = METRICS_EVIDENCE_PARTIAL
    refreshed = {
        **payload,
        **reduced,
        "metrics_reducer_version": METRICS_REDUCER_VERSION,
        **status,
    }
    changed = (
        not current_reducer
        or any(payload.get(key) != value for key, value in reduced.items())
        or any(payload.get(key) != value for key, value in status.items())
    )
    return refreshed, changed


def _non_negative_int(payload: Mapping[str, Any], field: str, event_id: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TrajectoryEvidenceError(f"model event {event_id} has invalid {field}={value!r}")
    return value


def _exact_version(value: Any, expected: int) -> bool:
    """Reject JSON booleans, which compare equal to integers in Python."""

    return isinstance(value, int) and not isinstance(value, bool) and value == expected
