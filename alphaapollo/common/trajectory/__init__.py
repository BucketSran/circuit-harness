"""Episode capture, read-back, and persistence."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "EpisodeRecorder": "alphaapollo.common.trajectory.recorder",
    "EpisodeResult": "alphaapollo.common.trajectory.episode",
    "EpisodeTurn": "alphaapollo.common.trajectory.episode",
    "EnvironmentCapture": "alphaapollo.common.trajectory.recorder",
    "EnvironmentCaptureKind": "alphaapollo.common.trajectory.recorder",
    "EnvironmentCaptureRecord": "alphaapollo.common.trajectory.recorder",
    "EnvironmentEventKind": "alphaapollo.common.trajectory.recorder",
    "EnvironmentEventSink": "alphaapollo.common.trajectory.recorder",
    "EnvironmentReplayStep": "alphaapollo.common.trajectory.recorder",
    "EnvironmentTrajectoryError": "alphaapollo.common.trajectory.recorder",
    "EnvironmentTrajectoryReader": "alphaapollo.common.trajectory.recorder",
    "EnvironmentTrajectoryView": "alphaapollo.common.trajectory.recorder",
    "NullEnvironmentEventSink": "alphaapollo.common.trajectory.recorder",
    "RecordedEnvironmentEvent": "alphaapollo.common.trajectory.recorder",
    "RecordingEnvironmentEventSink": "alphaapollo.common.trajectory.recorder",
    "SensitiveEnvironmentInputError": "alphaapollo.common.trajectory.recorder",
    "TRAJECTORY_EVENT_SCHEMA_VERSION": "alphaapollo.common.trajectory.schemas",
    "TRAJECTORY_SCHEMA_VERSION": "alphaapollo.common.trajectory.schemas",
    "TrajectoryCaptureReader": "alphaapollo.common.trajectory.recorder",
    "TrajectoryCaptureStore": "alphaapollo.common.trajectory.recorder",
    "TrajectoryEnvironmentEventSink": "alphaapollo.common.trajectory.recorder",
    "TrajectoryEvent": "alphaapollo.common.trajectory.schemas",
    "TrajectoryEventType": "alphaapollo.common.trajectory.schemas",
    "TrajectoryQuery": "alphaapollo.common.trajectory.episode",
    "TrajectoryRef": "alphaapollo.common.trajectory.schemas",
    "TrajectoryReader": "alphaapollo.common.trajectory.episode",
    "TrajectorySink": "alphaapollo.common.trajectory.episode",
    "TrajectoryStore": "alphaapollo.common.trajectory.store",
    "TrajectoryStoreError": "alphaapollo.common.trajectory.store",
    "build_capture_envelope": "alphaapollo.common.trajectory.recorder",
    "ensure_safe_environment_input": "alphaapollo.common.trajectory.recorder",
    "sanitize_capture_value": "alphaapollo.common.trajectory.recorder",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
