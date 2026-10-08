# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Stable imports for episode recording and environment capture read-back."""

from alphaapollo.common.trajectory._recorder.contracts import (
    EnvironmentCapture,
    EnvironmentCaptureKind,
    EnvironmentEventKind,
    EnvironmentEventSink,
    TrajectoryCaptureReader,
    TrajectoryCaptureStore,
)
from alphaapollo.common.trajectory._recorder.episode import EpisodeRecorder
from alphaapollo.common.trajectory._recorder.replay import (
    EnvironmentCaptureRecord,
    EnvironmentReplayStep,
    EnvironmentTrajectoryError,
    EnvironmentTrajectoryReader,
    EnvironmentTrajectoryView,
)
from alphaapollo.common.trajectory._recorder.sanitize import (
    SensitiveEnvironmentInputError,
    build_capture_envelope,
    ensure_safe_environment_input,
    sanitize_capture_value,
)
from alphaapollo.common.trajectory._recorder.sinks import (
    NullEnvironmentEventSink,
    RecordedEnvironmentEvent,
    RecordingEnvironmentEventSink,
    TrajectoryEnvironmentEventSink,
)

__all__ = [
    "EpisodeRecorder",
    "EnvironmentCapture",
    "EnvironmentCaptureKind",
    "EnvironmentCaptureRecord",
    "EnvironmentEventKind",
    "EnvironmentEventSink",
    "EnvironmentReplayStep",
    "EnvironmentTrajectoryError",
    "EnvironmentTrajectoryReader",
    "EnvironmentTrajectoryView",
    "NullEnvironmentEventSink",
    "RecordedEnvironmentEvent",
    "RecordingEnvironmentEventSink",
    "SensitiveEnvironmentInputError",
    "TrajectoryCaptureReader",
    "TrajectoryCaptureStore",
    "TrajectoryEnvironmentEventSink",
    "build_capture_envelope",
    "ensure_safe_environment_input",
    "sanitize_capture_value",
]
