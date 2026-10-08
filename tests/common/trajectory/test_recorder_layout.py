"""Regression coverage for the stable recorder facade."""

import importlib

from alphaapollo.common import trajectory
from alphaapollo.common.trajectory import recorder
from alphaapollo.common.trajectory._recorder.contracts import EnvironmentCapture
from alphaapollo.common.trajectory._recorder.episode import EpisodeRecorder
from alphaapollo.common.trajectory._recorder.replay import EnvironmentTrajectoryReader
from alphaapollo.common.trajectory._recorder.sinks import TrajectoryEnvironmentEventSink


def test_recorder_facade_reexports_canonical_implementations() -> None:
    assert recorder.EpisodeRecorder is EpisodeRecorder
    assert recorder.EnvironmentCapture is EnvironmentCapture
    assert recorder.EnvironmentTrajectoryReader is EnvironmentTrajectoryReader
    assert recorder.TrajectoryEnvironmentEventSink is TrajectoryEnvironmentEventSink


def test_package_reexports_every_recorder_name() -> None:
    """The package table and the facade must agree in both directions.

    `build_capture_envelope` was public on the facade and missing from the
    package table, so one of the two documented import paths raised
    `ImportError` for a name the other exported.
    """

    missing = sorted(set(recorder.__all__) - set(trajectory.__all__))
    assert missing == []
    for name in trajectory.__all__:
        owner = importlib.import_module(trajectory._EXPORTS[name])
        assert getattr(trajectory, name) is getattr(owner, name)
