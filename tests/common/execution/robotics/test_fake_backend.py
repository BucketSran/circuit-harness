"""Contract tests for the deterministic robotics backend."""

from __future__ import annotations

import pytest

from alphaapollo.common.artifacts.schemas import ArtifactRef
from alphaapollo.common.execution.robotics import (
    RobotAction,
    RobotBackend,
    RobotObservation,
    RobotTask,
    RobotTransition,
)
from alphaapollo.common.execution.robotics.backends import FakeBackend


def _observation(frame: int) -> RobotObservation:
    return RobotObservation(
        state=[float(frame), 0.0, 0.0],
        artifact_refs=(ArtifactRef(id=f"frame-{frame}", location=f"artifact://frame-{frame}"),),
        timestamp=f"2026-08-16T00:00:0{frame}Z",
        backend_metadata={"backend": "fake"},
    )


def _task() -> RobotTask:
    return RobotTask(
        task_id="task-1",
        benchmark="libero",
        instruction="pick up the red cup",
        environment_version="libero-0.1",
    )


def _action(kind: str) -> RobotAction:
    return RobotAction(kind=kind, provenance={"provider": "fake-vla"})


def test_fake_backend_replays_recovery_then_authoritative_success() -> None:
    backend = FakeBackend(
        initial_observation=_observation(0),
        transitions=(
            RobotTransition(
                observation=_observation(1),
                steps_used=2,
                terminated=False,
                truncated=False,
                success=None,
                info={"recoverable": True},
            ),
            RobotTransition(
                observation=_observation(2),
                steps_used=3,
                terminated=True,
                truncated=False,
                success=True,
                termination_reason="task_success",
            ),
        ),
        reset_info={"seed": 7},
    )

    assert isinstance(backend, RobotBackend)
    reset = backend.reset(_task())
    first = backend.execute(_action("reach"))

    assert reset.info["seed"] == 7
    assert first.done is False
    assert backend.observe().state == (1.0, 0.0, 0.0)
    assert backend.steps_used == 2
    assert backend.success is None

    second = backend.execute(_action("grasp"))

    assert second.success is True
    assert backend.terminated is True
    assert backend.truncated is False
    assert backend.success is True
    assert backend.termination_reason == "task_success"
    assert backend.steps_used == 5
    assert [action.kind for action in backend.actions] == ["reach", "grasp"]


def test_fake_backend_reset_restarts_script_and_clears_episode_state() -> None:
    backend = FakeBackend(
        initial_observation=_observation(0),
        transitions=(
            RobotTransition(
                observation=_observation(1),
                steps_used=4,
                terminated=False,
                truncated=True,
                success=False,
                termination_reason="time_limit",
            ),
        ),
    )
    backend.reset(_task())
    backend.execute(_action("wait"))

    backend.reset(_task())

    assert backend.observe() == _observation(0)
    assert backend.actions == ()
    assert backend.steps_used == 0
    assert backend.terminated is False
    assert backend.truncated is False
    assert backend.success is None
    assert backend.termination_reason is None


def test_fake_backend_rejects_invalid_lifecycle_and_unreachable_script() -> None:
    terminal = RobotTransition(
        observation=_observation(1),
        steps_used=1,
        terminated=True,
        truncated=False,
        success=False,
        termination_reason="task_failure",
    )
    nonterminal = RobotTransition(
        observation=_observation(2),
        steps_used=1,
        terminated=False,
        truncated=False,
        success=None,
    )

    with pytest.raises(ValueError, match="unreachable"):
        FakeBackend(
            initial_observation=_observation(0),
            transitions=(terminal, nonterminal),
        )

    backend = FakeBackend(initial_observation=_observation(0), transitions=(terminal,))
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()
    backend.reset(_task())
    backend.execute(_action("stop"))
    with pytest.raises(RuntimeError, match="terminal"):
        backend.execute(_action("again"))
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.reset(_task())


def test_fake_backend_fails_loudly_when_script_is_exhausted() -> None:
    backend = FakeBackend(initial_observation=_observation(0))
    backend.reset(_task())

    with pytest.raises(RuntimeError, match="script is exhausted"):
        backend.execute(_action("move"))
    assert backend.actions == ()
