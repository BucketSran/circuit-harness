"""Tests for backend-neutral robot execution records."""

from __future__ import annotations

import pytest

from alphaapollo.common.execution.robotics import (
    RobotAction,
    RobotGroundedPoint,
    RobotObservation,
    RobotResetResult,
    RobotTask,
    RobotTransition,
)
from alphaapollo.common.execution.robotics.geometry import ground_metric_depth


def _observation() -> RobotObservation:
    return RobotObservation(
        state={"joints": [0.0, 0.1]},
        artifact_refs=({"id": "frame-1", "location": "artifact://frame/1"},),
        timestamp="2026-08-16T00:00:00Z",
        backend_metadata={"backend": "fake"},
    )


def test_task_and_reset_result_are_json_safe_and_copied() -> None:
    metadata = {"suite": {"name": "libero-spatial"}}
    task = RobotTask(
        task_id="task-1",
        benchmark="libero",
        instruction="pick up the cup",
        environment_version="libero-0.1",
        backend_metadata=metadata,
    )
    metadata["suite"]["name"] = "mutated"
    reset = RobotResetResult(observation=_observation(), info={"seed": 7})

    assert task.to_dict()["backend_metadata"]["suite"]["name"] == "libero-spatial"
    assert reset.to_dict()["observation"]["artifact_refs"][0]["id"] == "frame-1"


def test_observation_is_json_only_and_copied_at_the_boundary() -> None:
    state = {"camera": {"frame": 1}}
    observation = RobotObservation(
        state=state,
        timestamp="2026-08-16T00:00:00Z",
    )
    state["camera"]["frame"] = 2

    assert observation.to_dict()["state"]["camera"]["frame"] == 1
    with pytest.raises(TypeError):
        observation.state["camera"] = {"frame": 3}
    with pytest.raises(ValueError, match="non-JSON"):
        RobotObservation(
            state={"camera": b"raw image bytes"},
            timestamp="2026-08-16T00:00:00Z",
        )


def test_grounded_point_is_validated_and_json_auditable() -> None:
    point = RobotGroundedPoint(
        camera="fixed_camera",
        pixel_x=1,
        pixel_y=2,
        image_width=4,
        image_height=5,
        depth_m=0.75,
        world_xyz=(0.1, -0.2, 0.75),
        steps_used=3,
        observation_timestamp="2026-08-18T00:00:00Z",
    )

    assert point.to_dict() == {
        "camera": "fixed_camera",
        "pixel": {"x": 1, "y": 2},
        "image_size": {"width": 4, "height": 5},
        "depth_m": 0.75,
        "world_xyz": [0.1, -0.2, 0.75],
        "frame": "world",
        "steps_used": 3,
        "observation_timestamp": "2026-08-18T00:00:00Z",
    }
    with pytest.raises(ValueError, match="inside"):
        RobotGroundedPoint(
            camera="fixed_camera",
            pixel_x=4,
            pixel_y=0,
            image_width=4,
            image_height=5,
            depth_m=0.75,
            world_xyz=(0.1, -0.2, 0.75),
            steps_used=0,
        )


def test_metric_depth_back_projection_uses_intrinsics_and_camera_pose() -> None:
    point = ground_metric_depth(
        camera="fixed_camera",
        pixel_x=3,
        pixel_y=2,
        image_width=8,
        image_height=6,
        depth_m=2.0,
        intrinsics=((2.0, 0.0, 1.0), (0.0, 2.0, 1.0), (0.0, 0.0, 1.0)),
        camera_to_world=(
            (1.0, 0.0, 0.0, 0.5),
            (0.0, 1.0, 0.0, -0.5),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        steps_used=0,
        observation_timestamp="now",
    )

    assert point.world_xyz == pytest.approx((2.5, 0.5, 3.0))


def test_action_is_typed_and_preserves_provenance() -> None:
    action = RobotAction(
        kind="joint_delta",
        arguments={"values": [0.1, 0.0]},
        provenance={"provider": "fake-vla", "model": "fake-v1"},
    )

    assert action.to_dict() == {
        "kind": "joint_delta",
        "arguments": {"values": [0.1, 0.0]},
        "provenance": {"provider": "fake-vla", "model": "fake-v1"},
    }


def test_transition_requires_backend_success_to_be_terminal() -> None:
    observation = _observation()

    with pytest.raises(ValueError, match="success requires"):
        RobotTransition(
            observation=observation,
            steps_used=1,
            terminated=False,
            truncated=False,
            success=True,
        )
    with pytest.raises(ValueError, match="termination_reason"):
        RobotTransition(
            observation=observation,
            steps_used=1,
            terminated=True,
            truncated=False,
            success=False,
        )


def test_nonterminal_transition_has_no_invented_success() -> None:
    transition = RobotTransition(
        observation=_observation(),
        steps_used=2,
        terminated=False,
        truncated=False,
        success=None,
        info={"backend": "fake"},
    )

    assert transition.done is False
    assert transition.success is None
    assert transition.to_dict()["info"] == {"backend": "fake"}

    # success is three-valued: any definitive value mid-episode is invented.
    with pytest.raises(ValueError, match="terminal"):
        RobotTransition(
            observation=_observation(),
            steps_used=2,
            terminated=False,
            truncated=False,
            success=False,
        )


def test_truncation_is_terminal_without_fabricating_success() -> None:
    transition = RobotTransition(
        observation=_observation(),
        steps_used=0,
        terminated=False,
        truncated=True,
        success=False,
        termination_reason="time_limit",
    )

    assert transition.done is True
    assert transition.success is False
