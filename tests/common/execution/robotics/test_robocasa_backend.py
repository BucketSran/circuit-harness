"""RoboCasa backend contract tests without importing MuJoCo or RoboCasa."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from alphaapollo.common.execution.robotics import (
    RobotAction,
    RobotBackend,
    RobotGeometryBackend,
    RobotTask,
)
from alphaapollo.common.execution.robotics.backends import (
    RoboCasaBackend,
    create_robocasa_environment,
)
from alphaapollo.common.execution.workspace import ArtifactStore


def _task(**metadata: Any) -> RobotTask:
    scene = {
        "task_name": "CoffeeSetup",
        "layout_id": 2,
        "style_id": 5,
        "scene_seed": 42,
        "max_episode_steps": 4,
    }
    scene.update(metadata)
    return RobotTask(
        task_id="coffee-setup-000",
        benchmark="robocasa",
        instruction="set up the coffee machine on the counter",
        environment_version="robocasa365-1.0.1",
        backend_metadata=scene,
    )


def _observation(step: int) -> dict[str, Any]:
    return {
        "robot0_proprio-state": [float(step), 0.0],
        "robot0_agentview_left_image": [[[step, 0, 0]]],
        "robot0_agentview_left_depth": [[[0.75]]],
    }


class _Environment:
    layout_id = 2
    style_id = 5
    horizon = 10
    action_dim = 2

    def __init__(self, *, succeeds_after: int | None = 2) -> None:
        self.succeeds_after = succeeds_after
        self.steps = 0
        self.closed = False

    def reset(self) -> dict[str, Any]:
        self.steps = 0
        return _observation(0)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        assert tuple(action) == (0.1, -0.2)
        self.steps += 1
        return _observation(self.steps), 0.25, False, {"step": self.steps}

    def get_ep_meta(self) -> dict[str, Any]:
        return {"lang": "set up the coffee machine on the counter"}

    def _check_success(self) -> bool:
        return self.succeeds_after is not None and self.steps >= self.succeeds_after

    def close(self) -> None:
        self.closed = True


def _action(values: list[float] | None = None) -> RobotAction:
    return RobotAction(
        kind="continuous",
        arguments={"values": values if values is not None else [0.1, -0.2]},
        provenance={"provider": "fake-vla"},
    )


def test_robocasa_backend_rebuilds_exact_scene_and_uses_native_success(
    tmp_path: Path,
) -> None:
    environment = _Environment()
    tasks: list[RobotTask] = []

    def factory(task: RobotTask) -> _Environment:
        tasks.append(task)
        return environment

    backend = RoboCasaBackend(
        environment_factory=factory,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: f"png:{image}".encode(),
        clock=lambda: "2026-08-16T00:00:00Z",
    )

    reset = backend.reset(_task())
    first = backend.execute(_action())
    second = backend.execute(_action())

    assert isinstance(backend, RobotBackend)
    assert tasks[0].backend_metadata["scene_seed"] == 42
    assert reset.info["scene"] == {
        "task_name": "CoffeeSetup",
        "layout_id": 2,
        "style_id": 5,
        "scene_seed": 42,
    }
    # Robot state nests under state.proprio, matching the LIBERO backend.
    assert reset.observation.state["proprio"]["robot0_proprio-state"] == (0.0, 0.0)
    assert "robot0_agentview_left_depth" not in reset.observation.state["proprio"]
    assert reset.observation.artifact_refs[0].type == "image/png"
    assert first.done is False
    assert first.success is None
    assert second.terminated is True
    assert second.success is True
    assert second.termination_reason == "task_success"
    assert backend.steps_used == 2
    assert backend.success is True


def test_robocasa_back_project_uses_current_metric_depth_and_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _Environment(succeeds_after=None)
    monkeypatch.setattr(
        "alphaapollo.common.execution.robotics.backends.robocasa._camera_metadata",
        lambda *args, **kwargs: {
            "robot0_agentview_left_image": {
                "name": "robot0_agentview_left",
                "height": 1,
                "width": 1,
                "intrinsics": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "extrinsics": [
                    [1.0, 0.0, 0.0, -0.1],
                    [0.0, 1.0, 0.0, 0.2],
                    [0.0, 0.0, 1.0, 0.3],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        },
    )
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        enable_depth=True,
        image_encoder=lambda image: b"png",
        depth_converter=lambda env, depth: depth,
        clock=lambda: "2026-08-18T00:00:00Z",
    )

    backend.reset(_task())
    point = backend.back_project(
        camera="robot0_agentview_left_image",
        pixel_x=0,
        pixel_y=0,
    )

    assert isinstance(backend, RobotGeometryBackend)
    assert point.depth_m == 0.75
    assert point.world_xyz == pytest.approx((-0.1, 0.2, 1.05))


def test_robocasa_depth_failure_keeps_images_and_the_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Depth is opt-in grounding, so its faults must not cost the observation."""

    monkeypatch.setattr(
        "alphaapollo.common.execution.robotics.backends.robocasa._camera_metadata",
        lambda *args, **kwargs: {
            "robot0_agentview_left_image": {
                "name": "robot0_agentview_left",
                "height": 1,
                "width": 1,
                "intrinsics": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "extrinsics": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        },
    )
    converted = 0

    def failing_converter(environment: Any, depth: Any) -> Any:
        nonlocal converted
        converted += 1
        if converted > 1:
            raise RuntimeError("depth conversion failed")
        return depth

    backend = RoboCasaBackend(
        environment_factory=lambda task: _Environment(succeeds_after=None),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        enable_depth=True,
        image_encoder=lambda image: b"png",
        depth_converter=failing_converter,
        clock=lambda: "2026-08-18T00:00:00Z",
    )
    backend.reset(_task())

    transition = backend.execute(_action())

    # The camera frames and the proprio state survive a depth-only fault.
    assert len(transition.observation.artifact_refs) == 1
    assert "observation_warning" not in transition.observation.backend_metadata
    assert transition.observation.backend_metadata["depth_warnings"] == (
        "robot0_agentview_left_image: RuntimeError: depth conversion failed",
    )
    assert transition.observation.state["proprio"]["robot0_proprio-state"] == (1.0, 0.0)
    # Grounding is reported as unavailable rather than answered from stale depth.
    with pytest.raises(RuntimeError, match="no depth for camera"):
        backend.back_project(
            camera="robot0_agentview_left_image",
            pixel_x=0,
            pixel_y=0,
        )


def test_robocasa_backend_truncates_at_prepared_episode_limit(tmp_path: Path) -> None:
    environment = _Environment(succeeds_after=None)
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )
    backend.reset(_task(max_episode_steps=1))

    transition = backend.execute(_action())

    assert transition.terminated is False
    assert transition.truncated is True
    assert transition.success is False
    assert transition.termination_reason == "time_limit"
    assert backend.success is False


def test_robocasa_distinguishes_environment_done_from_time_limit(tmp_path: Path) -> None:
    class _DoneEnvironment(_Environment):
        def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            observation, reward, _done, info = super().step(action)
            return observation, reward, True, info

    environment = _DoneEnvironment(succeeds_after=None)
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )
    backend.reset(_task(max_episode_steps=10))

    transition = backend.execute(_action())

    assert transition.truncated is True
    assert transition.termination_reason == "environment_done"


def test_robocasa_failed_limit_validation_closes_the_new_environment(tmp_path: Path) -> None:
    environment = _Environment()
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )

    with pytest.raises(ValueError, match="positive integer"):
        backend.reset(_task(max_episode_steps=0))

    assert environment.closed is True
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()


@pytest.mark.parametrize("horizon", (500.0,))
def test_robocasa_coerces_non_native_horizon_types(horizon: float, tmp_path: Path) -> None:
    class _FloatHorizon(_Environment):
        pass

    environment = _FloatHorizon()
    environment.horizon = horizon  # type: ignore[assignment]
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )

    backend.reset(_task(max_episode_steps=None))

    assert backend._max_episode_steps == 500


def test_robocasa_image_failure_still_publishes_post_step_state(tmp_path: Path) -> None:
    environment = _Environment(succeeds_after=1)
    calls = 0

    def encoder(image: Any) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("artifact disk full")
        return b"png"

    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=encoder,
    )
    backend.reset(_task())

    transition = backend.execute(_action())

    assert transition.terminated is True
    assert transition.success is True
    assert backend.observe().state["proprio"]["robot0_proprio-state"] == (1.0, 0.0)
    assert transition.observation.backend_metadata["observation_warning"]


def test_robocasa_noncanonical_step_data_is_degraded_without_losing_success(
    tmp_path: Path,
) -> None:
    class _DivergedEnvironment(_Environment):
        def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            observation, _reward, done, _info = super().step(action)
            observation["robot0_proprio-state"] = [float("nan"), 0.0]
            return observation, float("nan"), done, {"bad": object()}

    environment = _DivergedEnvironment(succeeds_after=1)
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )
    backend.reset(_task())

    transition = backend.execute(_action())

    assert transition.terminated is True and transition.success is True
    assert transition.observation is backend.observe()
    assert "robot0_proprio-state" not in transition.observation.state["proprio"]
    assert transition.observation.backend_metadata["dropped_state_fields"] == (
        "robot0_proprio-state",
    )
    assert "reward" not in transition.info
    assert "bad" not in transition.info
    assert len(transition.info["transition_warnings"]) == 2


def test_robocasa_post_step_contract_failure_invalidates_the_episode(tmp_path: Path) -> None:
    class _BrokenOracleEnvironment(_Environment):
        def _check_success(self) -> bool:
            # Answers at reset (where the backend refuses an already-solved
            # scene) and only breaks once the simulator has stepped.
            if self.steps == 0:
                return False
            raise RuntimeError("success oracle disconnected")

    environment = _BrokenOracleEnvironment()
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )
    backend.reset(_task())

    with pytest.raises(RuntimeError, match="oracle disconnected"):
        backend.execute(_action())

    assert environment.closed is True
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()


def test_robocasa_reset_observation_includes_resolved_camera_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = {
        "robot0_agentview_left_image": {
            "name": "robot0_agentview_left",
            "height": 1,
            "width": 1,
            "intrinsics": [[1.0]],
            "extrinsics": [[1.0]],
        }
    }
    monkeypatch.setattr(
        "alphaapollo.common.execution.robotics.backends.robocasa._camera_metadata",
        lambda environment, raw: cameras,
    )
    backend = RoboCasaBackend(
        environment_factory=lambda task: _Environment(),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )

    reset = backend.reset(_task())

    published = reset.observation.backend_metadata["cameras"]
    assert published["robot0_agentview_left_image"]["name"] == "robot0_agentview_left"
    assert published["robot0_agentview_left_image"]["intrinsics"] == ((1.0,),)


def test_robocasa_close_preserves_the_episode_outcome(tmp_path: Path) -> None:
    environment = _Environment(succeeds_after=1)
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )
    backend.reset(_task())
    backend.execute(_action())

    backend.close()

    assert environment.closed is True
    assert backend.terminated is True
    assert backend.success is True
    assert backend.termination_reason == "task_success"
    assert backend.steps_used == 1


@pytest.mark.parametrize("values", ([0.1], [0.1, -0.2, 0.3]))
def test_robocasa_backend_rejects_wrong_action_dimensions(
    tmp_path: Path, values: list[float]
) -> None:
    environment = _Environment()
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )
    backend.reset(_task())

    with pytest.raises(ValueError, match="wrong dimension: expected 2"):
        backend.execute(_action(values))
    assert environment.steps == 0


def test_robocasa_backend_requires_a_native_action_dimension(tmp_path: Path) -> None:
    class _NoActionDim(_Environment):
        action_dim = None

    environment = _NoActionDim()
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )

    with pytest.raises(RuntimeError, match="no usable action_dim"):
        backend.reset(_task())
    assert environment.closed is True


def test_robocasa_backend_rejects_wrong_or_incomplete_prepared_scene() -> None:
    calls = 0

    def factory(task: RobotTask) -> _Environment:
        nonlocal calls
        calls += 1
        return _Environment()

    backend = RoboCasaBackend(environment_factory=factory)
    with pytest.raises(ValueError, match="missing"):
        backend.reset(
            RobotTask(
                task_id="bad",
                benchmark="robocasa",
                instruction="do it",
                environment_version="robocasa-921c9a5",
                backend_metadata={"task_name": "CoffeeSetup"},
            )
        )
    with pytest.raises(ValueError, match="benchmark"):
        backend.reset(
            RobotTask(
                task_id="bad",
                benchmark="libero",
                instruction="do it",
                environment_version="robocasa-921c9a5",
                backend_metadata={
                    "task_name": "CoffeeSetup",
                    "layout_id": 2,
                    "style_id": 5,
                    "scene_seed": 42,
                },
            )
        )
    assert calls == 0


def test_robocasa_failed_rereset_invalidates_the_previous_episode(tmp_path: Path) -> None:
    environment = _Environment()
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        environment_version="robocasa365-1.0.1",
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )
    backend.reset(_task())

    with pytest.raises(ValueError, match="environment_version mismatch"):
        backend.reset(replace(_task(), environment_version="different-version"))

    assert environment.closed is True
    assert backend.success is None
    assert backend.steps_used == 0
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.execute(_action())


def test_robocasa_backend_rejects_reset_scene_mismatch_and_closes_it() -> None:
    environment = _Environment()
    environment.layout_id = 3
    backend = RoboCasaBackend(environment_factory=lambda task: environment)

    with pytest.raises(ValueError, match="layout_id"):
        backend.reset(_task())

    assert environment.closed is True


def test_robocasa_backend_requires_artifact_store_for_camera_observations() -> None:
    environment = _Environment()
    backend = RoboCasaBackend(environment_factory=lambda task: environment)

    with pytest.raises(RuntimeError, match="artifact_store"):
        backend.reset(_task())

    assert environment.closed is True


def test_official_factory_receives_pr256_scene_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    environment = _Environment()
    robocasa = ModuleType("robocasa")
    utils = ModuleType("robocasa.utils")
    env_utils = ModuleType("robocasa.utils.env_utils")

    def create_env(**kwargs: Any) -> _Environment:
        calls.append(kwargs)
        return environment

    env_utils.create_env = create_env  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "robocasa", robocasa)
    monkeypatch.setitem(sys.modules, "robocasa.utils", utils)
    monkeypatch.setitem(sys.modules, "robocasa.utils.env_utils", env_utils)

    actual = create_robocasa_environment(_task())

    assert actual is environment
    assert calls == [
        {
            "env_name": "CoffeeSetup",
            "layout_and_style_ids": [(2, 5)],
            "seed": 42,
            "render_onscreen": False,
        }
    ]


def test_robocasa_backend_rejects_an_environment_version_mismatch() -> None:
    calls = 0

    def factory(task: RobotTask) -> Any:
        nonlocal calls
        calls += 1
        return _Environment()

    backend = RoboCasaBackend(
        environment_factory=factory,
        environment_version="robocasa-921c9a5",
    )
    with pytest.raises(ValueError, match="environment_version mismatch"):
        backend.reset(
            RobotTask(
                task_id="t",
                benchmark="robocasa",
                instruction="set up the coffee machine on the counter",
                environment_version="robocasa-deadbeef",
                backend_metadata={
                    "task_name": "CoffeeSetup",
                    "layout_id": 2,
                    "style_id": 5,
                    "scene_seed": 42,
                },
            )
        )
    assert calls == 0


def test_robocasa_rejects_a_scene_already_successful_at_reset(tmp_path: Path) -> None:
    # The native oracle decides the episode, so a scene that already satisfies
    # it would score a success no policy earned. LIBERO refuses the same state.
    environment = _Environment(succeeds_after=0)
    backend = RoboCasaBackend(
        environment_factory=lambda task: environment,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )

    with pytest.raises(RuntimeError, match="already successful at reset"):
        backend.reset(_task())

    assert environment.closed is True
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()


def test_robocasa_failure_paths_report_the_cause_not_the_close_error(
    tmp_path: Path,
) -> None:
    class _UncloseableEnvironment(_Environment):
        def close(self) -> None:
            super().close()
            raise RuntimeError("mujoco context teardown failed")

    mismatched = _UncloseableEnvironment()
    mismatched.layout_id = 3
    backend = RoboCasaBackend(environment_factory=lambda task: mismatched)

    with pytest.raises(ValueError, match="layout_id"):
        backend.reset(_task())
    assert mismatched.closed is True

    class _ExplodingStep(_UncloseableEnvironment):
        def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            raise RuntimeError("mujoco step exploded")

    stepping = _ExplodingStep()
    stepping_backend = RoboCasaBackend(
        environment_factory=lambda task: stepping,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_encoder=lambda image: b"png",
    )
    stepping_backend.reset(_task())

    with pytest.raises(RuntimeError, match="mujoco step exploded"):
        stepping_backend.execute(_action())
    assert stepping.closed is True
