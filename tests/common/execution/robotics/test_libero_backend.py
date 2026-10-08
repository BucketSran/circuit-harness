"""In-process LIBERO backend tests against the #260 prepared-payload contract."""

from __future__ import annotations

import sys
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
from alphaapollo.common.execution.robotics.backends import LiberoBackend
from alphaapollo.common.execution.robotics.backends import libero as libero_backend
from alphaapollo.common.execution.workspace import ArtifactStore
from alphaapollo.data_preprocess.robotics import prepare_libero, prepare_liberopro

_INSTRUCTION = "pick up the black bowl and place it on the plate"
_TASK_NAME = "pick_up_the_black_bowl_and_place_it_on_the_plate"
_ACTION = (0.1, -0.2, 0.0, 0.0, 0.0, 0.0, -1.0)


def _task(**metadata: Any) -> RobotTask:
    values = {
        "suite": "libero_spatial",
        "task_order_index": 0,
        "task_index": 3,
        "task_name": _TASK_NAME,
        "problem_folder": "libero_spatial",
        "bddl_file": f"{_TASK_NAME}.bddl",
        "init_states_file": f"{_TASK_NAME}.pruned_init",
        "demonstration_id": "demo_0",
        "demonstration_path": f"libero_spatial/{_TASK_NAME}_demo.hdf5",
        "initial_state_index": 1,
        "max_episode_steps": 3,
    }
    values.update(metadata)
    return RobotTask(
        task_id="libero-spatial-3-1",
        benchmark="libero",
        instruction=_INSTRUCTION,
        environment_version="libero-v1.0.1",
        backend_metadata=values,
    )


class _SuiteTask:
    name = _TASK_NAME
    language = _INSTRUCTION
    problem_folder = "libero_spatial"
    bddl_file = f"{_TASK_NAME}.bddl"
    init_states_file = f"{_TASK_NAME}.pruned_init"


class _Suite:
    def __init__(self) -> None:
        self.requested: list[int] = []
        self.task = _SuiteTask()

    def get_task(self, index: int) -> _SuiteTask:
        self.requested.append(index)
        return self.task

    def get_task_init_states(self, index: int) -> list[list[float]]:
        return [[0.0, 0.0], [1.0, 2.0]]


class _Env:
    def __init__(self, *, succeeds_after: int | None = 2) -> None:
        self.succeeds_after = succeeds_after
        self.steps = 0
        self.closed = False
        self.init_state: Any = None
        self.actions: list[list[float]] = []

    def _obs(self) -> dict[str, Any]:
        return {
            "agentview_image": [[[self.steps, 0, 0]]],
            "agentview_depth": [[[0.5]]],
            "robot0_eye_in_hand_image": None,
            "robot0_joint_pos": [0.1] * 7,
            "robot0_eef_pos": [0.0, 0.1, 0.2],
        }

    def reset(self) -> dict[str, Any]:
        self.steps = 0
        return self._obs()

    def set_init_state(self, state: Any) -> dict[str, Any]:
        self.init_state = state
        return self._obs()

    def step(self, action: list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        assert len(action) == 7
        self.actions.append(list(action))
        self.steps += 1
        succeeded = self.succeeds_after is not None and self.steps >= self.succeeds_after
        return self._obs(), 1.0 if succeeded else 0.0, succeeded, {}

    def check_success(self) -> bool:
        return self.succeeds_after is not None and self.steps >= self.succeeds_after

    def close(self) -> None:
        self.closed = True


def _action(values: Any = _ACTION) -> RobotAction:
    return RobotAction(
        kind="continuous",
        arguments={"values": list(values)},
        provenance={"provider": "fake-vla"},
    )


def _backend(
    env: _Env,
    artifact_dir: Path | None,
    *,
    suite: _Suite | None = None,
    settle_steps: int | None = 0,
    **kwargs: Any,
) -> LiberoBackend:
    store = None if artifact_dir is None else ArtifactStore(artifact_dir)
    if settle_steps is not None:
        kwargs.setdefault("settle_steps", settle_steps)
    return LiberoBackend(
        artifact_store=store,
        image_orientation="raw",
        suite_loader=lambda name, order: suite or _Suite(),
        env_factory=lambda folder, bddl, **options: env,
        image_encoder=lambda image: f"png:{image}".encode(),
        clock=lambda: "2026-08-16T00:00:00Z",
        **kwargs,
    )


def test_libero_backend_uses_simulator_success_signal_and_publishes_artifacts(
    tmp_path: Path,
) -> None:
    env = _Env()
    suite = _Suite()
    backend = _backend(env, tmp_path / "artifacts", suite=suite)

    reset = backend.reset(_task())
    first = backend.execute(_action())
    second = backend.execute(_action())

    assert isinstance(backend, RobotBackend)
    assert suite.requested == [3]
    assert env.init_state == [1.0, 2.0]
    assert reset.info["environment"]["task_index"] == 3
    assert reset.info["initial_states_available"] == 2
    assert reset.observation.state["proprio"]["robot0_eef_pos"] == (0.0, 0.1, 0.2)
    assert reset.observation.artifact_refs[0].type == "image/png"
    assert reset.observation.backend_metadata["artifact_roles"]["fixed_camera"]
    assert first.done is False
    assert first.success is None
    assert second.terminated is True
    assert second.success is True
    assert second.termination_reason == "task_success"
    assert backend.steps_used == 2


def test_libero_back_project_uses_current_metric_depth_and_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _Env(succeeds_after=None)
    factory_options: list[dict[str, Any]] = []

    def factory(folder: str, bddl: str, **options: Any) -> _Env:
        factory_options.append(options)
        return env

    monkeypatch.setattr(
        "alphaapollo.common.execution.robotics.backends.libero._camera_metadata",
        lambda *args, **kwargs: {
            "fixed_camera": {
                "name": "agentview",
                "height": 1,
                "width": 1,
                "intrinsics": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "extrinsics": [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.2],
                    [0.0, 0.0, 1.0, 0.3],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        },
    )
    backend = LiberoBackend(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_orientation="raw",
        enable_depth=True,
        settle_steps=0,
        suite_loader=lambda name, order: _Suite(),
        env_factory=factory,
        image_encoder=lambda image: b"png",
        depth_converter=lambda environment, depth: depth,
        clock=lambda: "2026-08-18T00:00:00Z",
    )

    backend.reset(_task())
    point = backend.back_project(camera="fixed_camera", pixel_x=0, pixel_y=0)

    assert isinstance(backend, RobotGeometryBackend)
    assert factory_options == [{"camera_height": 256, "camera_width": 256, "camera_depths": True}]
    assert point.depth_m == 0.5
    assert point.world_xyz == pytest.approx((0.1, 0.2, 0.8))
    assert point.observation_timestamp == "2026-08-18T00:00:00Z"


# One 8x8 camera whose calibration is a plain pinhole at the image centre, so a
# hand-derived world point pins the pixel conventions on its own.
_GEOMETRY_SIZE = 8
_GEOMETRY_FOCAL = 8.0
_GEOMETRY_CENTRE = 4.0
# Ground truth: the object sits at upright pixel (6, 1), two metres along the
# optical axis.  With f=8 and the principal point at (4, 4) that is
# ((6-4)/8*2, (1-4)/8*2, 2) = (0.5, -0.75, 2.0) in the camera frame, which the
# identity extrinsic leaves unchanged in the world frame.
_GEOMETRY_UPRIGHT_PIXEL = (6, 1)
_GEOMETRY_DEPTH = 2.0
_GEOMETRY_WORLD = (0.5, -0.75, 2.0)
# MuJoCo renders bottom-up, so the depth sample lives in the mirrored row.
_GEOMETRY_RAW_PIXEL = (6, _GEOMETRY_SIZE - 1 - _GEOMETRY_UPRIGHT_PIXEL[1])
_GEOMETRY_CAMERA = {
    "fixed_camera": {
        "name": "agentview",
        "height": _GEOMETRY_SIZE,
        "width": _GEOMETRY_SIZE,
        "intrinsics": [
            [_GEOMETRY_FOCAL, 0.0, _GEOMETRY_CENTRE],
            [0.0, _GEOMETRY_FOCAL, _GEOMETRY_CENTRE],
            [0.0, 0.0, 1.0],
        ],
        "extrinsics": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
}


class _GeometryEnv(_Env):
    """Render an 8x8 frame whose depth buffer marks exactly one raw pixel."""

    def _obs(self) -> dict[str, Any]:
        size = _GEOMETRY_SIZE
        raw_x, raw_y = _GEOMETRY_RAW_PIXEL
        return {
            "agentview_image": [[[0, 0, 0] for _ in range(size)] for _ in range(size)],
            "agentview_depth": [
                [
                    _GEOMETRY_DEPTH if (row, column) == (raw_y, raw_x) else 9.0
                    for column in range(size)
                ]
                for row in range(size)
            ],
            "robot0_eye_in_hand_image": None,
            "robot0_joint_pos": [0.1] * 7,
        }


def _geometry_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, image_orientation: str
) -> LiberoBackend:
    monkeypatch.setattr(
        "alphaapollo.common.execution.robotics.backends.libero._camera_metadata",
        lambda *args, **kwargs: _GEOMETRY_CAMERA,
    )
    return LiberoBackend(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        camera_height=_GEOMETRY_SIZE,
        camera_width=_GEOMETRY_SIZE,
        image_orientation=image_orientation,
        enable_depth=True,
        settle_steps=0,
        suite_loader=lambda name, order: _Suite(),
        env_factory=lambda folder, bddl, **options: _GeometryEnv(succeeds_after=None),
        image_encoder=lambda image: b"png",
        depth_converter=lambda environment, depth: depth,
        clock=lambda: "2026-08-18T00:00:00Z",
    )


@pytest.mark.parametrize(
    ("image_orientation", "displayed_pixel"),
    [
        # The published frame is the upright view, so the planner names it directly.
        ("upright", _GEOMETRY_UPRIGHT_PIXEL),
        # The published frame is the bottom-up buffer, so the row is mirrored.
        ("raw", _GEOMETRY_RAW_PIXEL),
        # The published frame is upright and mirrored horizontally.
        (
            "rotate_180",
            (
                _GEOMETRY_SIZE - 1 - _GEOMETRY_UPRIGHT_PIXEL[0],
                _GEOMETRY_UPRIGHT_PIXEL[1],
            ),
        ),
    ],
)
def test_libero_back_project_maps_every_display_orientation_to_one_world_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_orientation: str,
    displayed_pixel: tuple[int, int],
) -> None:
    """The depth row and the calibration row are mirror images, never the same row.

    Indexing the depth buffer with the calibration row (or calibrating with the
    raw row) still returns a finite, plausible point — it just sits on the wrong
    side of the optical axis, which is why this pins both rows at once.
    """

    backend = _geometry_backend(tmp_path, monkeypatch, image_orientation=image_orientation)
    backend.reset(_task())

    point = backend.back_project(
        camera="fixed_camera",
        pixel_x=displayed_pixel[0],
        pixel_y=displayed_pixel[1],
    )

    assert point.depth_m == _GEOMETRY_DEPTH
    assert point.world_xyz == pytest.approx(_GEOMETRY_WORLD)


def test_libero_depth_failure_keeps_images_and_the_terminal_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Depth is opt-in grounding, so its faults must not cost the observation."""

    monkeypatch.setattr(
        "alphaapollo.common.execution.robotics.backends.libero._camera_metadata",
        lambda *args, **kwargs: _GEOMETRY_CAMERA,
    )
    converted = 0

    def failing_converter(environment: Any, depth: Any) -> Any:
        nonlocal converted
        converted += 1
        if converted > 1:
            raise RuntimeError("depth conversion failed")
        return depth

    backend = LiberoBackend(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        camera_height=_GEOMETRY_SIZE,
        camera_width=_GEOMETRY_SIZE,
        image_orientation="upright",
        enable_depth=True,
        settle_steps=0,
        suite_loader=lambda name, order: _Suite(),
        env_factory=lambda folder, bddl, **options: _GeometryEnv(succeeds_after=1),
        image_encoder=lambda image: b"png",
        depth_converter=failing_converter,
        clock=lambda: "2026-08-18T00:00:00Z",
    )
    backend.reset(_task())

    transition = backend.execute(_action())

    assert transition.terminated is True
    assert transition.success is True
    assert transition.termination_reason == "task_success"
    # The camera frames and the proprio state survive a depth-only fault.
    assert len(transition.observation.artifact_refs) == 1
    assert "observation_warning" not in transition.observation.backend_metadata
    assert transition.observation.backend_metadata["depth_warnings"] == (
        "fixed_camera: RuntimeError: depth conversion failed",
    )
    # Grounding is reported as unavailable rather than answered from stale depth.
    with pytest.raises(RuntimeError, match="no depth for camera"):
        backend.back_project(camera="fixed_camera", pixel_x=0, pixel_y=0)


def test_libero_image_failure_still_returns_the_successful_transition(tmp_path: Path) -> None:
    env = _Env(succeeds_after=1)
    calls = 0

    def encoder(image: Any) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("artifact disk full")
        return b"png"

    backend = LiberoBackend(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        suite_loader=lambda name, order: _Suite(),
        env_factory=lambda folder, bddl, *, camera_height, camera_width: env,
        image_encoder=encoder,
        image_orientation="raw",
        settle_steps=0,
    )
    backend.reset(_task())

    transition = backend.execute(_action())

    assert transition.terminated is True
    assert transition.success is True
    assert transition.observation.state["proprio"]["robot0_eef_pos"] == (0.0, 0.1, 0.2)
    assert transition.observation.backend_metadata["observation_warning"]


def test_libero_noncanonical_step_data_is_degraded_without_losing_success(
    tmp_path: Path,
) -> None:
    class _DivergedEnv(_Env):
        def _obs(self) -> dict[str, Any]:
            observation = super()._obs()
            if self.steps:
                observation["robot0_eef_pos"] = [float("nan"), 0.1, 0.2]
            return observation

        def step(self, action: list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            observation, _reward, done, info = super().step(action)
            return observation, float("nan"), done, info

    env = _DivergedEnv(succeeds_after=1)
    backend = _backend(env, tmp_path / "artifacts")
    backend.reset(_task())

    transition = backend.execute(_action())

    assert transition.terminated is True and transition.success is True
    assert transition.observation is backend.observe()
    assert "robot0_eef_pos" not in transition.observation.state["proprio"]
    assert transition.observation.backend_metadata["dropped_state_fields"] == ("robot0_eef_pos",)
    assert "reward" not in transition.info
    assert transition.info["transition_warnings"]


def test_libero_recorder_failure_does_not_discard_the_transition(tmp_path: Path) -> None:
    env = _Env(succeeds_after=1)

    class _FailingSink:
        calls = 0

        def __init__(self, path: Path, fps: int) -> None:
            self.path = path

        def append(self, frame: Any) -> None:
            type(self).calls += 1
            if type(self).calls > 1:
                raise OSError("artifact disk full")

        def close(self) -> None:
            pass

    backend = _backend(
        env,
        tmp_path / "artifacts",
        record_dir=tmp_path / "rollout",
        video_sink_factory=_FailingSink,
    )
    backend.reset(_task())

    transition = backend.execute(_action())

    assert transition.terminated is True
    assert transition.success is True
    assert backend._recorder is not None
    assert backend._recorder.active is False
    import json

    episode = json.loads((tmp_path / "rollout" / "episode-000" / "episode.json").read_text())
    assert episode["summary"]["reason"] == "recording_failed"
    assert "artifact disk full" in episode["summary"]["recorder_error"]
    assert episode["summary"]["steps_recorded"] == 1


def test_libero_backend_treats_truncation_as_failure(tmp_path: Path) -> None:
    env = _Env(succeeds_after=None)
    backend = _backend(env, tmp_path / "artifacts")
    backend.reset(_task(max_episode_steps=1))

    transition = backend.execute(_action())

    assert transition.terminated is False
    assert transition.truncated is True
    assert transition.success is False
    assert transition.termination_reason == "time_limit"


def test_libero_distinguishes_environment_done_from_time_limit(tmp_path: Path) -> None:
    class _DoneEnv(_Env):
        def step(self, action: list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            observation, reward, _done, info = super().step(action)
            return observation, reward, True, info

    env = _DoneEnv(succeeds_after=None)
    backend = _backend(env, tmp_path / "artifacts")
    backend.reset(_task(max_episode_steps=10))

    transition = backend.execute(_action())

    assert transition.truncated is True
    assert transition.termination_reason == "environment_done"


def test_libero_backend_executes_action_chunks_and_stops_at_terminal(
    tmp_path: Path,
) -> None:
    env = _Env(succeeds_after=2)
    backend = _backend(env, tmp_path / "artifacts")
    backend.reset(_task())

    transition = backend.execute(_action([list(_ACTION), list(_ACTION), list(_ACTION)]))

    assert transition.steps_used == 2
    assert transition.terminated is True
    assert transition.success is True
    assert len(env.actions) == 2
    assert backend.steps_used == 2


def test_libero_backend_rejects_a_different_suite_task(tmp_path: Path) -> None:
    backend = _backend(_Env(), tmp_path / "artifacts")

    with pytest.raises(ValueError, match="task spec mismatch"):
        backend.reset(_task(task_name="a_different_task"))

    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()


def test_libero_failed_rereset_invalidates_the_previous_episode(tmp_path: Path) -> None:
    env = _Env()
    backend = _backend(env, tmp_path / "artifacts")
    backend.reset(_task())

    with pytest.raises(ValueError, match="out of range"):
        backend.reset(_task(initial_state_index=99))

    assert env.closed is True
    assert backend.success is None
    assert backend.steps_used == 0
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.execute(_action())


def test_libero_backend_rejects_different_task_language(tmp_path: Path) -> None:
    suite = _Suite()
    suite.task.language = "a different task"
    backend = _backend(_Env(), tmp_path / "artifacts", suite=suite)

    with pytest.raises(ValueError, match="prepared instruction"):
        backend.reset(_task())


def test_libero_backend_validates_the_prepared_payload(tmp_path: Path) -> None:
    backend = _backend(_Env(), tmp_path / "artifacts")

    with pytest.raises(ValueError, match="missing"):
        backend.reset(
            RobotTask(
                task_id="t",
                benchmark="libero",
                instruction=_INSTRUCTION,
                environment_version="libero-v1.0.1",
                backend_metadata={"suite": "libero_spatial"},
            )
        )
    with pytest.raises(
        ValueError, match="benchmark 'libero' does not include LIBERO suite 'libero_11'"
    ):
        backend.reset(_task(suite="libero_11", problem_folder="libero_11"))
    with pytest.raises(ValueError, match="out of range"):
        backend.reset(_task(initial_state_index=9))
    # An empty environment_version no longer reaches the backend at all: the
    # RobotTask record refuses it as a required identity field.
    with pytest.raises(ValueError, match="environment_version"):
        RobotTask(
            task_id="t",
            benchmark="libero",
            instruction=_INSTRUCTION,
            environment_version="  ",
            backend_metadata=dict(_task().backend_metadata),
        )


def test_libero_backend_rejects_an_environment_version_mismatch(tmp_path: Path) -> None:
    backend = _backend(_Env(), tmp_path / "artifacts", environment_version="libero-v1.0.2")

    with pytest.raises(ValueError, match="environment_version mismatch"):
        backend.reset(_task())


def test_libero_backend_requires_artifact_store_for_camera_observations() -> None:
    backend = _backend(_Env(), None)

    with pytest.raises(RuntimeError, match="artifact_store"):
        backend.reset(_task())


def test_libero_backend_validates_actions_and_closes_the_environment(
    tmp_path: Path,
) -> None:
    env = _Env()
    backend = _backend(env, tmp_path / "artifacts")
    backend.reset(_task())

    with pytest.raises(TypeError, match="must be numeric"):
        backend.execute(_action([True] * 7))
    with pytest.raises(ValueError, match="7 values"):
        backend.execute(_action([0.1, -0.2]))
    assert env.actions == []

    backend.close()
    backend.close()
    assert env.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        backend.observe()


def test_libero_post_step_contract_failure_invalidates_the_episode(tmp_path: Path) -> None:
    class _BrokenOracleEnv(_Env):
        def check_success(self) -> bool:
            if self.steps:
                raise RuntimeError("success oracle disconnected")
            return False

    env = _BrokenOracleEnv()
    backend = _backend(env, tmp_path / "artifacts")
    backend.reset(_task())

    with pytest.raises(RuntimeError, match="oracle disconnected"):
        backend.execute(_action())

    assert env.closed is True
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()


def test_libero_backend_orients_images_upright(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    captured: list[Any] = []

    def _encoder(image: Any) -> bytes:
        captured.append(np.asarray(image))
        return b"png"

    class _TinyImageEnv(_Env):
        def _obs(self) -> dict[str, Any]:
            return {
                "agentview_image": np.asarray([[[0, 0, 0]], [[255, 255, 255]]], dtype=np.uint8),
                "robot0_joint_pos": [0.1] * 7,
            }

    backend = LiberoBackend(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        image_orientation="upright",
        suite_loader=lambda name, order: _Suite(),
        env_factory=lambda folder, bddl, *, camera_height, camera_width: _TinyImageEnv(),
        image_encoder=_encoder,
        clock=lambda: "2026-08-16T00:00:00Z",
        settle_steps=0,
    )

    backend.reset(_task())
    assert captured and captured[0][0][0][0] == 255


def test_libero_backend_records_a_dense_episode_trace_and_video(tmp_path: Path) -> None:
    import json

    videos: list[tuple[Path, list[Any], int]] = []

    class _Sink:
        def __init__(self, path: Path, fps: int) -> None:
            self.path = path
            self.fps = fps
            self.frames: list[Any] = []

        def append(self, frame: Any) -> None:
            self.frames.append(frame)

        def close(self) -> None:
            self.path.write_bytes(b"mp4")
            videos.append((self.path, self.frames, self.fps))

    env = _Env()
    backend = _backend(
        env,
        tmp_path / "artifacts",
        record_dir=tmp_path / "rollout",
        record_fps=10,
        video_sink_factory=_Sink,
    )
    backend.reset(_task())
    backend.execute(_action())
    backend.execute(_action())

    episode_dir = tmp_path / "rollout" / "episode-000"
    trace = [
        json.loads(line)
        for line in (episode_dir / "trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # Reset frame plus one line per simulator step, indexed by cumulative steps_used.
    assert [record["step"] for record in trace] == [0, 1, 2]
    assert trace[0]["action"] is None
    assert trace[1]["action"] == list(_ACTION)
    assert trace[1]["terminated"] is False and trace[1]["reward"] == 0.0
    assert trace[2]["terminated"] is True and trace[2]["success"] is True
    assert trace[2]["reward"] == 1.0
    assert trace[1]["proprio"]["robot0_eef_pos"] == [0.0, 0.1, 0.2]

    episode = json.loads((episode_dir / "episode.json").read_text())
    assert episode["instruction"] == _INSTRUCTION
    assert episode["environment"]["task_name"] == _TASK_NAME
    assert episode["summary"] == {
        "reason": "task_success",
        "steps_recorded": 2,
        "frames_recorded": 3,
        "finished_at": "2026-08-16T00:00:00Z",
    }
    assert videos and videos[0][0] == episode_dir / "episode.mp4"
    assert len(videos[0][1]) == 3 and videos[0][2] == 10


def test_libero_backend_recorder_closes_open_episodes(tmp_path: Path) -> None:
    videos: list[Path] = []

    class _Sink:
        def __init__(self, path: Path, fps: int) -> None:
            self.path = path

        def append(self, frame: Any) -> None:
            pass

        def close(self) -> None:
            self.path.write_bytes(b"mp4")
            videos.append(self.path)

    env = _Env(succeeds_after=None)
    backend = _backend(
        env,
        tmp_path / "artifacts",
        record_dir=tmp_path / "rollout",
        video_sink_factory=_Sink,
    )
    backend.reset(_task(max_episode_steps=99))
    backend.execute(_action())
    # A new reset supersedes the open episode; close() flushes the last one.
    backend.reset(_task(max_episode_steps=99))
    backend.close()

    import json

    first = json.loads((tmp_path / "rollout" / "episode-000" / "episode.json").read_text())
    second = json.loads((tmp_path / "rollout" / "episode-001" / "episode.json").read_text())
    assert first["summary"]["reason"] == "superseded_by_reset"
    assert first["summary"]["steps_recorded"] == 1
    assert second["summary"]["reason"] == "backend_closed"
    assert len(videos) == 2


def test_libero_backend_video_failure_still_closes_the_simulator(tmp_path: Path) -> None:
    import json

    class _BrokenSink:
        def __init__(self, path: Path, fps: int) -> None:
            pass

        def append(self, frame: Any) -> None:
            pass

        def close(self) -> None:
            raise RuntimeError("no ffmpeg plugin")

    env = _Env(succeeds_after=None)
    backend = _backend(
        env,
        tmp_path / "artifacts",
        record_dir=tmp_path / "rollout",
        video_sink_factory=_BrokenSink,
    )
    backend.reset(_task(max_episode_steps=99))
    backend.execute(_action())

    with pytest.raises(RuntimeError, match="no ffmpeg plugin"):
        backend.close()

    # The env must not leak behind an unrepeatable close(), and the episode
    # metadata must record the video failure honestly.
    assert env.closed is True
    episode = json.loads((tmp_path / "rollout" / "episode-000" / "episode.json").read_text())
    assert episode["summary"]["reason"] == "backend_closed"
    assert "no ffmpeg plugin" in episode["summary"]["video_error"]
    assert episode["summary"]["steps_recorded"] == 1


def test_libero_backend_failed_reset_leaves_the_backend_inactive(tmp_path: Path) -> None:
    env = _Env()
    calls = {"count": 0}

    def _factory(folder: str, bddl: str, *, camera_height: int, camera_width: int) -> Any:
        calls["count"] += 1
        if calls["count"] > 1:
            raise FileNotFoundError("bddl file gone")
        return env

    store = ArtifactStore(tmp_path / "artifacts")
    backend = LiberoBackend(
        artifact_store=store,
        image_orientation="raw",
        suite_loader=lambda name, order: _Suite(),
        env_factory=_factory,
        image_encoder=lambda image: b"png",
        clock=lambda: "2026-08-17T00:00:00Z",
        settle_steps=0,
    )
    backend.reset(_task())
    first = backend.observe()
    assert first is not None

    with pytest.raises(FileNotFoundError):
        backend.reset(_task())

    # The stale episode must not survive the failed re-reset.
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()


def test_episode_recorder_numbering_survives_directory_collisions(tmp_path: Path) -> None:
    from alphaapollo.common.execution.robotics.backends import EpisodeRecorder

    root = tmp_path / "rollout"
    (root / "episode-000").mkdir(parents=True)
    (root / "episode-002").mkdir()

    class _NullSink:
        def __init__(self, path: Path, fps: int) -> None:
            pass

        def append(self, frame: Any) -> None:
            pass

        def close(self) -> None:
            pass

    recorder = EpisodeRecorder(root, video_sink_factory=_NullSink)

    # Two existing dirs -> index 2 collides with the pre-created episode-002;
    # mkdir itself must be the collision test, retrying to 003.
    episode_dir = recorder.start({"task": "t"})

    assert episode_dir.name == "episode-003"
    recorder.finish(reason="test")


def test_libero_backend_refuses_an_env_without_a_success_oracle(tmp_path: Path) -> None:
    class _NoOracleEnv(_Env):
        check_success = None  # type: ignore[assignment]

    backend = _backend(_NoOracleEnv(), tmp_path / "artifacts")

    # robosuite's done is also True on horizon timeout, so success falling
    # back to done would fabricate positives; the backend must fail closed.
    with pytest.raises(RuntimeError, match="check_success oracle"):
        backend.reset(_task())
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()


def test_libero_backend_finds_the_oracle_one_wrapper_down(tmp_path: Path) -> None:
    inner = _Env()

    class _Wrapper:
        def __init__(self) -> None:
            self.env = inner

        def reset(self) -> dict[str, Any]:
            return inner.reset()

        def set_init_state(self, state: Any) -> dict[str, Any]:
            return inner.set_init_state(state)

        def step(self, action: list[float]) -> Any:
            return inner.step(action)

        def close(self) -> None:
            inner.close()

    backend = _backend(_Wrapper(), tmp_path / "artifacts")  # type: ignore[arg-type]
    backend.reset(_task())
    backend.execute(_action())
    transition = backend.execute(_action())

    assert transition.terminated is True and transition.success is True


def test_libero_rejects_success_reached_during_reset_settling(tmp_path: Path) -> None:
    env = _Env(succeeds_after=1)
    backend = _backend(env, tmp_path / "artifacts", settle_steps=2)

    with pytest.raises(RuntimeError, match="settle step reached task success"):
        backend.reset(_task())

    assert env.steps == 1
    assert env.closed is True
    with pytest.raises(RuntimeError, match="reset must be called"):
        backend.observe()


def test_libero_rejects_an_initially_satisfied_task(tmp_path: Path) -> None:
    env = _Env(succeeds_after=0)
    backend = _backend(env, tmp_path / "artifacts", settle_steps=0)

    with pytest.raises(RuntimeError, match="initial state is already successful"):
        backend.reset(_task())

    assert env.closed is True


def test_libero_default_reset_uses_the_official_ten_step_settle(tmp_path: Path) -> None:
    env = _Env(succeeds_after=None)
    backend = _backend(env, tmp_path / "artifacts", settle_steps=None)

    reset = backend.reset(_task())

    assert env.steps == 10
    assert backend.steps_used == 0
    assert reset.info["settle_steps_used"] == 10
    assert reset.info["settle_steps_count_toward_policy_budget"] is False


def test_libero_settle_rejects_an_unsuccessful_terminal_step(tmp_path: Path) -> None:
    class _TerminalEnv(_Env):
        def step(self, action: list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            observation, reward, _done, info = super().step(action)
            return observation, reward, True, info

    env = _TerminalEnv(succeeds_after=None)
    backend = _backend(env, tmp_path / "artifacts", settle_steps=1)

    with pytest.raises(RuntimeError, match="settle step terminated"):
        backend.reset(_task())
    assert env.closed is True


def test_libero_reset_recording_failure_aborts_recorder(tmp_path: Path) -> None:
    env = _Env()

    def failing_sink(path: Path, fps: int) -> Any:
        raise RuntimeError("video sink failed")

    backend = _backend(
        env,
        tmp_path / "artifacts",
        record_dir=tmp_path / "rollout",
        video_sink_factory=failing_sink,
    )

    with pytest.raises(RuntimeError, match="video sink failed"):
        backend.reset(_task())

    assert backend._recorder is not None
    assert backend._recorder.active is False


def test_libero_backend_refuses_a_suite_task_missing_spec_attributes(
    tmp_path: Path,
) -> None:
    class _BareTask:
        name = _TASK_NAME
        language = _INSTRUCTION
        problem_folder = "libero_spatial"
        bddl_file = f"{_TASK_NAME}.bddl"
        init_states_file = None  # absent attribute must fail closed

    suite = _Suite()
    suite.task = _BareTask()  # type: ignore[assignment]
    backend = _backend(_Env(), tmp_path / "artifacts", suite=suite)

    with pytest.raises(ValueError, match="cannot verify the prepared init_states_file"):
        backend.reset(_task())


def test_libero_failure_paths_report_the_cause_not_the_close_error(tmp_path: Path) -> None:
    class _UncloseableEnv(_Env):
        def close(self) -> None:
            super().close()
            raise RuntimeError("mujoco context teardown failed")

    solved = _UncloseableEnv(succeeds_after=0)
    backend = _backend(solved, tmp_path / "artifacts")

    with pytest.raises(RuntimeError, match="initial state is already successful"):
        backend.reset(_task())
    assert solved.closed is True

    class _ExplodingStep(_UncloseableEnv):
        def step(self, action: list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            raise RuntimeError("mujoco step exploded")

    stepping = _ExplodingStep()
    stepping_backend = _backend(stepping, tmp_path / "artifacts")
    stepping_backend.reset(_task())

    with pytest.raises(RuntimeError, match="mujoco step exploded"):
        stepping_backend.execute(_action())
    assert stepping.closed is True


# --- LIBERO-Pro: same episode machinery, a different package ---------------

_PRO_SUITE = "libero_spatial_swap"
_PRO_VERSION = "0.1.1"


def _liberopro_task(**metadata: Any) -> RobotTask:
    values = {
        "suite": _PRO_SUITE,
        "task_order_index": 0,
        "task_index": 3,
        "task_name": _TASK_NAME,
        "problem_folder": _PRO_SUITE,
        "bddl_file": f"{_TASK_NAME}.bddl",
        "init_states_file": f"{_TASK_NAME}.pruned_init",
        "initial_state_index": 1,
        "max_episode_steps": 3,
        "package_distribution": "rpent-liberopro",
        "package_version": _PRO_VERSION,
    }
    values.update(metadata)
    return RobotTask(
        task_id="liberopro-spatial-swap-3-1",
        benchmark="liberopro",
        instruction=_INSTRUCTION,
        environment_version=f"rpent-liberopro=={_PRO_VERSION}",
        backend_metadata=values,
    )


def _pro_suite() -> _Suite:
    suite = _Suite()
    suite.task.problem_folder = _PRO_SUITE
    return suite


def test_libero_backend_runs_a_prepared_liberopro_task(tmp_path: Path) -> None:
    env = _Env()
    suite = _pro_suite()
    backend = _backend(env, tmp_path / "artifacts", suite=suite)

    reset = backend.reset(_liberopro_task())
    backend.execute(_action())
    transition = backend.execute(_action())

    assert suite.requested == [3]
    environment = reset.info["environment"]
    assert environment["benchmark"] == "liberopro"
    assert environment["suite"] == _PRO_SUITE
    assert environment["package_distribution"] == "rpent-liberopro"
    assert environment["import_root"] == "liberopro.liberopro"
    assert environment["backend_metadata"]["package_version"] == _PRO_VERSION
    assert transition.terminated is True
    assert backend.success is True


def test_libero_backend_records_no_distribution_for_a_libero_checkout(tmp_path: Path) -> None:
    reset = _backend(_Env(), tmp_path / "artifacts").reset(_task())

    assert reset.info["environment"]["package_distribution"] is None
    assert reset.info["environment"]["import_root"] == "libero.libero"


@pytest.mark.parametrize(
    ("task_factory", "pattern"),
    [
        (
            lambda: _task(suite=_PRO_SUITE, problem_folder=_PRO_SUITE),
            "benchmark 'libero' does not include LIBERO suite 'libero_spatial_swap'",
        ),
        (
            lambda: _liberopro_task(suite="libero_spatial", problem_folder="libero_spatial"),
            "benchmark 'liberopro' does not include LIBERO suite 'libero_spatial'",
        ),
    ],
)
def test_libero_backend_keeps_each_benchmark_closed_to_the_other_suites(
    tmp_path: Path, task_factory: Any, pattern: str
) -> None:
    backend = _backend(_Env(), tmp_path / "artifacts")

    with pytest.raises(ValueError, match=pattern):
        backend.reset(task_factory())


def test_libero_backend_refuses_an_unknown_benchmark(tmp_path: Path) -> None:
    backend = _backend(_Env(), tmp_path / "artifacts")
    task = RobotTask(
        task_id="t",
        benchmark="robocasa",
        instruction=_INSTRUCTION,
        environment_version="robocasa@921c9a5",
        backend_metadata=dict(_task().backend_metadata),
    )

    with pytest.raises(ValueError, match="cannot reset benchmark 'robocasa'"):
        backend.reset(task)


def test_libero_backend_refuses_a_problem_folder_owned_by_another_package(
    tmp_path: Path,
) -> None:
    backend = _backend(_Env(), tmp_path / "artifacts")

    with pytest.raises(ValueError, match="which is not one of its suites"):
        backend.reset(_liberopro_task(problem_folder="libero_spatial"))


@pytest.mark.parametrize(
    ("task_factory", "pattern"),
    [
        (
            lambda: _liberopro_task(package_distribution="libero"),
            "declares package_distribution='libero'",
        ),
        (
            lambda: _liberopro_task(package_distribution=None),
            "declares package_distribution=None",
        ),
        (
            lambda: _task(package_distribution="rpent-liberopro"),
            "declares package_distribution='rpent-liberopro'",
        ),
        (
            lambda: _task(package_version="0.1.1"),
            "cannot accept package_version='0.1.1'",
        ),
    ],
)
def test_libero_backend_refuses_a_foreign_package_identity(
    tmp_path: Path, task_factory: Any, pattern: str
) -> None:
    backend = _backend(_Env(), tmp_path / "artifacts")

    with pytest.raises(ValueError, match=pattern):
        backend.reset(task_factory())


def test_libero_backend_requires_the_liberopro_package_version(tmp_path: Path) -> None:
    backend = _backend(_Env(), tmp_path / "artifacts")
    metadata = dict(_liberopro_task().backend_metadata)
    del metadata["package_version"]
    task = RobotTask(
        task_id="t",
        benchmark="liberopro",
        instruction=_INSTRUCTION,
        environment_version=f"rpent-liberopro=={_PRO_VERSION}",
        backend_metadata=metadata,
    )

    with pytest.raises(ValueError, match="require package_version"):
        backend.reset(task)


def test_libero_backend_refuses_an_environment_version_that_drops_the_package_version(
    tmp_path: Path,
) -> None:
    backend = _backend(_Env(), tmp_path / "artifacts")
    task = RobotTask(
        task_id="t",
        benchmark="liberopro",
        instruction=_INSTRUCTION,
        environment_version="rpent-liberopro==0.1.0",
        backend_metadata=dict(_liberopro_task().backend_metadata),
    )

    with pytest.raises(ValueError, match="does not state the prepared package identity"):
        backend.reset(task)


# --- Package selection in the default factories ----------------------------
#
# The defaults are the only code that reaches a real simulator package, so the
# choice of package is exercised here against stub modules rather than through
# the injected factories the episode tests use.


def _install_stub_packages(monkeypatch: pytest.MonkeyPatch, *, bddl_root: Path) -> dict[str, Any]:
    """Register importable stand-ins for both simulator packages.

    Each stub answers for exactly one distribution, so a lookup that reached
    the wrong package would return that package's marker instead of raising.
    """

    seen: dict[str, Any] = {}
    for import_root, suites in (
        ("libero.libero", ("libero_spatial", "libero_object", "libero_goal")),
        ("liberopro.liberopro", (_PRO_SUITE, "libero_spatial", "libero_goal_lan")),
    ):
        top = ModuleType(import_root.split(".", 1)[0])
        package = ModuleType(import_root)
        benchmark = ModuleType(f"{import_root}.benchmark")
        envs = ModuleType(f"{import_root}.envs")

        benchmark.get_benchmark_dict = (  # type: ignore[attr-defined]
            lambda root=import_root, names=suites: {
                name: (lambda order, root=root, name=name: f"{root}:{name}:{order}")
                for name in names
            }
        )
        package.get_libero_path = (  # type: ignore[attr-defined]
            lambda key, root=import_root: str(bddl_root / root.split(".", 1)[0] / key)
        )

        def _env(
            *,
            bddl_file_name: str,
            camera_heights: int,
            camera_widths: int,
            camera_depths: bool = False,
            root=import_root,
        ):
            seen[root] = bddl_file_name
            return f"{root}:{bddl_file_name}:{camera_heights}x{camera_widths}:{camera_depths}"

        envs.OffScreenRenderEnv = _env  # type: ignore[attr-defined]
        for name, module in (
            (import_root.split(".", 1)[0], top),
            (import_root, package),
            (f"{import_root}.benchmark", benchmark),
            (f"{import_root}.envs", envs),
        ):
            monkeypatch.setitem(sys.modules, name, module)
    return seen


@pytest.mark.parametrize(
    ("suite", "expected_root"),
    [("libero_spatial", "libero.libero"), (_PRO_SUITE, "liberopro.liberopro")],
)
def test_default_suite_loader_reads_each_suite_from_its_own_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suite: str, expected_root: str
) -> None:
    _install_stub_packages(monkeypatch, bddl_root=tmp_path)

    assert libero_backend._default_suite_loader(suite, 2) == f"{expected_root}:{suite}:2"


def test_default_suite_loader_refuses_a_suite_no_benchmark_owns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_stub_packages(monkeypatch, bddl_root=tmp_path)

    with pytest.raises(ValueError, match="belongs to no supported benchmark"):
        libero_backend._default_suite_loader("libero_spatial_trigger", 0)


@pytest.mark.parametrize(
    ("problem_folder", "expected_root"),
    [("libero_spatial", "libero.libero"), (_PRO_SUITE, "liberopro.liberopro")],
)
def test_default_env_factory_reads_bddl_assets_from_its_own_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, problem_folder: str, expected_root: str
) -> None:
    seen = _install_stub_packages(monkeypatch, bddl_root=tmp_path)
    top = expected_root.split(".", 1)[0]
    folder = tmp_path / top / "bddl_files" / problem_folder
    folder.mkdir(parents=True)
    (folder / f"{_TASK_NAME}.bddl").write_text("(define)")

    env = libero_backend._default_env_factory(
        problem_folder,
        f"{_TASK_NAME}.bddl",
        camera_height=128,
        camera_width=128,
    )

    assert env.startswith(f"{expected_root}:")
    assert seen == {expected_root: str(folder / f"{_TASK_NAME}.bddl")}
    # Depth is opt-in, so the package's own factory must be asked for it
    # explicitly rather than inheriting whatever the simulator defaults to.
    assert env.endswith(":False")


def test_libero_distributions_claim_disjoint_suites() -> None:
    seen: set[str] = set()
    for record in libero_backend.LIBERO_DISTRIBUTIONS:
        assert not seen & set(record.suites)
        seen |= set(record.suites)
    assert len(seen) == sum(len(record.suites) for record in libero_backend.LIBERO_DISTRIBUTIONS)


def test_libero_distributions_cover_exactly_the_prepared_suites() -> None:
    """Every suite a preparer can publish must be one this backend accepts."""

    suites = {record.benchmark: record.suites for record in libero_backend.LIBERO_DISTRIBUTIONS}
    assert suites["libero"] == prepare_libero.LIBERO_SUITES
    assert suites["liberopro"] == prepare_liberopro.LIBEROPRO_SUITES
    assert suites["liberopro"] != ()
