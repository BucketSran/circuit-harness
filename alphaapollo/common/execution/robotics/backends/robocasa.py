# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""RoboCasa365 adapter for the backend-neutral robot execution contract."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from numbers import Integral
from typing import Any, Protocol

from alphaapollo.common.artifacts.schemas import ArtifactRef, require_json_value
from alphaapollo.common.execution.robotics.geometry import ground_metric_depth
from alphaapollo.common.execution.robotics.schemas import (
    RobotAction,
    RobotGroundedPoint,
    RobotObservation,
    RobotResetResult,
    RobotTask,
    RobotTransition,
)
from alphaapollo.common.execution.workspace import ArtifactStore

logger = logging.getLogger(__name__)

ImageEncoder = Callable[[Any], bytes]
DepthConverter = Callable[[Any, Any], Any]
Clock = Callable[[], str]
EnvironmentFactory = Callable[[RobotTask], "RoboCasaEnvironment"]

_SCENE_FIELDS = ("task_name", "layout_id", "style_id", "scene_seed")


class RoboCasaEnvironment(Protocol):
    """The small upstream surface used by :class:`RoboCasaBackend`."""

    layout_id: int
    style_id: int
    horizon: int
    action_dim: int

    def reset(self) -> Mapping[str, Any]: ...

    def step(self, action: Sequence[float]) -> tuple[Any, Any, Any, Any]: ...

    def get_ep_meta(self) -> Mapping[str, Any]: ...

    def _check_success(self) -> Any: ...

    def close(self) -> None: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _camera_metadata(environment: Any, raw: Any) -> dict[str, Any]:
    """Best-effort intrinsics/extrinsics per observed camera for geometric tools.

    Camera names and render sizes are derived from the observation's image
    keys, so this tracks whatever camera set the RoboCasa scene exposes.
    Absent metadata is not an error: inspection tools report it as unavailable.
    """

    try:
        from robosuite.utils import camera_utils
    except ImportError:
        return {}
    sim = getattr(environment, "sim", None)
    if sim is None or not isinstance(raw, Mapping):
        return {}
    cameras: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        if not (name.endswith("_image") or name.endswith("_rgb")):
            continue
        shape = getattr(value, "shape", None)
        if not shape or len(shape) < 2:
            continue
        camera_name = name.rsplit("_", 1)[0]
        height, width = int(shape[0]), int(shape[1])
        try:
            intrinsics = camera_utils.get_camera_intrinsic_matrix(sim, camera_name, height, width)
            extrinsics = camera_utils.get_camera_extrinsic_matrix(sim, camera_name)
        except Exception:  # noqa: BLE001 - a missing camera only skips its metadata
            continue
        cameras[name] = {
            "name": camera_name,
            "height": height,
            "width": width,
            "intrinsics": [[float(v) for v in row] for row in intrinsics],
            "extrinsics": [[float(v) for v in row] for row in extrinsics],
        }
    return cameras


def _action_array(values: Sequence[float]) -> Any:
    """Hand robosuite a mutable array; composite controllers slice and copy it.

    A tuple breaks robosuite's mobile-base controllers, which call
    ``action.copy()`` on their slice of the action vector. NumPy is always
    present in a real simulator runtime; contract tests without it get a list,
    which supports the same slice-and-copy usage.
    """

    try:
        import numpy as np
    except ImportError:
        return list(values)
    return np.asarray(values, dtype=np.float64)


def _encode_png(image: Any) -> bytes:
    """Encode an RGB observation without importing robotics extras at module load."""

    try:
        import imageio.v3 as iio
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "RoboCasa image encoding requires the optional robotics dependencies"
        ) from exc
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"RoboCasa RGB image must have HxWx3 shape, got {array.shape}")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return iio.imwrite("<bytes>", array, extension=".png")


def _json_value(value: Any, *, path: str) -> Any:
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    elif hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, Mapping):
        value = {str(key): _json_value(item, path=f"{path}.{key}") for key, item in value.items()}
    elif isinstance(value, tuple):
        value = [_json_value(item, path=f"{path}[]") for item in value]
    elif isinstance(value, list):
        value = [_json_value(item, path=f"{path}[]") for item in value]
    require_json_value(value, path=path)
    return value


def _scalar_bool(value: Any, *, name: str) -> bool:
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except ValueError as exc:
            raise ValueError(f"{name} must be scalar") from exc
    if not isinstance(value, (bool, int)):
        raise TypeError(f"{name} must be bool-like")
    return bool(value)


def _resolve_action_dim(environment: Any) -> int:
    """Resolve the native controller width and fail closed when unavailable."""

    raw = getattr(environment, "action_dim", None)
    if raw is not None:
        if isinstance(raw, bool) or not isinstance(raw, Integral) or raw < 1:
            raise ValueError("RoboCasa environment action_dim must be a positive integer")
        return int(raw)

    action_spec = getattr(environment, "action_spec", None)
    shape = getattr(action_spec, "shape", None)
    if (
        shape is not None
        and not isinstance(shape, (str, bytes))
        and isinstance(shape, Sequence)
        and len(shape) == 1
        and isinstance(shape[0], Integral)
        and not isinstance(shape[0], bool)
        and shape[0] >= 1
    ):
        return int(shape[0])
    raise RuntimeError(
        "RoboCasa environment exposes no usable action_dim or one-dimensional action_spec; "
        "refusing to send an unchecked action vector"
    )


def _scene_metadata(task: RobotTask) -> dict[str, Any]:
    if task.benchmark.lower() not in {"robocasa", "robocasa365"}:
        raise ValueError(f"RoboCasaBackend cannot reset benchmark {task.benchmark!r}")
    metadata = dict(task.backend_metadata)
    missing = [field for field in _SCENE_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"RoboCasa task metadata is missing {missing}")
    task_name = metadata["task_name"]
    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("RoboCasa task_name must be a non-empty string")
    scene = {"task_name": task_name.strip()}
    for field in ("layout_id", "style_id", "scene_seed"):
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"RoboCasa {field} must be an integer")
        minimum = 0 if field == "scene_seed" else 1
        if value < minimum:
            raise ValueError(f"RoboCasa {field} must be >= {minimum}")
        scene[field] = value
    return scene


def create_robocasa_environment(
    task: RobotTask, *, enable_depth: bool = False
) -> RoboCasaEnvironment:
    """Create the exact scene triplet emitted by ``prepare_robocasa``.

    Imports stay inside the factory so base AlphaApollo does not require the
    RoboCasa runtime, MuJoCo, or NumPy merely to import execution contracts.
    """

    scene = _scene_metadata(task)
    try:
        from robocasa.utils.env_utils import create_env
    except ImportError as exc:
        raise RuntimeError(
            "creating a RoboCasa backend requires the optional 'robocasa' dependencies"
        ) from exc
    options = {
        "env_name": scene["task_name"],
        "layout_and_style_ids": [(scene["layout_id"], scene["style_id"])],
        "seed": scene["scene_seed"],
        "render_onscreen": False,
    }
    if enable_depth:
        options["camera_depths"] = True
    return create_env(**options)


def _default_depth_converter(environment: Any, depth: Any) -> Any:
    try:
        from robosuite.utils import camera_utils
    except ImportError as exc:
        raise RuntimeError(
            "RoboCasa depth conversion requires the optional robosuite dependency"
        ) from exc
    sim = getattr(environment, "sim", None)
    if sim is None:
        raise RuntimeError("RoboCasa environment exposes no simulator for depth conversion")
    return camera_utils.get_real_depth_map(sim, depth)


class RoboCasaBackend:
    """Execute one prepared RoboCasa scene through its native success oracle."""

    def __init__(
        self,
        *,
        environment_factory: EnvironmentFactory = create_robocasa_environment,
        artifact_store: ArtifactStore | None = None,
        environment_version: str | None = None,
        enable_depth: bool = False,
        image_encoder: ImageEncoder = _encode_png,
        depth_converter: DepthConverter = _default_depth_converter,
        clock: Clock = _utc_now,
    ) -> None:
        if not callable(environment_factory):
            raise TypeError("environment_factory must be callable")
        if artifact_store is not None and not isinstance(artifact_store, ArtifactStore):
            raise TypeError("artifact_store must be an ArtifactStore")
        if not isinstance(enable_depth, bool):
            raise TypeError("enable_depth must be a bool")
        if environment_version is not None:
            if not isinstance(environment_version, str) or not environment_version.strip():
                raise ValueError("environment_version must be a non-empty string when provided")
            environment_version = environment_version.strip()
        if not callable(image_encoder):
            raise TypeError("image_encoder must be callable")
        if not callable(depth_converter):
            raise TypeError("depth_converter must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._environment_factory = (
            (lambda task: create_robocasa_environment(task, enable_depth=enable_depth))
            if environment_factory is create_robocasa_environment
            else environment_factory
        )
        self._artifact_store = artifact_store
        self._environment_version = environment_version
        self._enable_depth = enable_depth
        self._image_encoder = image_encoder
        self._depth_converter = depth_converter
        self._clock = clock
        self._environment: RoboCasaEnvironment | None = None
        self._task: RobotTask | None = None
        self._observation: RobotObservation | None = None
        self._cameras: dict[str, Any] = {}
        self._depth_maps: dict[str, Any] = {}
        self._terminated = False
        self._truncated = False
        self._success: bool | None = None
        self._termination_reason: str | None = None
        self._steps_used = 0
        self._max_episode_steps: int | None = None
        self._action_dim: int | None = None
        self._closed = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def success(self) -> bool | None:
        return self._success

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    @property
    def steps_used(self) -> int:
        return self._steps_used

    def reset(self, task: RobotTask) -> RobotResetResult:
        self._require_open()
        # Invalidate before scene/version validation: a failed re-reset must
        # leave no stale environment or task available to observe/execute.
        self._invalidate_episode()
        if not isinstance(task, RobotTask):
            raise TypeError("task must be a RobotTask")
        scene = _scene_metadata(task)
        if (
            self._environment_version is not None
            and task.environment_version.strip() != self._environment_version
        ):
            # The sampled instructions depend on the robocasa/robosuite
            # checkout; running a task prepared on another one fails closed
            # here just like the LIBERO backend.
            raise ValueError(
                "RoboCasa environment_version mismatch: the prepared task requires "
                f"{task.environment_version!r}, this backend serves "
                f"{self._environment_version!r}"
            )
        environment = self._environment_factory(task)
        try:
            raw_observation = environment.reset()
            action_dim = _resolve_action_dim(environment)
            cameras = _camera_metadata(environment, raw_observation)
            self._validate_reset(environment, task, scene)
            # The oracle decides the episode, so a scene already satisfying it
            # would score a success no policy earned.  LIBERO refuses the same
            # initial state; a mis-prepared scene is a preparation bug.
            if _scalar_bool(environment._check_success(), name="_check_success"):
                raise RuntimeError(
                    "RoboCasa scene is already successful at reset; refusing to score a "
                    "task the policy did not solve"
                )
            observation = self._convert_observation(
                raw_observation,
                task=task,
                cameras=cameras,
                environment=environment,
            )
            # Resolve the horizon before publishing any episode state.  Some
            # robosuite versions expose numpy scalars or floats here.
            configured_limit = dict(task.backend_metadata).get("max_episode_steps")
            if configured_limit is None:
                horizon = getattr(environment, "horizon", None)
                configured_limit = None if horizon is None else int(horizon)
            max_episode_steps = self._positive_limit(configured_limit)
        except BaseException:
            self._depth_maps = {}
            self._quiet_close(environment)
            raise

        self._environment = environment
        self._task = task
        self._observation = observation
        self._cameras = cameras
        self._terminated = False
        self._truncated = False
        self._success = None
        self._termination_reason = None
        self._steps_used = 0
        self._action_dim = action_dim
        self._max_episode_steps = max_episode_steps
        return RobotResetResult(
            observation=observation,
            info={
                "scene": scene,
                "action_dim": action_dim,
                "environment": {
                    "backend": "robocasa",
                    "benchmark": task.benchmark,
                    "task_id": task.task_id,
                    "environment_version": task.environment_version,
                    "backend_metadata": dict(task.backend_metadata),
                    **scene,
                },
            },
        )

    def observe(self) -> RobotObservation:
        self._require_active("observe")
        observation = self._observation
        if observation is None:
            raise RuntimeError("reset must be called before observe")
        return observation

    def back_project(self, *, camera: str, pixel_x: int, pixel_y: int) -> RobotGroundedPoint:
        self._require_active("back_project")
        if not self._enable_depth:
            raise RuntimeError("RoboCasa metric depth is disabled; set enable_depth=true")
        depth = self._depth_maps.get(camera)
        if depth is None:
            raise RuntimeError(f"current RoboCasa observation has no depth for camera {camera!r}")
        camera_meta = self._cameras.get(camera)
        if not isinstance(camera_meta, Mapping):
            raise ValueError(f"unknown camera {camera!r}; available: {sorted(self._depth_maps)}")
        width = int(camera_meta["width"])
        height = int(camera_meta["height"])
        if not 0 <= pixel_x < width or not 0 <= pixel_y < height:
            raise ValueError(
                f"pixel ({pixel_x}, {pixel_y}) is outside camera {camera!r} image {width}x{height}"
            )
        depth_m = self._depth_value(depth, pixel_x=pixel_x, pixel_y=pixel_y)
        observation = self.observe()
        return ground_metric_depth(
            camera=camera,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            image_width=width,
            image_height=height,
            depth_m=depth_m,
            intrinsics=camera_meta["intrinsics"],
            camera_to_world=camera_meta["extrinsics"],
            steps_used=self._steps_used,
            observation_timestamp=observation.timestamp,
        )

    def execute(self, action: RobotAction) -> RobotTransition:
        self._require_active("execute")
        if not isinstance(action, RobotAction):
            raise TypeError("action must be a RobotAction")
        if self._terminated or self._truncated:
            raise RuntimeError("cannot execute an action after the RoboCasa episode is terminal")
        values = self._action_values(action)
        assert self._environment is not None
        try:
            result = self._environment.step(_action_array(values))
            if (
                not isinstance(result, Sequence)
                or isinstance(result, (str, bytes))
                or len(result) != 4
            ):
                raise ValueError("RoboCasa env.step must return (observation, reward, done, info)")
            raw_observation, reward, done, raw_info = result
            if not isinstance(raw_info, Mapping):
                raise TypeError("RoboCasa env.step info must be a mapping")
            steps_used = self._steps_used + 1
            succeeded = _scalar_bool(self._environment._check_success(), name="_check_success")
            limit_reached = (
                self._max_episode_steps is not None and steps_used >= self._max_episode_steps
            )
            ended = _scalar_bool(done, name="done")
            terminated = succeeded
            truncated = not succeeded and (ended or limit_reached)
            success = succeeded if terminated or truncated else None
            if terminated:
                termination_reason = "task_success"
            elif limit_reached:
                termination_reason = "time_limit"
            elif ended:
                termination_reason = "environment_done"
            else:
                termination_reason = None
            observation = self._convert_observation_after_step(
                raw_observation,
                task=self._task,
            )
            info = self._transition_info(raw_info, reward)
            transition = RobotTransition(
                observation=observation,
                steps_used=1,
                terminated=terminated,
                truncated=truncated,
                success=success,
                termination_reason=termination_reason,
                info=info,
            )
        except BaseException:
            # Once env.step has returned, any validation failure leaves the
            # simulator ahead of the last published state.  Fail closed instead
            # of allowing a retry against an unknown episode.
            self._invalidate_episode()
            raise

        self._steps_used = steps_used
        self._terminated = terminated
        self._truncated = truncated
        self._success = success
        self._termination_reason = termination_reason
        self._observation = observation
        return transition

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Closing resources must not erase the measured episode outcome.
        self._task = None
        self._observation = None
        self._cameras = {}
        self._max_episode_steps = None
        self._action_dim = None
        self._close_environment()

    def _validate_reset(
        self,
        environment: RoboCasaEnvironment,
        task: RobotTask,
        scene: Mapping[str, Any],
    ) -> None:
        if int(environment.layout_id) != scene["layout_id"]:
            raise ValueError("RoboCasa reset produced a different layout_id")
        if int(environment.style_id) != scene["style_id"]:
            raise ValueError("RoboCasa reset produced a different style_id")
        metadata = environment.get_ep_meta()
        if not isinstance(metadata, Mapping):
            raise TypeError("RoboCasa get_ep_meta must return a mapping")
        instruction = str(metadata.get("lang", "")).strip()
        # Strip both sides: a prepared instruction with stray whitespace must
        # not spuriously fail the correct scene (LIBERO compares the same way).
        if instruction != task.instruction.strip():
            raise ValueError(
                "RoboCasa reset instruction does not match the prepared task; "
                "refusing to run a different sampled scene"
            )

    def _convert_observation(
        self,
        raw: Any,
        *,
        task: RobotTask | None,
        cameras: Mapping[str, Any] | None = None,
        environment: Any | None = None,
    ) -> RobotObservation:
        if not isinstance(raw, Mapping):
            raise TypeError("RoboCasa observation must be a mapping")
        resolved_cameras = self._cameras if cameras is None else cameras
        depth_warnings = self._update_depth_maps(
            raw,
            environment=environment or self._environment,
            cameras=resolved_cameras,
        )
        state: dict[str, Any] = {}
        artifact_refs: list[ArtifactRef] = []
        artifact_roles: dict[str, str] = {}
        for key, value in raw.items():
            name = str(key)
            if name.endswith("_depth"):
                continue
            if name.endswith("_image") or name.endswith("_rgb"):
                if value is None:
                    continue
                if self._artifact_store is None:
                    raise RuntimeError(
                        f"RoboCasa observation contains {name!r}, but no artifact_store "
                        "was configured"
                    )
                encoded = self._image_encoder(value)
                if not isinstance(encoded, bytes):
                    raise TypeError("image_encoder must return bytes")
                artifact = self._artifact_store.put(
                    encoded,
                    type_="image/png",
                    created_by="robocasa-backend",
                )
                ref = self._artifact_store.ref(artifact)
                artifact_refs.append(ref)
                artifact_roles[name] = ref.id
                continue
            state[name] = _json_value(value, path=f"$.state.{name}")
        metadata: dict[str, Any] = {
            "backend": "robocasa",
            "artifact_roles": artifact_roles,
        }
        # Pin the contract shape both in-process backends publish: robot state
        # nests under state.proprio (matching LIBERO), so consumers never need
        # to guess between nested and flat layouts.
        state = {"proprio": state}
        if resolved_cameras:
            metadata["cameras"] = dict(resolved_cameras)
        if depth_warnings:
            metadata["depth_warnings"] = depth_warnings
        if task is not None:
            metadata["task_id"] = task.task_id
        return RobotObservation(
            state=state,
            artifact_refs=tuple(artifact_refs),
            timestamp=self._clock(),
            backend_metadata=metadata,
        )

    @staticmethod
    def _depth_value(depth: Any, *, pixel_x: int, pixel_y: int) -> float:
        try:
            value = depth[pixel_y][pixel_x]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("depth buffer does not match the advertised camera size") from exc
        if hasattr(value, "item") and callable(value.item):
            value = value.item()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) != 1:
                raise ValueError("depth pixels must be scalar or single-channel")
            value = value[0]
        depth_m = float(value)
        if not math.isfinite(depth_m) or depth_m <= 0:
            raise ValueError(f"invalid metric depth {depth_m!r} at pixel ({pixel_x}, {pixel_y})")
        return depth_m

    def _update_depth_maps(
        self,
        raw: Mapping[str, Any],
        *,
        environment: Any,
        cameras: Mapping[str, Any],
    ) -> list[str]:
        """Refresh the backend-private depth maps, degrading instead of raising.

        Depth is an opt-in grounding side channel. A conversion fault must cost
        the episode its ``back_project`` support for that camera, never the
        camera images, the proprio state, or the terminal transition that the
        observation carries, so failures are reported and the camera is simply
        left ungrounded.
        """

        self._depth_maps = {}
        if not self._enable_depth:
            return []
        maps: dict[str, Any] = {}
        warnings: list[str] = []
        for key, value in raw.items():
            name = str(key)
            if not name.endswith("_depth") or value is None:
                continue
            base = name[: -len("_depth")]
            camera = next(
                (
                    candidate
                    for candidate in (f"{base}_image", f"{base}_rgb", base)
                    if candidate in cameras
                ),
                None,
            )
            if camera is None:
                continue
            if environment is None:
                warnings.append(
                    f"{camera}: RoboCasa environment is unavailable for depth conversion"
                )
                continue
            try:
                maps[camera] = self._depth_converter(environment, value)
            except Exception as exc:  # noqa: BLE001 - optional grounding data degrades
                warnings.append(f"{camera}: {type(exc).__name__}: {exc}")
        self._depth_maps = maps
        for warning in warnings:
            logger.warning("RoboCasa depth unavailable (%s)", warning)
        return warnings

    def _convert_observation_after_step(
        self, raw: Any, *, task: RobotTask | None
    ) -> RobotObservation:
        """Publish the post-step state even when optional image storage fails."""

        try:
            return self._convert_observation(raw, task=task)
        except Exception as exc:
            if not isinstance(raw, Mapping):
                raise
            # A failed artifact encoder must not roll back the simulator step
            # or leave observe() returning the pre-step frame.  Preserve all
            # non-image state and annotate the degraded observation.
            state: dict[str, Any] = {}
            dropped: list[str] = []
            for key, value in raw.items():
                name = str(key)
                if name.endswith("_image") or name.endswith("_rgb") or name.endswith("_depth"):
                    continue
                try:
                    state[name] = _json_value(value, path=f"$.state.{name}")
                except Exception:
                    dropped.append(name)
            metadata: dict[str, Any] = {
                "backend": "robocasa",
                "artifact_roles": {},
                "observation_warning": (
                    f"observation conversion failed: {type(exc).__name__}: {exc}"
                ),
            }
            if self._cameras:
                metadata["cameras"] = self._cameras
            if task is not None:
                metadata["task_id"] = task.task_id
            if dropped:
                metadata["dropped_state_fields"] = dropped
            return RobotObservation(
                state={"proprio": state},
                artifact_refs=(),
                timestamp=self._clock(),
                backend_metadata=metadata,
            )

    @staticmethod
    def _transition_info(raw_info: Mapping[str, Any], reward: Any) -> dict[str, Any]:
        """Preserve a transition while dropping non-canonical auxiliary data."""

        info: dict[str, Any] = {}
        warnings: list[str] = []
        for key, value in raw_info.items():
            name = str(key)
            try:
                info[name] = _json_value(value, path=f"$.info.{name}")
            except Exception as exc:
                warnings.append(f"info.{name}: {type(exc).__name__}: {exc}")
        try:
            info["reward"] = _json_value(reward, path="$.reward")
        except Exception as exc:
            warnings.append(f"reward: {type(exc).__name__}: {exc}")
        if warnings:
            info["transition_warnings"] = warnings
        return info

    def _action_values(self, action: RobotAction) -> tuple[float, ...]:
        raw = action.arguments.get("values")
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise ValueError("RoboCasa actions require a numeric arguments.values array")
        values: list[float] = []
        for index, value in enumerate(raw):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"RoboCasa action value {index} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"RoboCasa action value {index} must be finite")
            values.append(number)
        if not values:
            raise ValueError("RoboCasa action values must not be empty")
        if self._action_dim is None:
            raise RuntimeError("RoboCasa action dimension is unavailable; reset must be called")
        if len(values) != self._action_dim:
            raise ValueError(
                "RoboCasa action values have the wrong dimension: "
                f"expected {self._action_dim}, got {len(values)}"
            )
        return tuple(values)

    @staticmethod
    def _positive_limit(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError("RoboCasa max_episode_steps must be a positive integer")
        return int(value)

    def _close_environment(self) -> None:
        environment = self._environment
        self._environment = None
        if environment is not None:
            environment.close()

    @staticmethod
    def _quiet_close(environment: RoboCasaEnvironment) -> None:
        """Release the simulator on a failure path without replacing the cause.

        A MuJoCo teardown fault raised here would surface instead of the reset
        or step failure that actually ended the episode, leaving the caller
        chasing the wrong error.
        """

        try:
            environment.close()
        except Exception:
            logger.warning("closing the RoboCasa environment failed", exc_info=True)

    def _invalidate_episode(self) -> None:
        """Discard the episode after a failure and release the environment."""

        self._task = None
        self._observation = None
        self._cameras = {}
        self._depth_maps = {}
        self._terminated = False
        self._truncated = False
        self._success = None
        self._termination_reason = None
        self._steps_used = 0
        self._max_episode_steps = None
        self._action_dim = None
        environment, self._environment = self._environment, None
        if environment is not None:
            self._quiet_close(environment)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("RoboCasa backend is closed")

    def _require_active(self, operation: str) -> None:
        self._require_open()
        if self._environment is None or self._task is None:
            raise RuntimeError(f"reset must be called before {operation}")


__all__ = ["RoboCasaBackend", "RoboCasaEnvironment", "create_robocasa_environment"]
