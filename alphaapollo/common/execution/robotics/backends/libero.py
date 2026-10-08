# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""In-process LIBERO backend driving the official simulator directly.

Tasks arrive through the prepared-dataset private payload produced by
``alphaapollo.data_preprocess.robotics.prepare_libero`` (#260) or
``prepare_liberopro`` (#271): the payload's task-spec fields land in
:attr:`RobotTask.backend_metadata` and identify one task inside one suite. The
backend owns the simulator lifecycle and remains the only authority on episode
termination and success.

LIBERO-Pro is a rename fork of LIBERO: it ships the same ``Benchmark`` registry
API, the same ``Task`` record, and the same ``OffScreenRenderEnv`` contract
under a ``liberopro.liberopro`` import root, so one backend drives both. What
it does not share is the *content* behind a suite name — the fork's registry
also registers ``libero_spatial``, ``libero_object``, ``libero_goal``,
``libero_90`` and ``libero_10``, whose tasks carry the same names, BDDL
filenames and initial-state filenames as the official ones while resolving to
its own asset tree. Choosing the package by anything looser than the prepared
``benchmark`` label would therefore run a different task under a name that
still cross-checks. :data:`LIBERO_DISTRIBUTIONS` is the single authority that
binds a benchmark label to its import root, its suites and its declared Python
distribution, and every reset resolves exactly one of its records.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from alphaapollo.common.artifacts.schemas import ArtifactRef
from alphaapollo.common.execution.robotics.backends.recording import (
    EpisodeRecorder,
    VideoSinkFactory,
    open_mp4_sink,
)
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
SuiteLoader = Callable[[str, int], Any]
EnvFactory = Callable[..., Any]

LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
)
# The 16 static perturbation suites rpent-liberopro registers, formed by
# crossing four base suites with four perturbations. Kept in step with
# ``prepare_liberopro.LIBEROPRO_SUITES`` by a registry test; the two layers do
# not import each other.
LIBEROPRO_SUITES = tuple(
    f"{base}_{perturbation}"
    for base in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    for perturbation in ("swap", "task", "lan", "object")
)


@dataclass(frozen=True)
class LiberoDistribution:
    """One prepared ``benchmark`` label bound to the package that executes it.

    ``distribution`` is the PyPI distribution the preparer pins, or ``None``
    when the benchmark runs from an official checkout that is not published as
    a wheel. It is what makes the prepared ``package_distribution`` and
    ``package_version`` fields load-bearing at reset.
    """

    benchmark: str
    import_root: str
    distribution: str | None
    suites: tuple[str, ...]


LIBERO_DISTRIBUTIONS = (
    LiberoDistribution(
        benchmark="libero",
        import_root="libero.libero",
        distribution=None,
        suites=LIBERO_SUITES,
    ),
    LiberoDistribution(
        benchmark="liberopro",
        import_root="liberopro.liberopro",
        distribution="rpent-liberopro",
        suites=LIBEROPRO_SUITES,
    ),
)


def _index_distributions() -> tuple[dict[str, LiberoDistribution], dict[str, LiberoDistribution]]:
    """Build the benchmark and suite lookups, refusing any ambiguous name.

    A suite that appeared under two benchmarks would make the package choice
    depend on which lookup ran first, which is the one failure this module
    exists to prevent. Reject it at import instead.
    """

    by_benchmark: dict[str, LiberoDistribution] = {}
    by_suite: dict[str, LiberoDistribution] = {}
    for record in LIBERO_DISTRIBUTIONS:
        if record.benchmark in by_benchmark:
            raise RuntimeError(f"duplicate LIBERO benchmark label {record.benchmark!r}")
        by_benchmark[record.benchmark] = record
        for suite in record.suites:
            owner = by_suite.get(suite)
            if owner is not None:
                raise RuntimeError(
                    f"LIBERO suite {suite!r} is claimed by both benchmark "
                    f"{owner.benchmark!r} and {record.benchmark!r}"
                )
            by_suite[suite] = record
    return by_benchmark, by_suite


_DISTRIBUTION_BY_BENCHMARK, _DISTRIBUTION_BY_SUITE = _index_distributions()

ACTION_DIM = 7

_IMAGE_ROLES = {
    "agentview_image": "fixed_camera",
    "robot0_eye_in_hand_image": "wrist_camera",
}
_DEPTH_ROLES = {
    "agentview_depth": "fixed_camera",
    "robot0_eye_in_hand_depth": "wrist_camera",
}
_PROPRIO_KEYS = (
    "robot0_joint_pos",
    "robot0_gripper_qpos",
    "robot0_eef_pos",
    "robot0_eef_quat",
)
_REQUIRED_SPEC_FIELDS = (
    "suite",
    "task_order_index",
    "task_index",
    "task_name",
    "problem_folder",
    "bddl_file",
    "init_states_file",
    "initial_state_index",
)
_OPTIONAL_SPEC_STRINGS = ("demonstration_id", "demonstration_path")
_IMAGE_ORIENTATIONS = ("raw", "upright", "rotate_180")
# The official LIBERO README resets the task and executes ten 7-D dummy
# actions before evaluation. Keep that protocol as the backend default.
_SETTLE_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _encode_png(image: Any) -> bytes:
    """Encode an RGB observation without importing robotics extras at module load."""

    try:
        import imageio.v3 as iio
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO image encoding requires the optional robotics dependencies"
        ) from exc
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"LIBERO RGB image must have HxWx3 shape, got {array.shape}")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return iio.imwrite("<bytes>", array, extension=".png")


def _distribution_for_suite(suite: str) -> LiberoDistribution:
    """Resolve the package that owns one suite name.

    Both defaults below take a suite name rather than a benchmark label, so
    this lookup is the point at which they choose a package. It is safe only
    because suite names are unique across :data:`LIBERO_DISTRIBUTIONS`, which
    :func:`_index_distributions` enforces at import, and because
    :func:`_task_spec` has already refused any suite that does not belong to
    the benchmark the task declared.
    """

    record = _DISTRIBUTION_BY_SUITE.get(suite)
    if record is None:
        raise ValueError(
            f"LIBERO suite {suite!r} belongs to no supported benchmark; expected one of "
            f"{sorted(_DISTRIBUTION_BY_SUITE)}"
        )
    return record


def _import_distribution(distribution: LiberoDistribution, submodule: str | None = None) -> Any:
    name = distribution.import_root
    if submodule is not None:
        name = f"{name}.{submodule}"
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        requirement = distribution.distribution or distribution.import_root.split(".", 1)[0]
        raise RuntimeError(
            f"the in-process LIBERO backend requires the optional {requirement!r} dependency "
            f"to run benchmark {distribution.benchmark!r}"
        ) from exc


def _default_suite_loader(suite: str, task_order_index: int) -> Any:
    """Instantiate the benchmark suite for one task ordering, from its own package."""

    distribution = _distribution_for_suite(suite)
    module = _import_distribution(distribution, "benchmark")
    get_benchmark_dict = getattr(module, "get_benchmark_dict", None)
    if not callable(get_benchmark_dict):
        raise RuntimeError(
            f"{module.__name__} exposes no get_benchmark_dict API for benchmark "
            f"{distribution.benchmark!r}"
        )
    benchmarks = get_benchmark_dict()
    if suite not in benchmarks:
        raise ValueError(
            f"LIBERO suite {suite!r} is not registered by {module.__name__} for benchmark "
            f"{distribution.benchmark!r}"
        )
    return benchmarks[suite](task_order_index)


def _default_env_factory(
    problem_folder: str,
    bddl_file: str,
    *,
    camera_height: int,
    camera_width: int,
    camera_depths: bool = False,
) -> Any:
    """Open an offscreen environment for one BDDL task file, from its own package.

    Both LIBERO and LIBERO-Pro build every ``Task`` with
    ``problem_folder=<suite name>``, so the folder names the owning package as
    well as the directory under its BDDL root. :func:`_task_spec` refuses any
    prepared task whose ``problem_folder`` is not a suite of its own benchmark,
    which is what keeps the registry and the BDDL root the same package.
    """

    distribution = _distribution_for_suite(problem_folder)
    package = _import_distribution(distribution)
    envs = _import_distribution(distribution, "envs")
    get_libero_path = getattr(package, "get_libero_path", None)
    off_screen_render_env = getattr(envs, "OffScreenRenderEnv", None)
    if not callable(get_libero_path) or not callable(off_screen_render_env):
        raise RuntimeError(
            f"{distribution.import_root} exposes no get_libero_path/OffScreenRenderEnv API "
            f"for benchmark {distribution.benchmark!r}"
        )
    bddl_path = os.path.join(get_libero_path("bddl_files"), problem_folder, bddl_file)
    if not os.path.isfile(bddl_path):
        raise FileNotFoundError(f"LIBERO BDDL file does not exist: {bddl_path}")
    return off_screen_render_env(
        bddl_file_name=bddl_path,
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_depths=camera_depths,
    )


def _default_depth_converter(env: Any, depth: Any) -> Any:
    """Convert MuJoCo's normalized depth buffer into metric z-depth."""

    try:
        from robosuite.utils import camera_utils
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO depth conversion requires the optional robosuite dependency"
        ) from exc
    sim = getattr(env, "sim", None)
    if sim is None:
        sim = getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        raise RuntimeError("LIBERO environment exposes no simulator for depth conversion")
    return camera_utils.get_real_depth_map(sim, depth)


def _resolve_success_oracle(env: Any) -> Callable[[], Any]:
    """Find the env's ``check_success`` oracle, walking one wrapper level.

    Fails closed: robosuite's ``done`` is also True on horizon timeout, so an
    env without the oracle cannot distinguish success from running out of
    time — falling back to ``done`` would fabricate positives into the
    reward signal.
    """

    for candidate in (env, getattr(env, "env", None)):
        if candidate is None:
            continue
        check = getattr(candidate, "check_success", None)
        if callable(check):
            return check
    raise RuntimeError(
        "LIBERO env exposes no check_success oracle (checked env and env.env); "
        "refusing to run with a fail-open success signal"
    )


def _camera_metadata(
    env: Any,
    *,
    names: Sequence[tuple[str, str]],
    height: int,
    width: int,
) -> dict[str, Any]:
    """Best-effort intrinsics/extrinsics per camera role for geometric tools.

    Absent metadata is not an error: simulators without a reachable ``sim``
    handle (or older robosuite APIs) simply omit it, and inspection tools
    report it as unavailable.
    """

    try:
        from robosuite.utils import camera_utils
    except ImportError:
        return {}
    sim = getattr(env, "sim", None)
    if sim is None:
        sim = getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        return {}
    cameras: dict[str, Any] = {}
    for camera_name, role in names:
        try:
            intrinsics = camera_utils.get_camera_intrinsic_matrix(sim, camera_name, height, width)
            extrinsics = camera_utils.get_camera_extrinsic_matrix(sim, camera_name)
        except Exception:  # noqa: BLE001 - a missing camera only skips its metadata
            continue
        cameras[role] = {
            "name": camera_name,
            "height": height,
            "width": width,
            "intrinsics": [[float(v) for v in row] for row in intrinsics],
            "extrinsics": [[float(v) for v in row] for row in extrinsics],
        }
    return cameras


def _to_json(value: Any, *, path: str) -> Any:
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    elif hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_json(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item, path=f"{path}[]") for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    raise TypeError(f"{path} has non-JSON type {type(value).__name__}")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"LIBERO {name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"LIBERO {name} must be an integer")
    if value < 0:
        raise ValueError(f"LIBERO {name} must be non-negative")
    return value


def _nonempty_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LIBERO {name} must be a non-empty string")
    return value.strip()


def _package_identity(
    task: RobotTask,
    metadata: Mapping[str, Any],
    distribution: LiberoDistribution,
) -> dict[str, str]:
    """Cross-check the prepared package identity against the one being executed.

    ``package_distribution`` and ``package_version`` are the preparer's own
    record of the wheel it validated the row against. Reading them here is what
    stops a row prepared for one distribution from executing against another,
    and stops ``environment_version`` from drifting from the version it
    encodes. A benchmark that runs from a checkout publishes no distribution,
    so for it both fields must be absent rather than merely unread.
    """

    declared_distribution = metadata.get("package_distribution")
    if declared_distribution is not None:
        declared_distribution = _nonempty_str(declared_distribution, name="package_distribution")
    if declared_distribution != distribution.distribution:
        raise ValueError(
            f"benchmark {task.benchmark!r} executes "
            f"{distribution.distribution or 'an official checkout'}, but the prepared task "
            f"declares package_distribution={declared_distribution!r}"
        )
    declared_version = metadata.get("package_version")
    if declared_version is not None:
        declared_version = _nonempty_str(declared_version, name="package_version")
    if distribution.distribution is None:
        if declared_version is not None:
            raise ValueError(
                f"benchmark {task.benchmark!r} executes an official checkout and cannot accept "
                f"package_version={declared_version!r}"
            )
        return {}
    if declared_version is None:
        raise ValueError(f"benchmark {task.benchmark!r} tasks require package_version")
    expected = f"{distribution.distribution}=={declared_version}"
    if task.environment_version.strip() != expected:
        raise ValueError(
            f"LIBERO environment_version {task.environment_version.strip()!r} does not state the "
            f"prepared package identity {expected!r}"
        )
    return {
        "package_distribution": distribution.distribution,
        "package_version": declared_version,
    }


def _task_spec(task: RobotTask) -> tuple[LiberoDistribution, dict[str, Any]]:
    """Validate the prepared-payload task spec carried in backend_metadata.

    Returns the distribution the task named together with its spec, so the
    caller cannot re-derive the package from anything looser than the label
    this function accepted.
    """

    distribution = _DISTRIBUTION_BY_BENCHMARK.get(task.benchmark.lower())
    if distribution is None:
        raise ValueError(
            f"LiberoBackend cannot reset benchmark {task.benchmark!r}; expected one of "
            f"{sorted(_DISTRIBUTION_BY_BENCHMARK)}"
        )
    if not task.environment_version.strip():
        raise ValueError("LIBERO tasks require a non-empty environment_version")
    metadata = dict(task.backend_metadata)
    missing = [field for field in _REQUIRED_SPEC_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"LIBERO task metadata is missing {missing}")
    suite = _nonempty_str(metadata["suite"], name="suite")
    if suite not in distribution.suites:
        raise ValueError(
            f"benchmark {task.benchmark!r} does not include LIBERO suite {suite!r}; expected one "
            f"of {list(distribution.suites)}"
        )
    problem_folder = _nonempty_str(metadata["problem_folder"], name="problem_folder")
    if problem_folder not in distribution.suites:
        raise ValueError(
            f"benchmark {task.benchmark!r} suite {suite!r} declares problem_folder "
            f"{problem_folder!r}, which is not one of its suites; refusing to read task assets "
            "from another package"
        )
    spec: dict[str, Any] = {
        "suite": suite,
        "task_order_index": _nonnegative_int(metadata["task_order_index"], name="task_order_index"),
        "task_index": _nonnegative_int(metadata["task_index"], name="task_index"),
        "task_name": _nonempty_str(metadata["task_name"], name="task_name"),
        "problem_folder": problem_folder,
        "bddl_file": _nonempty_str(metadata["bddl_file"], name="bddl_file"),
        "init_states_file": _nonempty_str(metadata["init_states_file"], name="init_states_file"),
        "initial_state_index": _nonnegative_int(
            metadata["initial_state_index"], name="initial_state_index"
        ),
        **_package_identity(task, metadata, distribution),
    }
    for field in _OPTIONAL_SPEC_STRINGS:
        if field in metadata and metadata[field] is not None:
            spec[field] = _nonempty_str(metadata[field], name=field)
    if "max_episode_steps" in metadata:
        spec["max_episode_steps"] = _positive_int(
            metadata["max_episode_steps"], name="max_episode_steps"
        )
    return distribution, spec


class LiberoBackend:
    """Execute prepared LIBERO tasks in-process with environment-owned success."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore | None = None,
        camera_height: int = 256,
        camera_width: int = 256,
        max_episode_steps: int | None = None,
        environment_version: str | None = None,
        image_orientation: str = "upright",
        enable_depth: bool = False,
        settle_steps: int = 10,
        record_dir: str | os.PathLike[str] | None = None,
        record_fps: int = 20,
        suite_loader: SuiteLoader = _default_suite_loader,
        env_factory: EnvFactory = _default_env_factory,
        image_encoder: ImageEncoder = _encode_png,
        depth_converter: DepthConverter = _default_depth_converter,
        video_sink_factory: VideoSinkFactory = open_mp4_sink,
        clock: Clock = _utc_now,
    ) -> None:
        if artifact_store is not None and not isinstance(artifact_store, ArtifactStore):
            raise TypeError("artifact_store must be an ArtifactStore")
        self._camera_height = _positive_int(camera_height, name="camera_height")
        self._camera_width = _positive_int(camera_width, name="camera_width")
        self._default_max_episode_steps = (
            None
            if max_episode_steps is None
            else _positive_int(max_episode_steps, name="max_episode_steps")
        )
        if environment_version is not None:
            environment_version = _nonempty_str(environment_version, name="environment_version")
        if image_orientation not in _IMAGE_ORIENTATIONS:
            raise ValueError(
                f"image_orientation must be one of {list(_IMAGE_ORIENTATIONS)}, "
                f"got {image_orientation!r}"
            )
        if not isinstance(enable_depth, bool):
            raise TypeError("enable_depth must be a bool")
        self._settle_steps = _nonnegative_int(settle_steps, name="settle_steps")
        if not callable(suite_loader):
            raise TypeError("suite_loader must be callable")
        if not callable(env_factory):
            raise TypeError("env_factory must be callable")
        if not callable(image_encoder):
            raise TypeError("image_encoder must be callable")
        if not callable(depth_converter):
            raise TypeError("depth_converter must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._artifact_store = artifact_store
        self._environment_version = environment_version
        self._image_orientation = image_orientation
        self._enable_depth = enable_depth
        self._recorder = (
            None
            if record_dir is None
            else EpisodeRecorder(
                record_dir, fps=record_fps, video_sink_factory=video_sink_factory, clock=clock
            )
        )
        self._suite_loader = suite_loader
        self._env_factory = env_factory
        self._image_encoder = image_encoder
        self._depth_converter = depth_converter
        self._clock = clock
        self._env: Any = None
        self._success_oracle: Callable[[], Any] | None = None
        self._task: RobotTask | None = None
        self._observation: RobotObservation | None = None
        self._environment_metadata: dict[str, Any] | None = None
        self._depth_maps: dict[str, Any] = {}
        self._terminated = False
        self._truncated = False
        self._success: bool | None = None
        self._termination_reason: str | None = None
        self._steps_used = 0
        self._max_episode_steps: int | None = None
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
        # A failed re-reset must never leave the previous episode usable.
        # Invalidate before task-spec/version/suite validation because all of
        # those checks are fallible too.
        self._invalidate_episode()
        if not isinstance(task, RobotTask):
            raise TypeError("task must be a RobotTask")
        distribution, spec = _task_spec(task)
        self._validate_environment_version(task)

        suite = self._suite_loader(spec["suite"], spec["task_order_index"])
        suite_task = suite.get_task(spec["task_index"])
        self._validate_suite_task(spec, suite_task, instruction=task.instruction)
        init_states = suite.get_task_init_states(spec["task_index"])
        if spec["initial_state_index"] >= len(init_states):
            raise ValueError(
                f"LIBERO initial_state_index {spec['initial_state_index']} is out of "
                f"range for {len(init_states)} prepared initial states"
            )

        env_options = {
            "camera_height": self._camera_height,
            "camera_width": self._camera_width,
        }
        if self._enable_depth:
            env_options["camera_depths"] = True
        env = self._env_factory(
            spec["problem_folder"],
            spec["bddl_file"],
            **env_options,
        )
        # Build everything fallible before committing any state, so a failure
        # closes the env and leaves the backend cleanly inactive.
        try:
            env.reset()
            raw = env.set_init_state(init_states[spec["initial_state_index"]])
            success_oracle = _resolve_success_oracle(env)
            initial_success = bool(success_oracle())
            if initial_success:
                raise RuntimeError(
                    "LIBERO initial state is already successful; refusing to score a task "
                    "the policy did not solve"
                )
            settle_steps_used = 0
            for _ in range(self._settle_steps):
                raw, _, done, _ = env.step(list(_SETTLE_ACTION))
                settle_steps_used += 1
                if bool(success_oracle()):
                    raise RuntimeError(
                        "LIBERO settle step reached task success before policy execution"
                    )
                if bool(done):
                    raise RuntimeError("LIBERO settle step terminated before task success")
            cameras = _camera_metadata(
                env,
                names=(("agentview", "fixed_camera"), ("robot0_eye_in_hand", "wrist_camera")),
                height=self._camera_height,
                width=self._camera_width,
            )
            environment_metadata = {
                "backend": "libero",
                "benchmark": task.benchmark,
                "package_distribution": distribution.distribution,
                "import_root": distribution.import_root,
                "task_id": task.task_id,
                "backend_metadata": dict(task.backend_metadata),
                "suite": spec["suite"],
                "task_order_index": spec["task_order_index"],
                "task_index": spec["task_index"],
                "task_name": spec["task_name"],
                "problem_folder": spec["problem_folder"],
                "bddl_file": spec["bddl_file"],
                "initial_state_index": spec["initial_state_index"],
                "environment_version": task.environment_version,
                "camera": {"height": self._camera_height, "width": self._camera_width},
                # Official reset settling is environment initialization, not a
                # policy action, and therefore does not consume the episode's
                # policy step budget.
                "settle_steps_used": settle_steps_used,
                "settle_steps_count_toward_policy_budget": False,
            }
            if cameras:
                environment_metadata["cameras"] = cameras
            observation = self._convert_observation(
                raw,
                task=task,
                environment_metadata=environment_metadata,
                steps_used=0,
                environment=env,
            )
        except BaseException:
            self._depth_maps = {}
            self._quiet_close(env)
            raise

        self._env = env
        self._success_oracle = success_oracle
        self._task = task
        self._environment_metadata = environment_metadata
        self._terminated = False
        self._truncated = False
        self._success = None
        self._termination_reason = None
        self._steps_used = 0
        self._max_episode_steps = spec.get("max_episode_steps", self._default_max_episode_steps)
        self._observation = observation
        if self._recorder is not None:
            try:
                self._recorder.start(
                    {
                        "task_id": task.task_id,
                        "instruction": task.instruction,
                        "environment": environment_metadata,
                    }
                )
                self._record_step(raw, action=None, reward=None)
            except BaseException:
                # Recording is fail-loud, but a recorder fault must not leave
                # a half-active backend behind the raised error.
                self._recorder.abort(reason="reset_failed")
                self._invalidate_episode()
                raise
        return RobotResetResult(
            observation=self._observation,
            info={
                "environment": environment_metadata,
                "initial_states_available": len(init_states),
                "initial_success": False,
                "settle_steps_used": settle_steps_used,
                "settle_steps_count_toward_policy_budget": False,
            },
        )

    def observe(self) -> RobotObservation:
        self._require_active("observe")
        observation = self._observation
        if observation is None:
            raise RuntimeError("reset must be called before observe")
        return observation

    def back_project(self, *, camera: str, pixel_x: int, pixel_y: int) -> RobotGroundedPoint:
        """Ground one displayed-image pixel against the current metric depth."""

        self._require_active("back_project")
        if not self._enable_depth:
            raise RuntimeError("LIBERO metric depth is disabled; set enable_depth=true")
        depth = self._depth_maps.get(camera)
        if depth is None:
            raise RuntimeError(f"current LIBERO observation has no depth for camera {camera!r}")
        metadata = self._environment_metadata or {}
        cameras = metadata.get("cameras", {})
        camera_meta = cameras.get(camera) if isinstance(cameras, Mapping) else None
        if not isinstance(camera_meta, Mapping):
            raise ValueError(f"unknown camera {camera!r}; available: {sorted(self._depth_maps)}")
        width = int(camera_meta["width"])
        height = int(camera_meta["height"])
        if not 0 <= pixel_x < width or not 0 <= pixel_y < height:
            raise ValueError(
                f"pixel ({pixel_x}, {pixel_y}) is outside camera {camera!r} image {width}x{height}"
            )
        raw_x, raw_y = self._raw_pixel(pixel_x, pixel_y, width=width, height=height)
        depth_m = self._depth_value(depth, pixel_x=raw_x, pixel_y=raw_y)
        observation = self.observe()
        return ground_metric_depth(
            camera=camera,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            # The MuJoCo buffers are bottom-up, so the depth sample is indexed
            # by the raw row.  robosuite's intrinsics pair with an extrinsic
            # whose y axis points down the *upright* image, so calibration
            # counts rows from the top instead.  The two rows are mirror
            # images and using one for both puts the point on the wrong side
            # of the optical axis.
            calibration_pixel_x=raw_x,
            calibration_pixel_y=height - 1 - raw_y,
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
            raise RuntimeError("cannot execute an action after the LIBERO episode is terminal")
        rows = self._action_rows(action)
        env = self._env
        task = self._task
        if env is None or task is None:
            # Ordinary error path, not an invariant: assert would vanish
            # under python -O.
            raise RuntimeError("reset must be called before execute")

        environment_metadata = self._environment_metadata
        if environment_metadata is None:
            raise RuntimeError("reset must be called before execute")
        executed = 0
        transition: RobotTransition | None = None
        for row in rows:
            try:
                raw, reward, done, _ = env.step(list(row))
                next_steps_used = self._steps_used + 1
                executed += 1
                reward_value, info = self._transition_info(reward)
                # The oracle was resolved fail-closed at reset; env-signaled
                # done without oracle success is truncation, never success.
                succeeded = bool(self._success_oracle())
                limit_reached = (
                    self._max_episode_steps is not None
                    and next_steps_used >= self._max_episode_steps
                )
                terminated = succeeded
                truncated = not succeeded and (bool(done) or limit_reached)
                success = succeeded if terminated or truncated else None
                if terminated:
                    termination_reason = "task_success"
                elif limit_reached:
                    termination_reason = "time_limit"
                elif bool(done):
                    termination_reason = "environment_done"
                else:
                    termination_reason = None
                observation = self._convert_observation_after_step(
                    raw,
                    task=task,
                    environment_metadata=environment_metadata,
                    steps_used=next_steps_used,
                )
                transition = RobotTransition(
                    observation=observation,
                    steps_used=executed,
                    terminated=terminated,
                    truncated=truncated,
                    success=success,
                    termination_reason=termination_reason,
                    info=info,
                )
            except BaseException as exc:
                if self._recorder is not None:
                    self._recorder.abort(reason="step_failed", error=exc)
                self._invalidate_episode()
                raise

            self._steps_used = next_steps_used
            self._terminated = terminated
            self._truncated = truncated
            self._success = success
            self._termination_reason = termination_reason
            self._observation = observation
            self._record_step_safely(raw, action=row, reward=reward_value)
            if terminated or truncated:
                break

        if transition is None:
            raise RuntimeError("LIBERO action contained no executable rows")
        if self._recorder is not None and (self._terminated or self._truncated):
            try:
                self._recorder.finish(reason=self._termination_reason or "terminal")
            except Exception:
                # The simulator transition and observation are authoritative;
                # a video/trace flush failure is recorded by EpisodeRecorder
                # but must not discard a successful terminal transition.
                pass
        return transition

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._recorder is not None:
                self._recorder.finish(reason="backend_closed")
        finally:
            # A recorder/video failure must not leak the MuJoCo env: close()
            # is unrepeatable once _closed is set.
            self._close_env()

    def _record_step(
        self, raw: Any, *, action: Sequence[float] | None, reward: float | None
    ) -> None:
        if self._recorder is None or not self._recorder.active:
            return
        proprio = None
        frame = None
        if isinstance(raw, Mapping):
            proprio = {
                key: _to_json(raw[key], path=f"$.record.{key}")
                for key in _PROPRIO_KEYS
                if key in raw
            }
            image = raw.get("agentview_image")
            if (
                image is not None
                and getattr(image, "size", None) != 0
                and not (isinstance(image, Sequence) and not image)
            ):
                frame = self._orient(image)
        self._recorder.record_step(
            step=self._steps_used,
            action=action,
            reward=reward,
            terminated=self._terminated,
            truncated=self._truncated,
            success=self._success,
            proprio=proprio,
            frame=frame,
        )

    def _record_step_safely(
        self, raw: Any, *, action: Sequence[float] | None, reward: float | None
    ) -> None:
        """Keep recorder failures out of the authoritative transition path."""

        if self._recorder is None or not self._recorder.active:
            return
        try:
            self._record_step(raw, action=action, reward=reward)
        except Exception as exc:
            self._recorder.abort(reason="recording_failed", error=exc)

    def _validate_environment_version(self, task: RobotTask) -> None:
        if (
            self._environment_version is not None
            and task.environment_version.strip() != self._environment_version
        ):
            raise ValueError(
                "LIBERO environment_version mismatch: the prepared task requires "
                f"{task.environment_version!r}, this backend serves "
                f"{self._environment_version!r}"
            )

    @staticmethod
    def _validate_suite_task(spec: Mapping[str, Any], suite_task: Any, *, instruction: str) -> None:
        # Fail closed on absent attributes too: a suite task that cannot be
        # cross-checked is a task that cannot be verified as the prepared one.
        for field, attribute in (
            ("task_name", "name"),
            ("problem_folder", "problem_folder"),
            ("bddl_file", "bddl_file"),
            ("init_states_file", "init_states_file"),
        ):
            actual = getattr(suite_task, attribute, None)
            if actual is None:
                raise ValueError(
                    f"LIBERO suite task does not expose {attribute!r}; "
                    f"cannot verify the prepared {field} — refusing to run"
                )
            if actual != spec[field]:
                raise ValueError(
                    "LIBERO task spec mismatch: prepared "
                    f"{field}={spec[field]!r}, suite has {actual!r}; "
                    "refusing to run a different task"
                )
        language = getattr(suite_task, "language", None)
        if not isinstance(language, str) or not language.strip():
            raise ValueError(
                "LIBERO suite task does not expose its language; "
                "cannot verify the prepared instruction — refusing to run"
            )
        if language.strip() != instruction.strip():
            raise ValueError(
                "LIBERO task language does not match the prepared instruction; "
                "refusing to run a different task"
            )

    def _convert_observation(
        self,
        raw: Any,
        *,
        task: RobotTask,
        environment_metadata: Mapping[str, Any],
        steps_used: int,
        include_images: bool = True,
        environment: Any | None = None,
    ) -> RobotObservation:
        if not isinstance(raw, Mapping):
            raise TypeError("LIBERO observation must be a mapping")
        depth_warnings = self._update_depth_maps(raw, environment=environment or self._env)
        artifact_refs: list[ArtifactRef] = []
        artifact_roles: dict[str, str] = {}
        for source, role in _IMAGE_ROLES.items():
            if not include_images:
                continue
            image = raw.get(source)
            if image is None or getattr(image, "size", None) == 0:
                continue
            if isinstance(image, Sequence) and not image:
                continue
            if self._artifact_store is None:
                raise RuntimeError(
                    f"LIBERO observation contains {source!r}, but no artifact_store was configured"
                )
            encoded = self._image_encoder(self._orient(image))
            if not isinstance(encoded, bytes):
                raise TypeError("image_encoder must return bytes")
            artifact = self._artifact_store.put(
                encoded,
                type_="image/png",
                created_by="libero-backend",
            )
            ref = self._artifact_store.ref(artifact)
            artifact_refs.append(ref)
            artifact_roles[role] = ref.id
        proprio = {
            key: _to_json(raw[key], path=f"$.state.{key}") for key in _PROPRIO_KEYS if key in raw
        }
        metadata: dict[str, Any] = {
            "backend": "libero",
            "task_id": task.task_id,
            "environment": dict(environment_metadata),
            "artifact_roles": artifact_roles,
            "steps_used": steps_used,
        }
        if depth_warnings:
            metadata["depth_warnings"] = depth_warnings
        return RobotObservation(
            state={"proprio": proprio},
            artifact_refs=tuple(artifact_refs),
            timestamp=self._clock(),
            backend_metadata=metadata,
        )

    def _convert_observation_after_step(
        self,
        raw: Any,
        *,
        task: RobotTask,
        environment_metadata: Mapping[str, Any],
        steps_used: int,
    ) -> RobotObservation:
        """Publish simulator state even when optional image encoding fails."""

        try:
            return self._convert_observation(
                raw,
                task=task,
                environment_metadata=environment_metadata,
                steps_used=steps_used,
            )
        except Exception as exc:
            if not isinstance(raw, Mapping):
                raise
            proprio: dict[str, Any] = {}
            dropped: list[str] = []
            for key in _PROPRIO_KEYS:
                if key not in raw:
                    continue
                try:
                    proprio[key] = _to_json(raw[key], path=f"$.state.{key}")
                except Exception:
                    dropped.append(key)
            metadata: dict[str, Any] = {
                "backend": "libero",
                "task_id": task.task_id,
                "environment": dict(environment_metadata),
                "artifact_roles": {},
                "steps_used": steps_used,
                "observation_warning": (
                    f"observation conversion failed: {type(exc).__name__}: {exc}"
                ),
            }
            if dropped:
                metadata["dropped_state_fields"] = dropped
            return RobotObservation(
                state={"proprio": proprio},
                artifact_refs=(),
                timestamp=self._clock(),
                backend_metadata=metadata,
            )

    @staticmethod
    def _transition_info(reward: Any) -> tuple[float | None, dict[str, Any]]:
        if reward is None:
            return None, {}
        try:
            value = float(reward)
            encoded = _to_json(value, path="$.reward")
        except Exception as exc:
            return None, {"transition_warnings": [f"reward: {type(exc).__name__}: {exc}"]}
        return value, {"reward": encoded}

    def _orient(self, image: Any) -> Any:
        if self._image_orientation == "raw":
            return image
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "LIBERO image orientation requires the optional robotics dependencies"
            ) from exc
        array = np.asarray(image)
        # MuJoCo offscreen buffers render bottom-up; "upright" restores the
        # human-upright view and "rotate_180" additionally mirrors horizontally
        # for VLA providers trained on that convention.
        if self._image_orientation == "upright":
            return array[::-1]
        return array[::-1, ::-1]

    def _raw_pixel(self, pixel_x: int, pixel_y: int, *, width: int, height: int) -> tuple[int, int]:
        if self._image_orientation == "raw":
            return pixel_x, pixel_y
        if self._image_orientation == "upright":
            return pixel_x, height - 1 - pixel_y
        return width - 1 - pixel_x, height - 1 - pixel_y

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

    def _update_depth_maps(self, raw: Mapping[str, Any], *, environment: Any) -> list[str]:
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
        for source, role in _DEPTH_ROLES.items():
            value = raw.get(source)
            if value is None or getattr(value, "size", None) == 0:
                continue
            if environment is None:
                warnings.append(f"{role}: LIBERO environment is unavailable for depth conversion")
                continue
            try:
                maps[role] = self._depth_converter(environment, value)
            except Exception as exc:  # noqa: BLE001 - optional grounding data degrades
                warnings.append(f"{role}: {type(exc).__name__}: {exc}")
        self._depth_maps = maps
        for warning in warnings:
            logger.warning("LIBERO depth unavailable (%s)", warning)
        return warnings

    @staticmethod
    def _action_rows(action: RobotAction) -> tuple[tuple[float, ...], ...]:
        raw = action.arguments.get("values")
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise ValueError("LIBERO actions require a numeric arguments.values array")
        if not raw:
            raise ValueError("LIBERO action values must not be empty")
        nested = all(
            isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
            for item in raw
        )
        rows = raw if nested else (raw,)
        converted: list[tuple[float, ...]] = []
        for row_index, row in enumerate(rows):
            values: list[float] = []
            for index, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"LIBERO action value [{row_index}][{index}] must be numeric")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"LIBERO action value [{row_index}][{index}] must be finite")
                values.append(number)
            if len(values) != ACTION_DIM:
                raise ValueError(
                    f"LIBERO actions require {ACTION_DIM} values per step, "
                    f"got {len(values)} at row {row_index}"
                )
            converted.append(tuple(values))
        return tuple(converted)

    def _close_env(self) -> None:
        self._success_oracle = None
        env, self._env = self._env, None
        if env is not None:
            self._safe_close(env)

    def _invalidate_episode(self) -> None:
        """Discard the episode after a failure and release the environment."""

        self._task = None
        self._observation = None
        self._environment_metadata = None
        self._depth_maps = {}
        self._terminated = False
        self._truncated = False
        self._success = None
        self._termination_reason = None
        self._steps_used = 0
        self._max_episode_steps = None
        self._success_oracle = None
        env, self._env = self._env, None
        if env is not None:
            self._quiet_close(env)

    @staticmethod
    def _safe_close(env: Any) -> None:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _quiet_close(env: Any) -> None:
        """Release the simulator on a failure path without replacing the cause.

        A MuJoCo teardown fault raised here would surface instead of the reset
        or step failure that actually ended the episode, leaving the caller
        chasing the wrong error.
        """

        try:
            LiberoBackend._safe_close(env)
        except Exception:
            logger.warning("closing the LIBERO environment failed", exc_info=True)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("LIBERO backend is closed")

    def _require_active(self, operation: str) -> None:
        self._require_open()
        if self._task is None or self._env is None:
            raise RuntimeError(f"reset must be called before {operation}")


__all__ = [
    "ACTION_DIM",
    "LIBERO_DISTRIBUTIONS",
    "LIBEROPRO_SUITES",
    "LIBERO_SUITES",
    "LiberoBackend",
    "LiberoDistribution",
]
