# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Transport-independent records exchanged with robot execution backends."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from alphaapollo.common.artifacts.schemas import ArtifactRef, require_json_value


def _non_empty(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _json_mapping(value: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    copied = deepcopy(dict(value))
    require_json_value(copied, path=path)
    return copied


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        # Tuples must recurse too: a tuple of dicts would otherwise keep
        # mutable members inside a "frozen" record.
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_json_value(value: Any, *, path: str) -> Any:
    copied = deepcopy(value)
    require_json_value(copied, path=path)
    return _freeze_json(copied)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return deepcopy(value)


def _freeze_mapping(value: Mapping[str, Any], *, path: str) -> Mapping[str, Any]:
    return _freeze_json(_json_mapping(value, path=path))


def _freeze_refs(
    values: Sequence[ArtifactRef | Mapping[str, Any]],
) -> tuple[ArtifactRef, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("$.artifact_refs must be a sequence of artifact references")
    refs: list[ArtifactRef] = []
    for index, value in enumerate(values):
        try:
            ref = (
                value.model_copy(deep=True)
                if isinstance(value, ArtifactRef)
                else ArtifactRef.model_validate(value)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"$.artifact_refs[{index}] is invalid: {exc}") from exc
        refs.append(ref)
    return tuple(refs)


@dataclass(frozen=True, slots=True)
class RobotTask:
    """One prepared benchmark task passed to a robot backend."""

    task_id: str
    benchmark: str
    instruction: str
    # Required and non-empty like the other identity fields: #260 makes the
    # environment version a build-time input every backend validates against,
    # so an empty value only defers the failure to reset time.
    environment_version: str
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("task_id", "benchmark", "instruction", "environment_version"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), path=f"$.{name}"))
        object.__setattr__(
            self,
            "backend_metadata",
            _freeze_mapping(self.backend_metadata, path="$.backend_metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "instruction": self.instruction,
            "environment_version": self.environment_version,
            "backend_metadata": _thaw_json(self.backend_metadata),
        }


@dataclass(frozen=True, slots=True)
class RobotObservation:
    """One backend observation with transport-independent artifact references."""

    state: Any
    artifact_refs: tuple[ArtifactRef, ...] = ()
    timestamp: str = ""
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _freeze_json_value(self.state, path="$.state"))
        if not isinstance(self.timestamp, str):
            raise TypeError("$.timestamp must be a string")
        object.__setattr__(self, "artifact_refs", _freeze_refs(self.artifact_refs))
        object.__setattr__(
            self,
            "backend_metadata",
            _freeze_mapping(self.backend_metadata, path="$.backend_metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": _thaw_json(self.state),
            "artifact_refs": [ref.model_dump(mode="json") for ref in self.artifact_refs],
            "timestamp": self.timestamp,
            "backend_metadata": _thaw_json(self.backend_metadata),
        }


@dataclass(frozen=True, slots=True)
class RobotGroundedPoint:
    """One current-observation pixel grounded in the backend's world frame."""

    camera: str
    pixel_x: int
    pixel_y: int
    image_width: int
    image_height: int
    depth_m: float
    world_xyz: tuple[float, float, float] | Sequence[float]
    steps_used: int
    observation_timestamp: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera", _non_empty(self.camera, path="$.camera"))
        for name in ("pixel_x", "pixel_y", "image_width", "image_height", "steps_used"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"$.{name} must be an int")
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError("$.image_width and $.image_height must be positive")
        if not 0 <= self.pixel_x < self.image_width or not 0 <= self.pixel_y < self.image_height:
            raise ValueError("$.pixel must lie inside the current camera image")
        if self.steps_used < 0:
            raise ValueError("$.steps_used must be non-negative")
        depth = float(self.depth_m)
        if not math.isfinite(depth) or depth <= 0:
            raise ValueError("$.depth_m must be finite and positive")
        if isinstance(self.world_xyz, (str, bytes, bytearray)) or not isinstance(
            self.world_xyz, Sequence
        ):
            raise TypeError("$.world_xyz must be a three-value sequence")
        world = tuple(float(value) for value in self.world_xyz)
        if len(world) != 3 or any(not math.isfinite(value) for value in world):
            raise ValueError("$.world_xyz must contain three finite values")
        if not isinstance(self.observation_timestamp, str):
            raise TypeError("$.observation_timestamp must be a string")
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "world_xyz", world)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "pixel": {"x": self.pixel_x, "y": self.pixel_y},
            "image_size": {"width": self.image_width, "height": self.image_height},
            "depth_m": self.depth_m,
            "world_xyz": list(self.world_xyz),
            "frame": "world",
            "steps_used": self.steps_used,
            "observation_timestamp": self.observation_timestamp,
        }


@dataclass(frozen=True, slots=True)
class RobotAction:
    """One backend-neutral action with provider provenance."""

    kind: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _non_empty(self.kind, path="$.kind"))
        object.__setattr__(
            self,
            "arguments",
            _freeze_mapping(self.arguments, path="$.arguments"),
        )
        object.__setattr__(
            self,
            "provenance",
            _freeze_mapping(self.provenance, path="$.provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "arguments": _thaw_json(self.arguments),
            "provenance": _thaw_json(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class RobotResetResult:
    """Initial observation and backend provenance returned by ``reset``."""

    observation: RobotObservation
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, RobotObservation):
            raise TypeError("observation must be a RobotObservation")
        object.__setattr__(self, "info", _freeze_mapping(self.info, path="$.info"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "info": _thaw_json(self.info),
        }


@dataclass(frozen=True, slots=True)
class RobotTransition:
    """Authoritative backend result of executing one robot action."""

    observation: RobotObservation
    steps_used: int
    terminated: bool
    truncated: bool
    success: bool | None
    termination_reason: str | None = None
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, RobotObservation):
            raise TypeError("observation must be a RobotObservation")
        if isinstance(self.steps_used, bool) or not isinstance(self.steps_used, int):
            raise TypeError("steps_used must be an int")
        if self.steps_used < 0:
            raise ValueError("steps_used must be non-negative")
        for name in ("terminated", "truncated"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.success is not None and not isinstance(self.success, bool):
            raise TypeError("success must be a bool or None")
        if self.success is not None and not self.done:
            # success is three-valued: unknown (None) until the environment
            # decides. A definitive False mid-episode would make consumers
            # misreport a still-running run as failed.
            raise ValueError("success requires a terminal transition")
        if self.done and self.termination_reason is None:
            raise ValueError("a terminal transition requires termination_reason")
        if not self.done and self.termination_reason is not None:
            raise ValueError("a non-terminal transition cannot have termination_reason")
        if self.termination_reason is not None:
            object.__setattr__(
                self,
                "termination_reason",
                _non_empty(self.termination_reason, path="$.termination_reason"),
            )
        object.__setattr__(self, "info", _freeze_mapping(self.info, path="$.info"))

    @property
    def done(self) -> bool:
        """Return whether the backend ended the episode for any reason."""

        return self.terminated or self.truncated

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "steps_used": self.steps_used,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "info": _thaw_json(self.info),
        }


__all__ = [
    "RobotAction",
    "RobotGroundedPoint",
    "RobotObservation",
    "RobotResetResult",
    "RobotTask",
    "RobotTransition",
]
