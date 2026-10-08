# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Provider-neutral perception contracts and semantic tool handlers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from alphaapollo.common.execution.robotics import (
    RobotBackend,
    RobotGeometryBackend,
    RobotObservation,
)
from alphaapollo.common.execution.tools.base import (
    ExecutionContext,
    ToolRequest,
    ToolResponse,
    ToolSpec,
)
from alphaapollo.common.execution.tools.robotics._shared import (
    json_mapping,
    object_schema,
    runtime_failure,
)

PERCEPTION_TOOL_SOURCE = "AlphaApollo robotics semantic perception tools"


_MIN_SCORE = {
    "type": "number",
    "minimum": 0.0,
    "maximum": 1.0,
    "description": "Minimum provider confidence included in the result.",
}

SEGMENT_SPEC = ToolSpec(
    tool_id="segment",
    description=(
        "Segment an object in the current camera frame with a text query; "
        "read-only, never moves the robot. Returns per-instance score, pixel "
        "bounding box, mask statistics, centroid_pixel, and a backend-verified "
        "world_xyz when live metric depth is available. Phrase the "
        "query as visible attributes plus a spatial relation — e.g. 'the "
        "black bowl on the stove' — and avoid dataset-internal or brand "
        "names, which ground poorly. Check the top instance is plausible "
        "before acting on it."
    ),
    parameters=object_schema(
        {
            "query": {
                "type": "string",
                "description": "Semantic description of the object to segment.",
            },
            "point": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Optional normalized [x, y] point prompt.",
            },
            "min_score": _MIN_SCORE,
        },
        required=("query",),
    ),
    source=PERCEPTION_TOOL_SOURCE,
)

DETECT_OBJECTS_SPEC = ToolSpec(
    tool_id="detect_objects",
    description=(
        "Detect object instances in the current camera frame; read-only, "
        "never moves the robot. Returns per-instance score and pixel bounding "
        "box. Use a short visual query (attributes plus spatial relation) "
        "rather than dataset-internal names."
    ),
    parameters=object_schema(
        {
            "query": {
                "type": "string",
                "description": "Optional semantic filter for returned objects.",
            },
            "min_score": _MIN_SCORE,
        }
    ),
    source=PERCEPTION_TOOL_SOURCE,
)

PerceptionOperation = Literal["segment", "detect_objects"]


@dataclass(frozen=True, slots=True)
class PerceptionRequest:
    operation: PerceptionOperation
    observation: RobotObservation
    query: str | None = None
    point: tuple[float, float] | None = None
    min_score: float = 0.2
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.operation not in ("segment", "detect_objects"):
            raise ValueError("unsupported perception operation")
        if not isinstance(self.observation, RobotObservation):
            raise TypeError("observation must be a RobotObservation")
        if self.query is not None and (not isinstance(self.query, str) or not self.query.strip()):
            raise ValueError("query must be non-empty when provided")
        if self.operation == "segment" and self.query is None:
            raise ValueError("segment requires a query")
        if self.point is not None:
            point = tuple(self.point)
            if len(point) != 2 or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in point
            ):
                raise ValueError("point must contain two finite numbers")
            object.__setattr__(self, "point", (float(point[0]), float(point[1])))
        if (
            isinstance(self.min_score, bool)
            or not isinstance(self.min_score, (int, float))
            or not math.isfinite(self.min_score)
            or not 0.0 <= self.min_score <= 1.0
        ):
            raise ValueError("min_score must be between zero and one")
        if self.timeout_s is not None and (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be finite and positive when provided")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "observation": self.observation.to_dict(),
            "min_score": self.min_score,
        }
        if self.query is not None:
            payload["query"] = self.query
        if self.point is not None:
            payload["point"] = list(self.point)
        return payload


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    items: tuple[Mapping[str, Any], ...]
    provider: str
    model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "items",
            tuple(
                json_mapping(item, path=f"$.items[{index}]")
                for index, item in enumerate(self.items)
            ),
        )
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("model must be non-empty when provided")
        object.__setattr__(self, "metadata", json_mapping(self.metadata, path="$.metadata"))


@runtime_checkable
class PerceptionProvider(Protocol):
    def inspect(self, request: PerceptionRequest) -> PerceptionResult: ...


@dataclass(slots=True)
class PerceptionTools:
    provider: PerceptionProvider
    backend: RobotBackend
    default_min_score: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.provider, PerceptionProvider):
            raise TypeError("provider must implement PerceptionProvider")
        if not isinstance(self.backend, RobotBackend):
            raise TypeError("backend must implement RobotBackend")
        if (
            isinstance(self.default_min_score, bool)
            or not isinstance(self.default_min_score, (int, float))
            or not 0.0 <= self.default_min_score <= 1.0
        ):
            raise ValueError("default_min_score must be between zero and one")

    @staticmethod
    def _centroid_pixel(item: Mapping[str, Any]) -> tuple[int, int]:
        raw = item.get("centroid_pixel", item.get("centroid"))
        if isinstance(raw, Mapping):
            raw = (raw.get("x"), raw.get("y"))
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise ValueError("segment item has no centroid_pixel")
        values = tuple(raw)
        if len(values) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("segment centroid_pixel must contain finite x/y values")
        return int(round(float(values[0]))), int(round(float(values[1])))

    def _ground_segment_items(self, items: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
        grounded: list[dict[str, Any]] = []
        for raw_item in items:
            item = deepcopy(dict(raw_item))
            item["world_xyz"] = None
            item["depth_m"] = None
            item["grounding_valid"] = False
            item["grounding_error"] = None
            camera = item.get("camera", "fixed_camera")
            try:
                if not isinstance(camera, str) or not camera.strip():
                    raise ValueError("segment item camera must be a non-empty string")
                pixel_x, pixel_y = self._centroid_pixel(item)
                item["camera"] = camera
                item["centroid_pixel"] = {"x": pixel_x, "y": pixel_y}
                if not isinstance(self.backend, RobotGeometryBackend):
                    raise RuntimeError("backend does not expose live metric depth")
                point = self.backend.back_project(
                    camera=camera,
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                )
                item.update(
                    {
                        "world_xyz": list(point.world_xyz),
                        "depth_m": point.depth_m,
                        "grounding_valid": True,
                        "grounding_error": None,
                        "grounding_steps_used": point.steps_used,
                        "grounding_observation_timestamp": point.observation_timestamp,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - partial perception remains useful
                item["grounding_error"] = f"{type(exc).__name__}: {exc}"
            grounded.append(item)
        return grounded

    def execute(self, request: ToolRequest, context: ExecutionContext) -> ToolResponse:
        if request.tool_id not in (SEGMENT_SPEC.tool_id, DETECT_OBJECTS_SPEC.tool_id):
            raise ValueError(f"PerceptionTools cannot execute {request.tool_id!r}")
        raw_point = request.arguments.get("point")
        point = None if raw_point is None else tuple(raw_point)
        try:
            observation = self.backend.observe()
            result = self.provider.inspect(
                PerceptionRequest(
                    operation=request.tool_id,
                    observation=observation,
                    query=request.arguments.get("query"),
                    point=point,
                    min_score=request.arguments.get("min_score", self.default_min_score),
                    timeout_s=context.effective_timeout_s(),
                )
            )
        except Exception as exc:  # noqa: BLE001 - provider faults are episode data
            return runtime_failure(request, exc)
        items = (
            self._ground_segment_items(result.items)
            if request.tool_id == SEGMENT_SPEC.tool_id
            else [deepcopy(dict(item)) for item in result.items]
        )
        payload: dict[str, Any] = {
            "operation": request.tool_id,
            "items": items,
            "count": len(result.items),
            "provider": result.provider,
            "model": result.model,
            "provider_metadata": deepcopy(dict(result.metadata)),
            "observation": observation.to_dict(),
        }
        return ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stdout=json.dumps(payload, sort_keys=True),
        )

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


__all__ = [
    "DETECT_OBJECTS_SPEC",
    "PERCEPTION_TOOL_SOURCE",
    "PerceptionOperation",
    "PerceptionProvider",
    "PerceptionRequest",
    "PerceptionResult",
    "PerceptionTools",
    "SEGMENT_SPEC",
]
