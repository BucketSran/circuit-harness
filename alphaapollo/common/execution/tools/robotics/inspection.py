# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read-only robot state, camera, and geometry inspection tools.

These tools inspect the backend's current observation and never produce robot
actions, mirroring the perception tools' read-only contract. Camera metadata
comes from the observation's environment metadata, where the in-process
backends publish best-effort intrinsics/extrinsics per camera role.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from alphaapollo.common.execution.robotics import RobotBackend, RobotGeometryBackend
from alphaapollo.common.execution.tools.base import (
    ExecutionContext,
    ToolRequest,
    ToolResponse,
    ToolSpec,
)
from alphaapollo.common.execution.tools.robotics._shared import (
    object_schema,
    proprio_view,
    runtime_failure,
    thaw_json,
)

INSPECTION_TOOL_SOURCE = "AlphaApollo robotics inspection tools"


VIEW_ENV_STATE_SPEC = ToolSpec(
    tool_id="view_env_state",
    description=(
        "Read the robot's current proprioceptive state (end-effector pose, "
        "gripper opening, joints) plus the episode's step accounting and "
        "termination flags. Read-only. Primitives already return the updated "
        "observation — use this only when re-checking state without acting."
    ),
    parameters=object_schema({}),
    source=INSPECTION_TOOL_SOURCE,
)

VIEW_CAMERA_META_SPEC = ToolSpec(
    tool_id="view_camera_meta",
    description=(
        "Read the calibrated camera metadata (intrinsics, extrinsics, "
        "resolution) per camera role. Read-only; use it for pixel-to-geometry "
        "reasoning alongside segment results."
    ),
    parameters=object_schema(
        {
            "camera": {
                "type": "string",
                "description": "Optional camera role to select; omit for all cameras.",
            },
        }
    ),
    source=INSPECTION_TOOL_SOURCE,
)

BACK_PROJECT_SPEC = ToolSpec(
    tool_id="back_project",
    description=(
        "Convert one pixel from the current camera image into a verified "
        "world-frame XYZ point using the backend's live metric depth and "
        "calibration. Read-only. Pixel coordinates use image x (column) and "
        "y (row); use the centroid_pixel returned by segment."
    ),
    parameters=object_schema(
        {
            "camera": {
                "type": "string",
                "description": "Camera role (default: fixed_camera).",
            },
            "pixel": object_schema(
                {
                    "x": {"type": "integer", "minimum": 0},
                    "y": {"type": "integer", "minimum": 0},
                },
                required=("x", "y"),
            ),
        },
        required=("pixel",),
    ),
    source=INSPECTION_TOOL_SOURCE,
)


def _camera_view(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    environment = metadata.get("environment")
    if isinstance(environment, Mapping):
        cameras = environment.get("cameras")
        if isinstance(cameras, Mapping) and cameras:
            return cameras
    cameras = metadata.get("cameras")
    if isinstance(cameras, Mapping) and cameras:
        return cameras
    return {}


@dataclass
class InspectionTools:
    backend: RobotBackend

    def __post_init__(self) -> None:
        if not isinstance(self.backend, RobotBackend):
            raise TypeError("backend must implement RobotBackend")

    def execute(self, request: ToolRequest, context: ExecutionContext) -> ToolResponse:
        del context
        if request.tool_id not in (
            VIEW_ENV_STATE_SPEC.tool_id,
            VIEW_CAMERA_META_SPEC.tool_id,
            BACK_PROJECT_SPEC.tool_id,
        ):
            raise ValueError(f"InspectionTools cannot execute {request.tool_id!r}")
        try:
            observation = self.backend.observe()
            if request.tool_id == VIEW_ENV_STATE_SPEC.tool_id:
                payload: dict[str, Any] = {
                    # Observation state is recursively frozen; thaw nested
                    # mappings before json.dumps.
                    "proprio": thaw_json(proprio_view(observation)),
                    "steps_used": self.backend.steps_used,
                    "terminated": self.backend.terminated,
                    "truncated": self.backend.truncated,
                    "success": self.backend.success,
                    "termination_reason": self.backend.termination_reason,
                }
            elif request.tool_id == VIEW_CAMERA_META_SPEC.tool_id:
                cameras = _camera_view(observation.to_dict()["backend_metadata"])
                if not cameras:
                    raise RuntimeError(
                        "this backend does not publish camera metadata for the current scene"
                    )
                selected = request.arguments.get("camera")
                if selected is not None:
                    if selected not in cameras:
                        raise ValueError(
                            f"unknown camera {selected!r}; available: {sorted(cameras)}"
                        )
                    cameras = {selected: cameras[selected]}
                payload = {"cameras": dict(cameras)}
            else:
                if not isinstance(self.backend, RobotGeometryBackend):
                    raise RuntimeError(
                        "this backend does not expose live metric depth for back_project"
                    )
                pixel = request.arguments["pixel"]
                payload = self.backend.back_project(
                    camera=request.arguments.get("camera", "fixed_camera"),
                    pixel_x=int(pixel["x"]),
                    pixel_y=int(pixel["y"]),
                ).to_dict()
            stdout = json.dumps(payload, sort_keys=True)
        except Exception as exc:  # noqa: BLE001 - inspection faults are episode data
            return runtime_failure(request, exc)
        return ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stdout=stdout,
        )

    def close(self) -> None:
        """Nothing owned: ``RobotEnvironment`` owns backend cleanup."""


__all__ = [
    "BACK_PROJECT_SPEC",
    "INSPECTION_TOOL_SOURCE",
    "InspectionTools",
    "VIEW_CAMERA_META_SPEC",
    "VIEW_ENV_STATE_SPEC",
]
