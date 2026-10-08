# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deterministic manipulation primitives executed through ``RobotBackend``.

Each primitive is a small feedback controller over the 7-dim OSC delta action
convention used by LIBERO-class arms: ``[dx, dy, dz, dax, day, daz, gripper]``
with every channel normalized to ``[-1, 1]``. The loops close on
proprioceptive feedback (``robot0_eef_pos`` / ``robot0_eef_quat``), so the
controller's exact per-unit scaling only affects step count, never the reached
target. Every simulator step goes through ``RobotBackend.execute`` and counts
against the episode budget; primitives stop immediately on a terminal
transition and report it.

World-frame yaw is recovered as ``atan2(R[1, 0], R[0, 0])`` from the
end-effector rotation matrix — robust in gripper-down configurations where
Euler-angle decompositions flip charts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from alphaapollo.common.execution.robotics import RobotAction, RobotBackend, RobotTransition
from alphaapollo.common.execution.tools.base import (
    ExecutionContext,
    ToolRequest,
    ToolResponse,
    ToolSpec,
)
from alphaapollo.common.execution.tools.robotics._shared import (
    object_schema,
    proprio_value,
    proprio_view,
    runtime_failure,
)

PRIMITIVE_TOOL_SOURCE = "AlphaApollo robotics manipulation primitives"

# Conservative normalized-command bounds; the closed loop absorbs scale error.
_POS_METERS_PER_UNIT = 0.05
_ROT_RADIANS_PER_UNIT = 0.5
_POS_COMMAND_CLIP = 0.6
_ROT_COMMAND_CLIP = 0.25


_HOLD_GRIPPER = {
    "hold_gripper_closed": {
        "type": "boolean",
        "description": "Keep the gripper closed while moving (default: open).",
    }
}

SET_GRIPPER_SPEC = ToolSpec(
    tool_id="set_gripper",
    description=(
        "Open or close the gripper in place by holding the command for a few "
        "simulator steps. Close to secure a grasp; re-issue close after a "
        "lift to lock a slipping grasp."
    ),
    parameters=object_schema(
        {
            "open": {"type": "boolean", "description": "True opens, false closes."},
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 40,
                "description": "Simulator steps to hold the command (default 8).",
            },
        },
        required=("open",),
    ),
    source=PRIMITIVE_TOOL_SOURCE,
)

RETREAT_SPEC = ToolSpec(
    tool_id="retreat",
    description=(
        "Lift the end effector straight up (world +z) by a distance — the "
        "standard recovery move after a failed or uncertain grasp, and the "
        "safe first move before a long traversal."
    ),
    parameters=object_schema(
        {
            "distance_m": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 0.5,
                "description": "Vertical retreat distance in meters (default 0.1).",
            },
            **_HOLD_GRIPPER,
        }
    ),
    source=PRIMITIVE_TOOL_SOURCE,
)

MOVE_TO_SPEC = ToolSpec(
    tool_id="move_to",
    description=(
        "Servo the end effector to a world-frame position in meters, closing "
        "on proprioceptive feedback. Holds orientation; the gripper stays "
        "open unless hold_gripper_closed. Prefer several short moves of a few "
        "centimeters over one long traversal, and approach targets from "
        "above: move at carry height first, then descend vertically. Use "
        "view_env_state for the current position."
    ),
    parameters=object_schema(
        {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
            "max_steps": {"type": "integer", "minimum": 1, "maximum": 200},
            "tolerance_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.1},
            **_HOLD_GRIPPER,
        },
        required=("x", "y", "z"),
    ),
    source=PRIMITIVE_TOOL_SOURCE,
)

ROTATE_WRIST_SPEC = ToolSpec(
    tool_id="rotate_wrist",
    description=(
        "Rotate the wrist about the world z-axis by a relative angle in "
        "radians, closing on measured yaw. Use to align the gripper with an "
        "object's orientation before grasping."
    ),
    parameters=object_schema(
        {
            "delta_yaw_rad": {
                "type": "number",
                "minimum": -3.15,
                "maximum": 3.15,
                "description": "Relative world-frame yaw in radians.",
            },
            "max_steps": {"type": "integer", "minimum": 1, "maximum": 120},
            "tolerance_rad": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.5},
            **_HOLD_GRIPPER,
        },
        required=("delta_yaw_rad",),
    ),
    source=PRIMITIVE_TOOL_SOURCE,
)


class PrimitiveFault(RuntimeError):
    """A primitive could not run against this backend's observation contract."""


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _yaw_from_quat(quat: Sequence[float]) -> float:
    """World-frame yaw from an xyzw quaternion via atan2(R[1,0], R[0,0])."""

    x, y, z, w = (float(v) for v in quat)
    r10 = 2.0 * (x * y + w * z)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(r10, r00)


@dataclass
class PrimitiveTools:
    backend: RobotBackend

    def __post_init__(self) -> None:
        if not isinstance(self.backend, RobotBackend):
            raise TypeError("backend must implement RobotBackend")

    # -- shared plumbing ------------------------------------------------------
    def _proprio(self) -> Mapping[str, Any]:
        # Same nested-or-flat resolution as view_env_state, so a backend one
        # tool reports is never a backend the other rejects.
        source = proprio_view(self.backend.observe())
        if proprio_value(source, "robot0_eef_pos") is None:
            raise PrimitiveFault(
                "this backend does not expose robot0_eef_pos proprioception; "
                "primitives require a LIBERO-class 7-dim OSC arm"
            )
        return source

    def _eef_pos(self) -> tuple[float, float, float]:
        pos = proprio_value(self._proprio(), "robot0_eef_pos")
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def _eef_yaw(self) -> float:
        quat = proprio_value(self._proprio(), "robot0_eef_quat")
        if not isinstance(quat, Sequence) or len(quat) != 4:
            raise PrimitiveFault("this backend does not expose robot0_eef_quat")
        return _yaw_from_quat(quat)

    def _step(
        self,
        tool_id: str,
        values: Sequence[float],
    ) -> RobotTransition:
        action = RobotAction(
            kind="continuous",
            arguments={"values": [float(v) for v in values]},
            provenance={"provider": "primitive", "primitive": tool_id},
        )
        return self.backend.execute(action)

    def _result(
        self,
        request: ToolRequest,
        *,
        reached: bool,
        error: float,
        steps_used: int,
        transition: RobotTransition | None,
        extra: Mapping[str, Any] | None = None,
    ) -> ToolResponse:
        done = transition.done if transition is not None else False
        payload: dict[str, Any] = {
            "primitive": request.tool_id,
            "reached": reached,
            "error": round(error, 5),
            "steps_used": steps_used,
            "environment_terminated": done,
            # Three-valued on purpose: null means the environment has not
            # decided yet, which the planner must not read as failure.
            "environment_success": transition.success if transition is not None else None,
            "next_action_required": not done,
            "terminated": transition.terminated if transition else False,
            "truncated": transition.truncated if transition else False,
            "termination_reason": transition.termination_reason if transition else None,
        }
        if extra:
            payload.update(extra)
        if transition is not None:
            payload["observation"] = transition.observation.to_dict()
        return ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stdout=json.dumps(payload, sort_keys=True),
        )

    # -- primitives -----------------------------------------------------------
    def _set_gripper(self, request: ToolRequest) -> ToolResponse:
        command = -1.0 if request.arguments["open"] else 1.0
        steps = int(request.arguments.get("steps", 8))
        transition = None
        used = 0
        for _ in range(steps):
            transition = self._step(request.tool_id, [0.0] * 6 + [command])
            used += 1
            if transition.done:
                break
        return self._result(
            request,
            reached=True,
            error=0.0,
            steps_used=used,
            transition=transition,
            extra={"gripper": "open" if command < 0 else "closed"},
        )

    def _servo_position(
        self,
        request: ToolRequest,
        target: tuple[float, float, float],
        *,
        max_steps: int,
        tolerance_m: float,
        grip: float,
    ) -> ToolResponse:
        transition = None
        used = 0
        error = self._distance(target)
        while used < max_steps:
            if error <= tolerance_m:
                break
            current = self._eef_pos()
            command = [
                _clip((target[axis] - current[axis]) / _POS_METERS_PER_UNIT, _POS_COMMAND_CLIP)
                for axis in range(3)
            ] + [0.0, 0.0, 0.0, grip]
            transition = self._step(request.tool_id, command)
            used += 1
            error = self._distance(target)
            if transition.done:
                break
        return self._result(
            request,
            reached=error <= tolerance_m,
            error=error,
            steps_used=used,
            transition=transition,
            extra={"target": list(target), "final_eef_pos": list(self._eef_pos())},
        )

    def _distance(self, target: tuple[float, float, float]) -> float:
        current = self._eef_pos()
        return math.sqrt(sum((target[i] - current[i]) ** 2 for i in range(3)))

    def _rotate_wrist(self, request: ToolRequest) -> ToolResponse:
        delta = float(request.arguments["delta_yaw_rad"])
        max_steps = int(request.arguments.get("max_steps", 40))
        tolerance = float(request.arguments.get("tolerance_rad", 0.02))
        grip = 1.0 if request.arguments.get("hold_gripper_closed") else -1.0
        target = self._eef_yaw() + delta
        transition = None
        used = 0
        error = abs(self._angle_error(target))
        while used < max_steps:
            if error <= tolerance:
                break
            command = [0.0] * 5 + [
                _clip(self._angle_error(target) / _ROT_RADIANS_PER_UNIT, _ROT_COMMAND_CLIP),
                grip,
            ]
            transition = self._step(request.tool_id, command)
            used += 1
            error = abs(self._angle_error(target))
            if transition.done:
                break
        return self._result(
            request,
            reached=error <= tolerance,
            error=error,
            steps_used=used,
            transition=transition,
            extra={"final_yaw_rad": round(self._eef_yaw(), 5)},
        )

    def _angle_error(self, target: float) -> float:
        error = target - self._eef_yaw()
        while error > math.pi:
            error -= 2.0 * math.pi
        while error < -math.pi:
            error += 2.0 * math.pi
        return error

    # -- dispatch -------------------------------------------------------------
    def execute(self, request: ToolRequest, context: ExecutionContext) -> ToolResponse:
        del context
        try:
            if self.backend.terminated or self.backend.truncated:
                raise PrimitiveFault("the episode is already terminal")
            if request.tool_id == SET_GRIPPER_SPEC.tool_id:
                return self._set_gripper(request)
            if request.tool_id == RETREAT_SPEC.tool_id:
                distance = float(request.arguments.get("distance_m", 0.1))
                grip = 1.0 if request.arguments.get("hold_gripper_closed") else -1.0
                x, y, z = self._eef_pos()
                return self._servo_position(
                    request,
                    (x, y, z + distance),
                    max_steps=max(20, int(distance / _POS_METERS_PER_UNIT) * 4),
                    tolerance_m=0.01,
                    grip=grip,
                )
            if request.tool_id == MOVE_TO_SPEC.tool_id:
                grip = 1.0 if request.arguments.get("hold_gripper_closed") else -1.0
                return self._servo_position(
                    request,
                    (
                        float(request.arguments["x"]),
                        float(request.arguments["y"]),
                        float(request.arguments["z"]),
                    ),
                    max_steps=int(request.arguments.get("max_steps", 60)),
                    tolerance_m=float(request.arguments.get("tolerance_m", 0.01)),
                    grip=grip,
                )
            if request.tool_id == ROTATE_WRIST_SPEC.tool_id:
                return self._rotate_wrist(request)
        except Exception as exc:  # noqa: BLE001 - primitive faults are episode data
            return runtime_failure(request, exc)
        raise ValueError(f"PrimitiveTools cannot execute {request.tool_id!r}")

    def close(self) -> None:
        """Nothing owned: ``RobotEnvironment`` owns backend cleanup."""


__all__ = [
    "MOVE_TO_SPEC",
    "PRIMITIVE_TOOL_SOURCE",
    "PrimitiveFault",
    "PrimitiveTools",
    "RETREAT_SPEC",
    "ROTATE_WRIST_SPEC",
    "SET_GRIPPER_SPEC",
]
