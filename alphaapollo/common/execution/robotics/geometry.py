# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small, dependency-free camera geometry primitives.

Backends own depth acquisition and camera calibration.  This module only turns
one validated metric-depth sample into a world-frame point, so semantic tools
never import simulator or array libraries and raw depth never enters the agent
transcript.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from alphaapollo.common.execution.robotics.schemas import RobotGroundedPoint


def _matrix(
    value: Sequence[Sequence[float]],
    *,
    rows: int,
    columns: int,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a matrix")
    result: list[tuple[float, ...]] = []
    for row_index, raw_row in enumerate(value):
        if isinstance(raw_row, (str, bytes, bytearray)) or not isinstance(raw_row, Sequence):
            raise TypeError(f"{name}[{row_index}] must be a row")
        row = tuple(float(item) for item in raw_row)
        if len(row) != columns or any(not math.isfinite(item) for item in row):
            raise ValueError(f"{name} must be a finite {rows}x{columns} matrix")
        result.append(row)
    if len(result) != rows:
        raise ValueError(f"{name} must be a finite {rows}x{columns} matrix")
    return tuple(result)


def _inverse_3x3(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError("camera intrinsics are singular")
    scale = 1.0 / determinant
    return (
        ((e * i - f * h) * scale, (c * h - b * i) * scale, (b * f - c * e) * scale),
        ((f * g - d * i) * scale, (a * i - c * g) * scale, (c * d - a * f) * scale),
        ((d * h - e * g) * scale, (b * g - a * h) * scale, (a * e - b * d) * scale),
    )


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, ...]:
    return tuple(sum(row[index] * vector[index] for index in range(len(vector))) for row in matrix)


def ground_metric_depth(
    *,
    camera: str,
    pixel_x: int,
    pixel_y: int,
    calibration_pixel_x: int | None = None,
    calibration_pixel_y: int | None = None,
    image_width: int,
    image_height: int,
    depth_m: float,
    intrinsics: Sequence[Sequence[float]],
    camera_to_world: Sequence[Sequence[float]],
    steps_used: int,
    observation_timestamp: str,
) -> RobotGroundedPoint:
    """Back-project one pixel using metric z-depth and a camera-to-world pose."""

    intrinsic_matrix = _matrix(intrinsics, rows=3, columns=3, name="intrinsics")
    extrinsic_matrix = _matrix(camera_to_world, rows=4, columns=4, name="camera_to_world")
    inverse_intrinsics = _inverse_3x3(intrinsic_matrix)
    projected_x = pixel_x if calibration_pixel_x is None else calibration_pixel_x
    projected_y = pixel_y if calibration_pixel_y is None else calibration_pixel_y
    camera_ray = _matvec(inverse_intrinsics, (float(projected_x), float(projected_y), 1.0))
    if abs(camera_ray[2]) < 1e-12:
        raise ValueError("camera ray has zero z component")
    scale = float(depth_m) / camera_ray[2]
    camera_point = (
        camera_ray[0] * scale,
        camera_ray[1] * scale,
        camera_ray[2] * scale,
        1.0,
    )
    world_homogeneous = _matvec(extrinsic_matrix, camera_point)
    if abs(world_homogeneous[3]) < 1e-12:
        raise ValueError("camera-to-world transform produced an invalid homogeneous point")
    world_xyz = tuple(value / world_homogeneous[3] for value in world_homogeneous[:3])
    return RobotGroundedPoint(
        camera=camera,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        image_width=image_width,
        image_height=image_height,
        depth_m=depth_m,
        world_xyz=world_xyz,
        steps_used=steps_used,
        observation_timestamp=observation_timestamp,
    )


__all__ = ["ground_metric_depth"]
