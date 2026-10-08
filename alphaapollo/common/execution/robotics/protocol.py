# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Backend-neutral robot execution contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from alphaapollo.common.execution.robotics.schemas import (
    RobotAction,
    RobotGroundedPoint,
    RobotObservation,
    RobotResetResult,
    RobotTask,
    RobotTransition,
)


@runtime_checkable
class RobotBackend(Protocol):
    """Own robot execution and expose authoritative cumulative episode state.

    Semantic tools call :meth:`execute`; the environment observes the resulting
    episode state through the properties below.  ``steps_used`` is cumulative
    since the latest successful :meth:`reset`, not the delta from the last
    transition.
    """

    def reset(self, task: RobotTask) -> RobotResetResult: ...

    def observe(self) -> RobotObservation: ...

    def execute(self, action: RobotAction) -> RobotTransition: ...

    def close(self) -> None: ...

    @property
    def terminated(self) -> bool: ...

    @property
    def truncated(self) -> bool: ...

    @property
    def success(self) -> bool | None: ...

    @property
    def termination_reason(self) -> str | None: ...

    @property
    def steps_used(self) -> int: ...


@runtime_checkable
class RobotGeometryBackend(Protocol):
    """Optional live-scene pixel-to-world capability implemented by a backend."""

    def back_project(
        self,
        *,
        camera: str,
        pixel_x: int,
        pixel_y: int,
    ) -> RobotGroundedPoint: ...


__all__ = ["RobotBackend", "RobotGeometryBackend"]
