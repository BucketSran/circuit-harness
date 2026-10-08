"""Robot execution contracts shared by tools, environments, and backends."""

from alphaapollo.common.execution.robotics.protocol import RobotBackend, RobotGeometryBackend
from alphaapollo.common.execution.robotics.schemas import (
    RobotAction,
    RobotGroundedPoint,
    RobotObservation,
    RobotResetResult,
    RobotTask,
    RobotTransition,
)

__all__ = [
    "RobotAction",
    "RobotBackend",
    "RobotGeometryBackend",
    "RobotGroundedPoint",
    "RobotObservation",
    "RobotResetResult",
    "RobotTask",
    "RobotTransition",
]
