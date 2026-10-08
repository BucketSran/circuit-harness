# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Robot environment: the agent-facing boundary for one manipulation episode."""

from alphaapollo.common.environment.robotics.environment import RobotEnvironment
from alphaapollo.common.environment.robotics.projection import (
    format_observation,
    observation_payload,
    observation_to_payload,
    project_model_action,
    robot_continuation_messages,
)
from alphaapollo.common.execution.robotics import (
    RobotBackend,
    RobotObservation,
    RobotTask,
)

__all__ = [
    "RobotBackend",
    "RobotEnvironment",
    "RobotObservation",
    "RobotTask",
    "format_observation",
    "observation_payload",
    "observation_to_payload",
    "project_model_action",
    "robot_continuation_messages",
]
