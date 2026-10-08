"""Concrete implementations of the :class:`RobotBackend` contract."""

from alphaapollo.common.execution.robotics.backends.fake import FakeBackend
from alphaapollo.common.execution.robotics.backends.libero import LiberoBackend
from alphaapollo.common.execution.robotics.backends.recording import EpisodeRecorder
from alphaapollo.common.execution.robotics.backends.remote import (
    HttpRobotRpcTransport,
    RemoteBackend,
    RobotRpcError,
    RobotRpcTransport,
)
from alphaapollo.common.execution.robotics.backends.robocasa import (
    RoboCasaBackend,
    create_robocasa_environment,
)

__all__ = [
    "EpisodeRecorder",
    "FakeBackend",
    "HttpRobotRpcTransport",
    "LiberoBackend",
    "RemoteBackend",
    "RoboCasaBackend",
    "RobotRpcError",
    "RobotRpcTransport",
    "create_robocasa_environment",
]
