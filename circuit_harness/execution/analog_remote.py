"""Analog public actions over the existing durable local or SSH transport."""

from .session_transport import LocalSessionTransport, RemoteSessionTransport


class RemoteAnalog(RemoteSessionTransport):
    request_command = "analog-request"
    response_command = "analog-response"


class LocalAnalog(LocalSessionTransport):
    request_command = "analog-request"
    response_command = "analog-response"
