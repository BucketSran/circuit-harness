"""Compatibility imports for VABench action transport."""

from circuit_harness.execution.transport.session_transport import (
    LocalSessionTransport,
    RemoteSessionTransport,
)


class RemoteVabench(RemoteSessionTransport):
    request_command = "vabench-request"
    response_command = "vabench-response"


class LocalVabench(LocalSessionTransport, RemoteVabench):
    """VABench local transport; remains a RemoteVabench subtype."""
