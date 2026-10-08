# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Stable imports for execution runtimes, gateways, and sessions.

Implementations are split by responsibility under ``execution._session``;
this module remains the supported compatibility and public import surface.
"""

from alphaapollo.common.execution._session.gateway import (
    ToolGateway,
    ToolGatewayResult,
    ToolGatewaySession,
)
from alphaapollo.common.execution._session.legacy import SandboxSession, SandboxSessionError
from alphaapollo.common.execution._session.runtime import (
    ExecutionRuntime,
    ExecutionRuntimeSession,
    ToolInvocationResult,
)

__all__ = [
    "ExecutionRuntime",
    "ExecutionRuntimeSession",
    "SandboxSession",
    "SandboxSessionError",
    "ToolGateway",
    "ToolGatewayResult",
    "ToolGatewaySession",
    "ToolInvocationResult",
]
