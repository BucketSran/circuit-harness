"""Tool execution, isolated backends, sessions, and workspaces."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ArtifactStore": "alphaapollo.common.execution.workspace",
    "CancellationToken": "alphaapollo.common.execution.sandbox.base",
    "ExecutionContext": "alphaapollo.common.execution.tools.base",
    "ExecutionRuntime": "alphaapollo.common.execution.session",
    "ExecutionRuntimeSession": "alphaapollo.common.execution.session",
    "OutputChunk": "alphaapollo.common.execution.sandbox.base",
    "OutputSink": "alphaapollo.common.execution.sandbox.base",
    "SandboxSession": "alphaapollo.common.execution.session",
    "SandboxSessionError": "alphaapollo.common.execution.session",
    "ToolError": "alphaapollo.common.execution.tools.base",
    "ToolExecutor": "alphaapollo.common.execution.tools.base",
    "ToolGateway": "alphaapollo.common.execution.session",
    "ToolGatewayResult": "alphaapollo.common.execution.session",
    "ToolGatewaySession": "alphaapollo.common.execution.session",
    "ToolInvocationResult": "alphaapollo.common.execution.session",
    "ToolRequest": "alphaapollo.common.execution.tools.base",
    "ToolResponse": "alphaapollo.common.execution.tools.base",
    "WorkspaceLease": "alphaapollo.common.execution.workspace",
    "WorkspaceProvider": "alphaapollo.common.execution.workspace",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
