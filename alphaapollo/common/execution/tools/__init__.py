"""Typed tool contracts, registry, and built-in implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "INTERNAL_PYTHON_TOOL_ID": "alphaapollo.common.execution.tools.base",
    "PI_SOURCE": "alphaapollo.common.execution.tools.base",
    "PI_SOURCE_COMMIT": "alphaapollo.common.execution.tools.base",
    "PUBLIC_TOOL_IDS": "alphaapollo.common.execution.tools.base",
    "BashTool": "alphaapollo.common.execution.tools.builtins.shell",
    "CostProgressRecord": "alphaapollo.common.execution.tools.schemas",
    "EditTool": "alphaapollo.common.execution.tools.builtins.files",
    "ExecutionContext": "alphaapollo.common.execution.tools.base",
    "ExecutionPolicy": "alphaapollo.common.execution.tools.gateway",
    "FindTool": "alphaapollo.common.execution.tools.builtins.files",
    "GrepTool": "alphaapollo.common.execution.tools.builtins.files",
    "LsTool": "alphaapollo.common.execution.tools.builtins.files",
    "PythonCompatibilityTool": "alphaapollo.common.execution.tools.builtins.shell",
    "PYTHON_EXECUTE_SPEC": "alphaapollo.common.execution.tools.python",
    "PYTHON_EXECUTE_TOOL_ID": "alphaapollo.common.execution.tools.python",
    "PythonExecuteTool": "alphaapollo.common.execution.tools.python",
    "PythonTool": "alphaapollo.common.execution.tools.builtins.shell",
    "ReadTool": "alphaapollo.common.execution.tools.builtins.files",
    "ToolCallRecord": "alphaapollo.common.execution.tools.schemas",
    "ToolCatalog": "alphaapollo.common.execution.tools.registry",
    "ToolError": "alphaapollo.common.execution.tools.base",
    "ToolExecutor": "alphaapollo.common.execution.tools.base",
    "ToolRequest": "alphaapollo.common.execution.tools.base",
    "ToolResponse": "alphaapollo.common.execution.tools.base",
    "ToolSpec": "alphaapollo.common.execution.tools.base",
    "WorkspaceTool": "alphaapollo.common.execution.tools.builtins.files",
    "WriteTool": "alphaapollo.common.execution.tools.builtins.files",
    "build_python_command": "alphaapollo.common.execution.tools.builtins.shell",
    "build_workspace_command": "alphaapollo.common.execution.tools.builtins.files",
    "export_openai_tools": "alphaapollo.common.execution.tools.base",
    "get_tool_spec": "alphaapollo.common.execution.tools.base",
    "list_tool_specs": "alphaapollo.common.execution.tools.base",
    "wrap_v2_python_code": "alphaapollo.common.execution.tools.builtins.shell",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
