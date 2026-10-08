# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Public, model-visible Python execution over the existing isolated runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from alphaapollo.common.execution.sandbox.base import (
    CancellationToken,
    OutputSink,
    SandboxBackend,
)
from alphaapollo.common.execution.tools.base import ToolRequest, ToolSpec
from alphaapollo.common.execution.tools.builtins._shell.bash import BashTool
from alphaapollo.common.execution.tools.builtins._shell.python import build_python_command
from alphaapollo.common.execution.tools.schemas import ToolCallRecord
from alphaapollo.common.execution.workspace import ArtifactStore

PYTHON_EXECUTE_TOOL_ID = "python_execute"

PYTHON_EXECUTE_SPEC = ToolSpec(
    tool_id=PYTHON_EXECUTE_TOOL_ID,
    description=(
        "Run one bounded Python program in AlphaApollo's separate networkless sandbox. "
        "Each call starts a fresh interpreter; files may persist between sandbox-tool calls "
        "in this session, but this sandbox cannot see the external Agent workspace. "
        "Common standard-library modules are pre-imported. The "
        "wrapper prints a final expression; when the last line is neither an expression nor "
        "print(...), it also prints the most recent simple assignment, even if earlier code "
        "already printed. "
        "Use stdout, stderr, and exit_code as execution evidence: exit 0 means the program "
        "ran, not that a task result was accepted by the Verifier. Example: "
        'code="print(sum(range(10)))".'
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Complete Python source for one fresh interpreter in the Python-tool "
                    "sandbox; print or leave a final expression to return a value"
                ),
            },
            "timeout": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": (
                    "Optional timeout in seconds; it may only shorten the system limit"
                ),
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    source="AlphaApollo v2 compatibility contract",
    adaptations=(
        "Adds a public structured function-call ID while retaining the internal "
        "<python_code> compatibility ingress.",
    ),
)


@dataclass(frozen=True, slots=True)
class PythonExecuteTool:
    """Execute public ``python_execute`` calls through a borrowed sandbox backend."""

    tool_id: str = PYTHON_EXECUTE_TOOL_ID
    _bash_tool: BashTool = field(default_factory=BashTool, repr=False)

    def validate_request(self, request: ToolRequest) -> str | None:
        code = request.arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return "python code must be a non-empty string"
        timeout = request.arguments.get("timeout")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            return "python timeout must be a finite positive number"
        return None

    def requested_timeout_s(self, request: ToolRequest) -> float | None:
        timeout = request.arguments.get("timeout")
        return (
            float(timeout)
            if isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
            else None
        )

    def execute(
        self,
        backend: SandboxBackend,
        request: ToolRequest,
        *,
        artifact_store: ArtifactStore | None = None,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ToolCallRecord:
        if request.tool_id != self.tool_id:
            raise ValueError(f"PythonExecuteTool cannot execute {request.tool_id!r}")
        problem = self.validate_request(request)
        if problem is not None:
            raise ValueError(problem)
        code = request.arguments["code"]
        assert isinstance(code, str)
        command_request = ToolRequest(
            call_id=request.call_id,
            tool_id=self._bash_tool.tool_id,
            arguments={"command": build_python_command(code)},
            source=request.source,
        )
        record = self._bash_tool.execute(
            backend,
            command_request,
            artifact_store=artifact_store,
            on_output=on_output,
            cancellation=cancellation,
        )
        return record.model_copy(update={"tool_id": self.tool_id})


__all__ = ["PYTHON_EXECUTE_SPEC", "PYTHON_EXECUTE_TOOL_ID", "PythonExecuteTool"]
