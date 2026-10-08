# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Environment-facing tool gateway and bound gateway sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from alphaapollo.common.execution._session.runtime import ExecutionRuntime, ToolInvocationResult
from alphaapollo.common.execution.sandbox.base import CancellationToken, OutputSink
from alphaapollo.common.execution.tools.base import (
    ExecutionContext,
    ToolError,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.execution.tools.schemas import ToolCallRecord


@runtime_checkable
class _RuntimeSession(Protocol):
    @property
    def context(self) -> ExecutionContext: ...

    @property
    def closed(self) -> bool: ...

    def invoke(
        self,
        request: ToolRequest,
        context: ExecutionContext | None = None,
        *,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ToolInvocationResult: ...

    def close(self) -> None: ...


# Environment-facing gateway -------------------------------------------------
@dataclass(frozen=True, slots=True)
class ToolGatewayResult:
    """Return the model-facing outcome alongside optional audit evidence."""

    outcome: ToolResponse | ToolError
    record: ToolCallRecord | None


@dataclass(slots=True)
class ToolGateway:
    """Invoke the shared runtime and preserve both response and audit views."""

    runtime: ExecutionRuntime = field(default_factory=ExecutionRuntime)

    def open_session(self, context: ExecutionContext) -> ToolGatewaySession:
        """Bind one actor context to a persistent isolated workspace."""
        return ToolGatewaySession(self.runtime.open_session(context))

    def invoke(
        self,
        request: ToolRequest,
        context: ExecutionContext,
        *,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ToolGatewayResult:
        result = self.runtime.invoke(
            request,
            context,
            on_output=on_output,
            cancellation=cancellation,
        )
        return _gateway_result(request, result)


class ToolGatewaySession:
    """Environment-facing gateway bound to one persistent execution session."""

    def __init__(self, runtime_session: _RuntimeSession) -> None:
        if not isinstance(runtime_session, _RuntimeSession):
            raise TypeError("ToolGatewaySession expects a runtime session")
        self._runtime_session = runtime_session

    @property
    def context(self) -> ExecutionContext:
        return self._runtime_session.context

    @property
    def closed(self) -> bool:
        return self._runtime_session.closed

    def invoke(
        self,
        request: ToolRequest,
        context: ExecutionContext | None = None,
        *,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ToolGatewayResult:
        result = self._runtime_session.invoke(
            request,
            context,
            on_output=on_output,
            cancellation=cancellation,
        )
        return _gateway_result(request, result)

    def close(self) -> None:
        self._runtime_session.close()

    def __enter__(self) -> ToolGatewaySession:
        if self.closed:
            raise RuntimeError("tool gateway session is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _gateway_result(
    request: ToolRequest,
    result: ToolInvocationResult,
) -> ToolGatewayResult:
    if isinstance(result, ToolError):
        return ToolGatewayResult(outcome=result, record=None)

    response = ToolResponse(
        call_id=request.call_id,
        tool_id=request.tool_id,
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        artifacts=tuple(result.artifacts),
    )
    return ToolGatewayResult(outcome=response, record=result)
