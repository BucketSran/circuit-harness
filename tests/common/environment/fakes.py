"""Test-only resolved Gateway fixtures for Environment unit tests."""

from __future__ import annotations

from typing import Any

from alphaapollo.common.environment import GatewayToolBridge, ToolBridgeResult
from alphaapollo.common.execution import (
    ExecutionContext,
    ToolError,
    ToolExecutor,
    ToolGateway,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.execution.tools.schemas import ToolCallRecord


class _ExecutorRuntime:
    def __init__(self, executor: ToolExecutor) -> None:
        self._executor = executor
        self._context: ExecutionContext | None = None
        self.closed = False

    @property
    def context(self) -> ExecutionContext:
        assert self._context is not None
        return self._context

    def open_session(self, context: ExecutionContext) -> _ExecutorRuntime:
        self._context = context
        return self

    def invoke(
        self,
        request: ToolRequest,
        context: ExecutionContext,
        **_: Any,
    ) -> ToolCallRecord | ToolError:
        try:
            response = self._executor.execute(request, context)
        except TimeoutError:
            return ToolCallRecord(
                tool_id=request.tool_id,
                args=request.arguments,
                stderr="tool execution timed out",
                exit_code=124,
            )
        except Exception:
            return ToolCallRecord(
                tool_id=request.tool_id,
                args=request.arguments,
                stderr="tool execution failed",
                exit_code=-1,
            )
        if not isinstance(response, ToolResponse):
            raise TypeError("test executor must return ToolResponse")
        if response.call_id != request.call_id or response.tool_id != request.tool_id:
            raise ValueError("test executor response identity mismatch")
        return ToolCallRecord(
            tool_id=request.tool_id,
            args=request.arguments,
            stdout=response.stdout,
            stderr=response.stderr,
            exit_code=response.exit_code,
            artifacts=list(response.artifacts),
        )

    def close(self) -> None:
        self.closed = True


class CanonicalTestToolBridge:
    """Use production ingress and Gateway adaptation with an injected executor."""

    def __init__(self, executor: ToolExecutor, **_: Any) -> None:
        gateway = ToolGateway(runtime=_ExecutorRuntime(executor))  # type: ignore[arg-type]
        self._bridge = GatewayToolBridge(gateway)

    def dispatch(self, action: Any, context: ExecutionContext) -> ToolBridgeResult:
        return self._bridge.dispatch(action, context)

    def close(self) -> None:
        self._bridge.close()


__all__ = ["CanonicalTestToolBridge"]
