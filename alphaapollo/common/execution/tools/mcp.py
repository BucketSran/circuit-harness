"""Transport-neutral MCP projection for an AlphaApollo ``ToolBridge``."""

from __future__ import annotations

import concurrent.futures
import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

import anyio
from mcp import types
from mcp.server.lowlevel import Server

from alphaapollo.common.environment.default.environment import ToolBridge
from alphaapollo.common.execution.tools.base import (
    PUBLIC_TOOL_IDS,
    ExecutionContext,
    ToolSpec,
    get_tool_spec,
)

__all__ = ["build_mcp_server"]

_INSTRUCTIONS = (
    "AlphaApollo tools. Calls use the configured execution bridge, policy, "
    "allowlist, budget, and artifact lifecycle. On failure, inspect the original "
    "error fields and follow deterministic guidance when the result provides it."
)

_RECOVERY_BY_ERROR_CODE = {
    "budget_exhausted": (
        "The tool-call budget is exhausted; do not retry this call. Use the available "
        "results and provide the best final answer."
    ),
    "tool_budget_exhausted": (
        "The tool-call budget is exhausted; do not retry this call. Use the available "
        "results and provide the best final answer."
    ),
    "execution_cancelled": (
        "The call was cancelled; do not assume partial output is complete. Retry only if "
        "budget remains and the cancellation cause is resolved."
    ),
    "execution_timeout": (
        "The call timed out; reduce the workload or request a shorter bounded operation "
        "before retrying. A requested timeout cannot extend the system limit."
    ),
}


def build_mcp_server(
    bridge: ToolBridge,
    context: ExecutionContext,
    *,
    tool_ids: Sequence[str] = PUBLIC_TOOL_IDS,
    specs: Sequence[ToolSpec] | None = None,
    failure_guidance: Mapping[str, str] | None = None,
    name: str = "alphaapollo",
) -> Server:
    """Build an MCP server that dispatches selected canonical tools."""

    if isinstance(tool_ids, (str, bytes)) or not tool_ids:
        raise ValueError("tool_ids must be a non-empty sequence of tool ids")
    requested_ids = tuple(str(tool_id) for tool_id in tool_ids)
    if failure_guidance is not None and not isinstance(failure_guidance, Mapping):
        raise TypeError("failure_guidance must be a mapping when provided")
    guidance_by_id = dict(failure_guidance or {})
    if any(
        not isinstance(tool_id, str) or not isinstance(guidance, str) or not guidance.strip()
        for tool_id, guidance in guidance_by_id.items()
    ):
        raise ValueError("failure_guidance requires non-empty tool ids and messages")
    unknown_guidance = sorted(set(guidance_by_id) - set(requested_ids))
    if unknown_guidance:
        raise ValueError(f"failure_guidance contains unselected tool ids: {unknown_guidance}")
    if specs is None:
        selected_specs = tuple(get_tool_spec(tool_id) for tool_id in requested_ids)
    else:
        if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
            raise TypeError("specs must be a sequence of ToolSpec records")
        supplied = tuple(specs)
        if any(not isinstance(spec, ToolSpec) for spec in supplied):
            raise TypeError("specs must contain ToolSpec records")
        by_id = {spec.tool_id: spec for spec in supplied}
        if len(by_id) != len(supplied):
            raise ValueError("specs must have unique tool ids")
        try:
            selected_specs = tuple(by_id[tool_id] for tool_id in requested_ids)
        except KeyError as exc:
            raise KeyError(f"unknown injected tool id: {exc.args[0]}") from exc
    for spec in selected_specs:
        if not spec.model_visible:
            raise ValueError(f"tool {spec.tool_id!r} is internal and must not be bridged")

    @asynccontextmanager
    async def bridge_lifespan(
        _: Server,
    ) -> AsyncIterator[concurrent.futures.ThreadPoolExecutor]:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="alphaapollo-mcp"
        ) as executor:
            yield executor

    server: Server = Server(name, instructions=_INSTRUCTIONS, lifespan=bridge_lifespan)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=spec.tool_id,
                description=spec.description,
                inputSchema=_thaw(spec.parameters),
            )
            for spec in selected_specs
        ]

    @server.call_tool(validate_input=False)
    async def call_tool(tool_id: str, arguments: dict[str, Any]) -> types.CallToolResult:
        action = {
            "id": f"mcp-{uuid.uuid4().hex[:12]}",
            "function": {"name": tool_id, "arguments": dict(arguments or {})},
        }
        result = await _dispatch_in_worker(
            server.request_context.lifespan_context, bridge, action, context
        )
        payload = result.observation_payload()
        guidance = _failure_guidance(payload, tool_id, guidance_by_id)
        if guidance is not None:
            payload["guidance"] = guidance
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, sort_keys=True))],
            structuredContent=payload,
            isError=not payload["ok"],
        )

    return server


def _failure_guidance(
    payload: Mapping[str, Any],
    tool_id: str,
    guidance_by_id: Mapping[str, str],
) -> str | None:
    """Prefer error-specific retry policy, then fall back to tool usage guidance."""

    if payload.get("ok") is not False:
        return None
    error = payload.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    if isinstance(code, str) and code in _RECOVERY_BY_ERROR_CODE:
        return _RECOVERY_BY_ERROR_CODE[code]
    return guidance_by_id.get(tool_id)


async def _dispatch_in_worker(
    executor: concurrent.futures.ThreadPoolExecutor,
    bridge: ToolBridge,
    action: dict[str, Any],
    context: ExecutionContext,
) -> Any:
    future = executor.submit(bridge.dispatch, action, context)
    while not future.done():
        await anyio.sleep(0.01)
    return future.result()


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value
