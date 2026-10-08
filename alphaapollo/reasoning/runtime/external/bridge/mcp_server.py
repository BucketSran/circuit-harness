# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve AlphaApollo's tools to an external coding agent over MCP stdio.

An external agent cannot be handed a Python object, so the only way to give it
*our* tools instead of its own is a protocol it already speaks.

This adds no tool semantics.  A call becomes the same function-call action a
model would emit and goes to the same ``ToolBridge`` a ``DefaultEnvironment``
uses, so catalog, policy, sandbox, budget, allowlist, and observation payload
stay one implementation.  MCP's own input validation is switched off for the
same reason: a second validator could accept or reject differently and quietly
turn "the same tools" into "similar tools".

Built-in file/shell calls and public ``python_execute`` use the configured
execution sandbox. Chips session tools forward to their Runtime-owned
Environment, preserving its budgets and recording boundary.

``mcp`` is an optional extra; nothing imports this module eagerly.  Run it
directly to serve one client on stdio::

    python -m alphaapollo.reasoning.runtime.external.bridge.mcp_server --tools read,bash
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Sequence
from pathlib import Path

import anyio
from mcp.server.stdio import stdio_server

from alphaapollo.common.environment.default import RoutedToolBridge, ToolBridge
from alphaapollo.common.execution.tools.base import (
    PUBLIC_TOOL_IDS,
    ExecutionContext,
    ToolSpec,
    list_tool_specs,
)
from alphaapollo.common.execution.tools.mcp import build_mcp_server as build_server
from alphaapollo.common.execution.tools.registry import ToolCatalog
from alphaapollo.reasoning.runtime.external.bridge.mcp_config import (
    BRIDGE_SERVER_NAME as SERVER_NAME,
)

__all__ = ["SERVER_NAME", "build_server", "main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Serve on stdio with one sandbox shared by every call in this process."""

    parser = argparse.ArgumentParser(
        prog="python -m alphaapollo.reasoning.runtime.external.bridge.mcp_server",
        description="Serve AlphaApollo workspace tools to an MCP client over stdio.",
    )
    parser.add_argument(
        "--tools",
        default=",".join(PUBLIC_TOOL_IDS),
        help=(
            "comma-separated explicit tool ids to advertise and permit "
            "(default: every public built-in tool)"
        ),
    )
    parser.add_argument("--session-id", default=None, help="execution session id")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="system-owned per-call limit; a tool may request less, never more",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="tool-call budget for this session (default: unbounded)",
    )
    parser.add_argument(
        "--environment-socket",
        default=None,
        help="forward calls using this Runtime-owned Environment endpoint file",
    )
    args = parser.parse_args(argv)

    tool_ids = tuple(dict.fromkeys(part.strip() for part in args.tools.split(",") if part.strip()))
    specs = _tool_specs(tool_ids)
    context = ExecutionContext(
        session_id=args.session_id or f"mcp-bridge-{uuid.uuid4().hex[:12]}",
        timeout_s=args.timeout_s,
    )
    if args.environment_socket is None:
        bridge = _local_tool_bridge(
            tool_ids,
            specs=specs,
            max_tool_calls=args.max_tool_calls,
        )
    else:
        from alphaapollo.reasoning.runtime.external.bridge.environment_socket import (
            EnvironmentSocketToolBridge,
        )

        bridge = EnvironmentSocketToolBridge(args.environment_socket)
    server = build_server(
        bridge,
        context,
        tool_ids=tool_ids,
        specs=specs,
    )

    async def serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    try:
        anyio.run(serve)
    finally:
        bridge.close()


def _local_tool_bridge(
    tool_ids: Sequence[str],
    *,
    specs: Sequence[ToolSpec],
    max_tool_calls: int | None,
) -> RoutedToolBridge:
    """Compose selected local tools without flattening their execution semantics."""

    from alphaapollo.common.environment.default import (
        ExecutorToolBridge,
        GatewayToolBridge,
    )
    from alphaapollo.common.execution import ExecutionRuntime, ToolGateway
    from alphaapollo.common.execution.tools.chips import EMX_TOOL_ID, SESSION_TOOL_SPECS, EmxTool
    from alphaapollo.common.execution.tools.python import (
        PYTHON_EXECUTE_SPEC,
        PYTHON_EXECUTE_TOOL_ID,
        PythonExecuteTool,
    )

    if any(spec.tool_id in tool_ids for spec in SESSION_TOOL_SPECS):
        raise ValueError("Chips session tools require --environment-socket")

    routes: dict[str, ToolBridge] = {}
    if EMX_TOOL_ID in tool_ids:
        routes[EMX_TOOL_ID] = ExecutorToolBridge(
            EmxTool(working_directory=Path.cwd()),
            allowed_tool_ids=(EMX_TOOL_ID,),
        )
    gateway_ids = tuple(
        tool_id
        for tool_id in tool_ids
        if tool_id in PUBLIC_TOOL_IDS or tool_id == PYTHON_EXECUTE_TOOL_ID
    )
    if gateway_ids:
        if PYTHON_EXECUTE_TOOL_ID in gateway_ids:
            runtime = ExecutionRuntime(
                catalog=ToolCatalog((*list_tool_specs(include_internal=True), PYTHON_EXECUTE_SPEC)),
                additional_tools=(PythonExecuteTool(),),
            )
        else:
            runtime = ExecutionRuntime()
        gateway = GatewayToolBridge(
            ToolGateway(runtime=runtime),
            allowed_tool_ids=gateway_ids,
        )
        routes.update({tool_id: gateway for tool_id in gateway_ids})

    return RoutedToolBridge(
        routes,
        catalog=ToolCatalog(specs),
        max_tool_calls=max_tool_calls,
    )


def _tool_specs(tool_ids: Sequence[str]) -> tuple[ToolSpec, ...]:
    """Resolve built-ins and selected domain tools through one catalog."""

    from alphaapollo.common.execution.tools.chips import EMX_TOOL_SPEC, SESSION_TOOL_SPECS
    from alphaapollo.common.execution.tools.python import PYTHON_EXECUTE_SPEC

    catalog = ToolCatalog(
        (
            *list_tool_specs(include_internal=True),
            PYTHON_EXECUTE_SPEC,
            EMX_TOOL_SPEC,
            *SESSION_TOOL_SPECS,
        )
    )
    return tuple(catalog.get(tool_id) for tool_id in tool_ids)


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
