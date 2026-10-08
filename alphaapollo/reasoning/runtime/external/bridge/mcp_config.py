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

"""Declare MCP servers once, hand them to whichever agent CLI is running.

The three agents reach MCP three different ways -- Claude Code takes a JSON
document, Codex takes dotted TOML overrides, and pi has no client at all -- so
one declaration here saves writing the same wiring three times and keeping the
copies honest.  It is the canonical ``mcpServers`` shape used across the MCP
ecosystem, so a third-party server joins a run with no code change; AlphaApollo's
own tool bridge is one entry produced by :func:`alphaapollo_bridge`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from alphaapollo.common.execution.tools.base import PUBLIC_TOOL_IDS

__all__ = [
    "BRIDGE_MODULE",
    "BRIDGE_SERVER_NAME",
    "alphaapollo_bridge",
    "claude_mcp_config",
    "codex_approval_overrides",
    "codex_config_overrides",
    "normalize_mcp_servers",
]

#: The module a bridged agent spawns; kept here so the wiring names it once.
BRIDGE_MODULE = "alphaapollo.reasoning.runtime.external.bridge.mcp_server"

#: The server name an MCP client sees, and the prefix in ``mcp__alphaapollo__*``.
#: Declared here rather than in the server so composition can name it without
#: importing the optional ``mcp`` dependency.
BRIDGE_SERVER_NAME = "alphaapollo"

# The external agent starts stdio MCP servers from its isolated workspace, not
# from the repository. Keep the bridge importable from both a source checkout
# and an installed wheel without relying on the operator's current directory.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[5])

_STDIO_KEYS = frozenset({"command", "args", "env"})
_HTTP_KEYS = frozenset({"url", "headers"})


def alphaapollo_bridge(
    *,
    tool_ids: Sequence[str] = PUBLIC_TOOL_IDS,
    timeout_s: float | None = None,
    max_tool_calls: int | None = None,
    environment_socket: str | None = None,
) -> dict[str, Any]:
    """Return the stdio server declaration for AlphaApollo's own tool bridge.

    ``sys.executable`` is the interpreter because the agent CLI inherits the
    operator's ``PATH``, not this process's virtualenv; a bare ``python`` there
    can easily be one without AlphaApollo installed.
    """

    if isinstance(tool_ids, (str, bytes)) or not tool_ids:
        raise ValueError("tool_ids must be a non-empty sequence of tool ids")
    args = ["-m", BRIDGE_MODULE, "--tools", ",".join(str(name) for name in tool_ids)]
    if timeout_s is not None:
        args.extend(("--timeout-s", str(float(timeout_s))))
    if max_tool_calls is not None:
        args.extend(("--max-tool-calls", str(int(max_tool_calls))))
    if environment_socket is not None:
        if not isinstance(environment_socket, str) or not environment_socket.strip():
            raise ValueError("environment_socket must be a non-empty string when provided")
        if timeout_s is not None or max_tool_calls is not None:
            raise ValueError("Environment-owned limits cannot also be declared on the MCP bridge")
        args.extend(("--environment-socket", environment_socket))
    return {
        "command": sys.executable,
        "args": args,
        "env": {"PYTHONPATH": _PACKAGE_PARENT},
    }


def normalize_mcp_servers(servers: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Validate a declaration map so a typo fails here, not inside the agent.

    Each entry is either stdio (``command`` plus optional ``args`` / ``env``) or
    HTTP (``url`` plus optional ``headers``).  Mixing the two is rejected rather
    than resolved by precedence, because which transport was meant is exactly
    what a reader would have to guess.
    """

    if servers is None:
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for name, spec in servers.items():
        unknown = set(spec) - _STDIO_KEYS - _HTTP_KEYS
        if unknown:
            raise ValueError(f"mcp_servers[{name!r}] has unknown keys: {sorted(unknown)}")
        if ("command" in spec) == ("url" in spec):
            raise ValueError(f"mcp_servers[{name!r}] must declare exactly one of command, url")
        normalized[name] = {
            key: _leaf(value, f"mcp_servers[{name!r}].{key}")
            for key, value in spec.items()
            if value is not None
        }
    return normalized


def claude_mcp_config(servers: Mapping[str, Mapping[str, Any]]) -> str:
    """Render the JSON document Claude Code's ``--mcp-config`` accepts inline."""

    return json.dumps({"mcpServers": dict(servers)}, sort_keys=True)


def codex_config_overrides(servers: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Render Codex ``-c key=value`` overrides for the same declarations.

    Every leaf is emitted as its own dotted override with a JSON-encoded value.
    JSON scalars and string arrays are valid TOML, which is what Codex parses,
    and flattening maps this way avoids hand-writing TOML inline tables.
    """

    overrides: list[str] = []
    for name, spec in servers.items():
        for key, value in spec.items():
            if isinstance(value, Mapping):
                for inner, item in value.items():
                    overrides.append(f"mcp_servers.{name}.{key}.{inner}={json.dumps(item)}")
            else:
                overrides.append(f"mcp_servers.{name}.{key}={json.dumps(value)}")
    return overrides


def codex_approval_overrides(
    approvals: Mapping[str, Sequence[str]],
) -> list[str]:
    """Render explicit per-tool approval settings for headless Codex runs."""

    overrides: list[str] = []
    for server, tool_ids in approvals.items():
        if not isinstance(server, str) or not server.strip():
            raise ValueError("MCP approval server names must be non-empty strings")
        if isinstance(tool_ids, (str, bytes)) or not isinstance(tool_ids, Sequence):
            raise TypeError(f"MCP approvals for {server!r} must be a sequence of tool ids")
        if not tool_ids:
            raise ValueError(f"MCP approvals for {server!r} must not be empty")
        for tool_id in tool_ids:
            if not isinstance(tool_id, str) or not tool_id.strip():
                raise ValueError("MCP approval tool ids must be non-empty strings")
            overrides.append(f'mcp_servers.{server}.tools.{tool_id}.approval_mode="approve"')
    return overrides


def _leaf(value: object, where: str) -> Any:
    """Accept only what every MCP client config allows, so YAML noise fails here."""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _leaf(item, where) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_leaf(item, where) for item in value]
    raise TypeError(f"{where} must be a string, a list of strings, or a mapping of them")
