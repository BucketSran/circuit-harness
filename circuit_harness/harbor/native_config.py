"""Codex CLI settings for the public MCP broker."""

import json
from collections.abc import Mapping, Sequence
from typing import Any


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
