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

"""Serve AlphaApollo's own tools to an agent that would otherwise use its own.

``mcp_config`` is how a run says which MCP servers an agent gets, and how each
CLI is told; ``mcp_server`` is the MCP server that answers. ``pi_extension.ts`` is
packaged data, not source: it is the MCP client pi does not have.

Only ``mcp_server`` needs the optional ``mcp`` extra, so composition can name and
validate a bridge without it installed.
"""

from alphaapollo.reasoning.runtime.external.bridge.mcp_config import (
    BRIDGE_MODULE,
    BRIDGE_SERVER_NAME,
    alphaapollo_bridge,
    claude_mcp_config,
    codex_config_overrides,
    normalize_mcp_servers,
)

__all__ = [
    "BRIDGE_MODULE",
    "BRIDGE_SERVER_NAME",
    "alphaapollo_bridge",
    "claude_mcp_config",
    "codex_config_overrides",
    "normalize_mcp_servers",
]
