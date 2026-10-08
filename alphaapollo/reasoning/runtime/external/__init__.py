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

"""The closed registry the composition root selects an external agent through."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from alphaapollo.reasoning.runtime.external.agents import (
    ClaudeCodeSession,
    CodexSession,
    CodexViaPiSession,
    FakeExternalSession,
    PiSession,
)
from alphaapollo.reasoning.runtime.external.bridge import (
    alphaapollo_bridge,
    normalize_mcp_servers,
)
from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalAgentSession

__all__ = [
    "ClaudeCodeSession",
    "CodexSession",
    "CodexViaPiSession",
    "FakeExternalSession",
    "PiSession",
    "alphaapollo_bridge",
    "available_agents",
    "normalize_mcp_servers",
    "session_factory",
]

_AGENTS: dict[str, type[Any]] = {
    "claude_code": ClaudeCodeSession,
    "codex": CodexSession,
    "codex_via_pi": CodexViaPiSession,
    "pi": PiSession,
    "fake": FakeExternalSession,
}


def available_agents() -> tuple[str, ...]:
    """Names accepted by :func:`session_factory`, in registration order."""

    return tuple(_AGENTS)


def session_factory(
    agent: str, options: Mapping[str, Any] | None = None
) -> Callable[[], ExternalAgentSession]:
    """Return a zero-argument factory building one fresh session per task.

    The registry is closed: an unknown name fails here rather than at the first
    subprocess call, so a typo in a run configuration is a configuration error.
    """

    if agent not in _AGENTS:
        raise ValueError(f"unknown external agent {agent!r}; available: {sorted(_AGENTS)}")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping")
    session_type = _AGENTS[agent]
    keyword_options = dict(options or {})

    def create() -> ExternalAgentSession:
        return session_type(**keyword_options)

    return create
