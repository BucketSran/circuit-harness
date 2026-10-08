# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Reasoning Runtime contracts and the AlphaApollo-managed implementation."""

from __future__ import annotations

from typing import Any

from alphaapollo.reasoning.runtime.agent_runtime import (
    AgentResult,
    AgentRuntime,
    AgentTask,
    AgentTurn,
)

_EXTERNAL_EXPORTS = frozenset(
    {
        "ExternalAgentRuntime",
        "ExternalAgentSession",
        "ExternalEvent",
        "ExternalRunOutcome",
        "project_outcome",
    }
)

__all__ = [
    "AgentResult",
    "AgentRuntime",
    "AgentTask",
    "AgentTurn",
    "AlphaApolloAgentRuntime",
    "EnvironmentProjector",
    *sorted(_EXTERNAL_EXPORTS),
]


def __getattr__(name: str) -> Any:
    """Keep pure Runtime contracts importable without loading Common integrations."""

    if name in {"AlphaApolloAgentRuntime", "EnvironmentProjector"}:
        from alphaapollo.reasoning.runtime.alphaapollo_agent_runtime import (
            AlphaApolloAgentRuntime,
            EnvironmentProjector,
        )

        return {
            "AlphaApolloAgentRuntime": AlphaApolloAgentRuntime,
            "EnvironmentProjector": EnvironmentProjector,
        }[name]
    if name in _EXTERNAL_EXPORTS:
        from alphaapollo.reasoning.runtime import external_agent_runtime

        return getattr(external_agent_runtime, name)
    raise AttributeError(name)
