# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Helpers shared by the robotics semantic tool modules.

These existed as per-module copies until review caught them drifting;
downstream slices copy whatever is here, so there is exactly one version.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from alphaapollo.common.artifacts.schemas import require_json_value
from alphaapollo.common.execution.robotics import RobotObservation
from alphaapollo.common.execution.tools.base import ToolRequest, ToolResponse

__all__ = [
    "json_mapping",
    "object_schema",
    "proprio_value",
    "proprio_view",
    "runtime_failure",
    "thaw_json",
]


def object_schema(
    properties: dict[str, dict[str, Any]],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Closed-object parameter schema: unknown arguments are rejected."""

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def runtime_failure(request: ToolRequest, exc: Exception) -> ToolResponse:
    """Convert a provider or backend fault into a failed, recoverable tool result.

    A flaky provider service or a refused robot action is episode data the
    planner must see and react to within the same episode (#252 demonstration
    point 4), not an environment-internal error that kills the episode.
    """

    return ToolResponse(
        call_id=request.call_id,
        tool_id=request.tool_id,
        stdout="",
        stderr=f"{type(exc).__name__}: {exc}",
        exit_code=1,
    )


def json_mapping(value: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    copied = deepcopy(dict(value))
    require_json_value(copied, path=path)
    return copied


def thaw_json(value: Any) -> Any:
    """Deep-convert frozen record values (MappingProxyType/tuple) to plain JSON.

    Frozen records nest ``MappingProxyType``, which neither ``deepcopy`` nor
    ``json.dumps`` accepts; tool payloads must thaw before serializing.
    """

    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


def proprio_view(observation: RobotObservation) -> dict[str, Any]:
    """Return the proprioceptive mapping from a nested or flat observation state.

    Backends either nest robot state under ``state.proprio`` or publish it
    flat; every consumer (inspection and primitives alike) must resolve both
    shapes the same way, or one tool reports a backend the other rejects.
    """

    state = observation.state
    if isinstance(state, Mapping):
        proprio = state.get("proprio")
        if isinstance(proprio, Mapping):
            return dict(proprio)
        return {
            str(key): value
            for key, value in state.items()
            if any(marker in str(key) for marker in ("eef", "gripper", "joint", "base"))
        }
    return {}


def proprio_value(proprio: Mapping[str, Any], key: str) -> Any:
    """Look up one proprioception field, tolerating an unprefixed key name."""

    if key in proprio:
        return proprio[key]
    prefix = "robot0_"
    if key.startswith(prefix):
        return proprio.get(key[len(prefix) :])
    return None
