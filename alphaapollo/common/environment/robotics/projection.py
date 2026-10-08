# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Projection between planner output and robot tool calls, plus multimodal views.

Tool-call normalization and tool-result formatting are shared with the coding
environment. What is robot-specific here is the observation payload: it always
carries the environment's authoritative ``environment_terminated``,
``environment_success``, and ``next_action_required`` fields, plus the updated
backend observation, so a planner can recover inside the episode.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from alphaapollo.common.environment.default.projection import (
    FeedbackMode,
    normalize,
    tool_response_payload,
)
from alphaapollo.common.execution import ToolError, ToolRequest, ToolResponse
from alphaapollo.common.execution.robotics import RobotObservation

__all__ = [
    "format_observation",
    "load_observation_image_blocks",
    "observation_payload",
    "observation_to_payload",
    "project_model_action",
    "robot_continuation_messages",
    "tool_call_payload",
]


def project_model_action(action: Any) -> ToolRequest | ToolError | None:
    """Return one structured robotics call without interpreting prose as code."""

    if isinstance(action, str):
        return None
    return normalize(_without_text_tool_syntax(action))


def observation_to_payload(observation: RobotObservation) -> dict[str, Any]:
    """Serialize one observation to a JSON-safe, transport-independent record."""

    return observation.to_dict()


def observation_payload(
    result: ToolResponse | ToolError | None,
    *,
    request: ToolRequest | None = None,
    observation: RobotObservation | None = None,
    environment_terminated: bool,
    environment_success: bool | None,
    next_action_required: bool,
    feedback_mode: FeedbackMode = "differentiated",
) -> dict[str, Any]:
    """Build the JSON-safe observation recorded after one planner action.

    ``result`` is ``None`` only for an advisory action such as ``finish`` that the
    environment records without executing. The three environment fields describe
    the episode; the ``tool`` field describes only the current operation.
    """

    payload: dict[str, Any] = {
        "environment_terminated": bool(environment_terminated),
        "environment_success": (None if environment_success is None else bool(environment_success)),
        "next_action_required": bool(next_action_required),
    }
    if result is not None:
        payload["tool"] = tool_response_payload(
            result,
            request=request,
            feedback_mode=feedback_mode,
        )
    elif request is not None:
        payload["tool"] = {
            "ok": True,
            "call_id": request.call_id,
            "tool_id": request.tool_id,
            "arguments": request.arguments,
            "advisory": True,
        }
    if observation is not None:
        payload["observation"] = observation_to_payload(observation)
    return payload


def format_observation(payload: Mapping[str, Any]) -> str:
    """Encode one observation payload for the next planner turn."""

    if not isinstance(payload, Mapping):
        raise TypeError("observation payload must be a mapping")
    return json.dumps(dict(payload), sort_keys=True)


def load_observation_image_blocks(
    observation: RobotObservation,
    load_artifact: Callable[[Any], bytes],
) -> tuple[Mapping[str, Any], ...]:
    """Resolve image artifacts while the environment can isolate failures.

    Artifact loading is intentionally separate from continuation assembly. A
    missing mid-episode frame must fail the owning environment step instead of
    escaping later from the Runtime's batch-wide transcript projection.
    """

    if not isinstance(observation, RobotObservation):
        raise TypeError("observation must be a RobotObservation")
    if not callable(load_artifact):
        raise TypeError("load_artifact must be callable")
    return tuple(_image_blocks(observation, load_artifact))


def robot_continuation_messages(
    *,
    assistant_message: Mapping[str, Any],
    observation_text: str,
    image_blocks: Sequence[Mapping[str, Any]] = (),
    call_id: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Return the assistant turn, tool result, and preloaded scene images."""

    if not isinstance(observation_text, str):
        raise TypeError("observation_text must be a string")
    tool_message: dict[str, Any] = (
        {"role": "tool", "tool_call_id": call_id, "content": observation_text}
        if call_id
        else {"role": "user", "content": observation_text}
    )
    messages: list[Mapping[str, Any]] = [dict(assistant_message), tool_message]
    if image_blocks:
        messages.append({"role": "user", "content": [dict(block) for block in image_blocks]})
    return tuple(messages)


def _image_blocks(
    observation: RobotObservation,
    load_artifact: Callable[[Any], bytes],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    roles = observation.backend_metadata.get("artifact_roles", {})
    role_by_id = (
        {
            artifact_id: role
            for role, artifact_id in roles.items()
            if isinstance(role, str) and isinstance(artifact_id, str)
        }
        if isinstance(roles, Mapping)
        else {}
    )
    for ref in observation.artifact_refs:
        media_type = ref.type or ""
        if not media_type.startswith("image/"):
            continue
        data = load_artifact(ref)
        if not isinstance(data, bytes):
            raise TypeError(f"image artifact {ref.id!r} loader must return bytes")
        if not data:
            raise ValueError(f"image artifact {ref.id!r} is empty")
        role = role_by_id.get(ref.id)
        if role is not None:
            blocks.append({"type": "text", "text": role.upper()})
        encoded = base64.b64encode(data).decode("ascii")
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
            }
        )
    return blocks


def _without_text_tool_syntax(action: Any) -> Any:
    """Keep structured calls while preventing coding-only tags from becoming tools."""

    raw: Any = action
    if not isinstance(raw, Mapping):
        model_dump = getattr(action, "model_dump", None)
        if not callable(model_dump):
            return action
        try:
            raw = model_dump()
        except (TypeError, ValueError):
            return action
    if not isinstance(raw, Mapping):
        return action

    payload = dict(raw)
    choices = payload.get("choices")
    if (
        isinstance(choices, Sequence)
        and not isinstance(choices, (str, bytes, bytearray))
        and len(choices) == 1
        and isinstance(choices[0], Mapping)
    ):
        choice = dict(choices[0])
        message = choice.get("message")
        if isinstance(message, Mapping):
            choice["message"] = {**dict(message), "content": ""}
            payload["choices"] = [choice]
        return payload
    message = payload.get("message")
    if isinstance(message, Mapping):
        payload["message"] = {**dict(message), "content": ""}
    else:
        payload["content"] = ""
    return payload


def tool_call_payload(call: object) -> dict[str, Any]:
    """Project one generation tool call into the shape ``normalize`` accepts."""

    return {
        "id": getattr(call, "id", ""),
        "type": "function",
        "function": {
            "name": getattr(call, "name", ""),
            "arguments": getattr(call, "arguments", ""),
        },
    }
