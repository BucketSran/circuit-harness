# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Default Environment grading inputs, validity, and runtime metadata projection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alphaapollo.common.environment.base import EnvironmentContext
from alphaapollo.common.environment.default.bridge import ToolBridgeResult
from alphaapollo.common.environment.default.projection import execution_failure


def _gold_from_payload(payload: Any, grader_id: str | None) -> str | None:
    """Read the gold answer an episode was given, if it was given one.

    Gold reaches an Environment only through ``task_payload``, which the prepared
    dataset's private projection populates. Training needs it in loop to produce
    a reward; evaluation grades after the fact and must never configure a grader
    here, which its config parser refuses outright.
    """

    if grader_id is None or not isinstance(payload, Mapping):
        return None
    gold = payload.get("answer")
    if gold is None:
        return None
    if not isinstance(gold, str):
        raise ValueError("task_payload['answer'] must be a string when a grader is configured")
    return gold


def _episode_context_from(context: EnvironmentContext) -> dict[str, Any]:
    """Project the canonical init context onto captured episode metadata."""

    episode_context = {
        key: value
        for key, value in context.metadata.items()
        if key not in {"task_payload", "workspace_snapshot_ref"}
    }
    episode_context["system_prompt"] = context.system_prompt
    if context.task_id is not None:
        episode_context["task_id"] = context.task_id
    if context.seed is not None:
        episode_context["seed"] = context.seed
    if context.environment_config:
        episode_context["environment_config"] = context.environment_config
    return episode_context


def _capture_model_action(action: Any) -> dict[str, Any]:
    if isinstance(action, str):
        return {"role": "assistant", "content": action}
    if isinstance(action, dict):
        return {"role": "assistant", "provider_response": action}
    model_dump = getattr(action, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except (TypeError, ValueError):
            dumped = None
        if isinstance(dumped, dict):
            return {"role": "assistant", "provider_response": dumped}
    return {
        "role": "assistant",
        "capture_omitted": "unsupported_action_type",
        "action_type": type(action).__name__,
    }


def _execution_failure(exit_code: int) -> tuple[str, str] | None:
    return execution_failure(exit_code)


def _tool_validity(result: ToolBridgeResult | None) -> tuple[bool, bool]:
    """Classify model-action validity separately from tool execution success."""

    if result is None:
        return True, True
    if result.refused:
        return True, False
    if result.error is None:
        return True, True
    assert result.error is not None
    if result.error.stage == "parse":
        return False, False
    if result.error.stage in {"catalog", "policy"}:
        return True, False
    # Resource acquisition and execution failures do not make the model action
    # malformed or unavailable in the Environment's action space.
    return True, True


def _runtime_tool_metadata(result: ToolBridgeResult) -> dict[str, Any]:
    """Project a bridge result onto the stable AgentRuntime-facing metadata."""
    if result.refused:
        return {"tool_refused": result.refusal_code}
    metadata: dict[str, Any] = {}
    if result.request is not None:
        metadata["tool_request"] = {
            "call_id": result.request.call_id,
            "tool_id": result.request.tool_id,
            "arguments": dict(result.request.arguments),
            "source": result.request.source,
        }
    elif result.error is not None:
        # Parse failures have no canonical ToolRequest, but still count as one
        # rejected tool turn for runtime budgets and failure metrics.
        metadata["tool_request"] = {
            "call_id": result.error.call_id,
            "tool_id": result.error.tool_id,
            "source": "rejected_tool_call",
            "stage": result.error.stage,
            "code": result.error.code,
        }

    if result.response is not None:
        metadata["tool_response"] = {
            "call_id": result.response.call_id,
            "tool_id": result.response.tool_id,
            "stdout": result.response.stdout,
            "stderr": result.response.stderr,
            "exit_code": result.response.exit_code,
            "artifacts": [ref.model_dump(mode="json") for ref in result.response.artifacts],
        }
    if result.record is not None:
        metadata["tool_record"] = result.record.model_dump(mode="json")

    if result.error is not None:
        metadata["tool_error"] = {
            "stage": result.error.stage,
            "code": result.error.code,
            "message": result.error.message,
            "attempted": False,
        }
    elif result.response is not None:
        failure = _execution_failure(result.response.exit_code)
        if failure is not None:
            stage, code = failure
            metadata["tool_error"] = {
                "stage": stage,
                "code": code,
                "message": f"tool exited with code {result.response.exit_code}",
                "attempted": True,
            }
    return metadata
