# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Provider-neutral VLA tool contracts and the ``vla_act`` handler."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from alphaapollo.common.execution.robotics import (
    RobotAction,
    RobotBackend,
    RobotObservation,
)
from alphaapollo.common.execution.tools.base import (
    ExecutionContext,
    ToolRequest,
    ToolResponse,
    ToolSpec,
)
from alphaapollo.common.execution.tools.robotics._shared import (
    json_mapping,
    object_schema,
    runtime_failure,
    thaw_json,
)

VLA_TOOL_SOURCE = "AlphaApollo robotics semantic VLA tool"

# Hard cap on the per-call action budget: the loop pays one provider inference
# round trip per re-prediction, so the budget bounds provider cost as well as
# episode consumption. Matches the largest primitive max_steps cap.
MAX_ACTIONS_LIMIT = 200

_monotonic = time.monotonic


VLA_ACT_SPEC = ToolSpec(
    tool_id="vla_act",
    description=(
        "Ask the configured vision-language-action provider for a short "
        "closed-loop action chunk and execute it on the robot, returning the "
        "updated observation. Best for contact-rich skill segments such as "
        "grasping or placing: delegate the skill, then verify the outcome from "
        "the returned observation before continuing. Phrase the instruction as "
        "the immediate objective in plain visual language. Prefer repeated "
        "small calls over one long run so you can react between chunks."
    ),
    parameters=object_schema(
        {
            "instruction": {
                "type": "string",
                "description": "The immediate robot objective stated without provider names.",
            },
            "max_actions": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_ACTIONS_LIMIT,
                "description": (
                    "Upper bound on VLA actions executed in this call "
                    "(default 25, maximum 200). The tool re-predicts between "
                    "chunks until the budget is spent or the episode ends, so "
                    "one call completes a whole skill segment; prefer the "
                    "default over small values. Simulator steps may exceed this "
                    "count when the backend expands one action into several "
                    "steps; the response reports both."
                ),
            },
        },
        required=("instruction",),
    ),
    source=VLA_TOOL_SOURCE,
)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class VLARequest:
    instruction: str
    observation: RobotObservation
    max_actions: int
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be non-empty")
        if not isinstance(self.observation, RobotObservation):
            raise TypeError("observation must be a RobotObservation")
        object.__setattr__(self, "max_actions", _positive_int(self.max_actions, name="max_actions"))
        if self.timeout_s is not None and (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be finite and positive when provided")

    def to_payload(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "observation": self.observation.to_dict(),
            "max_actions": self.max_actions,
        }


@dataclass(frozen=True, slots=True)
class VLAResult:
    actions: tuple[RobotAction, ...]
    provider: str
    model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if not actions:
            raise ValueError("actions must not be empty")
        if any(not isinstance(action, RobotAction) for action in actions):
            raise TypeError("actions must contain only RobotAction values")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("model must be non-empty when provided")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "metadata", json_mapping(self.metadata, path="$.metadata"))


@runtime_checkable
class VLAProvider(Protocol):
    def predict(self, request: VLARequest) -> VLAResult: ...


@dataclass(slots=True)
class VLAActTool:
    provider: VLAProvider
    backend: RobotBackend
    default_max_actions: int = 25

    def __post_init__(self) -> None:
        if not isinstance(self.provider, VLAProvider):
            raise TypeError("provider must implement VLAProvider")
        if not isinstance(self.backend, RobotBackend):
            raise TypeError("backend must implement RobotBackend")
        self.default_max_actions = _positive_int(
            self.default_max_actions,
            name="default_max_actions",
        )
        if self.default_max_actions > MAX_ACTIONS_LIMIT:
            raise ValueError(f"default_max_actions must be <= {MAX_ACTIONS_LIMIT}")

    def execute(self, request: ToolRequest, context: ExecutionContext) -> ToolResponse:
        if request.tool_id != VLA_ACT_SPEC.tool_id:
            raise ValueError(f"VLAActTool cannot execute {request.tool_id!r}")
        instruction = request.arguments["instruction"]
        max_actions = request.arguments.get("max_actions", self.default_max_actions)
        try:
            # The catalog accepts integral floats for integer parameters.
            if isinstance(max_actions, float) and max_actions.is_integer():
                max_actions = int(max_actions)
            max_actions = _positive_int(max_actions, name="max_actions")
            if max_actions > MAX_ACTIONS_LIMIT:
                raise ValueError(f"max_actions must be <= {MAX_ACTIONS_LIMIT}")
            if self.backend.terminated or self.backend.truncated:
                raise RuntimeError("the episode is already terminal")
            # Re-predict between chunks until the action budget is spent or the
            # episode ends: one call completes a whole closed-loop skill
            # segment, so task progress does not depend on how often the
            # planner is willing to call the tool. Providers bound the chunk
            # they return per predict; the loop supplies the fresh observation
            # each time. The call-level timeout is one budget across every
            # re-prediction, not a fresh allowance per iteration.
            total_timeout = context.effective_timeout_s()
            deadline = None if total_timeout is None else _monotonic() + total_timeout
            transition = None
            prediction = None
            actions_requested = 0
            actions_executed = 0
            steps_used = 0
            while actions_executed < max_actions:
                remaining_s: float | None = None
                if deadline is not None:
                    remaining_s = deadline - _monotonic()
                    if remaining_s <= 0:
                        if transition is None:
                            raise TimeoutError(
                                "vla_act timeout budget exhausted before any action executed"
                            )
                        break
                prediction = self.provider.predict(
                    VLARequest(
                        instruction=instruction,
                        observation=self.backend.observe(),
                        max_actions=max_actions - actions_executed,
                        timeout_s=remaining_s,
                    )
                )
                if len(prediction.actions) > max_actions - actions_executed:
                    raise ValueError(
                        f"VLA provider returned {len(prediction.actions)} actions; "
                        f"limit is {max_actions - actions_executed}"
                    )
                if not prediction.actions:
                    # A duck-typed provider can return an empty sequence on any
                    # iteration; without progress the loop would spin forever.
                    raise RuntimeError("VLA provider returned no executable actions")
                actions_requested += len(prediction.actions)
                for action in prediction.actions:
                    transition = self.backend.execute(action)
                    actions_executed += 1
                    steps_used += transition.steps_used
                    if transition.done:
                        break
                assert transition is not None
                if transition.done:
                    break
            assert transition is not None and prediction is not None
            payload = {
                "actions_requested": actions_requested,
                "actions_executed": actions_executed,
                "steps_used": steps_used,
                "provider": prediction.provider,
                "model": prediction.model,
                "provider_metadata": thaw_json(prediction.metadata),
                # transition.info is recursively frozen; thaw before json.dumps.
                "backend_info": thaw_json(transition.info),
                "observation": transition.observation.to_dict(),
                "environment_terminated": transition.done,
                # Three-valued on purpose: null means the environment has not
                # decided yet, which the planner must not read as failure.
                "environment_success": transition.success,
                "next_action_required": not transition.done,
                "terminated": transition.terminated,
                "truncated": transition.truncated,
                "termination_reason": transition.termination_reason,
            }
            stdout = json.dumps(payload, sort_keys=True)
        except Exception as exc:  # noqa: BLE001 - provider/backend faults are episode data
            return runtime_failure(request, exc)
        return ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stdout=stdout,
        )

    def close(self) -> None:
        """Close the provider; ``RobotEnvironment`` owns backend cleanup."""

        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


__all__ = [
    "VLA_TOOL_SOURCE",
    "VLA_ACT_SPEC",
    "VLAActTool",
    "VLAProvider",
    "VLARequest",
    "VLAResult",
]
