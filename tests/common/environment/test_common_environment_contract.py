from __future__ import annotations

from typing import Any

import pytest

from alphaapollo.common.environment import (
    EnvironmentContext,
    EnvironmentTransition,
    ExecutorToolBridge,
    TextOnlyToolBridge,
    available_environments,
)
from alphaapollo.common.execution import ExecutionContext, ToolRequest, ToolResponse


def _context(**payload: Any) -> EnvironmentContext:
    return EnvironmentContext(
        session_id="episode-1",
        actor="solver",
        seed=7,
        user_prompt="question",
        task_payload=payload,
    )


def test_environment_context_repr_excludes_task_payload() -> None:
    canary = "PRIVATE-ENVIRONMENT-CONTEXT-CANARY"

    context = _context(answer=canary)

    assert context.task_payload == {"answer": canary}
    assert canary not in repr(context)


def test_transition_rejects_inconsistent_terminal_fields() -> None:
    with pytest.raises(ValueError, match="require termination_reason"):
        EnvironmentTransition(observation="", reward=0.0, done=True)
    with pytest.raises(ValueError, match="success is defined only"):
        EnvironmentTransition(observation="", reward=0.0, done=False, success=False)


@pytest.mark.parametrize("reward", [True, "1.0", float("nan"), float("inf")])
def test_transition_rejects_non_numeric_or_non_finite_reward(reward: object) -> None:
    with pytest.raises((TypeError, ValueError), match="reward must"):
        EnvironmentTransition(observation="", reward=reward, done=False)  # type: ignore[arg-type]


def test_builtin_registry_is_task_agnostic_and_complete() -> None:
    assert available_environments() == ("default", "robotics")


class _Executor:
    def execute(self, request: ToolRequest, context: ExecutionContext) -> ToolResponse:
        assert context.session_id == "episode-1"
        return ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stdout="42\n",
        )


def test_managed_no_tool_and_offline_tool_bridges_use_default_contract() -> None:
    context = ExecutionContext(session_id="episode-1")
    final = TextOnlyToolBridge().dispatch("answer", context)
    bridge = ExecutorToolBridge(
        _Executor(),
        max_tool_calls=1,
        allowed_tool_ids=("python",),
    )
    tool = bridge.dispatch("<python_code>6 * 7</python_code>", context)
    refused = bridge.dispatch("<python_code>7 * 8</python_code>", context)

    assert final.model_output == "answer"
    assert tool.response is not None and tool.response.stdout == "42\n"
    assert tool.record is not None
    assert refused.refusal_code == "budget_exhausted"
