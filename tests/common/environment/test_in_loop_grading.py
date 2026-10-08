from __future__ import annotations

import pytest

from alphaapollo.common.environment.base import EnvironmentContext
from alphaapollo.common.environment.default import DefaultEnvironment, TextOnlyToolBridge


def _environment(grader_id: str | None) -> DefaultEnvironment:
    return DefaultEnvironment(tool_bridge=TextOnlyToolBridge(), grader_id=grader_id)


def _context(payload: dict | None) -> EnvironmentContext:
    return EnvironmentContext(
        session_id="s0",
        actor="solver",
        system_prompt="Solve it.",
        user_prompt="What is 40 + 2?",
        task_payload=payload or {},
    )


def _run(environment: DefaultEnvironment, answer: str) -> float:
    environment.init(_context({"answer": "42"}))
    transition = environment.step(answer)
    environment.close()
    return transition["reward"]


def test_graded_environment_rewards_a_correct_final_answer() -> None:
    assert _run(_environment("exact_match"), "So the answer is \\boxed{42}") == 1.0


def test_graded_environment_gives_no_reward_for_a_wrong_answer() -> None:
    assert _run(_environment("exact_match"), "So the answer is \\boxed{41}") == 0.0


def test_an_unscoreable_answer_is_zero_rather_than_a_penalty() -> None:
    # Failing to state an answer is not evidence of doing worse than answering
    # incorrectly, so it must not be distinguishable as a penalty.
    assert _run(_environment("exact_match"), "I could not finish.") == 0.0


def test_an_ungraded_environment_stays_reward_free() -> None:
    assert _run(_environment(None), "So the answer is \\boxed{42}") == 0.0


def test_gold_is_ignored_when_no_grader_is_configured() -> None:
    environment = _environment(None)
    environment.init(_context({"answer": "42"}))
    try:
        assert environment._gold is None
    finally:
        environment.close()


def test_an_unknown_grader_fails_at_construction() -> None:
    # Discovering this at the first terminal step would waste a whole matrix.
    with pytest.raises(ValueError, match="unknown grader_id"):
        _environment("no_such_grader")


def test_an_environment_graded_id_fails_at_construction() -> None:
    # It resolves, so only the offline-scoreable gate can catch it; without
    # that check the refusal moved to the first terminal step (#256 review).
    with pytest.raises(ValueError, match="execution backend's success"):
        _environment("environment_success")


def test_a_non_string_gold_is_rejected() -> None:
    environment = _environment("exact_match")
    with pytest.raises(ValueError, match="task_payload"):
        environment.init(_context({"answer": 42}))
