from __future__ import annotations

import pytest

from alphaapollo.common.environment import EnvironmentTransition
from alphaapollo.common.trajectory import EpisodeRecorder, EpisodeResult, EpisodeTurn


def _turn(index: int, *, reward: float, done: bool, success: bool | None = None) -> EpisodeTurn:
    return EpisodeTurn(
        index=index,
        generation_request=object(),
        generation_response=object(),
        environment_transition=EnvironmentTransition(
            observation=f"observation-{index}",
            reward=reward,
            done=done,
            success=success,
            termination_reason="completed" if done else None,
        ),
    )


def test_episode_preserves_order_reward_and_terminal_success() -> None:
    first = _turn(0, reward=0.25, done=False)
    final = _turn(1, reward=1.5, done=True, success=True)

    result = EpisodeResult(
        episode_id="episode-1",
        final_text="done",
        turns=(first, final),
        termination_reason="completed",
    )

    assert result.turns == (first, final)
    assert result.env_reward_total == 1.75
    assert result.terminal_success is True
    assert result.env_steps == 2


def test_episode_does_not_infer_success_from_positive_reward() -> None:
    result = EpisodeResult(
        episode_id="truncated",
        final_text="partial",
        turns=(_turn(0, reward=2.0, done=False),),
        termination_reason="max_turns",
    )

    assert result.env_reward_total == 2.0
    assert result.terminal_success is None


def test_recorder_assembles_one_ordered_episode_without_projecting_records() -> None:
    request = object()
    response = object()
    transition = EnvironmentTransition(
        observation="complete",
        reward=1.0,
        done=True,
        success=True,
        termination_reason="completed",
    )
    recorder = EpisodeRecorder("episode-1")

    turn = recorder.record_turn(request, response, transition)
    result = recorder.finish(final_text="answer", termination_reason="completed")

    assert turn.generation_request is request
    assert turn.generation_response is response
    assert turn.environment_transition is transition
    assert result.turns == (turn,)
    assert result.terminal_success is True
    with pytest.raises(RuntimeError, match="already finished"):
        recorder.record_turn(request, response, transition)


@pytest.mark.parametrize(
    ("turns", "message"),
    [
        ((_turn(1, reward=0.0, done=False),), "contiguous"),
        (
            (
                _turn(0, reward=0.0, done=True, success=False),
                _turn(1, reward=0.0, done=False),
            ),
            "final turn",
        ),
    ],
)
def test_episode_rejects_invalid_turn_order(
    turns: tuple[EpisodeTurn, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EpisodeResult(
            episode_id="invalid",
            final_text="",
            turns=turns,
            termination_reason="stopped",
        )
