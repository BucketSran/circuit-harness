"""Bridge from common.generation.GenerationResponse to verl rollout rows.

Pins that the NEW Generation contract feeds the DataProto builder (single-turn),
and that a multi-turn trajectory (assistant/tool/assistant) assembles the verl
response_mask (1 for LLM tokens, 0 for tool tokens) -- the #185 "two generations
form one trajectory" shape at the tensor-assembly layer. Pure lists, no torch.
"""

from __future__ import annotations

import pytest

from alphaapollo.common.generation import GenerationResponse, Provenance
from alphaapollo.learning.adapters import (
    RolloutTensorError,
    TurnSegment,
    assemble_trajectory_row,
    rollout_response_row,
)


def _trainable_response() -> GenerationResponse:
    return GenerationResponse(
        request_id="r0",
        content="hi",
        prompt_token_ids=[1, 2, 3],
        response_token_ids=[10, 11],
        response_logprobs=[-0.1, -0.2],
        provenance=Provenance(policy_model="qwen", tokenizer_id="qwen-tok"),
    )


def test_single_turn_rollout_response_row() -> None:
    row = rollout_response_row(
        _trainable_response(), prompt_length=4, response_length=3, pad_token_id=0, reward_score=1.0
    )
    assert row["prompts"] == [0, 1, 2, 3]  # left-padded
    assert row["responses"] == [10, 11, 0]  # right-padded
    assert row["response_mask"] == [1, 1, 0]  # single turn: all real tokens trained
    assert row["rollout_log_probs"] == [-0.1, -0.2, 0.0]
    assert row["rm_scores"] == [0.0, 1.0, 0.0]  # reward at last real token


def test_text_only_rollout_response_is_rejected() -> None:
    text_only = GenerationResponse(request_id="r0", content="hi")
    with pytest.raises(RolloutTensorError, match="not trainable"):
        rollout_response_row(text_only, prompt_length=4, response_length=3, pad_token_id=0)


def test_multiturn_trajectory_masks_tool_tokens_zero() -> None:
    # prompt [1,2] ; assistant [10,11] (train) ; tool obs [90,91,92] (no train) ;
    # assistant [12] (train). verl response_mask = 1,1,0,0,0,1 over the response.
    segments = [
        TurnSegment(token_ids=[10, 11], trainable=True, logprobs=[-0.1, -0.2]),
        TurnSegment(token_ids=[90, 91, 92], trainable=False),
        TurnSegment(token_ids=[12], trainable=True, logprobs=[-0.3]),
    ]
    row = assemble_trajectory_row(
        prompt_ids=[1, 2], segments=segments, prompt_length=3, response_length=8, pad_token_id=0
    )
    assert row["responses"][:6] == [10, 11, 90, 91, 92, 12]
    assert row["response_mask"][:6] == [1, 1, 0, 0, 0, 1]  # tool tokens masked out
    # logprobs: real for trainable spans, 0.0 for tool span
    assert row["rollout_log_probs"][:6] == [-0.1, -0.2, 0.0, 0.0, 0.0, -0.3]
    # padded tail is zero
    assert row["response_mask"][6:] == [0, 0]


def test_trainable_segment_requires_aligned_logprobs() -> None:
    with pytest.raises(RolloutTensorError, match="aligned logprobs"):
        assemble_trajectory_row(
            prompt_ids=[1],
            segments=[TurnSegment(token_ids=[10, 11], trainable=True, logprobs=[-0.1])],
            prompt_length=2,
            response_length=4,
            pad_token_id=0,
        )
