"""Rollout-tensor assembly must reproduce verl's padding byte-for-byte.

The pure-list core (build_rollout_row) carries no torch, so these run everywhere
and pin the exact padding, mask, position-id, and rm_score math against a
hand-computed example matching verl's documented scheme. The DataProto stacking
is exercised through an injected fake so it needs no verl; the real field-parity
check against a native verl rollout is the server test.
"""

from __future__ import annotations

import pytest

from alphaapollo.learning.adapters import (
    ROLLOUT_TENSOR_FIELDS,
    RolloutTensorError,
    build_rollout_row,
    stack_to_dataproto,
)


def test_build_rollout_row_matches_hand_computed_padding() -> None:
    # prompt [1,2,3,4] left-padded to 6; response [5,6,7,8] with token 7 a tool
    # token (mask 0), right-padded to 6. Every expected value computed by hand
    # from verl's _agent_loop_postprocess scheme.
    row = build_rollout_row(
        prompt_ids=[1, 2, 3, 4],
        response_ids=[5, 6, 7, 8],
        response_mask=[1, 1, 0, 1],
        prompt_length=6,
        response_length=6,
        pad_token_id=0,
        response_logprobs=[-0.1, -0.2, -0.3, -0.4],
        reward_score=1.0,
    )
    assert row["prompts"] == [0, 0, 1, 2, 3, 4]
    assert row["responses"] == [5, 6, 7, 8, 0, 0]
    assert row["response_mask"] == [1, 1, 0, 1, 0, 0]
    assert row["input_ids"] == [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0]
    assert row["attention_mask"] == [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    # cumsum-1 clipped at 0; trailing padding keeps the running max (7), NOT 0 —
    # this matches verl's compute_position_id_with_mask code, not its doc-comment.
    assert row["position_ids"] == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]
    assert row["rollout_log_probs"] == [-0.1, -0.2, -0.3, -0.4, 0.0, 0.0]
    assert row["rm_scores"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # reward at last real token


def test_build_rollout_row_does_not_invent_reward_when_untrained() -> None:
    # Absent logprobs are honest only where nothing is trained (mask all 0).
    row = build_rollout_row(
        prompt_ids=[1, 2],
        response_ids=[3],
        response_mask=[0],
        prompt_length=3,
        response_length=2,
        pad_token_id=9,
        response_logprobs=None,
        reward_score=None,
    )
    assert row["prompts"] == [9, 1, 2]
    assert row["responses"] == [3, 9]
    assert row["rollout_log_probs"] == [0.0, 0.0]
    assert "rm_scores" not in row
    assert set(row) == set(ROLLOUT_TENSOR_FIELDS) - {"rm_scores"}


def test_build_rollout_row_rejects_absent_logprobs_at_trainable_positions() -> None:
    # mask=1 with no logprobs would fabricate zero sampling-time logprobs at a
    # loss position; that must fail loudly rather than silently emit zeros.
    with pytest.raises(RolloutTensorError, match="response_logprobs is required"):
        build_rollout_row(
            prompt_ids=[1, 2],
            response_ids=[3],
            response_mask=[1],
            prompt_length=3,
            response_length=2,
            pad_token_id=9,
            response_logprobs=None,
            reward_score=None,
        )


def test_build_rollout_row_rejects_overflow_and_misalignment() -> None:
    with pytest.raises(RolloutTensorError, match="exceeds prompt_length"):
        build_rollout_row(
            prompt_ids=[1, 2, 3],
            response_ids=[4],
            response_mask=[1],
            prompt_length=2,
            response_length=2,
            pad_token_id=0,
        )
    with pytest.raises(RolloutTensorError, match="same length"):
        build_rollout_row(
            prompt_ids=[1],
            response_ids=[4, 5],
            response_mask=[1],
            prompt_length=2,
            response_length=2,
            pad_token_id=0,
        )


def _two_rows() -> list[dict[str, list[float]]]:
    return [
        build_rollout_row(
            prompt_ids=[1, 2, 3],
            response_ids=[5, 6],
            response_mask=[1, 1],
            prompt_length=4,
            response_length=3,
            pad_token_id=0,
            response_logprobs=[-0.1, -0.2],
        )
        for _ in range(2)
    ]


def test_stack_to_dataproto_uses_injected_class_and_dtypes() -> None:
    torch = pytest.importorskip("torch")

    captured: dict[str, object] = {}

    class _FakeDataProto:
        @classmethod
        def from_dict(cls, *, tensors, non_tensors, meta_info):
            captured["tensors"] = tensors
            captured["meta_info"] = meta_info
            return "DATAPROTO"

    rows = _two_rows()
    result = stack_to_dataproto(rows, meta_info={"n": 2}, dataproto_class=_FakeDataProto)
    assert result == "DATAPROTO"
    tensors = captured["tensors"]
    assert tensors["input_ids"].shape == (2, 7)  # prompt_length + response_length
    assert tensors["input_ids"].dtype == torch.int64
    assert tensors["rollout_log_probs"].dtype == torch.float32
    assert "rm_scores" not in tensors
    assert captured["meta_info"] == {"n": 2}


def test_stack_to_dataproto_rejects_mixed_reward_presence() -> None:
    rows = _two_rows()
    rows[0]["rm_scores"] = [0.0, 1.0, 0.0]

    with pytest.raises(RolloutTensorError, match="must be present in every row or none"):
        stack_to_dataproto(rows, dataproto_class=object)


def test_build_rollout_row_rejects_non_integer_and_non_finite_values() -> None:
    # #189 (X1angyuLu [P2]): the row is float-cast then restored to int64 tensors, so
    # a fractional token id, non-binary mask, or NaN/Inf logprob would silently
    # corrupt the batch. Each must be refused before conversion.
    base = dict(
        prompt_ids=[1, 2],
        response_ids=[3],
        response_mask=[1],
        prompt_length=3,
        response_length=2,
        pad_token_id=0,
        response_logprobs=[-0.1],
    )
    with pytest.raises(RolloutTensorError, match="integer token ids"):
        build_rollout_row(**{**base, "response_ids": [3.5]})
    with pytest.raises(RolloutTensorError, match="integer token ids"):
        build_rollout_row(**{**base, "prompt_ids": [True, 2]})
    with pytest.raises(RolloutTensorError, match="0 or 1"):
        build_rollout_row(**{**base, "response_mask": [2]})
    with pytest.raises(RolloutTensorError, match="finite"):
        build_rollout_row(**{**base, "response_logprobs": [float("nan")]})
    with pytest.raises(RolloutTensorError, match="finite"):
        build_rollout_row(**{**base, "reward_score": float("inf")})
    with pytest.raises(RolloutTensorError, match="pad_token_id"):
        build_rollout_row(**{**base, "pad_token_id": 0.5})


def test_build_rollout_row_accepts_numpy_integer_and_real_types() -> None:
    # Learning feeds rows out of NumPy/torch: integral NumPy ids and real NumPy
    # logprobs must be accepted; only a genuinely fractional / non-real value is
    # rejected. (Regression: a strict `isinstance(int)` wrongly rejected np.int64.)
    np = pytest.importorskip("numpy")
    row = build_rollout_row(
        prompt_ids=np.array([1, 2, 3], dtype=np.int64),
        response_ids=[np.int64(5), np.int32(6)],
        response_mask=[np.int64(1), 1],
        prompt_length=4,
        response_length=3,
        pad_token_id=np.int64(0),
        response_logprobs=[np.float32(-0.1), np.float64(-0.2)],
        reward_score=np.float32(1.0),
    )
    assert row["responses"][:2] == [5, 6]
    with pytest.raises(RolloutTensorError, match="integer token ids"):
        build_rollout_row(
            prompt_ids=[np.float64(1.5)],
            response_ids=[5],
            response_mask=[1],
            prompt_length=2,
            response_length=2,
            pad_token_id=0,
            response_logprobs=[-0.1],
        )


def test_assemble_trajectory_row_validates_segment_logprobs_before_cast() -> None:
    # Multi-turn path: segment logprobs are validated before the float() cast, so a
    # bool or numeric string can't be laundered into a valid logprob (which would
    # bypass build_rollout_row's downstream finiteness check).
    from alphaapollo.learning.adapters.rollout_tensors import (
        TurnSegment,
        assemble_trajectory_row,
    )

    for bad in ([True], ["-0.25"], [float("nan")]):
        with pytest.raises(RolloutTensorError, match="finite"):
            assemble_trajectory_row(
                [1],
                [TurnSegment(token_ids=[2], trainable=True, logprobs=bad)],
                prompt_length=2,
                response_length=2,
                pad_token_id=0,
            )


def test_build_rollout_row_rejects_reward_on_empty_response() -> None:
    # verl anchors the reward at the last real response token; with no response
    # token the reward would be silently dropped, so reject the combination.
    with pytest.raises(RolloutTensorError, match="empty response"):
        build_rollout_row(
            prompt_ids=[1],
            response_ids=[],
            response_mask=[],
            prompt_length=2,
            response_length=2,
            pad_token_id=0,
            reward_score=1.0,
        )


def test_turn_segment_enforces_bool_trainable_and_no_stray_logprobs() -> None:
    from alphaapollo.learning.adapters.rollout_tensors import TurnSegment

    with pytest.raises(RolloutTensorError, match="must be a bool"):
        TurnSegment(token_ids=[1], trainable="false", logprobs=[-0.1])  # type: ignore[arg-type]
    with pytest.raises(RolloutTensorError, match="must not carry logprobs"):
        TurnSegment(token_ids=[1], trainable=False, logprobs=[-0.1])


def test_stack_to_dataproto_preserves_provenance_non_tensor() -> None:
    # #189 (X1angyuLu [P1]): provenance is retained as a batch-aligned non-tensor so
    # Learning can verify policy/tokenizer/chat-template match before an update. This
    # adapter only retains it; it does not enforce the match.
    pytest.importorskip("torch")
    from alphaapollo.common.generation import Provenance

    captured: dict[str, object] = {}

    class _FakeDataProto:
        @classmethod
        def from_dict(cls, *, tensors, non_tensors, meta_info):
            captured["non_tensors"] = non_tensors
            return "DP"

    rows = _two_rows()
    provenances = [
        Provenance(policy_model="m", tokenizer_id="t", weights_version="s1"),
        Provenance(policy_model="m", tokenizer_id="t", weights_version="s2"),
    ]
    stack_to_dataproto(rows, provenances=provenances, dataproto_class=_FakeDataProto)
    provenance = captured["non_tensors"]["provenance"]  # type: ignore[index]
    assert provenance.dtype == object
    assert provenance[0]["weights_version"] == "s1"
    assert provenance[1]["weights_version"] == "s2"


def test_stack_to_dataproto_rejects_provenance_length_mismatch() -> None:
    pytest.importorskip("torch")
    from alphaapollo.common.generation import Provenance

    rows = _two_rows()
    with pytest.raises(RolloutTensorError, match="must align with rows"):
        stack_to_dataproto(
            rows,
            provenances=[Provenance(policy_model="m", tokenizer_id="t")],
            dataproto_class=object,
        )
