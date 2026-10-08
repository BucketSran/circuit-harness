from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from alphaapollo.learning.off_policy.algorithms.dpo import (
    make_dpo_loss_fn,
    sequence_logps,
    sigmoid_dpo_losses,
)


def _nested(rows, *, dtype=torch.float32):
    return torch.nested.as_nested_tensor(
        [torch.tensor(row, dtype=dtype) for row in rows],
        layout=torch.jagged,
    )


def test_sequence_logps_are_reduced_per_sequence():
    token_logps = _nested([[10.0, 2.0, 3.0, 100.0], [7.0, 8.0, 9.0]])
    loss_mask = _nested([[0, 0, 1, 1], [0, 1, 1]], dtype=torch.long)

    actual = sequence_logps(token_logps, loss_mask)

    # Row 0 shifted mask = [0, 1, 1, 0], row 1 = [1, 1, 0]. This
    # ordinary prompt-prefixed fixture checks the per-row sums; the all-ones
    # fixture below is what distinguishes zero-filling from wrap-around.
    torch.testing.assert_close(actual, torch.tensor([5.0, 15.0]))
    torch.testing.assert_close(
        sequence_logps(token_logps, loss_mask, average=True), torch.tensor([2.5, 7.5])
    )


def test_sequence_logps_zero_fills_the_tail_instead_of_wrapping_around():
    """The tail must be zeroed, never wrapped — reachable via `truncation: left`.

    For ordinary rows the mask starts at 0 (the first token is always prompt), which makes
    zero-filling and a wrap-around roll numerically identical — so normal fixtures cannot
    tell a correct implementation from a wrapping one. `_tokenize` with truncation='left'
    keeps `loss_mask[-max_length:]`, so a completion longer than max_length drops the prompt
    entirely and yields an all-ones mask. Only then does the last position matter: verl rolls
    labels over the FLATTENED batch, so token_logps[-1] scores the NEXT sequence's first
    token and must never be summed.
    """
    token_logps = _nested([[1.0, 2.0, 999.0], [4.0, 5.0, 888.0]])
    all_completion = _nested([[1, 1, 1], [1, 1, 1]], dtype=torch.long)

    actual = sequence_logps(token_logps, all_completion)

    # shifted mask = [1, 1, 0]: the trailing cross-sequence garbage is excluded.
    torch.testing.assert_close(actual, torch.tensor([3.0, 9.0]))


def test_sequence_logps_reject_empty_completion():
    with pytest.raises(ValueError, match="no completion tokens"):
        sequence_logps(_nested([[1.0, 2.0]]), _nested([[0, 0]], dtype=torch.long))


def test_sequence_logps_preserve_gradient_through_jagged_values():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    offsets = torch.tensor([0, 3, 5])
    token_logps = torch.nested.nested_tensor_from_jagged(values, offsets)
    loss_mask = torch.nested.nested_tensor_from_jagged(torch.tensor([0, 1, 1, 0, 1]), offsets)

    sequence_logps(token_logps, loss_mask).sum().backward()

    torch.testing.assert_close(values.grad, torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0]))


@pytest.mark.parametrize("reference_free", [False, True])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_sigmoid_dpo_matches_spin_oracle_before_reduction(smoothing, reference_free):
    """Parity against the REAL SPIN implementation, not a restatement of our formula.

    SPIN reduces to a scalar mean (recipe/spin/core_algos.py:158), so the contract is
    ``our_pair_losses.mean() == compute_online_dpo_loss(...)``. Re-deriving the formula
    inside the test would be a tautology: a shared misreading would still pass.
    """
    from recipe.spin.core_algos import compute_online_dpo_loss

    policy_chosen = torch.tensor([-2.0, -1.0, -0.5])
    policy_rejected = torch.tensor([-3.0, -1.5, -0.9])
    ref_chosen = torch.tensor([-2.2, -1.2, -0.4])
    ref_rejected = torch.tensor([-2.8, -1.4, -1.1])
    beta = 0.2

    losses, chosen_rewards, rejected_rewards = sigmoid_dpo_losses(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta=beta,
        label_smoothing=smoothing,
        reference_free=reference_free,
    )
    spin_loss = compute_online_dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta=beta,
        label_smoothing=smoothing,
        loss_type="sigmoid",
        reference_free=reference_free,
    )

    torch.testing.assert_close(losses.mean(), spin_loss)
    torch.testing.assert_close(chosen_rewards, beta * (policy_chosen - ref_chosen))
    torch.testing.assert_close(rejected_rewards, beta * (policy_rejected - ref_rejected))


def test_sequence_logps_matches_independent_padded_forward_under_verl_convention():
    """PR0 oracle: our nested reduction must equal the padded oracle that builds the cache.

    This pins the alignment assumption the whole design rests on. verl builds log_probs as
    ``logprobs_from_logits(logits, torch.roll(input_ids.values(), -1))``
    (fsdp/transformer_impl.py:1018,1214) — i.e. ``log_probs[t]`` scores ``input_ids[t+1]``,
    and the roll runs over the FLATTENED batch, so each sequence's final position scores the
    NEXT sequence's first token (garbage). Reproducing that here proves ``sequence_logps``
    masks the garbage away and still agrees with ``completion_logps_from_logits``, the
    independent padded forward used offline to write ref_logprob_* into the parquet.
    Reference logp and policy logp MUST share one definition or every logratio is wrong.

    We reproduce verl's INDEXING (the -1 roll) but score it with cross_entropy rather than
    verl's `logprobs_from_logits`: that helper dispatches to a Triton kernel and refuses CPU
    tensors. The two are mathematically identical — and the alignment, not the kernel, is
    what this test pins.
    """
    from alphaapollo.data_preprocess.prepare_dpo_reference import completion_logps_from_logits

    torch.manual_seed(0)
    vocab_size = 16
    sequences = [
        {"input_ids": torch.tensor([3, 5, 9, 2]), "loss_mask": torch.tensor([0, 0, 1, 1])},
        {"input_ids": torch.tensor([7, 1, 4]), "loss_mask": torch.tensor([0, 1, 1])},
    ]
    logits_rows = [torch.randn(item["input_ids"].shape[0], vocab_size) for item in sequences]

    # --- verl's engine convention, reproduced exactly (flattened roll included) ---
    flat_input_ids = torch.cat([item["input_ids"] for item in sequences])
    flat_logits = torch.cat(logits_rows, dim=0)
    rolled_labels = torch.roll(flat_input_ids, shifts=-1, dims=0)
    flat_token_logps = -F.cross_entropy(flat_logits, rolled_labels, reduction="none")
    offsets = torch.tensor([0, 4, 7])
    nested_token_logps = torch.nested.nested_tensor_from_jagged(flat_token_logps, offsets)
    nested_loss_mask = torch.nested.nested_tensor_from_jagged(
        torch.cat([item["loss_mask"] for item in sequences]), offsets
    )

    actual = sequence_logps(nested_token_logps, nested_loss_mask)

    # --- independent oracle: the padded HF-style forward used by the reference cache ---
    max_length = max(item["input_ids"].shape[0] for item in sequences)
    padded_logits = torch.zeros(len(sequences), max_length, vocab_size)
    padded_input_ids = torch.zeros(len(sequences), max_length, dtype=torch.long)
    padded_loss_mask = torch.zeros(len(sequences), max_length, dtype=torch.bool)
    for index, (item, row_logits) in enumerate(zip(sequences, logits_rows, strict=True)):
        length = item["input_ids"].shape[0]
        padded_logits[index, :length] = row_logits
        padded_input_ids[index, :length] = item["input_ids"]
        padded_loss_mask[index, :length] = item["loss_mask"].to(torch.bool)
    expected = completion_logps_from_logits(padded_logits, padded_input_ids, padded_loss_mask)

    torch.testing.assert_close(actual, expected)


def test_dpo_loss_hook_emits_pair_metrics():
    model_output = {
        "log_probs": _nested(
            [
                [0.0, -1.0, -2.0],
                [0.0, -2.0, -3.0],
                [0.0, -1.5, -2.5],
                [0.0, -1.0, -1.5],
            ]
        )
    }
    # global_batch_size counts PAIRS, not the collator's 2N flattened rows: DPOTrainer sets it
    # from data.train_batch_size (dpo.py:344) before collation, and PairedPreferenceDataset's
    # unit is one pair. 4 rows here = 2 pairs on 1 rank => global_batch_size=2.
    data = {
        "loss_mask": _nested([[0, 1, 1]] * 4, dtype=torch.long),
        "reference_logp": torch.tensor([-3.2, -4.8, -4.1, -2.7]),
        "pair_id": torch.tensor([7, 7, 9, 9]),
        "is_chosen": torch.tensor([True, False, True, False]),
        "global_batch_size": 2,
        "dp_size": 1,
    }

    loss, metrics = make_dpo_loss_fn(beta=0.1)(model_output=model_output, data=data)

    policy = torch.tensor([-1.0, -2.0, -1.5, -1.0])
    pair_losses, _, _ = sigmoid_dpo_losses(
        policy[0::2],
        policy[1::2],
        data["reference_logp"][0::2],
        data["reference_logp"][1::2],
        beta=0.1,
    )
    torch.testing.assert_close(loss, pair_losses.mean())  # 1 rank, 2 pairs => plain pair mean
    assert set(metrics) >= {"dpo/loss", "dpo/reward_accuracy", "dpo/reward_margin"}
    assert 0.0 <= metrics["dpo/reward_accuracy"] <= 1.0


def test_global_pair_normalization_aggregates_to_the_global_pair_mean_under_dp():
    """`sum / global_pair_batch_size * dp_size` must average to the global PAIR mean.

    FSDP/DDP average gradients across ranks, so each rank scales by dp_size to compensate —
    the same trick `sft_loss` plays with `/ batch_num_tokens * dp_size` (losses.py:49), except
    our unit is a pair, not a token. This shards 4 pairs over 2 ranks and checks the
    post-aggregation value, which is what a wrong denominator (e.g. reading global_batch_size
    as the 2N flattened row count) would silently halve.
    """
    torch.manual_seed(0)
    logps = torch.tensor([-1.0, -2.0, -1.5, -1.0, -0.5, -1.2, -2.0, -0.8])
    reference_logp = torch.tensor([-3.2, -4.8, -4.1, -2.7, -1.0, -2.0, -3.0, -1.5])
    pair_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    is_chosen = torch.tensor([True, False] * 4)
    global_pairs, dp_size = 4, 2

    def rank_loss(rank: int) -> torch.Tensor:
        rows = slice(rank * 4, rank * 4 + 4)  # 2 pairs = 4 flattened rows per rank
        # loss_mask [0, 1] shifts to [1, 0], so token_logps[0] — which scores the completion
        # token input_ids[1] — is the sequence logp; index 1 is the masked-away tail.
        model_output = {"log_probs": _nested([[value.item(), 0.0] for value in logps[rows]])}
        data = {
            "loss_mask": _nested([[0, 1]] * 4, dtype=torch.long),
            "reference_logp": reference_logp[rows],
            "pair_id": pair_ids[rows],
            "is_chosen": is_chosen[rows],
            "global_batch_size": global_pairs,
            "dp_size": dp_size,
        }
        loss, _ = make_dpo_loss_fn(beta=0.1)(model_output=model_output, data=data)
        return loss

    aggregated = torch.stack([rank_loss(0), rank_loss(1)]).mean()  # what FSDP does

    all_pairs, _, _ = sigmoid_dpo_losses(
        logps[0::2], logps[1::2], reference_logp[0::2], reference_logp[1::2], beta=0.1
    )
    torch.testing.assert_close(aggregated, all_pairs.mean())


def test_dpo_loss_hook_rejects_reordered_pairs():
    model_output = {"log_probs": _nested([[0.0, -1.0], [0.0, -2.0]])}
    data = {
        "loss_mask": _nested([[0, 1], [0, 1]], dtype=torch.long),
        "reference_logp": torch.zeros(2),
        "pair_id": torch.tensor([1, 2]),
        "is_chosen": torch.tensor([True, False]),
        "global_batch_size": 1,
        "dp_size": 1,
    }
    with pytest.raises(ValueError, match="split or reordered"):
        make_dpo_loss_fn()(model_output=model_output, data=data)
