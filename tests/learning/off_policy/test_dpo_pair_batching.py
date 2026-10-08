# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DPO pairs must survive verl's REAL micro-batching path, not just our own guard.

``_validate_and_split_pairs`` checks the contract at loss time; these tests check the
mechanism that has to uphold it — ``verl.workers.engine.utils.prepare_micro_batches``.
"""

from __future__ import annotations

import pytest
import torch
from verl.utils import tensordict_utils as tu
from verl.workers.engine.utils import prepare_micro_batches

from alphaapollo.data_preprocess.preference_dataset import (
    REF_CHOSEN_COLUMN,
    REF_REJECTED_COLUMN,
    PairedPreferenceCollator,
)


def _pair_item(pair_id: int, *, length: int = 4):
    sequence = {
        "input_ids": torch.arange(length, dtype=torch.long),
        "position_ids": torch.arange(length, dtype=torch.long),
        "loss_mask": torch.tensor([0] + [1] * (length - 1), dtype=torch.long),
    }
    return {
        "pair_id": pair_id,
        "sample_id": str(pair_id),
        "chosen": {key: value.clone() for key, value in sequence.items()},
        "rejected": {key: value.clone() for key, value in sequence.items()},
        REF_CHOSEN_COLUMN: -1.0 * pair_id,
        REF_REJECTED_COLUMN: -2.0 * pair_id,
    }


def _collated_tensordict(num_pairs: int, micro_batch_size_per_gpu: int):
    collated = PairedPreferenceCollator()([_pair_item(i) for i in range(num_pairs)])
    return tu.get_tensordict(
        tensor_dict=collated,
        non_tensor_dict={
            "use_dynamic_bsz": False,
            "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        },
    )


def test_static_chunking_keeps_each_pair_inside_one_micro_batch():
    data = _collated_tensordict(num_pairs=6, micro_batch_size_per_gpu=4)

    micro_batches, _ = prepare_micro_batches(data=data, dp_group=None)

    assert len(micro_batches) == 3  # 12 flattened sequences / 4 per micro-batch
    seen_pairs: list[int] = []
    for micro_batch in micro_batches:
        pair_ids = micro_batch["pair_id"].reshape(-1, 2)
        is_chosen = micro_batch["is_chosen"].reshape(-1, 2)
        # both rows of a pair land in the same micro-batch, adjacent, chosen-first
        assert torch.equal(pair_ids[:, 0], pair_ids[:, 1])
        assert torch.equal(is_chosen[:, 0], torch.ones_like(is_chosen[:, 0]))
        assert torch.equal(is_chosen[:, 1], torch.zeros_like(is_chosen[:, 1]))
        seen_pairs.extend(pair_ids[:, 0].tolist())

    assert seen_pairs == list(range(6))  # order preserved, nothing dropped or duplicated


def test_odd_micro_batch_size_would_split_pairs_and_is_rejected_upstream():
    """An odd flattened micro-batch size tears pairs apart — hence DPOTrainer's even check.

    prepare_micro_batches itself is happy to cut mid-pair; nothing in verl protects us.
    This pins WHY `_build_dataloader` rejects an odd micro_batch_size_per_gpu.
    """
    data = _collated_tensordict(num_pairs=6, micro_batch_size_per_gpu=3)

    micro_batches, _ = prepare_micro_batches(data=data, dp_group=None)

    owners: dict[int, set[int]] = {}
    for index, micro_batch in enumerate(micro_batches):
        for pair_id in micro_batch["pair_id"].tolist():
            owners.setdefault(pair_id, set()).add(index)

    torn = {pair_id for pair_id, indices in owners.items() if len(indices) > 1}
    assert torn, "expected an odd micro-batch size to split pairs across micro-batches"


def test_force_group_size_two_silently_doubles_the_static_micro_batch():
    """Guard the coupling trap: fgs is NOT a free safety net on the static path.

    verl computes ``chunk_tensordict(data, total // (mbs * fgs))`` — the 2nd arg is a
    CHUNK COUNT, so fgs=2 reinterprets micro_batch_size_per_gpu as a pair count and
    doubles the sequences per micro-batch. DPO v1 therefore must not inject it.
    """
    without_fgs = _collated_tensordict(num_pairs=8, micro_batch_size_per_gpu=4)
    with_fgs = _collated_tensordict(num_pairs=8, micro_batch_size_per_gpu=4)
    tu.assign_non_tensor(with_fgs, force_group_size=2)

    plain, _ = prepare_micro_batches(data=without_fgs, dp_group=None)
    grouped, _ = prepare_micro_batches(data=with_fgs, dp_group=None)

    assert len(plain[0]["pair_id"]) == 4
    assert len(grouped[0]["pair_id"]) == 8  # doubled — the trap
    assert len(plain) == 2 * len(grouped)


@pytest.mark.parametrize("num_pairs, micro_batch_size_per_gpu", [(3, 4), (5, 4)])
def test_indivisible_flattened_batch_fails_loud(num_pairs, micro_batch_size_per_gpu):
    data = _collated_tensordict(
        num_pairs=num_pairs, micro_batch_size_per_gpu=micro_batch_size_per_gpu
    )

    with pytest.raises(AssertionError, match="divisible"):
        prepare_micro_batches(data=data, dp_group=None)
