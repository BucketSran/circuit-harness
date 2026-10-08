"""Integration contract against the exact verl revision pinned by this repository."""

from __future__ import annotations

import numpy as np
import pytest

from alphaapollo.common.execution import ArtifactStore
from alphaapollo.learning.adapters import (
    OUTPUT_TO_INPUT_TENSOR_FIELD,
    DataProtoAdapter,
)
from tests.learning.capture_fixtures import inference_record

torch = pytest.importorskip("torch")
DataProto = pytest.importorskip("verl.protocol").DataProto


def test_real_verl_dataproto_round_trip_preserves_b_by_n_alignment(tmp_path) -> None:
    """Capture B=2 -> B*n=4 and restore a genuine verl DataProto on CPU."""

    output_tensors = {
        "prompts": torch.tensor([[10, 11], [10, 11], [20, 21], [20, 21]]),
        "responses": torch.tensor([[30, 31], [32, 33], [40, 41], [42, 43]]),
        "input_ids": torch.tensor(
            [
                [10, 11, 30, 31],
                [10, 11, 32, 33],
                [20, 21, 40, 41],
                [20, 21, 42, 43],
            ]
        ),
        "attention_mask": torch.ones((4, 4), dtype=torch.int64),
        "position_ids": torch.arange(4, dtype=torch.int64).repeat(4, 1),
    }

    store = ArtifactStore(tmp_path)
    record = inference_record(
        store,
        tensor_batch=output_tensors,
        non_tensor_batch={"sample_id": np.array(["p0-0", "p0-1", "p1-0", "p1-1"])},
        meta_info={"backend": "contract-test"},
        input_batch_size=2,
        output_to_input=np.array([0, 0, 1, 1], dtype=np.int64),
        model="contract-model",
    )
    restored = DataProtoAdapter(
        store,
        dataproto_class=DataProto,
        output_to_input_tensor_field=OUTPUT_TO_INPUT_TENSOR_FIELD,
    ).restore(record)

    assert isinstance(restored, DataProto)
    assert len(restored) == 4
    assert set(output_tensors).issubset(restored.batch.keys())
    assert "rollout_log_probs" not in restored.batch.keys()
    assert torch.equal(
        restored.batch[OUTPUT_TO_INPUT_TENSOR_FIELD],
        torch.tensor([0, 0, 1, 1], dtype=torch.int64),
    )
    assert restored.non_tensor_batch["sample_id"].tolist() == [
        "p0-0",
        "p0-1",
        "p1-0",
        "p1-1",
    ]
    assert restored.meta_info == {"backend": "contract-test"}
