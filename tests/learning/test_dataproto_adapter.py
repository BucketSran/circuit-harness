from __future__ import annotations

import sys

import numpy as np
import pytest

from alphaapollo.common.execution import ArtifactStore
from alphaapollo.learning.adapters import (
    OUTPUT_TO_INPUT_TENSOR_FIELD,
    DataProtoAdapter,
    DataProtoRestoreError,
)
from tests.learning.capture_fixtures import inference_record


class FakeDataProto:
    @classmethod
    def from_dict(cls, *, tensors, non_tensors, meta_info):
        return {
            "tensors": tensors,
            "non_tensors": non_tensors,
            "meta_info": meta_info,
        }


def _rollout_tensors(batch_size: int = 2) -> dict[str, np.ndarray]:
    return {
        "prompts": np.arange(batch_size * 3).reshape(batch_size, 3),
        "responses": np.arange(batch_size * 2).reshape(batch_size, 2),
        "input_ids": np.arange(batch_size * 5).reshape(batch_size, 5),
        "rollout_log_probs": np.zeros((batch_size, 2), dtype=np.float32),
        "attention_mask": np.ones((batch_size, 5), dtype=np.int64),
        "position_ids": np.tile(np.arange(5), (batch_size, 1)),
    }


def _record(tmp_path):
    store = ArtifactStore(tmp_path)
    raw_prompt_ids = np.empty(2, dtype=object)
    raw_prompt_ids[0] = [1, 2]
    raw_prompt_ids[1] = [3, 4, 5]
    record = inference_record(
        store,
        tensor_batch=_rollout_tensors(),
        non_tensor_batch={"raw_prompt_ids": raw_prompt_ids},
        meta_info={"eos_token_id": 2},
        model="m",
    )
    return store, record


def _adapter(store):
    return DataProtoAdapter(
        store,
        dataproto_class=FakeDataProto,
        tensor_factory=lambda array: array,
    )


def _aligned_record(tmp_path):
    store = ArtifactStore(tmp_path)
    record = inference_record(
        store,
        tensor_batch=_rollout_tensors(batch_size=2),
        input_batch_size=1,
        output_to_input=np.array([0, 0], dtype=np.int64),
        model="m",
    )
    return store, record


def test_restore_round_trips_tensor_object_and_meta_fields(tmp_path) -> None:
    store, record = _record(tmp_path)
    restored = _adapter(store).restore(record)

    np.testing.assert_array_equal(restored["tensors"]["input_ids"], _rollout_tensors()["input_ids"])
    assert restored["non_tensors"]["raw_prompt_ids"].shape == (2,)
    assert restored["non_tensors"]["raw_prompt_ids"].tolist() == [
        [1, 2],
        [3, 4, 5],
    ]
    assert restored["meta_info"] == {"eos_token_id": 2}


def test_restore_core_profile_allows_missing_optional_rollout_log_probs(tmp_path) -> None:
    store, record = _record(tmp_path)
    tensors = dict(record.tensor_batch)
    tensors.pop("rollout_log_probs")

    restored = _adapter(store).restore(record.model_copy(update={"tensor_batch": tensors}))

    assert "rollout_log_probs" not in restored["tensors"]


def test_output_to_input_mapping_is_opt_in_and_namespaced(tmp_path) -> None:
    store, record = _aligned_record(tmp_path)

    default_restored = _adapter(store).restore(record)
    assert OUTPUT_TO_INPUT_TENSOR_FIELD not in default_restored["tensors"]

    opted_in = DataProtoAdapter(
        store,
        dataproto_class=FakeDataProto,
        tensor_factory=lambda array: array,
        output_to_input_tensor_field=OUTPUT_TO_INPUT_TENSOR_FIELD,
    ).restore(record)
    np.testing.assert_array_equal(
        opted_in["tensors"][OUTPUT_TO_INPUT_TENSOR_FIELD],
        np.array([0, 0], dtype=np.int64),
    )


def test_output_to_input_mapping_refuses_tensor_name_collision(tmp_path) -> None:
    store, record = _aligned_record(tmp_path)
    adapter = DataProtoAdapter(
        store,
        dataproto_class=FakeDataProto,
        tensor_factory=lambda array: array,
        output_to_input_tensor_field="input_ids",
    )

    with pytest.raises(DataProtoRestoreError, match="collides"):
        adapter.restore(record)


def test_learning_profile_can_require_rollout_log_probs(tmp_path) -> None:
    store, record = _record(tmp_path)
    tensors = dict(record.tensor_batch)
    tensors.pop("rollout_log_probs")
    adapter = DataProtoAdapter(
        store,
        dataproto_class=FakeDataProto,
        tensor_factory=lambda array: array,
        required_tensor_fields=frozenset({"rollout_log_probs"}),
    )

    with pytest.raises(DataProtoRestoreError, match="rollout_log_probs"):
        adapter.restore(record.model_copy(update={"tensor_batch": tensors}))


def test_restore_rejects_unsupported_schema_and_missing_required_field(tmp_path) -> None:
    store, record = _record(tmp_path)
    with pytest.raises(DataProtoRestoreError, match="unsupported inference"):
        _adapter(store).restore(record.model_copy(update={"schema_version": "2.0"}))

    incomplete = record.model_copy(
        update={
            "tensor_batch": {
                key: value for key, value in record.tensor_batch.items() if key != "responses"
            }
        }
    )
    with pytest.raises(DataProtoRestoreError, match="responses"):
        _adapter(store).restore(incomplete)


@pytest.mark.parametrize(
    ("field_update", "message"),
    [
        ({"shape": [99, 5]}, "shape"),
        ({"dtype": "float64"}, "dtype"),
        ({"encoding": "json-array"}, "tensor field cannot use"),
    ],
)
def test_restore_rejects_tensor_manifest_mismatch(tmp_path, field_update, message) -> None:
    store, record = _record(tmp_path)
    tensors = dict(record.tensor_batch)
    tensors["input_ids"] = tensors["input_ids"].model_copy(update=field_update)
    with pytest.raises(DataProtoRestoreError, match=message):
        _adapter(store).restore(record.model_copy(update={"tensor_batch": tensors}))


def test_restore_rejects_invalid_non_tensor_encoding_and_json(tmp_path) -> None:
    store, record = _record(tmp_path)
    non_tensors = dict(record.non_tensor_batch)
    original = non_tensors["raw_prompt_ids"]
    non_tensors["raw_prompt_ids"] = original.model_copy(update={"encoding": "torch-raw"})
    with pytest.raises(DataProtoRestoreError, match="non-tensor field cannot use"):
        _adapter(store).restore(record.model_copy(update={"non_tensor_batch": non_tensors}))

    invalid = store.put(
        b"not-json",
        type_="application/x-alphaapollo-array",
        created_by="test",
    )
    non_tensors["raw_prompt_ids"] = original.model_copy(update={"artifact": store.ref(invalid)})
    with pytest.raises(DataProtoRestoreError, match="invalid JSON"):
        _adapter(store).restore(record.model_copy(update={"non_tensor_batch": non_tensors}))


def test_restore_rejects_batch_size_mismatch(tmp_path) -> None:
    store, record = _record(tmp_path)
    with pytest.raises(DataProtoRestoreError, match="batch_size=9"):
        _adapter(store).restore(record.model_copy(update={"batch_size": 9}))


def test_torch_restore_preserves_bfloat16_and_rejects_byte_order(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    store = ArtifactStore(tmp_path)
    fields = {
        "prompts": torch.tensor([[1, 2]], dtype=torch.int64),
        "responses": torch.tensor([[3, 4]], dtype=torch.int64),
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
        "rollout_log_probs": torch.tensor([[-0.2, -0.3]], dtype=torch.bfloat16),
        "attention_mask": torch.ones((1, 4), dtype=torch.int64),
        "position_ids": torch.arange(4, dtype=torch.int64).reshape(1, 4),
    }
    record = inference_record(store, tensor_batch=fields, model="m")
    restored = DataProtoAdapter(store, dataproto_class=FakeDataProto).restore(record)["tensors"]
    for key, original in fields.items():
        assert restored[key].dtype == original.dtype
        assert torch.equal(restored[key], original)

    tensors = dict(record.tensor_batch)
    opposite = "big" if sys.byteorder == "little" else "little"
    tensors["input_ids"] = tensors["input_ids"].model_copy(update={"byte_order": opposite})
    with pytest.raises(DataProtoRestoreError, match="byte order"):
        DataProtoAdapter(store, dataproto_class=FakeDataProto).restore(
            record.model_copy(update={"tensor_batch": tensors})
        )
