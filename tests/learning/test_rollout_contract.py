from __future__ import annotations

import numpy as np
import pytest

from alphaapollo.learning.adapters import RolloutDataProtoAdapter
from alphaapollo.learning.rollout_contract import (
    LearningTrajectory,
    LearningTrajectoryBatch,
    RolloutBatch,
    RolloutContractError,
)


class FakeDataProto:
    @classmethod
    def from_dict(cls, *, tensors, non_tensors, meta_info):
        return {
            "tensors": tensors,
            "non_tensors": non_tensors,
            "meta_info": meta_info,
        }


def _tensors(batch_size: int = 3) -> dict[str, np.ndarray]:
    return {
        "prompts": np.arange(batch_size * 2).reshape(batch_size, 2),
        "responses": np.arange(batch_size * 2).reshape(batch_size, 2),
        "response_mask": np.ones((batch_size, 2), dtype=np.int64),
        "input_ids": np.arange(batch_size * 4).reshape(batch_size, 4),
        "attention_mask": np.ones((batch_size, 4), dtype=np.int64),
        "position_ids": np.tile(np.arange(4), (batch_size, 1)),
        "rollout_log_probs": np.zeros((batch_size, 2), dtype=np.float32),
        "rm_scores": np.zeros((batch_size, 2), dtype=np.float32),
    }


def _parity_fixture() -> RolloutBatch:
    """Variable-length, invalid-action, terminal-success rollout rows."""

    return RolloutBatch(
        tensors=_tensors(),
        non_tensors={
            "traj_uid": np.array(["short", "long", "long"], dtype=object),
            "step_index": np.array([0, 0, 1]),
            "original_step_index": np.array([0, 0, 2]),
            "raw_env_reward": np.array([1.0, 0.0, -0.2]),
            "done": np.array([True, False, True]),
            "is_success_terminal": np.array([True, False, False]),
            "is_response_format_valid": np.array([True, False, True]),
            "is_env_action_valid": np.array([True, True, False]),
            "is_action_valid": np.array([True, False, False]),
        },
        meta_info={
            "policy_version": "policy-7",
            "checkpoint": "/checkpoints/step-42",
            "sampling_config": {"temperature": 0.7, "top_p": 0.9},
            "seed": 17,
            "environment_config": {"name": "fixture"},
            "code_sha": "abc123",
        },
    )


def test_rollout_dataproto_adapter_preserves_every_field_without_mutation() -> None:
    rollout = _parity_fixture()
    converted = RolloutDataProtoAdapter(dataproto_class=FakeDataProto).convert(rollout)

    assert converted["meta_info"] == dict(rollout.meta_info)
    assert converted["tensors"].keys() == rollout.tensors.keys()
    assert converted["non_tensors"].keys() == rollout.non_tensors.keys()
    for key, expected in rollout.tensors.items():
        np.testing.assert_array_equal(converted["tensors"][key], expected)
    for key, expected in rollout.non_tensors.items():
        np.testing.assert_array_equal(converted["non_tensors"][key], expected)


def test_parity_fixture_keeps_cleaned_to_original_step_mapping() -> None:
    rollout = _parity_fixture()

    assert rollout.non_tensors["step_index"].tolist() == [0, 0, 1]
    assert rollout.non_tensors["original_step_index"].tolist() == [0, 0, 2]
    assert rollout.non_tensors["raw_env_reward"].tolist() == [1.0, 0.0, -0.2]


def test_rollout_contract_rejects_missing_or_misaligned_fields() -> None:
    incomplete = _tensors()
    incomplete.pop("rollout_log_probs")
    with pytest.raises(RolloutContractError, match="rollout_log_probs"):
        RolloutBatch(tensors=incomplete, non_tensors={"done": np.zeros(3)})

    missing_mask = _tensors()
    missing_mask.pop("response_mask", None)
    with pytest.raises(RolloutContractError, match="response_mask"):
        RolloutBatch(tensors=missing_mask, non_tensors={"done": np.zeros(3)})

    with pytest.raises(RolloutContractError, match="row-aligned"):
        RolloutBatch(tensors=_tensors(), non_tensors={"done": np.zeros(2)})


def test_learning_trajectory_batch_rejects_duplicate_identities() -> None:
    trajectory = LearningTrajectory(
        task_id="task",
        trajectory_id="duplicate",
        group_id="group",
        sample_id=0,
        data_source="test",
        initial_observation="prompt",
        turns=(),
        termination_reason="final",
    )

    with pytest.raises(ValueError, match="identities must be unique"):
        LearningTrajectoryBatch((trajectory, trajectory))
