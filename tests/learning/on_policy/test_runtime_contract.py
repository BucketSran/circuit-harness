"""Pinned-verl contracts for rollout generation, reward, and configuration."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
DataProto = pytest.importorskip("verl").DataProto

collector_module = import_module("alphaapollo.learning.on_policy.rollout.collector")
episode_module = import_module("alphaapollo.learning.on_policy.reward_manager.episode")
penalty_module = import_module("alphaapollo.learning.on_policy.reward_manager.penalty")
provenance_module = import_module("alphaapollo.learning.on_policy.rollout.provenance")

TrajectoryCollector = collector_module.TrajectoryCollector
EpisodeRewardManager = episode_module.EpisodeRewardManager


def test_tokenizer_provenance_changes_with_revision_and_template() -> None:
    base = SimpleNamespace(
        _commit_hash="revision-a",
        init_kwargs={},
        vocab_size=100,
        added_tokens_encoder={"<special>": 100},
        chat_template="template-a",
    )

    first = provenance_module.tokenizer_fingerprints(base, "tokenizer")
    revised = provenance_module.tokenizer_fingerprints(
        SimpleNamespace(**{**vars(base), "_commit_hash": "revision-b"}),
        "tokenizer",
    )
    retemplated = provenance_module.tokenizer_fingerprints(
        SimpleNamespace(**{**vars(base), "chat_template": "template-b"}),
        "tokenizer",
    )

    assert first[0] != revised[0]
    assert first[1] == revised[1]
    assert first[0] == retemplated[0]
    assert first[1] != retemplated[1]


def test_rollout_version_is_published_only_after_weight_sync_succeeds() -> None:
    events = []
    checkpoint_manager = SimpleNamespace(update_weights=lambda step: events.append(("sync", step)))
    generation_backend = SimpleNamespace(
        update_weights_version=lambda version: events.append(("publish", version))
    )

    sync_count = provenance_module.synchronize_rollout_weights(
        checkpoint_manager=checkpoint_manager,
        generation_backend=generation_backend,
        global_step=12,
        previous_sync_count=3,
    )

    assert events == [("sync", 12), ("publish", "step-12-sync-4")]
    assert sync_count == 4

    def fail(_step):
        raise RuntimeError("sync failed")

    checkpoint_manager.update_weights = fail
    events.clear()
    with pytest.raises(RuntimeError, match="sync failed"):
        provenance_module.synchronize_rollout_weights(
            checkpoint_manager=checkpoint_manager,
            generation_backend=generation_backend,
            global_step=12,
            previous_sync_count=4,
        )
    assert events == []


def _object_array(*values):
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def test_collector_requires_reasoning_backend_after_legacy_producer_removal() -> None:
    collector = TrajectoryCollector(config={}, tokenizer=None)

    with pytest.raises(RuntimeError, match="legacy rollout producer has been removed"):
        collector.vanilla_multi_turn_loop(
            gen_batch=SimpleNamespace(),
            envs=SimpleNamespace(),
        )


def test_episode_reward_and_invalid_action_penalty_handle_empty_response() -> None:
    data = DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[1, 2], [1, 2]], dtype=torch.long),
            "responses": torch.tensor([[3, 0], [0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "episode_rewards": np.array([2.0, 3.0]),
            "episode_lengths": np.array([4.0, 0.0]),
            "data_source": _object_array("test", "test"),
        }
    )
    manager = EpisodeRewardManager(
        tokenizer=SimpleNamespace(), num_examine=0, normalize_by_length=True
    )
    reward = manager(data)

    assert reward.tolist() == [[0.5, 0.0], [0.0, 0.0]]

    data.batch["token_level_scores"] = reward.clone()
    data.batch["step_rewards"] = torch.zeros(2, 1)
    data.non_tensor_batch["is_action_valid"] = np.array([False, False])
    penalized, result_metrics = penalty_module.apply_invalid_action_penalty(data, 0.1)

    assert penalized.batch["token_level_scores"][0, 0].item() == pytest.approx(0.4)
    assert penalized.batch["token_level_scores"][1].tolist() == [0.0, 0.0]
    assert penalized.batch["step_rewards"][:, 0].tolist() == pytest.approx([-0.1, 0.0])
    assert result_metrics == {"episode/valid_action_ratio": 0.0}


def test_reward_and_penalty_prefer_canonical_explicit_signals() -> None:
    data = DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[1, 2]], dtype=torch.long),
            "responses": torch.tensor([[3, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "episode_reward": np.array([2.0]),
            "episode_length": np.array([4.0]),
            "episode_rewards": np.array([99.0]),
            "episode_lengths": np.array([1.0]),
            "data_source": _object_array("test"),
        }
    )
    manager = EpisodeRewardManager(
        tokenizer=SimpleNamespace(), num_examine=0, normalize_by_length=True
    )

    reward = manager(data)

    assert reward.tolist() == [[0.5, 0.0]]

    data.batch["token_level_scores"] = reward.clone()
    data.non_tensor_batch["is_response_format_valid"] = np.array([True])
    data.non_tensor_batch["is_env_action_valid"] = np.array([False])
    penalized, metrics = penalty_module.apply_invalid_action_penalty(data, 0.1)

    assert penalized.batch["token_level_scores"][0, 0].item() == pytest.approx(0.4)
    assert metrics == {"episode/valid_action_ratio": 0.0}


def _rollout_row(traj_uid: str, *, active: bool = True, response_id: int = 3) -> dict:
    return {
        "active_masks": active,
        "traj_uid": traj_uid,
        "prompts": torch.tensor([1, 2], dtype=torch.long),
        "responses": torch.tensor([response_id], dtype=torch.long),
        "response_mask": torch.tensor([1], dtype=torch.long),
        "input_ids": torch.tensor([1, 2, response_id], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        "position_ids": torch.tensor([0, 1, 2], dtype=torch.long),
        "rollout_log_probs": torch.tensor([-0.2]),
        "rm_scores": torch.zeros(1),
    }


def test_gather_preserves_per_trajectory_success_separately_from_batch_rate() -> None:
    collector = object.__new__(TrajectoryCollector)
    collector.config = {"alphaapollo": {"normalize_by_length": False}}
    trajectories = [
        [_rollout_row("success"), _rollout_row("success")],
        [_rollout_row("failure")],
    ]

    output = collector.gather_rollout_data(
        total_batch_list=trajectories,
        episode_rewards=np.array([1.0, 0.7]),
        episode_lengths=np.array([2.0, 1.0]),
        success={"success_rate": np.array([1.0, 0.0])},
        traj_uid=np.array(["success", "failure"], dtype=object),
        tool_callings=np.zeros(2),
        meta_info={"policy_version": "policy-7", "code_sha": "abc123"},
    )

    np.testing.assert_array_equal(
        output.non_tensor_batch["is_success_terminal"],
        [1.0, 1.0, 0.0],
    )
    np.testing.assert_array_equal(
        output.non_tensor_batch["success_rate"],
        [0.5, 0.5, 0.5],
    )
    np.testing.assert_array_equal(
        output.non_tensor_batch["episode_reward"],
        output.non_tensor_batch["episode_rewards"],
    )
    np.testing.assert_array_equal(
        output.non_tensor_batch["episode_length"],
        output.non_tensor_batch["episode_lengths"],
    )
    assert output.meta_info == {"policy_version": "policy-7", "code_sha": "abc123"}


def test_gather_drops_noop_rows_without_shifting_original_step_credit() -> None:
    collector = object.__new__(TrajectoryCollector)
    collector.config = {"alphaapollo": {"normalize_by_length": False}}
    kept_first = _rollout_row("trajectory", response_id=10)
    rejected_noop = _rollout_row("trajectory", active=False, response_id=11)
    kept_last = _rollout_row("trajectory", response_id=12)
    for original_step, row in enumerate((kept_first, rejected_noop, kept_last)):
        row["original_step_index"] = original_step

    output = collector.gather_rollout_data(
        total_batch_list=[[kept_first, rejected_noop, kept_last]],
        episode_rewards=np.array([1.0]),
        episode_lengths=np.array([2.0]),
        success={"success_rate": np.array([1.0])},
        traj_uid=np.array(["trajectory"], dtype=object),
        tool_callings=np.zeros(1),
    )

    assert output.non_tensor_batch["original_step_index"].tolist() == [0, 2]
    assert output.batch["responses"].squeeze(-1).tolist() == [10, 12]
    assert output.batch["rm_scores"].squeeze(-1).tolist() == [1.0, 1.0]


def test_dynamic_rollout_combines_sparse_task_metrics_across_attempts() -> None:
    combined = collector_module._concatenate_success_metrics(
        [
            {
                "success_rate": np.array([1.0, 0.0]),
                "robotics/task_a_success_rate": np.array([1.0]),
            },
            {
                "success_rate": np.array([0.0]),
                "robotics/task_b_success_rate": np.array([0.0]),
            },
        ]
    )

    np.testing.assert_array_equal(combined["success_rate"], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(combined["robotics/task_a_success_rate"], [1.0])
    np.testing.assert_array_equal(combined["robotics/task_b_success_rate"], [0.0])
