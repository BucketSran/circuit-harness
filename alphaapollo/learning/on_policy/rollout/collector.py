# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

"""Learning-owned multi-turn rollout collection for the pinned verl runtime."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn

from alphaapollo.common.environment.provider import EnvironmentProvider
from alphaapollo.learning.adapters import RolloutDataProtoAdapter
from alphaapollo.learning.on_policy.rollout.reasoning_consumer import ReasoningRolloutConsumer
from alphaapollo.learning.on_policy.rollout.utils import filter_group_data
from alphaapollo.learning.rollout_contract import LearningTrajectoryBatch, RolloutBatch


def _trajectory_success_metrics(
    trajectories: LearningTrajectoryBatch,
) -> dict[str, np.ndarray]:
    """Aggregate task outcomes from canonical trajectory transitions."""

    metrics: dict[str, list[float]] = {"success_rate": []}
    for trajectory in trajectories.trajectories:
        terminal = trajectory.turns[-1] if trajectory.turns else None
        won = float(bool(terminal and terminal.terminal_success))
        metrics["success_rate"].append(won)
    return {key: np.asarray(values, dtype=np.float32) for key, values in metrics.items()}


def _concatenate_success_metrics(
    batches: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Combine sparse task-specific metrics from dynamic rollout attempts."""

    keys = set().union(*(batch.keys() for batch in batches))
    return {
        key: np.concatenate([batch[key] for batch in batches if key in batch], axis=0)
        for key in keys
    }


class TrajectoryCollector:
    def __init__(
        self,
        config,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Initialize the TrajectoryProcessor class.

        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
        """
        self.config = config
        self.tokenizer = tokenizer
        self.reasoning_consumer: ReasoningRolloutConsumer | None = None
        self.trajectory_postprocessor = None
        self._rollout_is_train = True

    def set_generation_backend(self, backend: Any) -> None:
        """Inject Learning's live rollout backend into the Reasoning producer."""

        self.reasoning_consumer = ReasoningRolloutConsumer(
            backend=backend,
            tokenizer=self.tokenizer,
            config=self.config,
        )

    def set_trajectory_postprocessor(self, postprocessor) -> None:
        """Set the one shared algorithm hook over Learning trajectory records."""

        self.trajectory_postprocessor = postprocessor

    def gather_rollout_data(
        self,
        total_batch_list: list[list[dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: dict[str, np.ndarray],
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
        meta_info: dict[str, Any] | None = None,
    ) -> DataProto:
        """
        Collect per-step rows into the DataProto consumed by the trainer.

        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)
        trajectory_success = np.asarray(success["success_rate"]).reshape(-1)
        if len(trajectory_success) != batch_size:
            raise ValueError(
                "Per-trajectory success count does not match rollout batch size: "
                f"{len(trajectory_success)} != {batch_size}"
            )

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data["traj_uid"], "data is not from the same trajectory"
                if data["active_masks"]:
                    # episode_rewards
                    data["episode_rewards"] = episode_rewards[bs]
                    data["episode_reward"] = episode_rewards[bs]
                    # episode_lengths
                    data["episode_lengths"] = episode_lengths[bs]
                    data["episode_length"] = episode_lengths[bs]

                    # Bake the real episode reward into the kept rm_scores slot at the
                    # last valid response token, mirroring EpisodeRewardManager attribution
                    # (episode.py). episode_rewards/episode_lengths are np scalars here;
                    # rm_scores is the per-row [max_resp_len] float tensor produced by
                    # build_rollout_row. EpisodeRewardManager short-circuits on its
                    # presence and returns this real value instead of zeros.
                    normalize_by_length = self.config.get("alphaapollo", {}).get(
                        "normalize_by_length", False
                    )
                    prompt_length = data["prompts"].shape[-1]
                    valid_response_length = int(data["attention_mask"][prompt_length:].sum())
                    if normalize_by_length:
                        # Guard against a zero episode length (mirrors episode.py):
                        # avoids a ZeroDivisionError in the degenerate empty-episode
                        # case; fall back to the unnormalized reward.
                        _ep_len = float(episode_lengths[bs])
                        score = (
                            float(episode_rewards[bs]) / _ep_len
                            if _ep_len > 0
                            else float(episode_rewards[bs])
                        )
                    else:
                        score = float(episode_rewards[bs])
                    if "rm_scores" not in data:
                        data["rm_scores"] = torch.zeros_like(data["responses"], dtype=torch.float)
                    if valid_response_length > 0:
                        data["rm_scores"][valid_response_length - 1] = score
                    # With no valid response token, the row remains all-zero.

                    # tool_callings
                    data["tool_callings"] = tool_callings[bs]
                    # Preserve the terminal task outcome for this trajectory.
                    # ``success_rate`` below remains the batch aggregate for logging.
                    data["is_success_terminal"] = trajectory_success[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)

        # NOTE: state saver
        # Convert trajectory data to DataProto format
        collated = collate_fn(effective_batch)
        rollout = RolloutBatch.from_collated(collated, meta_info=meta_info)
        return RolloutDataProtoAdapter(dataproto_class=DataProto).convert(rollout)

    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        envs: EnvironmentProvider,
        group_size: int = 1,
    ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            envs (EnvironmentProvider): Common per-episode Environment provider.

        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        if self.reasoning_consumer is None:
            raise RuntimeError(
                "Reasoning GenerationBackend is required; the legacy rollout producer "
                "has been removed"
            )
        return self._reasoning_multi_turn_loop(
            gen_batch=gen_batch,
            envs=envs,
            group_size=group_size,
        )

    def _reasoning_multi_turn_loop(
        self,
        *,
        gen_batch: DataProto,
        envs: EnvironmentProvider,
        group_size: int,
    ):
        """Consume canonical Reasoning results without exposing them to algorithms."""

        assert self.reasoning_consumer is not None
        validate = bool(getattr(gen_batch, "meta_info", {}).get("validate", False))
        trajectories = self.reasoning_consumer.collect(
            gen_batch,
            envs,
            group_size=group_size,
            validate=validate,
        )
        if self.trajectory_postprocessor is not None:
            trajectories = self.trajectory_postprocessor(
                trajectories,
                is_train=self._rollout_is_train,
            )
            if not isinstance(trajectories, LearningTrajectoryBatch):
                raise TypeError("trajectory postprocessor must return LearningTrajectoryBatch")
        total_batch_list = self.reasoning_consumer.to_step_rows(trajectories)
        episode_rewards = np.asarray(
            [sum(float(row["raw_env_reward"]) for row in rows) for rows in total_batch_list],
            dtype=np.float32,
        )
        episode_lengths = np.asarray([len(rows) for rows in total_batch_list], dtype=np.float32)
        tool_callings = np.asarray(
            [sum(float(row["tool_callings"]) for row in rows) for rows in total_batch_list],
            dtype=np.float32,
        )
        success = _trajectory_success_metrics(trajectories)
        return (
            total_batch_list,
            episode_rewards,
            episode_lengths,
            success,
            np.asarray(
                [trajectory.trajectory_id for trajectory in trajectories.trajectories],
                dtype=object,
            ),
            tool_callings,
        )

    def dynamic_multi_turn_loop(
        self,
        gen_batch: DataProto,
        envs: EnvironmentProvider,
    ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met.
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            envs (EnvironmentProvider): Common Environment provider.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        # Absent-safe read — filter_groups may be None when dynamic sampling is disabled.
        # This site is only reached when the multi_turn_loop guard has already confirmed
        # filter_groups is enabled; the defensive read here ensures no AttributeError if ever
        # reached unexpectedly with filter_groups=None.
        _fg = self.config.algorithm.get("filter_groups") or {}
        max_try_count = _fg.get("max_num_gen_batches")
        # Fail-fast on a clear, actionable error instead of an opaque
        # `TypeError: '<' not supported between 'int' and 'NoneType'` mid-rollout.
        # This branch is only reached when filter_groups.enable is True (the
        # multi_turn_loop guard), so max_num_gen_batches MUST be a positive int.
        if max_try_count is None or max_try_count <= 0:
            raise ValueError(
                "algorithm.filter_groups.enable=True requires "
                "algorithm.filter_groups.max_num_gen_batches to be a positive int "
                f"(the dynamic-sampling retry cap); got {max_try_count!r}"
            )

        target_size = self.config.data.train_batch_size * self.config.env.rollout.n
        while len(total_batch_list) < target_size and try_count < max_try_count:
            if len(total_batch_list) > 0:
                print(
                    f"valid trajectories={len(total_batch_list)}; target={target_size}; "
                    f"retry={try_count}/{max_try_count}"
                )
            try_count += 1

            (
                batch_list,
                episode_rewards,
                episode_lengths,
                success,
                traj_uid,
                tool_callings,
            ) = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                envs=envs,
                group_size=self.config.env.rollout.n,
            )
            (
                batch_list,
                episode_rewards,
                episode_lengths,
                success,
                traj_uid,
                tool_callings,
            ) = filter_group_data(
                batch_list=batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                success=success,
                traj_uid=traj_uid,
                tool_callings=tool_callings,
                config=self.config,
                last_try=(try_count == max_try_count),
            )

            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = _concatenate_success_metrics(total_success)
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return (
            total_batch_list,
            total_episode_rewards,
            total_episode_lengths,
            total_success,
            total_traj_uid,
            total_tool_callings,
        )

    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        envs: EnvironmentProvider,
        is_train: bool = True,
    ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            envs (EnvironmentProvider): Common Environment provider.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        group_size = max(int(self.config.env.rollout.n), 1)
        self._rollout_is_train = is_train
        if is_train and group_size > 1:
            gen_batch = gen_batch.repeat(repeat_times=group_size, interleave=True)

        # Initial observations from the environment
        # Absent-safe read — algorithm.filter_groups defaults to None in verl 0.9
        # AlgoConfig; treat None as disabled (enable=False) to avoid AttributeError on the
        # smoke path. Only enter the dynamic branch when filter_groups is present AND enabled.
        _fg = self.config.algorithm.get("filter_groups")
        if _fg is not None and _fg.get("enable", False) and is_train:
            if group_size <= 1:
                raise ValueError("dynamic group filtering requires env.rollout.n > 1")
            # Dynamic Sampling (for DAPO)
            (
                total_batch_list,
                total_episode_rewards,
                total_episode_lengths,
                total_success,
                total_traj_uid,
                total_tool_callings,
            ) = self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                envs=envs,
            )
        else:
            # Vanilla Sampling
            (
                total_batch_list,
                total_episode_rewards,
                total_episode_lengths,
                total_success,
                total_traj_uid,
                total_tool_callings,
            ) = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                envs=envs,
                group_size=group_size if is_train else 1,
            )

        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(total_tool_callings)

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=total_tool_callings,
            meta_info=gen_batch.meta_info,
        )

        return gen_batch_output
