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

"""Invalid-action penalty — reward-signal shaping (not an algorithm).

Adjusts ``token_level_scores`` in place before advantage computation; the advantage
estimator itself stays verl-native / lives in ``../algorithms``.
"""

from __future__ import annotations

import numpy as np
import torch

from alphaapollo.learning.on_policy.reward_manager.signals import action_validity


def apply_invalid_action_penalty(data, invalid_action_penalty_coef: float):
    """Apply penalty to token_level_scores for invalid actions.

    Args:
        data: DataProto batch with batch['token_level_scores'], batch['prompts'],
            batch['attention_mask'], non_tensor_batch['is_action_valid'].
            Optionally batch['step_rewards'] for per-step reward adjustment.
        invalid_action_penalty_coef: Penalty coefficient subtracted from the
            last valid response token of each invalid action row.

    Returns:
        Tuple of (data, metrics) where metrics = {'episode/valid_action_ratio': float}.
    """
    reward_tensor = data.batch["token_level_scores"]
    if "step_rewards" in data.batch.keys():
        step_rewards = data.batch["step_rewards"]
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
        action_valids = action_validity(data_item.non_tensor_batch).astype(np.float32)
        action_invalids = torch.tensor(
            1 - action_valids, dtype=torch.float32, device=prompt_ids.device
        ).squeeze(0)
        if valid_response_length > 0:
            reward_tensor[i, valid_response_length - 1] -= (
                invalid_action_penalty_coef * action_invalids
            )
            if "step_rewards" in data.batch.keys():
                step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    valid_action_ratio = np.mean(action_validity(data.non_tensor_batch).astype(np.float32)).item()
    metrics = {"episode/valid_action_ratio": valid_action_ratio}
    return data, metrics
