# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Attribute each trajectory's episode outcome to its final response token."""

from __future__ import annotations

import numpy as np
import torch
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from alphaapollo.learning.on_policy.reward_manager.signals import (
    episode_length,
    episode_reward,
)

EXAMINE_PRINT_PROB = 0.1


@register("episode")
class EpisodeRewardManager(AbstractRewardManager):
    """Convert rollout-provided episode outcomes into token-level rewards."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        normalize_by_length=False,
        **kwargs,
    ) -> None:
        del kwargs
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length
        # Accepted for compatibility with verl's reward-manager factory contract.
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """Return a response-shaped reward tensor for a rollout batch."""

        if "rm_scores" in data.batch.keys():
            reward_tensor = data.batch["rm_scores"]
            return self._result(reward_tensor, return_dict)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        examined_by_source: dict[str, int] = {}

        for index in range(len(data)):
            item = data[index]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = item.batch["responses"]
            attention_mask = item.batch["attention_mask"]
            valid_prompt_length = int(attention_mask[:prompt_length].sum().item())
            valid_response_length = int(attention_mask[prompt_length:].sum().item())

            outcome = episode_reward(item)
            length = episode_length(item)
            score = outcome
            if self.normalize_by_length and length != 0:
                score /= length

            if valid_response_length > 0:
                reward_tensor[index, valid_response_length - 1] = score

            data_source = str(item.non_tensor_batch["data_source"])
            examined_by_source.setdefault(data_source, 0)
            if (
                examined_by_source[data_source] < self.num_examine
                and np.random.random() < EXAMINE_PRINT_PROB
            ):
                examined_by_source[data_source] += 1
                valid_prompt_ids = (
                    prompt_ids[-valid_prompt_length:] if valid_prompt_length > 0 else prompt_ids[:0]
                )
                valid_response_ids = response_ids[:valid_response_length]
                prompt = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
                response = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                print(f"[{data_source}][prompt]", prompt)
                print(f"[{data_source}][response]", response)
                print(f"[{data_source}][score]", score)

        return self._result(reward_tensor, return_dict)

    @staticmethod
    def _result(reward_tensor: torch.Tensor, return_dict: bool):
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        return reward_tensor
