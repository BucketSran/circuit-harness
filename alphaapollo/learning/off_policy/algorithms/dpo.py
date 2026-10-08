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

"""Offline Direct Preference Optimization on verl's FSDP SFT engine.

The engine already converts causal-LM logits into token-level log probabilities.
This module therefore adapts the SPIN DPO objective at the loss boundary instead
of copying SPIN's actor/update loop.  Preference pairs are flattened as adjacent
``chosen, rejected`` sequences by :class:`PairedPreferenceCollator`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F


def sequence_logps(
    token_logps: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    average: bool = False,
) -> torch.Tensor:
    """Reduce nested token log-probabilities to one value per sequence.

    verl's LM engine stores at position ``t`` the log-probability of token
    ``t + 1``. Its SFT loss may align a flattened loss mask with ``torch.roll``
    because SFT needs only one batch scalar. DPO needs one value per sequence,
    so each jagged row is reduced independently. The final position is always
    masked explicitly; this matters when left truncation removes the prompt and
    leaves an all-completion mask whose first value is one.

    Args:
        token_logps: Jagged nested tensor ``[batch, j1]`` from the verl engine.
        loss_mask: Jagged nested tensor with the same sequence layout. Prompt
            positions are zero and completion positions are one.
        average: Divide each sequence sum by its completion-token count.

    Returns:
        Dense tensor of shape ``[batch]``.
    """

    if not getattr(token_logps, "is_nested", False):
        raise TypeError("DPO sequence_logps expects a no-padding nested token_logps tensor")
    if not getattr(loss_mask, "is_nested", False):
        raise TypeError("DPO sequence_logps expects a no-padding nested loss_mask tensor")

    logp_rows = token_logps.unbind()
    mask_rows = loss_mask.unbind()
    if len(logp_rows) != len(mask_rows):
        raise ValueError(
            f"token_logps/loss_mask batch mismatch: {len(logp_rows)} != {len(mask_rows)}"
        )
    if not logp_rows:
        raise ValueError("DPO batch must contain at least one sequence")

    values: list[torch.Tensor] = []
    for row_index, (token_logps_i, loss_mask_i) in enumerate(
        zip(logp_rows, mask_rows, strict=True)
    ):
        if token_logps_i.shape != loss_mask_i.shape:
            raise ValueError(
                f"token_logps/loss_mask shape mismatch at row {row_index}: "
                f"{tuple(token_logps_i.shape)} != {tuple(loss_mask_i.shape)}"
            )
        if token_logps_i.numel() == 0:
            raise ValueError(f"DPO sequence {row_index} is empty")

        shifted_mask_i = torch.cat((loss_mask_i[1:], loss_mask_i.new_zeros(1))).to(
            token_logps_i.dtype
        )
        valid_tokens = shifted_mask_i.sum()
        if valid_tokens.detach().item() <= 0:
            raise ValueError(
                f"DPO sequence {row_index} has no completion tokens after causal shift"
            )

        value = (token_logps_i * shifted_mask_i).sum()
        if average:
            value = value / valid_tokens
        values.append(value)

    return torch.stack(values)


def sigmoid_dpo_losses(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-pair sigmoid-DPO losses and detached implicit rewards.

    The formula matches ``recipe/spin/core_algos.py::compute_online_dpo_loss``
    before SPIN's final ``mean()`` reduction.
    """

    tensors = (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("DPO log-probability inputs must all be rank-1 tensors")
    if len({tuple(tensor.shape) for tensor in tensors}) != 1:
        raise ValueError("DPO policy/reference chosen/rejected tensors must have identical shapes")
    if not 0.0 < beta:
        raise ValueError(f"beta must be positive, got {beta}")
    if not 0.0 <= label_smoothing <= 0.5:
        raise ValueError(f"label_smoothing must be in [0, 0.5], got {label_smoothing}")

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if reference_free:
        ref_logratios = torch.zeros_like(pi_logratios)
    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits) * (1.0 - label_smoothing)
    losses -= F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def _validate_and_split_pairs(
    sequence_values: torch.Tensor,
    pair_ids: torch.Tensor,
    is_chosen: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the adjacent pair contract and return chosen/rejected views."""

    batch_size = sequence_values.shape[0]
    if batch_size == 0 or batch_size % 2:
        raise ValueError(
            f"DPO flattened sequence batch must be non-empty and even, got {batch_size}"
        )
    if pair_ids.shape != (batch_size,) or is_chosen.shape != (batch_size,):
        raise ValueError(
            "pair_id and is_chosen must be rank-1 tensors aligned with the flattened batch"
        )

    paired_ids = pair_ids.reshape(-1, 2)
    if not torch.equal(paired_ids[:, 0], paired_ids[:, 1]):
        raise ValueError("A chosen/rejected pair was split or reordered inside a DPO micro-batch")

    expected = torch.tensor([True, False], dtype=torch.bool, device=is_chosen.device).unsqueeze(0)
    expected = expected.expand(batch_size // 2, -1)
    if not torch.equal(is_chosen.to(torch.bool).reshape(-1, 2), expected):
        raise ValueError("DPO rows must be ordered [chosen, rejected] for every pair")

    return sequence_values[0::2], sequence_values[1::2]


def make_dpo_loss_fn(
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
):
    """Build a verl loss hook for offline sigmoid DPO."""

    def loss_fn(model_output, data, dp_group=None):  # noqa: ARG001 - verl loss contract
        policy_sequence_logps = sequence_logps(model_output["log_probs"], data["loss_mask"])
        policy_chosen_logps, policy_rejected_logps = _validate_and_split_pairs(
            policy_sequence_logps,
            data["pair_id"],
            data["is_chosen"],
        )

        reference_sequence_logps = data["reference_logp"].to(
            device=policy_sequence_logps.device,
            dtype=policy_sequence_logps.dtype,
        )
        reference_chosen_logps, reference_rejected_logps = _validate_and_split_pairs(
            reference_sequence_logps,
            data["pair_id"],
            data["is_chosen"],
        )

        pair_losses, chosen_rewards, rejected_rewards = sigmoid_dpo_losses(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=beta,
            label_smoothing=label_smoothing,
            reference_free=reference_free,
        )

        global_pair_batch_size = int(data["global_batch_size"])
        dp_size = int(data["dp_size"])
        if global_pair_batch_size <= 0:
            raise ValueError(
                f"global pair batch size must be positive, got {global_pair_batch_size}"
            )
        loss = pair_losses.sum() / global_pair_batch_size * dp_size

        reward_margins = chosen_rewards - rejected_rewards
        metrics = {
            "dpo/loss": pair_losses.detach().mean().item(),
            "dpo/reward_accuracy": (reward_margins > 0).float().mean().item(),
            "dpo/reward_margin": reward_margins.mean().item(),
            "dpo/chosen_reward": chosen_rewards.mean().item(),
            "dpo/rejected_reward": rejected_rewards.mean().item(),
            "dpo/policy_logratio": (policy_chosen_logps - policy_rejected_logps)
            .detach()
            .mean()
            .item(),
            "dpo/reference_logratio": (reference_chosen_logps - reference_rejected_logps)
            .detach()
            .mean()
            .item(),
        }
        return loss, metrics

    return loss_fn


def _flatten_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_numbers(item)
    elif isinstance(value, torch.Tensor):
        yield float(value.detach().item())
    else:
        yield float(value)


def _mean_nested_numbers(value: Any) -> float:
    values = list(_flatten_numbers(value))
    if not values:
        return 0.0
    return sum(values) / len(values)


def run_dpo(config) -> None:
    """Initialize distributed state, train offline DPO, and tear it down."""

    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

    initialize_global_process_group()
    try:
        trainer = build_dpo_trainer(config)
        trainer.fit()
    finally:
        destroy_global_process_group()


def build_dpo_trainer(config):
    """Construct the verl SFTTrainer subclass lazily.

    Heavy verl imports remain out of module import time so pure CPU math tests and
    Hydra ``--cfg job`` introspection stay lightweight.
    """

    from torch.utils.data import DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils import tensordict_utils as tu
    from verl.utils.device import get_device_name
    from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig

    from alphaapollo.data_preprocess.preference_dataset import PairedPreferenceCollator

    class DPOTrainingWorker(TrainingWorker):
        def _postprocess_output(self, output, **kwargs):
            final_output = super()._postprocess_output(output, **kwargs)
            metrics = tu.get(final_output, "metrics")
            for key in tuple(metrics):
                if key.startswith("dpo/"):
                    metrics[key] = _mean_nested_numbers(metrics[key])
            return final_output

    class DPOTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            trainer_config = kwargs.get("config", args[0] if args else None)
            dpo_config = trainer_config.get("alphaapollo", {}).get("dpo", {})
            if bool(dpo_config.get("reference_free", False)):
                trainer_config.data.require_reference_logps = False
                trainer_config.data.validate_reference_metadata = False
            super().__init__(*args, **kwargs)

        def _build_engine(self):
            dpo_config = self.config.get("alphaapollo", {}).get("dpo", {})
            loss_type = str(dpo_config.get("loss_type", "sigmoid"))
            if loss_type != "sigmoid":
                raise ValueError(
                    f"Offline DPO currently supports only loss_type='sigmoid'; got {loss_type!r}. "
                    "IPO is deferred until its logp-normalization contract is selected."
                )

            self.loss_fn = make_dpo_loss_fn(
                beta=float(dpo_config.get("beta", 0.1)),
                label_smoothing=float(dpo_config.get("label_smoothing", 0.0)),
                reference_free=bool(dpo_config.get("reference_free", False)),
            )
            worker_config = TrainingWorkerConfig(
                model_type="language_model",
                model_config=self.model_config,
                engine_config=self.engine_config,
                optimizer_config=self.optimizer_config,
                checkpoint_config=self.checkpoint_config,
                profiler_config=self.profiler_config,
            )
            self.training_client = DPOTrainingWorker(config=worker_config)
            self.training_client.set_loss_fn(loss_fn=self.loss_fn)
            self.engine = self.training_client.engine

        def _build_dataloader(self):
            dp_rank = self.engine.get_data_parallel_rank()
            dp_size = self.engine.get_data_parallel_size()
            global_pair_batch_size = int(self.config.data.train_batch_size)
            micro_sequence_batch_size = int(self.config.data.micro_batch_size_per_gpu)

            if bool(self.config.data.use_dynamic_bsz):
                raise ValueError(
                    "Offline DPO v1 requires data.use_dynamic_bsz=false to keep pairs atomic"
                )
            if global_pair_batch_size % dp_size:
                raise ValueError(
                    f"pair train_batch_size={global_pair_batch_size} must be divisible "
                    f"by dp_size={dp_size}"
                )
            if micro_sequence_batch_size <= 0 or micro_sequence_batch_size % 2:
                raise ValueError(
                    "DPO data.micro_batch_size_per_gpu counts flattened sequences "
                    "and must be a positive even number"
                )

            local_pair_batch_size = global_pair_batch_size // dp_size
            local_sequence_batch_size = local_pair_batch_size * 2
            if local_sequence_batch_size % micro_sequence_batch_size:
                raise ValueError(
                    f"local flattened batch ({local_sequence_batch_size} sequences) "
                    "must be divisible by "
                    f"micro_batch_size_per_gpu={micro_sequence_batch_size}"
                )

            self.train_sampler = DistributedSampler(
                self.train_dataset,
                shuffle=True,
                num_replicas=dp_size,
                rank=dp_rank,
                drop_last=True,
            )
            self.global_batch_size = global_pair_batch_size
            self.train_batch_size_per_dp = local_pair_batch_size
            self.collate_fn = PairedPreferenceCollator()

            dataloader_kwargs = dict(
                batch_size=local_pair_batch_size,
                collate_fn=self.collate_fn,
                num_workers=int(self.config.data.num_workers),
                pin_memory=False,
                drop_last=True,
                pin_memory_device=get_device_name(),
            )
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                sampler=self.train_sampler,
                **dataloader_kwargs,
            )

            if self.val_dataset:
                self.val_sampler = DistributedSampler(
                    self.val_dataset,
                    shuffle=False,
                    num_replicas=dp_size,
                    rank=dp_rank,
                    drop_last=True,
                )
                self.val_dataloader = StatefulDataLoader(
                    dataset=self.val_dataset,
                    sampler=self.val_sampler,
                    **dataloader_kwargs,
                )
            else:
                self.val_dataloader = None

    return DPOTrainer(config=config)


__all__ = [
    "build_dpo_trainer",
    "make_dpo_loss_fn",
    "run_dpo",
    "sequence_logps",
    "sigmoid_dpo_losses",
]
