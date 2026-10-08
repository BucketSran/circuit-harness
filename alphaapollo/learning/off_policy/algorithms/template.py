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

"""TEMPLATE — off-policy preference-optimization algorithm (DPO / SimPO / KTO).

Copy this file to ``<algo>.py`` (e.g. ``dpo.py``) and fill in the three seams below.
This is a GUIDE, not an active algorithm — nothing imports it, and it does not
self-register. Delete the parts you don't need.

An off-policy algorithm here differs from the SFT baseline in exactly one thing: the
LOSS. It runs on verl's static-data SFT engine (no sampling loop). The verl-native
baseline (SFT) is a pure passthrough with no file; a new algorithm plugs in through
THREE verl public seams — ZERO edits to third_party/verl:

  (1) DATASET  — a paired (chosen/rejected) dataset, selected via config
                 ``data.custom_cls.path`` + ``data.custom_cls.name``. verl's
                 ``create_sft_dataset`` (sft_trainer.py) already honors this hook.
  (2) LOSS     — a ``loss_fn(config, model_output, data, dp_group=None)`` matching
                 verl's contract (same shape as ``verl.workers.utils.losses.sft_loss``).
  (3) WIRING   — a thin ``SFTTrainer`` subclass that swaps ``self.loss_fn`` in
                 ``_build_engine`` before ``set_loss_fn``; ``main_off_policy`` picks it
                 by config. (SFT's ``run_sft`` hardcodes the base trainer + ``sft_loss``,
                 so a custom loss needs a subclass — the off-policy analogue of how
                 on_policy subclasses ``RayPPOTrainer``.)

DESIGN NOTE — the two-model problem (read before implementing DPO/KTO):
  DPO/KTO need a frozen REFERENCE model's log-probs. verl's engine assumes a single
  trained model; threading a second frozen model through it is expensive. INSTEAD,
  precompute reference log-probs during preprocessing
  (``alphaapollo/data_preprocess/prepare_dpo_*.py``) and write them as parquet columns
  (``ref_logprob_chosen`` / ``ref_logprob_rejected``). The loss then just READS them —
  no second model at train time, and DPO becomes structurally identical to SFT.
  (SimPO needs no reference model at all — it drops the ref term for a length-normalized
  margin, so it skips this entirely.)

CONFIG (all AA-custom keys live under ``alphaapollo.*`` / ``data.*`` — never inside a
verl dataclass namespace, which is strict-validated):
    data:
      custom_cls:
        path: alphaapollo/data_preprocess/preference_dataset.py   # your Dataset file
        name: PairedPreferenceDataset
      train_files: .../dpo_train.parquet     # chosen/rejected (+ ref_logprob_* cols)
    alphaapollo:
      off_policy_algo: dpo                    # read by main_off_policy to pick the trainer
      dpo:
        beta: 0.1
"""

from __future__ import annotations

from typing import Any

import torch


# ---------------------------------------------------------------------------
# (2) LOSS — the only mathematically-distinct part. Engine-agnostic core so it is
#     unit-testable on CPU without verl. Example shows DPO; SimPO/KTO variants noted.
# ---------------------------------------------------------------------------
def dpo_loss_core(
    policy_chosen_logps: torch.Tensor,  # [batch] sum log p_theta(y_chosen | x)
    policy_rejected_logps: torch.Tensor,  # [batch] sum log p_theta(y_rejected | x)
    ref_chosen_logps: torch.Tensor,  # [batch] precomputed frozen-ref log-probs
    ref_rejected_logps: torch.Tensor,  # [batch]
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Standard DPO loss (Rafailov et al. 2023). Pure torch — no verl, no engine.

    L = -E[ log sigmoid( beta * ( (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected) ) ) ]

    SimPO variant: drop the ref terms, use length-normalized logps and a margin gamma:
        logits = beta * (pi_chosen/len_chosen - pi_rejected/len_rejected) - gamma
    KTO variant: unpaired; per-example desirable/undesirable with a KL baseline term.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -torch.nn.functional.logsigmoid(beta * logits).mean()

    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    metrics = {
        "dpo/loss": loss.item(),
        "dpo/reward_margin": (chosen_reward - rejected_reward).mean().item(),
        "dpo/accuracy": (chosen_reward > rejected_reward).float().mean().item(),
    }
    return loss, metrics


def make_preference_loss_fn(beta: float = 0.1):
    """Return a ``loss_fn`` matching verl's ``set_loss_fn`` contract.

    Signature mirrors ``verl.workers.utils.losses.sft_loss``:
        loss_fn(config: ActorConfig, model_output, data: TensorDict, dp_group=None)
            -> (loss: torch.Tensor, metrics: dict)

    ``model_output["log_probs"]`` holds per-token log-probs; ``data`` carries the
    dataset fields (loss masks, and the precomputed ``ref_logprob_*`` columns your
    PairedPreferenceDataset emitted). TODO: reduce per-token log-probs to per-sequence
    sums for chosen/rejected using the loss masks, then call ``dpo_loss_core``.
    """

    def loss_fn(config, model_output, data, dp_group=None):  # noqa: ANN001 — verl contract
        # TODO(implementer): from model_output["log_probs"] + data["loss_mask"],
        #   build policy_chosen_logps / policy_rejected_logps (sum over response tokens,
        #   split by the chosen/rejected halves of the micro-batch).
        # TODO(implementer): read data["ref_logprob_chosen"] / data["ref_logprob_rejected"]
        #   (precomputed at preprocessing — see DESIGN NOTE).
        raise NotImplementedError(
            "Copy template.py -> <algo>.py and implement the log-prob reduction + "
            "reference-logprob read, then call dpo_loss_core(...)."
        )

    return loss_fn


# ---------------------------------------------------------------------------
# (3) WIRING — thin SFTTrainer subclass that swaps the loss. Import verl lazily so this
#     template stays import-safe on CPU / without verl installed.
# ---------------------------------------------------------------------------
def build_trainer(config):
    """Return an SFTTrainer whose loss_fn is the preference loss (not sft_loss).

    ``main_off_policy`` calls this when ``alphaapollo.off_policy_algo`` selects this
    algorithm, instead of the passthrough ``verl.trainer.sft_trainer.run_sft``.
    """
    from verl.trainer.sft_trainer import SFTTrainer  # lazy — verl only needed at run time

    beta = float(config.get("alphaapollo", {}).get("dpo", {}).get("beta", 0.1))

    class PreferenceSFTTrainer(SFTTrainer):
        def _build_engine(self):
            # Reproduce the parent's engine build, but inject our loss_fn.
            super()._build_engine()
            self.loss_fn = make_preference_loss_fn(beta=beta)
            self.training_client.set_loss_fn(loss_fn=self.loss_fn)

    return PreferenceSFTTrainer(config=config)


# ---------------------------------------------------------------------------
# (1) DATASET — sketch only. Put the real class in
#     alphaapollo/data_preprocess/preference_dataset.py and point data.custom_cls at it.
#     It must inherit torch.utils.data.Dataset (verl checks this) and yield, per item,
#     the chosen+rejected token ids/masks plus the precomputed ref_logprob_* fields.
# ---------------------------------------------------------------------------
# class PairedPreferenceDataset(torch.utils.data.Dataset):
#     def __init__(self, data_paths, data_config, tokenizer, processor, max_samples=-1): ...
#     def __getitem__(self, idx) -> dict: ...   # chosen/rejected ids+mask + ref_logprob_*
