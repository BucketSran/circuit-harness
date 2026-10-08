"""Masked discrete-diffusion (MDLM/LLaDA) forward step + loss.

Per training step: sample a diffusion timestep ``t ~ U(t_min, t_max)``, mask each real token to
``mask_id`` independently with probability ``t``, run the (bidirectional) model with
``lm_labels=originals`` to obtain per-token cross-entropy ``[b,s]``, and weight it by the
MDLM/LLaDA x0-prediction schedule ``1/t`` (the absorb-diffusion ELBO for the linear/uniform
masking schedule). Pair with ``--arch bert`` (BertModel returns per-token CE
when ``lm_labels`` is passed).

Select with ``--forward masked_diffusion``. Requires ``--diff-mask-id <token id>``,
``--diff-t-min`` (default 1e-3), ``--diff-t-max`` (default 0.999).

This component overrides ``BaseForwardStep.forward_step`` (the base template threads
``batch['loss_mask']``, which does not fit diffusion); the per-token CE is produced in ``compute``
via the ``_diffuse`` helper, and ``forward_step`` threads ``(mask, weight)`` into ``loss_func``.
"""

from __future__ import annotations

from functools import partial

import torch

from alphaapollo.learning.off_policy.pretrain.base import BaseForwardStep
from alphaapollo.learning.off_policy.pretrain.forward import FORWARD


@FORWARD.register("masked_diffusion")
class MaskedDiffusionForward(BaseForwardStep):
    """MDLM-style masked discrete diffusion training step (absorb / mask diffusion)."""

    name = "masked_diffusion"

    def get_batch(self, data_iterator) -> dict:
        batch = next(data_iterator)
        out = {}
        for key, val in batch.items():
            out[key] = val.cuda(non_blocking=True) if torch.is_tensor(val) else val
        return out

    def _diffusion_args(self, args):
        mask_id = int(getattr(args, "diff_mask_id", -1) or -1)
        if mask_id < 0:
            raise ValueError(
                "--forward masked_diffusion requires --diff-mask-id <token id> "
                "(the [MASK] token used for corruption)."
            )
        t_min = float(getattr(args, "diff_t_min", 1e-3))
        t_max = float(getattr(args, "diff_t_max", 0.999))
        if not (0.0 < t_min < t_max < 1.0):
            raise ValueError(
                f"require 0 < diff-t-min < diff-t-max < 1; got t_min={t_min}, t_max={t_max}."
            )
        return mask_id, t_min, t_max

    def _diffuse(self, batch, model):
        """Corrupt tokens to [MASK] per timestep; return (per_token_ce[b,s], mask[b,s], weight)."""
        from megatron.training import get_args

        mask_id, t_min, t_max = self._diffusion_args(get_args())
        tokens = batch["tokens"]  # [b,s] clean originals
        b, s = tokens.shape

        # one timestep per step; mask each token independently with prob t (absorb diffusion)
        t = torch.empty(1, device=tokens.device).uniform_(t_min, t_max).item()
        mask = torch.rand(b, s, device=tokens.device) < t  # [b,s] bool: which positions to mask
        corrupted = torch.where(mask, torch.full_like(tokens, mask_id), tokens)
        padding_mask = torch.ones(b, s, device=tokens.device, dtype=tokens.dtype)  # no-pad path

        # BertModel(input_ids, attention_mask, lm_labels=...) -> (per_token_ce[b,s], binary_logits)
        per_tok_ce, _ = model(corrupted, padding_mask, lm_labels=tokens)
        # MDLM/LLaDA x0-prediction ELBO weight for the linear/uniform masking schedule is 1/t (the
        # absorb-diffusion variational bound). 1/(1-t) (the prior value) is WRONG — it overweights
        # nearly-fully-masked steps (up to 1000x at t_max=0.999) instead of lightly-masked ones.
        # t >= t_min (>=1e-3) so no division by zero.
        weight = 1.0 / t
        return per_tok_ce, mask, weight

    def compute(self, batch, model):
        """Per-token loss tensor (Megatron reduces it via loss_func). Used by the base template."""
        return self._diffuse(batch, model)[0]

    def forward_step(self, data_iterator, model):
        """Override the base template to thread (mask, weight) into loss_func."""
        batch = self.get_batch(data_iterator)
        per_tok_ce, mask, weight = self._diffuse(batch, model)
        return per_tok_ce, partial(self.loss_func, mask, weight)

    def loss_func(self, mask, weight, output_tensor):
        """MDLM loss: (1/t) * sum(per-token CE over masked positions), normalized by num_valid.

        Returns ``(loss_scalar, num_valid, report)`` — Megatron's loss-callable contract; it
        normalizes the summed loss by ``num_tokens`` across ranks. Per the standard MDLM/LLaDA
        continuous-time estimator this is ``sum(mask * CE / t) / num_valid_tokens`` (reviewer #4
        option a): normalize by the VALID token count (b*s for the no-pad path), NOT the masked
        count — the masked count varies with the sampled t and would make the loss scale
        schedule-dependent.
        """
        per_tok_ce = output_tensor.float()
        mask_f = mask.to(per_tok_ce.dtype)
        loss = weight * torch.sum(per_tok_ce * mask_f)  # weight = 1/t (per-step scalar)
        num_tokens = torch.tensor(
            mask_f.numel(), dtype=torch.int, device=per_tok_ce.device
        )  # num_valid (b*s)
        report = {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

        from megatron.training import get_args

        args = get_args()
        if getattr(args, "check_for_nan_in_loss_and_grad", False) and torch.isnan(loss):
            raise FloatingPointError("found NaN in diffusion forward loss")
        return loss, num_tokens, report
