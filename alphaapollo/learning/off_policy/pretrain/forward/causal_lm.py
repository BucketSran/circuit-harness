"""Causal-LM forward step + loss — the default forward component.

Moved verbatim from the former ``forward_step.py`` and mapped onto the
``BaseForwardStep`` hooks (``get_batch`` / ``compute`` / ``loss_func``); the concrete
``forward_step`` template method comes from the base class.
"""

from __future__ import annotations

import torch

from alphaapollo.learning.off_policy.pretrain.base import BaseForwardStep
from alphaapollo.learning.off_policy.pretrain.forward import FORWARD


@FORWARD.register("causal_lm")
class CausalLMForward(BaseForwardStep):
    """Standard causal-LM training step (GPTDataset / MockGPTDataset batches)."""

    name = "causal_lm"

    def get_batch(self, data_iterator) -> dict:
        """Lean batch fetcher for TP=PP=CP=1: pull next batch, move tensors to cuda.

        ``GPTDataset`` / ``MockGPTDataset`` yield a dict with at least ``tokens``,
        ``position_ids``, ``attention_mask``, ``labels``, ``loss_mask``.
        """
        batch = next(data_iterator)
        out = {}
        for key, val in batch.items():
            out[key] = val.cuda(non_blocking=True) if torch.is_tensor(val) else val
        return out

    def compute(self, batch, model):
        """One model forward; returns the per-token loss tensor the mcore GPTModel produces
        when ``labels`` + ``loss_mask`` are passed."""
        return model(
            batch["tokens"],
            batch["position_ids"],
            batch.get("attention_mask"),
            labels=batch.get("labels"),
            loss_mask=batch.get("loss_mask"),
        )

    def loss_func(self, loss_mask, output_tensor):
        """Standard causal-LM loss: sum(per-token loss * loss_mask) / num_tokens.

        Returns ``(loss_scalar, num_tokens, report)`` — the contract Megatron's
        forward-backward loop expects from a loss callable.
        """
        losses = output_tensor.float().view(-1)
        loss_mask = loss_mask.float().view(-1)
        loss = torch.sum(losses * loss_mask)
        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {"lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

        # NaN guard (cheap; Megatron's full version also checks Inf / spiky loss via the
        # rerun state machine — add if needed).
        from megatron.training import get_args

        args = get_args()
        if getattr(args, "check_for_nan_in_loss_and_grad", False) and torch.isnan(loss):
            raise FloatingPointError("found NaN in local forward loss calculation")

        return loss, num_tokens, report
