"""Fully-custom architecture: a CAUSAL MLP-Mixer LM with NO attention / NO QKV.

Deliberately NOT a GPT/transformer, to exercise the "completely off the GPT template" path:

* **token mixing** = a CAUSAL depthwise conv (each channel looks back only ``kernel`` tokens;
  left-padded so position t sees positions <= t — keeps it autoregressive, which RL needs);
* **channel mixing** = a GLU MLP;
* residual pre-norm blocks.

It trains through the SAME Megatron ``pretrain()`` loop as GPT (TP=1 / DP), but the model is a
hand-written :class:`megatron.core.transformer.module.MegatronModule` — NOT a ``GPTModel``. Its
``forward`` signature matches what the default ``causal_lm`` forward step calls
(``model(tokens, position_ids, attention_mask, labels=, loss_mask=)`` returning per-token loss),
so ``--forward causal_lm`` is reused unchanged.

Reaching RL with this arch is the hard part: vLLM cannot load it, so RL must use
``rollout.name=hf``; and the HF side needs a matching ``modeling_causal_mlp.py``
(trust_remote_code) plus the dedicated mcore->HF converter at
``alphaapollo.learning.off_policy.pretrain.converters.causal_mlp`` (select with ``--converter
causal_mlp``). The checked-in HF mirror lives at ``tests/fixtures/causal_mlp_hf/``.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule

from alphaapollo.learning.off_policy.pretrain.archs import ARCHS
from alphaapollo.learning.off_policy.pretrain.base import BaseModelBuilder


class CausalDepthwiseConv(nn.Module):
    """Depthwise conv over the sequence dim, one filter per channel, CAUSAL via left-padding."""

    def __init__(self, hidden: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=kernel, groups=hidden, bias=True)

    def forward(self, x):  # x: [b, s, h]
        xt = x.transpose(1, 2)  # [b, h, s]
        xt = F.pad(xt, (self.kernel - 1, 0))  # left-pad -> [b, h, s+k-1] (causal)
        out = self.conv(xt)  # [b, h, s]
        return out.transpose(1, 2)  # [b, s, h]


class GLUMLP(nn.Module):
    """SwiGLU-style channel mixing: fc1 (H->2F) -> gate*value -> fc2 (F->H)."""

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 2 * ffn, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        gate, value = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * value)


class CausalMLPLayer(nn.Module):
    """One block: causal-conv token mixing + GLU channel mixing, both pre-norm residual."""

    def __init__(self, hidden: int, ffn: int, kernel: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.conv = CausalDepthwiseConv(hidden, kernel)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = GLUMLP(hidden, ffn)

    def forward(self, x):
        x = x + self.conv(self.norm1(x))  # token mixing (looks back `kernel` tokens)
        x = x + self.mlp(self.norm2(x))  # channel mixing (per-token)
        return x


class CausalMLPModel(MegatronModule):
    """A fully-custom causal LM. Returns per-token CE loss ``[b, s]`` when ``labels`` is given
    (aligned to ``[b, s]`` so ``causal_lm.loss_func``'s ``loss_mask`` of the same shape multiplies
    cleanly); returns logits ``[b, s, V]`` otherwise (used by the HF export / RL side).
    """

    def __init__(self, config, vocab_size, num_layers, hidden_size, ffn_size, conv_kernel=8):
        super().__init__(config)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ffn_size = ffn_size
        self.conv_kernel = conv_kernel
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [CausalMLPLayer(hidden_size, ffn_size, conv_kernel) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def set_input_tensor(self, input_tensor):
        """Pipeline-parallel input forwarding. No-op for this arch — Megatron's fwd/bwd schedule
        calls this even at PP=1 (GPTModel has it via inheritance); we don't support PP here."""
        pass

    def forward(self, tokens, position_ids=None, attention_mask=None, labels=None, loss_mask=None):
        h = self.embed(tokens)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        logits = self.lm_head(h)  # [b, s, V]
        if labels is None:
            return logits
        # Megatron GPTDataset PRE-SHIFTS the data (gpt_dataset.py: tokens=text[:-1],
        # labels=text[1:], already next-token aligned), and GPTModel.compute_language_model_loss
        # compares logits vs labels POSITION-BY-POSITION (no extra shift). So we must NOT shift
        # again — a logits[:,:-1]/labels[:,1:] here trains token t to predict t+2.
        b, s, v = logits.shape
        ce = F.cross_entropy(
            logits.reshape(-1, v), labels.reshape(-1), reduction="none", ignore_index=-100
        )
        return ce.view(b, s)  # [b, s], aligned to loss_mask


@ARCHS.register("causal_mlp")
class CausalMLPArch(BaseModelBuilder):
    """Pluggable builder for the fully-custom causal MLP-Mixer LM."""

    name = "causal_mlp"

    def build(
        self, args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None
    ):
        from megatron.training import print_rank_0
        from megatron.training.arguments import core_transformer_config_from_args

        print_rank_0("[causal_mlp] building FULLY-CUSTOM causal MLP-Mixer LM (no attention) ...")
        if config is None:
            config = core_transformer_config_from_args(args)
        return CausalMLPModel(
            config=config,
            vocab_size=args.padded_vocab_size,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            ffn_size=args.ffn_hidden_size,
            conv_kernel=8,
        )
