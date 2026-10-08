"""Bidirectional BERT architecture for diffusion LMs (MDLM/LLaDA).

Wraps ``megatron.core.models.bert.BertModel`` — a *bidirectional* encoder (padding attention
mask, no causal mask) whose ``forward(input_ids, attention_mask, lm_labels=...)`` returns
per-token vocab-parallel cross-entropy ``[b,s]``. Pair with ``--forward masked_diffusion``:
that forward step corrupts tokens to ``[MASK]`` and the model predicts the originals.

Notes: MHA only (the BERT layer spec has no GQA / no qk-layernorm); RoPE/SwiGLU via config;
RMSNorm works only with the TE spec (the local spec hardcodes LayerNorm). ``rotary_base`` is not
plumbed on BertModel (it uses the RotaryEmbedding default). Do NOT pass ``--group-query-attention``
/ ``--add-qkv-bias`` (GPT-only).

Select with ``--arch bert``.
"""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.archs import ARCHS
from alphaapollo.learning.off_policy.pretrain.base import BaseModelBuilder


@ARCHS.register("bert")
class BertArch(BaseModelBuilder):
    """Args-driven bidirectional BertModel — the diffusion-LM encoder."""

    name = "bert"

    def build(
        self, args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None
    ):
        from megatron.core.models.bert.bert_layer_specs import (
            bert_layer_local_spec,
            get_bert_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.bert.bert_model import BertModel
        from megatron.training import print_rank_0
        from megatron.training.arguments import core_transformer_config_from_args

        print_rank_0("building BERT model (AA diffusion encoder) ...")
        if config is None:
            config = core_transformer_config_from_args(args)
        use_te = args.transformer_impl == "transformer_engine"
        transformer_layer_spec = (
            get_bert_layer_with_transformer_engine_spec() if use_te else bert_layer_local_spec
        )
        model = BertModel(
            config=config,
            num_tokentypes=0,  # no binary head -> 0 token types
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,  # keep vocab split; CE is vocab-parallel inside the model
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            add_binary_head=False,  # diffusion needs no NSP / binary head
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )
        return model
