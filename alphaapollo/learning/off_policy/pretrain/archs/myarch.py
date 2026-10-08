"""Custom architecture for the end-to-end pipeline demo (pretrain -> HF -> verl RL).

Registered as ``--arch myarch``. This is the pluggable custom-architecture slot: a
``BaseModelBuilder`` subclass that returns a Megatron ``GPTModel``. ``main_pretrain`` calls
``build(...)`` and handles data / optimizer / checkpoint save / multi-GPU for you.

This demo builder produces a standard **decoder-only** transformer so that its mcore checkpoint
flows through the *generic* ``decoder`` converter (``get_converter('decoder')``) into a
HuggingFace model that verl can load for RL. The exact topology (Qwen2-style w/ qkv-bias, or
Llama-style w/o) is selected by the launch args — ``--add-qkv-bias``, ``--group-query-attention``,
``--num-query-groups``, etc. — so the SAME custom builder covers a family of decoder variants.

CUSTOMIZE HERE for a real, non-decoder topology: swap ``transformer_layer_spec`` for your own
layer spec / custom attention+MLP modules. A non-decoder arch then needs its own
``BaseCheckpointConverter`` (the declarative ``mapping`` converter covers arbitrary layouts) before
it can reach RL.

The ``build`` signature is fixed (the adapter calls it) — do not change it.
"""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.archs import ARCHS
from alphaapollo.learning.off_policy.pretrain.base import BaseModelBuilder


@ARCHS.register("myarch")
class MyArch(BaseModelBuilder):
    """Pipeline-demo custom architecture (decoder-only; topology via launch args)."""

    name = "myarch"

    def build(
        self, args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None
    ):
        from megatron.core.models.gpt import GPTModel
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.training import print_rank_0
        from megatron.training.arguments import core_transformer_config_from_args

        print_rank_0("[myarch] building CUSTOM decoder architecture (pipeline demo) ...")
        if config is None:
            config = core_transformer_config_from_args(args)

        # CUSTOMIZE HERE: replace with your own layer spec / custom modules for a non-decoder arch.
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            config.num_moe_experts,
            config.moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
            config.experimental_attention_variant,
            qk_l2_norm=config.qk_l2_norm,
        )

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=None,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )
        return model
