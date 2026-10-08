"""Example custom-architecture builder (skeleton) — plug your own structure here.

Use it with::

    --arch example_custom
    # or
    --arch custom:alphaapollo.learning.off_policy.pretrain.archs.example_custom:ExampleCustomArch

This skeleton builds the SAME dense GPT as the default ``GPTArch``, so it runs out-of-the-box
and you can see the pluggable path works end-to-end (multi-GPU from-scratch). To make it a REAL
custom architecture, replace the ``transformer_layer_spec`` (and/or the GPTModel construction)
with YOUR custom layer spec / custom modules.

A custom builder MUST subclass ``BaseModelBuilder`` and implement ``build`` with the signature::

    build(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None)
        -> megatron.core.models.gpt.GPTModel

It receives the live Megatron ``args`` (so it can read --num-layers / --hidden-size /
parallelism / your own --arch-* args etc.) and must return a Megatron model. Everything else
(data, optimizer, checkpoint save, multi-GPU) is handled by ``main_pretrain``.

NOTE: only the PRETRAIN side. A custom arch also needs its own mcore->HF converter (register a
``BaseCheckpointConverter``; the default ``"qwen2"`` is Qwen2-specific) before it can reach RL.
"""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.archs import ARCHS
from alphaapollo.learning.off_policy.pretrain.base import BaseModelBuilder


@ARCHS.register("example_custom")
class ExampleCustomArch(BaseModelBuilder):
    """Runnable custom-arch skeleton (dense-GPT placeholder). Replace the spec to customize."""

    name = "example_custom"

    def build(
        self, args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None
    ):
        """Build a Megatron GPTModel. Replace the spec below with YOUR custom structure."""
        from megatron.core.models.gpt import GPTModel
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.training import print_rank_0
        from megatron.training.arguments import core_transformer_config_from_args

        print_rank_0(
            "[custom-arch] building EXAMPLE custom architecture (dense-GPT placeholder) ..."
        )
        if config is None:
            config = core_transformer_config_from_args(args)

        # TODO(custom): swap this for YOUR custom layer spec / custom transformer layer.
        # The default below is the standard TE dense spec.
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
