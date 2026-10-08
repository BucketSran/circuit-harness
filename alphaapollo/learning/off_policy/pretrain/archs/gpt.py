"""Default dense-GPT architecture (args-driven; Qwen/Llama-style). TE or local layer spec.

Moved verbatim from the former ``model_provider.py`` (``gpt_builder`` +
``_get_transformer_layer_spec`` + the ``--load-hf`` continued-pretrain hook), now wrapped as
the ``"gpt"`` :class:`BaseModelBuilder`.
"""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.archs import ARCHS
from alphaapollo.learning.off_policy.pretrain.base import BaseModelBuilder


@ARCHS.register("gpt")
class GPTArch(BaseModelBuilder):
    """Args-driven dense GPTModel (GQA/SwiGLU/RMSNorm/RoPE/tied embeddings via CLI flags)."""

    name = "gpt"

    def build(
        self, args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None
    ):
        """Build a dense mcore GPTModel. TE vs local layer-spec is chosen by --transformer-impl."""
        from megatron.core.models.gpt import GPTModel
        from megatron.training import print_rank_0
        from megatron.training.arguments import core_transformer_config_from_args

        print_rank_0("building GPT model (AA) ...")
        if config is None:
            config = core_transformer_config_from_args(args)
        use_te = args.transformer_impl == "transformer_engine"
        transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
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

        # Continued-pretrain: inject real HF weights into the freshly-built mcore model via
        # megatron-bridge (Qwen2 HF -> mcore mapping). Arch args must match the HF model.
        # NOTE: currently BLOCKED by a bridge<->mcore0.18 load_hf_weights API drift; the
        # from-scratch path (no --load-hf) is the working one.
        if getattr(args, "load_hf", None):
            from megatron.bridge import AutoBridge

            # mcore 0.18 TransformerConfig doesn't expose share_embeddings_and_output_weights,
            # but bridge.load_hf_weights reads it off model.config — set it explicitly.
            model.config.share_embeddings_and_output_weights = (
                not args.untie_embeddings_and_output_weights
            )
            print_rank_0(f"[AA] continued-pretrain: loading HF weights from {args.load_hf} ...")
            bridge = AutoBridge.from_hf_pretrained(args.load_hf)
            bridge.load_hf_weights([model])
            print_rank_0("[AA] HF weights loaded into mcore model.")

        return model


def _get_transformer_layer_spec(use_te, config):
    """Verbatim from Megatron-LM 0.18 gpt_builders (dense path only)."""
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )

    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            config.num_moe_experts,
            config.moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
            config.experimental_attention_variant,
            qk_l2_norm=config.qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_te_activation_func=config.use_te_activation_func,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
            mla_down_proj_fusion=getattr(config, "mla_down_proj_fusion", False),
            use_grouped_gemm_for_dense_mlp=config.use_grouped_gemm_for_dense_mlp,
        )
    return get_gpt_layer_local_spec(
        config.num_moe_experts,
        config.moe_grouped_gemm,
        config.qk_layernorm,
        config.multi_latent_attention,
        config.experimental_attention_variant,
        normalization=config.normalization,
        use_kitchen=config.use_kitchen,
        use_kitchen_attention=config.use_kitchen_attention,
        kitchen_attention_backend=config.kitchen_attention_backend,
    )
