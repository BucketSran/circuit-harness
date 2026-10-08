"""Bridge-based mcore -> HuggingFace converter (automatic, ~50 model families).

Uses ``megatron.bridge.AutoBridge`` so any of bridge's supported families (Qwen2/3, Llama,
Mistral, Gemma, Deepseek, MoE, MLA, ...) converts out-of-the-box with bridge's battle-tested
mappings — which get GQA per-group QKV / MoE expert / MLA layouts right automatically, with no
hand-written tensor math on our side.

Works on **native** mcore ``torch_dist`` checkpoints via "Route 1" — bypassing bridge's STRICT
checkpoint loader, which can't tolerate three mismatches with native mcore-0.18 ckpts:
  * bridge's ``load_megatron_model`` builds a TE model and loads STRICTLY (no ``strict`` flag) and
    fails on ``decoder.final_layernorm._extra_state`` — mcore-0.18's GPTModel uses a NATIVE RMSNorm
    for the final_layernorm (no TE extra state), so the ckpt lacks it;
  * bridge's model build defaults ``gradient_accumulation_fusion=True`` (needs APEX, not in our
    env) and DDP-wrapping (needs a ``ddp_config``).
Route 1 instead: ``to_megatron_provider`` -> clear ``gradient_accumulation_fusion`` -> build
without DDP (``wrap_with_ddp=False``) -> ``dist_checkpointing.load`` (``_extra_state`` stripped,
``validate_access_integrity=False``) -> ``save_hf_pretrained``. Needs a GPU (bridge builds the
mcore model).

Verified on a real 28L Qwen2.5-0.5B (2-group) ckpt: bridge's auto export is **bit-exact**
(112/112 attention weights) vs the hand-written ``decoder`` converter — independent confirmation.

For a novel architecture bridge has no spec for (e.g. CausalMLP), use ``--converter mapping``
(B, JSON rules) or a custom converter (D, ``--converter custom:<module>:<Cls>``).
"""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base import BaseCheckpointConverter
from alphaapollo.learning.off_policy.pretrain.converters import CONVERTERS


@CONVERTERS.register("bridge")
class BridgeConverter(BaseCheckpointConverter):
    """Convert a Megatron-LM torch_dist checkpoint to HuggingFace via megatron-bridge AutoBridge.

    ``hf_config`` (passed to :meth:`export`) identifies the target HF architecture and so the
    conversion mapping bridge applies. Pass a ``transformers.PretrainedConfig`` or anything
    ``AutoConfig.from_pretrained`` accepts (path / dir / hub id).
    """

    name = "bridge"

    def export(self, mcore_ckpt_dir: str, hf_out_dir: str, *, hf_config, **opts) -> str:
        """Convert ``mcore_ckpt_dir`` -> ``hf_out_dir`` via AutoBridge.

        Args:
            mcore_ckpt_dir: Megatron-LM ``torch_dist`` checkpoint dir (must carry ``args``).
            hf_out_dir: output HuggingFace model dir.
            hf_config: target HF architecture — a ``PretrainedConfig`` or a path/dir/model-id.
        """
        import megatron.core.dist_checkpointing as dc
        import transformers
        from megatron.bridge import AutoBridge

        from alphaapollo.learning.off_policy.pretrain.converters.qwen2 import _latest_iter_dir

        if isinstance(hf_config, str):
            hf_config = transformers.AutoConfig.from_pretrained(hf_config)
        if not AutoBridge.supports(hf_config):
            raise ValueError(
                f"[bridge] AutoBridge has no spec for model_type="
                f"{getattr(hf_config, 'model_type', '?')!r}. For a novel architecture use "
                f"`--converter mapping` (B) or a custom converter (D)."
            )

        bridge = AutoBridge.from_hf_config(hf_config)
        iter_dir = _latest_iter_dir(mcore_ckpt_dir)
        # Route 1 — bypass bridge's STRICT checkpoint loader. bridge's load_megatron_model builds a
        # TE model and loads STRICTLY (no strict flag), which fails on native mcore-0.18 ckpts:
        # mcore-0.18's GPTModel uses a NATIVE RMSNorm for decoder.final_layernorm (no TE
        # _extra_state), so the ckpt lacks decoder.final_layernorm._extra_state that the TE model
        # expects. So: build the model WITHOUT loading (to_megatron_model(load_weights=False)), then
        # load the ckpt via dist_checkpointing with validate_access_integrity=False (tolerate that
        # one missing FP8-metadata key — irrelevant for weight export), then export HF weights.
        # Build via the provider so we can clear gradient_accumulation_fusion: bridge specs set it
        # True (needs APEX's fused_weight_gradient_mlp_cuda CUDA ext — NOT in our env; AA pretrain
        # runs with --no-gradient-accumulation-fusion). This mirrors to_megatron_model's flow
        # (to_megatron_provider -> finalize -> provide_distributed_model) but with the flag cleared
        # so the model builds without APEX. (to_megatron_model rebuilds the config internally, so a
        # transformer_config() override does not propagate; set it on the provider instead.)
        provider = bridge.to_megatron_provider(load_weights=False)
        if hasattr(provider, "finalize"):
            provider.finalize()
        try:
            provider.gradient_accumulation_fusion = False
        except Exception:
            pass
        models = provider.provide_distributed_model(wrap_with_ddp=False)
        # Load ALL pipeline stages into the ckpt weights (PP>1 → multiple models in
        # the list; loading
        # only models[0] leaves later stages at random init → a half-correct HF export). Strip TE
        # _extra_state keys: native mcore-0.18 ckpts use a NATIVE RMSNorm for
        # decoder.final_layernorm
        # (no _extra_state), so dc.load raises "Missing key" on the model's expected
        # shards. They are
        # FP8 metadata, irrelevant for weight export (export_hf_weights reads
        # named_parameters only).
        for model in models:
            sharded_sd = model.sharded_state_dict()
            sharded_sd = {k: v for k, v in sharded_sd.items() if "_extra_state" not in k}
            dc.load(sharded_sd, iter_dir, validate_access_integrity=False)
        # sharded_sd references each model's parameters in-place -> all PP stages
        # now have ckpt weights.
        # save_hf_pretrained takes the model LIST (it unwraps Float16Module etc. internally).
        bridge.save_hf_pretrained(models, hf_out_dir)
        print(
            f"[bridge] exported via AutoBridge ({getattr(hf_config, 'model_type', '?')}) "
            f"-> {hf_out_dir}"
        )
        return hf_out_dir
