"""Generic decoder-LLM mcore -> HuggingFace converter (Llama / Qwen2 / Mistral-style).

Generalizes the Qwen2 converter to the **standard HF decoder-only layout**, so a CUSTOM
decoder architecture — any ``BaseModelBuilder`` whose model is a Megatron ``GPTModel`` with the
standard embedding / decoder / MLP layout — can export to HuggingFace for the post-train
(SFT/RL) stage. The mcore->HF tensor mapping is identical across Llama / Qwen2 / Mistral dense
decoders; only ``config.json`` differs (``architectures`` / ``model_type`` + ``attention_bias``
/ ``tie_word_embeddings``).

This converter AUTO-DERIVES from the checkpoint:
- ``num_hidden_layers`` (from the per-layer tensor's leading dim),
- ``vocab_size`` (from the embedding shape),
- ``attention_bias`` (whether ``linear_qkv.bias`` is present),
- ``tie_word_embeddings`` (``True`` iff no separate ``output_layer.weight``).
So for a standard decoder arch you only pass the hidden/head/ffn dims + ``hf_model_type``.

Select with ``--converter decoder`` (alias ``--converter llama``). For NON-decoder architectures
(e.g. the ``bert`` diffusion encoder, MoE, dual-encoder) write your own
``BaseCheckpointConverter`` under ``converters/`` — this one covers dense decoder-only LMs only.
"""

from __future__ import annotations

import json
import os

import torch

from alphaapollo.learning.off_policy.pretrain.base import BaseCheckpointConverter
from alphaapollo.learning.off_policy.pretrain.converters import CONVERTERS
from alphaapollo.learning.off_policy.pretrain.converters.qwen2 import (
    _latest_iter_dir,
    _load_mcore_state_dict,
    _num_layers_from_ckpt,
    _resolve_eos,
    _split_qkv_grouped,
    _warn_unrepresentable_biases,
)

_HF_PRESETS = {
    "qwen2": ("Qwen2ForCausalLM", "qwen2"),
    "llama": ("LlamaForCausalLM", "llama"),
    "mistral": ("MistralForCausalLM", "mistral"),
}


def _mcore_to_decoder(
    sd, num_layers, hidden, num_heads, num_kv_groups, ffn_hidden, hf_model_type="qwen2"
):
    """Map a flat Megatron GPT state dict to the standard HF decoder-only state dict.

    Handles GQA (num_kv_groups), optional q/k/v bias (auto), and tied vs untied lm_head (auto).
    QKV is split per query group via ``_split_qkv_grouped`` (mcore's interleaved layout).
    o_proj.bias is emitted only for HF targets whose o_proj carries bias — Llama when
    ``attention_bias=True`` (Qwen2/Mistral hardcode o_proj ``bias=False``) — from the ckpt's
    ``linear_proj.bias``; otherwise omitted (HF would flag it UNEXPECTED).
    """
    head_dim = hidden // num_heads
    q_dim = num_heads * head_dim
    kv_dim = num_kv_groups * head_dim
    has_bias = "decoder.layers.self_attention.linear_qkv.bias" in sd
    # Bias slots differ per target (transformers 5.8.1): Qwen2 hardcodes q/k/v bias on;
    # Llama follows attention_bias (q/k/v and o_proj); Mistral hardcodes everything off.
    # Emitting a bias the target drops at load would make the export differ from what
    # survives, so emission and the warning set share one decision.
    representable = set()
    if hf_model_type in ("qwen2", "llama"):
        representable.add("decoder.layers.self_attention.linear_qkv.bias")
    if hf_model_type == "llama" and has_bias:
        representable.add("decoder.layers.self_attention.linear_proj.bias")
    _warn_unrepresentable_biases(sd, representable)
    out = {}

    emb = sd["embedding.word_embeddings.weight"].float()
    out["model.embed_tokens.weight"] = emb
    # lm_head: emit only if untied (mcore keeps a separate output_layer). If tied, transformers
    # re-ties lm_head from embed_tokens via config (tie_word_embeddings=True) — emit nothing.
    if "output_layer.weight" in sd:
        out["lm_head.weight"] = sd["output_layer.weight"].float()

    out["model.norm.weight"] = sd["decoder.final_layernorm.weight"].float()

    qkv_w = sd["decoder.layers.self_attention.linear_qkv.weight"]
    qkv_b = sd.get("decoder.layers.self_attention.linear_qkv.bias")  # (L, q+2kv) or None
    proj_w = sd["decoder.layers.self_attention.linear_proj.weight"]
    proj_b = sd.get("decoder.layers.self_attention.linear_proj.bias")  # o_proj bias, or None
    qkv_ln = sd["decoder.layers.self_attention.linear_qkv.layer_norm_weight"]
    fc1_w = sd["decoder.layers.mlp.linear_fc1.weight"]
    fc1_ln = sd["decoder.layers.mlp.linear_fc1.layer_norm_weight"]
    fc2_w = sd["decoder.layers.mlp.linear_fc2.weight"]
    # LlamaAttention.o_proj carries bias when attention_bias=True; Qwen2/Mistral hardcode False.
    emit_o_bias = hf_model_type == "llama" and has_bias

    for i in range(num_layers):
        p = f"model.layers.{i}."
        # _split_qkv_grouped returns fresh contiguous tensors (no aliasing into sd) -> no .clone().
        q, k, v = _split_qkv_grouped(qkv_w[i], q_dim, kv_dim, num_kv_groups)
        out[p + "self_attn.q_proj.weight"] = q
        out[p + "self_attn.k_proj.weight"] = k
        out[p + "self_attn.v_proj.weight"] = v
        if qkv_b is not None and hf_model_type != "mistral":
            # arch has q/k/v bias (mcore fused bias uses the same per-group layout);
            # mistral targets drop them at load (no bias slot), so they are not emitted
            bq, bk, bv = _split_qkv_grouped(qkv_b[i], q_dim, kv_dim, num_kv_groups)
            out[p + "self_attn.q_proj.bias"] = bq
            out[p + "self_attn.k_proj.bias"] = bk
            out[p + "self_attn.v_proj.bias"] = bv
        elif hf_model_type == "qwen2":
            # Qwen2Attention hardcodes q/k/v bias=True: a bias-free checkpoint must still
            # ship zero biases or the load reports missing keys (zero bias == the trained
            # bias-free math, exactly like the dedicated qwen2 converter).
            out[p + "self_attn.q_proj.bias"] = torch.zeros(q_dim)
            out[p + "self_attn.k_proj.bias"] = torch.zeros(kv_dim)
            out[p + "self_attn.v_proj.bias"] = torch.zeros(kv_dim)
        if emit_o_bias:
            # Llama ties q/k/v AND o_proj bias to attention_bias: a qkv-only-bias ckpt (the
            # documented --disable-bias-linear --add-qkv-bias layout) still owes the o slot,
            # or the load reports missing keys (zero bias == the trained bias-free math).
            out[p + "self_attn.o_proj.bias"] = (
                proj_b[i].contiguous() if proj_b is not None else torch.zeros(hidden)
            )
        out[p + "self_attn.o_proj.weight"] = proj_w[i].contiguous()
        out[p + "input_layernorm.weight"] = qkv_ln[i].contiguous()
        fc1 = fc1_w[i]
        out[p + "mlp.gate_proj.weight"] = fc1[0:ffn_hidden].contiguous()
        out[p + "mlp.up_proj.weight"] = fc1[ffn_hidden : 2 * ffn_hidden].contiguous()
        out[p + "mlp.down_proj.weight"] = fc2_w[i].contiguous()
        out[p + "post_attention_layernorm.weight"] = fc1_ln[i].contiguous()
    return out


def _decoder_config(
    hf_model_type,
    num_layers,
    hidden,
    num_heads,
    num_kv_groups,
    ffn_hidden,
    vocab_size,
    rope_theta,
    max_position_embeddings,
    attention_bias,
    tie_word_embeddings,
    rms_norm_eps=1e-5,
    eos_token_id=None,
):
    arch, model_type = _HF_PRESETS.get(
        hf_model_type, (f"{hf_model_type.title()}ForCausalLM", hf_model_type)
    )
    return {
        "architectures": [arch],
        "model_type": model_type,
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_groups,
        "intermediate_size": ffn_hidden,
        "vocab_size": vocab_size,
        "hidden_act": "silu",
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "max_position_embeddings": max_position_embeddings,
        "tie_word_embeddings": tie_word_embeddings,
        "attention_bias": attention_bias,
        **({"eos_token_id": eos_token_id} if eos_token_id is not None else {}),
        "torch_dtype": "bfloat16",
        "use_cache": True,
    }


@CONVERTERS.register("llama")
@CONVERTERS.register("decoder")
class DecoderConverter(BaseCheckpointConverter):
    """Generic Megatron-GPT -> HuggingFace decoder-only LM (Llama/Qwen2/Mistral layout)."""

    name = "decoder"

    def export(
        self,
        mcore_ckpt_dir: str,
        hf_out_dir: str,
        *,
        num_layers: int = 28,
        hidden_size: int = 896,
        num_attention_heads: int = 14,
        num_query_groups: int = 2,
        ffn_hidden_size: int = 4864,
        vocab_size: int = 151936,
        rope_theta: float = 1_000_000,
        max_position_embeddings: int = 32_768,
        hf_model_type: str = "qwen2",
        rms_norm_eps: float = None,
        eos_token_id: int = None,
    ) -> str:
        """Convert a Megatron-LM decoder dist-checkpoint to a HuggingFace decoder-only LM dir.

        ``hf_model_type`` picks the HF target (``qwen2`` | ``llama`` | ``mistral``). vocab,
        attention_bias, and tie_word_embeddings are auto-derived from the checkpoint.
        """
        from safetensors.torch import save_file

        iter_dir = _latest_iter_dir(mcore_ckpt_dir)
        sd = _load_mcore_state_dict(iter_dir)

        # auto-derive the things that must match the real ckpt or the HF load shape-mismatches
        actual_vocab = int(sd["embedding.word_embeddings.weight"].shape[0])
        actual_layers = _num_layers_from_ckpt(sd)
        if num_layers != actual_layers:
            print(
                f"[decoder] WARNING: num_layers={num_layers} but checkpoint has {actual_layers} "
                f"layers; using the checkpoint's count (prevents silent truncation)."
            )
        has_bias = "decoder.layers.self_attention.linear_qkv.bias" in sd
        tied = "output_layer.weight" not in sd  # no separate output layer -> embeddings tied
        # RMSNorm eps must match the training eps or HF logits drift. The caller's explicit
        # value; else the mcore training default 1e-5 (the checkpoint's saved args do not carry
        # the eps; 1e-6 — an old hard-code — was wrong for the documented training path).
        actual_eps = rms_norm_eps if rms_norm_eps is not None else 1e-5
        eos = _resolve_eos(eos_token_id, hf_out_dir, "decoder")

        hf = _mcore_to_decoder(
            sd,
            actual_layers,
            hidden_size,
            num_attention_heads,
            num_query_groups,
            ffn_hidden_size,
            hf_model_type=hf_model_type,
        )
        # Save bf16 to match config torch_dtype=bfloat16 (the model trains bf16; the ckpt loads as
        # float32, so casting back to bf16 on save is lossless and halves disk vs fp32).
        hf = {k: v.to(torch.bfloat16).contiguous() for k, v in hf.items()}
        os.makedirs(hf_out_dir, exist_ok=True)
        save_file(hf, os.path.join(hf_out_dir, "model.safetensors"), metadata={"format": "pt"})
        with open(os.path.join(hf_out_dir, "config.json"), "w") as f:
            json.dump(
                _decoder_config(
                    hf_model_type,
                    actual_layers,
                    hidden_size,
                    num_attention_heads,
                    num_query_groups,
                    ffn_hidden_size,
                    actual_vocab,
                    rope_theta,
                    max_position_embeddings,
                    # Per-target capability, not the checkpoint: Qwen2 slots are hardcoded
                    # on, Mistral's are off (and inert), Llama's follow the trained flag.
                    attention_bias=(
                        has_bias if hf_model_type == "llama" else hf_model_type == "qwen2"
                    ),
                    tie_word_embeddings=tied,
                    rms_norm_eps=actual_eps,
                    eos_token_id=eos,
                ),
                f,
                indent=2,
            )
        print(
            f"[decoder] exported {len(hf)} tensors -> {hf_out_dir} "
            f"(hf={hf_model_type}, layers={actual_layers}, vocab={actual_vocab}, "
            f"attn_bias={has_bias}, tied={tied})"
        )
        return hf_out_dir
