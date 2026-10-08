"""mcore (Megatron-LM) dist-checkpoint  ->  HuggingFace Qwen2ForCausalLM converter.

Moved verbatim from the former ``checkpoint_bridge.py``; wrapped as the ``"qwen2"``
``BaseCheckpointConverter``. The only interface between the Megatron pretrain process and
verl/vLLM: pretrain writes a Megatron torch_dist checkpoint; this converter exports it to
HuggingFace ``Qwen2ForCausalLM`` format, which verl/vLLM consume via ``model.path=<hf_dir>``.

Hand-written converter — the validated primary mcore->HF path. megatron-bridge's
``AutoBridge`` is installed in the pretrain env and could replace this for standard
families, but this hand path is kept because it explicitly handles the Qwen2.5 nuances
(tied-vs-untied lm_head and q/k/v bias derived from the checkpoint, unrepresentable-bias
warning) that a generic auto-export can silently get wrong. NOTE: writes
only config.json + model.safetensors — copy tokenizer files into ``hf_out_dir`` separately.
"""

from __future__ import annotations

import json
import os
import pickle

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

from alphaapollo.learning.off_policy.pretrain.base import BaseCheckpointConverter
from alphaapollo.learning.off_policy.pretrain.converters import CONVERTERS


def _latest_iter_dir(ckpt_dir: str) -> str:
    latest = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
    with open(latest) as f:
        it = f.read().strip()
    return os.path.join(ckpt_dir, f"iter_{int(it):07d}")


def _load_mcore_state_dict(iter_dir: str) -> dict:
    """Load the model tensors from a Megatron torch_dist checkpoint (standalone)."""
    with open(os.path.join(iter_dir, ".metadata"), "rb") as f:
        meta = pickle.load(f)
    sd: dict[str, torch.Tensor] = {}
    for k, v in meta.state_dict_metadata.items():
        ks = str(k)
        if ks.startswith(("optimizer.", "rng_state", "rerun_state", "dataloader")):
            continue
        try:
            sh = tuple(v.size)
        except Exception:
            continue
        if not (sh and all(isinstance(d, int) for d in sh)):
            continue
        sd[ks] = torch.zeros(sh, dtype=torch.float32)
    dcp.load(sd, checkpoint_id=iter_dir, storage_reader=FileSystemReader(iter_dir))
    return sd


def _split_qkv_grouped(qkv_block, q_dim, kv_dim, num_kv_groups):
    """Split a mcore fused-QKV tensor into HF q/k/v, honoring mcore's PER-QUERY-GROUP layout.

    mcore stores ``linear_qkv`` as ``[g0:q|k|v | g1:q|k|v | ...]`` — one
    ``(q_per_group, k_per_group, v_per_group)`` triple per query group (see mcore
    ``transformer/attention.py::_clip_linear_qkv``, which reshapes the weight to
    ``(num_query_groups_per_partition, (q+2kv)/g, -1)`` before splitting q/k/v within each
    group; verl's ``models/mcore/saver.py`` chunks by group identically). A flat ``[Q|K|V]``
    slice is correct ONLY when ``num_kv_groups == 1``; for GQA (``num_kv_groups > 1``, e.g.
    Qwen2.5-0.5B's 2 groups) a flat slice silently permutes the heads.

    This regroups so HF receives q_proj = [all heads, group order], k_proj/v_proj = [all
    kv-groups, group order]. Works for both the weight ``[q+2kv, hidden]`` and the bias
    ``[q+2kv]`` (mcore stores the bias in the same interleaved layout).
    """
    g = num_kv_groups
    q_per = q_dim // g
    kv_per = kv_dim // g
    rest = qkv_block.shape[1:]
    grouped = qkv_block.reshape(g, q_per + 2 * kv_per, *rest)
    q = grouped[:, :q_per].reshape(q_dim, *rest).contiguous()
    k = grouped[:, q_per : q_per + kv_per].reshape(kv_dim, *rest).contiguous()
    v = grouped[:, q_per + kv_per :].reshape(kv_dim, *rest).contiguous()
    return q, k, v


def _grouped_reinterleave(q, k, v, q_dim, kv_dim, num_kv_groups):
    """Inverse of :func:`_split_qkv_grouped`: rebuild mcore's per-query-group fused-QKV tensor
    ``[g0:q|k|v | g1:q|k|v | ...]`` from HF q/k/v.

    Used by the parity gate to compare the converter's HF OUTPUT back against the RAW mcore
    checkpoint tensor — this is NON-circular: a wrong extraction (flat slice, k<->v swap, wrong
    group count) yields a re-interleave that does NOT equal the raw grouped tensor, so the gate
    fails. (Comparing the converter output to a re-derived grouped reference, as a prior version
    did, was circular — both sides used ``_split_qkv_grouped`` and so agreed for any bug inside
    that helper.)
    """
    g = num_kv_groups
    q_per = q_dim // g
    kv_per = kv_dim // g
    rest = q.shape[1:]
    qr = q.reshape(g, q_per, *rest)
    kr = k.reshape(g, kv_per, *rest)
    vr = v.reshape(g, kv_per, *rest)
    per_group = torch.cat([qr, kr, vr], dim=1)  # [g, q_per + 2*kv_per, *rest]
    return per_group.reshape((q_dim + 2 * kv_dim), *rest)


def _num_layers_from_ckpt(sd) -> int:
    """Layer count from the checkpoint's per-layer tensor leading dim.

    ``num_layers`` historically came from the caller (default 28), and ``range(num_layers)``
    silently exported a TRUNCATED model when it was smaller than the checkpoint's real layer count
    (shapes still lined up, so HF loaded "clean" but wrong); too large a value IndexErrored. Derive
    from the checkpoint — the same pattern as vocab — so the export always matches the real model.
    """
    return int(sd["decoder.layers.self_attention.linear_qkv.weight"].shape[0])


def _warn_unrepresentable_biases(sd, representable_bias_keys) -> None:
    """Warn when the checkpoint carries bias tensors the HF target cannot represent.

    HF decoder targets (Qwen2/Mistral/Llama) give ``o_proj`` and the MLP projections no bias
    slot — those are hard-coded ``bias=False`` in their modeling code. A checkpoint trained
    WITHOUT ``--disable-bias-linear`` (mcore's default ``add_bias_linear=True``) therefore holds
    trained ``linear_proj.bias`` / ``mlp.linear_fc1.bias`` / ``mlp.linear_fc2.bias`` values that
    the export must drop: the exported model is then NOT the trained model. That drop is
    silent in the safetensors output, so it must be loud here. ``representable_bias_keys`` is
    the set of mcore bias suffixes the target DOES carry (e.g. the fused qkv bias).
    """
    dropped = sorted(k for k in sd if k.endswith(".bias") and k not in representable_bias_keys)
    if dropped:
        print(
            "[mcore->hf] WARNING: checkpoint trains bias tensors the HF target cannot represent "
            f"({len(dropped)} tensors, e.g. {dropped[:3]}); they are DROPPED, so the export "
            "differs from the trained model. Retrain with --disable-bias-linear "
            "(keep --add-qkv-bias for Qwen2.5's q/k/v bias) to match the HF layout."
        )


def _eos_from_tokenizer_dir(hf_out_dir):
    """eos_token_id from the tokenizer files already copied into the HF output dir.

    The checkpoint's saved Megatron args carry neither the eps nor the eod (verified against
    mcore 0.18 torch_dist saves), so the tokenizer the run will actually load is the only
    in-band source. Real tokenizer configs (Qwen family included) often carry NO
    ``eos_token_id`` key — only the ``eos_token`` STRING, with the id recoverable from
    ``added_tokens_decoder``; some carry a LIST of eos ids (transformers supports multi-eos).
    Returns the primary (first) id, or ``None`` when nothing is staged/resolvable.
    """
    try:
        with open(os.path.join(hf_out_dir, "tokenizer_config.json")) as f:
            cfg = json.load(f)
        v = cfg.get("eos_token_id")
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None  # multi-eos config: the primary stop is the first id
        if v is not None:
            return int(v)
        token = cfg.get("eos_token")
        if isinstance(token, dict):  # AddedToken-serialized form
            token = token.get("content")
        if isinstance(token, str):
            decoder = cfg.get("added_tokens_decoder")
            if isinstance(decoder, dict):  # null/list on malformed third-party files
                for sid, spec in decoder.items():
                    if isinstance(spec, dict) and spec.get("content") == token:
                        return int(sid)
        return None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _resolve_eos(eos_token_id, hf_out_dir, converter_name):
    """Explicit param wins; else the tokenizer already staged in the output dir; else warn.

    A config without eos_token_id disables EOS early-stop everywhere downstream (the HF
    rollout server resolves eos from model.config only), so an underivable value must be loud.
    """
    if eos_token_id is not None:
        return int(eos_token_id)
    eos = _eos_from_tokenizer_dir(hf_out_dir)
    if eos is None:
        print(
            f"[{converter_name}] WARNING: no eos_token_id (none passed, no tokenizer files in "
            f"{hf_out_dir}); the exported config will have none, so HF-rollout generation "
            "NEVER early-stops on EOS. Pass eos_token_id=<trained eod> or copy the tokenizer "
            "files into the output dir before exporting."
        )
    return eos


def _mcore_to_qwen2(
    sd: dict,
    num_layers=28,
    hidden=896,
    num_heads=14,
    num_kv_groups=2,
    ffn_hidden=4864,
    untied: bool = False,
) -> dict:
    head_dim = hidden // num_heads
    q_dim = num_heads * head_dim  # 896
    kv_dim = num_kv_groups * head_dim  # 128
    _warn_unrepresentable_biases(sd, {"decoder.layers.self_attention.linear_qkv.bias"})
    out: dict[str, torch.Tensor] = {}
    emb = sd["embedding.word_embeddings.weight"].float()
    out["model.embed_tokens.weight"] = emb
    # untied: the checkpoint trained a separate output layer (mcore
    # --untie-embeddings-and-output-weights). When tied, transformers re-ties lm_head from
    # embed_tokens via config — emitting a duplicate would double-store the same weights.
    if untied:
        out["lm_head.weight"] = sd["output_layer.weight"].float()
    out["model.norm.weight"] = sd["decoder.final_layernorm.weight"].float()
    qkv_w = sd["decoder.layers.self_attention.linear_qkv.weight"]
    qkv_b = sd.get("decoder.layers.self_attention.linear_qkv.bias")  # (L,1152) or None
    proj_w = sd["decoder.layers.self_attention.linear_proj.weight"]
    qkv_ln = sd["decoder.layers.self_attention.linear_qkv.layer_norm_weight"]
    fc1_w = sd["decoder.layers.mlp.linear_fc1.weight"]
    fc1_ln = sd["decoder.layers.mlp.linear_fc1.layer_norm_weight"]
    fc2_w = sd["decoder.layers.mlp.linear_fc2.weight"]
    for i in range(num_layers):
        p = f"model.layers.{i}."
        # _split_qkv_grouped returns fresh contiguous tensors (no aliasing into sd) -> no .clone().
        q, k, v = _split_qkv_grouped(qkv_w[i], q_dim, kv_dim, num_kv_groups)
        out[p + "self_attn.q_proj.weight"] = q
        out[p + "self_attn.k_proj.weight"] = k
        out[p + "self_attn.v_proj.weight"] = v
        # Attn bias: transformers/vllm create q/k/v bias even with attention_bias=False, so the
        # checkpoint MUST provide them. mcore's fused linear_qkv.bias uses the SAME per-group
        # interleaved layout as the weight -> regroup with the same helper (a flat slice would
        # permute heads for GQA, num_kv_groups>1).
        if qkv_b is not None:
            bq, bk, bv = _split_qkv_grouped(qkv_b[i], q_dim, kv_dim, num_kv_groups)
            out[p + "self_attn.q_proj.bias"] = bq
            out[p + "self_attn.k_proj.bias"] = bk
            out[p + "self_attn.v_proj.bias"] = bv
        else:
            out[p + "self_attn.q_proj.bias"] = torch.zeros(q_dim)
            out[p + "self_attn.k_proj.bias"] = torch.zeros(kv_dim)
            out[p + "self_attn.v_proj.bias"] = torch.zeros(kv_dim)
        out[p + "self_attn.o_proj.weight"] = proj_w[i].contiguous()
        out[p + "input_layernorm.weight"] = qkv_ln[i].contiguous()
        fc1 = fc1_w[i]
        out[p + "mlp.gate_proj.weight"] = fc1[0:ffn_hidden].contiguous()
        out[p + "mlp.up_proj.weight"] = fc1[ffn_hidden : 2 * ffn_hidden].contiguous()
        out[p + "mlp.down_proj.weight"] = fc2_w[i].contiguous()
        out[p + "post_attention_layernorm.weight"] = fc1_ln[i].contiguous()
    return out


def _qwen2_config(
    num_layers=28,
    hidden=896,
    num_heads=14,
    num_kv_groups=2,
    ffn_hidden=4864,
    vocab_size=151936,
    rope_theta=1_000_000,
    max_position_embeddings=32_768,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    eos_token_id=None,
) -> dict:
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
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
        "attention_bias": True,
        **({"eos_token_id": eos_token_id} if eos_token_id is not None else {}),
        "torch_dtype": "bfloat16",
        "use_cache": True,
    }


@CONVERTERS.register("qwen2")
class Qwen2Converter(BaseCheckpointConverter):
    """Convert a Megatron-LM Qwen-arch dist-checkpoint to HuggingFace Qwen2."""

    name = "qwen2"

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
        rms_norm_eps: float = None,
        eos_token_id: int = None,
    ) -> str:
        """Convert a Megatron-LM Qwen-arch dist-checkpoint to HuggingFace Qwen2."""
        from safetensors.torch import save_file

        iter_dir = _latest_iter_dir(mcore_ckpt_dir)
        sd = _load_mcore_state_dict(iter_dir)
        # Megatron pads vocab to make_vocab_size_divisible_by and may use the tokenizer's
        # actual vocab (overriding --vocab-size); the HF config MUST match the real embedding
        # size in the checkpoint or the HF load shape-mismatches.
        actual_vocab = int(sd["embedding.word_embeddings.weight"].shape[0])
        # Derive the layer count from the checkpoint too: a caller-supplied
        # num_layers smaller than the real count silently truncated the export. Warn (don't silently
        # override without trace) when the caller's value conflicts; always use the real count.
        actual_layers = _num_layers_from_ckpt(sd)
        if num_layers != actual_layers:
            print(
                f"[qwen2] WARNING: num_layers={num_layers} but checkpoint has {actual_layers} "
                f"layers; using the checkpoint's count (prevents silent truncation)."
            )
        # RMSNorm eps must match the training eps or HF logits drift. The caller's explicit
        # value; else the mcore training default 1e-5 (the checkpoint's saved args do not carry
        # the eps, and 1e-6 — an old hard-code — was wrong for the documented training path).
        actual_eps = rms_norm_eps if rms_norm_eps is not None else 1e-5
        eos = _resolve_eos(eos_token_id, hf_out_dir, "qwen2")
        # Tie derives from the checkpoint, not the Qwen2.5 default: a checkpoint trained with
        # --untie-embeddings-and-output-weights carries output_layer.weight, which the export
        # emits as lm_head.weight — tying it anyway would silently discard the trained head.
        untied = "output_layer.weight" in sd
        qwen2 = _mcore_to_qwen2(
            sd,
            actual_layers,
            hidden_size,
            num_attention_heads,
            num_query_groups,
            ffn_hidden_size,
            untied=untied,
        )
        # Save bf16 to match config torch_dtype=bfloat16 (lossless: the model trains bf16, the
        # ckpt loads as float32, so casting back is exact) and halve disk vs fp32.
        qwen2 = {k: v.to(torch.bfloat16).contiguous() for k, v in qwen2.items()}
        os.makedirs(hf_out_dir, exist_ok=True)
        save_file(qwen2, os.path.join(hf_out_dir, "model.safetensors"), metadata={"format": "pt"})
        with open(os.path.join(hf_out_dir, "config.json"), "w") as f:
            json.dump(
                _qwen2_config(
                    actual_layers,
                    hidden_size,
                    num_attention_heads,
                    num_query_groups,
                    ffn_hidden_size,
                    actual_vocab,
                    rope_theta,
                    max_position_embeddings,
                    rms_norm_eps=actual_eps,
                    tie_word_embeddings=not untied,
                    eos_token_id=eos,
                ),
                f,
                indent=2,
            )
        print(
            f"[checkpoint_bridge] exported {len(qwen2)} tensors -> {hf_out_dir} "
            f"(layers={actual_layers}, vocab={actual_vocab})"
        )
        return hf_out_dir
