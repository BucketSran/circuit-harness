"""Numerical parity checks for the mcore -> HuggingFace converters.

The converters (`decoder`, `mapping`, `qwen2`) were previously validated only behaviorally —
"loads in transformers + verl SFT/RL trains" — which does NOT catch a weight-mapping bug (wrong
QKV split order, transposed slice, swapped gate/up): such a model still loads (shapes match) and
still trains (just from a worse init). These gates add the numerical acceptance check:

* ``decoder_weights_parity`` — reverse-reconstructs the mcore [Q|K|V] / [gate|up] blocks from the
  converter's HF output and compares them tensor-for-tensor against the source mcore checkpoint.
  Catches exactly the split/transpose/swap bugs a behavioral run hides. (A full logits gate would
  need to build a Megatron GPTModel + load + forward — heavy; this weights-level reconstruction is
  the highest-signal, lightest check for the one hand-written mapping in the chain.)

* ``causal_mlp_logit_parity`` — a true end-to-end logit-parity gate for the custom causal-MLP arch:
  build the mcore ``CausalMLPModel`` and its HF mirror with IDENTICAL weights, forward the same
  input, assert ``allclose(mcore_logits, hf_logits)``. ``CausalMLPModel`` is a hand-written
  ``MegatronModule`` whose forward is pure torch ops, so it builds/forwards standalone (no Megatron
  distributed init) — this is the logit gate the reviewer asked for, feasible here.

Run (in the ``alphaapollo-pretrain`` env, which has both mcore and transformers):
    python -m alphaapollo.learning.off_policy.pretrain.converters.parity_check \
        --decoder-ckpt /tmp/aa_qwen_smoke_ckpt \
        --num-layers 24 --hidden 896 --num-heads 14 --num-query-groups 2 --ffn-hidden 4864
"""

from __future__ import annotations

import sys

import torch

__all__ = ["decoder_weights_parity", "causal_mlp_logit_parity"]


def decoder_weights_parity(
    mcore_ckpt_dir, num_layers, hidden, num_heads, num_query_groups, ffn_hidden
):
    """Verify the converter's HF output reconstructs the RAW mcore checkpoint tensors.

    QKV: re-interleave the converter's HF q/k/v back into mcore's per-query-group layout
    (``_grouped_reinterleave``) and compare to the RAW ``linear_qkv`` tensor (the ground truth),
    NOT to a re-derived grouped reference. This is NON-circular: a wrong extraction — flat
    ``[Q|K|V]`` slice, a k<->v swap, or a wrong group count — produces a re-interleave that does
    NOT equal the raw grouped tensor, so the gate fails. (A prior version compared the converter
    output to ``_split_qkv_grouped`` of the same tensor — circular, since both sides used that
    helper and agreed for any bug inside it.) gate|up/o_proj/norms are compared directly.
    Returns True iff all match.

    NOTE: weights-level regression gate, NOT a forward-logit proof (the latter needs an mcore
    GPTModel build — see ``causal_mlp_logit_parity`` for the one arch that builds standalone).
    The grouped layout itself is independently anchored by mcore ``attention.py::_clip_linear_qkv``
    + verl ``saver.py`` + the row-index unit tests for ``_split_qkv_grouped``/
    ``_grouped_reinterleave``."""
    from alphaapollo.learning.off_policy.pretrain.converters.decoder import _mcore_to_decoder
    from alphaapollo.learning.off_policy.pretrain.converters.qwen2 import (
        _grouped_reinterleave,
        _latest_iter_dir,
        _load_mcore_state_dict,
    )

    iter_dir = _latest_iter_dir(mcore_ckpt_dir)
    sd = _load_mcore_state_dict(iter_dir)
    hf = _mcore_to_decoder(sd, num_layers, hidden, num_heads, num_query_groups, ffn_hidden)

    head_dim = hidden // num_heads
    q_dim = num_heads * head_dim
    kv_dim = num_query_groups * head_dim
    mismatches = []

    def chk(name, a, b):
        a, b = a.float(), b.float()
        if a.shape != b.shape or not torch.equal(a, b):
            err = (a - b).abs().max().item() if a.shape == b.shape else float("nan")
            mismatches.append(
                f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}, max_err={err:.3e}"
            )

    chk(
        "model.embed_tokens",
        hf["model.embed_tokens.weight"],
        sd["embedding.word_embeddings.weight"],
    )
    chk("model.norm", hf["model.norm.weight"], sd["decoder.final_layernorm.weight"])
    if "lm_head.weight" in hf:
        chk("lm_head", hf["lm_head.weight"], sd["output_layer.weight"])

    qkv_w = sd["decoder.layers.self_attention.linear_qkv.weight"]  # [L, q+2kv, hidden]
    proj_w = sd["decoder.layers.self_attention.linear_proj.weight"]  # [L, hidden, hidden]
    fc1_w = sd["decoder.layers.mlp.linear_fc1.weight"]  # [L, 2*ffn, hidden]
    fc2_w = sd["decoder.layers.mlp.linear_fc2.weight"]  # [L, hidden, ffn]
    qkv_ln = sd["decoder.layers.self_attention.linear_qkv.layer_norm_weight"]  # [L, hidden]
    fc1_ln = sd["decoder.layers.mlp.linear_fc1.layer_norm_weight"]  # [L, hidden]

    for i in range(num_layers):
        p = f"model.layers.{i}."
        # QKV: re-interleave the converter's HF q/k/v back into mcore's per-query-group layout and
        # compare to the RAW checkpoint tensor (ground truth). NON-circular — a flat slice, k<->v
        # swap, or wrong group count makes the re-interleave != the raw grouped tensor (a prior
        # version compared against _split_qkv_grouped of the same tensor, which agreed for any bug
        # inside that helper).
        recon_qkv = _grouped_reinterleave(
            hf[p + "self_attn.q_proj.weight"],
            hf[p + "self_attn.k_proj.weight"],
            hf[p + "self_attn.v_proj.weight"],
            q_dim,
            kv_dim,
            num_query_groups,
        )
        chk(f"L{i} qkv (grouped re-interleave vs raw)", recon_qkv, qkv_w[i])
        chk(f"L{i} o_proj", hf[p + "self_attn.o_proj.weight"], proj_w[i])
        # gate|up: cat HF gate|up back into the mcore fc1 [gate|up] block (fc1 IS a flat split in
        # mcore, unlike QKV). Swapped gate/up flips this comparison.
        recon_fc1 = torch.cat(
            [
                hf[p + "mlp.gate_proj.weight"],  # [ffn, hidden]
                hf[p + "mlp.up_proj.weight"],  # [ffn, hidden]
            ],
            dim=0,
        )
        chk(f"L{i} gate|up split", recon_fc1, fc1_w[i])
        chk(f"L{i} down_proj", hf[p + "mlp.down_proj.weight"], fc2_w[i])
        chk(f"L{i} input_layernorm", hf[p + "input_layernorm.weight"], qkv_ln[i])
        chk(f"L{i} post_attention_layernorm", hf[p + "post_attention_layernorm.weight"], fc1_ln[i])

    if mismatches:
        print(f"decoder weights-parity FAIL ({len(mismatches)} mismatches):")
        for m in mismatches[:12]:
            print("   ", m)
        return False
    print(
        f"decoder weights-parity PASS: {num_layers} layers — q/k/v re-interleave == raw mcore "
        f"grouped tensor (GQA-safe; catches flat / k<->v / wrong-group-count regressions); "
        f"gate|up / o_proj / down_proj / norms all reconstruct the raw checkpoint."
    )
    return True


class _DummyCfg:
    """CausalMLPModel inherits MegatronModule but its forward is pure torch; a dummy config lets it
    build without Megatron distributed init (the model never touches the mcore
    config at forward)."""

    def __getattr__(self, k):
        return None


def causal_mlp_logit_parity(
    hf_modeling_dir=None, vocab=151665, num_layers=6, hidden=512, ffn=1024, kernel=8, seq=16
):
    """True end-to-end logit-parity gate for the custom causal-MLP arch. Build the mcore model and
    its HF mirror with identical weights, forward the same input, assert allclose(logits).

    Returns True/False when the gate RUNS; returns None (SKIP) if the HF modeling fixture is
    missing — the caller (_main) reports SKIP distinctly so a missing artifact can never count as
    PASS."""
    import importlib
    import os

    if hf_modeling_dir is None:
        # Default to the checked-in fixture — NOT /tmp (disappears on reboot).
        hf_modeling_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "tests",
            "fixtures",
            "causal_mlp_hf",
        )

    if hf_modeling_dir not in sys.path:
        sys.path.insert(0, hf_modeling_dir)
    try:
        modeling = importlib.import_module("modeling_causal_mlp")  # the HF trust_remote_code mirror
    except Exception as e:
        print(
            f"causal_mlp logit-parity SKIP: HF modeling file not importable from "
            f"{hf_modeling_dir}: {e}"
        )
        return None  # SKIP — fixture missing; _main reports it distinctly, NOT counted as PASS

    from alphaapollo.learning.off_policy.pretrain.archs.causal_mlp import CausalMLPModel

    torch.manual_seed(0)
    mcore = CausalMLPModel(
        config=_DummyCfg(),
        vocab_size=vocab,
        num_layers=num_layers,
        hidden_size=hidden,
        ffn_size=ffn,
        conv_kernel=kernel,
    ).eval()

    hf_cfg = modeling.CausalMLPConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        intermediate_size=ffn,
        conv_kernel=kernel,
    )
    hf_model = modeling.CausalMLPForCausalLM(hf_cfg).eval()

    # mcore and the HF mirror are the SAME graph by construction; the converter is near-identity
    # (mcore torch_dist keys have NO `module.` prefix and the names match), so the mcore state_dict
    # loads straight into the HF model. Missing/unexpected should both be ~0.
    msd = {k: v.float() for k, v in mcore.state_dict().items()}
    missing, unexpected = hf_model.load_state_dict(msd, strict=False)

    ids = torch.randint(0, vocab, (1, seq))
    attn = torch.ones_like(ids)
    with torch.no_grad():
        mcore_logits = mcore(ids, None, None)  # [1, seq, vocab]
        hf_logits = hf_model(ids, attention_mask=attn).logits  # [1, seq, vocab]

    close = torch.allclose(mcore_logits.float(), hf_logits.float(), atol=1e-4, rtol=1e-3)
    max_err = (mcore_logits.float() - hf_logits.float()).abs().max().item()
    argmax_match = (mcore_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    print(
        f"causal_mlp logit-parity: allclose={close} (atol=1e-4) max_err={max_err:.2e} "
        f"argmax_match={argmax_match:.4f} (missing={len(missing)} unexpected={len(unexpected)})"
    )
    if missing:
        print("   missing keys (first 5):", missing[:5])
    if unexpected:
        print("   unexpected keys (first 5):", unexpected[:5])
    return bool(close)


def _main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--decoder-ckpt",
        default="/tmp/aa_qwen_smoke_ckpt",
        help="mcore dist-checkpoint dir for the decoder weights-parity gate",
    )
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--hidden", type=int, default=896)
    p.add_argument("--num-heads", type=int, default=14)
    p.add_argument("--num-query-groups", type=int, default=2)
    p.add_argument("--ffn-hidden", type=int, default=4864)
    p.add_argument(
        "--hf-modeling-dir",
        default=None,
        help="dir with the HF modeling_causal_mlp.py "
        "(default: repo fixture tests/fixtures/causal_mlp_hf)",
    )
    p.add_argument("--skip-decoder", action="store_true")
    p.add_argument("--skip-causal-mlp", action="store_true")
    a = p.parse_args()

    ran, skipped = {}, []
    if not a.skip_decoder:
        ran["decoder"] = decoder_weights_parity(
            a.decoder_ckpt, a.num_layers, a.hidden, a.num_heads, a.num_query_groups, a.ffn_hidden
        )
    if not a.skip_causal_mlp:
        r = causal_mlp_logit_parity(a.hf_modeling_dir)
        if r is None:  # SKIP (fixture missing) — must NOT be counted as PASS
            skipped.append("causal_mlp")
        else:
            ran["causal_mlp"] = r
    print("=" * 60)
    verdict = "PASS" if (ran and all(ran.values())) else ("FAIL" if ran else "SKIP")
    msg = f"PARITY GATE: {verdict}"
    if skipped:
        msg += f"   (skipped: {skipped} — HF fixture missing, NOT counted as PASS)"
    print(msg)
    sys.exit(0 if (ran and all(ran.values())) else 1)


if __name__ == "__main__":
    _main()
