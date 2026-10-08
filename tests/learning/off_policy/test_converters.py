"""Unit tests for the mcore->HF QKV per-group extraction (review issues #1 + #2).

These pin the ROW ASSIGNMENT of ``_split_qkv_grouped`` against mcore's per-query-group
interleaved layout (mcore ``transformer/attention.py::_clip_linear_qkv`` / verl
``models/mcore/saver.py``). They are deliberately NON-circular: a sentinel value is placed at
each source row so the test asserts exactly which rows land in q vs k vs v (rather than
round-tripping the converter's own op, which is true for any split order). Pure torch, no
mcore/transformers/GPU needed.
"""

from __future__ import annotations

import pytest

# The base CI job installs only [dev] (no torch); these tests are meaningless without it,
# and a module-level import error would fail COLLECTION and redden CI for the whole repo.
pytest.importorskip("torch")
pytest.importorskip("safetensors")

import torch  # noqa: E402

from alphaapollo.learning.off_policy.pretrain.converters.qwen2 import (  # noqa: E402
    _grouped_reinterleave,
    _split_qkv_grouped,
)


def _sentinel(q_dim, kv_dim, hidden=3):
    """A ``[q+2kv, hidden]`` tensor whose every row r is the constant r, so a row's destination
    (q vs k vs v, which group) is read straight off its value."""
    total = q_dim + 2 * kv_dim
    return torch.arange(total, dtype=torch.float).view(total, 1).expand(total, hidden).clone()


def test_split_qkv_grouped_row_assignment_gqa2():
    # 4 heads, 2 kv groups, head_dim 2 -> q=8, kv=4, per-group (q4|k2|v2)=8; total 16.
    # mcore layout: g0=rows[0..7] (q0-3,k4-5,v6-7), g1=rows[8..15] (q8-11,k12-13,v14-15).
    w = _sentinel(q_dim=8, kv_dim=4)
    q, k, v = _split_qkv_grouped(w, q_dim=8, kv_dim=4, num_kv_groups=2)
    assert q.shape == (8, 3)
    # q_proj = [g0 heads, g1 heads] concatenated in group order
    assert q[:, 0].tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
    assert k[:, 0].tolist() == [4, 5, 12, 13]  # g0 k=rows4-5, g1 k=rows12-13
    assert v[:, 0].tolist() == [6, 7, 14, 15]  # g0 v=rows6-7, g1 v=rows14-15


def test_split_qkv_grouped_flat_differs_for_gqa():
    # TEETH: for GQA (>1 group) a flat [Q|K|V] slice is NOT the grouped extraction. This is the
    # core of review issue #1 (flat silently permutes heads) and what the new parity gate detects.
    w = _sentinel(q_dim=8, kv_dim=4)
    q_grouped, _, _ = _split_qkv_grouped(w, 8, 4, num_kv_groups=2)
    q_flat = w[0:8]
    assert not torch.equal(q_grouped, q_flat)


def test_split_qkv_grouped_equals_flat_for_single_group():
    # Degenerate: num_kv_groups==1 -> the per-group layout collapses to flat, so flat is
    # coincidentally correct. This is why qwen2_mapping.json split_sizes [128,32,32] (1 kv group)
    # masked the bug in the example.
    w = _sentinel(q_dim=8, kv_dim=2)
    q_g, k_g, v_g = _split_qkv_grouped(w, 8, 2, num_kv_groups=1)
    assert torch.equal(q_g, w[0:8])
    assert torch.equal(k_g, w[8:10])
    assert torch.equal(v_g, w[10:12])


def test_split_qkv_grouped_bias_vector():
    # The bias ``[q+2kv]`` uses the SAME per-group interleaved layout as the weight.
    b = torch.arange(16, dtype=torch.float)  # q=8, kv=4 -> 16
    q, k, v = _split_qkv_grouped(b, q_dim=8, kv_dim=4, num_kv_groups=2)
    assert q.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]
    assert k.tolist() == [4, 5, 12, 13]
    assert v.tolist() == [6, 7, 14, 15]


# --------------------------------------------------------------------------- #
# Path B: the mapping converter's qkv_grouped rule (GQA-correct JSON path)
# --------------------------------------------------------------------------- #
from alphaapollo.learning.off_policy.pretrain.converters.mapping import _chunks  # noqa: E402


def test_chunks_qkv_grouped_rule_produces_grouped_layout():
    # A "split":"qkv_grouped" mapping rule must regroup fused QKV per query group (not flat-split).
    w = torch.arange(16, dtype=torch.float).view(16, 1).expand(16, 3).clone()  # sentinel rows
    rule = {"split": "qkv_grouped", "q_dim": 8, "kv_dim": 4, "num_query_groups": 2}
    q, k, v = _chunks(w, rule)
    assert q[:, 0].tolist() == [0, 1, 2, 3, 8, 9, 10, 11]  # grouped q (g0 heads then g1 heads)
    assert k[:, 0].tolist() == [4, 5, 12, 13]
    assert v[:, 0].tolist() == [6, 7, 14, 15]


def test_chunks_qkv_grouped_differs_from_flat_rule():
    # The qkv_grouped rule must NOT equal a flat split for GQA (>1 group): a flat split
    # silently permutes heads, which is exactly the bug class the converter must avoid.
    w = torch.arange(16, dtype=torch.float).view(16, 1).expand(16, 3).clone()
    grouped_q = _chunks(
        w, {"split": "qkv_grouped", "q_dim": 8, "kv_dim": 4, "num_query_groups": 2}
    )[0]
    flat_q = _chunks(w, {"split_sizes": [8, 4, 4], "split_dim": 0})[0]
    assert not torch.equal(grouped_q, flat_q)


# --------------------------------------------------------------------------- #
# Path A: BridgeConverter resolves + bridge supports a standard family (CPU; no ckpt build)
# --------------------------------------------------------------------------- #
def test_bridge_converter_resolves_and_supports_qwen2():
    try:
        from megatron.bridge import AutoBridge
    except Exception:
        import pytest

        pytest.skip("megatron.bridge not installed")
    from alphaapollo.learning.off_policy.pretrain.converters import get_converter

    bc = get_converter("bridge")
    assert bc.name == "bridge"
    # bridge covers the standard families via its registry (AutoBridge.supports needs an
    # `architectures` field; list_supported_models is the reliable coverage check).
    assert "Qwen2ForCausalLM" in AutoBridge.list_supported_models(), (
        "bridge should cover the qwen2 family"
    )


# --------------------------------------------------------------------------- #
# Parity gate non-circularity: _grouped_reinterleave (inverse of split) vs raw
# --------------------------------------------------------------------------- #
def _sentinel_qkv(q_dim, kv_dim, hidden=3):
    total = q_dim + 2 * kv_dim
    return torch.arange(total, dtype=torch.float).view(total, 1).expand(total, hidden).clone()


def test_grouped_reinterleave_is_inverse_of_split():
    # reinterleave(split(x)) == x: the re-interleave faithfully rebuilds the raw grouped tensor
    # from a CORRECT grouped extraction. (This is the property the parity gate relies on.)
    w = _sentinel_qkv(q_dim=8, kv_dim=4)
    q, k, v = _split_qkv_grouped(w, 8, 4, num_kv_groups=2)
    assert torch.equal(_grouped_reinterleave(q, k, v, 8, 4, 2), w)


def test_grouped_reinterleave_catches_flat_extraction():
    # TEETH: if q/k/v came from a FLAT [Q|K|V] slice (the review-#1 bug) instead of grouped, the
    # re-interleave must NOT equal the raw grouped tensor — so the parity gate FAILs. (A prior
    # gate version compared against _split_qkv_grouped of the same tensor, which passed for a
    # flat bug too — circular.)
    w = _sentinel_qkv(q_dim=8, kv_dim=4)
    q_flat, k_flat, v_flat = w[0:8], w[8:12], w[12:16]  # FLAT (wrong for GQA=2)
    assert not torch.equal(_grouped_reinterleave(q_flat, k_flat, v_flat, 8, 4, 2), w)


def test_grouped_reinterleave_catches_k_v_swap():
    # TEETH: swapping k and v (another mapping bug) must also fail the re-interleave vs raw check.
    w = _sentinel_qkv(q_dim=8, kv_dim=4)
    q, k, v = _split_qkv_grouped(w, 8, 4, num_kv_groups=2)
    assert not torch.equal(_grouped_reinterleave(q, v, k, 8, 4, 2), w)  # k,v swapped


# --------------------------------------------------------------------------- #
# export derives num_layers from the checkpoint (no silent truncation)
# --------------------------------------------------------------------------- #
def _synth_decoder_sd(num_layers, hidden=4, heads=2, kv_groups=1, ffn=3, vocab=5):
    """Minimal flat-Megatron-GPT state dict with ``num_layers`` layers (tiny dims; kv_groups=1 so
    the grouped QKV split is degenerate==flat — this test is about LAYER COUNT, not QKV layout)."""
    head_dim = hidden // heads
    q, kv = heads * head_dim, kv_groups * head_dim

    def t(*shape):
        n = 1
        for s in shape:
            n *= s
        return torch.arange(n, dtype=torch.float).reshape(*shape)

    return {
        "embedding.word_embeddings.weight": t(vocab, hidden),
        "output_layer.weight": t(vocab, hidden),
        "decoder.final_layernorm.weight": t(hidden),
        "decoder.layers.self_attention.linear_qkv.weight": t(num_layers, q + 2 * kv, hidden),
        "decoder.layers.self_attention.linear_qkv.bias": t(num_layers, q + 2 * kv),
        "decoder.layers.self_attention.linear_proj.weight": t(num_layers, hidden, hidden),
        "decoder.layers.self_attention.linear_qkv.layer_norm_weight": t(num_layers, hidden),
        "decoder.layers.mlp.linear_fc1.weight": t(num_layers, 2 * ffn, hidden),
        "decoder.layers.mlp.linear_fc1.layer_norm_weight": t(num_layers, hidden),
        "decoder.layers.mlp.linear_fc2.weight": t(num_layers, hidden, ffn),
    }


def _exported_layer_ids(out_dir):
    from safetensors import safe_open

    ids = set()
    with safe_open(out_dir + "/model.safetensors", framework="pt") as f:
        for k in f.keys():
            if ".layers." in k:
                ids.add(int(k.split(".layers.")[1].split(".")[0]))
    return ids


def test_decoder_export_derives_num_layers_from_ckpt(tmp_path, monkeypatch):
    """Teeth-check: export must derive num_layers from the checkpoint (like vocab),
    not trust the caller's value. A too-small value used to silently export a TRUNCATED model; a
    too-large value used to IndexError. Build a 3-layer synthetic ckpt, pass num_layers=99, assert
    exactly 3 layers come out in BOTH the config and the safetensors."""
    import json

    from alphaapollo.learning.off_policy.pretrain.converters import decoder as _dec

    monkeypatch.setattr(_dec, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_dec, "_load_mcore_state_dict", lambda d: _synth_decoder_sd(num_layers=3))
    out = str(tmp_path)
    _dec.DecoderConverter().export(
        out,
        out,
        num_layers=99,
        hidden_size=4,
        num_attention_heads=2,
        num_query_groups=1,
        ffn_hidden_size=3,
    )

    cfg = json.load(open(tmp_path / "config.json"))
    assert cfg["num_hidden_layers"] == 3, "export used the caller's num_layers (silent truncation)"
    assert _exported_layer_ids(out) == {0, 1, 2}, "exported the wrong layer count"


def test_qwen2_export_derives_num_layers_from_ckpt(tmp_path, monkeypatch):
    """Same teeth-check for the qwen2 converter (the DEFAULT converter — same bug
    class as decoder)."""
    import json

    from alphaapollo.learning.off_policy.pretrain.converters import qwen2 as _q

    monkeypatch.setattr(_q, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_q, "_load_mcore_state_dict", lambda d: _synth_decoder_sd(num_layers=3))
    out = str(tmp_path)
    _q.Qwen2Converter().export(
        out,
        out,
        num_layers=1,
        hidden_size=4,
        num_attention_heads=2,
        num_query_groups=1,
        ffn_hidden_size=3,
    )  # num_layers=1 is WRONG (would truncate to 1 layer)

    cfg = json.load(open(tmp_path / "config.json"))
    assert cfg["num_hidden_layers"] == 3, "qwen2 export used the caller's num_layers (truncation)"
    assert _exported_layer_ids(out) == {0, 1, 2}


# --------------------------------------------------------------------------- #
# CausalMLP generation must be padding-isolated (batched == unbatched)
# --------------------------------------------------------------------------- #
def test_causal_mlp_generation_is_padding_independent():
    """Teeth-check: the causal Conv1d has a bias, so pad positions become non-zero
    after a conv layer and leak into neighbouring real tokens' causal conv windows -> the same
    prompt yields a different greedy decode when run alone vs left-padded in a batch. Per-layer
    re-masking (pad -> 0 at every conv input) makes batched == unbatched. Without the fix, the
    assert fails (solo != batched)."""
    import importlib
    import os
    import sys

    import torch

    fixture_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "causal_mlp_hf")
    )
    if fixture_dir not in sys.path:
        sys.path.insert(0, fixture_dir)
    modeling = importlib.import_module("modeling_causal_mlp")

    torch.manual_seed(0)
    cfg = modeling.CausalMLPConfig(
        vocab_size=64,
        hidden_size=24,
        num_hidden_layers=4,
        intermediate_size=48,
        conv_kernel=4,
        eos_token_id=63,
        pad_token_id=0,
    )
    model = modeling.CausalMLPForCausalLM(cfg).eval()

    prompt_a = [5, 6, 7]
    prompt_b = [10, 11, 12, 13, 14]
    max_p = max(len(prompt_a), len(prompt_b))  # 5
    new = 6

    # Unbatched A (no padding).
    solo = model.generate(
        input_ids=torch.tensor([prompt_a]),
        attention_mask=torch.ones(1, len(prompt_a), dtype=torch.long),
        max_new_tokens=new,
        do_sample=False,
        use_cache=False,
    )[0].tolist()
    resp_solo = solo[len(prompt_a) :]

    # Batched: A left-padded to max_p alongside B; A's real tokens are NOT at the sequence start.
    def left_pad(ids):
        return [0] * (max_p - len(ids)) + ids

    batch = model.generate(
        input_ids=torch.tensor([left_pad(prompt_a), prompt_b]),
        attention_mask=torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long),
        max_new_tokens=new,
        do_sample=False,
        use_cache=False,
    )
    resp_batched = batch[0].tolist()[max_p:]

    assert resp_solo == resp_batched, (
        f"CausalMLP generation is NOT padding-isolated: solo={resp_solo} batched={resp_batched}"
    )


# --------------------------------------------------------------------------- #
# RMSNorm eps must default to the mcore training default (1e-5), not 1e-6
# --------------------------------------------------------------------------- #
def test_decoder_export_default_eps_is_mcore_default(tmp_path, monkeypatch):
    """The exported ``rms_norm_eps`` must default to 1e-5 (mcore ``layernorm_epsilon``
    default), NOT the stale hard-coded 1e-6 — and an explicit value must be honored. Without this,
    a ckpt trained via the documented command (mcore default 1e-5) is exported
    with 1e-6 -> HF logits
    drift. Reverting to 1e-6 fails the first assertion."""
    import json

    from alphaapollo.learning.off_policy.pretrain.converters import decoder as _dec

    monkeypatch.setattr(_dec, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_dec, "_load_mcore_state_dict", lambda d: _synth_decoder_sd(num_layers=3))

    out = str(tmp_path)
    _dec.DecoderConverter().export(
        out,
        out,
        num_layers=3,
        hidden_size=4,
        num_attention_heads=2,
        num_query_groups=1,
        ffn_hidden_size=3,
    )
    cfg = json.load(open(tmp_path / "config.json"))
    assert cfg["rms_norm_eps"] == 1e-5, (
        f"default rms_norm_eps must be the mcore default 1e-5, got {cfg['rms_norm_eps']}"
    )

    out2 = str(tmp_path / "o2")
    _dec.DecoderConverter().export(
        out2,
        out2,
        num_layers=3,
        hidden_size=4,
        num_attention_heads=2,
        num_query_groups=1,
        ffn_hidden_size=3,
        rms_norm_eps=1e-6,
    )
    cfg2 = json.load(open(tmp_path / "o2" / "config.json"))
    assert cfg2["rms_norm_eps"] == 1e-6, "explicit rms_norm_eps override not honored"


def test_normalization_eps_materially_changes_logits():
    """Demonstration: normalization eps is NOT weight-preserved — identical weights with
    eps=1e-5 vs 1e-6 yield DIFFERENT logits. Hence the converter must emit the TRAINING eps, not a
    hard-coded constant. (Uses the CausalMLP fixture's LayerNorm; the principle is identical for the
    RMSNorm in the qwen2/decoder targets.)"""
    import importlib
    import os
    import sys

    import torch

    fixture_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "causal_mlp_hf")
    )
    if fixture_dir not in sys.path:
        sys.path.insert(0, fixture_dir)
    modeling = importlib.import_module("modeling_causal_mlp")

    def build(eps):
        torch.manual_seed(1)
        cfg = modeling.CausalMLPConfig(
            vocab_size=64, hidden_size=24, num_hidden_layers=3, intermediate_size=48, conv_kernel=4
        )
        cfg.layer_norm_eps = eps
        return modeling.CausalMLPForCausalLM(cfg).eval()

    m5, m6 = build(1e-5), build(1e-6)
    m6.load_state_dict(m5.state_dict(), strict=False)  # identical weights; only eps differs
    ids = torch.tensor([[5, 6, 7, 8]])
    with torch.no_grad():
        logits_5 = m5(ids).logits
        logits_6 = m6(ids).logits
    assert not torch.allclose(logits_5, logits_6, atol=1e-5), (
        "normalization eps did not change logits — then the F1 fix would be a no-op"
    )


# --------------------------------------------------------------------------- #
# CausalMLP converter exists + export path round-trips
# --------------------------------------------------------------------------- #
def test_causal_mlp_converter_round_trip(tmp_path, monkeypatch):
    """The CausalMLP converter must (a) be registered, and (b) export a torch_dist
    checkpoint to an HF dir whose weights reload cleanly and reproduce logits. The old parity gate
    loaded an in-memory mcore state_dict DIRECTLY into the HF fixture, bypassing the export path
    entirely; this drives the converter's real read->map->write (``_load_mcore_state_dict`` is fed a
    synthetic CausalMLP sd; producing a real training torch_dist ckpt needs Megatron parallel_state
    init and is a manual GPU step, like the bridge converter)."""
    import importlib
    import os
    import sys

    import torch
    from safetensors.torch import load_file

    from alphaapollo.learning.off_policy.pretrain.converters import causal_mlp as _cm
    from alphaapollo.learning.off_policy.pretrain.converters import get_converter

    # (a) registered
    assert get_converter("causal_mlp").name == "causal_mlp"

    # Build a real HF fixture model; its state_dict stands in for the (synthetic) torch_dist sd.
    fixture_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "causal_mlp_hf")
    )
    if fixture_dir not in sys.path:
        sys.path.insert(0, fixture_dir)
    modeling = importlib.import_module("modeling_causal_mlp")

    torch.manual_seed(0)
    cfg = modeling.CausalMLPConfig(
        vocab_size=64, hidden_size=24, num_hidden_layers=3, intermediate_size=48, conv_kernel=4
    )
    cfg.layer_norm_eps = 1e-5
    src = modeling.CausalMLPForCausalLM(cfg).eval()
    synth_sd = {k: v.detach().clone() for k, v in src.state_dict().items()}

    monkeypatch.setattr(_cm, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_cm, "_load_mcore_state_dict", lambda d: synth_sd)

    out = str(tmp_path)
    _cm.CausalMLPConverter().export(out, out, hidden_size=24, ffn_size=48, conv_kernel=4)

    # (b) exported safetensors reload cleanly into a fresh model and reproduce
    # logits (bf16 round-trip).
    exported = load_file(os.path.join(out, "model.safetensors"))
    fresh = modeling.CausalMLPForCausalLM(cfg).eval()
    missing, unexpected = fresh.load_state_dict(exported, strict=False)
    assert not missing, f"missing keys after export: {missing[:5]}"
    assert not unexpected, f"unexpected keys after export: {unexpected[:5]}"

    ids = torch.tensor([[5, 6, 7, 8]])
    with torch.no_grad():
        src_logits = src(ids).logits.float()
        fresh_logits = fresh(ids).logits.float()
    assert torch.allclose(src_logits, fresh_logits, atol=1e-2), (
        "CausalMLP converter export did not round-trip logits"
    )


def test_causal_mlp_converter_strips_prefix(tmp_path, monkeypatch):
    """If Megatron saved the CausalMLP tensors under a top-level prefix (e.g.
    ``model.``), the converter must strip it so the keys match the HF mirror."""
    import os

    import torch

    from alphaapollo.learning.off_policy.pretrain.converters import causal_mlp as _cm

    synth = {
        "model.embed.weight": torch.zeros(8, 4),
        "model.layers.0.norm1.weight": torch.zeros(4),
        "model.layers.0.norm1.bias": torch.zeros(4),
        "model.final_norm.weight": torch.zeros(4),
        "model.lm_head.weight": torch.zeros(8, 4),
    }
    monkeypatch.setattr(_cm, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_cm, "_load_mcore_state_dict", lambda d: synth)
    out = str(tmp_path)
    _cm.CausalMLPConverter().export(
        out, out, hidden_size=4, ffn_size=8, conv_kernel=4, vocab_size=8
    )
    from safetensors.torch import load_file

    keys = set(load_file(os.path.join(out, "model.safetensors")).keys())
    assert "embed.weight" in keys and "layers.0.norm1.weight" in keys, (
        f"prefix not stripped: {sorted(keys)}"
    )
    assert all(not k.startswith("model.") for k in keys), "model. prefix leaked into export"


# --------------------------------------------------------------------------- #
# Unrepresentable-bias warning + tie derivation (the exported model must BE
# the trained model)
# --------------------------------------------------------------------------- #
def test_qwen2_export_warns_when_ckpt_trains_unrepresentable_biases(tmp_path, monkeypatch, capsys):
    """Teeth-check: HF Qwen2 hardcodes o_proj/MLP bias=False, so a checkpoint trained
    without --disable-bias-linear carries biases the export must DROP — the exported
    model then differs from the trained one, which must be LOUD (naming the tensors and
    the retrain flag), not silent."""
    from alphaapollo.learning.off_policy.pretrain.converters import qwen2 as _q

    sd = dict(_synth_decoder_sd(num_layers=2))
    sd["decoder.layers.self_attention.linear_proj.bias"] = torch.full((2, 4), 0.5)
    sd["decoder.layers.mlp.linear_fc1.bias"] = torch.full((2, 6), 0.5)
    monkeypatch.setattr(_q, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_q, "_load_mcore_state_dict", lambda d: sd)
    out = str(tmp_path)
    _q.Qwen2Converter().export(
        out,
        out,
        num_layers=2,
        hidden_size=4,
        num_attention_heads=2,
        num_query_groups=1,
        ffn_hidden_size=3,
    )
    printed = capsys.readouterr().out
    assert "linear_proj.bias" in printed, "warning must name a dropped tensor"
    assert "--disable-bias-linear" in printed, "warning must name the retrain fix"

    # Qwen2.5 layout (qkv bias only) exports clean — no warning.
    capsys.readouterr()
    monkeypatch.setattr(_q, "_load_mcore_state_dict", lambda d: _synth_decoder_sd(num_layers=2))
    _q.Qwen2Converter().export(
        out,
        out,
        num_layers=2,
        hidden_size=4,
        num_attention_heads=2,
        num_query_groups=1,
        ffn_hidden_size=3,
    )
    assert "cannot represent" not in capsys.readouterr().out


def test_decoder_export_warns_when_ckpt_trains_unrepresentable_biases(
    tmp_path, monkeypatch, capsys
):
    """Same warning on the generic decoder converter (qwen2 target drops o_proj bias too)."""
    from alphaapollo.learning.off_policy.pretrain.converters import decoder as _dec

    sd = dict(_synth_decoder_sd(num_layers=2))
    sd["decoder.layers.mlp.linear_fc2.bias"] = torch.full((2, 4), 0.5)
    monkeypatch.setattr(_dec, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_dec, "_load_mcore_state_dict", lambda d: sd)
    out = str(tmp_path)
    _dec.DecoderConverter().export(
        out,
        out,
        num_layers=2,
        hidden_size=4,
        num_attention_heads=2,
        num_query_groups=1,
        ffn_hidden_size=3,
    )
    assert "linear_fc2.bias" in capsys.readouterr().out


def test_qwen2_export_tie_follows_checkpoint(tmp_path, monkeypatch):
    """Untied checkpoint (output_layer.weight present) must export lm_head.weight and set
    tie_word_embeddings=False; tied checkpoint must emit no lm_head.weight and tie=True.
    The old converter hard-coded tie=True, silently discarding a trained output layer."""
    import json

    from safetensors import safe_open

    from alphaapollo.learning.off_policy.pretrain.converters import qwen2 as _q

    monkeypatch.setattr(_q, "_latest_iter_dir", lambda d: d)
    kwargs = dict(
        num_layers=2, hidden_size=4, num_attention_heads=2, num_query_groups=1, ffn_hidden_size=3
    )

    def keys_and_cfg(sd):
        monkeypatch.setattr(_q, "_load_mcore_state_dict", lambda d: sd)
        out = str(tmp_path / ("untied" if "output_layer.weight" in sd else "tied"))
        _q.Qwen2Converter().export(out, out, **kwargs)
        with safe_open(out + "/model.safetensors", framework="pt") as f:
            keys = set(f.keys())
        return keys, json.load(open(out + "/config.json"))

    untied_sd = _synth_decoder_sd(num_layers=2)  # fixture carries output_layer.weight
    keys, cfg = keys_and_cfg(untied_sd)
    assert "lm_head.weight" in keys, "trained output layer silently dropped"
    assert cfg["tie_word_embeddings"] is False

    tied_sd = dict(untied_sd)
    del tied_sd["output_layer.weight"]
    keys, cfg = keys_and_cfg(tied_sd)
    assert "lm_head.weight" not in keys, "tied ckpt double-stores embed as lm_head"
    assert cfg["tie_word_embeddings"] is True


# --------------------------------------------------------------------------- #
# EOS in the exported config (without it, HF rollout never early-stops)
# --------------------------------------------------------------------------- #
def test_qwen2_export_writes_eos_param_tokenizer_or_warns(tmp_path, monkeypatch, capsys):
    """The exported config must carry eos_token_id when it is knowable: explicit param wins;
    else the tokenizer already staged in the output dir; when neither exists the export must
    WARN (a config without eos silently disables early-stop in the HF rollout server)."""
    import json
    import os

    from alphaapollo.learning.off_policy.pretrain.converters import qwen2 as _q

    monkeypatch.setattr(_q, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_q, "_load_mcore_state_dict", lambda d: _synth_decoder_sd(num_layers=2))
    kwargs = dict(
        num_layers=2, hidden_size=4, num_attention_heads=2, num_query_groups=1, ffn_hidden_size=3
    )

    # explicit param
    out = str(tmp_path / "a")
    _q.Qwen2Converter().export(out, out, eos_token_id=151915, **kwargs)
    assert json.load(open(out + "/config.json"))["eos_token_id"] == 151915

    # tokenizer staged in the output dir
    out = str(tmp_path / "b")
    os.makedirs(out, exist_ok=True)
    with open(out + "/tokenizer_config.json", "w") as f:
        json.dump({"eos_token_id": 151643}, f)
    _q.Qwen2Converter().export(out, out, **kwargs)
    assert json.load(open(out + "/config.json"))["eos_token_id"] == 151643

    # underivable: loud warning, key omitted
    capsys.readouterr()
    out = str(tmp_path / "c")
    os.makedirs(out, exist_ok=True)
    _q.Qwen2Converter().export(out, out, **kwargs)
    printed = capsys.readouterr().out
    assert "NEVER early-stops" in printed
    assert "eos_token_id" not in json.load(open(out + "/config.json"))


def test_causal_mlp_export_ships_modeling_file_and_fixture_config_shape(tmp_path, monkeypatch):
    """auto_map names modeling_causal_mlp.py — the converter must ship it (from the package,
    not tests/) so trust_remote_code resolves at RL load. The config must mirror the validated
    fixture: architectures TransformersForCausalLM (vLLM generic-wrapper fallback) and the
    transformers-5 `dtype` key."""
    import importlib
    import json
    import os
    import sys

    from alphaapollo.learning.off_policy.pretrain.converters import causal_mlp as _cm

    fixture_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "causal_mlp_hf")
    )
    if fixture_dir not in sys.path:
        sys.path.insert(0, fixture_dir)
    modeling = importlib.import_module("modeling_causal_mlp")
    cfg = modeling.CausalMLPConfig(
        vocab_size=64, hidden_size=24, num_hidden_layers=2, intermediate_size=48, conv_kernel=4
    )
    synth_sd = {
        k: v.detach().clone() for k, v in modeling.CausalMLPForCausalLM(cfg).state_dict().items()
    }
    monkeypatch.setattr(_cm, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_cm, "_load_mcore_state_dict", lambda d: synth_sd)

    out = str(tmp_path)
    _cm.CausalMLPConverter().export(
        out, out, hidden_size=24, ffn_size=48, conv_kernel=4, eos_token_id=63
    )

    assert os.path.exists(os.path.join(out, "modeling_causal_mlp.py")), (
        "converter wrote an auto_map it does not satisfy"
    )
    shipped = open(os.path.join(out, "modeling_causal_mlp.py")).read()
    fixture = open(os.path.join(fixture_dir, "modeling_causal_mlp.py")).read()
    assert shipped == fixture, "shipped modeling file drifted from the validated fixture"

    exported_cfg = json.load(open(out + "/config.json"))
    assert exported_cfg["architectures"] == ["TransformersForCausalLM"]
    assert "dtype" in exported_cfg and "torch_dtype" not in exported_cfg
    assert exported_cfg["eos_token_id"] == 63


def test_mistral_target_skips_qkv_bias_and_warns(tmp_path, monkeypatch, capsys):
    """MistralAttention hardcodes bias=False on q/k/v too: a biased checkpoint exported to
    mistral must not emit q/k/v biases HF would drop, and the unrepresentable-bias warning
    must fire for them."""

    from safetensors import safe_open

    from alphaapollo.learning.off_policy.pretrain.converters import decoder as _dec

    monkeypatch.setattr(_dec, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_dec, "_load_mcore_state_dict", lambda d: _synth_decoder_sd(num_layers=2))
    out = str(tmp_path)
    _dec.DecoderConverter().export(
        out,
        out,
        num_layers=2,
        hidden_size=4,
        num_attention_heads=2,
        num_query_groups=1,
        ffn_hidden_size=3,
        hf_model_type="mistral",
    )
    assert "linear_qkv.bias" in capsys.readouterr().out
    with safe_open(out + "/model.safetensors", framework="pt") as f:
        assert not [k for k in f.keys() if k.endswith("_proj.bias")], (
            "mistral export emitted bias tensors the target drops at load"
        )


def test_mapping_export_refuses_empty_result(tmp_path, monkeypatch):
    """An all-optional mapping that matches nothing must fail loudly, not write an empty
    safetensors and exit 0."""
    import pytest

    from alphaapollo.learning.off_policy.pretrain.converters import mapping as _mp

    monkeypatch.setattr(_mp, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_mp, "_load_mcore_state_dict", lambda d: _synth_decoder_sd(num_layers=2))
    mapping = {"rules": [{"from": "absent.key", "to": "x", "optional": True}]}
    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 4,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 3,
        "vocab_size": 5,
    }
    with pytest.raises(ValueError, match="0 tensors"):
        _mp.MappingConverter().export(
            str(tmp_path), str(tmp_path), mapping=mapping, config=cfg, num_layers=2
        )


# --------------------------------------------------------------------------- #
# Round-4: real tokenizer shapes and per-target bias metadata
# --------------------------------------------------------------------------- #
def test_qwen2_export_resolves_eos_from_real_shaped_tokenizer(tmp_path, monkeypatch):
    """Real Qwen-family tokenizer_config.json carries NO eos_token_id key — only the
    eos_token STRING with the id in added_tokens_decoder (verified against the actual
    Qwen2.5-0.5B files). The fallback must resolve that shape, and a LIST-valued
    eos_token_id (multi-eos, a transformers-supported shape) must not crash the export."""
    import json
    import os

    from alphaapollo.learning.off_policy.pretrain.converters import qwen2 as _q

    monkeypatch.setattr(_q, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_q, "_load_mcore_state_dict", lambda d: _synth_decoder_sd(num_layers=2))
    kwargs = dict(
        num_layers=2, hidden_size=4, num_attention_heads=2, num_query_groups=1, ffn_hidden_size=3
    )

    # real Qwen2.5 shape: string eos_token, id only in added_tokens_decoder
    out = str(tmp_path / "real")
    os.makedirs(out, exist_ok=True)
    with open(out + "/tokenizer_config.json", "w") as f:
        json.dump(
            {
                "eos_token": "<|endoftext|>",
                "added_tokens_decoder": {
                    "151643": {"content": "<|endoftext|>"},
                    "151644": {"content": "<|im_end|>"},
                },
            },
            f,
        )
    _q.Qwen2Converter().export(out, out, **kwargs)
    assert json.load(open(out + "/config.json"))["eos_token_id"] == 151643, (
        "real-shaped Qwen tokenizer failed to yield eos"
    )

    # multi-eos list: primary stop is the first id, and the export must not crash
    out = str(tmp_path / "list")
    os.makedirs(out, exist_ok=True)
    with open(out + "/tokenizer_config.json", "w") as f:
        json.dump({"eos_token_id": [128001, 128009]}, f)
    _q.Qwen2Converter().export(out, out, **kwargs)
    assert json.load(open(out + "/config.json"))["eos_token_id"] == 128001


def test_causal_mlp_export_normalizes_string_eos(tmp_path, monkeypatch):
    """The eos param flows through _resolve_eos's int() normalization — a string "63" must
    land in config.json as the int 63, not as a string that breaks GenerationConfig at
    generate time."""
    import importlib
    import json
    import os
    import sys

    from alphaapollo.learning.off_policy.pretrain.converters import causal_mlp as _cm

    fixture_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "causal_mlp_hf")
    )
    if fixture_dir not in sys.path:
        sys.path.insert(0, fixture_dir)
    modeling = importlib.import_module("modeling_causal_mlp")
    cfg = modeling.CausalMLPConfig(
        vocab_size=64, hidden_size=24, num_hidden_layers=2, intermediate_size=48, conv_kernel=4
    )
    synth_sd = {
        k: v.detach().clone() for k, v in modeling.CausalMLPForCausalLM(cfg).state_dict().items()
    }
    monkeypatch.setattr(_cm, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_cm, "_load_mcore_state_dict", lambda d: synth_sd)

    out = str(tmp_path)
    _cm.CausalMLPConverter().export(
        out, out, hidden_size=24, ffn_size=48, conv_kernel=4, eos_token_id="63"
    )
    assert json.load(open(out + "/config.json"))["eos_token_id"] == 63


def test_decoder_attention_bias_follows_target_capability(tmp_path, monkeypatch):
    """attention_bias is target capability, not checkpoint state: Qwen2 slots are hardcoded
    on (and a bias-free ckpt zero-fills them so the load report stays clean); Mistral's are
    off; Llama's follow the trained flag."""
    import json

    from safetensors import safe_open

    from alphaapollo.learning.off_policy.pretrain.converters import decoder as _dec

    monkeypatch.setattr(_dec, "_latest_iter_dir", lambda d: d)
    kwargs = dict(
        num_layers=2, hidden_size=4, num_attention_heads=2, num_query_groups=1, ffn_hidden_size=3
    )

    # qwen2 target, bias-FREE checkpoint: flag True + zero-filled bias tensors
    sd = _synth_decoder_sd(num_layers=2)
    del sd["decoder.layers.self_attention.linear_qkv.bias"]
    monkeypatch.setattr(_dec, "_load_mcore_state_dict", lambda d: sd)
    out = str(tmp_path / "q2")
    _dec.DecoderConverter().export(out, out, hf_model_type="qwen2", **kwargs)
    cfg = json.load(open(out + "/config.json"))
    assert cfg["attention_bias"] is True
    with safe_open(out + "/model.safetensors", framework="pt") as f:
        assert f.get_tensor("model.layers.0.self_attn.q_proj.bias") is not None

    # mistral target, biased checkpoint: flag False, no bias tensors
    monkeypatch.setattr(_dec, "_load_mcore_state_dict", lambda d: _synth_decoder_sd(num_layers=2))
    out = str(tmp_path / "ms")
    _dec.DecoderConverter().export(out, out, hf_model_type="mistral", **kwargs)
    cfg = json.load(open(out + "/config.json"))
    assert cfg["attention_bias"] is False
    with safe_open(out + "/model.safetensors", framework="pt") as f:
        assert not [k for k in f.keys() if k.endswith("_proj.bias")]

    # llama target, qkv-only-bias ckpt (the documented --disable-bias-linear --add-qkv-bias
    # layout; the synth sd carries qkv bias but no linear_proj.bias): attention_bias=True ties
    # q/k/v AND o_proj to the flag, so o_proj.bias must be zero-filled or the load reports
    # missing keys.
    out = str(tmp_path / "ll")
    _dec.DecoderConverter().export(out, out, hf_model_type="llama", **kwargs)
    cfg = json.load(open(out + "/config.json"))
    assert cfg["attention_bias"] is True
    with safe_open(out + "/model.safetensors", framework="pt") as f:
        o = f.get_tensor("model.layers.0.self_attn.o_proj.bias")
        assert float(o.abs().max()) == 0.0, "unbacked o_proj.bias must be zero-filled"


def test_qwen2_export_tolerates_malformed_added_tokens_decoder(tmp_path, monkeypatch, capsys):
    """A staged tokenizer_config.json with a null/list added_tokens_decoder (third-party
    malformation) must degrade to the loud no-eos warning, not crash the export."""
    import json
    import os

    from alphaapollo.learning.off_policy.pretrain.converters import qwen2 as _q

    out = str(tmp_path)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tokenizer_config.json"), "w") as f:
        json.dump({"eos_token": "<|eot|>", "added_tokens_decoder": None}, f)
    monkeypatch.setattr(_q, "_latest_iter_dir", lambda d: d)
    monkeypatch.setattr(_q, "_load_mcore_state_dict", lambda d: _synth_decoder_sd(num_layers=2))
    _q.Qwen2Converter().export(
        out,
        out,
        num_layers=2,
        hidden_size=4,
        num_attention_heads=2,
        num_query_groups=1,
        ffn_hidden_size=3,
    )
    assert "NEVER early-stops" in capsys.readouterr().out
    assert "eos_token_id" not in json.load(open(out + "/config.json"))
