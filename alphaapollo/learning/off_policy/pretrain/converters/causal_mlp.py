"""CausalMLP Megatron -> HuggingFace converter.

``CausalMLPModel`` (``archs/causal_mlp.py``) is a fully-custom causal LM — causal depthwise-conv
token mixing + GLU channel mixing, residual pre-norm blocks. Its module tree
(``embed`` / ``layers.{i}.{norm1,conv,norm2,mlp}`` / ``final_norm`` / ``lm_head``) is IDENTICAL to
the HF mirror at ``tests/fixtures/causal_mlp_hf/modeling_causal_mlp.py`` (same names, including the
double ``conv.conv`` — ``CausalDepthwiseConv.conv`` is an ``nn.Conv1d``). So this converter is a
near-identity copy: it reads the ``torch_dist`` checkpoint via ``_load_mcore_state_dict`` —
the same real read path the qwen2/decoder converters use, so the export is proven on the path
production checkpoints take —
strips any common top-level prefix Megatron may have prepended, and writes HF safetensors + a
CausalMLP ``config.json`` whose ``auto_map`` points at the modeling file.

Verification: the export path (``torch_dist`` read -> key map -> safetensors + config -> reload ->
logit parity) is covered by ``test_causal_mlp_converter_round_trip``. A full end-to-end run against
a checkpoint produced by Megatron *training* is a manual GPU step — producing that sharded
``torch_dist`` checkpoint needs Megatron ``parallel_state`` init (``sharded_state_dict`` asserts the
TP group is up), the same bar as the ``bridge`` converter. The converter itself calls the real
``_load_mcore_state_dict`` at runtime and so works unchanged on a real training checkpoint.

Select with ``--converter causal_mlp`` (or ``get_converter('causal_mlp')``).
"""

from __future__ import annotations

import json
import os
import re
import shutil

import torch

from alphaapollo.learning.off_policy.pretrain.base import BaseCheckpointConverter
from alphaapollo.learning.off_policy.pretrain.converters import CONVERTERS
from alphaapollo.learning.off_policy.pretrain.converters.qwen2 import (
    _latest_iter_dir,
    _load_mcore_state_dict,
    _resolve_eos,
)

# Canonical CausalMLP top-level keys (mcore model == HF mirror, identical tree).
_CAUSAL_MLP_TOPS = {"embed", "layers", "final_norm", "lm_head"}


def _detect_prefix(sd) -> str:
    """Strip a single common top-level prefix Megatron may have prepended to every key
    (e.g. ``model.`` / ``module.``). If the checkpoint stored the CausalMLP keys flat (the common
    case), returns ''. Returns '' rather than guessing when the layout is ambiguous."""
    tops = {k.split(".", 1)[0] for k in sd}
    if len(tops) == 1 and tops.isdisjoint(_CAUSAL_MLP_TOPS):
        return next(iter(tops)) + "."
    return ""


def _causal_mlp_num_layers(hf: dict) -> int:
    """Layer count from the exported keys (``layers.{i}.norm1.weight``)."""
    ids = [
        int(m.group(1)) for k in hf for m in [re.fullmatch(r"layers\.(\d+)\.norm1\.weight", k)] if m
    ]
    return (max(ids) + 1) if ids else 0


@CONVERTERS.register("causal_mlp")
class CausalMLPConverter(BaseCheckpointConverter):
    """Convert a Megatron CausalMLP torch_dist checkpoint to the HF CausalMLP mirror dir."""

    name = "causal_mlp"

    def export(
        self,
        mcore_ckpt_dir: str,
        hf_out_dir: str,
        *,
        hidden_size: int = 512,
        ffn_size: int = 1024,
        conv_kernel: int = 8,
        vocab_size: int = None,
        layer_norm_eps: float = 1e-5,
        eos_token_id: int = None,
        bos_token_id: int = None,
    ) -> str:
        """Read the CausalMLP torch_dist checkpoint and write an HF CausalMLP dir (safetensors +
        config.json + the modeling file the config's auto_map names). ``num_layers``/``vocab_size``
        are derived from the checkpoint when present."""
        from safetensors.torch import save_file

        iter_dir = _latest_iter_dir(mcore_ckpt_dir)
        sd = _load_mcore_state_dict(iter_dir)  # the REAL torch_dist read path
        if not sd:
            raise ValueError(
                f"CausalMLP converter read 0 tensors from {iter_dir}. The checkpoint must be a "
                "Megatron torch_dist checkpoint with the CausalMLP model tensors in its metadata."
            )

        prefix = _detect_prefix(sd)
        hf = {
            (k[len(prefix) :] if prefix and k.startswith(prefix) else k): v for k, v in sd.items()
        }

        actual_layers = _causal_mlp_num_layers(hf)
        actual_vocab = int(hf["embed.weight"].shape[0]) if "embed.weight" in hf else vocab_size
        if actual_vocab is None:
            raise ValueError("vocab_size not derivable (no embed.weight) and none passed.")
        eos = _resolve_eos(eos_token_id, hf_out_dir, "causal_mlp")

        os.makedirs(hf_out_dir, exist_ok=True)
        # auto_map names modeling_causal_mlp.py; without it in the dir, trust_remote_code
        # cannot resolve the arch at RL load time. Ship it from the package copy.
        src = os.path.join(os.path.dirname(__file__), "hf_models", "modeling_causal_mlp.py")
        shutil.copyfile(src, os.path.join(hf_out_dir, "modeling_causal_mlp.py"))
        # bf16 to match config dtype=bfloat16 (lossless if trained bf16; ckpt loads fp32).
        save_file(
            {k: v.to(torch.bfloat16).contiguous() for k, v in hf.items()},
            os.path.join(hf_out_dir, "model.safetensors"),
            metadata={"format": "pt"},
        )
        cfg = {
            # TransformersForCausalLM (NOT CausalMLPForCausalLM) keeps the vLLM generic-wrapper
            # fallback loadable; auto_map is what the HF-rollout path follows. Mirrors the
            # checked-in fixture config.
            "architectures": ["TransformersForCausalLM"],
            "model_type": "causal_mlp",
            "auto_map": {
                "AutoConfig": "modeling_causal_mlp.CausalMLPConfig",
                "AutoModelForCausalLM": "modeling_causal_mlp.CausalMLPForCausalLM",
            },
            "hidden_size": hidden_size,
            "num_hidden_layers": actual_layers,
            "intermediate_size": ffn_size,
            "vocab_size": actual_vocab,
            "conv_kernel": conv_kernel,
            **({"eos_token_id": eos} if eos is not None else {}),
            "layer_norm_eps": layer_norm_eps,
            "tie_word_embeddings": False,
            "dtype": "bfloat16",
            "use_cache": True,
        }
        if bos_token_id is not None:
            cfg["bos_token_id"] = bos_token_id
        with open(os.path.join(hf_out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(
            f"[causal_mlp] exported {len(hf)} tensors -> {hf_out_dir} "
            f"(layers={actual_layers}, vocab={actual_vocab}, prefix={prefix!r})"
        )
        return hf_out_dir
