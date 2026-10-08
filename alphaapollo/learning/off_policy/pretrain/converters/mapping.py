"""Declarative, architecture-agnostic mcore -> HuggingFace converter.

Works for **any** architecture: the caller supplies a *mapping* (which mcore checkpoint tensor
maps to which HuggingFace state-dict key, with optional split / per-layer expansion / zero-fill)
and the target ``config.json``. This module does the mechanics (load the Megatron torch_dist
checkpoint, apply the rules, write ``model.safetensors`` + ``config.json``). No architecture
knowledge is hard-coded here — that lives entirely in the mapping the user provides.

Mapping format (JSON object loaded from a path or passed as a dict):
```
{
  "num_layers": 28,            # int, or omitted to infer from a layered tensor's dim-0
  "rules": [
    {"from": "<mcore_key>",                       "to": "<hf_key>"},                  # rename
    {"from": "<mcore_key>",                "to": "<hf_key with {i}>", "layers": true},  # per-layer
    {"from": "<mcore_key>",                       "to": ["<hf1>", "<hf2>"],            # split
                   "split_dim": 0, "split_sizes": [n1, n2]},
    {"from": "<mcore_key>",          "to": ["<hf1 {i}>", ...], "layers": true,  # per-layer split
                   "split_dim": 0, "split_sizes": [...]},
    {"from": "<maybe-absent mcore_key>",   "to": "...", "optional": true},  # skip if absent
    {   # "from" omitted                                    "to": "<hf_key {i}>", "layers": true,
    "fill_zeros": [d]},  # zero-fill
    {"from": "...linear_qkv.weight",           "to": ["...q_proj.weight {i}",  # GQA QKV (grouped)
                  "...k_proj.weight {i}", "...v_proj.weight {i}"], "layers": true,
                  "split": "qkv_grouped", "num_query_groups": 2, "q_dim": 896,
                  "kv_dim": 128}
  ]
}
```
Rule fields:
- ``from``    : mcore checkpoint key. Omit (or set ``optional``) when only zero-filling.
- ``to``      : a single HF key, or a list of HF keys (=> split). ``{i}`` = layer index.
- ``layers``  : true => ``from`` is stacked ``[L, ...]`` over layers; emit one (or N split) HF
               key(s) per layer ``i``, slicing ``from[i]``.
- ``split_dim``/``split_sizes`` : when ``to`` is a list, flat-split the (per-layer) tensor along
               ``split_dim`` into ``split_sizes`` chunks (torch.split). Use for gate|up etc.
- ``split``   : ``"qkv_grouped"`` => IGNORE split_dim/split_sizes and regroup the fused QKV into
               q/k/v by query group (GQA-correct; needs ``num_query_groups``,
               ``q_dim``, ``kv_dim``).
               Use for ``self_attention.linear_qkv.{weight,bias}``; a flat split_sizes is WRONG for
               >1 kv group (silently permutes heads).
- ``optional``: if ``from`` is absent in the checkpoint, skip the rule instead of erroring.
- ``fill_zeros``: shape list; if ``from`` is absent, emit a zero tensor of this shape instead.

Select via the registry (``get_converter('mapping')``) or run as a tool:
``python -m alphaapollo.learning.off_policy.pretrain.converters.mapping --mcore <dir> --out <dir>
   --mapping map.json --config cfg.json [--num-layers N]``
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from alphaapollo.learning.off_policy.pretrain.base import BaseCheckpointConverter
from alphaapollo.learning.off_policy.pretrain.converters import CONVERTERS, get_converter
from alphaapollo.learning.off_policy.pretrain.converters.qwen2 import (
    _latest_iter_dir,
    _load_mcore_state_dict,
    _split_qkv_grouped,
)


def _load_json_or_dict(obj: Any) -> Any:
    """Accept a path (str/Path) -> parsed JSON, or pass a dict/list through."""
    if isinstance(obj, str):
        with open(obj) as f:
            return json.load(f)
    return obj


def _infer_num_layers(sd: dict, rules: list[dict]) -> int:
    """Infer layer count from the first layered rule's stacked tensor (dim-0)."""
    for rule in rules:
        if rule.get("layers") and rule.get("from") in sd:
            return int(sd[rule["from"]].shape[0])
    raise ValueError("could not infer num_layers; set mapping 'num_layers' or --num-layers.")


def _chunks(src: torch.Tensor, rule: dict) -> list[torch.Tensor]:
    """Split a source tensor into the rule's target chunks.

    Two modes:
    - ``"split": "qkv_grouped"`` (GQA-correct): regroup mcore's per-query-group fused QKV into HF
      q/k/v via :func:`_split_qkv_grouped` (needs ``num_query_groups``, ``q_dim``, ``kv_dim``).
      Returns 3 chunks. Correct for any MHA/MQA/GQA; a flat ``split_sizes`` here would silently
      permute heads for >1 kv group.
    - default (flat): ``torch.split(src, split_sizes, dim=split_dim)`` (gate|up, etc.).
    """
    if rule.get("split") == "qkv_grouped":
        return list(
            _split_qkv_grouped(src, rule["q_dim"], rule["kv_dim"], rule["num_query_groups"])
        )
    return list(torch.split(src, rule["split_sizes"], dim=rule["split_dim"]))


def _apply_rule(hf: dict, sd: dict, rule: dict, num_layers: int) -> None:
    if "to" not in rule:
        return  # comment / doc entry (e.g. {"_note": ...}) — not a real rule, skip
    frm = rule.get("from")
    to = rule["to"]
    layered = bool(rule.get("layers", False))
    is_split = isinstance(to, list)
    optional = bool(rule.get("optional", False))
    fill = rule.get("fill_zeros")  # list[int] shape, or None

    def source(i=None):
        """Return the tensor for this rule (layer-sliced if layered), or None to skip."""
        if frm is None or frm not in sd:
            if fill is not None:
                return torch.zeros(tuple(fill), dtype=torch.float32)
            if optional:
                return None
            raise KeyError(
                f"mapping rule expects mcore key {frm!r} but it is absent; "
                f"set 'optional' or 'fill_zeros'."
            )
        t = sd[frm]
        return t[i] if (layered and i is not None) else t

    def emit(keys, tensors):
        for k, t in zip(keys, tensors, strict=True):
            hf[k] = t

    if not layered:
        src = source()
        if src is None:
            return
        if is_split:
            emit(to, _chunks(src, rule))
        else:
            hf[to] = src
        return

    for i in range(num_layers):
        src = source(i)
        if src is None:
            continue
        if is_split:
            keys = [k.replace("{i}", str(i)) for k in to]
            emit(keys, _chunks(src, rule))
        else:
            hf[to.replace("{i}", str(i))] = src


@CONVERTERS.register("mapping")
class MappingConverter(BaseCheckpointConverter):
    """Convert any Megatron checkpoint to HuggingFace via a user-supplied mapping + config."""

    name = "mapping"

    def export(
        self,
        mcore_ckpt_dir: str,
        hf_out_dir: str,
        *,
        mapping: Any,
        config: Any,
        num_layers: int | None = None,
    ) -> str:
        """Apply ``mapping`` to the mcore checkpoint; write ``hf_out_dir`` (config + safetensors).

        Args:
            mcore_ckpt_dir: Megatron torch_dist checkpoint dir.
            hf_out_dir: output HuggingFace model dir.
            mapping: path to a mapping JSON file, or the mapping dict.
            config:  path to a HuggingFace ``config.json`` file, or the config dict.
            num_layers: override layer count (else mapping['num_layers'] or inferred).
        """
        from safetensors.torch import save_file

        mapping = _load_json_or_dict(mapping)
        config = _load_json_or_dict(config)
        rules = mapping["rules"]
        sd = _load_mcore_state_dict(_latest_iter_dir(mcore_ckpt_dir))

        nl = num_layers or mapping.get("num_layers") or _infer_num_layers(sd, rules)

        hf: dict[str, torch.Tensor] = {}
        for rule in rules:
            _apply_rule(hf, sd, rule, nl)
        if not hf:
            raise ValueError(
                f"the mapping produced 0 tensors from {mcore_ckpt_dir} "
                f"({len(rules)} rules, {len(sd)} ckpt tensors): every rule must have missed. "
                "Refusing to write an empty model."
            )
        # Save bf16 (the user's config.json is expected to declare torch_dtype=bfloat16; lossless
        # if the source trained bf16) and halve disk vs fp32.
        hf = {k: v.to(torch.bfloat16).contiguous() for k, v in hf.items()}

        # soft check: warn if any model tensor in the ckpt was not referenced by a rule
        referenced = {r.get("from") for r in rules if r.get("from")}
        unmapped = sorted(set(sd) - referenced)
        if unmapped:
            print(
                f"[mapping] WARNING: {len(unmapped)} ckpt tensor(s) not covered by any rule: "
                f"{unmapped[:6]}{' ...' if len(unmapped) > 6 else ''}"
            )

        os.makedirs(hf_out_dir, exist_ok=True)
        save_file(hf, os.path.join(hf_out_dir, "model.safetensors"), metadata={"format": "pt"})
        with open(os.path.join(hf_out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"[mapping] exported {len(hf)} tensors -> {hf_out_dir} (num_layers={nl})")
        return hf_out_dir


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Convert any mcore checkpoint to HuggingFace via a mapping."
    )
    ap.add_argument("--mcore", required=True, help="Megatron torch_dist checkpoint dir")
    ap.add_argument("--out", required=True, help="output HuggingFace model dir")
    ap.add_argument("--mapping", required=True, help="mapping JSON path (or 'name' registry key)")
    ap.add_argument("--config", required=True, help="HuggingFace config.json path")
    ap.add_argument("--num-layers", type=int, default=None)
    args = ap.parse_args()
    get_converter("mapping").export(
        args.mcore,
        args.out,
        mapping=args.mapping,
        config=args.config,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    _main()
