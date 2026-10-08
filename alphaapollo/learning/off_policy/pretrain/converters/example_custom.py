"""Example custom mcore -> HuggingFace converter skeleton (converter path D).

Copy this file, implement :meth:`export` for YOUR architecture, then select it with
``--converter custom:<your_module>:<YourConverter>`` — or register it under a name with
``@CONVERTERS.register("your_name")`` and use ``--converter your_name``.

Compose from the reusable helpers in :mod:`...converters.qwen2`:
- :func:`_load_mcore_state_dict` — load a native mcore ``torch_dist`` ckpt into a plain
  ``{key: tensor}`` dict (no model build; works on CPU).
- :func:`_latest_iter_dir` — resolve the parent ckpt dir to its ``iter_XXXXXXX`` dir.
- :func:`_split_qkv_grouped` — regroup mcore's per-query-group fused QKV into HF q/k/v
  (GQA-correct; also takes the bias).

See ``qwen2.py`` / ``decoder.py`` for full worked examples, and ``mapping.py`` for the no-Python
declarative alternative (path B) which may suffice for your arch without writing any tensor code.
"""

from __future__ import annotations

import torch

from alphaapollo.learning.off_policy.pretrain.base import BaseCheckpointConverter
from alphaapollo.learning.off_policy.pretrain.converters.qwen2 import (
    _latest_iter_dir,
    _load_mcore_state_dict,
)


class ExampleCustomConverter(BaseCheckpointConverter):
    """Template: load the mcore ckpt to a dict, map tensors to HF keys, write the HF dir.

    Replace the body of :meth:`export` with your architecture's mcore-key -> HF-key mapping.
    """

    name = "example_custom"

    def export(self, mcore_ckpt_dir: str, hf_out_dir: str, **opts) -> str:
        # 1. Load the native mcore torch_dist checkpoint into a plain {key: tensor} dict.
        sd = _load_mcore_state_dict(_latest_iter_dir(mcore_ckpt_dir))  # noqa: F841 — template
        hf: dict[str, torch.Tensor] = {}  # noqa: F841 — filled by YOUR mapping below

        # 2. TODO: map mcore tensors -> HF state-dict keys for YOUR architecture, e.g.:
        #    hf["model.embed_tokens.weight"] = sd["embedding.word_embeddings.weight"]
        #    qkv = sd["decoder.layers.self_attention.linear_qkv.weight"]   # [L, q+2kv, hidden]
        #    for i in range(num_layers):
        #        q, k, v = _split_qkv_grouped(qkv[i], q_dim, kv_dim,
        #                                    num_query_groups)  # GQA-correct
        #        hf[f"model.layers.{i}.self_attn.q_proj.weight"] = q
        #        ...
        #
        # 3. Write the HF dir (bf16 to match config torch_dtype=bfloat16); copy
        #    the tokenizer separately:
        #    import json, os
        #    from safetensors.torch import save_file
        #    os.makedirs(hf_out_dir, exist_ok=True)
        #    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in hf.items()},
        #              os.path.join(hf_out_dir, "model.safetensors"), metadata={"format": "pt"})
        #    json.dump(your_hf_config, open(os.path.join(hf_out_dir, "config.json"), "w"), indent=2)
        #    return hf_out_dir
        raise NotImplementedError(
            "ExampleCustomConverter is a skeleton — implement the mcore->HF mapping above (step 2) "
            "and the save (step 3), then select with --converter custom:<module>:<YourConverter>."
        )
