"""mcore -> HuggingFace checkpoint converters (registry + selection).

Built-in hand-written converters (the primary path for the standard dense families they cover;
audited for that family's nuances):
- ``qwen2``  : Qwen2.5 (tied lm_head, q/k/v bias) — the validated primary mcore->HF path.
- ``decoder``: generic Llama / Qwen2 / Mistral dense decoder-only.

And three generalization paths for architectures the built-ins don't cover:

- **A — ``bridge``** : automatic, via ``megatron.bridge.AutoBridge``. Any of bridge's ~50
  supported families (Qwen2/3, Llama, Mistral, Gemma, Deepseek, MoE, MLA, ...) converts with
  bridge's maintained mappings (GQA/MoE/MLA handled for you). Use for families beyond the dense
  built-ins, or when you prefer bridge's mappings. Needs a GPU (bridge builds the mcore model);
  pass the target HF config via ``hf_config=``.
- **B — ``mapping``** : declarative JSON rules (no Python). Any architecture you can describe as
  tensor-rename/split/per-layer/zero-fill rules, incl. GQA QKV via ``"split": "qkv_grouped"``. Best
  when bridge has no spec and you want a config-driven, no-code path.
- **D — ``custom:<module>:<Cls>``** : a user converter subclassing
  :class:`BaseCheckpointConverter` and implementing ``export(mcore_ckpt_dir, hf_out_dir, **opts)``.
  Best for novel architectures needing custom logic. Reusable helpers for D authors:
  ``_split_qkv_grouped``, ``_grouped_reinterleave``, ``_load_mcore_state_dict``,
  ``_latest_iter_dir``
  (from :mod:`...converters.qwen2`).

Resolve any of the above with ``get_converter(name)`` or the CLI ``--converter <name|custom:...>``.
The HF directory on disk is the only crossing from pretrain to the RL/SFT side.
"""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base import BaseCheckpointConverter, Registry

CONVERTERS = Registry("converter")


def get_converter(name: str) -> BaseCheckpointConverter:
    """Resolve a converter by name (or ``custom:<mod>:<Cls>``), e.g. ``get_converter('bridge')``."""
    return CONVERTERS.get_or_custom(name)


# Side-effect registration — MUST follow the CONVERTERS definition above.
from alphaapollo.learning.off_policy.pretrain.converters import (  # noqa: E402,F401
    bridge,
    causal_mlp,
    decoder,
    mapping,
    qwen2,
)

__all__ = ["CONVERTERS", "get_converter"]
