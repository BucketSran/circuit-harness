"""Backward-compat shim: re-exports the Qwen2 mcore -> HuggingFace converter.

The converter now lives at ``converters/qwen2.py`` (``Qwen2Converter``, registered as
``"qwen2"``). This module keeps the historical ``export_mcore_to_hf`` callable so existing
docs/scripts imports (e.g. ``docs/runbooks/pretrain-usage.md``, ``examples/pretrain/*.sh``)
keep working. New code should use ``converters.get_converter("qwen2").export(...)``.
"""

from __future__ import annotations


def export_mcore_to_hf(mcore_ckpt_dir, hf_out_dir, **opts):
    """Convert a Megatron-LM Qwen-arch dist-checkpoint to HuggingFace Qwen2.

    Delegates to the registered ``"qwen2"`` ``BaseCheckpointConverter``. The import is lazy
    so importing this shim does not pull torch/megatron.
    """
    from alphaapollo.learning.off_policy.pretrain.converters import get_converter

    return get_converter("qwen2").export(mcore_ckpt_dir, hf_out_dir, **opts)


__all__ = ["export_mcore_to_hf"]
