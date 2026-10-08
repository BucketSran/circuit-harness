"""Pluggable architecture framework for multi-GPU from-scratch pretrain.

Select the model architecture with ``--arch`` (an AlphaApollo arg added in
``main_pretrain``):

* ``--arch gpt`` (default): the standard args-driven dense GPT (Qwen/Llama-style) —
  :class:`alphaapollo.learning.off_policy.pretrain.archs.gpt.GPTArch`.
* ``--arch example_custom``: the runnable skeleton
  :class:`alphaapollo.learning.off_policy.pretrain.archs.example_custom.ExampleCustomArch`
  (dense-GPT placeholder).
* ``--arch myarch``: pipeline-demo custom decoder
  :class:`alphaapollo.learning.off_policy.pretrain.archs.myarch.MyArch` (decoder-only; topology
  via launch args; flows through the generic ``decoder`` converter to HF/RL).
* ``--arch custom:<module>:<Class>``: import ``<module>.<Class>``, instantiate it, and use it
  as the model builder. The class MUST subclass
  :class:`alphaapollo.learning.off_policy.pretrain.base.BaseModelBuilder`. See
  ``example_custom.py``.

This covers the PRETRAIN side only. Converting a custom arch's mcore checkpoint to
HuggingFace (for the RL env) is a separate, per-arch step — register a
``BaseCheckpointConverter`` under ``converters/`` (the default ``"qwen2"`` is Qwen2-specific).
"""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base import BaseModelBuilder, Registry

ARCHS = Registry("architecture")


def get_model_builder(name: str) -> BaseModelBuilder:
    """Resolve ``--arch`` to a ``BaseModelBuilder`` instance (name or ``custom:<mod>:<Cls>``)."""
    if name is None:
        name = "gpt"
    return ARCHS.get_or_custom(name)


# Side-effect registration — MUST follow the ARCHS definition above.
from alphaapollo.learning.off_policy.pretrain.archs import (  # noqa: E402,F401
    bert,
    causal_mlp,
    example_custom,
    gpt,
    myarch,
)

# Backward-compat alias for the old module-global name (now a ``Registry``, not a dict).
ARCH_REGISTRY = ARCHS

__all__ = ["ARCHS", "ARCH_REGISTRY", "get_model_builder"]
