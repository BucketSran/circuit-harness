"""Forward-step components for pretrain (registry + selection)."""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base import BaseForwardStep, Registry

FORWARD = Registry("forward step")


def get_forward_step(name: str) -> BaseForwardStep:
    """Resolve ``--forward`` to a ``BaseForwardStep`` instance (name or ``custom:<mod>:<Cls>``)."""
    return FORWARD.get_or_custom(name)


# Side-effect registration — MUST follow the FORWARD definition above.
from alphaapollo.learning.off_policy.pretrain.forward import (  # noqa: E402,F401
    causal_lm,
    masked_diffusion,
)

__all__ = ["FORWARD", "get_forward_step"]
