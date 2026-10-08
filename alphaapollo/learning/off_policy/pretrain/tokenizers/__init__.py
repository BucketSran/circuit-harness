"""Tokenizer components for pretrain (registry + selection)."""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base import BaseTokenizer, Registry

TOKENIZERS = Registry("tokenizer")


def get_tokenizer(name: str) -> BaseTokenizer:
    """Resolve ``--tokenizer`` to a ``BaseTokenizer`` (name or ``custom:<mod>:<Cls>``)."""
    return TOKENIZERS.get_or_custom(name)


# Side-effect registration — MUST follow the TOKENIZERS definition above.
from alphaapollo.learning.off_policy.pretrain.tokenizers import (  # noqa: E402,F401
    hf_tokenizer,
    null_tokenizer,
)

__all__ = ["TOKENIZERS", "get_tokenizer"]
