"""Abstract base for pluggable pretrain tokenizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTokenizer(ABC):
    """A pluggable tokenizer component.

    Megatron builds its tokenizer internally from ``args`` (``args.tokenizer_type`` +
    ``args.tokenizer_model`` etc.) via ``megatron.core.tokenizers.utils.build_tokenizer``.
    A ``BaseTokenizer`` does NOT rebuild that machinery; it *configures* ``args`` so that
    Megatron's own ``build_tokenizer(args)`` produces this tokenizer. ``apply`` must run
    before the dataset provider (which calls ``build_tokenizer``).

    Subclasses set the class attribute ``name`` (the registry key).
    """

    name: str

    @abstractmethod
    def apply(self, args: Any) -> None:
        """Mutate ``args`` in place so Megatron builds THIS tokenizer.

        Typically sets ``args.tokenizer_type`` (+ ``args.tokenizer_model`` for HF,
        ``args.null_tokenizer_eod_id`` for NullTokenizer).
        """
        ...

    def build(self, args: Any) -> Any:
        """Apply, then return Megatron's built tokenizer (convenience default)."""
        self.apply(args)
        from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer

        return build_tokenizer(args)
