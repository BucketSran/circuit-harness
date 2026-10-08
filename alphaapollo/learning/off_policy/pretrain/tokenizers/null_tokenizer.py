"""Null tokenizer component (synthesizes mock tokens; wraps Megatron's ``NullTokenizer``)."""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base import BaseTokenizer
from alphaapollo.learning.off_policy.pretrain.tokenizers import TOKENIZERS


@TOKENIZERS.register("NullTokenizer")
@TOKENIZERS.register("null")
class NullTokenizer(BaseTokenizer):
    """Configures Megatron to build a NullTokenizer (no corpus;
    uses ``--null-tokenizer-eod-id``)."""

    name = "null"

    def apply(self, args) -> None:
        args.tokenizer_type = "NullTokenizer"
        if getattr(args, "null_tokenizer_eod_id", None) is None:
            raise ValueError("NullTokenizer requires --null-tokenizer-eod-id <id>.")
