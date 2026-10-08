"""HuggingFace tokenizer component (wraps Megatron's ``HuggingFaceTokenizer``)."""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base import BaseTokenizer
from alphaapollo.learning.off_policy.pretrain.tokenizers import TOKENIZERS


@TOKENIZERS.register("HuggingFaceTokenizer")
@TOKENIZERS.register("huggingface")
class HuggingFaceTokenizer(BaseTokenizer):
    """Configures Megatron to build a HuggingFace tokenizer from ``--tokenizer-model``."""

    name = "huggingface"

    def apply(self, args) -> None:
        args.tokenizer_type = "HuggingFaceTokenizer"
        if getattr(args, "tokenizer_model", None) in (None, "None"):
            raise ValueError("HuggingFaceTokenizer requires --tokenizer-model <HF tokenizer dir>.")
