"""Registry completeness for the pretrain package's component families.

Pins the PACKAGED surface in both directions: a component that silently stops
registering (broken import, renamed decorator) fails here, and adding/removing an
entry becomes a deliberate edit to the expected set. All registry modules import
with torch only (megatron/transformers are function-level inside the members), so
this runs anywhere the converter tests do.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from alphaapollo.learning.off_policy.pretrain.archs import ARCHS
from alphaapollo.learning.off_policy.pretrain.converters import CONVERTERS
from alphaapollo.learning.off_policy.pretrain.datasets import DATASETS
from alphaapollo.learning.off_policy.pretrain.forward import FORWARD
from alphaapollo.learning.off_policy.pretrain.tokenizers import TOKENIZERS


def test_arch_registry_is_the_packaged_set():
    assert sorted(ARCHS.names()) == ["bert", "causal_mlp", "example_custom", "gpt", "myarch"]


def test_forward_registry_is_the_packaged_set():
    assert sorted(FORWARD.names()) == ["causal_lm", "masked_diffusion"]


def test_dataset_registry_is_the_packaged_set():
    assert sorted(DATASETS.names()) == ["gpt_mmap", "mock"]


def test_tokenizer_registry_is_the_packaged_set():
    assert sorted(TOKENIZERS.names()) == [
        "HuggingFaceTokenizer",
        "NullTokenizer",
        "huggingface",
        "null",
    ]


def test_converter_registry_is_the_packaged_set():
    assert sorted(CONVERTERS.names()) == [
        "bridge",
        "causal_mlp",
        "decoder",
        "llama",
        "mapping",
        "qwen2",
    ]
