"""Dataset-provider components for pretrain (registry + selection)."""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base import BaseDatasetProvider, Registry

DATASETS = Registry("dataset")


def get_dataset_provider(name: str) -> BaseDatasetProvider:
    """Resolve ``--dataset`` to a ``BaseDatasetProvider`` (name or ``custom:<mod>:<Cls>``)."""
    return DATASETS.get_or_custom(name)


# Side-effect registration — MUST follow the DATASETS definition above.
from alphaapollo.learning.off_policy.pretrain.datasets import gpt_mmap, mock  # noqa: E402,F401

__all__ = ["DATASETS", "get_dataset_provider"]
