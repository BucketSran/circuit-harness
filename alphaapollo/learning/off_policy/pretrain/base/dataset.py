"""Abstract base for pluggable pretrain dataset providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDatasetProvider(ABC):
    """A pluggable train/valid/test dataset provider.

    ``build_datasets`` returns the ``(train, valid, test)`` Megatron datasets for the
    requested per-split sample counts — exactly what Megatron's
    ``train_valid_test_datasets_provider`` contract expects. Subclasses set ``name``.
    """

    name: str

    @abstractmethod
    def build_datasets(
        self, train_val_test_num_samples: list[int], vp_stage: Any = None
    ) -> tuple[Any, Any, Any]:
        """Return ``(train, valid, test)`` Megatron datasets."""
        ...
