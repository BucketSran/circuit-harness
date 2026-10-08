"""Mock dataset provider — ``MockGPTDataset`` (synthesizes tokens; no corpus needed).

The bring-up / smoke path (``--dataset mock``). ``MockGPTDataset`` generates random tokens,
so no corpus or mmap files are required. Body moved verbatim from the former ``data.py``
(MockGPTDataset branch).
"""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base import BaseDatasetProvider
from alphaapollo.learning.off_policy.pretrain.datasets import DATASETS
from alphaapollo.learning.off_policy.pretrain.datasets._config import gpt_dataset_config_from_args


@DATASETS.register("mock")
class MockProvider(BaseDatasetProvider):
    """Mock provider: builds train/valid/test ``MockGPTDataset`` (random tokens)."""

    name = "mock"

    def build_datasets(self, train_val_test_num_samples, vp_stage=None):
        from megatron.core.datasets.blended_megatron_dataset_builder import (
            BlendedMegatronDatasetBuilder,
        )
        from megatron.core.datasets.gpt_dataset import MockGPTDataset
        from megatron.training import get_args, print_rank_0

        args = get_args()
        config = gpt_dataset_config_from_args(args)

        print_rank_0("> building MOCK GPT datasets ...")
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            MockGPTDataset, train_val_test_num_samples, lambda: True, config
        ).build()
        print_rank_0("> finished creating MOCK GPT datasets ...")
        return train_ds, valid_ds, test_ds
