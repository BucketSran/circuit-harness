"""GPT mmap dataset provider — real corpus via Megatron ``GPTDataset``.

Points ``--data-path`` at mmap ``.bin``/``.idx`` files (preprocess text with Megatron's
``tools/preprocess_data.py``). The GPTDataset-building body is moved verbatim from the
former ``data.py`` (GPTDataset branch).
"""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base import BaseDatasetProvider
from alphaapollo.learning.off_policy.pretrain.datasets import DATASETS
from alphaapollo.learning.off_policy.pretrain.datasets._config import gpt_dataset_config_from_args


@DATASETS.register("gpt_mmap")
class GPTMmapProvider(BaseDatasetProvider):
    """Real-corpus provider: builds train/valid/test ``GPTDataset`` from mmap files."""

    name = "gpt_mmap"

    def build_datasets(self, train_val_test_num_samples, vp_stage=None):
        from megatron.core.datasets.blended_megatron_dataset_builder import (
            BlendedMegatronDatasetBuilder,
        )
        from megatron.core.datasets.gpt_dataset import GPTDataset
        from megatron.training import get_args, print_rank_0

        args = get_args()
        config = gpt_dataset_config_from_args(args)

        print_rank_0("> building GPT datasets ...")
        # NOTE: ``is_dataset_built`` is simplified to "build on every rank" for the
        # single-parallel-degree case. pretrain_gpt.py uses a rank-aware partial
        # (``is_dataset_built_on_rank``) — swap that in for TP/PP.
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset, train_val_test_num_samples, lambda: True, config
        ).build()
        print_rank_0("> finished creating GPT datasets ...")
        return train_ds, valid_ds, test_ds
