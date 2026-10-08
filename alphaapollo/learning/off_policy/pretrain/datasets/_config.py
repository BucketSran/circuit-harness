"""Shared helper: build a Megatron ``GPTDatasetConfig`` from args.

Used by every GPT-style dataset provider (real mmap corpus and mock). The tokenizer is
built here via Megatron's ``build_tokenizer(args)`` — so the active ``BaseTokenizer.apply``
must have run first (it configures ``args`` so the right tokenizer is built).
"""

from __future__ import annotations


def gpt_dataset_config_from_args(args):
    """Build a ``GPTDatasetConfig`` from the live Megatron ``args`` (mcore 0.18 kwargs)."""
    from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
    from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
    from megatron.training.utils import get_blend_and_blend_per_split

    tokenizer = build_tokenizer(args)
    blend, blend_per_split = get_blend_and_blend_per_split(args)
    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )
