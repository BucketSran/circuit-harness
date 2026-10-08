"""AlphaApollo pretrain entry point (Megatron-LM, verl-free) — modular component version.

Drives Megatron-LM's config-container pretrain path against mcore 0.18 (the version
``megatron-bridge`` 0.5.0 pins), assembling the pipeline from pluggable, registry-backed
components:

    --dataset   <name|custom:mod:Cls>   datasets/     -> BaseDatasetProvider
    --arch      <name|custom:mod:Cls>   archs/        -> BaseModelBuilder
    --forward   <name|custom:mod:Cls>   forward/      -> BaseForwardStep
    --converter <name|custom:mod:Cls>   converters/   -> BaseCheckpointConverter (post-train)

The component instances are wrapped by ``adapters.py`` into Megatron's fixed provider
signatures, then passed to the 0.18 ``pretrain(cfg, datasets_provider, model_provider,
ModelType, forward_step)``. Megatron uses its OWN argparse (NOT Hydra) — launch with torchrun.

Tokenizer note: there is intentionally **no** ``--tokenizer`` selector. Megatron builds the
tokenizer *during* ``parse_and_validate_args`` (``megatron.training.global_vars._build_tokenizer``),
so the tokenizer type is fixed at parse time by Megatron's own ``--tokenizer-type`` flag and
cannot be changed afterward. The ``BaseTokenizer`` component (``tokenizers/``) is therefore
selected BY ``args.tokenizer_type`` and used only to *validate* its required args
(``--tokenizer-model`` for HF, ``--null-tokenizer-eod-id`` for NullTokenizer).

Backward-compat: if ``--dataset`` is omitted, it is inferred from the legacy Megatron args
(``--mock-data`` -> mock, else ``--data-path`` -> gpt_mmap), so existing scripts keep working.

Runs in the unified env or ``alphaapollo-pretrain`` (torch 2.12+cu13 + TE 2.17 +
megatron-core 0.18 clone at ``$MEGATRON_LM_BRIDGE``). Heavy imports are inside
``main_entry`` so the
module is import-safe without a GPU.
"""

from __future__ import annotations

import time

# Capture program start time BEFORE any heavy import (Megatron's timing report).
_PROGRAM_START_TIME = time.time()


def _extra_args(parser):
    """AlphaApollo-specific Megatron args — the pluggable component selectors."""
    group = parser.add_argument_group(title="alphaapollo")
    group.add_argument(
        "--arch",
        type=str,
        default="gpt",
        help="Model architecture: a registry name ('gpt', 'example_custom') or "
        "'custom:<module>:<Class>' (a BaseModelBuilder subclass). See archs/.",
    )
    group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset provider: 'gpt_mmap' (real corpus; --data-path) or 'mock' (--mock-data), "
        "or 'custom:<module>:<Class>'. Default: 'mock' if --mock-data else 'gpt_mmap'.",
    )
    group.add_argument(
        "--forward",
        type=str,
        default="causal_lm",
        help="Forward-step component (default 'causal_lm'), or 'custom:<module>:<Class>'.",
    )
    group.add_argument(
        "--converter",
        type=str,
        default="qwen2",
        help="mcore->HF checkpoint converter used post-training (default 'qwen2'), or "
        "'custom:<module>:<Class>'. Resolved lazily by the export tooling, not at train time.",
    )
    group.add_argument(
        "--diff-mask-id",
        type=int,
        default=None,
        help="[masked_diffusion only] token id used as [MASK] for corruption. "
        "Required when --forward masked_diffusion.",
    )
    group.add_argument(
        "--diff-t-min",
        type=float,
        default=1e-3,
        help="[masked_diffusion] lower bound of the diffusion timestep t~U(t_min, t_max).",
    )
    group.add_argument(
        "--diff-t-max",
        type=float,
        default=0.999,
        help="[masked_diffusion] upper bound of the timestep (keep <1 to bound the 1/t weight).",
    )
    group.add_argument(
        "--load-hf",
        type=str,
        default=None,
        help="HuggingFace model dir to load weights from (continued-pretrain; arch args MUST "
        "match this HF model). Currently blocked by a bridge<->mcore0.18 API drift.",
    )
    return parser


def main_entry() -> None:
    # Heavy / Megatron imports are deferred so `import ...main_pretrain` works without a GPU.
    from megatron.core.enums import ModelType
    from megatron.training import inprocess_restart, pretrain, set_startup_timestamps
    from megatron.training.argument_utils import pretrain_cfg_container_from_args
    from megatron.training.arguments import parse_and_validate_args

    from alphaapollo.learning.off_policy.pretrain.adapters import (
        make_dataset_provider,
        make_forward_step,
        make_model_provider,
    )
    from alphaapollo.learning.off_policy.pretrain.archs import get_model_builder
    from alphaapollo.learning.off_policy.pretrain.datasets import get_dataset_provider
    from alphaapollo.learning.off_policy.pretrain.forward import get_forward_step
    from alphaapollo.learning.off_policy.pretrain.tokenizers import TOKENIZERS, get_tokenizer

    _main_entry_time = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_main_entry_time)

    # Megatron parses its own argparse namespace AND builds the tokenizer during this call
    # (global_vars._build_tokenizer), so --tokenizer-type must already be correct on the CLI.
    args = parse_and_validate_args(
        extra_args_provider=_extra_args,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )

    # Validate the chosen tokenizer's required args via its BaseTokenizer component, selected BY
    # Megatron's --tokenizer-type. Tokenizer types not registered here are left to Megatron.
    if getattr(args, "tokenizer_type", None) in TOKENIZERS:
        get_tokenizer(args.tokenizer_type).apply(args)

    full_config = pretrain_cfg_container_from_args(args)

    # --- resolve the pluggable components from their registries (with legacy-arg fallbacks) ---
    dataset_name = args.dataset
    if dataset_name is None:  # --mock-data shortcut stays supported
        dataset_name = "mock" if getattr(args, "mock_data", False) else "gpt_mmap"
    dataset_provider = get_dataset_provider(dataset_name)
    model_builder = get_model_builder(getattr(args, "arch", "gpt"))
    forward = get_forward_step(args.forward)
    # converter (args.converter) is resolved lazily by export tooling (converters/get_converter).

    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    # Bind the components into Megatron's fixed-signature providers.
    datasets_provider = make_dataset_provider(dataset_provider)
    model_provider = make_model_provider(model_builder)
    forward_step = make_forward_step(forward)

    pretrain(
        full_config,
        datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
    )


if __name__ == "__main__":
    main_entry()
