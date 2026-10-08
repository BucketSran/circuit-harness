"""AlphaApollo pretrain — Megatron-LM-backed pretraining extension.

Lives under ``alphaapollo.learning.off_policy.pretrain``: an **off-policy**
training mode (static corpus -> forward -> LM loss -> update) that runs on
standalone Megatron-LM (``megatron-core``), NOT on verl. It is therefore a
**verl-free leaf** — this package imports nothing from ``verl`` and nothing
from the verl-coupled parts of ``alphaapollo`` (trainer / workflows), so it is
importable in the ``alphaapollo-pretrain`` runtime that has no verl on the path.
(The sibling ``algorithms/`` dir, by contrast, holds verl-SFT-engine off-policy
algorithms like DPO/KTO.) The user-facing entry point is
``alphaapollo.learning.off_policy.pretrain.main_pretrain``.

Design contract (see ``docs/design/pretrain-extension.md``):
  * Pretrain runs on standalone Megatron-LM (``megatron-core``), driven by
    Megatron's own argparse + ``PretrainConfigContainer`` — *not* Hydra.
  * The only interface to the RL/SFT side is a HuggingFace-format checkpoint
    on disk (see :mod:`alphaapollo.learning.off_policy.pretrain.checkpoint_bridge`).
  * Additive only — zero edits to ``third_party/verl`` or existing trainer files.

NOTE: written against megatron-core 0.18.0 (bridge-pinned; clone at
``$MEGATRON_LM_BRIDGE``, commit ``d0b3b7754``). Runs in the
``alphaapollo-pretrain`` conda env (torch 2.12+cu13 + TE 2.17 + megatron-bridge 0.5.0).
"""

__all__: list[str] = []
