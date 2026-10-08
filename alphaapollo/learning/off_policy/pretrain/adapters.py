"""Thin adapters binding pretrain components into Megatron's fixed provider signatures.

Megatron 0.18's ``pretrain()`` calls three providers with fixed signatures (see
``main_pretrain``): ``datasets_provider(train_val_test_num_samples, vp_stage=None)``,
``model_provider(pre_process, post_process, vp_stage, config, pg_collection)``, and
``forward_step(data_iterator, model)``. These factories wrap a component INSTANCE into exactly
those signatures, so the ABCs in ``base/`` stay free of Megatron coupling — all Megatron-shape
knowledge lives here.
"""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base import (
    BaseDatasetProvider,
    BaseForwardStep,
    BaseModelBuilder,
)


def make_dataset_provider(provider: BaseDatasetProvider):
    """Wrap a dataset provider as Megatron's ``train_valid_test_datasets_provider`` callable."""

    def _datasets_provider(train_val_test_num_samples, vp_stage=None):
        return provider.build_datasets(train_val_test_num_samples, vp_stage=vp_stage)

    _datasets_provider.is_distributed = True  # Megatron expects this attr on the provider.
    return _datasets_provider


def make_model_provider(builder: BaseModelBuilder):
    """Wrap a model builder as Megatron's ``model_provider(pre_process, post_process, ...)``."""

    def _model_provider(
        pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
    ):
        from megatron.training import get_args

        args = get_args()
        return builder.build(
            args,
            pre_process,
            post_process,
            vp_stage=vp_stage,
            config=config,
            pg_collection=pg_collection,
        )

    return _model_provider


def make_forward_step(step: BaseForwardStep):
    """Return a Megatron-signature ``forward_step(data_iterator, model)`` bound to ``step``."""
    return step.forward_step
