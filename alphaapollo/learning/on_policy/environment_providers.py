"""Learning-owned train/validation Environment provider composition."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from alphaapollo.common.environment.provider import (
    EnvironmentProvider,
    make_environment_provider,
)
from alphaapollo.common.environment.registry import integer, value_or_default


def make_environment_providers(
    config: Any,
) -> tuple[EnvironmentProvider, EnvironmentProvider]:
    """Size train and validation providers from the Learning configuration."""

    group_n = integer(config.env.rollout.n, name="env.rollout.n", minimum=1)
    validation_repeats = integer(
        value_or_default(OmegaConf.select(config, "actor_rollout_ref.rollout.val_kwargs.n"), 1),
        name="actor_rollout_ref.rollout.val_kwargs.n",
        minimum=1,
    )
    train = make_environment_provider(
        config.env,
        is_train=True,
        env_num=int(config.data.train_batch_size),
        group_n=group_n,
    )
    try:
        validation = make_environment_provider(
            config.env,
            is_train=False,
            env_num=int(config.data.val_batch_size) * validation_repeats,
            group_n=1,
        )
    except Exception:
        train.close()
        raise
    return train, validation


__all__ = ["make_environment_providers"]
