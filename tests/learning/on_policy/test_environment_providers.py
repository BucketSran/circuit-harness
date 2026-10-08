"""Tests for Learning-owned train/validation Environment composition."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from alphaapollo.learning.on_policy import environment_providers


def test_learning_sizes_train_and_validation_environment_providers(monkeypatch) -> None:
    calls = []

    def make_environment_provider(env_config, **kwargs):
        calls.append((env_config, kwargs))
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        environment_providers,
        "make_environment_provider",
        make_environment_provider,
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 3}}},
            "data": {"train_batch_size": 4, "val_batch_size": 2},
            "env": {"env_name": "search", "rollout": {"n": 5}},
        }
    )

    train, validation = environment_providers.make_environment_providers(config)

    assert train.env_num == 4
    assert train.group_n == 5
    assert train.is_train is True
    assert validation.env_num == 6
    assert validation.group_n == 1
    assert validation.is_train is False
    assert calls[0][0] is config.env
    assert calls[1][0] is config.env


def test_learning_closes_train_provider_when_validation_creation_fails(monkeypatch) -> None:
    train = SimpleNamespace(closed=False)
    train.close = lambda: setattr(train, "closed", True)

    def make_environment_provider(_env_config, *, is_train, **_kwargs):
        if is_train:
            return train
        raise RuntimeError("validation startup failed")

    monkeypatch.setattr(
        environment_providers,
        "make_environment_provider",
        make_environment_provider,
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 1}}},
            "data": {"train_batch_size": 1, "val_batch_size": 1},
            "env": {"env_name": "search", "rollout": {"n": 1}},
        }
    )

    with pytest.raises(RuntimeError, match="validation startup failed"):
        environment_providers.make_environment_providers(config)

    assert train.closed is True
