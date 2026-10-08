"""Contracts for per-episode Environment providers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alphaapollo.common.environment.provider import EnvironmentProvider


class _Pool:
    batch_size = 2

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_provider_vends_distinct_environments_by_explicit_slot() -> None:
    pool = _Pool()
    provider = EnvironmentProvider(
        pool,
        lambda slot, payload: SimpleNamespace(slot=slot, payload=dict(payload)),
    )

    first = provider.environment_factory(
        SimpleNamespace(environment_slot=0, task_payload={"case": 1})
    )
    second = provider.environment_factory(
        SimpleNamespace(environment_slot=1, task_payload={"case": 2})
    )

    assert first is not second
    assert (first.slot, first.payload) == (0, {"case": 1})
    assert (second.slot, second.payload) == (1, {"case": 2})


def test_provider_prepares_one_environment_seed_stream_per_batch() -> None:
    reset_count = 0

    def prepare(active_size: int):
        nonlocal reset_count
        reset_count += 1
        return [100 + reset_count] * active_size

    provider = EnvironmentProvider(_Pool(), lambda slot, payload: object(), prepare_batch=prepare)

    assert provider.episode_seeds([{}, {}]) == (101, 101)
    assert provider.episode_seeds([{"task_seed": 7}, {"seed": 8}]) == (7, 8)

    with pytest.raises(ValueError, match="active_size"):
        provider.episode_seeds([{}, {}, {}])


def test_provider_owns_pool_lifecycle() -> None:
    pool = _Pool()
    provider = EnvironmentProvider(pool, lambda slot, payload: object())

    provider.close()
    provider.close()

    assert pool.closed is True
    with pytest.raises(RuntimeError, match="closed provider"):
        provider.environment_factory(SimpleNamespace(environment_slot=0, task_payload={}))


@pytest.mark.parametrize("slot", [-1, 2])
def test_provider_rejects_slots_outside_the_pool(slot: int) -> None:
    provider = EnvironmentProvider(_Pool(), lambda index, payload: object())

    with pytest.raises(ValueError, match="environment_slot"):
        provider.environment_factory(SimpleNamespace(environment_slot=slot, task_payload={}))


def test_removed_pool_is_refused_before_acquiring_resources() -> None:
    from alphaapollo.common.environment.provider import make_environment_provider

    with pytest.raises(ValueError, match="not registered"):
        make_environment_provider(
            SimpleNamespace(env_name="sokoban"), is_train=True, env_num=1, group_n=1
        )


def test_registered_application_pool_receives_the_training_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import alphaapollo.common.environment.provider as module

    monkeypatch.setattr(module, "_PROVIDER_FACTORIES", {})
    calls = []
    expected = EnvironmentProvider(_Pool(), lambda slot, payload: object())

    def build(config, **kwargs):
        calls.append((config.env_name, kwargs))
        return expected

    module.register_environment_provider("fixture", build)
    actual = module.make_environment_provider(
        SimpleNamespace(env_name="fixture"), is_train=False, env_num=2, group_n=3
    )
    assert actual is expected
    assert calls == [("fixture", {"is_train": False, "env_num": 2, "group_n": 3})]
    with pytest.raises(ValueError, match="already registered"):
        module.register_environment_provider("fixture", build)
    actual.close()
