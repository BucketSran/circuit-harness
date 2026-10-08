from __future__ import annotations

from typing import Any

import pytest

from alphaapollo.common.environment import Env, EnvStepOutput


class _StringEnv(Env[str, str]):
    def __init__(self) -> None:
        self.closed = False

    def init(self, *args: Any, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return "initial", {"args": args, "kwargs": kwargs}

    def step(self, action: str) -> EnvStepOutput:
        return {
            "observations": action.upper(),
            "reward": 1.0,
            "done": True,
            "metadata": None,
        }

    def close(self) -> None:
        self.closed = True


def test_contract_preserves_init_step_close_and_context_manager() -> None:
    env = _StringEnv()
    with env as active:
        assert active is env
        assert active.init("seed", difficulty="hard") == (
            "initial",
            {"args": ("seed",), "kwargs": {"difficulty": "hard"}},
        )
        assert active.step("answer") == {
            "observations": "ANSWER",
            "reward": 1.0,
            "done": True,
            "metadata": None,
        }
    assert env.closed is True


def test_base_methods_fail_loud_except_idempotent_close() -> None:
    env: Env[Any, Any] = Env()
    with pytest.raises(NotImplementedError):
        env.init()
    with pytest.raises(NotImplementedError):
        env.step(None)
    assert env.close() is None


def test_contract_does_not_invent_reset() -> None:
    assert not hasattr(Env, "reset")


def test_step_output_preserves_legacy_required_keys() -> None:
    assert EnvStepOutput.__required_keys__ == {
        "observations",
        "reward",
        "done",
        "metadata",
    }
