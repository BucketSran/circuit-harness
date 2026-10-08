"""Regression coverage for the stable DefaultEnvironment facade."""

from alphaapollo.common.environment.default import environment
from alphaapollo.common.environment.default._environment import DefaultEnvironment
from alphaapollo.common.environment.default.bridge import (
    ExecutorToolBridge,
    GatewayToolBridge,
    ToolBridgeResult,
)


def test_default_environment_facade_reexports_canonical_implementations() -> None:
    assert environment.DefaultEnvironment is DefaultEnvironment
    assert environment.ExecutorToolBridge is ExecutorToolBridge
    assert environment.GatewayToolBridge is GatewayToolBridge
    assert environment.ToolBridgeResult is ToolBridgeResult
