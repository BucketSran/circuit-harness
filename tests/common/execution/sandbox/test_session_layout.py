"""Compatibility tests for the split execution session implementation."""

from alphaapollo.common.execution._session.gateway import ToolGateway as SplitGateway
from alphaapollo.common.execution._session.legacy import SandboxSession as SplitSandboxSession
from alphaapollo.common.execution._session.runtime import ExecutionRuntime as SplitRuntime
from alphaapollo.common.execution.session import ExecutionRuntime, SandboxSession, ToolGateway


def test_session_facade_reexports_split_implementations() -> None:
    assert ExecutionRuntime is SplitRuntime
    assert ToolGateway is SplitGateway
    assert SandboxSession is SplitSandboxSession
