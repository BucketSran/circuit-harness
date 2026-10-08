from __future__ import annotations

import math

import pytest

from alphaapollo.common.execution import (
    ExecutionContext,
    ToolError,
    ToolExecutor,
    ToolRequest,
    ToolResponse,
)


class EchoExecutor:
    def execute(self, request: ToolRequest, context: ExecutionContext) -> ToolResponse:
        return ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stdout=f"{context.session_id}:{request.arguments['text']}",
        )


def test_minimum_execution_protocol_is_structural_and_backend_neutral() -> None:
    executor = EchoExecutor()
    assert isinstance(executor, ToolExecutor)

    response = executor.execute(
        ToolRequest(call_id="call-1", tool_id="echo", arguments={"text": "hello"}),
        ExecutionContext(session_id="session-1"),
    )

    assert response.stdout == "session-1:hello"


def test_invocation_separates_provider_metadata_from_model_arguments() -> None:
    arguments = {"path": "README.md"}
    request = ToolRequest(
        call_id="provider-call-1",
        tool_id="read",
        arguments=arguments,
        source="openai_tool_call",
    )
    arguments["path"] = "changed-after-construction"

    assert request.call_id == "provider-call-1"
    assert request.source == "openai_tool_call"
    assert request.arguments == {"path": "README.md"}
    assert "call_id" not in request.arguments
    assert "source" not in request.arguments


def test_invocation_owns_nested_arguments_without_changing_json_container_types() -> None:
    arguments = {"path": "notes.txt", "edits": [{"oldText": "old", "newText": "new"}]}
    request = ToolRequest(call_id="call-edit", tool_id="edit", arguments=arguments)

    arguments["edits"][0]["newText"] = "mutated"

    assert request.arguments["edits"][0]["newText"] == "new"
    assert isinstance(request.arguments, dict)
    assert isinstance(request.arguments["edits"], list)
    assert isinstance(request.arguments["edits"][0], dict)


def test_invocation_context_carries_system_owned_execution_metadata() -> None:
    context = ExecutionContext(
        session_id="session-1",
        branch_id="branch-7",
        workspace_snapshot_ref="sha256:snapshot",
        actor="verifier",
        mode="isolated",
        timeout_s=12.5,
    )

    assert context.actor == "verifier"
    assert context.mode == "isolated"
    assert context.timeout_s == 12.5
    assert context.workspace_snapshot_ref == "sha256:snapshot"


def test_effective_timeout_uses_the_smaller_system_or_tool_value() -> None:
    bounded = ExecutionContext(session_id="session", timeout_s=30)
    unbounded = ExecutionContext(session_id="session")

    assert bounded.effective_timeout_s() == 30
    assert bounded.effective_timeout_s(10) == 10
    assert bounded.effective_timeout_s(60) == 30
    assert unbounded.effective_timeout_s(10) == 10
    assert unbounded.effective_timeout_s() is None
    with pytest.raises(ValueError):
        bounded.effective_timeout_s(math.inf)


def test_tool_error_is_typed_by_pre_execution_stage_and_safe_identity() -> None:
    error = ToolError(
        stage="policy",
        code="path_escape",
        message="path is outside /workspace",
        call_id="call-1",
        tool_id="read",
    )

    assert error.stage == "policy"
    assert error.code == "path_escape"
    assert error.call_id == "call-1"
    assert error.tool_id == "read"


@pytest.mark.parametrize("stage", ["parse", "catalog", "policy", "acquire"])
def test_tool_error_accepts_each_pre_execution_stage(stage: str) -> None:
    assert ToolError(stage=stage, code="failure", message="safe message").stage == stage


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ExecutionContext(session_id=""),
        lambda: ExecutionContext(session_id="session", branch_id=""),
        lambda: ExecutionContext(session_id="session", actor=""),
        lambda: ExecutionContext(session_id="session", mode=""),
        lambda: ExecutionContext(session_id="session", workspace_snapshot_ref=""),
        lambda: ExecutionContext(session_id="session", timeout_s=True),
        lambda: ExecutionContext(session_id="session", timeout_s=0),
        lambda: ExecutionContext(session_id="session", timeout_s=-1),
        lambda: ExecutionContext(session_id="session", timeout_s=math.inf),
        lambda: ToolRequest(call_id="", tool_id="echo"),
        lambda: ToolRequest(call_id="call", tool_id=""),
        lambda: ToolRequest(call_id="call", tool_id="echo", source=""),
        lambda: ToolRequest(call_id="call", tool_id="echo", arguments=[]),
        lambda: ToolRequest(call_id="call", tool_id="echo", arguments={"timeout": math.nan}),
        lambda: ToolResponse(call_id="call", tool_id="", exit_code=0),
        lambda: ToolResponse(call_id="call", tool_id="echo", exit_code=True),
        lambda: ToolError(stage="execute", code="failure", message="safe"),
        lambda: ToolError(stage="parse", code="", message="safe"),
        lambda: ToolError(stage="parse", code="failure", message=""),
        lambda: ToolError(stage="parse", code="failure", message="safe", call_id=""),
    ],
)
def test_runtime_execution_records_fail_loudly_on_invalid_contracts(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()
