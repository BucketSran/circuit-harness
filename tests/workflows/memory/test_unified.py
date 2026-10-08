"""Unified memory adapter contract tests (#347)."""

from __future__ import annotations

import json

import pytest

from alphaapollo.common.environment.base import EnvironmentTransition
from alphaapollo.evolving.memory import canonical_entry_id
from alphaapollo.reasoning.runtime import AgentResult, AgentTurn
from alphaapollo.reasoning.verification import VerificationResult
from alphaapollo.workflows.config import ConfigError, MemoryConfig
from alphaapollo.workflows.memory import WORKFLOW_MEMORY_HEADER, WorkflowMemorySession
from alphaapollo.workflows.memory.adapter import (
    WorkflowMemoryContext,
    build_workflow_memory_adapter,
)
from alphaapollo.workflows.memory.robotics import RoboticsMemoryAdapter
from alphaapollo.workflows.memory.unified import UnifiedMemoryAdapter
from alphaapollo.workflows.memory.verification import VerificationMemoryAdapter
from alphaapollo.workflows.records import WorkflowInput


def _context(step_id: str = "solve") -> WorkflowMemoryContext:
    return WorkflowMemoryContext(
        workflow_name="unified_smoke",
        run_id="run-1",
        workflow_input=WorkflowInput(input_id="task-one", problem="problem ERR42"),
        branch_index=0,
        step_id=step_id,
    )


def _verification_output() -> VerificationResult:
    return VerificationResult(
        request_id="verify-1",
        verdict="fail",
        candidate="candidate ERR42",
        candidate_ref="candidate-ref",
        feedback="repair ERR42 exactly",
    )


def _robotics_output() -> AgentResult:
    return AgentResult(
        task_id="task-one:solve:1",
        final_text="The drawer stayed shut.",
        # The environment-authored episode identity the unified dispatch
        # requires before treating a terminal AgentResult as robotics output.
        metadata={"environment_init": {"benchmark": "libero", "task_id": "task-one"}},
        turns=(
            AgentTurn(
                index=0,
                generation_request=object(),
                generation_response=object(),
                environment_transition=EnvironmentTransition(
                    observation={"environment_success": False},
                    reward=0.0,
                    done=True,
                    success=False,
                    termination_reason="time_limit",
                    metadata={"event_id": "task-one:terminal", "steps_used": 12},
                ),
            ),
        ),
    )


def test_config_and_factory_select_the_unified_adapter() -> None:
    MemoryConfig(adapter="unified")
    adapter = build_workflow_memory_adapter("unified")
    assert isinstance(adapter, UnifiedMemoryAdapter)


def test_unified_scope_is_the_workflow_scheme() -> None:
    scope = UnifiedMemoryAdapter().scope(context=_context())
    assert scope.namespace_id == "workflow:unified_smoke"
    assert scope.task_id == "task-one:branch-0"
    assert scope.run_id == "run-1"


def test_unified_projections_are_byte_identical_to_the_single_adapters() -> None:
    """Delegation, verified: same output, same scope — same canonical id."""

    unified = UnifiedMemoryAdapter()
    context = _context()
    scope = unified.scope(context=context)
    singles = (
        (VerificationMemoryAdapter(), _verification_output()),
        (RoboticsMemoryAdapter(), _robotics_output()),
    )

    for single, output in singles:
        unified_entry = unified.project_output(output, context=context, scope=scope)
        single_entry = single.project_output(output, context=context, scope=scope)
        assert unified_entry is not None and single_entry is not None
        assert canonical_entry_id(unified_entry) == canonical_entry_id(single_entry)


def test_unified_dispatch_fails_closed() -> None:
    unified = UnifiedMemoryAdapter()
    context = _context()
    scope = unified.scope(context=context)

    # Unknown output types never write.
    assert unified.project_output("free text", context=context, scope=scope) is None
    # A non-terminal robotics episode is rejected by the delegated gating.
    nonterminal = AgentResult(
        task_id="task-one:solve:2",
        final_text="still working",
        turns=(
            AgentTurn(
                index=0,
                generation_request=object(),
                generation_response=object(),
                environment_transition=EnvironmentTransition(
                    observation={}, reward=0.0, done=False
                ),
            ),
        ),
    )
    assert unified.project_output(nonterminal, context=context, scope=scope) is None


def test_unified_never_projects_a_math_terminal_as_a_robotics_outcome() -> None:
    """The default (tool) environment also ends episodes with done=True and a
    boolean success — budget exhaustion returns success=False — so terminal
    shape alone must not satisfy the robotics branch. Without the episode
    identity from a robotics initialization, the entry is refused."""

    unified = UnifiedMemoryAdapter()
    context = _context()
    scope = unified.scope(context=context)
    math_terminal = AgentResult(
        task_id="task-one:solve:1",
        final_text="ran out of tool budget",
        # No environment_init metadata: this is what a default-environment
        # (Math with tools) result looks like after the env exhausts its turns.
        turns=(
            AgentTurn(
                index=0,
                generation_request=object(),
                generation_response=object(),
                environment_transition=EnvironmentTransition(
                    observation={"tool": "python"},
                    reward=0.0,
                    done=True,
                    success=False,
                    termination_reason="limit",
                ),
            ),
        ),
    )

    assert unified.project_output(math_terminal, context=context, scope=scope) is None


def test_unified_adapter_writes_and_recalls_through_the_session(tmp_path) -> None:
    session = WorkflowMemorySession(
        MemoryConfig(adapter="unified"),
        workflow_name="unified_smoke",
        run_id="run-1",
        journal_path=tmp_path / "memory.jsonl",
    )
    workflow_input = WorkflowInput(input_id="task-one", problem="problem ERR42")

    session.observe_output(
        workflow_input,
        branch_index=0,
        step_id="verify",
        output=_verification_output(),
    )
    session.observe_output(
        workflow_input,
        branch_index=0,
        step_id="solve",
        output=_robotics_output(),
    )

    prompt = session.augment_prompt(
        workflow_input,
        branch_index=0,
        step_id="revise",
        prompt="fix ERR42",
    )

    assert WORKFLOW_MEMORY_HEADER in prompt
    assert "repair ERR42 exactly" in prompt
    assert '"termination_reason":"time_limit"' in prompt
    events = [
        json.loads(line)
        for line in (tmp_path / "memory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    committed = [event["entry_id"] for event in events if event["kind"] == "write_committed"]
    retrieved = next(event for event in events if event["kind"] == "memory_retrieve")
    assert len(committed) == 2
    assert set(retrieved["entry_ids"]) == set(committed)


def test_unknown_adapter_names_stay_rejected() -> None:
    with pytest.raises(ConfigError, match="unified"):
        MemoryConfig(adapter="everything")
    with pytest.raises(ValueError, match="unified"):
        build_workflow_memory_adapter("everything")


def test_unified_cross_run_switches_scope_and_lifetime() -> None:
    """#340 through the unified adapter: one switch, all three projections."""

    from alphaapollo.evolving.memory import MemoryLifetime

    unified = build_workflow_memory_adapter("unified", cross_run=True)
    context = _context()
    scope = unified.scope(context=context)
    assert scope.task_id == "task-one"  # branch suffix dropped

    entry = unified.project_output(_verification_output(), context=context, scope=scope)
    assert entry is not None and entry.lifetime is MemoryLifetime.PERSISTENT
    robotics_entry = unified.project_output(_robotics_output(), context=context, scope=scope)
    assert robotics_entry is not None and robotics_entry.lifetime is MemoryLifetime.PERSISTENT
