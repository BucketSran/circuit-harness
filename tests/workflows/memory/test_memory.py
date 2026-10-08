"""Configured Workflow memory projects domain evidence and recalls by task."""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from alphaapollo.common.environment.base import EnvironmentTransition
from alphaapollo.evolving.memory import (
    MemoryBackendRegistry,
    MemoryEntry,
    MemoryKind,
    MemoryLifetime,
    MemoryProvenance,
    MemoryScope,
    TieredMemory,
    WorkingMemory,
    canonical_entry_id,
)
from alphaapollo.evolving.memory.scratchpad import SCRATCHPAD_FILE_CAP
from alphaapollo.reasoning.runtime import AgentResult, AgentRuntime, AgentTask, AgentTurn
from alphaapollo.reasoning.verification import (
    VerificationRequest,
    VerificationResult,
    Verifier,
)
from alphaapollo.workflows.config import ConfigError, MemoryConfig, parse_workflow_config
from alphaapollo.workflows.executor import WorkflowExecutor
from alphaapollo.workflows.memory import (
    WORKFLOW_MEMORY_HEADER,
    WORKFLOW_SCRATCHPAD_HEADER,
    WorkflowMemorySession,
)
from alphaapollo.workflows.memory.adapter import (
    WorkflowMemoryContext,
    build_workflow_memory_adapter,
)
from alphaapollo.workflows.memory.robotics import RoboticsMemoryAdapter
from alphaapollo.workflows.records import Workflow, WorkflowInput


class RecordingRuntime(AgentRuntime):
    def __init__(self) -> None:
        self.tasks: list[AgentTask] = []

    def run_batch(self, tasks: Sequence[AgentTask]) -> list[AgentResult]:
        self.tasks.extend(tasks)
        return [
            AgentResult(
                task_id=task.task_id,
                final_text=f"candidate for {task.metadata['workflow_input_id']}",
                metadata=task.metadata,
            )
            for task in tasks
        ]


class RejectingVerifier(Verifier):
    def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
        return [
            VerificationResult(
                request_id=request.request_id,
                verdict="fail",
                candidate=request.candidate,
                candidate_ref=request.candidate_ref,
                feedback=f"repair only {request.request_id.split(':', 1)[0]}",
            )
            for request in requests
        ]


class TerminalRecordingRuntime(AgentRuntime):
    def __init__(self) -> None:
        self.tasks: list[AgentTask] = []

    def run_batch(self, tasks: Sequence[AgentTask]) -> list[AgentResult]:
        self.tasks.extend(tasks)
        return [
            AgentResult(
                task_id=task.task_id,
                final_text="environment-backed episode output",
                metadata={
                    **task.metadata,
                    "terminal_success": False,
                    "terminal_event_id": f"{task.task_id}:terminal",
                },
            )
            for task in tasks
        ]


class TerminalTransitionRecordingRuntime(AgentRuntime):
    """Emit the terminal `EnvironmentTransition` shape a Robotics episode ends on."""

    def __init__(self) -> None:
        self.tasks: list[AgentTask] = []

    def run_batch(self, tasks: Sequence[AgentTask]) -> list[AgentResult]:
        self.tasks.extend(tasks)
        return [
            AgentResult(
                task_id=task.task_id,
                final_text="episode stalled with the drawer still shut",
                metadata={
                    **task.metadata,
                    # The robotics projector requires environment identity.
                    "environment_init": {"benchmark": "libero"},
                },
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
                            metadata={
                                "event_id": f"{task.task_id}:terminal",
                                "turns_used": 60,
                            },
                        ),
                    ),
                ),
            )
            for task in tasks
        ]


class TerminalResultMemoryAdapter:
    """Test producer standing in for a non-verifier environment adapter."""

    def scope(self, *, context: WorkflowMemoryContext) -> MemoryScope:
        return MemoryScope(
            namespace_id="test-environment:v1",
            task_id=f"episode:{context.workflow_input.input_id}",
            run_id=context.run_id,
        )

    def retrieval_query(
        self,
        *,
        context: WorkflowMemoryContext,
        prompt: str,
    ) -> str:
        return f"{context.workflow_input.problem}\n{context.step_id}\n{prompt}"

    def project_output(
        self,
        output: object,
        *,
        context: WorkflowMemoryContext,
        scope: MemoryScope,
    ) -> MemoryEntry | None:
        if not isinstance(output, AgentResult):
            return None
        success = output.metadata.get("terminal_success")
        event_id = output.metadata.get("terminal_event_id")
        if not isinstance(success, bool) or not isinstance(event_id, str):
            return None
        return MemoryEntry(
            kind=MemoryKind.RESULT if success else MemoryKind.FAILURE,
            content=json.dumps(
                {
                    "schema_version": 1,
                    "workflow_name": context.workflow_name,
                    "input_id": context.workflow_input.input_id,
                    "branch_index": context.branch_index,
                    "step_id": context.step_id,
                    "success": success,
                    "summary": output.final_text,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            scope=scope,
            provenance=MemoryProvenance(
                actor_id="test-environment",
                source_event_id=event_id,
                artifact_ids=(output.task_id,),
            ),
            lifetime=MemoryLifetime.RUN,
        )


def _workflow() -> Workflow:
    return Workflow.from_config(
        parse_workflow_config(
            {
                "version": 1,
                "name": "aime_memory_smoke",
                "roles": [
                    {"id": "solver", "target": "solver", "system_prompt": "solve"},
                    {
                        "id": "judge",
                        "target": "judge",
                        "system_prompt": "judge",
                        "input_template": "{problem}\n{candidate}",
                        "output_format": "verdict-line",
                    },
                ],
                "steps": [
                    {"id": "propose", "kind": "agent", "role": "solver"},
                    {"id": "verify", "kind": "verifier", "role": "judge"},
                    {
                        "id": "revise",
                        "kind": "agent",
                        "role": "solver",
                        "input_template": "{problem}\n{candidate}\n{feedback}",
                        "output": True,
                    },
                ],
                "entry_step": "propose",
                "transitions": [
                    {"source": "propose", "target": "verify"},
                    {"source": "verify", "target": "revise"},
                ],
            }
        )
    )


def _agent_only_workflow() -> Workflow:
    return Workflow.from_config(
        parse_workflow_config(
            {
                "version": 1,
                "name": "environment_memory_smoke",
                "roles": [
                    {"id": "solver", "target": "solver", "system_prompt": "act"},
                ],
                "steps": [
                    {"id": "act", "kind": "agent", "role": "solver", "output": True},
                ],
                "entry_step": "act",
                "transitions": [],
            }
        )
    )


def test_workflow_memory_accepts_a_non_verifier_adapter(tmp_path) -> None:
    runtime = TerminalRecordingRuntime()
    journal = tmp_path / "workflow-memory.jsonl"
    memory = WorkflowMemorySession(
        MemoryConfig(top_k=3),
        workflow_name="environment_memory_smoke",
        run_id="run-1",
        journal_path=journal,
        adapter=TerminalResultMemoryAdapter(),
    )
    executor = WorkflowExecutor(
        _agent_only_workflow(),
        runtimes={"solver": runtime},
        verifiers={},
        memory=memory,
    )
    workflow_input = WorkflowInput(input_id="episode-one", problem="complete the task")

    executor.run(workflow_input)
    executor.run(workflow_input)

    first, second = runtime.tasks
    assert WORKFLOW_MEMORY_HEADER not in first.prompt
    assert WORKFLOW_MEMORY_HEADER in second.prompt
    assert "environment-backed episode output" in second.prompt
    assert '"success":false' in second.prompt
    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    intents = [event for event in events if event["kind"] == "write_intent"]
    assert intents[0]["entry"]["provenance"]["actor_id"] == "test-environment"
    assert intents[0]["entry"]["scope"]["namespace_id"] == "test-environment:v1"
    assert any(event["kind"] == "memory_retrieve" for event in events)


def test_robotics_memory_adapter_projects_only_terminal_environment_results(tmp_path) -> None:
    session = WorkflowMemorySession(
        MemoryConfig(adapter="robotics", top_k=3),
        workflow_name="robotics_memory_smoke",
        run_id="run-1",
        journal_path=tmp_path / "workflow-memory.jsonl",
    )
    workflow_input = WorkflowInput(input_id="episode-one", problem="place the cup")

    transition = EnvironmentTransition(
        observation={"environment_success": True},
        reward=1.0,
        done=True,
        success=True,
        termination_reason="task_success",
        metadata={"event_id": "episode-one:terminal", "steps_used": 12},
    )
    output = AgentResult(
        task_id="episode-one:solve:1",
        final_text="The cup is placed.",
        metadata={"environment_init": {"benchmark": "libero"}},
        turns=(
            AgentTurn(
                index=0,
                generation_request=object(),
                generation_response=object(),
                environment_transition=transition,
            ),
        ),
    )

    assert (
        session.observe_output(
            workflow_input,
            branch_index=0,
            step_id="solve",
            output=output,
        )
        is not None
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "workflow-memory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    intent = next(event for event in events if event["kind"] == "write_intent")
    assert intent["entry"]["provenance"]["actor_id"] == "robotics-environment"
    assert intent["entry"]["provenance"]["source_event_id"] == "episode-one:terminal"
    assert json.loads(intent["entry"]["content"])["success"] is True

    nonterminal = AgentResult(
        task_id="episode-one:solve:2",
        final_text="Still working.",
        metadata={"environment_init": {"benchmark": "libero"}},
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
    assert (
        session.observe_output(
            workflow_input,
            branch_index=0,
            step_id="solve",
            output=nonterminal,
        )
        is None
    )


def test_robotics_memory_adapter_is_the_configured_builtin() -> None:
    adapter = build_workflow_memory_adapter("robotics")
    assert isinstance(adapter, RoboticsMemoryAdapter)


def test_robotics_memory_returns_its_own_terminal_entry_to_a_later_model_call(tmp_path) -> None:
    """Cover the recall half of the Robotics adapter contract.

    `project_output` is covered above, but `retrieval_query` is only reachable
    through `augment_prompt`, and `libero_demo` is a single-step workflow whose
    one augmentation happens before the terminal write. Without this test the
    Robotics half of the adapter never demonstrates that a committed episode
    outcome comes back to a subsequent model call.
    """

    runtime = TerminalTransitionRecordingRuntime()
    journal = tmp_path / "workflow-memory.jsonl"
    memory = WorkflowMemorySession(
        MemoryConfig(adapter="robotics", top_k=3),
        workflow_name="robotics_memory_smoke",
        run_id="run-1",
        journal_path=journal,
    )
    executor = WorkflowExecutor(
        _agent_only_workflow(),
        runtimes={"solver": runtime},
        verifiers={},
        memory=memory,
    )
    workflow_input = WorkflowInput(input_id="episode-one", problem="open the top drawer")

    executor.run(workflow_input)
    executor.run(workflow_input)

    first, second = runtime.tasks
    assert WORKFLOW_MEMORY_HEADER not in first.prompt
    assert WORKFLOW_MEMORY_HEADER in second.prompt
    # The environment outcome, not the model's own claim, is what comes back.
    assert '"success":false' in second.prompt
    assert '"termination_reason":"time_limit"' in second.prompt
    # The model-authored summary must stay marked untrusted across the handoff.
    assert '"agent_summary_trust":"untrusted"' in second.prompt

    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    committed = [event for event in events if event["kind"] == "write_committed"]
    retrieved = [event for event in events if event["kind"] == "memory_retrieve"]
    assert committed
    assert retrieved
    assert committed[0]["entry_id"] in retrieved[0]["entry_ids"]

    intent = next(event for event in events if event["kind"] == "write_intent")
    assert intent["entry"]["provenance"]["actor_id"] == "robotics-environment"
    assert intent["entry"]["scope"]["namespace_id"] == "robotics:robotics_memory_smoke"
    assert intent["entry"]["scope"]["task_id"] == "episode-one:branch-0"


def test_robotics_retrieval_query_is_step_scoped_and_bounds_the_prompt_tail() -> None:
    context = WorkflowMemoryContext(
        workflow_name="robotics_memory_smoke",
        run_id="run-1",
        workflow_input=WorkflowInput(input_id="episode-one", problem="open the top drawer"),
        branch_index=0,
        step_id="solve",
    )

    query = RoboticsMemoryAdapter().retrieval_query(context=context, prompt="a" * 5_000)

    # Only the prompt tail is carried, so an unbounded episode transcript cannot
    # grow the retrieval query without limit.
    assert query == "open the top drawer\nrobotics step solve\n" + "a" * 4_000


def test_workflow_memory_injects_only_same_task_verified_judgments(tmp_path) -> None:
    runtime = RecordingRuntime()
    journal = tmp_path / "workflow-memory.jsonl"
    memory = WorkflowMemorySession(
        MemoryConfig(top_k=3),
        workflow_name="aime_memory_smoke",
        run_id="run-1",
        journal_path=journal,
    )
    executor = WorkflowExecutor(
        _workflow(),
        runtimes={"solver": runtime},
        verifiers={"judge": RejectingVerifier()},
        memory=memory,
    )

    executor.run_batch(
        (
            WorkflowInput(input_id="aime-one", problem="first problem"),
            WorkflowInput(input_id="aime-two", problem="second problem"),
        )
    )

    propose, propose_two, revise, revise_two = runtime.tasks
    assert WORKFLOW_MEMORY_HEADER not in propose.prompt
    assert WORKFLOW_MEMORY_HEADER not in propose_two.prompt
    assert "repair only aime-one" in revise.prompt
    assert "candidate for aime-one" in revise.prompt
    assert "aime-two" not in revise.prompt
    assert "repair only aime-two" in revise_two.prompt
    assert "candidate for aime-two" in revise_two.prompt
    assert "aime-one" not in revise_two.prompt

    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [event["kind"] for event in events].count("write_committed") == 2
    retrieves = [event for event in events if event["kind"] == "memory_retrieve"]
    assert len(retrieves) == 2
    assert all(WORKFLOW_MEMORY_HEADER in event["rendered_context"] for event in retrieves)


def test_persistent_workflow_memory_uses_registered_backend(tmp_path) -> None:
    captured: dict[str, object] = {}
    registry = MemoryBackendRegistry()

    def build_example(config):
        captured["config"] = dict(config)
        backend = WorkingMemory()
        captured["backend"] = backend
        return backend

    registry.register("example-framework", build_example)
    session = WorkflowMemorySession(
        MemoryConfig(
            mode="persistent",
            persistent_backend="example-framework",
            persistent_config={"endpoint": "memory.example"},
        ),
        workflow_name="aime_memory_smoke",
        run_id="run-1",
        journal_path=tmp_path / "workflow-memory.jsonl",
        backends=registry,
    )
    workflow_input = WorkflowInput(input_id="aime-one", problem="problem ERR42")
    session.observe_verification(
        workflow_input,
        branch_index=0,
        step_id="verify",
        result=VerificationResult(
            request_id="verify-1",
            verdict="fail",
            candidate="candidate ERR42",
            candidate_ref="candidate-ref",
            feedback="repair ERR42 exactly",
        ),
    )

    prompt = session.augment_prompt(
        workflow_input,
        branch_index=0,
        step_id="revise",
        prompt="fix ERR42",
    )

    assert captured["config"] == {"endpoint": "memory.example"}
    assert "repair ERR42 exactly" in prompt
    backend = captured["backend"]
    assert isinstance(backend, WorkingMemory)
    scope = MemoryScope(
        namespace_id="workflow:aime_memory_smoke",
        task_id="aime-one:branch-0",
        run_id="run-1",
    )
    assert len(backend.snapshot(reader=scope)) == 1


def test_workflow_memory_skips_a_verifier_without_a_bound_candidate(tmp_path) -> None:
    session = WorkflowMemorySession(
        MemoryConfig(),
        workflow_name="verifier_entry",
        run_id="run-1",
        journal_path=tmp_path / "memory.jsonl",
    )
    workflow_input = WorkflowInput(input_id="aime-one", problem="problem")
    result = VerificationResult(
        request_id="verify-1",
        verdict="inconclusive",
        candidate="problem",
        feedback="no candidate was produced",
    )

    assert (
        session.observe_verification(
            workflow_input,
            branch_index=0,
            step_id="verify",
            result=result,
        )
        is None
    )
    assert not session.journal_path.exists()


def test_full_profile_recovers_memory_and_branch_scratchpad(tmp_path) -> None:
    journal = tmp_path / "memory.jsonl"
    config = MemoryConfig(profile="full")
    workflow_input = WorkflowInput(input_id="aime-one", problem="problem ERR42")
    session = WorkflowMemorySession(
        config,
        workflow_name="workflow",
        run_id="run-1",
        journal_path=journal,
    )
    result = VerificationResult(
        request_id="verify-1",
        verdict="fail",
        candidate="candidate ERR42",
        candidate_ref="candidate-ref",
        feedback="repair ERR42 exactly",
    )
    session.observe_verification(
        workflow_input,
        branch_index=0,
        step_id="verify",
        result=result,
    )
    session.sync_scratchpad(
        input_id="aime-one",
        branch_index=0,
        step_id="revise",
        content="## Failed approach\nDo not repeat ERR42",
    )

    recovered = WorkflowMemorySession(
        config,
        workflow_name="workflow",
        run_id="run-1",
        journal_path=journal,
    )
    prompt = recovered.augment_prompt(
        workflow_input,
        branch_index=0,
        step_id="finalize",
        prompt="fix ERR42",
    )

    assert WORKFLOW_MEMORY_HEADER in prompt
    assert WORKFLOW_SCRATCHPAD_HEADER in prompt
    assert "repair ERR42 exactly" in prompt
    assert "Do not repeat ERR42" in prompt
    retrieves = [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["kind"] == "memory_retrieve"
    ]
    assert retrieves[-1]["tiers"] == ["semantic", "lexical"]


def test_full_profile_mirrors_external_agent_workspace(tmp_path) -> None:
    session = WorkflowMemorySession(
        MemoryConfig(profile="full"),
        workflow_name="workflow",
        run_id="run-1",
        journal_path=tmp_path / "memory.jsonl",
    )
    session.sync_scratchpad(
        input_id="aime-one",
        branch_index=0,
        step_id="propose",
        content="## Facts\nx=7",
    )
    task = SimpleNamespace(
        metadata={
            "actor": "solver",
            "workflow_input_id": "aime-one",
            "workflow_step_id": "revise",
            "branch_index": 0,
        }
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    session.prepare_agent_workspace(task, workspace)
    scratchpad = workspace / ".alphaapollo" / "scratchpad.md"
    assert scratchpad.read_text(encoding="utf-8") == "## Facts\nx=7"
    scratchpad.write_text("## Facts\nx=8", encoding="utf-8")
    session.observe_agent_workspace(task, workspace)

    assert session.scratchpad_content(input_id="aime-one", branch_index=0).endswith("x=8")


def test_external_agent_scratchpad_rejects_symlinks_without_reading_target(tmp_path) -> None:
    journal = tmp_path / "memory.jsonl"
    session = WorkflowMemorySession(
        MemoryConfig(profile="full"),
        workflow_name="workflow",
        run_id="run-1",
        journal_path=journal,
    )
    task = SimpleNamespace(
        metadata={
            "actor": "solver",
            "workflow_input_id": "aime-one",
            "workflow_step_id": "revise",
            "branch_index": 0,
        }
    )
    workspace = tmp_path / "workspace"
    notes = workspace / ".alphaapollo"
    notes.mkdir(parents=True)
    target = tmp_path / "outside-secret.txt"
    target.write_text("must-not-enter-memory", encoding="utf-8")
    (notes / "scratchpad.md").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        session.observe_agent_workspace(task, workspace)

    assert session.scratchpad_content(input_id="aime-one", branch_index=0) == ""
    assert not journal.exists()


def test_external_agent_scratchpad_is_read_with_a_hard_size_bound(tmp_path) -> None:
    session = WorkflowMemorySession(
        MemoryConfig(profile="full"),
        workflow_name="workflow",
        run_id="run-1",
        journal_path=tmp_path / "memory.jsonl",
    )
    task = SimpleNamespace(
        metadata={
            "actor": "solver",
            "workflow_input_id": "aime-one",
            "workflow_step_id": "revise",
            "branch_index": 0,
        }
    )
    scratchpad = tmp_path / "workspace" / ".alphaapollo" / "scratchpad.md"
    scratchpad.parent.mkdir(parents=True)
    scratchpad.write_text("x" * (SCRATCHPAD_FILE_CAP + 1), encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds"):
        session.observe_agent_workspace(task, tmp_path / "workspace")


def test_external_agent_scratchpad_replaces_invalid_utf8_after_a_successful_call(tmp_path) -> None:
    session = WorkflowMemorySession(
        MemoryConfig(profile="full"),
        workflow_name="workflow",
        run_id="run-1",
        journal_path=tmp_path / "memory.jsonl",
    )
    task = SimpleNamespace(
        metadata={
            "actor": "solver",
            "workflow_input_id": "aime-one",
            "workflow_step_id": "revise",
            "branch_index": 0,
        }
    )
    scratchpad = tmp_path / "workspace" / ".alphaapollo" / "scratchpad.md"
    scratchpad.parent.mkdir(parents=True)
    scratchpad.write_bytes(b"facts:\xff")

    session.observe_agent_workspace(task, tmp_path / "workspace")

    assert session.scratchpad_content(input_id="aime-one", branch_index=0) == "facts:\ufffd"


def test_workflow_prompt_renders_plain_text_memory_without_crashing(tmp_path) -> None:
    session = WorkflowMemorySession(
        MemoryConfig(),
        workflow_name="workflow",
        run_id="run-1",
        journal_path=tmp_path / "memory.jsonl",
    )
    workflow_input = WorkflowInput(input_id="aime-one", problem="problem")
    session.write(
        MemoryEntry(
            kind=MemoryKind.CLAIM,
            content="plain verified note",
            scope=MemoryScope(
                namespace_id="workflow:workflow",
                task_id="aime-one:branch-0",
                run_id="run-1",
            ),
            provenance=MemoryProvenance(actor_id="test", source_event_id="event-1"),
        )
    )

    prompt = session.augment_prompt(
        workflow_input,
        branch_index=0,
        step_id="revise",
        prompt="try again",
    )

    assert "plain verified note" in prompt


def test_workflow_recovery_replays_exact_write_instead_of_similarity_dedup(tmp_path) -> None:
    class SimilarityTrapMemory(WorkingMemory):
        def __init__(self) -> None:
            super().__init__()
            self.writes = 0
            self.dedup_calls = 0

        def write(self, entry: MemoryEntry) -> str:
            self.writes += 1
            return super().write(entry)

        def dedup(self, entry: MemoryEntry, *, threshold=None) -> bool:
            del entry, threshold
            self.dedup_calls += 1
            return True

    entry = MemoryEntry(
        kind=MemoryKind.RESULT,
        content='{"answer":205}',
        scope=MemoryScope(
            namespace_id="workflow:workflow",
            task_id="aime-one:branch-0",
            run_id="run-1",
        ),
        provenance=MemoryProvenance(actor_id="verifier", source_event_id="verify-205"),
    )
    entry_id = canonical_entry_id(entry)
    journal = tmp_path / "memory.jsonl"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "run_id": "run-1",
                "kind": "write_intent",
                "entry_id": entry_id,
                "entry": entry.model_dump(mode="json"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    persistent = SimilarityTrapMemory()

    WorkflowMemorySession(
        MemoryConfig(),
        workflow_name="workflow",
        run_id="run-1",
        journal_path=journal,
        memory=TieredMemory(persistent=persistent),
    )

    assert persistent.writes == 1
    assert persistent.dedup_calls == 0
    assert "write_committed" in journal.read_text(encoding="utf-8")


def test_workflow_recovery_ignores_only_a_torn_final_journal_record(tmp_path) -> None:
    journal = tmp_path / "memory.jsonl"
    config = MemoryConfig(profile="full")
    session = WorkflowMemorySession(
        config,
        workflow_name="workflow",
        run_id="run-1",
        journal_path=journal,
    )
    session.sync_scratchpad(
        input_id="aime-one",
        branch_index=0,
        step_id="revise",
        content="durable notes",
    )
    with journal.open("ab") as handle:
        handle.write(b'{"version":1,"kind":')

    recovered = WorkflowMemorySession(
        config,
        workflow_name="workflow",
        run_id="run-1",
        journal_path=journal,
    )

    assert recovered.scratchpad_content(input_id="aime-one", branch_index=0) == "durable notes"


def test_workflow_recovery_rejects_corruption_before_the_final_record(tmp_path) -> None:
    journal = tmp_path / "memory.jsonl"
    journal.write_text("{}\nnot-json\n{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="line 2"):
        WorkflowMemorySession(
            MemoryConfig(),
            workflow_name="workflow",
            run_id="run-1",
            journal_path=journal,
        )


def test_recall_returns_entries_and_scratchpad_without_rendering(tmp_path) -> None:
    """The non-rendering twin of render_context, for callers that own their
    prompt format: same retrieval and journal audit, structured results out."""

    journal = tmp_path / "memory.jsonl"
    session = WorkflowMemorySession(
        MemoryConfig(profile="full"),
        workflow_name="workflow",
        run_id="run-1",
        journal_path=journal,
    )
    workflow_input = WorkflowInput(input_id="aime-one", problem="problem ERR42")
    session.observe_verification(
        workflow_input,
        branch_index=0,
        step_id="verify",
        result=VerificationResult(
            request_id="verify-1",
            verdict="fail",
            candidate="candidate ERR42",
            candidate_ref="candidate-ref",
            feedback="repair ERR42 exactly",
        ),
    )
    session.sync_scratchpad(
        input_id="aime-one",
        branch_index=0,
        step_id="revise",
        content="own notes about ERR42",
    )

    entries, scratchpad = session.recall(
        workflow_input,
        branch_index=0,
        step_id="revise",
        prompt="fix ERR42",
    )

    assert len(entries) == 1
    assert "repair ERR42 exactly" in entries[0].content
    assert scratchpad == "own notes about ERR42"
    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    retrieve = next(e for e in events if e["kind"] == "memory_retrieve")
    assert retrieve["entry_ids"] and "rendered_context" not in retrieve
    assert any(e["kind"] == "scratchpad_retrieve" for e in events)

    # Journaled even when nothing is found: distinguishable from "never recalled".
    other_input = WorkflowInput(input_id="aime-two", problem="different problem")
    empty_entries, _ = session.recall(
        other_input, branch_index=0, step_id="revise", prompt="anything"
    )
    assert empty_entries == ()
    last_retrieve = [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["kind"] == "memory_retrieve"
    ][-1]
    assert last_retrieve["entry_ids"] == []


def test_cross_run_requires_persistent_mode() -> None:
    with pytest.raises(ConfigError, match="cross_run"):
        MemoryConfig(cross_run=True)


def test_default_adapters_keep_todays_scope_and_lifetime() -> None:
    """cross_run=False is byte-identical to the pre-#340 behavior."""

    from alphaapollo.workflows.memory.robotics import RoboticsMemoryAdapter
    from alphaapollo.workflows.memory.verification import VerificationMemoryAdapter

    context = WorkflowMemoryContext(
        workflow_name="wf",
        run_id="run-1",
        workflow_input=WorkflowInput(input_id="task-one", problem="problem"),
        branch_index=0,
        step_id="verify",
    )
    verification = VerificationMemoryAdapter()
    scope = verification.scope(context=context)
    assert scope.task_id == "task-one:branch-0"
    entry = verification.project_output(
        VerificationResult(
            request_id="verify-1",
            verdict="fail",
            candidate="candidate",
            candidate_ref="ref",
            feedback="feedback",
        ),
        context=context,
        scope=scope,
    )
    assert entry is not None and entry.lifetime is MemoryLifetime.RUN
    assert RoboticsMemoryAdapter().scope(context=context).task_id == "task-one:branch-0"


def _cross_run_registry(store: WorkingMemory) -> MemoryBackendRegistry:
    registry = MemoryBackendRegistry()
    registry.register("example-framework", lambda config: store)
    return registry


def _cross_run_session(run_id: str, store: WorkingMemory, tmp_path, adapter: str):
    return WorkflowMemorySession(
        MemoryConfig(
            adapter=adapter,
            mode="persistent",
            cross_run=True,
            persistent_backend="example-framework",
            persistent_config={},
        ),
        workflow_name="wf",
        run_id=run_id,
        journal_path=tmp_path / f"memory-{run_id}.jsonl",
        backends=_cross_run_registry(store),
    )


def test_verification_cross_run_recall_reaches_the_next_run(tmp_path) -> None:
    """#340: run B recalls run A's verifier judgment for the same task."""

    store = WorkingMemory()
    workflow_input = WorkflowInput(input_id="task-one", problem="problem ERR42")

    first = _cross_run_session("run-1", store, tmp_path, "verification")
    first.observe_verification(
        workflow_input,
        branch_index=0,
        step_id="verify",
        result=VerificationResult(
            request_id="verify-1",
            verdict="fail",
            candidate="candidate ERR42",
            candidate_ref="candidate-ref",
            feedback="repair ERR42 exactly",
        ),
    )

    second = _cross_run_session("run-2", store, tmp_path, "verification")
    prompt = second.augment_prompt(
        workflow_input,
        branch_index=1,  # a different branch too: persistent scope has no branch
        step_id="revise",
        prompt="fix ERR42",
    )

    assert WORKFLOW_MEMORY_HEADER in prompt
    assert "repair ERR42 exactly" in prompt


def test_robotics_cross_run_recall_reaches_the_next_run(tmp_path) -> None:
    store = WorkingMemory()
    workflow_input = WorkflowInput(input_id="episode-one", problem="open the drawer")
    terminal = AgentResult(
        task_id="episode-one:solve:1",
        final_text="episode stalled",
        metadata={"environment_init": {"benchmark": "libero"}},
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
                    metadata={"event_id": "episode-one:terminal", "turns_used": 60},
                ),
            ),
        ),
    )

    first = _cross_run_session("run-1", store, tmp_path, "robotics")
    first.observe_output(workflow_input, branch_index=0, step_id="solve", output=terminal)

    second = _cross_run_session("run-2", store, tmp_path, "robotics")
    prompt = second.augment_prompt(
        workflow_input, branch_index=0, step_id="solve", prompt="try again"
    )

    assert WORKFLOW_MEMORY_HEADER in prompt
    assert '"termination_reason":"time_limit"' in prompt


def test_run_lifetime_entries_stay_invisible_across_runs(tmp_path) -> None:
    """Isolation regression: without cross_run, nothing crosses the boundary."""

    store = WorkingMemory()
    workflow_input = WorkflowInput(input_id="task-one", problem="problem ERR42")

    def session(run_id: str):
        return WorkflowMemorySession(
            MemoryConfig(
                mode="persistent",
                persistent_backend="example-framework",
                persistent_config={},
            ),
            workflow_name="wf",
            run_id=run_id,
            journal_path=tmp_path / f"memory-{run_id}.jsonl",
            backends=_cross_run_registry(store),
        )

    session("run-1").observe_verification(
        workflow_input,
        branch_index=0,
        step_id="verify",
        result=VerificationResult(
            request_id="verify-1",
            verdict="fail",
            candidate="candidate ERR42",
            candidate_ref="candidate-ref",
            feedback="repair ERR42 exactly",
        ),
    )
    prompt = session("run-2").augment_prompt(
        workflow_input, branch_index=0, step_id="revise", prompt="fix ERR42"
    )

    assert WORKFLOW_MEMORY_HEADER not in prompt


def test_memory_config_without_adapter_name_is_injection_only(tmp_path) -> None:
    """A configuration must not name an adapter it does not honor (P2a).

    adapter=None is the injection-only form custom adapters use; constructing
    a session from it without an adapter instance fails before any side
    effect instead of silently selecting a default.
    """

    with pytest.raises(ValueError, match="injection-only"):
        WorkflowMemorySession(
            MemoryConfig(adapter=None),
            workflow_name="wf",
            run_id="run-1",
            journal_path=tmp_path / "memory.jsonl",
        )
