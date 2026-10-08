from __future__ import annotations

from pathlib import Path

from alphaapollo.common.environment import (
    DefaultEnvironment,
    EnvironmentCaptureKind,
    EnvironmentTrajectoryReader,
    TrajectoryEnvironmentEventSink,
)
from alphaapollo.common.execution import (
    ArtifactStore,
    ExecutionContext,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.trajectory import TrajectoryStore
from tests.common.environment.fakes import CanonicalTestToolBridge


class _UnusedExecutor:
    def execute(
        self,
        request: ToolRequest,
        context: ExecutionContext,
    ) -> ToolResponse:
        raise AssertionError("no-tool revision fixture must not execute a tool")


def _run_no_tool_episode(
    store: TrajectoryStore,
    *,
    actor: str,
    branch_id: str,
    round_index: int,
    initial: str,
    output: str,
) -> None:
    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_UnusedExecutor()),
        event_sink=TrajectoryEnvironmentEventSink(store),
    )
    env.init(
        session_id="fail-revise-session",
        actor=actor,
        branch_id=branch_id,
        round_index=round_index,
        initial_observation=initial,
        episode_context={
            "workflow": "vanilla",
            "max_rounds": 2,
        },
    )
    result = env.step(output)
    assert result["done"] is True
    env.close()


def test_fail_feedback_revision_is_replayable_across_rounds(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        jsonl_path=tmp_path / "events.jsonl",
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    _run_no_tool_episode(
        store,
        actor="solver",
        branch_id="solver",
        round_index=0,
        initial="Problem statement",
        output="Candidate answer: 203",
    )
    _run_no_tool_episode(
        store,
        actor="verifier",
        branch_id="verifier",
        round_index=0,
        initial="Verify candidate answer 203",
        output='{"verdict":"FAIL","feedback":"recheck the arithmetic"}',
    )
    _run_no_tool_episode(
        store,
        actor="solver",
        branch_id="solver",
        round_index=1,
        initial="Revise using feedback: recheck the arithmetic",
        output="Revised answer: 204",
    )

    reader = EnvironmentTrajectoryReader(store)
    solver = reader.read_session(
        "fail-revise-session",
        branch_id="solver",
        actor="solver",
    )
    verifier = reader.read_session(
        "fail-revise-session",
        branch_id="verifier",
        actor="verifier",
    )
    solver_final = solver.records_of(EnvironmentCaptureKind.FINAL_OUTPUT)
    verifier_final = verifier.records_of(EnvironmentCaptureKind.FINAL_OUTPUT)
    assert [(record.round_index, record.content["content"]) for record in solver_final] == [
        (0, "Candidate answer: 203"),
        (1, "Revised answer: 204"),
    ]
    assert verifier_final[0].content["content"] == (
        '{"verdict":"FAIL","feedback":"recheck the arithmetic"}'
    )
    assert [(step.round_index, step.step_index) for step in solver.replay_steps()] == [
        (0, 1),
        (1, 1),
    ]
    assert [event.sequence for event in store.read_events()] == list(
        range(len(store.read_events()))
    )
    store.close()
