from __future__ import annotations

import ast
import inspect
from pathlib import Path

from alphaapollo.common.artifacts.schemas import ArtifactRef
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
from alphaapollo.common.trajectory import recorder as environment_trajectory
from tests.common.environment.fakes import CanonicalTestToolBridge


class _LearningFixtureExecutor:
    def execute(
        self,
        request: ToolRequest,
        context: ExecutionContext,
    ) -> ToolResponse:
        return ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stdout="fixture-output\n",
        )


class _ReadOnlyCaptureSource:
    def __init__(self, store: TrajectoryStore) -> None:
        self._store = store

    def read_events(self, query=None):
        return self._store.read_events(query)

    def get_blob(self, ref: ArtifactRef) -> bytes:
        return self._store.get_blob(ref)


def test_learning_reads_environment_capture_without_reasoning_import(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(
        jsonl_path=tmp_path / "events.jsonl",
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )
    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_LearningFixtureExecutor()),
        event_sink=TrajectoryEnvironmentEventSink(store),
    )
    env.init(
        session_id="learning-readback",
        actor="solver",
        initial_observation="Fixture problem",
    )
    env.step("<python_code>print('fixture-output')</python_code>")
    env.step("Fixture final")
    env.close()

    view = EnvironmentTrajectoryReader(_ReadOnlyCaptureSource(store)).read_session(
        "learning-readback"
    )
    assert view.records_of(EnvironmentCaptureKind.TOOL_REQUEST)[0].content["arguments"] == {
        "code": "print('fixture-output')"
    }
    tool_response = view.records_of(EnvironmentCaptureKind.TOOL_RESPONSE)[0].content
    assert tool_response["stdout"] == "fixture-output\n"
    assert tool_response["record"]["tool_id"] == "python"
    assert tool_response["record"]["args"] == {"code": "print('fixture-output')"}
    assert (
        view.records_of(EnvironmentCaptureKind.FINAL_OUTPUT)[0].content["content"]
        == "Fixture final"
    )

    tree = ast.parse(inspect.getsource(environment_trajectory))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(name.startswith("alphaapollo.reasoning") for name in imports)
    store.close()
