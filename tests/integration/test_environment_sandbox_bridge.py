from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from alphaapollo.common.environment import (
    DefaultEnvironment,
    EnvironmentCaptureKind,
    EnvironmentTrajectoryReader,
    GatewayToolBridge,
    TrajectoryEnvironmentEventSink,
)
from alphaapollo.common.execution import ArtifactStore, ExecutionRuntime, ToolGateway
from alphaapollo.common.trajectory import TrajectoryStore


def _apollo_containers() -> set[str]:
    completed = subprocess.run(
        [
            "podman",
            "ps",
            "-a",
            "--filter",
            "name=apollo_pod_",
            "--format",
            "{{.Names}}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return {name for name in completed.stdout.splitlines() if name}


@pytest.mark.skipif(
    os.environ.get("ALPHAAPOLLO_RUN_REAL_PODMAN") != "1",
    reason="set ALPHAAPOLLO_RUN_REAL_PODMAN=1 on a rootless-Podman host",
)
def test_real_environment_gateway_podman_round_trip_is_captured(tmp_path: Path) -> None:
    containers_before = _apollo_containers()
    artifacts = ArtifactStore(tmp_path / "artifacts")
    store = TrajectoryStore(
        jsonl_path=tmp_path / "events.jsonl",
        artifacts=artifacts,
    )
    gateway = ToolGateway(runtime=ExecutionRuntime(artifact_store=artifacts))
    env = DefaultEnvironment(
        tool_bridge=GatewayToolBridge(gateway),
        event_sink=TrajectoryEnvironmentEventSink(store),
        execution_mode="isolated",
        tool_timeout_s=30,
    )
    env.init(
        session_id="real-podman-environment-smoke",
        actor="solver",
        branch_id="aime-60/solver",
        workspace_snapshot_ref="solver-workspace-ref",
        initial_observation="Compute six times seven.",
        episode_context={
            "runtime_snapshot": {
                "task": "aime-smoke",
                "model": "fixture",
                "budget": 30,
            },
        },
    )

    try:
        write_step = env.step(
            "<python_code>from pathlib import Path\n"
            "Path('/workspace/state.txt').write_text(str(6 * 7))</python_code>"
        )
        read_step = env.step(
            "<python_code>from pathlib import Path\n"
            "Path('/workspace/state.txt').read_text()</python_code>"
        )
        final_step = env.step("The answer is 42.")
    finally:
        env.close()

    assert write_step["done"] is False
    assert read_step["done"] is False
    assert '"stdout": "42\\n"' in read_step["observations"]
    assert final_step["done"] is True
    assert final_step["observations"] == "The answer is 42."
    assert _apollo_containers() == containers_before

    view = EnvironmentTrajectoryReader(store).read_session("real-podman-environment-smoke")
    responses = view.records_of(EnvironmentCaptureKind.TOOL_RESPONSE)
    assert len(responses) == 2
    response = responses[1].content
    assert response["attempted"] is True
    assert response["record"]["tool_id"] == "python"
    assert response["record"]["stdout"] == "42\n"
    assert response["record"]["sandbox"] == {
        "actor": "solver",
        "branch_id": "aime-60/solver",
        "effective_timeout_s": 30.0,
        "image": "python:3.11-slim",
        "kind": "podman",
        "mode": "isolated",
        "network": False,
        "profile": "podman_default",
        "resource_limits": {
            "cpu_seconds": 30,
            "memory_bytes": 2 * 1024 * 1024 * 1024,
            "max_processes": 64,
            "max_open_files": 4096,
            "cpus": None,
        },
        "session_id": "real-podman-environment-smoke",
        "workspace_root": "/workspace",
        "workspace_snapshot_ref": "solver-workspace-ref",
    }
    model_input = view.records_of(EnvironmentCaptureKind.MODEL_INPUT)[0]
    assert model_input.content["episode_context"]["runtime_snapshot"] == {
        "task": "aime-smoke",
        "model": "fixture",
        "budget": 30,
    }
    store.close()
