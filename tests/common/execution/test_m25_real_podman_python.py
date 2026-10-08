"""Real-container effect sample for the M2.5 v2 Python compatibility adapter.

Set ``ALPHAAPOLLO_RUN_REAL_PODMAN=1`` on a dedicated rootless-Podman host.
Ordinary CI skips this opt-in evidence test.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from alphaapollo.common.environment.default.projection import format_tool_response, normalize
from alphaapollo.common.execution import (
    ExecutionContext,
    ToolGateway,
    ToolGatewayResult,
    ToolRequest,
    ToolResponse,
)


def _podman(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["podman", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _require_rootless_podman() -> None:
    if os.environ.get("ALPHAAPOLLO_RUN_REAL_PODMAN") != "1":
        pytest.skip("set ALPHAAPOLLO_RUN_REAL_PODMAN=1 on a dedicated rootless-Podman host")
    if shutil.which("podman") is None:
        pytest.skip("Podman is not installed")
    info = _podman("info", "--format", "{{.Host.Security.Rootless}}")
    if info.returncode != 0:
        pytest.skip(f"Podman is unavailable: {info.stderr.strip()}")
    if info.stdout.strip().lower() != "true":
        pytest.skip("this integration test requires rootless Podman")


def _apollo_containers() -> set[str]:
    result = _podman(
        "ps",
        "-a",
        "--filter",
        "name=apollo_pod_",
        "--format",
        "{{.Names}}",
    )
    assert result.returncode == 0, result.stderr
    return {name for name in result.stdout.splitlines() if name}


def _invoke_tagged(
    gateway: ToolGateway,
    source: str,
    context: ExecutionContext,
) -> tuple[ToolRequest, ToolGatewayResult]:
    request = _tagged_request(source)
    return request, gateway.invoke(request, context)


def _tagged_request(source: str) -> ToolRequest:
    request = normalize(f"<python_code>{source}</python_code>")
    assert isinstance(request, ToolRequest)
    return request


def test_real_gateway_python_round_trip_for_solver_and_verifier() -> None:
    """Exercise both actors through the shared gateway and isolated runtime."""
    _require_rootless_podman()
    containers_before = _apollo_containers()
    gateway = ToolGateway()
    solver_request, solver = _invoke_tagged(
        gateway,
        "6 * 7",
        ExecutionContext(
            session_id="m25-real",
            branch_id="candidate/solver",
            actor="solver",
            mode="isolated",
        ),
    )
    verifier_request, verifier = _invoke_tagged(
        gateway,
        "sqrt(81)",
        ExecutionContext(
            session_id="m25-real",
            branch_id="candidate/verifier",
            actor="verifier",
            mode="isolated",
        ),
    )
    _, timeout = _invoke_tagged(
        gateway,
        "import time\ntime.sleep(2)",
        ExecutionContext(
            session_id="m25-real",
            branch_id="candidate/solver",
            actor="solver",
            mode="isolated",
            timeout_s=0.5,
        ),
    )

    assert isinstance(solver.outcome, ToolResponse)
    assert isinstance(verifier.outcome, ToolResponse)
    assert isinstance(timeout.outcome, ToolResponse)
    assert solver.record is not None
    assert verifier.record is not None
    assert timeout.record is not None
    assert solver.outcome.stdout == "42\n"
    assert verifier.outcome.stdout == "9.0\n"
    assert timeout.outcome.exit_code == 124
    assert solver.record.sandbox is not None
    assert verifier.record.sandbox is not None
    assert timeout.record.sandbox is not None
    assert solver.record.sandbox["actor"] == "solver"
    assert solver.record.sandbox["branch_id"] == "candidate/solver"
    assert verifier.record.sandbox["actor"] == "verifier"
    assert verifier.record.sandbox["branch_id"] == "candidate/verifier"
    assert timeout.record.sandbox["effective_timeout_s"] == 0.5

    observation = format_tool_response(solver.outcome)
    payload = json.loads(
        observation.removeprefix("<tool_response>\n").removesuffix("\n</tool_response>")
    )

    print("\nM2.5 solver action: <python_code>6 * 7</python_code>")
    print("M2.5 solver record:", solver.record.model_dump_json())
    print("M2.5 verifier record:", verifier.record.model_dump_json())
    print("M2.5 timeout record:", timeout.record.model_dump_json())
    print("M2.5 model observation:", observation)

    assert payload == {
        "ok": True,
        "call_id": solver_request.call_id,
        "tool_id": "python",
        "stdout": "42\n",
        "stderr": "",
        "exit_code": 0,
        "artifacts": [],
    }
    assert verifier.outcome.call_id == verifier_request.call_id
    assert _apollo_containers() == containers_before


def test_real_sessions_persist_per_actor_workspace_and_remain_isolated() -> None:
    """Write/read across calls while keeping Solver and Verifier files separate."""
    _require_rootless_podman()
    containers_before = _apollo_containers()
    gateway = ToolGateway()
    solver_context = ExecutionContext(
        session_id="m25-session-real",
        branch_id="candidate/solver",
        actor="solver",
        mode="isolated",
        workspace_snapshot_ref="workspace-solver",
        timeout_s=10,
    )
    verifier_context = ExecutionContext(
        session_id="m25-session-real",
        branch_id="candidate/verifier",
        actor="verifier",
        mode="isolated",
        workspace_snapshot_ref="workspace-verifier",
        timeout_s=10,
    )

    with (
        gateway.open_session(solver_context) as solver,
        gateway.open_session(verifier_context) as verifier,
    ):
        solver_write = solver.invoke(
            _tagged_request(
                "from pathlib import Path\nPath('/workspace/actor.txt').write_text('solver-only')"
            )
        )
        solver_read = solver.invoke(
            _tagged_request("from pathlib import Path\nPath('/workspace/actor.txt').read_text()")
        )
        verifier_cannot_read_solver = verifier.invoke(
            _tagged_request("from pathlib import Path\nPath('/workspace/actor.txt').read_text()")
        )
        verifier_write = verifier.invoke(
            _tagged_request(
                "from pathlib import Path\nPath('/workspace/actor.txt').write_text('verifier-only')"
            )
        )
        verifier_read = verifier.invoke(
            _tagged_request("from pathlib import Path\nPath('/workspace/actor.txt').read_text()")
        )
        solver_read_again = solver.invoke(
            _tagged_request("from pathlib import Path\nPath('/workspace/actor.txt').read_text()")
        )

        assert isinstance(solver_write.outcome, ToolResponse)
        assert isinstance(solver_read.outcome, ToolResponse)
        assert isinstance(verifier_cannot_read_solver.outcome, ToolResponse)
        assert isinstance(verifier_write.outcome, ToolResponse)
        assert isinstance(verifier_read.outcome, ToolResponse)
        assert isinstance(solver_read_again.outcome, ToolResponse)
        assert solver_write.outcome.exit_code == 0
        assert solver_read.outcome.stdout == "solver-only\n"
        assert verifier_cannot_read_solver.outcome.exit_code != 0
        assert "FileNotFoundError" in verifier_cannot_read_solver.outcome.stderr
        assert verifier_write.outcome.exit_code == 0
        assert verifier_read.outcome.stdout == "verifier-only\n"
        assert solver_read_again.outcome.stdout == "solver-only\n"
        assert solver_read.record is not None
        assert verifier_read.record is not None
        assert solver_read.record.sandbox is not None
        assert verifier_read.record.sandbox is not None
        assert solver_read.record.sandbox["workspace_snapshot_ref"] == "workspace-solver"
        assert verifier_read.record.sandbox["workspace_snapshot_ref"] == ("workspace-verifier")
        assert len(_apollo_containers() - containers_before) == 2

    assert _apollo_containers() == containers_before
