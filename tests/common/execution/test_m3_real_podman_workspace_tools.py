from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from alphaapollo.common.execution import ExecutionContext, ToolGateway, ToolRequest, ToolResponse


def _podman(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["podman", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _require_rootless_podman() -> None:
    if os.environ.get("ALPHAAPOLLO_RUN_REAL_PODMAN") != "1":
        pytest.skip("set ALPHAAPOLLO_RUN_REAL_PODMAN=1 on a dedicated rootless-Podman host")
    if shutil.which("podman") is None:
        pytest.skip("podman CLI is not installed")
    info = _podman("info", "--format", "{{.Host.Security.Rootless}}")
    if info.returncode != 0:
        pytest.skip(f"podman is unavailable: {info.stderr.strip()}")
    if info.stdout.strip().lower() != "true":
        pytest.skip("real workspace-tool test requires rootless Podman")


def _apollo_containers() -> set[str]:
    result = _podman("ps", "-a", "--format", "{{.Names}}")
    if result.returncode != 0:
        return set()
    return {name for name in result.stdout.splitlines() if name.startswith("apollo_pod_")}


def _request(tool_id: str, **arguments: object) -> ToolRequest:
    return ToolRequest(
        call_id=f"m3-{tool_id}",
        tool_id=tool_id,
        arguments=arguments,
        source="openai_tool_call",
    )


def _response(result: object) -> ToolResponse:
    assert hasattr(result, "outcome")
    outcome = result.outcome
    assert isinstance(outcome, ToolResponse)
    return outcome


def test_real_podman_session_runs_six_tools_and_isolates_actors() -> None:
    _require_rootless_podman()
    containers_before = _apollo_containers()
    gateway = ToolGateway()
    solver_context = ExecutionContext(
        session_id="m3-workspace-tools",
        branch_id="candidate/solver",
        actor="solver",
        mode="isolated",
        timeout_s=20,
    )
    verifier_context = ExecutionContext(
        session_id="m3-workspace-tools",
        branch_id="candidate/verifier",
        actor="verifier",
        mode="isolated",
        timeout_s=20,
    )

    with (
        gateway.open_session(solver_context) as solver,
        gateway.open_session(verifier_context) as verifier,
    ):
        written = solver.invoke(
            _request(
                "write",
                path="src/example.py",
                content="value = 1\nprint(value)\n",
            )
        )
        read = solver.invoke(_request("read", path="src/example.py"))
        edited = solver.invoke(
            _request(
                "edit",
                path="src/example.py",
                edits=[{"oldText": "value = 1", "newText": "value = 42"}],
            )
        )
        grep = solver.invoke(_request("grep", pattern="value = 42", path=".", literal=True))
        found = solver.invoke(_request("find", pattern="**/*.py", path="."))
        listed = solver.invoke(_request("ls", path="src"))
        isolated = verifier.invoke(_request("read", path="src/example.py"))

        assert _response(written).exit_code == 0
        assert _response(read).stdout == "value = 1\nprint(value)\n"
        assert _response(edited).exit_code == 0
        assert "src/example.py:1: value = 42" in _response(grep).stdout
        assert _response(found).stdout == "src/example.py"
        assert _response(listed).stdout == "example.py"
        assert _response(isolated).exit_code == 2
        assert isolated.record is not None
        assert isolated.record.sandbox is not None
        assert isolated.record.sandbox["tool_error"]["code"] == "path_not_found"
        assert len(_apollo_containers() - containers_before) == 2

    assert _apollo_containers() == containers_before


def test_real_podman_resolved_symlink_escape_is_rejected() -> None:
    _require_rootless_podman()
    containers_before = _apollo_containers()
    gateway = ToolGateway()
    context = ExecutionContext(
        session_id="m3-symlink-escape",
        actor="solver",
        mode="isolated",
        timeout_s=20,
    )

    with gateway.open_session(context) as session:
        symlink = session.invoke(_request("bash", command="ln -s /tmp /workspace/escape"))
        escaped_read = session.invoke(_request("read", path="escape/secret.txt"))
        escaped_write = session.invoke(
            _request("write", path="escape/created.txt", content="blocked")
        )

        assert _response(symlink).exit_code == 0
        assert _response(escaped_read).exit_code == 2
        assert _response(escaped_write).exit_code == 2
        assert escaped_read.record is not None
        assert escaped_write.record is not None
        assert escaped_read.record.sandbox is not None
        assert escaped_write.record.sandbox is not None
        assert escaped_read.record.sandbox["tool_error"]["code"] == "path_forbidden"
        assert escaped_write.record.sandbox["tool_error"]["code"] == "path_forbidden"

    assert _apollo_containers() == containers_before
