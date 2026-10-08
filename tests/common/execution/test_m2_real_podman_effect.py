"""Real-container effect sample for the M2 isolated bash execution path.

This test is intended for a dedicated rootless-Podman host. Set
``ALPHAAPOLLO_RUN_REAL_PODMAN=1`` to opt in; ordinary CI skips it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import asdict
from json import dumps

import pytest

from alphaapollo.common.execution import (
    ArtifactStore,
    CancellationToken,
    ExecutionContext,
    ExecutionRuntime,
    ToolRequest,
)
from alphaapollo.common.execution.tools.schemas import ToolCallRecord


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


def test_real_rootless_podman_effect_sample() -> None:
    """Show one concrete ToolRequest -> /workspace -> ToolCallRecord result."""
    _require_rootless_podman()
    containers_before = _apollo_containers()
    request = ToolRequest(
        call_id="m2-effect-sample",
        tool_id="bash",
        arguments={"command": 'pwd; python -c "print(23 * 17)"'},
    )

    result = ExecutionRuntime().invoke(
        request,
        ExecutionContext(session_id="m2-effect-sample"),
    )

    print("\nM2 input:", dumps(asdict(request), ensure_ascii=False))
    if isinstance(result, ToolCallRecord):
        print("M2 output:", result.model_dump_json())

    assert isinstance(result, ToolCallRecord)
    assert result.exit_code == 0
    assert result.stdout == "/workspace\n391\n"
    assert result.stderr == ""
    assert result.sandbox == {
        "kind": "podman",
        "workspace_root": "/workspace",
        "network": False,
        "profile": "podman_default",
        "image": "python:3.11-slim",
        "resource_limits": {
            "cpu_seconds": 30,
            "memory_bytes": 2 * 1024 * 1024 * 1024,
            "max_processes": 64,
            "max_open_files": 4096,
            "cpus": None,
        },
        "effective_timeout_s": 30,
        "session_id": "m2-effect-sample",
        "branch_id": "main",
        "actor": "solver",
        "mode": "default",
    }
    assert result.cost.tool_calls == 1
    assert _apollo_containers() == containers_before


def test_real_rootless_podman_keeps_pi_tail_and_stores_pre_truncation_capture(
    tmp_path,
) -> None:
    """Exercise Pi's 2000-line bash tail through the real AlphaApollo backend."""
    _require_rootless_podman()
    containers_before = _apollo_containers()
    store = ArtifactStore(tmp_path / "artifacts")
    request = ToolRequest(
        call_id="m2-pi-tail-sample",
        tool_id="bash",
        arguments={
            "command": """python -c 'for i in range(2105): print(f"line-{i}")'""",
        },
    )

    result = ExecutionRuntime(artifact_store=store).invoke(
        request,
        ExecutionContext(session_id="m2-pi-tail-sample"),
    )

    assert isinstance(result, ToolCallRecord)
    print("\nPi-tail output preview:", result.stdout[:120], "...", result.stdout[-160:])
    assert result.exit_code == 0
    assert "line-0\n" not in result.stdout
    assert result.stdout.startswith("line-105\n")
    assert "line-2104" in result.stdout
    assert "[stdout truncated: showing lines 106-2105 of 2105" in result.stdout
    assert len(result.artifacts) == 1
    captured_output = store.get(result.artifacts[0]).decode("utf-8")
    assert captured_output.startswith("line-0\n")
    assert captured_output.endswith("line-2104\n")
    assert result.cost.artifacts_produced == 1
    assert result.sandbox is not None
    truncation = result.sandbox["output_truncation"]["stdout"]
    assert truncation["total_lines"] == 2_105
    assert truncation["output_lines"] == 2_000
    assert _apollo_containers() == containers_before


def test_real_rootless_podman_streams_then_actively_cancels() -> None:
    _require_rootless_podman()
    containers_before = _apollo_containers()
    token = CancellationToken()
    chunks = []

    def on_output(chunk) -> None:
        chunks.append(chunk)
        if chunk.stream == "stdout" and "ready" in chunk.text:
            token.cancel()

    started_at = time.monotonic()
    result = ExecutionRuntime().invoke(
        ToolRequest(
            call_id="m2-stream-cancel",
            tool_id="bash",
            arguments={
                "command": """python -c 'import time; print("ready", flush=True); time.sleep(30)'"""
            },
        ),
        ExecutionContext(session_id="m2-stream-cancel"),
        on_output=on_output,
        cancellation=token,
    )
    elapsed = time.monotonic() - started_at

    assert isinstance(result, ToolCallRecord)
    assert any(chunk.stream == "stdout" and "ready" in chunk.text for chunk in chunks)
    assert result.exit_code == 130
    assert "command cancelled" in result.stderr
    assert elapsed < 10
    assert _apollo_containers() == containers_before


def test_real_rootless_podman_streams_multimegabyte_output_to_artifact(tmp_path) -> None:
    _require_rootless_podman()
    containers_before = _apollo_containers()
    store = ArtifactStore(tmp_path / "artifacts")
    size = 2 * 1024 * 1024

    result = ExecutionRuntime(artifact_store=store).invoke(
        ToolRequest(
            call_id="m2-large-stream",
            tool_id="bash",
            arguments={
                "command": (f"head -c {size} /dev/zero | tr '\\0' x; printf '\\nactual-tail\\n'")
            },
        ),
        ExecutionContext(session_id="m2-large-stream"),
    )

    assert isinstance(result, ToolCallRecord)
    assert result.exit_code == 0
    assert len(result.artifacts) == 1
    full_output = store.get(result.artifacts[0])
    assert len(full_output) == size + len(b"\nactual-tail\n")
    assert full_output.endswith(b"\nactual-tail\n")
    assert result.stdout.startswith("actual-tail")
    assert "[stdout truncated:" in result.stdout
    assert _apollo_containers() == containers_before
