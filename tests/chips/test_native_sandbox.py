"""Real outer Codex sandbox checks, without a model call or private user data."""

import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from circuit_harness.execution.native_sandbox import NativeSandbox


@pytest.mark.skipif(shutil.which("codex") is None, reason="Codex CLI not installed")
def test_outer_sandbox_allows_public_work_but_denies_private_reads(tmp_path):
    workspace = tmp_path / "public"
    workspace.mkdir()
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    secret = hidden / "fixture.txt"
    secret.write_text("private-fixture")
    (workspace / "escape").symlink_to(secret)
    sandbox = NativeSandbox(
        codex=Path(shutil.which("codex")),
        directory=tmp_path / "policy",
        workspace=workspace,
        protected_paths=(hidden,),
    )
    command, overrides = sandbox.command(
        [
            "/bin/sh",
            "-c",
            'printf public > visible.txt; if cat "$1"; then exit 9; fi; '
            "if cat escape; then exit 10; fi; test -f visible.txt",
            "probe",
            str(secret),
        ]
    )
    result = subprocess.run(
        command,
        cwd=workspace,
        env={**os.environ, **overrides},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert (workspace / "visible.txt").read_text() == "public"
    assert "private-fixture" not in result.stdout
    assert sandbox.receipt["enforcement"] == "outer_codex_sandbox"
    assert sandbox.probe()["access_probe"] == "synthetic_file_boundary_passed"


def test_sandbox_refuses_to_grant_access_to_verifier_material(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    with pytest.raises(ValueError, match="protected"):
        NativeSandbox(
            codex=Path("/usr/bin/false"),
            directory=tmp_path / "policy",
            workspace=public,
            readonly_paths=(tmp_path,),
            protected_paths=(private,),
        )


@pytest.mark.skipif(shutil.which("codex") is None, reason="Codex CLI not installed")
def test_outer_sandbox_denies_direct_network_access(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    sandbox = NativeSandbox(
        codex=Path(shutil.which("codex")),
        directory=tmp_path / "policy",
        workspace=public,
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(0.1)
        port = listener.getsockname()[1]
        command, overrides = sandbox.command(
            [
                "/usr/bin/curl",
                "--noproxy",
                "*",
                "--connect-timeout",
                "1",
                f"http://127.0.0.1:{port}/",
            ]
        )
        result = subprocess.run(
            command,
            cwd=public,
            env={**os.environ, **overrides},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        with pytest.raises(TimeoutError):
            listener.accept()
