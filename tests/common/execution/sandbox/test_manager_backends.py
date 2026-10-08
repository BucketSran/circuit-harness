"""Unit tests for the non-podman SandboxManager backends.

`LocalSubprocessBackend` runs a real hardened subprocess (no docker/GPU).
`DockerBackend` is exercised without a docker daemon by bypassing `__init__`
(which would build a real CLISandboxEnv) and injecting a fake env + a patched
`subprocess.run`, so exec/copy_out/release/guards are covered structurally.
"""

from __future__ import annotations

import pathlib
import time

import pytest

from alphaapollo.common.execution.sandbox import docker as mgr
from alphaapollo.common.execution.sandbox.base import SandboxBackend
from alphaapollo.common.execution.sandbox.docker import (
    DockerBackend,
    LocalSubprocessBackend,
    SandboxError,
)


# --------------------------------------------------------------------------
# LocalSubprocessBackend (real subprocess)
# --------------------------------------------------------------------------
def test_local_exec_runs_python_and_captures_stdout() -> None:
    be = LocalSubprocessBackend()
    try:
        rec = be.exec("print(23 * 17)")
        assert rec.exit_code == 0
        assert rec.stdout.strip() == "391"
    finally:
        be.release()


def test_local_exec_nonzero_on_error_never_raises() -> None:
    be = LocalSubprocessBackend()
    try:
        rec = be.exec("raise ValueError('boom')")
        assert rec.exit_code != 0
        assert "ValueError" in rec.stderr
    finally:
        be.release()


def test_local_empty_code_is_reported_not_crashed() -> None:
    be = LocalSubprocessBackend()
    try:
        rec = be.exec("   ")
        assert rec.exit_code != 0
    finally:
        be.release()


def test_local_copy_out_copies_workdir_tree(tmp_path: pathlib.Path) -> None:
    be = LocalSubprocessBackend()
    try:
        # write a file into the backend workdir via exec
        be.exec("open('out.txt', 'w').write('local-data')")
        dest = tmp_path / "exported"
        be.copy_out("/ignored", str(dest))
        assert (dest / "out.txt").read_text() == "local-data"
    finally:
        be.release()


def test_local_release_is_idempotent_and_blocks_exec() -> None:
    be = LocalSubprocessBackend()
    workdir = pathlib.Path(be.workdir)
    be.release()
    be.release()  # idempotent, no raise
    assert not workdir.exists()
    with pytest.raises(SandboxError):
        be.exec("print(1)")
    with pytest.raises(SandboxError):
        be.copy_out("/x", "/y")


def test_local_backend_satisfies_protocol() -> None:
    assert isinstance(LocalSubprocessBackend(), SandboxBackend)


# --------------------------------------------------------------------------
# LocalSubprocessBackend timeout cleanup (real subprocess, real grandchild)
# --------------------------------------------------------------------------
# Sandboxed code that spawns a background writer. The writer appends one line
# every 0.25s to a file in the backend workdir, so counting lines measures
# whether it is still running after the parent command has been stopped.
_BACKGROUND_WRITER = """
import os, subprocess, sys, time

writer = '''
import sys, time
for _ in range(400):
    with open(sys.argv[1], "a") as fh:
        fh.write("x\\\\n")
    time.sleep(0.25)
'''
marker = os.path.join(os.getcwd(), "writer.log")
subprocess.Popen([sys.executable, "-c", writer, marker])
time.sleep(600)
"""


def _line_count(path: pathlib.Path) -> int:
    try:
        return len(path.read_text().splitlines())
    except FileNotFoundError:
        return 0


def test_local_exec_timeout_kills_the_process_group_not_only_the_child() -> None:
    """A timed-out command must stop what it spawned, not just its own process.

    ``subprocess.run(timeout=)`` and ``Popen.kill()`` signal the direct child
    only. A grandchild then keeps running on the host after the sandbox has
    reported a timeout, and — because it inherited stdout/stderr — keeps those
    pipes open, so draining them before killing the group blocks past the
    caller's own timeout. ``exec`` therefore starts the child with
    ``start_new_session=True`` and kills the group before it drains.

    A refactor back to ``subprocess.run(timeout=)`` reintroduces the orphan
    silently; this test is what catches it.
    """
    be = LocalSubprocessBackend(timeout=2)
    marker = pathlib.Path(be.workdir) / "writer.log"
    try:
        started = time.monotonic()
        rec = be.exec(_BACKGROUND_WRITER)
        elapsed = time.monotonic() - started

        assert elapsed < 20, "exec did not return near its own timeout"
        at_timeout = _line_count(marker)
        assert at_timeout > 0, "sandboxed code did not fork; the orphan cannot be exercised"

        time.sleep(1.5)
        assert _line_count(marker) == at_timeout, (
            f"grandchild kept writing after the timeout "
            f"({at_timeout} -> {_line_count(marker)} lines): the process group was orphaned"
        )
        assert rec.exit_code == -1
    finally:
        be.release()


def test_local_exec_timeout_still_reports_as_a_timeout() -> None:
    """Group cleanup must not change how a timeout reads to a caller.

    A caller distinguishing a timeout from a crash reads ``exit_code`` and
    ``stderr``; killing the group must leave both exactly as they were.
    """
    be = LocalSubprocessBackend(timeout=1)
    try:
        timed_out = be.exec("import time; time.sleep(30)")
        assert timed_out.exit_code == -1
        assert timed_out.stderr == "Code execution timed out after 1s"
        assert timed_out.stdout == ""

        crashed = be.exec("raise ValueError('boom')")
        assert crashed.exit_code != timed_out.exit_code
        assert "ValueError" in crashed.stderr
    finally:
        be.release()


def test_local_exec_is_reusable_across_calls_after_a_timeout() -> None:
    """Killing a group must reach the timed-out command and nothing else.

    Each ``exec`` leads its own session, so one call's cleanup cannot touch a
    later call — nor the interpreter that owns the backend.
    """
    be = LocalSubprocessBackend(timeout=1)
    try:
        assert be.exec("import time; time.sleep(30)").exit_code == -1
        after = be.exec("print(23 * 17)")
        assert after.exit_code == 0
        assert after.stdout.strip() == "391"
    finally:
        be.release()


def test_local_exec_normal_command_is_unaffected_by_group_cleanup() -> None:
    """Regression: a clean run keeps its streams, exit code, and file effects."""
    be = LocalSubprocessBackend(timeout=30)
    try:
        rec = be.exec(
            "import sys; print('out'); print('err', file=sys.stderr); "
            "open('kept.txt', 'w').write('data')"
        )
        assert rec.exit_code == 0
        assert rec.stdout.strip() == "out"
        assert rec.stderr.strip() == "err"
        assert (pathlib.Path(be.workdir) / "kept.txt").read_text() == "data"
    finally:
        be.release()


# --------------------------------------------------------------------------
# Network policy on the docker branch (no docker daemon: record the arguments)
# --------------------------------------------------------------------------
@pytest.fixture
def recorded_docker(monkeypatch):
    """Capture what SandboxManager would build, without a docker daemon."""
    built: dict = {}

    class _RecordingDockerBackend:
        def __init__(self, **kwargs):
            built.clear()
            built.update(kwargs)

    monkeypatch.setattr(mgr, "DockerBackend", _RecordingDockerBackend)
    return built


@pytest.mark.parametrize(
    ("kind", "kwargs", "expected"),
    [
        # A stated policy is honoured in both directions...
        ("cli", {"network": False}, "none"),
        ("cli", {"network": True}, None),
        # ...and an omitted one leaves each kind's own default: lean compiles
        # without egress, everything else keeps Docker's bridge.
        ("cli", {}, None),
    ],
)
def test_docker_branch_honours_the_network_policy(recorded_docker, kind, kwargs, expected) -> None:
    """Unlike the local backend, Docker can provide both answers — so it is never refused."""
    mgr.SandboxManager().acquire(kind, **kwargs)
    assert recorded_docker.get("network_mode") == expected


def test_docker_branch_refuses_two_declarations_of_one_policy(recorded_docker) -> None:
    """``network`` and ``network_mode`` both set this property, so stating both is an error.

    ``network_mode`` used to win by ``setdefault``: ``network=False`` with
    ``network_mode="host"`` ran on the host network and discarded the stricter
    request without a word. Precedence is the wrong answer for two deliberately
    stated inputs, so the combination is refused and names both sides.
    """
    with pytest.raises(SandboxError) as excinfo:
        mgr.SandboxManager().acquire("cli", network=False, network_mode="host")
    message = str(excinfo.value)
    assert "network=False" in message
    assert "network_mode='host'" in message
    assert recorded_docker == {}  # refused before the backend was built

    # A kind's derived default is not a declaration, so overriding it still works.
    mgr.SandboxManager().acquire("cli", network_mode="host")
    assert recorded_docker["network_mode"] == "host"


# --------------------------------------------------------------------------
# DockerBackend (no docker daemon: bypass __init__, inject fake env)
# --------------------------------------------------------------------------
class _FakeEnvResult:
    def __init__(self, stdout="", stderr="", exit_code=0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code

    @property
    def fs_diff(self):
        from dataclasses import dataclass, field

        @dataclass
        class _FsDiff:
            added: list = field(default_factory=list)
            modified: list = field(default_factory=list)
            deleted: list = field(default_factory=list)

        return _FsDiff()


class _FakeEnv:
    def __init__(self):
        self.executed: list[str] = []
        self.closed = False

    def execute(self, cmd):
        self.executed.append(cmd)
        return _FakeEnvResult(stdout="docker-out", exit_code=0)

    def close(self):
        self.closed = True


def _docker_backend_without_daemon(instance_id="abc123"):
    be = object.__new__(DockerBackend)
    be._env = _FakeEnv()
    be._instance_id = instance_id
    be._tool_id = "docker"
    be._network_mode = None
    be._released = False
    return be


def test_docker_exec_maps_envresult_to_record() -> None:
    be = _docker_backend_without_daemon()
    rec = be.exec("echo hi")
    assert rec.stdout == "docker-out"
    assert rec.exit_code == 0
    assert be._env.executed == ["echo hi"]


def test_docker_copy_out_builds_docker_cp_and_fails_loud(monkeypatch) -> None:
    calls = {}

    class _Completed:
        def __init__(self, rc, stderr=""):
            self.returncode = rc
            self.stderr = stderr

    def fake_run(argv, capture_output, text):
        calls["argv"] = argv
        return _Completed(0)

    monkeypatch.setattr(mgr.subprocess, "run", fake_run)
    be = _docker_backend_without_daemon(instance_id="xyz")
    be.copy_out("/workspace", "/host/dest")
    assert calls["argv"] == ["docker", "cp", "openclaw_xyz:/workspace", "/host/dest"]

    def fail_run(argv, capture_output, text):
        return _Completed(1, stderr="no such path")

    monkeypatch.setattr(mgr.subprocess, "run", fail_run)
    with pytest.raises(SandboxError, match="failed to copy"):
        be.copy_out("/nope", "/host/dest")


def test_docker_release_idempotent_and_blocks_exec() -> None:
    be = _docker_backend_without_daemon()
    be.release()
    assert be._env.closed is True
    be.release()  # idempotent
    with pytest.raises(SandboxError):
        be.exec("echo hi")
    with pytest.raises(SandboxError):
        be.copy_out("/x", "/y")


def test_docker_backend_satisfies_protocol() -> None:
    assert isinstance(_docker_backend_without_daemon(), SandboxBackend)
