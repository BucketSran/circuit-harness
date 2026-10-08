"""GPU/Podman-free tests for podman_cli (argv construction + injectable runner).

A FAKE runner records every podman argv and returns a canned PodmanResult, so these
tests assert the COMMANDS are built correctly — no Podman binary required. The
security boundary is asserted directly: --network none by default, and NEVER a
-v/--volume host mount.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from alphaapollo.common.execution.sandbox._podman import runner as podman_runner
from alphaapollo.common.execution.sandbox.base import CancellationToken
from alphaapollo.common.execution.sandbox.podman import (
    WORKSPACE,
    PodmanCliError,
    PodmanResult,
    container_exists,
    copy_out,
    exec_argv,
    exec_in,
    inspect_container_image_digest,
    kill_container,
    remove_container,
    run_container,
    streaming_runner,
)


class _FakeRunner:
    """Records every argv it is called with; returns a configurable result."""

    def __init__(self, result: PodmanResult | None = None) -> None:
        self.calls: list[list[str]] = []
        self._result = result or PodmanResult(stdout="", stderr="", exit_code=0)

    def __call__(self, argv):
        self.calls.append(list(argv))
        return self._result

    @property
    def last(self) -> list[str]:
        return self.calls[-1]


# --- run_container -----------------------------------------------------------
def test_run_container_argv_detached_named_no_network_no_mount() -> None:
    r = _FakeRunner(PodmanResult(stdout="abc123", stderr="", exit_code=0))
    result = run_container("box1", "python:3.11-slim", runner=r)
    argv = r.last
    assert argv[:2] == ["podman", "run"]
    assert "-d" in argv
    assert argv[argv.index("--name") + 1] == "box1"
    # security: no network by default
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    # security: NEVER a host mount
    assert "-v" not in argv and "--volume" not in argv
    # long-lived container creates the stable in-container workspace first
    assert argv[-3:] == [
        "sh",
        "-c",
        f"mkdir -p {WORKSPACE} && exec sleep infinity",
    ]
    assert argv[-4] == "python:3.11-slim"
    assert result.stdout == "abc123"


def test_run_container_network_true_omits_network_none() -> None:
    r = _FakeRunner()
    run_container("box1", "img", network=True, runner=r)
    assert "--network" not in r.last


def test_run_container_resource_caps_emitted() -> None:
    r = _FakeRunner()
    run_container("box1", "img", memory_bytes=2 * 1024**3, cpus=1.5, pids_limit=64, runner=r)
    argv = r.last
    assert argv[argv.index("--memory") + 1] == f"{2 * 1024**3}b"
    assert argv[argv.index("--cpus") + 1] == "1.5"
    assert argv[argv.index("--pids-limit") + 1] == "64"


def test_run_container_rejects_empty_image_and_name() -> None:
    r = _FakeRunner()
    with pytest.raises(PodmanCliError, match="image"):
        run_container("box1", "", runner=r)
    with pytest.raises(PodmanCliError, match="name"):
        run_container("", "img", runner=r)


def test_run_container_rejects_nonpositive_caps() -> None:
    r = _FakeRunner()
    with pytest.raises(PodmanCliError, match="memory_bytes"):
        run_container("b", "img", memory_bytes=0, runner=r)
    with pytest.raises(PodmanCliError, match="cpus"):
        run_container("b", "img", cpus=0, runner=r)
    with pytest.raises(PodmanCliError, match="pids_limit"):
        run_container("b", "img", pids_limit=-1, runner=r)


# --- exec_in -----------------------------------------------------------------
def test_exec_in_passes_command_as_single_argv_element() -> None:
    r = _FakeRunner(PodmanResult(stdout="391\n", stderr="", exit_code=0))
    dangerous = "print(1); rm -rf /  # metachars stay one arg"
    result = exec_in("box1", dangerous, runner=r)
    argv = r.last
    assert argv[:7] == ["podman", "exec", "--workdir", WORKSPACE, "box1", "bash", "-c"]
    # the whole command is ONE argv element — no host-side shell splitting/injection
    assert argv[7] == dangerous
    assert len(argv) == 8
    assert result.stdout == "391\n"


def test_exec_in_rejects_empty_name() -> None:
    with pytest.raises(PodmanCliError, match="name"):
        exec_in("", "echo hi", runner=_FakeRunner())


def test_inspect_container_image_digest_uses_running_container() -> None:
    runner = _FakeRunner(PodmanResult(stdout=f"sha256:{'a' * 64}\n", stderr="", exit_code=0))

    result = inspect_container_image_digest("box1", runner=runner)

    assert result.stdout.strip() == f"sha256:{'a' * 64}"
    assert runner.last == [
        "podman",
        "container",
        "inspect",
        "box1",
        "--format",
        "{{.ImageDigest}}",
    ]


def test_exec_in_propagates_nonzero_exit() -> None:
    r = _FakeRunner(PodmanResult(stdout="", stderr="boom", exit_code=1))
    result = exec_in("box1", "false", runner=r)
    assert result.exit_code == 1
    assert result.stderr == "boom"


def test_exec_argv_passes_timeout_to_production_runner(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def run(argv, *, timeout_seconds=None):
        captured.update(argv=list(argv), timeout_seconds=timeout_seconds)
        return PodmanResult(stdout="", stderr="", exit_code=124, timed_out=True)

    monkeypatch.setattr(
        "alphaapollo.common.execution.sandbox._podman.cli.default_runner",
        run,
    )
    result = exec_argv("box1", ["sleep", "2"], timeout_seconds=0.1, runner=run)

    assert result.timed_out is True
    assert captured["timeout_seconds"] == 0.1
    assert captured["argv"][-2:] == ["sleep", "2"]


def test_streaming_runner_emits_live_chunks_and_can_be_cancelled() -> None:
    token = CancellationToken()
    chunks = []

    def on_output(chunk) -> None:
        chunks.append(chunk)
        if "ready" in chunk.text:
            token.cancel()

    result = streaming_runner(
        [
            sys.executable,
            "-c",
            "import time; print('ready', flush=True); time.sleep(30)",
        ],
        timeout_seconds=10,
        on_output=on_output,
        cancellation=token,
    )

    assert result.cancelled is True
    assert result.timed_out is False
    assert result.exit_code == 130
    assert any(chunk.stream == "stdout" and "ready" in chunk.text for chunk in chunks)
    assert result.stdout == "ready\n"
    assert result.stdout_capture_path is not None
    result.stdout_capture_path.unlink()
    assert result.stderr_capture_path is not None
    result.stderr_capture_path.unlink()


def _spawn_background_writer_script(marker) -> str:
    """Source for a command that leaves a writer running and then idles."""
    writer = (
        "import sys, time\n"
        "for _ in range(400):\n"
        "    with open(sys.argv[1], 'a') as fh:\n"
        "        fh.write('x\\n')\n"
        "    time.sleep(0.25)\n"
    )
    return (
        f"import subprocess, sys, time;"
        f"subprocess.Popen([sys.executable, '-c', {writer!r}, {str(marker)!r}]);"
        f"time.sleep(600)"
    )


def _assert_writer_stopped(marker, label: str) -> None:
    at_stop = len(marker.read_text().splitlines()) if marker.exists() else 0
    if at_stop == 0:
        pytest.skip("the command could not fork on this host; nothing to orphan")
    time.sleep(1.5)
    assert len(marker.read_text().splitlines()) == at_stop, (
        f"background writer survived the timeout in {label}: the process group was orphaned"
    )


def test_default_runner_timeout_kills_the_process_group(tmp_path) -> None:
    """A timed-out podman CLI invocation must not leave host-side helpers behind.

    ``default_runner`` is the production path for every bounded podman command
    (``cli.py`` routes the wall-clock timeout here), and ``subprocess.run(timeout=)``
    signals only the direct child. The argv is a plain interpreter here because the
    runner is argv-generic; the orphan it reproduces is the same one a timed-out
    ``podman run`` would leave.
    """
    marker = tmp_path / "writer.log"
    script = _spawn_background_writer_script(marker)

    started = time.monotonic()
    result = podman_runner.default_runner([sys.executable, "-c", script], timeout_seconds=2)
    elapsed = time.monotonic() - started

    assert result.timed_out is True
    assert result.exit_code == 124
    assert "command timed out after 2s" in result.stderr
    assert elapsed < 20
    _assert_writer_stopped(marker, "default_runner")


def test_default_runner_normal_command_is_unaffected() -> None:
    """Regression: a clean invocation keeps its streams and exit code."""
    result = podman_runner.default_runner(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
        timeout_seconds=30,
    )
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"


def test_streaming_runner_timeout_kills_the_process_group(tmp_path) -> None:
    """A timed-out command must stop the children it spawned, not just itself.

    ``streaming_runner`` is the generic (non-container) path, so nothing else
    cleans up after it: signalling the direct child alone leaves a background
    writer running on the host and still appending to the file being drained.
    The command is therefore spawned with ``start_new_session=True`` and the
    group is signalled as a unit, before the drain.

    A refactor back to ``process.terminate()``/``process.kill()`` reintroduces
    the orphan silently; this test is what catches it.
    """
    marker = tmp_path / "writer.log"
    script = _spawn_background_writer_script(marker)

    started = time.monotonic()
    result = streaming_runner([sys.executable, "-c", script], timeout_seconds=2)
    elapsed = time.monotonic() - started

    assert result.timed_out is True
    assert result.exit_code == 124
    assert elapsed < 20, "the drain waited on a writer the timeout had not stopped"
    _assert_writer_stopped(marker, "streaming_runner")
    for path in (result.stdout_capture_path, result.stderr_capture_path):
        if path is not None:
            path.unlink(missing_ok=True)


def test_streaming_runner_bounds_drain_on_background_writer(monkeypatch) -> None:
    """A detached background writer keeps output flowing forever — drain must be bounded.

    Without the wall-clock bound, ``follow_output_files`` never exits and
    ``output_reader.join()`` hangs the episode (review BLOCKING B1). The test runs
    ``streaming_runner`` on its own thread so a regression fails instead of hanging CI.
    """
    monkeypatch.setattr(podman_runner, "_OUTPUT_DRAIN_TIMEOUT", 0.5)
    child_writer = (
        "import time\n"
        "e = time.time() + 2\n"
        "while time.time() < e:\n"
        "    print('bg', flush=True)\n"
        "    time.sleep(0.01)\n"
    )
    script = (
        f"import subprocess, sys, time;"
        f"subprocess.Popen([sys.executable, '-c', {child_writer!r}], "
        f"stdout=sys.stdout, stderr=sys.stderr, start_new_session=True);"
        f"print('main done', flush=True);"
        f"time.sleep(2)"
    )
    holder: dict[str, object] = {}

    def run() -> None:
        holder["result"] = streaming_runner([sys.executable, "-c", script])

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=10)
    assert worker.is_alive() is False, "streaming_runner hung on post-exit drain"
    result = holder["result"]
    assert isinstance(result, PodmanResult)
    assert result.timed_out is False
    assert result.cancelled is False
    assert "main done" in result.stdout
    assert result.stdout_capture_path is not None
    result.stdout_capture_path.unlink(missing_ok=True)
    assert result.stderr_capture_path is not None
    result.stderr_capture_path.unlink(missing_ok=True)


def test_streaming_podman_exec_drain_is_bounded_on_background_writer(monkeypatch) -> None:
    """The exec drain loop stops on a wall-clock budget even if a writer never stops.

    ``_streaming_podman_exec`` drains remote output after the command exits. A
    background sandbox process keeps the remote files growing, so ``poll_once()``
    never returns False; the drain must be bounded (review BLOCKING B1) or the
    episode hangs forever. This test fakes the reader so every poll makes progress.
    """
    monkeypatch.setattr(podman_runner, "_OUTPUT_DRAIN_TIMEOUT", 0.2)

    class FakeProcess:
        """Doubles both the exec client and the bounded output readers.

        Both are ``Popen`` objects now that the readers are spawned into their own
        session rather than run through ``subprocess.run``; the reader half is the
        context-manager plus ``communicate`` protocol. Every chunk read reports
        progress, which is what keeps the drain loop from ever finishing on its own.
        """

        def __enter__(self) -> FakeProcess:
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            return b"bg-line\n", b""

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(podman_runner.subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr(podman_runner, "default_runner", lambda *a, **k: None)

    started = time.monotonic()
    result = podman_runner._streaming_podman_exec(
        "box",
        "echo START; (while true; do echo bg; sleep 0.01; done) & echo DONE",
        timeout_seconds=None,
        on_output=None,
        cancellation=None,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 5, "drain loop was not wall-clock bounded"
    assert result.timed_out is False
    assert result.cancelled is False
    assert result.exit_code == 0


# --- remove_container --------------------------------------------------------
def test_remove_container_argv() -> None:
    r = _FakeRunner()
    remove_container("box1", runner=r)
    assert r.last == ["podman", "rm", "-f", "box1"]


def test_remove_container_rejects_empty_name() -> None:
    with pytest.raises(PodmanCliError, match="name"):
        remove_container("", runner=_FakeRunner())


# --- kill_container ----------------------------------------------------------
def test_kill_container_argv() -> None:
    r = _FakeRunner()
    kill_container("box1", runner=r)
    assert r.last == ["podman", "kill", "box1"]


# --- container_exists --------------------------------------------------------
def test_container_exists_true_on_zero_exit() -> None:
    r = _FakeRunner(PodmanResult(stdout="", stderr="", exit_code=0))
    assert container_exists("box1", runner=r) is True
    assert r.last == ["podman", "container", "exists", "box1"]


def test_container_exists_false_on_nonzero_exit() -> None:
    r = _FakeRunner(PodmanResult(stdout="", stderr="", exit_code=1))
    assert container_exists("box1", runner=r) is False


# --- copy_out ----------------------------------------------------------------
def test_copy_out_builds_podman_cp_argv() -> None:
    r = _FakeRunner(PodmanResult(stdout="", stderr="", exit_code=0))
    copy_out("box1", "/workspace", "/host/dest", runner=r)
    assert r.last == ["podman", "cp", "box1:/workspace", "/host/dest"]


def test_copy_out_returns_nonzero_not_raise() -> None:
    r = _FakeRunner(PodmanResult(stdout="", stderr="no such path", exit_code=1))
    res = copy_out("box1", "/nope", "/host/dest", runner=r)
    assert res.exit_code == 1


def test_copy_out_rejects_empty_args() -> None:
    for a, b, c in [("", "/w", "/d"), ("box", "", "/d"), ("box", "/w", "")]:
        with pytest.raises(PodmanCliError):
            copy_out(a, b, c, runner=_FakeRunner())


# --- resource caps: cpu_seconds / max_open_files / cpus (Yiming P1+P2) ---------
def test_run_container_emits_ulimit_cpu_and_nofile() -> None:
    r = _FakeRunner()
    run_container(
        "box",
        "img",
        network=False,
        cpu_seconds=3,
        max_open_files=5000,
        runner=r,
    )
    assert "--ulimit" in r.last
    assert "cpu=3" in r.last
    assert "nofile=5000" in r.last


def test_nofile_uses_the_profile_value_without_a_hidden_override() -> None:
    r = _FakeRunner()
    run_container("box", "img", max_open_files=16, runner=r)
    assert "nofile=16" in r.last


def test_cpus_rate_cap_still_emitted() -> None:
    r = _FakeRunner()
    run_container("box", "img", cpus=0.5, runner=r)
    assert "--cpus" in r.last and "0.5" in r.last


def test_cpu_seconds_must_be_positive() -> None:
    with pytest.raises(PodmanCliError):
        run_container("box", "img", cpu_seconds=0, runner=_FakeRunner())


def test_max_open_files_must_be_positive() -> None:
    with pytest.raises(PodmanCliError):
        run_container("box", "img", max_open_files=0, runner=_FakeRunner())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
