"""GPU/Podman-free tests for PodmanBackend (contract + profile + result mapping).

A FAKE podman_cli runner records every argv and returns canned results, so these
tests verify the backend WITHOUT Podman: it starts a container from the profile,
maps podman exec output onto ToolCallRecord, releases idempotently, and enforces the
security boundary (--network none, no -v).

The fixture profile is PODMAN_DEFAULT. These tests previously ran the backend from
DOCKER_PROFILE, a ``docker``-kind profile that was convenient because it carries an
image — exactly the mismatch the backend now refuses. The fixture was corrected, not
the rule; DOCKER_PROFILE survives here only as the refusal's negative case.
"""

from __future__ import annotations

import dataclasses

import pytest

from alphaapollo.common.execution.sandbox.base import (
    PODMAN_DEFAULT,
    PYTHON_DEFAULT,
    SandboxProfile,
    SandboxProfileError,
)
from alphaapollo.common.execution.sandbox.podman import (
    PodmanBackend,
    PodmanBackendError,
    PodmanCliError,
    PodmanResult,
)
from alphaapollo.common.execution.tools.schemas import ToolCallRecord

DOCKER_PROFILE = SandboxProfile(
    name="docker_test", kind="docker", image="python:3.11-slim", network=False
)


class _ScriptedRunner:
    """Records argv and returns queued/canned PodmanResults keyed by podman subcommand."""

    def __init__(self, *, exec_result: PodmanResult | None = None, run_exit: int = 0) -> None:
        self.calls: list[list[str]] = []
        self._exec_result = exec_result or PodmanResult(stdout="", stderr="", exit_code=0)
        self._run_exit = run_exit

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        sub = argv[1] if len(argv) > 1 else ""
        if sub == "run":
            return PodmanResult(stdout="containerid", stderr="", exit_code=self._run_exit)
        if sub == "exec":
            return self._exec_result
        if sub == "container":
            return PodmanResult(stdout=f"sha256:{'a' * 64}\n", stderr="", exit_code=0)
        if sub == "rm":
            return PodmanResult(stdout="", stderr="", exit_code=0)
        return PodmanResult(stdout="", stderr="", exit_code=0)

    def argvs_for(self, sub: str) -> list[list[str]]:
        return [c for c in self.calls if len(c) > 1 and c[1] == sub]


# --- profile kind must match the backend --------------------------------------
def test_backend_refuses_a_profile_targeting_another_backend_family() -> None:
    """A profile is honoured only by the backend family its ``kind`` names.

    DOCKER_PROFILE is a ``docker`` profile with a real image, so nothing but ``kind``
    distinguishes it here; before this check it started a Podman container from the
    Lean image. PYTHON_DEFAULT is the reverse direction: a host-subprocess profile
    must not be served by a container. Both messages name the declared kind and the
    selected backend, and neither reaches the Podman CLI.
    """
    for profile in (DOCKER_PROFILE, PYTHON_DEFAULT):
        runner = _ScriptedRunner()
        with pytest.raises(SandboxProfileError) as excinfo:
            PodmanBackend(profile=profile, runner=runner)
        message = str(excinfo.value)
        assert f"kind={profile.kind!r}" in message
        assert "'podman' backend was selected" in message
        assert runner.calls == []  # refused before any container start


# --- container image required ------------------------------------------------
def test_backend_requires_profile_with_image() -> None:
    """A podman-kind profile still needs an image; the kind check does not replace it."""
    imageless = dataclasses.replace(PODMAN_DEFAULT, image=None)
    with pytest.raises(PodmanBackendError, match="image"):
        PodmanBackend(profile=imageless, runner=_ScriptedRunner())


# --- startup ------------------------------------------------------------------
def test_backend_starts_container_from_profile_image_no_network_no_mount() -> None:
    r = _ScriptedRunner()
    backend = PodmanBackend(profile=PODMAN_DEFAULT, instance_id="abc", runner=r)
    run_argv = r.argvs_for("run")[0]
    assert backend.container_name == "apollo_pod_abc"
    assert "apollo_pod_abc" in run_argv
    assert PODMAN_DEFAULT.image in run_argv
    # security: PODMAN_DEFAULT.network is False → --network none, and never -v
    assert "--network" in run_argv and run_argv[run_argv.index("--network") + 1] == "none"
    assert "-v" not in run_argv and "--volume" not in run_argv
    assert backend.image_digest == f"sha256:{'a' * 64}"


def test_profile_overrides_emit_the_complete_container_hardening_set() -> None:
    runner = _ScriptedRunner()
    profile = PODMAN_DEFAULT.with_overrides(
        read_only_root=True,
        drop_all_capabilities=True,
        no_new_privileges=True,
        run_as_user="65532:65532",
        workspace_tmpfs_bytes=1536 * 1024 * 1024,
        tmp_tmpfs_bytes=256 * 1024 * 1024,
    )
    PodmanBackend(profile=profile, instance_id="hardened", runner=runner)

    argv = runner.argvs_for("run")[0]
    assert "--read-only" in argv
    assert argv[argv.index("--user") + 1] == "65532:65532"
    assert argv[argv.index("--cap-drop") + 1] == "all"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"
    tmpfs = [argv[index + 1] for index, value in enumerate(argv) if value == "--tmpfs"]
    assert any(value.startswith("/workspace:rw,size=") for value in tmpfs)
    assert any(value.startswith("/tmp:rw,size=") for value in tmpfs)


def test_backend_removes_partial_container_when_start_fails() -> None:
    r = _ScriptedRunner(run_exit=125)
    with pytest.raises(PodmanBackendError, match="failed to start"):
        PodmanBackend(profile=PODMAN_DEFAULT, instance_id="failed", runner=r)

    assert r.argvs_for("rm") == [["podman", "rm", "-f", "apollo_pod_failed"]]


# --- exec mapping -------------------------------------------------------------
def test_exec_maps_podman_result_to_toolcallrecord() -> None:
    r = _ScriptedRunner(exec_result=PodmanResult(stdout="391\n", stderr="", exit_code=0))
    backend = PodmanBackend(profile=PODMAN_DEFAULT, instance_id="abc", runner=r)
    record = backend.exec("python3 -c 'print(23*17)'")
    assert isinstance(record, ToolCallRecord)
    assert record.stdout == "391\n"
    assert record.exit_code == 0
    exec_argv = r.argvs_for("exec")[0]
    assert exec_argv[:5] == ["podman", "exec", "--workdir", "/workspace", "apollo_pod_abc"]
    assert exec_argv[-1] == "python3 -c 'print(23*17)'"  # command is one argv element


def test_exec_propagates_nonzero_exit() -> None:
    r = _ScriptedRunner(exec_result=PodmanResult(stdout="", stderr="err", exit_code=1))
    backend = PodmanBackend(profile=PODMAN_DEFAULT, runner=r)
    record = backend.exec("false")
    assert record.exit_code == 1
    assert record.stderr == "err"


def test_exec_with_timeout_overrides_profile_for_one_call() -> None:
    r = _ScriptedRunner(
        exec_result=PodmanResult(stdout="", stderr="", exit_code=124, timed_out=True)
    )
    backend = PodmanBackend(profile=PODMAN_DEFAULT, runner=r)

    record = backend.exec_with_timeout("sleep 2", timeout_seconds=1.5)

    assert record.exit_code == 124
    assert "command timed out after 1.5s" in record.stderr


@pytest.mark.parametrize(
    "error",
    [
        PodmanCliError("model-controlled secret"),
        RuntimeError("model-controlled secret"),
    ],
)
def test_exec_does_not_return_or_log_exception_details(
    error: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _RaisingRunner(_ScriptedRunner):
        def __call__(self, argv):
            if list(argv)[1:2] == ["exec"]:
                raise error
            return super().__call__(argv)

    with caplog.at_level("WARNING"):
        record = PodmanBackend(profile=PODMAN_DEFAULT, runner=_RaisingRunner()).exec("secret")

    assert record.exit_code == -1
    assert record.stderr == "podman exec failed"
    assert "model-controlled secret" not in caplog.text


# --- release ------------------------------------------------------------------
def test_release_removes_container_and_is_idempotent() -> None:
    r = _ScriptedRunner()
    backend = PodmanBackend(profile=PODMAN_DEFAULT, instance_id="abc", runner=r)
    backend.release()
    backend.release()  # second release is a no-op
    kill_calls = r.argvs_for("kill")
    rm_calls = r.argvs_for("rm")
    assert kill_calls == [["podman", "kill", "apollo_pod_abc"]]
    assert rm_calls == [["podman", "rm", "-f", "apollo_pod_abc"]]  # exactly one rm


def test_release_failure_is_observable_and_retryable() -> None:
    class _FailOnceRunner(_ScriptedRunner):
        def __init__(self) -> None:
            super().__init__()
            self.rm_attempts = 0

        def __call__(self, argv):
            if list(argv)[1:2] == ["rm"]:
                self.rm_attempts += 1
                if self.rm_attempts == 1:
                    return PodmanResult(stdout="", stderr="busy", exit_code=125)
            return super().__call__(argv)

    r = _FailOnceRunner()
    backend = PodmanBackend(profile=PODMAN_DEFAULT, runner=r)
    with pytest.raises(PodmanBackendError, match="failed to remove"):
        backend.release()
    backend.release()
    assert r.rm_attempts == 2


def test_timeout_record_poisoned_and_container_killed() -> None:
    class _TimeoutRunner(_ScriptedRunner):
        def __call__(self, argv):
            argv = list(argv)
            self.calls.append(argv)
            if argv[1:2] == ["exec"]:
                return PodmanResult(
                    stdout="partial",
                    stderr="",
                    exit_code=124,
                    timed_out=True,
                )
            if argv[1:2] == ["kill"]:
                return PodmanResult(stdout="", stderr="", exit_code=0)
            if argv[1:2] == ["run"]:
                return PodmanResult(stdout="containerid", stderr="", exit_code=0)
            return PodmanResult(stdout="", stderr="", exit_code=0)

    r = _TimeoutRunner()
    backend = PodmanBackend(profile=PODMAN_DEFAULT, runner=r)
    record = backend.exec("sleep 2")
    assert record.exit_code == 124
    assert any(call[1:2] == ["kill"] for call in r.calls)
    assert backend.exec("echo later").exit_code == -1


def test_cancelled_record_uses_130_and_kills_the_container() -> None:
    class _CancelledRunner(_ScriptedRunner):
        def __call__(self, argv):
            argv = list(argv)
            self.calls.append(argv)
            if argv[1:2] == ["exec"]:
                return PodmanResult(
                    stdout="partial\n",
                    stderr="",
                    exit_code=130,
                    cancelled=True,
                )
            if argv[1:2] == ["kill"]:
                return PodmanResult(stdout="", stderr="", exit_code=0)
            if argv[1:2] == ["run"]:
                return PodmanResult(stdout="containerid", stderr="", exit_code=0)
            return PodmanResult(stdout="", stderr="", exit_code=0)

    runner = _CancelledRunner()
    backend = PodmanBackend(profile=PODMAN_DEFAULT, runner=runner)

    record = backend.exec("long-running")

    assert record.exit_code == 130
    assert "command cancelled" in record.stderr
    assert any(call[1:2] == ["kill"] for call in runner.calls)


def test_exec_after_release_raises() -> None:
    backend = PodmanBackend(profile=PODMAN_DEFAULT, runner=_ScriptedRunner())
    backend.release()
    with pytest.raises(PodmanBackendError, match="released"):
        backend.exec("echo hi")


# --- isolation ----------------------------------------------------------------
def test_distinct_instances_get_distinct_container_names() -> None:
    r = _ScriptedRunner()
    a = PodmanBackend(profile=PODMAN_DEFAULT, runner=r)
    b = PodmanBackend(profile=PODMAN_DEFAULT, runner=r)
    assert a.container_name != b.container_name


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))


def test_backend_forwards_profile_cpu_and_nofile_caps_to_run() -> None:
    r = _ScriptedRunner()
    prof = dataclasses.replace(PODMAN_DEFAULT, cpu_seconds=5, max_open_files=5000, cpus=0.5)
    PodmanBackend(profile=prof, instance_id="caps", runner=r)
    run_argv = r.argvs_for("run")[0]
    assert "cpu=5" in run_argv  # RLIMIT_CPU wired (Yiming P1)
    assert "nofile=5000" in run_argv  # RLIMIT_NOFILE wired (Yiming P1)
    assert "--cpus" in run_argv and "0.5" in run_argv  # rate cap wired (Yiming P2)
