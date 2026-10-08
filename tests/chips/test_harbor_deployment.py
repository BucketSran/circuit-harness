"""Operator preflight over local files; no model, SSH or simulator execution."""

import json
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from circuit_harness.harbor.deployment import main


def recipe(tmp_path):
    root = Path(__file__).resolve().parents[2] / "examples/chips/harbor"
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text("exit 0\n")
    (task / "instruction.md").write_text("Fixture\n")
    (task / "task.toml").write_text('[environment]\ndocker_image="sha256:' + "a" * 64 + '"\n')
    public = json.loads((root / "public-session.example.json").read_text())
    for name, directory in (("materials", True), ("checkout", True), ("kernel", False)):
        path = tmp_path / name
        path.mkdir() if directory else path.write_bytes(b"\x7fELFfixture")
        public[name] = str(path)
    session = tmp_path / "public.json"
    session.write_text(json.dumps(public))
    final = tmp_path / "final.json"
    package = tmp_path / "final-package"
    package.mkdir()
    final_data = json.loads((root / "final-evaluation.example.json").read_text())
    final_data["task_package"] = str(package)
    final.write_text(json.dumps(final_data))
    job = json.loads((root / "job.example.json").read_text())
    job["tasks"] = [{"path": str(task)}]
    job["jobs_dir"] = str(tmp_path / "jobs")
    job["environment"]["kwargs"]["session_config"] = str(session)
    job["environment"]["kwargs"]["gateway_host"] = "host.docker.internal"
    job["verifier"]["kwargs"]["config_path"] = str(final)
    job["agents"] = [
        {
            "import_path": "circuit_harness.harbor.installed_agent:CircuitAgent",
            "model_name": "openai/fixture",
            "kwargs": {"agent_name": "pi"},
            "env": {"OPENAI_API_KEY": "${FIXTURE_KEY}"},
        }
    ]
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    return path, job


def test_static_cli_reports_limits_without_launching(tmp_path, capsys):
    path, _ = recipe(tmp_path)
    assert main(["--job", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    checks = {item["name"]: item for item in result["checks"]}
    assert checks["configuration"]["status"] == "passed"
    for name in ("environment", "cleanup", "model_auth", "commercial_license", "final_evaluation"):
        assert checks[name]["status"] == "not_checked"
    assert result["acceptance"] == "not_evaluated"


def test_unknown_native_agent_options_fail_before_environment(tmp_path, capsys):
    path, job = recipe(tmp_path)
    job["agents"][0]["kwargs"]["agent_kwargs"] = {"not_a_pi_option": True}
    path.write_text(json.dumps(job))
    assert main(["--job", str(path), "--check-environment"]) == 1
    checks = json.loads(capsys.readouterr().out)["checks"]
    assert checks[0]["code"] == "configuration_invalid"
    assert next(c for c in checks if c["name"] == "environment")["status"] == "not_checked"


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_image_command_is_bounded_and_never_installs(tmp_path, capsys, monkeypatch, failure):
    import os
    import sys

    path, _ = recipe(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "commands.jsonl"
    executable = bin_dir / "docker"
    script = bin_dir / "docker_fixture.py"
    script.write_text(
        "import json,sys,time\n"
        f"with open({str(calls)!r}, 'a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')\n"
        + "if sys.argv[1:3] == ['context', 'inspect']:\n"
        + " print('unix:///fixture/socket'); sys.exit(0)\n"
        + ("sys.exit(1)\n" if failure == "exit" else "time.sleep(10)\n")
    )
    executable.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    executable.chmod(0o755)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    assert main(["--job", str(path), "--check-environment", "--timeout-s", "2"]) == 1
    checks = json.loads(capsys.readouterr().out)["checks"]
    result = next(c for c in checks if c["name"] == "environment")
    assert result["code"] == (
        "local_image_unavailable" if failure == "exit" else "image_inspection_timeout"
    )
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert commands[-1][:2] == ["image", "inspect"]
    assert all(
        command[:2] in (["context", "inspect"], ["image", "inspect"]) for command in commands
    )
    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize(
    "mutation",
    ["job_unknown", "task_unknown", "private_in_task", "duplicate_key", "missing_resource"],
)
def test_static_rejects_bad_private_declarations_without_external_commands(
    tmp_path, capsys, mutation
):
    path, job = recipe(tmp_path)
    public_path = Path(job["environment"]["kwargs"]["session_config"])
    if mutation == "job_unknown":
        job["environment"]["not_a_harbor_option"] = True
    elif mutation == "task_unknown":
        public = json.loads(public_path.read_text())
        public["task"]["final_secret"] = "do not expose"
        public_path.write_text(json.dumps(public))
    elif mutation == "private_in_task":
        private = tmp_path / "task/private.json"
        private.write_bytes(public_path.read_bytes())
        job["environment"]["kwargs"]["session_config"] = str(private)
    elif mutation == "missing_resource":
        (tmp_path / "kernel").unlink()
    path.write_text(json.dumps(job))
    if mutation == "duplicate_key":
        path.write_text('{"agents":[], "agents":[]}')
    assert main(["--job", str(path), "--check-environment"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["checks"][0]["status"] == "failed"
    assert (
        next(c for c in result["checks"] if c["name"] == "environment")["code"]
        == "configuration_failed"
    )
    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize(
    "startup,cleanup,expected",
    [
        ("ok", "ok", ("passed", "passed")),
        ("fail", "ok", ("failed", "passed")),
        ("timeout", "ok", ("failed", "passed")),
        ("ok", "fail", ("passed", "failed")),
        ("ok", "timeout", ("passed", "failed")),
    ],
)
def test_environment_lifecycle_reports_failure_and_cleans_after_partial_start(
    startup, cleanup, expected
):
    import asyncio

    from circuit_harness.harbor.deployment import check_environment

    class ExternalEnvironment:
        """Harbor environment protocol double; no Harness implementation mocked."""

        closed = False

        async def start(self, force_build):
            assert force_build is False
            if startup == "fail":
                raise RuntimeError("private-detail-must-not-escape")
            if startup == "timeout":
                await asyncio.sleep(10)

        async def stop(self, delete):
            assert delete is True
            self.closed = True
            if cleanup == "fail":
                raise RuntimeError("private-detail-must-not-escape")
            if cleanup == "timeout":
                await asyncio.sleep(10)

    environment = ExternalEnvironment()
    report = asyncio.run(check_environment(environment, timeout_s=0.02, cleanup_timeout_s=0.02))
    assert tuple(item["status"] for item in report) == expected
    assert environment.closed
    assert "private-detail" not in json.dumps(report)


def test_unknown_task_toml_fields_fail_static(tmp_path, capsys):
    path, _ = recipe(tmp_path)
    with (tmp_path / "task/task.toml").open("a") as stream:
        stream.write('secret_runtime_option="ignored"\n')
    assert main(["--job", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["checks"][0]["code"] == "configuration_invalid"


def test_job_exports_cannot_be_created_inside_public_task(tmp_path, capsys):
    path, job = recipe(tmp_path)
    job["jobs_dir"] = str(tmp_path / "task/jobs")
    path.write_text(json.dumps(job))
    assert main(["--job", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["checks"][0]["code"] == "configuration_invalid"


def test_multitask_preflight_uses_shared_binding_identity_and_checks_every_task(tmp_path, capsys):
    import subprocess

    entries = []
    for name in ("a", "b"):
        base = tmp_path / name
        base.mkdir()
        path, job = recipe(base)
        checkout = base / "checkout"
        source = checkout / "evas/src/evas/__init__.py"
        source.parent.mkdir(parents=True)
        source.write_text("VERSION = 1\n")
        (checkout / ".gitignore").write_text("__pycache__/\n")
        subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
        subprocess.run(["git", "add", "."], cwd=checkout, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            cwd=checkout,
            check=True,
        )
        public_path = Path(job["environment"]["kwargs"]["session_config"])
        public = json.loads(public_path.read_text())
        public["task"].update(task_id=name, task_version="v1")
        public_path.write_text(json.dumps(public))
        final_path = Path(job["verifier"]["kwargs"]["config_path"])
        package = Path(json.loads(final_path.read_text())["task_package"])
        (package / "manifest.json").write_text(json.dumps({"task_id": name, "task_version": "v1"}))
        entries.append(
            {
                "task_path": job["tasks"][0]["path"],
                "task_id": name,
                "task_version": "v1",
                "session_config": str(public_path),
                "final_config": str(final_path),
            }
        )
    manifest = tmp_path / "bindings.json"
    manifest.write_text(json.dumps({"schema_version": 1, "tasks": entries}))
    job["tasks"] = [{"path": entry["task_path"]} for entry in entries]
    job["environment"]["kwargs"].pop("session_config")
    job["environment"]["kwargs"]["task_bindings"] = str(manifest)
    job["verifier"]["kwargs"] = {"task_bindings": str(manifest)}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    assert main(["--job", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["checks"][0]["status"] == "passed"
    # Task B identity drift cannot be hidden by task A's valid declaration.
    (tmp_path / "b/final-package/manifest.json").write_text(
        '{"task_id":"wrong", "task_version":"v1"}'
    )
    assert main(["--job", str(path), "--check-environment"]) == 1
    assert json.loads(capsys.readouterr().out)["checks"][0]["status"] == "failed"


def test_cancelled_harbor_command_does_not_survive_deadline(tmp_path):
    import asyncio
    import os
    import sys

    from circuit_harness.harbor.deployment import PreflightEnvironment

    async def scenario():
        marker = tmp_path / "late-write"
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import pathlib,time; time.sleep(2); pathlib.Path("
            + repr(str(marker))
            + ").write_text('late')",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                PreflightEnvironment._collect_buffered_output(
                    process, timeout_sec=None, stdin_data=None
                ),
                0.1,
            )
        assert process.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
        assert not marker.exists()

    asyncio.run(scenario())


def test_missing_selected_model_fails_before_any_environment(tmp_path, capsys):
    path, job = recipe(tmp_path)
    job["agents"][0].pop("model_name")
    path.write_text(json.dumps(job))
    assert main(["--job", str(path), "--check-environment"]) == 1
    assert json.loads(capsys.readouterr().out)["checks"][0]["code"] == "configuration_invalid"


@pytest.mark.parametrize("seconds", ["nan", "inf", "0", "-1", "301"])
def test_cli_rejects_nonfinite_or_out_of_range_budgets(tmp_path, seconds):
    path, _ = recipe(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main(["--job", str(path), "--timeout-s", seconds])
    assert exit_info.value.code == 2


def test_cancelled_cli_terminates_its_plugin_child(tmp_path):
    import asyncio
    import os
    import sys

    from circuit_harness.harbor.deployment import PreflightEnvironment

    async def scenario():
        child_pid = tmp_path / "child.pid"
        code = (
            "import subprocess,sys,signal,time,pathlib; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)']); "
            "signal.signal(signal.SIGTERM,lambda *args: (child.wait(),sys.exit(0))); "
            f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid)); "
            "time.sleep(10)"
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        async with asyncio.timeout(3):
            while not child_pid.exists():
                await asyncio.sleep(0.01)
        pid = int(child_pid.read_text())
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    PreflightEnvironment._collect_buffered_output(process, timeout_sec=None), 0.1
                )
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        finally:
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
            await process.wait()

    asyncio.run(scenario())


def test_environment_preflight_rejects_ssh_docker_without_connecting(tmp_path, capsys, monkeypatch):
    path, _ = recipe(tmp_path)
    monkeypatch.setenv("DOCKER_HOST", "ssh://private-server")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    assert main(["--job", str(path), "--check-environment"]) == 1
    checks = json.loads(capsys.readouterr().out)["checks"]
    assert next(c for c in checks if c["name"] == "environment")["code"] == "local_docker_required"
    assert not (tmp_path / "jobs").exists()


def test_preflight_preserves_stock_harbor_task_version_alias(tmp_path, capsys):
    path, _ = recipe(tmp_path)
    task_config = tmp_path / "task/task.toml"
    task_config.write_text('version = "1.0"\n' + task_config.read_text())
    assert main(["--job", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["checks"][0]["status"] == "passed"


def test_podman_recipe_compiles_and_static_preflight_preserves_cpu_declaration(tmp_path, capsys):
    from circuit_harness.harbor.config import PublicSessionConfig
    from circuit_harness.harbor.profiles import compile_job

    path, job = recipe(tmp_path)
    job["environment"]["import_path"] = (
        "circuit_harness.harbor.podman_environment:CircuitPodmanEnvironment"
    )
    job["environment"]["kwargs"]["gateway_host"] = "host.containers.internal"
    public_path = Path(job["environment"]["kwargs"]["session_config"])
    public = json.loads(public_path.read_text())
    public.update(public_backend="podman", public_cpu_limit=None)
    public_path.write_text(json.dumps(public))
    catalog = json.loads(
        (
            Path(__file__).resolve().parents[2] / "examples/chips/harbor/profiles.example.json"
        ).read_text()
    )
    compiled = compile_job(catalog, "pi", "glm", job, protocol="openai-chat-completions")
    assert compiled.agents[0].import_path.endswith(":CircuitAgent")
    path.write_text(compiled.model_dump_json(context={"redact_sensitive_env": False}))
    assert main(["--job", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["checks"][0]["status"] == "passed"
    assert PublicSessionConfig.read(public_path).public_cpu_limit is None


def test_podman_preflight_rejects_public_cpu_quota_before_starting_agent_container(
    tmp_path, capsys, monkeypatch
):
    import os
    import sys

    path, job = recipe(tmp_path)
    job["environment"]["import_path"] = (
        "circuit_harness.harbor.podman_environment:CircuitPodmanEnvironment"
    )
    path.write_text(json.dumps(job))
    public_path = Path(job["environment"]["kwargs"]["session_config"])
    public = json.loads(public_path.read_text())
    public["public_backend"] = "podman"
    public_path.write_text(json.dumps(public))
    engine = tmp_path / "podman"
    engine.write_text(f"#!{sys.executable}\nimport sys\nif sys.argv[1]=='run': sys.exit(126)\n")
    engine.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("DOCKER_HOST", "unix:///fixture/socket")
    assert main(["--job", str(path), "--check-environment"]) == 1
    checks = json.loads(capsys.readouterr().out)["checks"]
    assert (
        next(c for c in checks if c["name"] == "environment")["code"]
        == "public_runtime_unavailable"
    )
    assert next(c for c in checks if c["name"] == "cleanup")["status"] == "passed"
