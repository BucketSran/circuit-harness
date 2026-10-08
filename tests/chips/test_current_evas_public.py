"""Real Docker boundary probes using a constructed CLI, not EVAS certification."""

import json
import os

import pytest

from tests.chips.test_current_evas_session import call, make_session


@pytest.mark.skipif(
    not os.environ.get("CHIPS_TEST_DOCKER_IMAGE"),
    reason="set immutable local Docker image for isolation probes",
)
def test_docker_public_execution_enforces_visibility_and_network(tmp_path):
    # The fixture engine tests actual OS boundaries before emitting protocol output.
    main = """
import json, os, pathlib, socket, sys
assert not pathlib.Path("HOST_SENTINEL").exists()
assert not pathlib.Path("/var/run/docker.sock").exists()
for filename in ["/inputs/dut.va", "/inputs/manifest.json",
                 "/engine/evas/__init__.py", "/private-write"]:
    try:
        pathlib.Path(filename).write_text("unexpected")
    except OSError:
        pass
    else:
        raise AssertionError("write escaped: " + filename)
s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(("192.0.2.1", 9))
except OSError:
    pass
else:
    raise AssertionError("network available")
assert not any(k.endswith("API_KEY") for k in os.environ)
m = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert pathlib.Path("/inputs/dut.va").read_text() == "bad syntax preserved"
print(json.dumps({"engine":"constructed", "nodes":["z"], "solutions":[{"voltages":[0]}],
 "transient":{"times":m["transient"]["output_times"], "events":[]}}))
""".replace("HOST_SENTINEL", str(tmp_path / "secret"))
    (tmp_path / "secret").write_text("hidden")
    directory = make_session(
        tmp_path,
        main=main,
        image=os.environ["CHIPS_TEST_DOCKER_IMAGE"],
        feedback_fields=["diagnostics", "observations"],
    )
    call(directory, "write", "evas_write", path="dut.va", content="bad syntax preserved")
    result = call(directory, "simulate", "evas_simulate")
    assert result["ok"], result
    assert result["result"]["execution"] == "ok", result
    assert result["result"]["cleanup_confirmed"]
    assert result["result"]["task_correctness"] == "not_evaluated"
    request = json.loads((directory / "actions/simulate/execution/request.json").read_text())
    assert request["image"] == os.environ["CHIPS_TEST_DOCKER_IMAGE"]


@pytest.mark.skipif(
    not os.environ.get("CHIPS_TEST_NATIVE_CODEX"),
    reason="set native Codex executable for OS sandbox probes",
)
def test_native_backend_denies_private_reads_and_writes(tmp_path):
    import sys
    from pathlib import Path

    private = tmp_path / "private"
    private.write_text("private fixture")
    main = """
import json,pathlib,sys,socket
try:
    pathlib.Path("PRIVATE").read_text()
except PermissionError:
    pass
else:
    raise AssertionError("private read escaped")
try:
    pathlib.Path(sys.argv[2]).write_text("changed")
except PermissionError:
    pass
else:
    raise AssertionError("fixed input writable")
m=json.loads(pathlib.Path(sys.argv[2]).read_text())
print(json.dumps({"engine":"constructed", "nodes":["z"], "solutions":[{"voltages":[0]}],
 "transient":{"times":m["transient"]["output_times"], "events":[]}}))
""".replace("PRIVATE", str(private))
    directory = make_session(
        tmp_path,
        main=main,
        image=None,
        backend="native_codex_sandbox",
        codex=Path(os.environ["CHIPS_TEST_NATIVE_CODEX"]),
        python=sys.executable,
    )
    call(directory, "write", "evas_write", path="dut.va", content="bad syntax preserved")
    result = call(directory, "sim", "evas_simulate")["result"]
    assert result["execution"] == "ok", result
    assert result["cleanup_confirmed"]


@pytest.mark.skipif(
    not os.environ.get("CHIPS_TEST_DOCKER_IMAGE"), reason="requires local Docker image"
)
def test_docker_public_experiment_has_complete_identity_and_no_host_access(tmp_path):
    image = os.environ["CHIPS_TEST_DOCKER_IMAGE"]
    private = tmp_path / "private"
    private.write_text("hidden grading sentinel")
    directory = make_session(
        tmp_path,
        image=image,
        feedback_fields=["observations"],
        experiments={
            "version": "v1",
            "files": ["probe.py", "stimulus.json"],
            "analyses": ["python_measurement"],
        },
    )
    script = """import json,os,pathlib,socket
assert not pathlib.Path("PRIVATE").exists()
assert not pathlib.Path('/var/run/docker.sock').exists()
assert not os.environ.get('CHIPS_SECRET_TEST')
try:
    pathlib.Path('/inputs/dut.va').write_text('tampered')
except OSError:
    pass
else:
    raise AssertionError('candidate writable')
connection=socket.socket(); connection.settimeout(.2)
try:
    connection.connect(('192.0.2.1',80))
except OSError:
    pass
else:
    raise AssertionError('network escaped')
print(json.dumps({'length':len(pathlib.Path('/inputs/dut.va').read_text()),
                  'stimulus':json.loads(pathlib.Path('/inputs/stimulus.json').read_text())}))
""".replace("PRIVATE", str(private))
    call(directory, "candidate", "evas_write", path="dut.va", content="complete")
    call(directory, "script", "evas_write", path="probe.py", content=script)
    call(directory, "stimulus", "evas_write", path="stimulus.json", content='{"level":0.2}')
    result = call(
        directory, "run", "evas_experiment", analysis="python_measurement", script="probe.py"
    )["result"]
    assert result["execution"] == "ok", result
    assert result["cleanup_confirmed"]
    assert result["observations"] == {"length": 8, "stimulus": {"level": 0.2}}
    assert result["authority"] == "agent_measurement"
    assert result["task_correctness"] == "not_evaluated"
    assert result["experiment_sha256"] != result["candidate_sha256"]
    assert "diagnostics" not in result


@pytest.mark.skipif(
    not os.environ.get("CHIPS_TEST_DOCKER_IMAGE"), reason="requires local Docker image"
)
@pytest.mark.parametrize(
    "script, expected",
    [
        ("import time; time.sleep(30)", "timeout"),
        ("print('broken json')", "invalid_result"),
        ("print('{\"reward\":1}')", "invalid_result"),
        ("print('x'*500000)", "output_limit"),
    ],
)
def test_docker_experiment_failures_are_unscored_and_cleaned(tmp_path, script, expected):
    import json

    directory = make_session(
        tmp_path,
        image=os.environ["CHIPS_TEST_DOCKER_IMAGE"],
        experiments={"version": "v1", "files": ["probe.py"], "analyses": ["python_measurement"]},
    )
    config_path = directory / "session.json"
    config = json.loads(config_path.read_text())
    config["timeout_s"] = 2
    config["max_output_bytes"] = 65536
    config_path.write_text(json.dumps(config))
    call(directory, "candidate", "evas_write", path="dut.va", content="complete")
    call(directory, "script", "evas_write", path="probe.py", content=script)
    result = call(
        directory, "run", "evas_experiment", analysis="python_measurement", script="probe.py"
    )["result"]
    assert result["execution"] == expected, result
    assert result["cleanup_confirmed"]
    assert result["task_correctness"] == "not_evaluated"
    assert not result.get("observations")
    assert "reward" not in result
    assert call(directory, "again", "evas_simulate")["error"] == "simulation_budget_exhausted"


@pytest.mark.skipif(
    not os.environ.get("CHIPS_TEST_DOCKER_IMAGE"), reason="requires local Docker image"
)
def test_docker_missing_declared_program_is_infrastructure_failure(tmp_path):
    from alphaapollo.common.execution.chips.current_evas_public import run_isolated_docker

    result = run_isolated_docker(
        image=os.environ["CHIPS_TEST_DOCKER_IMAGE"],
        readonly={},
        writable={},
        command=["missing-declared-program"],
        directory=tmp_path / "execution",
        action_id="missing-program",
        timeout_s=2,
        max_output_bytes=65536,
        workdir="/tmp",
    )
    assert result["execution"] == "infrastructure_error"
    assert result["cleanup_confirmed"]


def test_watchdog_checks_aggregate_output_after_fast_child_exit(tmp_path):
    import subprocess
    import sys

    from alphaapollo.common.execution.chips.current_evas_public import _CONTAINER_WATCHDOG

    for count, expected in ((1, 0), (20, 123)):
        output = tmp_path / str(count)
        output.mkdir()
        code = (
            "import pathlib,resource;resource.setrlimit(resource.RLIMIT_FSIZE,(1024,1024));"
            f"root=pathlib.Path({str(output)!r});"
            f"[(root/str(i)).write_bytes(b'x'*512) for i in range({count})]"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                _CONTAINER_WATCHDOG,
                "3",
                json.dumps([sys.executable, "-I", "-S", "-c", code]),
                json.dumps([[str(output)], 1024]),
            ],
            capture_output=True,
            timeout=5,
        )
        assert result.returncode == expected, result.stderr
        assert all(path.stat().st_size == 512 for path in output.iterdir())


def test_podman_public_measurement_records_explicit_cpu_policy(tmp_path, monkeypatch):
    """Constructed engine CLI tests routing/evidence, not container isolation."""
    import sys

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    engine = bin_dir / "podman"
    calls = tmp_path / "engine.jsonl"
    engine.write_text(
        f"#!{sys.executable}\nimport json,sys\n"
        f"with open({str(calls)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if sys.argv[1] == 'run': print('{\"voltage\":0.2}')\n"
    )
    engine.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    directory = make_session(
        tmp_path,
        backend="podman",
        cpu_limit=None,
        feedback_fields=["observations"],
        experiments={"version": "v1", "files": ["probe.py"], "analyses": ["python_measurement"]},
    )
    call(directory, "candidate", "evas_write", path="dut.va", content="candidate")
    call(directory, "script", "evas_write", path="probe.py", content="print('{}')")
    result = call(
        directory, "measure", "evas_experiment", analysis="python_measurement", script="probe.py"
    )["result"]
    assert result["execution"] == "ok", result
    assert result["cleanup_confirmed"]
    assert result["observations"] == {"voltage": 0.2}
    evidence = json.loads((directory / "actions/measure/execution/backend.json").read_text())
    assert evidence["backend"] == "podman"
    assert evidence["cpu_limit"] is None
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert not any(arg.startswith("--cpus") for arg in commands[0])
    assert "--network=none" in commands[0]
    assert "--read-only" in commands[0]
    assert "--memory=512m" in commands[0]
    assert "--pids-limit=32" in commands[0]
    assert commands[-1][0:2] == ["rm", "-f"]


@pytest.mark.parametrize("cpu_limit", [1, None])
def test_podman_quota_failure_is_not_silently_retried(tmp_path, monkeypatch, cpu_limit):
    import sys

    from alphaapollo.common.execution.chips.current_evas_public import run_isolated_container

    engine = tmp_path / "podman"
    engine.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "if sys.argv[1]=='run': sys.exit(126 if '--cpus=1' in sys.argv else 0)\n"
    )
    engine.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    result = run_isolated_container(
        backend="podman",
        cpu_limit=cpu_limit,
        image="sha256:" + "a" * 64,
        readonly={},
        writable={},
        command=["true"],
        directory=tmp_path / "result",
        action_id="quota",
        timeout_s=2,
        max_output_bytes=65536,
    )
    assert result["execution"] == ("infrastructure_error" if cpu_limit else "ok")
    assert result["cleanup_confirmed"]
    assert json.loads((tmp_path / "result/backend.json").read_text())["cpu_limit"] == cpu_limit
