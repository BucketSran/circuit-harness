"""Private runtime launcher with a real local process and a constructed engine API."""

import json
import sys
import tempfile
from pathlib import Path


def test_podman_runner_scopes_store_endpoint_and_cleans_service(tmp_path):
    from alphaapollo.workflows.harbor_chips import podman_runner

    engine = tmp_path / "engine"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import json,os,signal,socket,sys,time\n"
        "a=sys.argv[1:]\n"
        "if 'service' in a:\n"
        " s=socket.socket(socket.AF_UNIX); s.bind(a[-1][7:]); s.listen()\n"
        " while True: time.sleep(.1)\n"
        "elif 'ps' in a or 'ls' in a: print('')\n"
        "else: print(json.dumps(a))\n"
    )
    engine.chmod(0o755)
    short_root = tempfile.TemporaryDirectory(prefix="chips-", dir="/tmp")
    control = Path(short_root.name) / "control"
    record = tmp_path / "child.json"
    config = {
        "executable": str(engine),
        "root": str(tmp_path / "store"),
        "runroot": str(tmp_path / "run"),
        "control_dir": str(control),
        "ignore_chown_errors": True,
    }
    child = (
        "import json,os,subprocess,pathlib,sys; "
        f"pathlib.Path({str(record)!r}).write_text(json.dumps({{"
        "'endpoint':os.environ['DOCKER_HOST'],"
        "'argv':json.loads(subprocess.check_output(['podman','info'])),"
        "'utf8':sys.flags.utf8_mode,'context':os.environ.get('DOCKER_CONTEXT')}))"
    )
    assert podman_runner.run(config, [sys.executable, "-c", child]) == 0
    observed = json.loads(record.read_text())
    assert observed["argv"] == [
        "--root",
        str(tmp_path / "store"),
        "--runroot",
        str(tmp_path / "run"),
        "--storage-opt",
        "ignore_chown_errors=true",
        "info",
    ]
    assert observed["endpoint"] == "unix://" + str(control / "api.sock")
    assert observed["context"] is None
    assert observed["utf8"] == 1
    assert not (control / "api.sock").exists()
    evidence = json.loads((control / "runtime.json").read_text())
    assert evidence["cleanup_confirmed"]
    assert (
        not Path("/proc/" + str(evidence["service_pid"])).exists()
        if sys.platform == "linux"
        else True
    )
    assert control.stat().st_mode & 0o077 == 0
    short_root.cleanup()


def test_runtime_schema_matches_model():
    from alphaapollo.workflows.harbor_chips import podman_runner
    from alphaapollo.workflows.harbor_chips.podman_runner import PodmanRuntimeConfig

    schema = json.loads(Path(podman_runner.__file__).with_suffix(".schema.json").read_text())
    assert schema == PodmanRuntimeConfig.model_json_schema()


def test_runner_terminates_service_when_child_fails(tmp_path):
    # The same process boundary as the success case, with a failing child.
    import os

    import pytest

    from alphaapollo.workflows.harbor_chips.podman_runner import run

    with tempfile.TemporaryDirectory(prefix="chips-", dir="/tmp") as short:
        control = Path(short) / "control"
        engine = tmp_path / "engine"
        engine.write_text(
            f"#!{sys.executable}\nimport socket,sys,time\n"
            "if 'service' in sys.argv:\n"
            " s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[-1][7:]); s.listen()\n"
            " while True: time.sleep(.1)\n"
        )
        engine.chmod(0o755)
        config = dict(
            executable=str(engine),
            root=str(tmp_path / "store"),
            runroot=str(tmp_path / "run"),
            control_dir=str(control),
        )
        assert run(config, [sys.executable, "-c", "raise SystemExit(7)"]) == 7
        receipt = json.loads((control / "runtime.json").read_text())
        assert receipt["cleanup_confirmed"]
        with pytest.raises(ProcessLookupError):
            os.kill(receipt["service_pid"], 0)
        with pytest.raises(ProcessLookupError):
            os.kill(receipt["child_pid"], 0)
        assert not (control / "api.sock").exists()


def test_runner_sigterm_closes_active_child_and_service(tmp_path):
    import os
    import signal
    import subprocess
    import time

    import pytest

    with tempfile.TemporaryDirectory(prefix="chips-", dir="/tmp") as short:
        control = Path(short) / "control"
        engine = tmp_path / "engine"
        engine.write_text(
            f"#!{sys.executable}\nimport socket,sys,time\n"
            "if 'service' in sys.argv:\n"
            " s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[-1][7:]); s.listen()\n"
            " while True: time.sleep(.1)\n"
        )
        engine.chmod(0o755)
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                dict(
                    executable=str(engine),
                    root=str(tmp_path / "store"),
                    runroot=str(tmp_path / "run"),
                    control_dir=str(control),
                )
            )
        )
        ready = tmp_path / "ready"
        command = [
            sys.executable,
            "-m",
            "alphaapollo.workflows.harbor_chips.podman_runner",
            "--config",
            str(config),
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); time.sleep(60)",
        ]
        process = subprocess.Popen(command)
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                assert process.poll() is None
                time.sleep(0.05)
            assert ready.exists()
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=15) == 130
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
        receipt = json.loads((control / "runtime.json").read_text())
        assert receipt["cleanup_confirmed"]
        for pid in (receipt["child_pid"], receipt["service_pid"]):
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert not (control / "api.sock").exists()
