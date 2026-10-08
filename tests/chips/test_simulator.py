"""Real local processes/GDS; constructed SSH and EMX fixtures, not lab evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from circuit_harness.execution.emx import (
    EmxConfig,
    SshTransport,
    TransportFailure,
    build_worker,
    run_emx,
)
from circuit_harness.execution.journal import Journal, read_events, status
from circuit_harness.execution.process import _directory_output_bytes, run_process
from circuit_harness.execution.simulator import SimulationRequest, simulate

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/chips"
EMX_EXAMPLE = EXAMPLE / "emx"


def test_output_limit_tolerates_atomic_temp_rename(tmp_path, monkeypatch):
    stable = tmp_path / "stable.log"
    stable.write_bytes(b"four")
    temporary = tmp_path / "status.process.json.tmp"
    temporary.write_bytes(b"vanishing")
    original = Path.stat

    def stat(path, *args, **kwargs):
        if path == temporary:
            raise FileNotFoundError(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    assert _directory_output_bytes(tmp_path) == 4


@pytest.fixture
def fake_lab(tmp_path, monkeypatch):
    pytest.importorskip("gdstk")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    shim = """import os, pathlib, shlex, shutil, subprocess, sys
marker = pathlib.Path(os.environ['TEST_MARKER'])
mode = os.environ.get('TEST_FAULT', '')
if mode == 'auth':
    print('Permission denied (publickey).', file=sys.stderr)
    sys.exit(255)
if pathlib.Path(sys.argv[0]).name == 'ssh':
    command = shlex.split(sys.argv[-1])
    result = subprocess.run(command, capture_output=True)
    if 'submit' in command and mode == 'lost_submit' and not marker.exists():
        marker.touch()
        print('Connection reset by peer', file=sys.stderr)
        sys.exit(255)
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)
source, destination = sys.argv[-2:]
if ':' in source:
    if mode == 'lost_download' and not marker.exists():
        marker.touch()
        print('Connection reset by peer', file=sys.stderr)
        sys.exit(1)
    source = source.split(':', 1)[1]
else:
    destination = destination.split(':', 1)[1]
shutil.copyfile(source, destination)
"""
    for name in ("ssh", "scp"):
        path = binaries / name
        path.write_text(f"#!{sys.executable}\n{shim}")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_MARKER", str(tmp_path / "fault-used"))
    tool = tmp_path / "constructed_emx.py"
    tool.write_text(
        "from pathlib import Path\nimport sys\n"
        "root=Path(sys.argv[2]).parent\n"
        "with (root/'launches').open('a') as f: f.write('launch\\n')\n"
        "assert Path(sys.argv[1]).read_bytes()[:2] == b'\\x00\\x06'\n"
        "Path('result.s2p').write_text('# constructed protocol fixture, NOT EMX\\n')\n"
        "print('constructed EMX fixture completed')\n"
    )
    config = EmxConfig(
        ssh_host="fixture",
        remote_root=str(tmp_path / "remote"),
        converter=[
            sys.executable,
            str(EMX_EXAMPLE / "json_to_gds.py"),
            "{input_json}",
            "{output_gds}",
        ],
        emx_command=[sys.executable, str(tool), "{gds}", "{job_dir}"],
        remote_input_files=[str(tool)],
        outputs=["result.s2p"],
        remote_python=sys.executable,
        timeout_s=25,
        command_timeout_s=5,
    )
    return config, json.loads((EMX_EXAMPLE / "layout.json").read_text())


@pytest.mark.parametrize("fault", ["", "lost_submit", "lost_download"])
def test_gds_ssh_download_and_resume_once(fake_lab, tmp_path, monkeypatch, fault):
    import gdstk

    config, layout = fake_lab
    monkeypatch.setenv("TEST_FAULT", fault)
    directory = tmp_path / "run"
    result = run_emx(layout, config, directory)
    assert result["execution"] == "ok", result
    assert result["verdict"] == "not_evaluated"
    assert not result["certified"]
    library = gdstk.read_gds(directory / "input.gds")
    assert library.cells[0].name == "CHIPS_SMOKE"
    assert (directory / "artifacts/emx.stdout.log").exists()
    assert (directory / "artifacts/result.s2p").exists()
    resumed = run_emx(layout, config, directory, resume=True)
    assert resumed["job_id"] == result["job_id"]
    assert (Path(config.remote_root) / "launches").read_text().splitlines() == ["launch"]
    events, _ = read_events(directory / "events.jsonl")
    assert any(e["event"] == "remote_job_reused" for e in events)
    assert any(e["event"] == "download_reused" for e in events)
    if fault:
        assert any(e["event"] == "retry_wait" for e in events)


def test_authentication_does_not_retry(fake_lab, tmp_path, monkeypatch):
    config, layout = fake_lab
    monkeypatch.setenv("TEST_FAULT", "auth")
    result = run_emx(layout, config, tmp_path / "run")
    assert result["execution"] == "unknown"
    events, _ = read_events(tmp_path / "run/events.jsonl")
    assert len([e for e in events if e["event"] == "transport_attempt"]) == 1
    assert not Path(config.remote_root).exists()
    assert status(tmp_path / "run")["state"] == "attention_required"


@pytest.mark.parametrize("deadline_s, attempts", [(10, 3), (0.2, 1)])
def test_persistent_network_failure_is_bounded(fake_lab, tmp_path, deadline_s, attempts):
    config, _ = fake_lab
    journal = Journal(tmp_path, call_id="test")
    transport = SshTransport(
        config, tmp_path, journal, threading.Event(), time.monotonic() + deadline_s
    )
    with pytest.raises(TransportFailure):
        transport.command([sys.executable, "-c", "raise SystemExit(255)"], "failure")
    rows, _ = read_events(journal.path)
    assert len([row for row in rows if row["event"] == "transport_attempt"]) == attempts


def test_corrupt_download_is_not_published(fake_lab, tmp_path, monkeypatch):
    config, layout = fake_lab
    original = SshTransport.copy

    def corrupt(self, local, remote, *, upload):
        original(self, local, remote, upload=upload)
        if not upload:
            local.write_bytes(b"transfer corruption")

    monkeypatch.setattr(SshTransport, "copy", corrupt)
    result = run_emx(layout, config, tmp_path / "run")
    assert result["execution"] == "unknown"
    assert "integrity mismatch" in result["error"]
    assert all(p.suffix == ".partial" for p in (tmp_path / "run/artifacts").iterdir())


def test_invalid_layout_never_connects(fake_lab, tmp_path):
    config, _ = fake_lab
    result = run_emx({"cell": "bad", "polygons": []}, config, tmp_path / "run")
    assert result["execution"] == "conversion_failed"
    assert not Path(config.remote_root).exists()


def test_process_output_limit_and_start_failure(tmp_path):
    for stage, argv, limit, expected in (
        ("flood", [sys.executable, "-c", "print('x'*20000)"], 4096, "output_limit"),
        ("missing", ["/not/a/chips/executable"], 1024**2, "infrastructure_error"),
    ):
        directory = tmp_path / stage
        result = run_process(
            argv,
            directory=directory,
            stage=stage,
            deadline=time.monotonic() + 5,
            cancel=threading.Event(),
            journal=Journal(directory, call_id=stage),
            max_output_bytes=limit,
        )
        assert result["execution"] == expected


def test_logging_failure_still_cleans_process(tmp_path, monkeypatch):
    marker = tmp_path / "should-not-exist"
    journal = Journal(tmp_path, call_id="test")

    def fail(*args, **kwargs):
        raise OSError("injected journal disk failure")

    monkeypatch.setattr(journal, "emit", fail)
    command = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
    with pytest.raises(OSError, match="disk failure"):
        run_process(
            [sys.executable, "-c", command],
            directory=tmp_path,
            stage="logging",
            deadline=time.monotonic() + 5,
            cancel=threading.Event(),
            journal=journal,
        )
    time.sleep(1.1)
    assert not marker.exists()


def test_resume_rejects_changed_input_gds_and_missing_submitted_job(fake_lab, tmp_path):
    config, layout = fake_lab
    directory = tmp_path / "run"
    result = run_emx(layout, config, directory)
    with pytest.raises(ValueError, match="identical"):
        run_emx({**layout, "cell": "OTHER"}, config, directory, resume=True)
    original = (directory / "input.gds").read_bytes()
    (directory / "input.gds").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        run_emx(layout, config, directory, resume=True)
    (directory / "input.gds").write_bytes(original)
    shutil.rmtree(Path(config.remote_root) / result["job_id"])
    resumed = run_emx(layout, config, directory, resume=True)
    assert resumed["execution"] == "unknown"
    assert "refusing resubmission" in resumed["error"]
    assert (Path(config.remote_root) / "launches").read_text().splitlines() == ["launch"]


def test_emx_failure_and_missing_outputs_are_not_success(fake_lab, tmp_path):
    config, layout = fake_lab
    for name, command, expected in (
        ("failure", [sys.executable, "-c", "raise SystemExit(3)", "{gds}"], "tool_error"),
        ("missing", [sys.executable, "-c", "print('no output')", "{gds}"], "missing_output"),
    ):
        result = run_emx(layout, replace(config, emx_command=command), tmp_path / name)
        assert result["execution"] == expected, result
        assert result["verdict"] == "not_evaluated"


def test_remote_cancel_stops_process(fake_lab, tmp_path):
    config, layout = fake_lab
    config = replace(
        config, emx_command=[sys.executable, "-c", "import time; time.sleep(30)", "{gds}"]
    )
    cancel = threading.Event()
    directory = tmp_path / "run"

    def cancel_after_submission():
        until = time.monotonic() + 15
        while time.monotonic() < until:
            if (directory / "submit-requested.json").exists():
                time.sleep(0.5)
                cancel.set()
                return
            time.sleep(0.05)

    watcher = threading.Thread(target=cancel_after_submission)
    watcher.start()
    result = run_emx(layout, config, directory, cancel=cancel)
    watcher.join(1)
    assert result["execution"] == "cancelled", result
    path = Path(config.remote_root) / result["job_id"] / "result.json"
    until = time.monotonic() + 5
    while not path.exists() and time.monotonic() < until:
        time.sleep(0.1)
    assert json.loads(path.read_text())["execution"] == "cancelled"


@pytest.mark.parametrize("mode", ["timeout", "cancel", "parent_exit"])
def test_process_group_cleanup(tmp_path, mode):
    marker = tmp_path / "orphan-wrote"
    child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); "
    parent += "time.sleep(30)" if mode != "parent_exit" else "sys.exit(0)"
    cancel = threading.Event()
    timer = threading.Timer(0.2, cancel.set) if mode == "cancel" else None
    if timer:
        timer.start()
    result = run_process(
        [sys.executable, "-c", parent],
        directory=tmp_path,
        stage="child",
        deadline=time.monotonic() + (0.3 if mode == "timeout" else 5),
        cancel=cancel,
        journal=Journal(tmp_path, call_id="test"),
    )
    assert (
        result["execution"]
        == {"timeout": "timeout", "cancel": "cancelled", "parent_exit": "ok"}[mode]
    )
    time.sleep(1.1)
    assert not marker.exists()


def test_torn_journal_and_malformed_complete_record(tmp_path):
    journal = Journal(tmp_path, call_id="test")
    journal.emit("start")
    with journal.path.open("ab") as f:
        f.write(b'{"partial"')
    assert status(tmp_path)["torn_tail"]
    repaired = Journal(tmp_path, call_id="test")
    repaired.emit("next")
    rows, torn = read_events(journal.path)
    assert not torn and [e["sequence"] for e in rows] == [1, 2, 3]
    with journal.path.open("ab") as f:
        f.write(b"broken\n")
    with pytest.raises(ValueError):
        read_events(journal.path)


def test_remote_status_uses_heartbeat_when_filesystem_clock_is_behind(tmp_path):
    """Construct NFS clock skew at the filesystem boundary; query the real worker CLI."""
    root = tmp_path / "remote"
    job_id = "a" * 32
    directory = root / job_id
    journal = Journal(directory, call_id=job_id)
    journal.emit("process_heartbeat")
    earlier = time.time() - 120
    os.utime(journal.path, (earlier, earlier))
    worker = tmp_path / "worker.pyz"
    build_worker(worker)

    reply = subprocess.run(
        [sys.executable, str(worker), "status", str(root), job_id],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )

    observed = json.loads(reply.stdout)
    assert observed["state"] == "running"
    assert 0 <= observed["last_event_age_s"] < 15


@pytest.mark.parametrize("heartbeat_age", [120, -120, None])
def test_fresh_filesystem_metadata_does_not_certify_a_live_worker(tmp_path, heartbeat_age):
    root = tmp_path / "remote"
    job_id = "b" * 32
    directory = root / job_id
    directory.mkdir(parents=True)
    (directory / "request.json").write_text("{}")
    if heartbeat_age is not None:
        journal = Journal(directory, call_id=job_id)
        journal.emit("process_heartbeat")
        row = json.loads(journal.path.read_text())
        row["time"] = time.time() - heartbeat_age
        journal.path.write_text(json.dumps(row) + "\n")
    worker = tmp_path / "worker.pyz"
    build_worker(worker)

    reply = subprocess.run(
        [sys.executable, str(worker), "status", str(root), job_id],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )

    assert json.loads(reply.stdout)["state"] == "unknown"


@pytest.mark.skipif(not shutil.which("iverilog"), reason="optional Icarus not installed")
@pytest.mark.parametrize("case", ["pass", "fail", "syntax", "timeout"])
def test_actual_icarus(tmp_path, case):
    design = (EXAMPLE / "rtl/design.sv").read_text()
    tb = (EXAMPLE / "rtl/tb.sv").read_text()
    if case == "fail":
        design = design.replace("a ^ b", "a & b")
    elif case == "syntax":
        design = "bad verilog input"
    elif case == "timeout":
        tb = "module tb; initial begin forever #1; end endmodule"
    result = simulate(
        SimulationRequest(design, tb, timeout_s=1 if case == "timeout" else 10), tmp_path
    )
    assert result["execution"] == ("timeout" if case == "timeout" else "ok")
    assert (
        result["verdict"]
        == {"pass": "pass", "fail": "fail", "syntax": "fail", "timeout": "not_evaluated"}[case]
    )
