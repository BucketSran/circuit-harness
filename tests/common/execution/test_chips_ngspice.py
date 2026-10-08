"""RC analytical oracle plus real local process fixtures, never fake lab evidence."""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from alphaapollo.common.execution.chips.bundle import build_cli
from alphaapollo.common.execution.chips.journal import file_digest, read_events
from alphaapollo.common.execution.chips.ngspice import run_rc
from alphaapollo.workflows.chips import main


def test_missing_ngspice_is_a_recorded_dependency_failure(tmp_path, capsys):
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"resistance_ohm": 1000, "capacitance_f": 1e-9}))
    output = tmp_path / "run"
    code = main(
        [
            "ngspice-rc",
            "--input",
            str(task),
            "--output",
            str(output),
            "--ngspice",
            str(tmp_path / "absent-ngspice"),
        ]
    )
    assert code == 1
    result = json.loads((output / "result.json").read_text())
    assert result["execution"] == "infrastructure_error"
    assert result["reason"] == "dependency_missing"
    assert result["verdict"] == "not_evaluated"
    assert result["certified"] is False


def test_zero_exit_without_waveforms_is_not_success(tmp_path):
    tool = tmp_path / "empty-simulator"
    tool.write_text(f"#!{sys.executable}\nprint('constructed ngspice fixture')\n")
    tool.chmod(0o700)
    result = run_rc(
        {"resistance_ohm": 1000, "capacitance_f": 1e-9}, tmp_path / "run", ngspice=str(tool)
    )
    assert result["execution"] == "missing_output"
    assert result["verdict"] == "not_evaluated"
    assert (tmp_path / "run/ngspice.stdout.log").exists()


@pytest.fixture
def constructed_simulator(tmp_path):
    # Analytical data fixture, NOT a SPICE simulation. Live evidence is separate.
    tool = tmp_path / "fixture-ngspice"
    tool.write_text(
        f"#!{sys.executable}\n"
        + """
import json, math, os, pathlib, sys, time
if '-v' in sys.argv:
    print('constructed ngspice test fixture')
    sys.exit(0)
p = pathlib.Path('.').resolve()
with (p.parent / 'launches').open('a') as stream:
    stream.write('simulation\\n')
time.sleep(float(os.environ.get('RC_FIXTURE_DELAY', '0')))
mode = os.environ.get('RC_FIXTURE_MODE', '')
if mode == 'sleep':
    time.sleep(30)
if mode == 'exit':
    sys.exit(7)
task = json.loads((p / 'request.json').read_text())['identity']['task']
tau = task['resistance_ohm'] * task['capacitance_f']
fc = 1 / (2 * math.pi * tau)
with (p / 'ac.dat').open('w') as f:
    for i in range(121):
        frequency = fc * 10 ** (-3 + i / 20)
        h = 1 / complex(1, frequency / fc)
        if mode == 'wrong': h *= 0.5
        f.write(f'{frequency} {h.real} {h.imag}\\n')
with (p / 'transient.dat').open('w') as f:
    for i in range(1001):
        t = tau * i / 100
        v = task['voltage_v'] * (1 - math.exp(-t / tau))
        f.write(f'{t} {v}\\n')
if mode == 'truncated':
    (p / 'ac.dat').write_text((p / 'ac.dat').read_text().splitlines()[0])
if mode == 'nan':
    (p / 'transient.dat').write_text('0 nan\\n')
"""
    )
    tool.chmod(0o700)
    return str(tool)


TASK = {"resistance_ohm": 1000, "capacitance_f": 1e-9}


def test_finished_run_resume_verifies_artifacts_without_solver_relaunch(
    tmp_path, constructed_simulator
):
    directory = tmp_path / "run"
    result = run_rc(TASK, directory, ngspice=constructed_simulator)
    assert result["execution"] == "ok" and result["verdict"] == "pass"
    assert result["metrics"]["ac_samples"] == 121
    assert run_rc(TASK, directory, ngspice=constructed_simulator, resume=True) == result
    assert (tmp_path / "launches").read_text() == "simulation\n"
    events, torn = read_events(directory / "events.jsonl")
    assert not torn and events[-1]["event"] == "result_reused"
    # Downloaded evidence must also be checkable without the server's executable.
    assert main(["verify-rc", str(directory)]) == 0
    (directory / "ac.dat").write_text("modified")
    assert main(["verify-rc", str(directory)]) == 2
    with pytest.raises(ValueError, match="artifact missing or modified"):
        run_rc(TASK, directory, ngspice=constructed_simulator, resume=True)


@pytest.mark.parametrize(
    "mode,execution,verdict",
    [
        ("wrong", "ok", "fail"),
        ("truncated", "invalid_output", "not_evaluated"),
        ("nan", "invalid_output", "not_evaluated"),
        ("exit", "tool_error", "not_evaluated"),
    ],
)
def test_bad_solver_evidence_is_rejected(
    tmp_path, constructed_simulator, monkeypatch, mode, execution, verdict
):
    monkeypatch.setenv("RC_FIXTURE_MODE", mode)
    result = run_rc(TASK, tmp_path / "run", ngspice=constructed_simulator)
    assert (result["execution"], result["verdict"]) == (execution, verdict)


def test_timeout_and_cancel_have_cleanup_evidence(tmp_path, constructed_simulator, monkeypatch):
    monkeypatch.setenv("RC_FIXTURE_MODE", "sleep")
    directory = tmp_path / "timeout"
    result = run_rc(TASK, directory, ngspice=constructed_simulator, timeout_s=0.5)
    assert result["execution"] == "timeout"
    events, _ = read_events(directory / "events.jsonl")
    assert any(row["event"] == "process_cleanup" and row["cleanup_confirmed"] for row in events)
    cancel = threading.Event()
    cancel.set()
    result = run_rc(TASK, tmp_path / "cancel", ngspice=constructed_simulator, cancel=cancel)
    assert result["execution"] == "cancelled"


def test_changed_tool_and_incomplete_run_cannot_be_resubmitted(tmp_path, constructed_simulator):
    directory = tmp_path / "run"
    run_rc(TASK, directory, ngspice=constructed_simulator)
    (directory / "result.json").unlink()
    with pytest.raises(ValueError, match="automatic resubmit refused"):
        run_rc(TASK, directory, ngspice=constructed_simulator, resume=True)
    with open(constructed_simulator, "a") as stream:
        stream.write("\n# changed tool\n")
    with pytest.raises(ValueError, match="identical task/tool/Harness"):
        run_rc(TASK, directory, ngspice=constructed_simulator, resume=True)
    assert (tmp_path / "launches").read_text() == "simulation\n"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {**TASK, "resistance_ohm": True},
        {**TASK, "capacitance_f": float("nan")},
        {**TASK, "shell": "exit"},
        {**TASK, "voltage_v": -1},
    ],
)
def test_invalid_inputs_are_rejected_before_execution(tmp_path, payload):
    with pytest.raises(ValueError):
        run_rc(payload, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_offline_bundle_runs_without_site_packages(tmp_path):
    bundle = tmp_path / "chips.pyz"
    assert build_cli(bundle) == build_cli(tmp_path / "second.pyz")
    task = tmp_path / "rc.json"
    task.write_text(json.dumps(TASK))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(bundle),
            "ngspice-rc",
            "--input",
            str(task),
            "--output",
            str(tmp_path / "run"),
            "--ngspice",
            str(tmp_path / "absent"),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["reason"] == "dependency_missing"
    # The packaged CLI must still build the pre-existing EMX remote worker.
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from pathlib import Path; "
            "from alphaapollo.common.execution.chips.emx import build_worker; "
            "build_worker(Path(sys.argv[2]))",
            str(bundle),
            str(tmp_path / "worker.pyz"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice is not installed locally")
def test_real_ngspice_rc_and_resume(tmp_path):
    result = run_rc(TASK, tmp_path / "run")
    assert (result["execution"], result["verdict"]) == ("ok", "pass"), result
    assert result["metrics"]["cutoff_hz"] == pytest.approx(159154.9430918953)
    assert run_rc(TASK, tmp_path / "run", resume=True) == result


def test_download_regrade_accepts_roundoff_but_not_changed_verdict(tmp_path, constructed_simulator):
    directory = tmp_path / "run"
    result = run_rc(TASK, directory, ngspice=constructed_simulator)
    # Model a tiny libm difference observed when regrading Linux data on macOS.
    result["metrics"]["transient_max_normalized_error"] += 1e-16
    metrics = directory / "metrics.json"
    metrics.write_text(json.dumps(result["metrics"]))
    result["artifacts"]["metrics.json"] = {
        "sha256": file_digest(metrics),
        "bytes": metrics.stat().st_size,
    }
    (directory / "result.json").write_text(json.dumps(result))
    assert main(["verify-rc", str(directory)]) == 0
    result["verdict"] = "fail"
    (directory / "result.json").write_text(json.dumps(result))
    assert main(["verify-rc", str(directory)]) == 2


def wait_for_file(path, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for {path}")


def test_detached_job_finishes_after_submitting_session_is_killed(
    tmp_path, constructed_simulator, monkeypatch
):
    monkeypatch.setenv("RC_FIXTURE_DELAY", "2")
    bundle = tmp_path / "chips.pyz"
    build_cli(bundle)
    task = tmp_path / "task.json"
    task.write_text(json.dumps(TASK))
    job = tmp_path / "jobs" / "disconnect"
    # A real submitting process/session, held open like an SSH monitoring session.
    # Only the solver's numerical output is a constructed analytical fixture.
    with (tmp_path / "client.log").open("w") as log:
        client = subprocess.Popen(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys,time; sys.path.insert(0,sys.argv[1]); "
                "from alphaapollo.workflows.chips import main; "
                "code=main(sys.argv[2:]); print('submitted',code,flush=True); "
                "time.sleep(30) if code == 0 else sys.exit(code)",
                str(bundle),
                "submit-rc",
                "--input",
                str(task),
                "--root",
                str(job.parent),
                "--job-id",
                job.name,
                "--ngspice",
                constructed_simulator,
                "--timeout",
                "10",
                "--archive-root",
                str(tmp_path / "archives"),
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            wait_for_file(job / "launches")
            assert not (job / "completion.json").exists()
            os.killpg(client.pid, signal.SIGKILL)
            client.wait(timeout=5)
            wait_for_file(job / "completion.json")
        finally:
            if client.poll() is None:
                os.killpg(client.pid, signal.SIGKILL)
                client.wait(timeout=5)
    from alphaapollo.common.execution.chips.jobs import inspect_job, verify_job

    assert inspect_job(job)["state"] == "finished"
    assert verify_job(job)["result"]["verdict"] == "pass"
    assert (job / "launches").read_text() == "simulation\n"
    wait_for_archive(job, "verified")
    assert main(["verify-archive", str(tmp_path / "archives/disconnect")]) == 0


def test_detached_job_id_prevents_duplicate_submission_and_pins_bundle(
    tmp_path, constructed_simulator, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    from alphaapollo.common.execution.chips.jobs import submit_rc, verify_job

    monkeypatch.setenv("RC_FIXTURE_DELAY", "1")
    root = tmp_path / "jobs"
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(
            pool.map(
                lambda _: submit_rc(TASK, root, "one", ngspice=constructed_simulator), range(2)
            )
        )
    assert all(reply["job_id"] == "one" for reply in replies)
    job = root / "one"
    wait_for_file(job / "completion.json")
    assert submit_rc(TASK, root, "one", ngspice=constructed_simulator)["state"] == "finished"
    assert verify_job(job)["result"]["verdict"] == "pass"
    request = json.loads((job / "request.json").read_text())
    assert file_digest(job / "worker.pyz") == request["worker_sha256"]
    assert (job / "launches").read_text() == "simulation\n"
    with pytest.raises(ValueError, match="different task"):
        submit_rc({**TASK, "resistance_ohm": 2000}, root, "one", ngspice=constructed_simulator)
    (job / "run/ac.dat").write_text("tampered")
    with pytest.raises(ValueError, match="artifact missing or modified"):
        verify_job(job)


@pytest.mark.parametrize("action", ["timeout", "cancel"])
def test_detached_worker_records_interruptions_without_client_supervision(
    tmp_path, constructed_simulator, monkeypatch, action
):
    from alphaapollo.common.execution.chips.jobs import cancel_job, submit_rc, verify_job

    monkeypatch.setenv("RC_FIXTURE_MODE", "sleep")
    root = tmp_path / "jobs"
    job = root / action
    submit_rc(
        TASK,
        root,
        action,
        ngspice=constructed_simulator,
        # Allow the detached zipapp's version probe to finish before timing out
        # the intentionally sleeping simulation on slower hosts.
        timeout_s=3 if action == "timeout" else 10,
    )
    wait_for_file(job / "launches")
    if action == "cancel":
        assert cancel_job(job)["cancel_requested"]
    wait_for_file(job / "completion.json")
    result = verify_job(job)["result"]
    assert result["execution"] == ("cancelled" if action == "cancel" else "timeout")
    assert result["verdict"] == "not_evaluated"
    events, _ = read_events(job / "run/events.jsonl")
    assert any(e["event"] == "process_cleanup" and e["cleanup_confirmed"] for e in events)
    assert (job / "launches").read_text() == "simulation\n"


def test_dead_worker_is_unknown_and_never_implicitly_restarted(tmp_path, constructed_simulator):
    from alphaapollo.common.execution.chips.jobs import inspect_job, submit_rc

    root = tmp_path / "jobs"
    submit_rc(TASK, root, "incomplete", ngspice=constructed_simulator)
    job = root / "incomplete"
    wait_for_file(job / "completion.json")
    # Construct an interrupted finalization from a finished fixture. This does
    # not claim a real crash; it checks the unknown-state / no-relaunch contract.
    (job / "completion.json").unlink()
    deadline = time.monotonic() + 5
    while inspect_job(job)["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert inspect_job(job)["state"] == "unknown"
    assert submit_rc(TASK, root, "incomplete", ngspice=constructed_simulator)["state"] == "unknown"
    assert (job / "launches").read_text() == "simulation\n"


def test_detached_missing_dependency_is_a_finished_failure(tmp_path):
    from alphaapollo.common.execution.chips.jobs import submit_rc, verify_job

    root = tmp_path / "jobs"
    submit_rc(TASK, root, "missing", ngspice=str(tmp_path / "absent"))
    job = root / "missing"
    wait_for_file(job / "completion.json")
    assert verify_job(job)["result"]["reason"] == "dependency_missing"
    assert main(["verify-job", str(job)]) == 1


@pytest.mark.parametrize("job_id", ["../escape", "", "a/b", "x" * 81])
def test_invalid_detached_job_ids_have_no_side_effects(tmp_path, job_id):
    from alphaapollo.common.execution.chips.jobs import submit_rc

    with pytest.raises(ValueError, match="job_id"):
        submit_rc(TASK, tmp_path / "jobs", job_id)
    assert not (tmp_path / "jobs").exists()


def test_detached_job_archives_sealed_evidence_and_keeps_work(tmp_path, constructed_simulator):
    task = tmp_path / "task.json"
    task.write_text(json.dumps(TASK))
    work, archive = tmp_path / "scratch", tmp_path / "persistent"
    assert (
        main(
            [
                "submit-rc",
                "--input",
                str(task),
                "--root",
                str(work),
                "--job-id",
                "archived",
                "--ngspice",
                constructed_simulator,
                "--archive-root",
                str(archive),
            ]
        )
        == 0
    )
    job, record = work / "archived", archive / "archived"
    wait_for_file(record / "receipt.json")
    assert main(["verify-archive", str(record)]) == 0
    assert main(["verify-job", str(job)]) == 0
    assert (record / "job.tar.gz").is_file()
    assert (job / "launches").read_text() == "simulation\n"
    assert json.loads((job / "archive-status.json").read_text())["state"] == "verified"


def wait_for_archive(job, state, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        path = job / "archive-status.json"
        if path.exists() and json.loads(path.read_text())["state"] == state:
            return
        time.sleep(0.05)
    pytest.fail(f"archive did not reach {state}: {job}")


def test_archive_retry_does_not_rerun_completed_simulation(
    tmp_path, constructed_simulator, monkeypatch
):
    from alphaapollo.common.execution.chips.jobs import inspect_job, submit_rc

    monkeypatch.setenv("RC_FIXTURE_DELAY", "1")
    work, archive = tmp_path / "scratch", tmp_path / "persistent"
    submit_rc(TASK, work, "retry", ngspice=constructed_simulator, archive_root=archive)
    job, record = work / "retry", archive / "retry"
    # Construct a publication failure in this isolated job, not a real NFS outage.
    (record / "job.tar.gz.partial").mkdir()
    wait_for_archive(job, "failed")
    sealed = (job / "completion.json").read_bytes()
    assert inspect_job(job)["result"]["verdict"] == "pass"
    assert not (record / "receipt.json").exists()
    (record / "job.tar.gz.partial").rmdir()
    assert main(["job-archive", str(job)]) == 0
    wait_for_archive(job, "verified")
    assert (job / "completion.json").read_bytes() == sealed
    assert (job / "launches").read_text() == "simulation\n"
    assert main(["verify-archive", str(record)]) == 0


def test_cleanup_requires_verified_archive_and_preserves_job_identity(
    tmp_path, constructed_simulator
):
    from alphaapollo.common.execution.chips.jobs import inspect_job, submit_rc

    work, archive = tmp_path / "scratch", tmp_path / "persistent"
    submit_rc(TASK, work, "clean", ngspice=constructed_simulator, archive_root=archive)
    job, record = work / "clean", archive / "clean"
    wait_for_archive(job, "verified")
    package = record / "job.tar.gz"
    original = package.read_bytes()
    package.write_bytes(original[:100])
    assert main(["job-cleanup", str(job)]) == 2
    assert (job / "run/ac.dat").exists()
    package.write_bytes(original)
    assert main(["job-cleanup", str(job)]) == 0
    assert not (job / "run").exists()
    assert inspect_job(job)["state"] == "finished"
    assert (
        submit_rc(TASK, work, "clean", ngspice=constructed_simulator, archive_root=archive)["state"]
        == "finished"
    )
    assert not (job / "run").exists()
    # Model later scratch expiration: the persistent registration still blocks relaunch.
    shutil.rmtree(job)
    assert (
        submit_rc(TASK, work, "clean", ngspice=constructed_simulator, archive_root=archive)["state"]
        == "finished"
    )
    assert not job.exists()
    with pytest.raises(ValueError, match="different"):
        submit_rc(
            {**TASK, "resistance_ohm": 2000},
            work,
            "clean",
            ngspice=constructed_simulator,
            archive_root=archive,
        )


def test_job_timings_separate_backend_process_from_total(
    tmp_path, constructed_simulator, monkeypatch, capsys
):
    from alphaapollo.common.execution.chips.jobs import submit_rc

    monkeypatch.setenv("RC_FIXTURE_DELAY", "0.3")
    submit_rc(TASK, tmp_path / "jobs", "timed", ngspice=constructed_simulator)
    job = tmp_path / "jobs/timed"
    wait_for_file(job / "completion.json")
    assert main(["job-timings", str(job)]) == 0
    timings = json.loads(capsys.readouterr().out)
    assert timings["job_s"]["run_backend"] >= timings["backend_s"]["process:ngspice"] >= 0.3
    assert timings["job_s"]["elapsed"] >= timings["job_s"]["run_backend"]
    assert timings["backend_s"]["parse_and_grade"] > 0
    assert timings["backend_s"]["prepare_input"] > 0
    assert timings["archive"] == {"state": "not_configured"}


def test_registered_job_with_lost_scratch_is_unknown_not_resubmitted(tmp_path, monkeypatch):
    from alphaapollo.common.execution.chips import jobs

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *args, **kwargs: None)
    monkeypatch.setattr(jobs, "inspect_job", lambda directory: {"state": "running"})
    root, archive = tmp_path / "jobs", tmp_path / "archives"
    jobs.submit_rc(TASK, root, "lost", archive_root=archive)
    shutil.rmtree(root / "lost")
    assert jobs.submit_rc(TASK, root, "lost", archive_root=archive)["state"] == "unknown"
    assert not (root / "lost").exists()
    with pytest.raises(ValueError, match="different"):
        jobs.submit_rc(TASK, tmp_path / "other-work", "lost", archive_root=archive)


@pytest.mark.parametrize("archive_path", ["jobs", "jobs/nested", "."])
def test_archive_roots_must_be_disjoint(tmp_path, archive_path):
    from alphaapollo.common.execution.chips.jobs import submit_rc

    with pytest.raises(ValueError, match="disjoint"):
        submit_rc(TASK, tmp_path / "jobs", "overlap", archive_root=tmp_path / archive_path)
    assert not (tmp_path / "jobs/overlap").exists()


def test_failed_simulation_is_also_archived(tmp_path):
    from alphaapollo.common.execution.chips.jobs import submit_rc

    submit_rc(
        TASK,
        tmp_path / "jobs",
        "missing",
        ngspice=str(tmp_path / "no-tool"),
        archive_root=tmp_path / "archives",
    )
    wait_for_archive(tmp_path / "jobs/missing", "verified")
    assert main(["verify-archive", str(tmp_path / "archives/missing")]) == 0
    receipt = json.loads((tmp_path / "archives/missing/receipt.json").read_text())
    assert receipt["completion"]["result"]["execution"] == "infrastructure_error"


def test_archive_rejects_unsafe_member_even_with_matching_package_hash(
    tmp_path, constructed_simulator
):
    import io
    import tarfile

    from alphaapollo.common.execution.chips.jobs import submit_rc

    submit_rc(
        TASK,
        tmp_path / "jobs",
        "unsafe",
        ngspice=constructed_simulator,
        archive_root=tmp_path / "archives",
    )
    wait_for_archive(tmp_path / "jobs/unsafe", "verified")
    record = tmp_path / "archives/unsafe"
    package = record / "job.tar.gz"
    with tarfile.open(package, "w:gz") as tar:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    receipt = json.loads((record / "receipt.json").read_text())
    receipt["package"] = {"sha256": file_digest(package), "bytes": package.stat().st_size}
    receipt["members"]["../escape"] = {"sha256": "unused", "bytes": 1}
    (record / "receipt.json").write_text(json.dumps(receipt))
    assert main(["verify-archive", str(record)]) == 2
    assert main(["job-cleanup", str(tmp_path / "jobs/unsafe")]) == 2
    assert (tmp_path / "jobs/unsafe/run/ac.dat").is_file()
