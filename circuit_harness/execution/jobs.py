"""Detached, single-host simulator jobs: submission is separate from server execution.

A private directory and stable job ID own one attempt. Unknown attempts are never
restarted automatically. This is not a reboot-persistent scheduler.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .archive import archive_job, archived_job_status, private_root, reserve_archive, verify_archive
from .benchmark_spectre import benchmark_identity, run_benchmark, stage_inputs, verify_benchmark
from .bundle import build_cli
from .journal import Journal, atomic_json, file_digest, read_events
from .ngspice import rc_identity, run_rc, verify_rc
from .spectre import run_spectre_rc, spectre_identity, verify_spectre_rc
from .spectre_testbench import run_testbench, testbench_identity, verify_testbench
from .vabench import freeze_candidate, run_vabench, vabench_identity, verify_vabench


def inspect_job(directory: Path) -> dict:
    """Read status without advancing, retrying, or signalling the job."""
    directory = Path(directory)
    if not directory.exists():
        return {"job_id": directory.name, "state": "missing"}
    if (directory / "archived.json").exists():
        request = json.loads((directory / "request.json").read_text())
        return archived_job_status(Path(request["archive_directory"]))
    completion = directory / "completion.json"
    if completion.exists():
        state = {
            "job_id": directory.name,
            "state": "finished",
            **json.loads(completion.read_text()),
        }
        request = json.loads((directory / "request.json").read_text())
        if "archive_directory" in request:
            local_status = directory / "archive-status.json"
            state["archive"] = (
                json.loads(local_status.read_text())
                if local_status.exists()
                else {"state": "pending"}
            )
        return state
    running = False
    lock_path = directory / ".worker.lock"
    if lock_path.exists():
        with lock_path.open("r") as lock:
            try:
                # Linux NFS emulates flock with record locks: an exclusive lock
                # needs a writable descriptor. A shared read probe still conflicts
                # with the worker's exclusive lock and never writes status files.
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                running = (directory / "worker.json").exists()
    return {
        "job_id": directory.name,
        "state": "running" if running else "unknown",
        "cancel_requested": (directory / "cancel").exists(),
    }


def submit_rc(
    payload: dict,
    root: Path,
    job_id: str,
    *,
    ngspice="ngspice",
    timeout_s=60,
    archive_root: Path | None = None,
) -> dict:
    """Acknowledge only a started worker or a finished attempt; retries never relaunch."""
    return _submit(
        rc_identity(payload, ngspice, timeout_s), root, job_id, archive_root=archive_root
    )


def submit_vabench(
    pin: dict,
    submission: Path,
    root: Path,
    job_id: str,
    *,
    timeout_s=300,
    archive_root: Path | None = None,
) -> dict:
    """Freeze an operator submission before acknowledging a detached replay."""
    identity = vabench_identity(pin, submission, timeout_s)
    return _submit(identity, root, job_id, candidate=submission, archive_root=archive_root)


def submit_spectre_rc(payload: dict, profile_path: Path, job_id: str) -> dict:
    """Submit one operator-configured Spectre RC run with private work and archive roots."""
    identity = spectre_identity(payload, profile_path)
    return _submit(
        identity,
        Path(identity["run_root"]),
        job_id,
        archive_root=Path(identity["archive_root"]),
    )


def submit_spectre_gain(payload: dict, profile_path: Path, job_id: str) -> dict:
    """Submit an operator-confirmed gain testbench through the existing durable worker."""
    identity = testbench_identity(payload, profile_path)
    profile = identity["profile"]
    return _submit(
        identity, Path(profile["run_root"]), job_id, archive_root=Path(profile["archive_root"])
    )


def submit_benchmark_spectre(
    candidate: Path,
    task_package: Path,
    profile_path: Path,
    job_id: str,
    *,
    purpose="final",
    isolated_public=False,
) -> dict:
    """Submit a frozen benchmark candidate once; query unknown attempts, never restart."""
    identity = benchmark_identity(
        candidate, task_package, profile_path, purpose=purpose, isolated_public=isolated_public
    )
    config = identity["profile"]["configuration"]
    root = Path(config["run_root"])
    reply = _submit(
        identity,
        root,
        job_id,
        benchmark_inputs=(candidate, task_package),
        archive_root=Path(config["archive_root"]),
    )
    return {**reply, "directory": str(root.resolve() / job_id)}


def _submit(
    identity: dict,
    root: Path,
    job_id: str,
    *,
    candidate: Path | None = None,
    benchmark_inputs: tuple[Path, Path] | None = None,
    archive_root: Path | None = None,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job_id):
        raise ValueError("job_id must contain 1-80 letters, digits, underscores or hyphens")
    root = Path(root).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise ValueError("job root must be owned by this user with mode 0700")
    archive_root = Path(archive_root).resolve() if archive_root is not None else None
    if archive_root is not None:
        if root == archive_root or root in archive_root.parents or archive_root in root.parents:
            raise ValueError("work and archive roots must be disjoint")
        private_root(archive_root)
    directory = root / job_id
    # Serialize preparation/spawn and retry lookup. The worker owns another lock.
    with (root / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if directory.exists():
            if directory.is_symlink():
                raise ValueError("job directory cannot be a symlink")
            request = json.loads((directory / "request.json").read_text())
            if request["identity"] != identity:
                raise ValueError("job ID already belongs to a different task/tool/Harness")
            if request.get("archive_directory") != (
                str(archive_root / job_id) if archive_root else None
            ):
                raise ValueError("job ID already has a different archive configuration")
            return inspect_job(directory)
        if archive_root is not None:
            if not reserve_archive(archive_root / job_id, directory, identity):
                return archived_job_status(archive_root / job_id)
        directory.mkdir(mode=0o700)
        if benchmark_inputs is not None:
            stage_inputs(*benchmark_inputs, directory, identity)
        if candidate is not None:
            freeze_candidate(candidate, directory / "candidate", identity)
        bundle = directory / "worker.pyz"
        bundle_hash = build_cli(bundle)
        atomic_json(
            directory / "request.json",
            {
                "schema_version": 1,
                "job_id": job_id,
                "identity": identity,
                "worker_sha256": bundle_hash,
                **({"archive_directory": str(archive_root / job_id)} if archive_root else {}),
            },
        )
        _launch_worker(directory)
    # This wait is only a handoff acknowledgement; it never drives the simulation.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = inspect_job(directory)
        if state["state"] != "unknown":
            return state
        time.sleep(0.05)
    return inspect_job(directory)


def _launch_worker(directory: Path) -> None:
    with (
        (directory / "worker.stdout.log").open("ab") as out,
        (directory / "worker.stderr.log").open("ab") as err,
    ):
        subprocess.Popen(
            [sys.executable, "-S", str(directory / "worker.pyz"), "job-execute", str(directory)],
            cwd=directory,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
            close_fds=True,
            umask=0o077,
        )


def retry_archive(directory: Path) -> dict:
    """Start a detached archive-only attempt using the original pinned worker."""
    directory = Path(directory).resolve()
    if (directory / "archived.json").exists():
        return inspect_job(directory)
    if not (directory / "completion.json").is_file():
        raise ValueError("archive retry requires a completed job; simulation is never retried")
    request = json.loads((directory / "request.json").read_text())
    if "archive_directory" not in request:
        raise ValueError("job was not submitted with --archive-root")
    if file_digest(directory / "worker.pyz") != request["worker_sha256"]:
        raise ValueError("worker bundle changed after submission")
    _launch_worker(directory)
    return {"job_id": directory.name, "archive_retry": "dispatched"}


def cleanup_job(directory: Path) -> dict:
    """Explicitly release work files only after independently verifying the archive.

    Keep a small tombstone and request to prevent duplicate submission while the
    scratch root exists. The persistent registration survives scratch expiration.
    """
    directory = Path(directory).absolute()
    if directory.is_symlink():
        raise ValueError("job directory cannot be a symlink")
    directory = directory.resolve()
    request = json.loads((directory / "request.json").read_text())
    if "archive_directory" not in request:
        raise ValueError("cleanup requires a configured verified archive")
    with (directory / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("worker is still active; retry cleanup after it exits") from error
        record = Path(request["archive_directory"])
        receipt = verify_archive(record)
        registration = json.loads((record / "registration.json").read_text())
        if receipt["request"] != request or registration["work_directory"] != str(directory):
            raise ValueError("archive belongs to a different work directory")
        if not (directory / "archived.json").exists():
            if verify_job(directory) != receipt["completion"]:
                raise ValueError("archive completion differs from work directory")
            atomic_json(directory / "archived.json", {"archive_directory": str(record)})
        for path in directory.iterdir():
            if path.name in {"request.json", "archived.json", ".worker.lock"}:
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    return {"job_id": directory.name, "work_files": "removed", "archive": str(record)}


def cancel_job(directory: Path) -> dict:
    """Request cooperative cancellation without PID reuse or process-group guesses."""
    state = inspect_job(directory)
    if state["state"] in {"running", "unknown"}:
        (Path(directory) / "cancel").touch(mode=0o600)
        return {**state, "cancel_requested": True}
    return state


def execute_job(directory: Path) -> None:
    """Own the whole server pipeline, including grading and final evidence receipts."""
    directory = Path(directory).resolve()
    # No controlling terminal and no SSH pipe is inherited. Also ignore a direct
    # hangup; explicit cancellation uses the file or SIGTERM, never a disconnect.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    cancel = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancel.set())
    with (directory / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (directory / "archived.json").exists():
            return
        if (directory / "completion.json").exists():
            if "archive_directory" in json.loads((directory / "request.json").read_text()):
                archive_job(directory)
            return
        if (directory / "worker.json").exists():
            return  # An interrupted attempt requires inspection, not a hidden restart.
        atomic_json(
            directory / "worker.json",
            {
                "pid": os.getpid(),
                "started_at": time.time(),
                "python": sys.executable,
                "python_version": sys.version,
            },
        )
        journal = Journal(directory, call_id=directory.name)
        started = time.monotonic()
        journal.emit("job_started")
        stop = threading.Event()

        def watch_cancel():
            while not stop.is_set():
                if (directory / "cancel").exists():
                    cancel.set()
                stop.wait(0.05)

        watcher = threading.Thread(target=watch_cancel, daemon=True)
        watcher.start()
        request = {}
        try:
            with journal.stage("validate_request"):
                request = json.loads((directory / "request.json").read_text())
                identity = request["identity"]
                if file_digest(directory / "worker.pyz") != request["worker_sha256"]:
                    raise ValueError("worker bundle changed after submission")
                backend = identity.get("backend", "ngspice_rc")
                if backend == "spectre_rc":
                    if (
                        spectre_identity(identity["task"], Path(identity["profile_path"]))
                        != identity
                    ):
                        raise ValueError(
                            "Spectre task/profile/tool/Harness changed after submission"
                        )
                elif backend == "ngspice_rc":
                    if (
                        rc_identity(identity["task"], identity["executable"], identity["timeout_s"])
                        != identity
                    ):
                        raise ValueError("task/tool/Harness changed after submission")
                elif backend not in {"vabench", "spectre_gain", "benchmark_spectre"}:
                    raise ValueError("unknown chips backend")
            with journal.stage("run_backend"):
                if backend == "benchmark_spectre":
                    result = run_benchmark(identity, directory / "run", cancel)
                elif backend == "vabench":
                    result = run_vabench(
                        identity, directory / "candidate", directory / "run", cancel
                    )
                elif backend == "spectre_gain":
                    result = run_testbench(identity, directory / "run", cancel)
                elif backend == "spectre_rc":
                    result = run_spectre_rc(identity, directory / "run", cancel)
                else:
                    result = run_rc(
                        identity["task"],
                        directory / "run",
                        ngspice=identity["executable"],
                        timeout_s=identity["timeout_s"],
                        cancel=cancel,
                    )
            with journal.stage("verify_result"):
                {
                    "benchmark_spectre": verify_benchmark,
                    "vabench": verify_vabench,
                    "spectre_rc": verify_spectre_rc,
                    "spectre_gain": verify_testbench,
                    "ngspice_rc": verify_rc,
                }[backend](directory / "run")
        except Exception as error:
            result = {
                "execution": "infrastructure_error",
                "verdict": "not_evaluated",
                "reason": f"{type(error).__name__}: {error}",
                "certified": False,
            }
            identity = request.get("identity", {})
            if identity.get("backend") == "benchmark_spectre":
                frozen = identity["candidate_manifest"]
                result.update(
                    score=None,
                    backend="spectre",
                    criteria_sha256=identity["package"]["manifest"]["criteria_sha256"],
                    condition_id=identity["package"]["manifest"]["condition_id"],
                    task_set=identity["package"]["manifest"]["task_set"],
                    purpose=identity["purpose"],
                    task_id=frozen["task_id"],
                    task_version=frozen["task_version"],
                    candidate_sha256=frozen["candidate_sha256"],
                )
        finally:
            stop.set()
            watcher.join()
        journal.emit("job_finished", execution=result["execution"], verdict=result["verdict"])
        sealing_started = time.monotonic()
        paths = [
            directory / name
            for name in ("request.json", "worker.json", "worker.pyz", "events.jsonl")
        ]
        for name in ("run", "candidate", "task-package"):
            if (directory / name).exists():
                paths.extend(
                    p
                    for p in (directory / name).rglob("*")
                    if p.name != ".lock" and "__pycache__" not in p.parts
                )
        artifacts = {
            str(path.relative_to(directory)): {
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(paths)
            if path.is_file() and not path.is_symlink()
        }
        atomic_json(
            directory / "completion.json",
            {
                "schema_version": 1,
                "finished_at": time.time(),
                "result": result,
                "artifacts": artifacts,
                "timings": {
                    "job_s": {
                        **_stage_times(directory),
                        "seal_artifacts": time.monotonic() - sealing_started,
                        "elapsed": time.monotonic() - started,
                    },
                    "backend_s": _stage_times(directory / "run"),
                    "note": "Backend stages are within run_backend; do not sum parent and child. "
                    "process:replay includes original simulator and scorer. "
                    "elapsed excludes final completion publication and archiving.",
                },
            },
        )

        if "archive_directory" in request:
            archive_job(directory)


def _stage_times(directory: Path) -> dict:
    events, _ = read_events(directory / "events.jsonl")
    return {
        ("process:" if event["event"] == "process_finished" else "") + event["stage"]: event[
            "elapsed_s"
        ]
        for event in events
        if event["event"] in {"process_finished", "stage_finished"}
    }


def job_timings(directory: Path) -> dict:
    """Read sealed measurements from work or archive; old jobs remain unmeasured."""
    if (directory / "receipt.json").exists():
        state = archived_job_status(directory)
    else:
        state = inspect_job(directory)
    timings = state.get("timings", {"state": "not_measured"})
    return {**timings, "archive": state.get("archive", {"state": "not_configured"})}


def verify_job(directory: Path) -> dict:
    """Verify a downloaded completed job, including backend-specific receipt checks."""
    directory = Path(directory).resolve()
    completion = json.loads((directory / "completion.json").read_text())
    for name, receipt in completion["artifacts"].items():
        path = directory / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or path.is_symlink()
            or directory not in path.resolve().parents
            or not path.is_file()
            or path.stat().st_size != receipt["bytes"]
            or file_digest(path) != receipt["sha256"]
        ):
            raise ValueError(f"job artifact missing or modified: {name}")
    if "run/result.json" in completion["artifacts"]:
        request = json.loads((directory / "request.json").read_text())
        backend = request["identity"].get("backend", "ngspice_rc")
        verify = {
            "benchmark_spectre": verify_benchmark,
            "vabench": verify_vabench,
            "spectre_rc": verify_spectre_rc,
            "spectre_gain": verify_testbench,
            "ngspice_rc": verify_rc,
        }[backend]
        if (
            backend == "benchmark_spectre"
            and json.loads((directory / "run/identity.json").read_text()) != request["identity"]
        ):
            raise ValueError("benchmark run identity differs from job request")
        result = verify(directory / "run")
        if result != completion["result"]:
            raise ValueError("job completion differs from run result")
    elif completion["result"]["execution"] == "ok":
        raise ValueError("successful job has no run result receipt")
    return completion
