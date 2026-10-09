"""Stdlib-only remote job protocol, shipped with process/journal in a zipapp."""

from __future__ import annotations

import fcntl
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from circuit_harness.execution.runtime.journal import Journal, atomic_json, file_digest, read_events
from circuit_harness.execution.runtime.process import run_process


def job_status(directory: Path) -> dict:
    if not directory.exists():
        return {"state": "missing"}
    result = directory / "result.json"
    if result.exists():
        return {"state": "finished", "result": json.loads(result.read_text())}
    # NFS metadata may use a different clock from the host executing this worker.
    # Compare application timestamps written and read on the execution host.
    rows, _ = read_events(directory / "events.jsonl")
    age = time.time() - rows[-1]["time"] if rows else None
    return {
        "state": "running" if age is not None and 0 <= age < 15 else "unknown",
        "last_event_age_s": age,
    }


def submit(root: Path, job_id: str, expected_hash: str) -> dict:
    directory = root / job_id
    request_file = root / f"{job_id}.request.json"
    if directory.exists():
        receipt = directory / "request.json"
        if not receipt.exists() or file_digest(receipt) != expected_hash:
            raise ValueError("existing job identity mismatch or incomplete receipt")
        return job_status(directory)
    if file_digest(request_file) != expected_hash:
        raise ValueError("request transfer hash mismatch")
    request = json.loads(request_file.read_text())
    if file_digest(root / f"{job_id}.gds") != request["gds_sha256"]:
        raise ValueError("GDS transfer hash mismatch")
    # The root lock serializes duplicate submits, including after lost SSH replies.
    directory.mkdir()
    shutil.copyfile(request_file, directory / "request.json")
    shutil.copyfile(root / f"{job_id}.gds", directory / "input.gds")
    Journal(directory, call_id=job_id).emit("job_submitted")
    subprocess.Popen(
        [sys.executable, sys.argv[0], "execute", str(root), job_id],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return job_status(directory)


def execute(directory: Path) -> None:
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "result.json").exists():
            return
        request = json.loads((directory / "request.json").read_text())
        journal = Journal(directory, call_id=directory.name)
        cancel, stop = threading.Event(), threading.Event()

        def watch_cancel():
            while not stop.wait(0.1):
                if (directory / "cancel").exists():
                    cancel.set()
                    return

        monitor = threading.Thread(target=watch_cancel, daemon=True)
        monitor.start()
        try:
            command = [
                part.replace("{gds}", str(directory / "input.gds")).replace(
                    "{job_dir}", str(directory)
                )
                for part in request["command"]
            ]
            executable = shutil.which(command[0])
            if executable is None:
                raise FileNotFoundError("configured EMX executable not found")
            command[0] = executable
            provenance = {
                str(path): file_digest(Path(path)) for path in [executable, *request["input_files"]]
            }
            atomic_json(directory / "provenance.json", provenance)
            journal.emit("run_started", command=command, timeout_s=request["timeout_s"])
            result = run_process(
                command,
                directory=directory,
                stage="emx",
                deadline=time.monotonic() + request["timeout_s"],
                cancel=cancel,
                journal=journal,
                max_output_bytes=request["max_output_bytes"],
            )
            if result["execution"] == "ok" and result["returncode"] != 0:
                result["execution"] = "tool_error"
            names = [
                "input.gds",
                "request.json",
                "provenance.json",
                "emx.stdout.log",
                "emx.stderr.log",
                *request["outputs"],
            ]
            missing = [name for name in request["outputs"] if not (directory / name).is_file()]
            if result["execution"] == "ok" and missing:
                result["execution"] = "missing_output"
            result.update(verdict="not_evaluated", certified=False, missing_outputs=missing)
            artifacts = {}
            for name in names:
                path = directory / name
                if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                    raise ValueError("output escapes job directory")
                if path.is_file():
                    artifacts[name] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
            result["artifacts"] = artifacts
        except Exception as error:
            result = {
                "execution": "infrastructure_error",
                "verdict": "not_evaluated",
                "certified": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "artifacts": {},
            }
        finally:
            stop.set()
            monitor.join(timeout=1)
        journal.emit("run_finished", execution=result["execution"], verdict=result["verdict"])
        result["artifacts"]["events.jsonl"] = {
            "sha256": file_digest(directory / "events.jsonl"),
            "bytes": (directory / "events.jsonl").stat().st_size,
        }
        atomic_json(directory / "result.json", result)


def main() -> None:
    action, root_arg, job_id, *extra = sys.argv[1:]
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise ValueError("invalid job ID")
    root = Path(root_arg).resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / job_id
    if action == "execute":
        execute(directory)
        return
    with (root / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if action == "submit":
            reply = submit(root, job_id, extra[0])
        elif action == "status":
            reply = job_status(directory)
        elif action == "cancel":
            reply = job_status(directory)
            if reply["state"] in {"running", "unknown"}:
                (directory / "cancel").touch()
                reply = {"state": "cancellation_requested"}
        else:
            raise ValueError("unknown worker action")
    print(json.dumps({"job_id": job_id, **reply}, allow_nan=False))


if __name__ == "__main__":
    main()
