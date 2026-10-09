"""Owned POSIX process groups, durable logs, cancellation and deadlines."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from importlib.resources import files
from pathlib import Path

from circuit_harness.execution.runtime.journal import Journal

# Capture at this module's own load, even when current_evas is imported later.
_SOURCE_SHA256 = hashlib.sha256(files(__package__).joinpath("process.py").read_bytes()).hexdigest()

# Keep a live group leader until the owner has signalled the group. On macOS,
# getpgid/killpg of an exited leader can fail even before wait() reaps it.
# The guardian also imposes a backstop if its owner disappears unexpectedly.
_GUARDIAN = """
import json, os, signal, subprocess, sys, time
receipt, lifetime, *argv = sys.argv[1:]
deadline = time.monotonic() + float(lifetime)
try:
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
    result = {"returncode": child.wait(timeout=max(0.01, deadline-time.monotonic())),
              "pid": child.pid}
except subprocess.TimeoutExpired:
    os.killpg(os.getpgrp(), signal.SIGKILL)
except OSError as error:
    result = {"returncode": None, "error": str(error), "error_type": type(error).__name__}
with open(receipt + '.tmp', 'w') as stream:
    json.dump(result, stream)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(receipt + '.tmp', receipt)
while time.monotonic() < deadline:
    time.sleep(0.05)
os.killpg(os.getpgrp(), signal.SIGKILL)
"""


def _directory_output_bytes(directory: Path) -> int:
    total = 0
    for root, _, names in os.walk(directory, followlinks=False):
        for name in names:
            try:
                info = (Path(root) / name).lstat()
            except FileNotFoundError:
                continue  # A worker may atomically replace its receipt during traversal.
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def run_process(
    argv: list[str],
    *,
    directory: Path,
    stage: str,
    deadline: float,
    cancel: threading.Event,
    journal: Journal,
    max_output_bytes: int = 16 * 1024 * 1024,
) -> dict:
    """Run argv without a shell; resource control is not a security sandbox."""
    if os.name != "posix":
        raise RuntimeError("chips process cleanup currently requires POSIX")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = (
        directory / f"{stage}.{suffix}.log" for suffix in ("stdout", "stderr")
    )
    receipt = directory / f"{stage}.process.json"
    receipt.unlink(missing_ok=True)
    started = time.monotonic()
    execution, returncode, cleanup = "ok", None, True
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        if cancel.is_set() or started >= deadline:
            execution = "cancelled" if cancel.is_set() else "timeout"
        else:
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _GUARDIAN,
                        str(receipt),
                        str(deadline - started + 3),
                        *argv,
                    ],
                    cwd=directory,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
            except OSError as error:
                execution = "infrastructure_error"
                err.write(f"{type(error).__name__}: {error}".encode())
                journal.emit("process_start_failed", stage=stage, error_type=type(error).__name__)
            else:
                try:
                    journal.emit("process_started", stage=stage, group_pid=process.pid, argv=argv)
                    heartbeat = started
                    while True:
                        now = time.monotonic()
                        if cancel.is_set():
                            execution = "cancelled"
                            break
                        if now >= deadline:
                            execution = "timeout"
                            break
                        if _directory_output_bytes(directory) > max_output_bytes:
                            execution = "output_limit"
                            break
                        if receipt.exists():
                            finished = json.loads(receipt.read_text())
                            returncode = finished["returncode"]
                            if "error" in finished:
                                execution = "infrastructure_error"
                                err.write(finished["error"].encode())
                            break
                        if now - heartbeat >= 1:
                            journal.emit(
                                "process_heartbeat",
                                stage=stage,
                                elapsed_s=now - started,
                                remaining_s=max(0, deadline - now),
                            )
                            heartbeat = now
                        cancel.wait(min(0.05, max(0, deadline - now)))
                finally:
                    # Never poll/reap this owned leader before group cleanup.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError:
                        cleanup = False
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        cleanup = False
                    journal.emit("process_cleanup", stage=stage, cleanup_confirmed=cleanup)
    if not cleanup:
        execution = "cleanup_failed"
    result = {
        "execution": execution,
        "returncode": returncode,
        "elapsed_s": time.monotonic() - started,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
        "cleanup_confirmed": cleanup,
    }
    journal.emit("process_finished", stage=stage, **result)
    return result
