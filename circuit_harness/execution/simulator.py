"""Icarus compile/simulate receipts and content-bound stage recovery."""

from __future__ import annotations

import fcntl
import json
import math
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .journal import Journal, atomic_json, digest
from .process import run_process


@dataclass(frozen=True)
class SimulationRequest:
    design: str
    testbench: str
    top: str = "tb"
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        for name in ("design", "testbench"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value.encode()) > 1024 * 1024:
                raise ValueError(f"{name} must contain 1..1048576 bytes of Verilog")
        if not isinstance(self.top, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", self.top):
            raise ValueError("top must be a Verilog identifier")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or not 0 < self.timeout_s <= 600
        ):
            raise ValueError("timeout_s must be finite and in (0, 600]")


def simulate(
    request: SimulationRequest,
    directory: Path,
    *,
    call_id: str = "direct",
    cancel: threading.Event | None = None,
    resume: bool = False,
    iverilog: str = "iverilog",
    vvp: str = "vvp",
) -> dict:
    """The supplied testbench is a development check, never a certification."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("simulation directory already has an active writer") from exc
        return _simulate(
            request, directory, call_id, cancel or threading.Event(), resume, iverilog, vvp
        )


def _simulate(request, directory, call_id, cancel, resume, iverilog, vvp):
    intent_path = directory / "request.json"
    intent = asdict(request)
    if intent_path.exists():
        if not resume:
            raise ValueError("run already exists; use resume with identical inputs")
        if json.loads(intent_path.read_text()) != intent:
            raise ValueError("resume input/configuration mismatch")
    else:
        atomic_json(intent_path, intent)
    journal = Journal(directory, call_id=call_id)
    attempt_id = uuid.uuid4().hex
    attempt = directory / "attempts" / attempt_id
    attempt.mkdir(parents=True)
    deadline = time.monotonic() + request.timeout_s
    journal.emit(
        "run_started",
        attempt_id=attempt_id,
        timeout_s=request.timeout_s,
        input_sha256=digest(json.dumps(intent, sort_keys=True).encode()),
    )
    tools = {}
    for name, command in (("iverilog", iverilog), ("vvp", vvp)):
        executable = shutil.which(command)
        if executable is None:
            return _finish(
                directory,
                attempt,
                journal,
                "infrastructure_error",
                "not_evaluated",
                "dependency_missing",
                missing=name,
            )
        probe = run_process(
            [executable, "-V"],
            directory=attempt,
            stage=f"probe_{name}",
            deadline=deadline,
            cancel=cancel,
            journal=journal,
        )
        if probe["execution"] != "ok" or probe["returncode"] != 0:
            return _finish(
                directory,
                attempt,
                journal,
                probe["execution"] if probe["execution"] != "ok" else "infrastructure_error",
                "not_evaluated",
                "version_probe_failed",
            )
        tools[name] = {
            "executable": executable,
            "binary_sha256": digest(Path(executable).read_bytes()),
            "version": (attempt / probe["stdout"]).read_text(errors="replace")[:4096]
            + (attempt / probe["stderr"]).read_text(errors="replace")[:4096],
        }
    fingerprint = digest(json.dumps({"inputs": intent, "tools": tools}, sort_keys=True).encode())
    (attempt / "design.sv").write_text(request.design)
    (attempt / "tb.sv").write_text(request.testbench)
    atomic_json(attempt / "tools.json", tools)
    checkpoint_path = directory / "compile.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text()) if resume and checkpoint_path.exists() else None
    )
    compiled = attempt / "simulation.vvp"
    if checkpoint and checkpoint["fingerprint"] == fingerprint:
        previous = directory / checkpoint["path"]
        if (
            not previous.resolve().is_relative_to(directory)
            or not previous.is_file()
            or digest(previous.read_bytes()) != checkpoint["sha256"]
        ):
            raise ValueError("compile checkpoint is missing or has been modified")
        shutil.copyfile(previous, compiled)
        journal.emit("compile_reused", attempt_id=attempt_id, source=checkpoint["path"])
    else:
        result = run_process(
            [
                tools["iverilog"]["executable"],
                "-g2012",
                "-s",
                request.top,
                "-o",
                "simulation.vvp",
                "design.sv",
                "tb.sv",
            ],
            directory=attempt,
            stage="compile",
            deadline=deadline,
            cancel=cancel,
            journal=journal,
        )
        if result["execution"] != "ok":
            return _finish(
                directory,
                attempt,
                journal,
                result["execution"],
                "not_evaluated",
                "compile_interrupted",
            )
        if result["returncode"] != 0:
            return _finish(directory, attempt, journal, "ok", "fail", "compile_error")
        if not compiled.is_file():
            return _finish(
                directory,
                attempt,
                journal,
                "infrastructure_error",
                "not_evaluated",
                "missing_compiler_output",
            )
        atomic_json(
            checkpoint_path,
            {
                "fingerprint": fingerprint,
                "path": str(compiled.relative_to(directory)),
                "sha256": digest(compiled.read_bytes()),
            },
        )
        journal.emit("compile_checkpoint", attempt_id=attempt_id)
    result = run_process(
        [tools["vvp"]["executable"], "-n", "simulation.vvp"],
        directory=attempt,
        stage="simulate",
        deadline=deadline,
        cancel=cancel,
        journal=journal,
    )
    if result["execution"] != "ok":
        return _finish(
            directory,
            attempt,
            journal,
            result["execution"],
            "not_evaluated",
            "simulation_interrupted",
        )
    output = (attempt / result["stdout"]).read_text(errors="replace")
    passed = result["returncode"] == 0 and "CHIPS_TEST_PASS" in output.splitlines()
    reason = "testbench_passed" if passed else "testbench_failed_or_missing_pass_marker"
    return _finish(directory, attempt, journal, "ok", "pass" if passed else "fail", reason)


def _finish(directory, attempt, journal, execution, verdict, reason, **extra):
    artifacts = {
        str(p.relative_to(directory)): {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
        for p in sorted(attempt.iterdir())
        if p.is_file()
    }
    result = {
        "schema_version": 1,
        "call_id": journal.call_id,
        "attempt_id": attempt.name,
        "execution": execution,
        "verdict": verdict,
        "reason": reason,
        "certified": False,
        "artifacts": artifacts,
        **extra,
    }
    atomic_json(attempt / "result.json", result)
    atomic_json(directory / "result.json", result)
    journal.emit(
        "run_finished", execution=execution, verdict=verdict, reason=reason, attempt_id=attempt.name
    )
    return result
