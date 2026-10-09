"""Server-local ngspice execution for the public analytical RC baseline."""

from __future__ import annotations

import fcntl
import json
import math
import shutil
import threading
import time
import uuid
from importlib.resources import files
from pathlib import Path

from circuit_harness.execution.evaluation.rc_validation import grade_rc
from circuit_harness.execution.runtime.journal import Journal, atomic_json, digest, file_digest
from circuit_harness.execution.runtime.process import run_process

MAX_OUTPUT_BYTES = 100 * 1024 * 1024


def verify_rc(directory: Path) -> dict:
    """Verify downloaded evidence and regrade successful executions without the solver."""
    result = json.loads((directory / "result.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    if result["identity"] != request["identity"] or result["run_id"] != request["run_id"]:
        raise ValueError("result identity mismatch")
    if "request.json" not in result["artifacts"]:
        raise ValueError("missing request receipt")
    for name, receipt in result["artifacts"].items():
        path = directory / name
        if (
            Path(name).name != name
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != receipt["bytes"]
            or file_digest(path) != receipt["sha256"]
        ):
            raise ValueError(f"artifact missing or modified: {name}")
    if result["execution"] == "ok":
        if not {"ac.dat", "transient.dat", "metrics.json"} <= result["artifacts"].keys():
            raise ValueError("missing numerical evidence receipts")
        metrics = grade_rc(
            rc_parameters(request["identity"]["task"]),
            directory / "ac.dat",
            directory / "transient.dat",
        )
        recorded = json.loads((directory / "metrics.json").read_text())
        if recorded != result["metrics"] or metrics["verdict"] != result["verdict"]:
            raise ValueError("stored verdict/metrics differ from independent regrade")
        if recorded.keys() != metrics.keys():
            raise ValueError("stored metric fields differ from independent regrade")
        # libm can differ at the last bits across Linux/macOS. Artifact hashes and
        # the independently computed pass/fail remain strict; only these derived
        # numerical error estimates allow tiny roundoff, far below task tolerances.
        error_metrics = {"ac_max_abs_error", "transient_max_normalized_error"}
        for key, value in metrics.items():
            if key in error_metrics:
                matches = isinstance(recorded[key], (int, float)) and math.isclose(
                    value, recorded[key], rel_tol=1e-10, abs_tol=1e-12
                )
            else:
                matches = value == recorded[key]
            if not matches:
                raise ValueError(f"stored metric differs from independent regrade: {key}")
    return result


def rc_parameters(payload: dict) -> dict:
    """Normalize the bounded public task; no arbitrary netlist or shell input."""
    allowed = {"resistance_ohm": (1, 1e9), "capacitance_f": (1e-15, 1e-3), "voltage_v": (1e-3, 100)}
    if not isinstance(payload, dict) or set(payload) - allowed.keys():
        raise ValueError("unknown RC task fields")
    values = {**payload}
    values.setdefault("voltage_v", 1.0)
    for name, (low, high) in allowed.items():
        value = values.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be finite and in [{low}, {high}] SI units")
        values[name] = float(value)
    if not 1e-12 <= values["resistance_ohm"] * values["capacitance_f"] <= 1e3:
        raise ValueError("RC time constant must be in [1e-12, 1e3] seconds")
    return values


def _netlist(task: dict) -> str:
    r, c, voltage = (task[k] for k in ("resistance_ohm", "capacitance_f", "voltage_v"))
    tau = r * c
    cutoff = 1 / (2 * math.pi * tau)
    return f"""Chips public RC baseline; SI units
V1 in 0 DC {voltage:.17g} AC 1
R1 in out {r:.17g}
C1 out 0 {c:.17g} IC=0
.options reltol=1e-6 vntol=1e-9 abstol=1e-12
.control
set noaskquit
set numdgt=15
set wr_singlescale
ac dec 20 {cutoff / 1000:.17g} {cutoff * 1000:.17g}
wrdata ac.dat v(out)
tran {tau / 100:.17g} {tau * 10:.17g} 0 {tau / 100:.17g} uic
wrdata transient.dat v(out)
quit
.endc
.end
"""


def rc_identity(payload: dict, ngspice="ngspice", timeout_s=60) -> dict:
    """Validate and identify the exact task/tool/grader before creating an attempt."""
    task = rc_parameters(payload)
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (float, int))
        or not math.isfinite(timeout_s)
        or not 0 < timeout_s <= 600
    ):
        raise ValueError("timeout_s must be finite and in (0, 600]")
    executable = shutil.which(ngspice)
    if executable:
        executable = str(Path(executable).resolve())
    return {
        "task": task,
        "timeout_s": timeout_s,
        "executable": executable or ngspice,
        "binary_sha256": file_digest(Path(executable)) if executable else None,
        "harness_sha256": digest(
            b"".join(
                files("circuit_harness.execution").joinpath(name).read_bytes()
                for name in (
                    "backends/ngspice.py",
                    "evaluation/rc_validation.py",
                    "runtime/process.py",
                    "runtime/journal.py",
                )
            )
        ),
    }


def run_rc(
    payload: dict, directory: Path, *, ngspice="ngspice", timeout_s=60, resume=False, cancel=None
) -> dict:
    """Run locally beside the simulator; finished evidence can be resumed once verified."""
    identity = rc_identity(payload, ngspice, timeout_s)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("run already has an active writer") from error
        return _run_rc(identity, directory, resume, cancel or threading.Event())


def _run_rc(identity, directory, resume, cancel):
    task = identity["task"]
    timeout_s = identity["timeout_s"]
    executable = identity["executable"] if identity["binary_sha256"] else None
    request_path = directory / "request.json"
    if request_path.exists():
        if not resume or json.loads(request_path.read_text())["identity"] != identity:
            raise ValueError("existing run requires --resume with identical task/tool/Harness")
        result_path = directory / "result.json"
        if not result_path.exists():
            raise ValueError("incomplete run: inspect status/processes; automatic resubmit refused")
        result = verify_rc(directory)
        Journal(directory, call_id=result["run_id"]).emit("result_reused")
        return result
    occupied = [path for path in directory.iterdir() if path.name != ".lock"]
    if occupied:
        raise ValueError("new run requires an empty directory")
    run_id = uuid.uuid4().hex
    atomic_json(request_path, {"run_id": run_id, "identity": identity})
    journal = Journal(directory, call_id=run_id)
    journal.emit("run_started", backend="ngspice", timeout_s=timeout_s)

    def finish(execution, verdict, reason, **extra):
        artifacts = {
            p.name: {"sha256": file_digest(p), "bytes": p.stat().st_size}
            for p in sorted(directory.iterdir())
            if p.is_file()
            and not p.is_symlink()
            and p.name not in {".lock", "events.jsonl", "result.json"}
        }
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "identity": identity,
            "execution": execution,
            "verdict": verdict,
            "reason": reason,
            "certified": False,
            "artifacts": artifacts,
            **extra,
        }
        atomic_json(directory / "result.json", result)
        journal.emit("run_finished", execution=execution, verdict=verdict, reason=reason)
        return result

    if not executable:
        return finish("infrastructure_error", "not_evaluated", "dependency_missing")
    deadline = time.monotonic() + timeout_s
    with journal.stage("prepare_input"):
        (directory / "circuit.cir").write_text(_netlist(task))
    for stage, argv in (
        ("version", [executable, "-v"]),
        ("ngspice", [executable, "-n", "-b", "circuit.cir"]),
    ):
        process = run_process(
            argv,
            directory=directory,
            stage=stage,
            deadline=deadline,
            cancel=cancel,
            journal=journal,
            max_output_bytes=MAX_OUTPUT_BYTES,
        )
        if process["execution"] != "ok":
            return finish(process["execution"], "not_evaluated", f"{stage}_interrupted")
        if process["returncode"] != 0:
            return finish("tool_error", "not_evaluated", f"{stage}_nonzero_exit")
    for name in ("ac.dat", "transient.dat"):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            return finish("missing_output", "not_evaluated", f"missing_or_unsafe_{name}")
    try:
        with journal.stage("parse_and_grade"):
            metrics = grade_rc(task, directory / "ac.dat", directory / "transient.dat")
    except ValueError as error:
        return finish("invalid_output", "not_evaluated", str(error))
    atomic_json(directory / "metrics.json", metrics)
    journal.emit("grade_finished", **metrics)
    return finish("ok", metrics["verdict"], "analytical_rc_validation", metrics=metrics)
