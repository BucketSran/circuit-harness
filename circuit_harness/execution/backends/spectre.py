"""Server-local Spectre execution for the bounded, PDK-free RC-001 task."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from importlib.resources import files
from pathlib import Path

from circuit_harness.execution.backends.ngspice import rc_parameters
from circuit_harness.execution.evaluation.rc_validation import grade_rc
from circuit_harness.execution.runtime.journal import Journal, atomic_json, digest, file_digest
from circuit_harness.execution.runtime.process import run_process

_SAFE_PATH = re.compile(r"/[A-Za-z0-9_./-]+\Z")
_PROFILE_KEYS = {
    "schema_version",
    "task_id",
    "shell",
    "setup_scripts",
    "spectre",
    "run_root",
    "archive_root",
    "timeout_s",
    "license_queue_s",
}
_NUMBER = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:nan|inf))"
_AXIS = re.compile(rf'"(freq|time)"\s+({_NUMBER})\Z')
_REAL = re.compile(rf'"(in|out)"\s+({_NUMBER})\Z')
_COMPLEX = re.compile(rf'"(in|out)"\s+\(({_NUMBER})\s+({_NUMBER})\)\Z')


def _safe_path(value: object, label: str, *, allow_symlink=False) -> Path:
    if not isinstance(value, str) or not _SAFE_PATH.fullmatch(value) or ".." in Path(value).parts:
        raise ValueError(f"{label} must be a safe absolute path")
    path = Path(value)
    if path.is_symlink() and not allow_symlink:
        raise ValueError(f"{label} cannot be a symlink")
    return path


def load_profile(profile_path: Path, *, task_id: str = "RC-001") -> dict:
    path = Path(profile_path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_uid != os.getuid()
        or path.stat().st_mode & 0o077
    ):
        raise ValueError("profile must be an owned private regular file (0600)")
    if path.stat().st_size > 64 * 1024:
        raise ValueError("profile is too large")
    profile = json.loads(path.read_text())
    if not isinstance(profile, dict) or set(profile) != _PROFILE_KEYS:
        raise ValueError("profile fields must match the Spectre profile schema")
    if profile["schema_version"] != 1 or profile["task_id"] != task_id:
        raise ValueError("unsupported Spectre profile or task")
    if profile["shell"] != "/bin/csh" or not Path(profile["shell"]).is_file():
        raise ValueError("shell must be /bin/csh")
    scripts = profile["setup_scripts"]
    if not isinstance(scripts, list) or not 1 <= len(scripts) <= 4:
        raise ValueError("setup_scripts must contain 1-4 files")
    for entry in scripts:
        if not _safe_path(entry, "setup_scripts").is_file():
            raise ValueError("setup_scripts must refer to readable files")
    executable = _safe_path(profile["spectre"], "spectre", allow_symlink=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("spectre must be executable")
    for key in ("run_root", "archive_root"):
        root = _safe_path(profile[key], key)
        if not root.is_dir() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
            raise ValueError(f"{key} must exist, be owned by this user and be private (0700)")
    run_root, archive_root = (Path(profile[key]) for key in ("run_root", "archive_root"))
    if (
        run_root == archive_root
        or run_root in archive_root.parents
        or archive_root in run_root.parents
    ):
        raise ValueError("run_root and archive_root must be disjoint")
    for key, low, high in (("timeout_s", 0, 600), ("license_queue_s", 0, 60)):
        value = profile[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low < value <= high
        ):
            raise ValueError(f"{key} must be finite and in ({low}, {high}]")
    if int(profile["license_queue_s"]) != profile["license_queue_s"]:
        raise ValueError("license_queue_s must be an integer")
    return profile


def spectre_identity(task: dict, profile_path: Path) -> dict:
    profile = load_profile(profile_path)
    return {
        "backend": "spectre_rc",
        "task": rc_parameters(task),
        "profile_path": str(Path(profile_path).absolute()),
        "profile_sha256": file_digest(Path(profile_path)),
        "setup_scripts": [
            {"path": name, "sha256": file_digest(Path(name))} for name in profile["setup_scripts"]
        ],
        "shell": profile["shell"],
        "spectre": profile["spectre"],
        "spectre_resolved": str(Path(profile["spectre"]).resolve()),
        "spectre_sha256": file_digest(Path(profile["spectre"])),
        "run_root": profile["run_root"],
        "archive_root": profile["archive_root"],
        "timeout_s": profile["timeout_s"],
        "license_queue_s": profile["license_queue_s"],
        "harness_sha256": digest(
            b"".join(
                files("circuit_harness.execution").joinpath(name).read_bytes()
                for name in (
                    "backends/spectre.py",
                    "backends/ngspice.py",
                    "evaluation/rc_validation.py",
                    "runtime/process.py",
                    "runtime/journal.py",
                )
            )
        ),
    }


def read_psf(
    path: Path, *, axis: str, complex_values: bool, expected_input: float | None = None
) -> list[tuple]:
    """Read the exact two-voltage psfascii subset emitted by the fixed netlist."""
    if axis not in {"freq", "time"} or complex_values != (axis == "freq"):
        raise ValueError("unsupported PSF axis or value type")
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("missing or unsafe PSF")
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        trace = lines.index("TRACE")
        value = lines.index("VALUE", trace + 1)
    except ValueError as error:
        raise ValueError("missing PSF trace/value section") from error
    if lines[trace + 1 : value] != ['"in" "V"', '"out" "V"']:
        raise ValueError("unexpected PSF trace set")
    samples = []
    position = value + 1
    pattern = _COMPLEX if complex_values else _REAL
    while position < len(lines) and lines[position] != "END":
        if position + 2 >= len(lines):
            raise ValueError("truncated PSF sample")
        match_axis = _AXIS.fullmatch(lines[position])
        first = pattern.fullmatch(lines[position + 1])
        second = pattern.fullmatch(lines[position + 2])
        if (
            not match_axis
            or match_axis[1] != axis
            or not first
            or not second
            or first[1] != "in"
            or second[1] != "out"
        ):
            raise ValueError("invalid PSF sample or trace order")
        coordinate = float(match_axis[2])
        values = tuple(float(v) for v in (*first.groups()[1:], *second.groups()[1:]))
        if not all(math.isfinite(v) for v in (coordinate, *values)):
            raise ValueError("nonfinite PSF value")
        input_values = values[: len(values) // 2]
        if expected_input is not None and (
            not math.isclose(input_values[0], expected_input, rel_tol=1e-5, abs_tol=1e-8)
            or (complex_values and not math.isclose(input_values[1], 0, abs_tol=1e-8))
        ):
            raise ValueError("PSF input voltage differs from the fixed source")
        if samples and coordinate <= samples[-1][0]:
            raise ValueError("non-increasing PSF coordinate")
        samples.append((coordinate, *values[len(values) // 2 :]))
        if len(samples) > 100_000:
            raise ValueError("too many PSF samples")
        position += 3
    if not samples or position != len(lines) - 1 or lines[position] != "END":
        raise ValueError("empty or unterminated PSF")
    return samples


def _netlist(task: dict) -> str:
    r, c, voltage = (task[k] for k in ("resistance_ohm", "capacitance_f", "voltage_v"))
    tau = r * c
    cutoff = 1 / (2 * math.pi * tau)
    return f"""simulator lang=spectre
global 0
V1 (in 0) vsource dc={voltage:.17g} mag=1
R1 (in out) resistor r={r:.17g}
C1 (out 0) capacitor c={c:.17g} ic=0
simulatorOptions options reltol=1e-6 vabstol=1e-9 iabstol=1e-12
ac1 ac start={cutoff / 1000:.17g} stop={cutoff * 1000:.17g} dec=20
tran1 tran stop={tau * 10:.17g} maxstep={tau / 100:.17g} skipdc=yes
save in out
"""


def _table_text(rows: list[tuple]) -> str:
    return "".join(" ".join(f"{x:.17g}" for x in row) + "\n" for row in rows)


def _grade(task: dict, directory: Path) -> dict:
    return grade_rc(task, directory / "ac.dat", directory / "transient.dat")


def run_spectre_rc(identity: dict, directory: Path, cancel=None) -> dict:
    """Run once in an isolated worker and retain raw output for regrading."""
    if spectre_identity(identity["task"], Path(identity["profile_path"])) != identity:
        raise ValueError("Spectre task/profile/tool/Harness changed after submission")
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    atomic_json(directory / "request.json", {"identity": identity})
    run_id = uuid.uuid4().hex
    journal = Journal(directory, call_id=run_id)
    journal.emit("run_started", backend="spectre_rc", timeout_s=identity["timeout_s"])

    def finish(execution: str, verdict: str, reason: str, **extra) -> dict:
        artifacts = {
            str(p.relative_to(directory)): {"sha256": file_digest(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*"))
            if p.is_file() and not p.is_symlink() and p.name not in {"result.json", "events.jsonl"}
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

    with journal.stage("prepare_input"):
        (directory / "rc.scs").write_text(_netlist(identity["task"]))
        script = (
            "\n".join(
                ["#!/bin/csh -f"]
                + [f"source {item['path']}" for item in identity["setup_scripts"]]
                + [
                    f"{identity['spectre']} -64 rc.scs +log spectre.log "
                    f"-format psfascii -raw psf +lqtimeout {int(identity['license_queue_s'])} +mt=1"
                ]
            )
            + "\n"
        )
        (directory / "run.csh").write_text(script)
    process = run_process(
        [identity["shell"], "-f", "run.csh"],
        directory=directory,
        stage="spectre",
        deadline=time.monotonic() + identity["timeout_s"],
        cancel=cancel or threading.Event(),
        journal=journal,
        max_output_bytes=100 * 1024 * 1024,
    )
    if process["execution"] != "ok":
        return finish(process["execution"], "not_evaluated", "spectre_interrupted")
    if process["returncode"] != 0:
        log = directory / "spectre.log"
        reason = (
            "license_checkout_failed"
            if log.is_file() and "SPECTRE-209" in log.read_text(errors="replace")
            else "spectre_nonzero_exit"
        )
        execution = "infrastructure_error" if reason == "license_checkout_failed" else "tool_error"
        return finish(execution, "not_evaluated", reason)
    try:
        with journal.stage("parse_and_grade"):
            ac = read_psf(
                directory / "psf" / "ac1.ac",
                axis="freq",
                complex_values=True,
                expected_input=1.0,
            )
            tran = read_psf(
                directory / "psf" / "tran1.tran.tran",
                axis="time",
                complex_values=False,
                expected_input=identity["task"]["voltage_v"],
            )
            (directory / "ac.dat").write_text(_table_text(ac))
            (directory / "transient.dat").write_text(_table_text(tran))
            metrics = _grade(identity["task"], directory)
    except (ValueError, OSError, UnicodeError) as error:
        return finish("invalid_output", "not_evaluated", str(error))
    atomic_json(directory / "metrics.json", metrics)
    journal.emit("grade_finished", **metrics)
    return finish("ok", metrics["verdict"], "analytical_rc_validation", metrics=metrics)


def verify_spectre_rc(directory: Path) -> dict:
    """Verify raw PSF and regrade without Spectre or access to the live profile."""
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    if (
        result["identity"] != request["identity"]
        or result["identity"].get("backend") != "spectre_rc"
    ):
        raise ValueError("Spectre result identity mismatch")
    if "request.json" not in result["artifacts"]:
        raise ValueError("missing Spectre request receipt")
    for name, receipt in result["artifacts"].items():
        path = directory / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or path.is_symlink()
            or directory.resolve() not in path.resolve().parents
            or not path.is_file()
            or path.stat().st_size != receipt["bytes"]
            or file_digest(path) != receipt["sha256"]
        ):
            raise ValueError(f"Spectre artifact missing or modified: {name}")
    if result["execution"] == "ok":
        required = {
            "request.json",
            "rc.scs",
            "run.csh",
            "psf/ac1.ac",
            "psf/tran1.tran.tran",
            "ac.dat",
            "transient.dat",
            "metrics.json",
        }
        if not required <= result["artifacts"].keys():
            raise ValueError("missing Spectre numerical evidence receipts")
        ac = read_psf(
            directory / "psf" / "ac1.ac",
            axis="freq",
            complex_values=True,
            expected_input=1.0,
        )
        tran = read_psf(
            directory / "psf" / "tran1.tran.tran",
            axis="time",
            complex_values=False,
            expected_input=result["identity"]["task"]["voltage_v"],
        )
        if (directory / "ac.dat").read_text() != _table_text(ac) or (
            directory / "transient.dat"
        ).read_text() != _table_text(tran):
            raise ValueError("derived tables differ from raw Spectre PSF")
        metrics = _grade(result["identity"]["task"], directory)
        recorded = json.loads((directory / "metrics.json").read_text())
        if (
            recorded != result["metrics"]
            or metrics["verdict"] != result["verdict"]
            or metrics.keys() != recorded.keys()
        ):
            raise ValueError("Spectre verdict/metrics differ from independent regrade")
        for key, value in metrics.items():
            matches = (
                math.isclose(value, recorded[key], rel_tol=1e-10, abs_tol=1e-12)
                if key in {"ac_max_abs_error", "transient_max_normalized_error"}
                else value == recorded[key]
            )
            if not matches:
                raise ValueError(f"Spectre metric differs from independent regrade: {key}")
    return result
