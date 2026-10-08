"""Bounded differential-gain testbench construction and independent PSF checks."""

import cmath
import copy
import json
import math
import re
import shutil
import threading
import time
import uuid
from importlib.resources import files
from pathlib import Path

from .journal import Journal, atomic_json, digest, file_digest
from .process import run_process
from .spectre import load_profile
from .task_authoring import require_confirmation


def testbench_identity(payload, profile_path):
    """Bind a confirmed candidate, operator model and simulator deployment."""
    if not isinstance(payload, dict) or set(payload) != {
        "draft",
        "confirmation",
        "materials",
        "candidate",
        "reference",
        "model",
    }:
        raise ValueError("invalid testbench payload")
    prepare_testbench(
        payload["draft"],
        payload["materials"],
        payload["confirmation"],
        payload["candidate"],
        payload["reference"],
    )
    model = Path(payload["model"])
    if model.is_symlink() or not model.is_file() or model.stat().st_size > 1024 * 1024:
        raise ValueError("invalid operator reference model")
    profile = load_profile(profile_path, task_id="OPAMP-GAIN-001")
    return {
        "backend": "spectre_gain",
        "task": copy.deepcopy(payload),
        "profile_path": str(Path(profile_path).absolute()),
        "profile_sha256": file_digest(Path(profile_path)),
        "profile": profile,
        "model_sha256": file_digest(model),
        "setup_sha256": {name: file_digest(Path(name)) for name in profile["setup_scripts"]},
        "spectre_sha256": file_digest(Path(profile["spectre"])),
        "harness_sha256": digest(
            b"".join(
                files(__package__).joinpath(name).read_bytes()
                for name in (
                    "spectre_testbench.py",
                    "task_authoring.py",
                    "spectre.py",
                    "process.py",
                    "journal.py",
                )
            )
        ),
    }


def run_testbench(identity, directory, cancel=None):
    """Execute a confirmed bounded testbench and retain independently graded evidence."""
    if testbench_identity(identity["task"], identity["profile_path"]) != identity:
        raise ValueError("testbench inputs or deployment changed after submission")
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    atomic_json(directory / "request.json", {"identity": identity})
    task, profile = identity["task"], identity["profile"]
    journal = Journal(directory, call_id=uuid.uuid4().hex)
    journal.emit("run_started", backend="spectre_gain")

    def finish(execution, verdict, reason, **extra):
        artifacts = {
            str(p.relative_to(directory)): {"sha256": file_digest(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*"))
            if p.is_file() and not p.is_symlink() and p.name not in {"result.json", "events.jsonl"}
        }
        result = {
            "schema_version": 1,
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
        for source in task["draft"]["sources"]:
            target = directory / "materials" / source["path"]
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(Path(task["materials"]) / source["path"], target)
        prepared = prepare_testbench(
            task["draft"],
            directory / "materials",
            task["confirmation"],
            task["candidate"],
            task["reference"],
        )
        shutil.copyfile(task["model"], directory / "reference.va")
        if file_digest(directory / "reference.va") != identity["model_sha256"]:
            raise ValueError("reference model changed during snapshot")
        (directory / "candidate.scs").write_text(prepared["netlist"])
        script = (
            "\n".join(
                ["#!/bin/csh -f"]
                + [f"source {name}" for name in profile["setup_scripts"]]
                + [
                    f"{profile['spectre']} -64 candidate.scs +log spectre.log -format psfascii "
                    f"-raw psf +lqtimeout {int(profile['license_queue_s'])} +mt=1"
                ]
            )
            + "\n"
        )
        (directory / "run.csh").write_text(script)
    process = run_process(
        [profile["shell"], "-f", "run.csh"],
        directory=directory,
        stage="spectre",
        deadline=time.monotonic() + profile["timeout_s"],
        cancel=cancel or threading.Event(),
        journal=journal,
        max_output_bytes=100 * 1024 * 1024,
    )
    if process["execution"] != "ok":
        return finish(process["execution"], "not_evaluated", "spectre_interrupted")
    if process["returncode"] != 0:
        log = directory / "spectre.log"
        license_error = log.is_file() and "SPECTRE-209" in log.read_text(errors="replace")
        return finish(
            "infrastructure_error" if license_error else "tool_error",
            "not_evaluated",
            "license_checkout_failed" if license_error else "spectre_nonzero_exit",
        )
    try:
        with journal.stage("parse_and_grade"):
            metrics = _grade(directory, task, prepared["values"])
    except (ValueError, OSError, UnicodeError) as error:
        return finish("invalid_output", "not_evaluated", str(error))
    atomic_json(directory / "metrics.json", metrics)
    return finish(
        "ok",
        "pass" if metrics["testbench_valid"] else "fail",
        "analytical_gain_validation",
        metrics=metrics,
    )


def _grade(directory, task, values):
    return grade_testbench(
        directory / "psf/ac1.ac",
        task["candidate"],
        values["frequency_hz"],
        values["minimum_gain_db"],
        task["reference"],
    )


def verify_testbench(directory):
    """Recheck an archived testbench run without calling a simulator."""
    directory = Path(directory).resolve()
    result = json.loads((directory / "result.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    if result["identity"] != request["identity"] or result["identity"]["backend"] != "spectre_gain":
        raise ValueError("testbench identity mismatch")
    if "request.json" not in result["artifacts"]:
        raise ValueError("missing request receipt")
    for name, receipt in result["artifacts"].items():
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
            raise ValueError(f"testbench artifact missing or modified: {name}")
    if result["execution"] == "ok":
        task = result["identity"]["task"]
        required = {"candidate.scs", "reference.va", "run.csh", "psf/ac1.ac", "metrics.json"}
        required.update("materials/" + item["path"] for item in task["draft"]["sources"])
        if not required <= result["artifacts"].keys():
            raise ValueError("missing testbench evidence receipts")
        prepared = prepare_testbench(
            task["draft"],
            directory / "materials",
            task["confirmation"],
            task["candidate"],
            task["reference"],
        )
        if (directory / "candidate.scs").read_text() != prepared["netlist"] or file_digest(
            directory / "reference.va"
        ) != result["identity"]["model_sha256"]:
            raise ValueError("testbench does not match confirmed inputs")
        metrics = _grade(directory, task, prepared["values"])
        recorded = json.loads((directory / "metrics.json").read_text())
        if recorded != result["metrics"] or metrics.keys() != recorded.keys():
            raise ValueError("testbench metrics differ from regrade")
        for name, value in metrics.items():
            matches = (
                math.isclose(value, recorded[name], rel_tol=1e-10, abs_tol=1e-12)
                if type(value) is float
                else value == recorded[name]
            )
            if not matches:
                raise ValueError(f"testbench metric differs from regrade: {name}")
        if result["verdict"] != ("pass" if metrics["testbench_valid"] else "fail"):
            raise ValueError("testbench verdict differs from regrade")
    return result


REQUIREMENTS = {
    "supply_v": "V",
    "common_mode_v": "V",
    "temperature_c": "degC",
    "load_ohm": "ohm",
    "frequency_hz": "Hz",
    "minimum_gain_db": "dB",
    "dut_ports": None,
}


def _number(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"invalid bounded value: {name}")


def _validate_candidate(candidate, reference, values):
    if not isinstance(candidate, dict) or set(candidate) != {
        "ports",
        "positive_ac",
        "negative_ac",
        "measurement",
    }:
        raise ValueError("candidate must contain only ports, two AC sources and measurement")
    nodes = {"vip", "vin", "out", "vdd", "0"}
    ports = candidate["ports"]
    if (
        not isinstance(ports, list)
        or len(ports) != 5
        or any(not isinstance(node, str) or node not in nodes for node in ports)
        or set(ports) != nodes
        or ports != values["dut_ports"]
    ):
        raise ValueError("ports must match the confirmed five-terminal interface")
    for name in ("positive_ac", "negative_ac"):
        source = candidate[name]
        if not isinstance(source, dict) or set(source) != {"magnitude", "phase_deg"}:
            raise ValueError("invalid AC source")
        _number(source["magnitude"], 0, 1, name)
        _number(source["phase_deg"], -360, 360, name)
    measurement = candidate["measurement"]
    if not isinstance(measurement, dict) or set(measurement) != {"numerator", "denominator"}:
        raise ValueError("invalid measurement")
    for pair in measurement.values():
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or pair[0] == pair[1]
            or any(not isinstance(node, str) or node not in nodes for node in pair)
        ):
            raise ValueError("measurement requires two distinct allowed nodes")
    if not isinstance(reference, dict) or set(reference) != {"gain0", "pole_hz"}:
        raise ValueError("invalid operator reference")
    for name, low, high in (("gain0", 1, 1e6), ("pole_hz", 1, 1e6)):
        _number(reference[name], low, high, name)
    for name, low, high in (
        ("supply_v", 0.1, 5),
        ("common_mode_v", 0, values["supply_v"]),
        ("temperature_c", -40, 125),
        ("load_ohm", 100, 1e9),
        ("frequency_hz", 1, 1e6),
        ("minimum_gain_db", 0, 120),
    ):
        _number(values[name], low, high, name)


def prepare_testbench(draft, materials, confirmation, candidate, reference):
    """Compile a bounded candidate only after its conditions have been confirmed."""
    values = require_confirmation(draft, materials, REQUIREMENTS, confirmation)
    _validate_candidate(candidate, reference, values)
    p, n = candidate["positive_ac"], candidate["negative_ac"]
    cm, supply = values["common_mode_v"], values["supply_v"]
    ports = " ".join(candidate["ports"])
    gain, pole = reference["gain0"], reference["pole_hz"]
    netlist = f"""simulator lang=spectre
global 0
ahdl_include "reference.va"
VDD (vdd 0) vsource dc={supply:.17g}
VP (vip 0) vsource dc={cm:.17g} mag={p["magnitude"]:.17g} phase={p["phase_deg"]:.17g}
VN (vin 0) vsource dc={cm:.17g} mag={n["magnitude"]:.17g} phase={n["phase_deg"]:.17g}
DUT ({ports}) gain_fixture gain0={gain:.17g} pole_hz={pole:.17g}
RL (out 0) resistor r={values["load_ohm"]:.17g}
simulatorOptions options temp={values["temperature_c"]:.17g} reltol=1e-6 vabstol=1e-9 iabstol=1e-12
ac1 ac values=[{values["frequency_hz"]:.17g}]
save vip vin out vdd
"""
    return {"netlist": netlist, "values": values, "candidate": copy.deepcopy(candidate)}


def grade_testbench(path, candidate, frequency_hz, minimum_gain_db, reference):
    """Check measurement accuracy separately from whether the DUT meets its spec."""
    measured = measure_gain(path, candidate, frequency_hz)
    expected_gain = 20 * math.log10(reference["gain0"]) - 10 * math.log10(
        1 + (frequency_hz / reference["pole_hz"]) ** 2
    )
    expected_phase = -math.degrees(math.atan(frequency_hz / reference["pole_hz"]))
    gain_error = abs(measured["gain_db"] - expected_gain)
    phase_error = abs((measured["phase_deg"] - expected_phase + 180) % 360 - 180)
    valid = gain_error <= 0.2 and phase_error <= 1.0
    return {
        **measured,
        "expected_gain_db": expected_gain,
        "expected_phase_deg": expected_phase,
        "gain_error_db": gain_error,
        "phase_error_deg": phase_error,
        "testbench_valid": valid,
        "dut_pass": measured["gain_db"] >= minimum_gain_db if valid else None,
    }


def measure_gain(path, candidate, frequency_hz):
    """Measure the declared voltage ratio from original AC data."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("missing, unsafe or oversized AC PSF")
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        trace = lines.index("TRACE")
        start = lines.index("VALUE", trace + 1)
        end = lines.index("END", start + 1)
        names = []
        for line in lines[trace + 1 : start]:
            match = re.fullmatch(r'"(vip|vin|out|vdd)" "V"', line)
            if not match:
                raise ValueError("unexpected voltage trace")
            names.append(match[1])
        if len(names) != 4 or set(names) != {"vip", "vin", "out", "vdd"}:
            raise ValueError("missing or duplicate voltage trace")
        if end - start != 6 or end != len(lines) - 1:
            raise ValueError("expected exactly one complete AC point")
        axis = re.fullmatch(r'"freq" (\S+)', lines[start + 1])
        if not axis or not math.isclose(float(axis[1]), frequency_hz, rel_tol=1e-9):
            raise ValueError("wrong AC frequency")
        values = {"0": 0j}
        for name, line in zip(names, lines[start + 2 : end], strict=True):
            match = re.fullmatch(r'"([a-z]+)" \((\S+) (\S+)\)', line)
            if not match or match[1] != name:
                raise ValueError("invalid AC sample order")
            value = complex(float(match[2]), float(match[3]))
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise ValueError("nonfinite AC value")
            values[name] = value
        measurement = candidate["measurement"]
        op, on = measurement["numerator"]
        ip, inn = measurement["denominator"]
        numerator, denominator = values[op] - values[on], values[ip] - values[inn]
        if abs(denominator) < 1e-12 or abs(numerator) < 1e-12:
            raise ValueError("zero excitation or response")
        ratio = numerator / denominator
        return {
            "gain_db": 20 * math.log10(abs(ratio)),
            "phase_deg": math.degrees(cmath.phase(ratio)),
            "input_magnitude_v": abs(denominator),
            "frequency_hz": frequency_hz,
        }
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("invalid AC data or measurement") from error
