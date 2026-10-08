"""Public passive-candidate rules and diagnostics for pinned Analog RLC tasks."""

from __future__ import annotations

import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import time
import uuid
from pathlib import Path

from .analog_design_bench import TASKS, rlc_contract, tree_digest
from .journal import atomic_json, file_digest

TASK_ID = "rlc-rf-bandpass-100mhz"
_MEASURE = re.compile(r"(?im)^\s*meas\s+ac\s+([a-z_][a-z0-9_]*)\b")
_RESULT = re.compile(
    r"(?im)^\s*([a-z_][a-z0-9_]*)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\b"
)

_VALUE = re.compile(r"(?i)^((?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)(meg|[tgkmunpf])?$")
_NAME = re.compile(r"(?i)^[rlc][a-z0-9_]+$")
_NODE = re.compile(r"(?i)^[a-z0-9_]+$")
_SCALE = {
    None: 1.0,
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}


def validate_rlc_candidate(content: str, *, task_id: str = TASK_ID) -> dict[str, int]:
    """Accept only a positive-valued R/L/C subcircuit, never SPICE commands."""
    contract = rlc_contract(task_id)
    if not isinstance(content, str) or not 0 < len(content.encode("utf-8")) <= 100_000:
        raise ValueError("candidate must be nonempty UTF-8 text of at most 100 KB")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("*")
    ]
    interface = [".subckt", contract.subcircuit, *contract.ports]
    if len(lines) < 3 or [part.casefold() for part in lines[0].split()] != [
        part.casefold() for part in interface
    ]:
        raise ValueError(f"expected the declared {contract.subcircuit} subcircuit and port order")
    if [part.casefold() for part in lines[-1].split()] != [".ends", contract.subcircuit.casefold()]:
        raise ValueError("expected the matching .ends directive")
    elements = lines[1:-1]
    if not 1 <= len(elements) <= 256:
        raise ValueError("candidate must contain 1–256 passive elements")
    names: set[str] = set()
    for line in elements:
        parts = line.split()
        if len(parts) != 4 or not _NAME.fullmatch(parts[0]):
            raise ValueError("only R/L/C elements with two nodes and one value are allowed")
        name = parts[0].casefold()
        if name in names or not all(_NODE.fullmatch(node) for node in parts[1:3]):
            raise ValueError("duplicate element name or invalid node")
        names.add(name)
        match = _VALUE.fullmatch(parts[3])
        if not match:
            raise ValueError("element value must be a positive finite literal")
        value = float(match.group(1)) * _SCALE[match.group(2).lower() if match.group(2) else None]
        if not math.isfinite(value) or value <= 0:
            raise ValueError("element value must be a positive finite literal")
    return {"components": len(elements)}


def run_public_rlc(
    source_root: Path,
    candidate: Path,
    output: Path,
    *,
    task_id: str = TASK_ID,
    podman: str = "podman",
    podman_root: Path | None = None,
    podman_runroot: Path | None = None,
    runtime_image: str | None = None,
    offline_image_archive: Path | None = None,
    podman_single_id: bool = False,
    podman_no_cpu_limit: bool = False,
    timeout_s: int = 120,
) -> dict:
    """Run pinned public benches only; no reference, tests or reward enter the container."""
    contract = rlc_contract(task_id)
    if not 1 <= timeout_s <= 300:
        raise ValueError("public simulation timeout must be between 1 and 300 seconds")
    if (podman_root is None) != (podman_runroot is None):
        raise ValueError("Podman root and runroot must be provided together")
    if (runtime_image is None) != (offline_image_archive is None):
        raise ValueError("runtime image and offline image archive must be provided together")
    if runtime_image is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_image):
        raise ValueError("runtime image must be a local sha256 image ID")
    if offline_image_archive is not None and not Path(offline_image_archive).is_file():
        raise ValueError("offline image archive is unavailable")

    candidate = Path(candidate).absolute()
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size > 100_000:
        raise ValueError("candidate must be a regular file of at most 100 KB")
    validate_rlc_candidate(candidate.read_text(encoding="utf-8"), task_id=task_id)
    source = (Path(source_root) / "tasks" / task_id).absolute()
    task = TASKS[task_id]
    source_hash = tree_digest(source)
    if source_hash != task.source_sha256:
        raise ValueError("public task source pin mismatch")
    output = Path(output).absolute()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("use a new output directory outside the task source")
    expected: set[str] = set()
    bench_contents = {}
    for name in contract.benches:
        bench = source / "environment/starter/testbench" / name
        content = bench.read_text(encoding="utf-8")
        measurements = _MEASURE.findall(content)
        if not contract.analyzer and (
            not measurements or '.include "/app/circuit.spi"' not in content
        ):
            raise ValueError("pinned public bench has an unsupported format")
        expected.update(value.lower() for value in measurements)
        bench_contents[name] = content
    if contract.analyzer:
        bench_contents[contract.analyzer] = (
            source / "environment/starter/testbench" / contract.analyzer
        ).read_text(encoding="utf-8")
        expected = {"gamma", "transducer_gain"}

    output.mkdir(mode=0o700, parents=True)
    inputs = output / "inputs"
    public = output / "public"
    inputs.mkdir(mode=0o700)
    public.mkdir(mode=0o700)
    snapshot = inputs / "circuit.spi"
    shutil.copyfile(candidate, snapshot)
    snapshot.chmod(0o600)
    validate_rlc_candidate(snapshot.read_text(encoding="utf-8"), task_id=task_id)
    for name, content in bench_contents.items():
        target = public / name
        target.write_text(content, encoding="utf-8")
        target.chmod(0o600)

    image = runtime_image or task.image
    offline_hash = file_digest(Path(offline_image_archive)) if offline_image_archive else None
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "task_id": task_id,
            "source_commit": task.commit,
            "source_sha256": source_hash,
            "candidate_sha256": file_digest(snapshot),
            "public_benches": {name: file_digest(public / name) for name in bench_contents},
            "authority": "public_diagnostic",
            "backend": "podman",
            "image": task.image,
            "runtime_image": image,
            "offline_image_archive_sha256": offline_hash,
        },
    )
    base = [podman]
    if podman_root is not None:
        base += ["--root", str(podman_root), "--runroot", str(podman_runroot)]
    if podman_single_id:
        base += ["--storage-opt", "ignore_chown_errors=true"]
    container_name = "chips-public-" + uuid.uuid4().hex[:16]
    command = base + [
        "run",
        "--rm",
        "--name",
        container_name,
        "--pull=never",
        "--network=none",
        "--memory=2g",
    ]
    if not podman_no_cpu_limit:
        command += ["--cpus=4"]
    diagnostic_command = (
        "python3 " + shlex.quote("/app/public/" + contract.analyzer)
        if contract.analyzer
        else " && ".join(
            "ngspice -b " + shlex.quote("/app/public/" + name) for name in contract.benches
        )
    )
    command += [
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,size=512m",
        "--volume",
        f"{snapshot}:/app/circuit.spi:ro",
        "--volume",
        f"{public}:/app/public:ro",
        image,
        "/bin/bash",
        "-c",
        "command -v ngspice && " + diagnostic_command,
    ]
    safe_keys = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "PYTHONUTF8",
        "XDG_RUNTIME_DIR",
        "XDG_CONFIG_HOME",
        "DBUS_SESSION_BUS_ADDRESS",
    }
    process_env = {key: value for key, value in os.environ.items() if key in safe_keys}
    started = time.monotonic()
    exit_code = None
    state = "simulation_error"
    cleanup_confirmed = True
    log_path = output / "public.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=process_env,
                timeout=timeout_s,
                check=False,
            )
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            state = "timeout"
            try:
                cleanup = subprocess.run(
                    base + ["rm", "--force", "--ignore", container_name],
                    capture_output=True,
                    env=process_env,
                    timeout=20,
                    check=False,
                )
                cleanup_confirmed = cleanup.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                cleanup_confirmed = False
            if not cleanup_confirmed:
                state = "cleanup_failed"
        except OSError as error:
            state = "infrastructure_error"
            log.write(f"{type(error).__name__}: {error}\n")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    observed = {}
    sweep = None
    missing = expected.copy()
    if contract.analyzer:
        observed, sweep, missing = _broadband_diagnostics(log_text, contract.sweep_hz)
    else:
        for name, raw in _RESULT.findall(log_text):
            value = float(raw)
            if name.lower() in expected and math.isfinite(value):
                observed[name.lower()] = value
        missing -= set(observed)
    if exit_code == 0 and not missing:
        state = "simulated"
    result = {
        "state": state,
        "authority": "public_diagnostic",
        "task_correctness": "not_evaluated",
        "task_id": task_id,
        "candidate_sha256": file_digest(snapshot),
        "measurements": observed,
        "missing_measurements": sorted(missing),
        "container_exit": exit_code,
        "cleanup_confirmed": cleanup_confirmed,
        "wall_clock_s": time.monotonic() - started,
        "diagnostic_excerpt": log_text[-4000:] if state != "simulated" else "",
    }
    if sweep is not None:
        result["sweep"] = sweep
    atomic_json(output / "result.json", result)
    return result


def _broadband_diagnostics(text: str, sweep_hz: tuple[float, float, int] | None):
    """Read upstream ngspice print tables; an incomplete sweep is not a measurement."""
    if sweep_hz is None:
        raise ValueError("broadband public diagnostic requires a declared frequency sweep")
    start, stop, count = sweep_hz
    frequencies = [start + index * (stop - start) / (count - 1) for index in range(count)]
    rows: dict[str, list[tuple[int, float, float]]] = {"gamma": [], "transducer_gain": []}
    active = None
    for line in text.splitlines():
        header = re.fullmatch(r"\s*Index\s+frequency\s+(gamma|transducer_gain)\s*", line)
        if header:
            active = header.group(1)
        elif line.lstrip().startswith("Index"):
            active = None
        elif active and re.match(r"\s*\d+\s", line):
            parts = line.split()
            try:
                if len(parts) != 3:
                    raise ValueError("not a real scalar row")
                rows[active].append((int(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                rows[active].append((-1, math.nan, math.nan))
    missing = {
        name
        for name, samples in rows.items()
        if len(samples) != count
        or any(
            index != position
            or not math.isfinite(frequency)
            or not math.isclose(frequency, frequencies[position], rel_tol=1e-6)
            or not math.isfinite(value)
            or value < 0
            or (name == "transducer_gain" and value == 0)
            for position, (index, frequency, value) in enumerate(samples)
        )
    }
    if missing:
        return {}, {}, missing
    gamma = [row[2] for row in rows["gamma"]]
    gain = [row[2] for row in rows["transducer_gain"]]
    return (
        {
            "worst_gamma": max(gamma),
            "median_gamma": statistics.median(gamma),
            "min_transducer_gain": min(gain),
            "max_insertion_loss_db": -10 * math.log10(min(gain)),
            "center_insertion_loss_db": -10 * math.log10(gain[count // 2]),
        },
        {"frequency_hz": frequencies, "gamma": gamma, "transducer_gain": gain},
        set(),
    )
