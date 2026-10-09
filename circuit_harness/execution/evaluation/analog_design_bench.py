"""Pinned, operator-side Analog Design Bench verification.

The task source, including reference solutions and verifier, never enters an
agent workspace. This module runs the upstream test.sh against one candidate;
it does not claim that the candidate was produced by an agent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from circuit_harness.execution.runtime.journal import atomic_json, file_digest


@dataclass(frozen=True)
class RlcContract:
    """Public passive-circuit interface and upstream diagnostic entry point."""

    subcircuit: str
    ports: tuple[str, ...]
    benches: tuple[str, ...]
    analyzer: str | None = None
    sweep_hz: tuple[float, float, int] | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @property
    def public_files(self) -> dict[str, str]:
        files = {
            "instruction.md": "instruction.md",
            "starter/circuit.spi": "environment/starter/circuit.spi",
        }
        for name in (*self.benches, *((self.analyzer,) if self.analyzer else ())):
            files[f"testbench/{name}"] = f"environment/starter/testbench/{name}"
        return files


@dataclass(frozen=True)
class Task:
    commit: str
    source_sha256: str
    image: str
    public_rlc: RlcContract | None = None


TASKS = {
    "rlc-rf-bandpass-100mhz": Task(
        "fb0ec30463d005d3e463caf4e48ab9a26008e869",
        "5ff88ba55bd7108fba3f37a4d257dd47eb04ee5c0771102229c2704c0f728310",
        "ghcr.io/arcadia-1/circuit-bench-sky130-ngspice@sha256:bd5c425675eb99fc1a2c3bca10b63a871c457613767e2c6984d6c207b3160500",
        RlcContract("rlc_rf_bandpass", ("IN", "OUT", "COM"), ("tb_ac.spi", "tb_stopband.spi")),
    ),
    "rlc-broadband-50-to-200-match": Task(
        "fb0ec30463d005d3e463caf4e48ab9a26008e869",
        "7f4efffcb5d2d50ea90cc8c520c8f17b959f26cbe2ebb23f7d455af2c5953b21",
        "ghcr.io/arcadia-1/circuit-bench-sky130-ngspice@sha256:bd5c425675eb99fc1a2c3bca10b63a871c457613767e2c6984d6c207b3160500",
        RlcContract(
            "rlc_broadband_match",
            ("IN", "OUT", "COM"),
            ("tb_ac.spi",),
            "analyze_broadband.py",
            (3.30e9, 3.80e9, 11),
        ),
    ),
    "sky130-ota-5t-gain40-pm60-noise50uv-pvt": Task(
        "c23f124de1e461655d2e02ce6cfae2654ccea0d3",
        "55e6145bf287bf0dd8ab18fe969ae85ebc5f6d3bee0d361dd5df0092ccf3a00f",
        "ghcr.io/arcadia-1/analog-arena-sky130-tools@sha256:668783dd8cbcd5ebce5264785803ce73a9466e79dbe4c98cdf4566348a33437a",
    ),
}


def rlc_contract(task_id: str) -> RlcContract:
    task = TASKS.get(task_id)
    if task is None or task.public_rlc is None:
        raise ValueError(f"task has no supported public RLC contract: {task_id}")
    return task.public_rlc


def tree_digest(root: Path) -> str:
    """Hash path, length and bytes of every source file in stable order."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("task source must be a regular directory")
    value = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symbolic link in task source")
        if path.is_file():
            data = path.read_bytes()
            value.update(path.relative_to(root).as_posix().encode() + b"\0")
            value.update(str(len(data)).encode() + b"\0")
            value.update(data)
        elif not path.is_dir():
            raise ValueError("unsupported task source entry")
    return value.hexdigest()


def read_result(path: Path, *, container_exit: int | None) -> dict:
    """A verifier failure never becomes a circuit score, even with stale output."""
    result = {"state": "verifier_error", "score": None, "container_exit": container_exit}
    if container_exit != 0 or not path.is_file():
        return result
    try:
        reward = json.loads(path.read_text(encoding="utf-8"))
        score = reward["reward"]
        total = reward["tests_total"]
        passed = reward["tests_passed"]
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 0 <= score <= 1
            or type(total) is not int
            or type(passed) is not int
            or not 0 < total
            or not 0 <= passed <= total
        ):
            return result
    except (OSError, ValueError, KeyError, TypeError):
        return result
    return {
        **result,
        "state": "graded",
        "score": float(score),
        "tests_total": total,
        "tests_passed": passed,
    }


def ngspice_version(banner: str) -> str | None:
    match = re.search(r"\bngspice-\d+(?:\.\d+)?\b", banner, re.IGNORECASE)
    return match.group(0) if match else None


def run_case(
    task_id: str,
    source_root: Path,
    candidate: Path,
    output: Path,
    *,
    podman: str = "podman",
    podman_root: Path | None = None,
    podman_runroot: Path | None = None,
    runtime_image: str | None = None,
    offline_image_archive: Path | None = None,
    podman_single_id: bool = False,
    podman_no_cpu_limit: bool = False,
    backend: str = "podman",
    ngspice: Path | None = None,
    timeout_s: int = 900,
    archive_root: Path | None = None,
) -> dict:
    """Run an upstream verifier in Podman or a networkless native sandbox."""
    if task_id not in TASKS:
        raise ValueError(f"unsupported Analog Design Bench task: {task_id}")
    if backend not in {"podman", "bubblewrap"}:
        raise ValueError("backend must be podman or bubblewrap")
    if not 1 <= timeout_s <= 3600:
        raise ValueError("timeout_s must be between 1 and 3600")
    if (podman_root is None) != (podman_runroot is None):
        raise ValueError("Podman root and runroot must be provided together")
    if backend != "podman" and (
        runtime_image or offline_image_archive or podman_single_id or podman_no_cpu_limit
    ):
        raise ValueError("offline image options require the Podman backend")
    if (runtime_image is None) != (offline_image_archive is None):
        raise ValueError("runtime image and offline image archive must be provided together")
    if runtime_image is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_image):
        raise ValueError("runtime image must be a local sha256 image ID")
    if offline_image_archive is not None and not Path(offline_image_archive).is_file():
        raise ValueError("offline image archive is unavailable")
    if backend == "bubblewrap" and (podman_root is not None or ngspice is None):
        raise ValueError("bubblewrap requires ngspice and excludes Podman storage options")
    if backend == "bubblewrap":
        ngspice = Path(ngspice).absolute()
        if not ngspice.is_file() or not os.access(ngspice, os.X_OK):
            raise ValueError("ngspice executable is unavailable")
        if task_id.startswith("sky130-") and not all(
            Path(path).is_file()
            for path in (
                "/opt/sky130/continuous/sky130.lib.spice",
                "/opt/analog-arena/check_circuit.py",
            )
        ):
            raise ValueError(
                "Sky130 model and upstream checker are required for native OTA verification"
            )
    task = TASKS[task_id]
    source = (Path(source_root) / "tasks" / task_id).absolute()
    actual_hash = tree_digest(source)
    if actual_hash != task.source_sha256:
        raise ValueError(f"task source pin mismatch for {task_id}: {actual_hash}")
    candidate = Path(candidate).absolute()
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or not 0 < candidate.stat().st_size <= 100_000
    ):
        raise ValueError("candidate must be a nonempty regular file of at most 100 KB")
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("use a new output directory for each verifier attempt")
    archive = None
    if archive_root is not None:
        archive = Path(archive_root).absolute() / output.name
        if archive == output or archive.exists():
            raise ValueError("archive target already exists or equals output")
    output.mkdir(parents=True, mode=0o700)
    (output / "verifier").mkdir(mode=0o700)
    (output / "inputs").mkdir(mode=0o700)
    snapshot = output / "inputs/circuit.spi"
    shutil.copyfile(candidate, snapshot)
    snapshot.chmod(0o600)
    simulator_version = None
    offline_archive_hash = (
        file_digest(Path(offline_image_archive)) if offline_image_archive is not None else None
    )
    if backend == "bubblewrap":
        version = subprocess.run(
            [str(ngspice), "--version"], capture_output=True, text=True, timeout=5, check=False
        )
        simulator_version = (
            ngspice_version(version.stdout or version.stderr) if version.returncode == 0 else None
        )
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "task_id": task_id,
            "source_commit": task.commit,
            "source_sha256": actual_hash,
            "candidate_sha256": file_digest(snapshot),
            "image": task.image,
            "runtime_image": runtime_image or task.image,
            "offline_image_archive_sha256": offline_archive_hash,
            "podman_cpu_quota": 4 if backend == "podman" and not podman_no_cpu_limit else None,
            "authority": "upstream_verifier_operator_run",
            "backend": backend,
            "ngspice": str(ngspice) if ngspice is not None else None,
            "ngspice_version": simulator_version,
        },
    )
    if backend == "podman":
        command = [podman]
        if podman_root is not None:
            command += ["--root", str(podman_root), "--runroot", str(podman_runroot)]
        if podman_single_id:
            command += ["--storage-opt", "ignore_chown_errors=true"]
        command += [
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--memory=2g",
        ]
        if not podman_no_cpu_limit:
            command += ["--cpus=4"]
        command += [
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,size=512m",
            "--volume",
            f"{source / 'tests'}:/app/analog_arena_tests:ro",
            "--volume",
            f"{snapshot}:/app/circuit.spi:ro",
            "--volume",
            f"{output / 'verifier'}:/logs/verifier:rw",
            runtime_image or task.image,
            "/bin/bash",
            "-c",
            (
                "command -v ngspice && test -x /opt/analog-arena/check_circuit.py "
                "&& test -s /opt/sky130/continuous/sky130.lib.spice "
                "&& exec /bin/bash /app/analog_arena_tests/test.sh"
            )
            if task_id.startswith("sky130-")
            else ("command -v ngspice && exec /bin/bash /app/analog_arena_tests/test.sh"),
        ]
    else:
        command = [
            "bwrap",
            "--die-with-parent",
            "--unshare-net",
            "--unshare-pid",
            "--tmpfs",
            "/",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/etc",
            "/etc",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--dir",
            "/tmp",
            "--dir",
            "/app",
            "--dir",
            "/logs",
            "--dir",
            "/opt",
            "--dir",
            "/opt/chips-python",
            "--symlink",
            str(Path(sys.executable).resolve()),
            "/opt/chips-python/python3",
            "--ro-bind",
            str(ngspice.parent.parent),
            "/opt/chips-ngspice",
            "--ro-bind",
            str(source / "tests"),
            "/app/analog_arena_tests",
            "--ro-bind",
            str(snapshot),
            "/app/circuit.spi",
            "--bind",
            str(output / "verifier"),
            "/logs/verifier",
        ]
        if task_id.startswith("sky130-"):
            command += [
                "--ro-bind",
                "/opt/sky130",
                "/opt/sky130",
                "--ro-bind",
                "/opt/analog-arena",
                "/opt/analog-arena",
            ]
        command += [
            "--setenv",
            "PATH",
            "/opt/chips-python:/opt/chips-ngspice/bin:/usr/local/bin:/usr/bin:/bin",
            "/bin/bash",
            "/app/analog_arena_tests/test.sh",
        ]
    # Pass only runtime variables; an inherited provider key must not reach the verifier.
    safe_env = {
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
    process_env = {key: value for key, value in os.environ.items() if key in safe_env}
    started = time.monotonic()
    timed_out = False
    with (output / "verifier.log").open("w", encoding="utf-8") as log:
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
            timed_out = True
            exit_code = None
    result = read_result(output / "verifier/reward.json", container_exit=exit_code)
    if timed_out:
        result["state"] = "timeout"
    result.update(
        {
            "task_id": task_id,
            "backend": backend,
            "wall_clock_s": time.monotonic() - started,
            "timed_out": timed_out,
        }
    )
    atomic_json(output / "result.json", result)
    if archive is not None:
        archive_root = archive.parent
        archive_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copytree(output, archive)
        for original in output.rglob("*"):
            if original.is_file() and file_digest(original) != file_digest(
                archive / original.relative_to(output)
            ):
                raise OSError(f"archive integrity check failed: {original.name}")
        result["archive"] = str(archive)
        atomic_json(output / "result.json", result)
        atomic_json(archive / "result.json", result)
    return result
