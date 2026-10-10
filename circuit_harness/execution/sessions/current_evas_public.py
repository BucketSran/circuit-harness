"""OS-isolated public EVAS execution with no hidden mounts or host fallback."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from circuit_harness.execution.backends.current_evas import _CLI, _validate_result, read_diagnostic
from circuit_harness.execution.evaluation.candidate_bundle import regular_file, verify_candidate
from circuit_harness.execution.runtime.journal import Journal, atomic_json, file_digest
from circuit_harness.execution.runtime.native_sandbox import NativeSandbox
from circuit_harness.execution.runtime.process import run_process

_CONTAINER_WATCHDOG = """
import json, os, signal, stat, subprocess, sys, time
seconds, encoded, limits = sys.argv[1:]
roots, maximum = json.loads(limits)
def output_bytes():
    total = 0
    for root in roots:
        for parent, _, names in os.walk(root, followlinks=False):
            for name in names:
                try:
                    info = os.lstat(os.path.join(parent, name))
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(info.st_mode): total += info.st_size
    return total
try:
    child = subprocess.Popen(json.loads(encoded), start_new_session=True)
except OSError as error:
    print(type(error).__name__, file=sys.stderr)
    sys.exit(127)
try:
    deadline = time.monotonic() + float(seconds)
    while child.poll() is None:
        if output_bytes() > maximum:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            sys.exit(123)
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(child.args, float(seconds))
        time.sleep(0.05)
    status = 123 if output_bytes() > maximum else child.returncode
except subprocess.TimeoutExpired:
    os.killpg(child.pid, signal.SIGKILL)
    child.wait()
    status = 124
sys.exit(status if status >= 0 else 128 - status)
"""


def validate_cpu_limit(value):
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1
    ):
        raise ValueError("cpu_limit must be in (0, 1] or explicit null")


def run_isolated_docker(**kwargs):
    """Compatible Docker entry for existing callers and independent replay."""
    return run_isolated_container(**kwargs)


def run_isolated_container(
    *,
    image: str,
    readonly: dict[Path, str],
    writable: dict[Path, str],
    command: list[str],
    directory: Path,
    action_id: str,
    timeout_s: float,
    max_output_bytes: int,
    workdir="/output",
    cancel=None,
    backend="docker",
    cpu_limit=1,
) -> dict:
    """Run one declared command with fixed container access and explicit CPU policy."""
    if backend not in {"docker", "podman"}:
        raise ValueError("unsupported container backend")
    validate_cpu_limit(cpu_limit)
    if not isinstance(image, str) or not re.fullmatch(
        r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}", image
    ):
        raise ValueError("Docker requires an immutable local image ID")
    if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 300:
        raise ValueError("invalid Docker deadline")
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 16 * 1024 * 1024:
        raise ValueError("invalid Docker output limit")
    if not command or any(not isinstance(part, str) or "\0" in part for part in command):
        raise ValueError("invalid Docker command")
    destinations = list(readonly.values()) + list(writable.values())
    if any(not name.startswith("/") or ".." in Path(name).parts for name in destinations):
        raise ValueError("mount destinations must be absolute and canonical")
    if len(set(destinations)) != len(destinations):
        raise ValueError("overlapping mount destinations")
    if (
        any(
            a == "/" or Path(a) in Path(b).parents or Path(b) in Path(a).parents
            for index, a in enumerate(destinations)
            for b in destinations[index + 1 :]
        )
        or "/" in destinations
    ):
        raise ValueError("overlapping mount destinations")
    name = "chips-public-" + uuid.uuid4().hex
    argv = [
        backend,
        "run",
        "--rm",
        "--name",
        name,
        "--pull=never",
        "--log-driver=none",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--memory=512m",
        "--memory-swap=512m",
        "--ulimit",
        f"fsize={max_output_bytes}:{max_output_bytes}",
        "--tmpfs",
        "/tmp:rw,nosuid,noexec,size=16m",
        f"--workdir={workdir}",
    ]
    if cpu_limit is not None:
        argv.append(f"--cpus={cpu_limit}")
    for mapping, is_readonly in ((readonly, True), (writable, False)):
        for source, destination in mapping.items():
            if "," in str(source) or "," in destination:
                raise ValueError("Docker mount paths cannot contain commas")
            argv.extend(
                [
                    "--mount",
                    f"type=bind,src={Path(source).absolute()},dst={destination}"
                    + (",readonly" if is_readonly else ""),
                ]
            )
    # This in-container watchdog also expires when the broker/CLI disappears.
    # Supported images must contain Python 3; no package install or image pull occurs.
    argv.extend(
        [
            "--entrypoint=python3",
            image,
            "-I",
            "-S",
            "-B",
            "-c",
            _CONTAINER_WATCHDOG,
            str(timeout_s),
            json.dumps(command),
            json.dumps([list(writable.values()), max_output_bytes]),
        ]
    )
    journal = Journal(directory, call_id=action_id)
    atomic_json(
        directory / "backend.json",
        {"backend": backend, "image": image, "cpu_limit": cpu_limit, "argv": argv},
    )
    try:
        process = run_process(
            argv,
            directory=directory,
            stage="evas",
            deadline=time.monotonic() + timeout_s,
            cancel=cancel if cancel is not None else threading.Event(),
            journal=journal,
            max_output_bytes=max_output_bytes,
        )
    finally:
        try:
            removed = subprocess.run(
                [backend, "rm", "-f", name],
                capture_output=True,
                timeout=30 if backend == "podman" else 10,
            )
            cleanup = removed.returncode == 0 or any(
                message in removed.stderr.lower()
                for message in (b"no such container", b"no container with name")
            )
            if not cleanup:
                # --rm may race rm -f after an output-limit kill. Confirm absence,
                # rather than treating an in-progress automatic removal as a leak.
                for _ in range(10):
                    inspect = subprocess.run(
                        [backend, "inspect", name], capture_output=True, timeout=2
                    )
                    if inspect.returncode != 0 and any(
                        message in inspect.stderr.lower()
                        for message in (
                            b"no such object",
                            b"no such container",
                            b"no container with name",
                        )
                    ):
                        cleanup = True
                        break
                    time.sleep(0.1)
            atomic_json(
                directory / "cleanup.json",
                {
                    "container": name,
                    "confirmed": cleanup,
                    "remove_returncode": removed.returncode,
                    "remove_stderr": removed.stderr.decode(errors="replace")[-2000:],
                },
            )
        except (OSError, subprocess.TimeoutExpired):
            cleanup = False
    process["cleanup_confirmed"] = cleanup and process["cleanup_confirmed"]
    if not cleanup:
        process["execution"] = "cleanup_failed"
    elif process["execution"] == "ok" and process["returncode"] == 123:
        process["execution"] = "output_limit"
    elif process["execution"] == "ok" and process["returncode"] == 124:
        process["execution"] = "timeout"
    elif process["execution"] == "ok" and process["returncode"] in (125, 126, 127):
        process["execution"] = "infrastructure_error"
    return process


def _native_process(*, directory, config, inputs, runtime, output, artifacts, action_id):
    python = Path(config["python"]).resolve()
    codex = Path(config["codex"])
    if (
        file_digest(python) != config["python_sha256"]
        or file_digest(codex) != config["codex_sha256"]
    ):
        raise ValueError("native executable identity changed")
    base_prefix = subprocess.check_output(
        [str(python), "-I", "-S", "-c", "import sys; print(sys.base_prefix)"], text=True, timeout=10
    ).strip()
    sandbox = NativeSandbox(
        codex=codex,
        directory=output / "policy",
        workspace=artifacts,
        readonly_paths=(
            inputs,
            runtime / "source/files/evas/src",
            runtime / "evas-kernel",
            Path(base_prefix),
            python.resolve().parent,
        ),
        protected_paths=(
            Path(config["source"]["checkout"]),
            directory / "session.json",
            directory / "submission",
            directory / "public",
        ),
    )
    sandbox.probe(timeout_s=min(config["timeout_s"], 20))
    # Limits are inherited by the kernel; no credentials or ambient EVAS controls enter.
    cpu = math.ceil(config["timeout_s"])
    quota = config["max_output_bytes"]
    limits = f"""
import resource
for kind, requested in ((resource.RLIMIT_CPU, {cpu}),
                        (resource.RLIMIT_FSIZE, {quota}),
                        (resource.RLIMIT_NOFILE, 64)):
    _, hard = resource.getrlimit(kind)
    limit = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    resource.setrlimit(kind, (limit, limit))
"""
    command, overrides = sandbox.command(
        [
            str(python),
            "-I",
            "-S",
            "-B",
            "-c",
            limits + _CLI,
            str(runtime / "source/files/evas/src"),
            str(artifacts / "runtime.json"),
            "transient",
            str(inputs / "manifest.json"),
            "--kernel",
            str(runtime / "evas-kernel"),
            "--timeout",
            str(config["timeout_s"]),
        ]
    )
    atomic_json(
        output / "backend.json",
        {
            "backend": "native_codex_sandbox",
            "policy": sandbox.receipt,
            "memory_limit": "not_enforced_on_native_backend",
            "pids_limit": "not_enforced_on_native_backend",
            "argv": command,
            "python_sha256": config["python_sha256"],
            "codex_sha256": config["codex_sha256"],
        },
    )
    return run_process(
        [
            "/usr/bin/env",
            "-i",
            "PATH=" + os.defpath,
            *[key + "=" + value for key, value in overrides.items()],
            *command,
        ],
        directory=output,
        stage="evas",
        deadline=time.monotonic() + config["timeout_s"],
        cancel=threading.Event(),
        journal=Journal(output, call_id=action_id),
        max_output_bytes=config["max_output_bytes"],
    )


def run_public(
    *,
    directory: Path,
    config: dict,
    candidate: Path,
    output: Path,
    action_id: str,
    experiment: Path | None = None,
    script: str | None = None,
) -> dict:
    """Execute the pinned source and compatible kernel inside the selected OS boundary."""
    frozen = verify_candidate(candidate)
    if (
        frozen["task_id"] != config["task"]["task_id"]
        or frozen["task_version"] != config["task"]["task_version"]
    ):
        raise ValueError("candidate identity belongs to another task")
    runtime = directory / "runtime"
    for name, record in config["source"]["files"].items():
        if file_digest(regular_file(runtime / "source/files", name)) != record["sha256"]:
            raise ValueError("session EVAS source changed")
    source_root = runtime / "source/files"
    present = set()
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in session EVAS source")
        if path.is_file():
            present.add(path.relative_to(source_root).as_posix())
    if present != set(config["source"]["files"]):
        raise ValueError("session EVAS source inventory changed")
    kernel = runtime / "evas-kernel"
    if kernel.is_symlink() or file_digest(kernel) != config["kernel_sha256"]:
        raise ValueError("session EVAS kernel changed")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    artifacts.mkdir(mode=0o700)
    inputs = output.parent / "inputs"
    complete = verify_candidate(experiment) if experiment is not None else frozen
    if experiment is not None:
        declaration = config["task"].get("experiments")
        if (
            config["backend"] not in {"docker", "podman"}
            or not declaration
            or script not in declaration["files"]
            or not script.endswith(".py")
            or complete["task_id"] != config["task"]["task_id"]
            or complete["task_version"]
            != json.dumps(
                [config["task"]["task_version"], declaration["version"]], separators=(",", ":")
            )
            or set(complete["files"])
            != set(
                config["task"]["candidate_files"]
                + config["task"]["public_files"]
                + declaration["files"]
            )
            or any(complete["files"][name] != entry for name, entry in frozen["files"].items())
        ):
            raise ValueError("experiment snapshot does not match declared frozen candidate")
    shutil.copytree((experiment if experiment is not None else candidate) / "files", inputs)
    for name, record in complete["files"].items():
        if file_digest(regular_file(inputs, name)) != record["sha256"]:
            raise ValueError("frozen candidate changed while staging")
    atomic_json(inputs / "manifest.json", config["task"]["manifest"])
    backend = config["backend"]
    atomic_json(
        output / "request.json",
        {
            "backend": backend,
            "image": config["image"],
            "candidate": frozen,
            "source": config["source"],
            "kernel_sha256": config["kernel_sha256"],
            "manifest": config["task"]["manifest"],
            "experiment": complete if experiment is not None else None,
            "script": script,
        },
    )
    if backend in {"docker", "podman"}:
        process = run_isolated_container(
            backend=backend,
            cpu_limit=config.get("cpu_limit", 1),
            image=config["image"],
            readonly={
                runtime / "source/files/evas/src": "/engine",
                kernel: "/kernel",
                inputs: "/inputs",
            },
            writable={artifacts: "/output"},
            command=(
                ["python3", "-I", "-S", "-B", "/inputs/" + script]
                if experiment is not None
                else [
                    "python3",
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    _CLI,
                    "/engine",
                    "/output/runtime.json",
                    "transient",
                    "/inputs/manifest.json",
                    "--kernel",
                    "/kernel",
                    "--timeout",
                    str(config["timeout_s"]),
                ]
            ),
            directory=output,
            action_id=action_id,
            timeout_s=config["timeout_s"],
            max_output_bytes=config["max_output_bytes"],
        )
    elif backend == "native_codex_sandbox":
        try:
            process = _native_process(
                directory=directory,
                config=config,
                inputs=inputs,
                runtime=runtime,
                output=output,
                artifacts=artifacts,
                action_id=action_id,
            )
        except (RuntimeError, subprocess.SubprocessError, OSError) as error:
            (output / "evas.stderr.log").write_text(type(error).__name__)
            (output / "evas.stdout.log").touch()
            process = {
                "execution": "infrastructure_error",
                "returncode": None,
                "cleanup_confirmed": True,
            }
    else:
        raise ValueError("unsupported public backend")
    execution = process["execution"]
    if execution == "ok" and process["returncode"] != 0:
        execution = "backend_error"
    data = None
    if execution == "ok":
        try:
            data = json.loads((output / "evas.stdout.log").read_text())
            if experiment is None:
                _validate_result(data, config["task"]["manifest"])
                data = {key: data[key] for key in ("engine", "nodes", "solutions", "transient")}
            elif not isinstance(data, dict) or set(data) & {
                "reward",
                "score",
                "task_correctness",
                "authority",
            }:
                raise ValueError("public measurement must be an object without grading fields")
        except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError):
            execution = "invalid_result"
    diagnostics = (output / "evas.stderr.log").read_text(errors="replace")[-4000:]
    for value in (str(directory), str(candidate), str(output), config["source"]["checkout"]):
        diagnostics = diagnostics.replace(value, "[operator-runtime]")
    result = {
        "execution": execution,
        "backend": backend,
        "image": config["image"],
        "cleanup_confirmed": process["cleanup_confirmed"],
        "diagnostics": diagnostics,
        "observations": data if execution == "ok" else None,
    }
    if experiment is None and execution == "backend_error":
        result["diagnostic"] = read_diagnostic(output / "evas.stderr.log")
    atomic_json(output / "result.json", result)
    return result
