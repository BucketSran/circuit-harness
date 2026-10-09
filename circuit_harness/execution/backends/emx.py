"""JSON → GDS → idempotent SSH job → verified local EMX artifacts."""

from __future__ import annotations

import fcntl
import json
import math
import re
import shlex
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path

from circuit_harness.execution.runtime.journal import Journal, atomic_json, file_digest
from circuit_harness.execution.runtime.process import run_process


@dataclass(frozen=True)
class EmxConfig:
    ssh_host: str
    remote_root: str
    converter: list[str]
    emx_command: list[str]
    outputs: list[str] = field(default_factory=list)
    remote_input_files: list[str] = field(default_factory=list)
    remote_python: str = "python3"
    timeout_s: float = 1800
    command_timeout_s: float = 30
    max_attempts: int = 3
    max_output_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", self.ssh_host):
            raise ValueError("ssh_host must be an SSH config alias or user@hostname")
        if (
            not re.fullmatch(r"/[A-Za-z0-9_./-]+", self.remote_root)
            or ".." in Path(self.remote_root).parts
            or self.remote_root == "/"
        ):
            raise ValueError("remote_root must be a dedicated absolute path without spaces or '..'")
        for name in ("converter", "emx_command"):
            command = getattr(self, name)
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(x, str) or not x or "\x00" in x for x in command)
            ):
                raise ValueError(f"{name} must be a nonempty argv list")
        if not all(
            marker in " ".join(self.converter) for marker in ("{input_json}", "{output_gds}")
        ):
            raise ValueError("converter requires {input_json} and {output_gds}")
        if "{gds}" not in " ".join(self.emx_command):
            raise ValueError("emx_command requires {gds}")
        reserved = {
            "input.gds",
            "request.json",
            "result.json",
            "events.jsonl",
            "provenance.json",
            "cancel",
            ".lock",
            "emx.stdout.log",
            "emx.stderr.log",
        }
        if (
            not isinstance(self.outputs, list)
            or len(set(self.outputs)) != len(self.outputs)
            or any(
                not isinstance(x, str)
                or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", x)
                or x in reserved
                for x in self.outputs
            )
        ):
            raise ValueError("outputs must be unique non-reserved file basenames")
        if not isinstance(self.remote_input_files, list) or any(
            not isinstance(x, str) or not x.startswith("/") for x in self.remote_input_files
        ):
            raise ValueError("remote_input_files must contain absolute paths")
        if (
            not isinstance(self.remote_python, str)
            or not self.remote_python
            or "\x00" in self.remote_python
        ):
            raise ValueError("remote_python must name an executable")
        for name in ("timeout_s", "command_timeout_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 86400
            ):
                raise ValueError(f"{name} must be finite and in (0, 86400]")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 10:
            raise ValueError("max_attempts must be in [1, 10]")
        if type(self.max_output_bytes) is not int or not 1024 <= self.max_output_bytes <= 1024**3:
            raise ValueError("max_output_bytes must be in [1024, 1073741824]")

    @classmethod
    def load(cls, path: Path) -> EmxConfig:
        return cls(**json.loads(path.read_text()))


def build_worker(destination: Path) -> str:
    package = files("circuit_harness.execution")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        sources = {
            f"circuit_harness/execution/{name}": package.joinpath(name).read_bytes()
            for name in (
                "ssh_worker.py",
                "journal.py",
                "process.py",
                "transport/ssh_worker.py",
                "runtime/journal.py",
                "runtime/process.py",
                "transport/__init__.py",
                "runtime/__init__.py",
            )
        }
        for prefix in (
            "circuit_harness",
            "circuit_harness/execution",
        ):
            sources[f"{prefix}/__init__.py"] = b""
        sources["__main__.py"] = b"from circuit_harness.execution.ssh_worker import main\nmain()\n"
        for name, content in sorted(sources.items()):
            archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), content)
    return file_digest(destination)


class TransportFailure(RuntimeError):
    pass


class SshTransport:
    def __init__(
        self,
        config: EmxConfig,
        directory: Path,
        journal: Journal,
        cancel: threading.Event,
        deadline: float,
    ):
        self.config, self.directory, self.journal = config, directory, journal
        self.cancel, self.deadline = cancel, deadline
        self.sequence = 0
        self.attempt_id = uuid.uuid4().hex[:12]

    def command(self, argv: list[str], stage: str) -> str:
        import random

        for attempt in range(1, self.config.max_attempts + 1):
            self.sequence += 1
            log_stage = f"{self.attempt_id}-{self.sequence:04d}-{stage}-{attempt}"
            self.journal.emit("transport_attempt", stage=stage, attempt=attempt)
            result = run_process(
                argv,
                directory=self.directory / "transport",
                stage=log_stage,
                deadline=min(self.deadline, time.monotonic() + self.config.command_timeout_s),
                cancel=self.cancel,
                journal=self.journal,
                max_output_bytes=self.config.max_output_bytes,
            )
            out = self.directory / "transport" / result["stdout"]
            err = self.directory / "transport" / result["stderr"]
            if result["execution"] == "ok" and result["returncode"] == 0:
                return out.read_text(errors="replace")
            diagnostic = err.read_text(errors="replace")[-4000:]
            fatal = any(
                text in diagnostic.lower()
                for text in (
                    "permission denied",
                    "host key verification failed",
                    "remote host identification has changed",
                    "could not resolve hostname",
                )
            )
            connection_failure = any(
                message in diagnostic.lower()
                for message in (
                    "connection reset",
                    "connection closed",
                    "connection timed out",
                    "connection refused",
                    "broken pipe",
                    "lost connection",
                    "no route to host",
                )
            )
            retryable = result["execution"] == "timeout" or (
                result["execution"] == "ok"
                and (
                    result["returncode"] == 255
                    or (result["returncode"] == 1 and connection_failure)
                )
            )
            if (
                self.cancel.is_set()
                or fatal
                or not retryable
                or attempt == self.config.max_attempts
            ):
                raise TransportFailure(
                    f"{stage}: {result['execution']}, exit={result['returncode']}; {diagnostic}"
                )
            delay = min(0.5 * 2 ** (attempt - 1) * random.uniform(0.8, 1.2), 5)
            if time.monotonic() + delay >= self.deadline:
                raise TransportFailure(f"{stage}: deadline exhausted")
            self.journal.emit("retry_wait", stage=stage, next_attempt=attempt + 1, delay_s=delay)
            self.cancel.wait(delay)
        raise AssertionError("unreachable")

    def ssh(self, argv: list[str], stage: str) -> str:
        return self.command(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=10",
                self.config.ssh_host,
                shlex.join(argv),
            ],
            stage,
        )

    def copy(self, local: Path, remote: str, *, upload: bool) -> None:
        source, destination = (
            (str(local), f"{self.config.ssh_host}:{remote}")
            if upload
            else (f"{self.config.ssh_host}:{remote}", str(local))
        )
        self.command(
            [
                "scp",
                "-q",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=10",
                source,
                destination,
            ],
            "upload" if upload else "download",
        )


def run_emx(
    payload: dict,
    config: EmxConfig,
    directory: Path,
    *,
    resume: bool = False,
    cancel: threading.Event | None = None,
    call_id: str = "direct",
) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("layout input must be a JSON object")
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > 1024 * 1024:
        raise ValueError("layout JSON exceeds 1 MiB")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_emx(payload, config, directory, resume, cancel or threading.Event(), call_id)


def _run_emx(payload, config, directory, resume, cancel, call_id):
    worker = directory / "worker.pyz"
    worker_hash = build_worker(worker)
    converter_files = {
        str(Path(x).resolve()): file_digest(Path(x)) for x in config.converter if Path(x).is_file()
    }
    identity = {
        "input": payload,
        "config": asdict(config),
        "converter_files": converter_files,
        "worker_sha256": worker_hash,
    }
    receipt_path = directory / "intent.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if not resume or receipt["identity"] != identity:
            raise ValueError(
                "existing run requires resume with identical input, config and tool content"
            )
    else:
        receipt = {"job_id": uuid.uuid4().hex, "identity": identity}
        atomic_json(receipt_path, receipt)
    job_id = receipt["job_id"]
    journal = Journal(directory, call_id=call_id)
    journal.emit("run_started", job_id=job_id, mode="ssh_emx", attempt_id=uuid.uuid4().hex)
    deadline = time.monotonic() + config.timeout_s
    transport = SshTransport(config, directory, journal, cancel, deadline)
    remote_worker = f"{config.remote_root}/worker-{worker_hash}.pyz"
    remote_job = f"{config.remote_root}/{job_id}"
    base = [config.remote_python, remote_worker]

    def query(action, *args):
        reply = json.loads(
            transport.ssh([*base, action, config.remote_root, job_id, *args], action)
        )
        if reply.get("job_id") != job_id:
            raise TransportFailure("remote job identity mismatch")
        return reply

    submitted = False
    try:
        checkpoint = directory / "gds.json"
        gds = directory / "input.gds"
        if resume and checkpoint.exists():
            if (
                not gds.is_file()
                or file_digest(gds) != json.loads(checkpoint.read_text())["sha256"]
            ):
                raise ValueError("GDS checkpoint has been modified")
            journal.emit("gds_reused")
        else:
            atomic_json(directory / "input.json", payload)
            command = [
                x.replace("{input_json}", str(directory / "input.json")).replace(
                    "{output_gds}", str(gds)
                )
                for x in config.converter
            ]
            conversion = run_process(
                command,
                directory=directory,
                stage="convert",
                deadline=deadline,
                cancel=cancel,
                journal=journal,
                max_output_bytes=config.max_output_bytes,
            )
            if (
                conversion["execution"] != "ok"
                or conversion["returncode"] != 0
                or not gds.is_file()
                or gds.stat().st_size < 4
            ):
                return _finish(
                    directory,
                    journal,
                    {
                        "execution": conversion["execution"]
                        if conversion["execution"] != "ok"
                        else "conversion_failed",
                        "verdict": "not_evaluated",
                        "stage": "json_to_gds",
                        "job_id": job_id,
                    },
                )
            atomic_json(checkpoint, {"sha256": file_digest(gds)})
            journal.emit("gds_checkpoint", sha256=file_digest(gds))
        transport.ssh(["mkdir", "-p", config.remote_root], "prepare")
        transport.copy(worker, remote_worker, upload=True)
        state = query("status")
        if state["state"] == "missing":
            if (directory / "submit-requested.json").exists():
                raise TransportFailure("previously submitted job is missing; refusing resubmission")
            request = {
                "gds_sha256": file_digest(gds),
                "command": config.emx_command,
                "outputs": config.outputs,
                "input_files": config.remote_input_files,
                "timeout_s": max(0.1, deadline - time.monotonic()),
                "max_output_bytes": config.max_output_bytes,
            }
            request_path = directory / "remote-request.json"
            # The receipt stays byte-identical across retries/resumes.
            if not request_path.exists():
                atomic_json(request_path, request)
            transport.copy(gds, f"{config.remote_root}/{job_id}.gds", upload=True)
            transport.copy(request_path, f"{config.remote_root}/{job_id}.request.json", upload=True)
            submitted = True
            atomic_json(directory / "submit-requested.json", {"job_id": job_id})
            journal.emit("submit_requested", job_id=job_id)
            state = query("submit", file_digest(request_path))
        else:
            submitted = True
            journal.emit("remote_job_reused", job_id=job_id, state=state["state"])
        while state["state"] == "running":
            journal.emit("remote_poll", job_id=job_id, state="running")
            if cancel.wait(0.5) or time.monotonic() >= deadline:
                raise TransportFailure("cancelled" if cancel.is_set() else "deadline exhausted")
            state = query("status")
        if state["state"] != "finished":
            raise TransportFailure("remote state unknown; do not resubmit")
        result = state["result"]
        artifacts = result["artifacts"]
        downloads = directory / "artifacts"
        downloads.mkdir(exist_ok=True)
        for name, metadata in artifacts.items():
            if (
                not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name)
                or metadata["bytes"] > config.max_output_bytes
            ):
                raise TransportFailure("invalid remote artifact metadata")
            destination = downloads / name
            if destination.exists() and file_digest(destination) == metadata["sha256"]:
                journal.emit("download_reused", artifact=name)
                continue
            partial = downloads / (name + ".partial")
            transport.copy(partial, f"{remote_job}/{name}", upload=False)
            if (
                partial.stat().st_size != metadata["bytes"]
                or file_digest(partial) != metadata["sha256"]
            ):
                raise TransportFailure("download integrity mismatch")
            partial.replace(destination)
            journal.emit("artifact_downloaded", artifact=name, **metadata)
        return _finish(
            directory,
            journal,
            {
                **result,
                "job_id": job_id,
                "remote_directory": remote_job,
                "artifacts_directory": "artifacts",
            },
        )
    except (TransportFailure, json.JSONDecodeError) as error:
        cleanup = "not_requested"
        if submitted and (cancel.is_set() or time.monotonic() >= deadline):
            transport.cancel = threading.Event()
            transport.deadline = time.monotonic() + min(5, config.command_timeout_s)
            try:
                cleanup = query("cancel")["state"]
            except (TransportFailure, ValueError):
                cleanup = "unknown"
        return _finish(
            directory,
            journal,
            {
                "execution": "cancelled" if cancel.is_set() else "unknown",
                "verdict": "not_evaluated",
                "job_id": job_id,
                "error": str(error),
                "remote_cancellation": cleanup,
                "resume_required": True,
            },
        )


def _finish(directory, journal, result):
    result = {"schema_version": 1, "certified": False, **result}
    atomic_json(directory / "result.json", result)
    journal.emit("run_finished", execution=result["execution"], verdict=result["verdict"])
    return result
