"""Operator-only SSH transport for frozen benchmark jobs; no simulator controller.

The existing session transport owns SSH waits and artifact downloads; jobs owns
execution, deduplication and archival. Disconnects never create a new job ID.
"""

from __future__ import annotations

import base64
import fcntl
import json
import re
import tempfile
from pathlib import Path

from circuit_harness.execution.backends.vabench import candidate_files
from circuit_harness.execution.evaluation.archive import private_root, verify_archive
from circuit_harness.execution.evaluation.benchmark_spectre import package_identity
from circuit_harness.execution.evaluation.candidate_bundle import verify_candidate
from circuit_harness.execution.runtime.journal import atomic_json, digest
from circuit_harness.execution.transport.session_transport import RemoteSessionTransport


def transfer_payload(candidate: Path, task_package: Path, *, purpose="final") -> dict:
    verify_candidate(candidate)
    package_identity(task_package, purpose=purpose)
    return {
        kind: {
            name: base64.b64encode((root / name).read_bytes()).decode("ascii")
            for name in candidate_files(root)
        }
        for kind, root in (("candidate", candidate), ("task-package", task_package))
    }


def stage_transfer(root: Path, payload: dict, *, purpose="final") -> dict:
    """Stage bounded declared bytes under a content address, never unpack a tar."""
    if not isinstance(payload, dict) or set(payload) != {"candidate", "task-package"}:
        raise ValueError("invalid benchmark transfer")
    root = private_root(root)
    key = digest(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    destination = root / key
    with (root / ".stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                directory = Path(temporary)
                total = count = 0
                for kind, inventory in payload.items():
                    if not isinstance(inventory, dict):
                        raise ValueError("invalid transfer inventory")
                    for name, encoded in inventory.items():
                        if (
                            not isinstance(name, str)
                            or not name
                            or "\\" in name
                            or "\x00" in name
                            or Path(name).is_absolute()
                            or any(p in {"", ".", ".."} for p in name.split("/"))
                        ):
                            raise ValueError("unsafe transfer filename")
                        data = base64.b64decode(encoded, validate=True)
                        total += len(data)
                        count += 1
                        if total > 32 * 1024 * 1024 or count > 2000:
                            raise ValueError("benchmark transfer exceeds limit")
                        path = directory / kind / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                verify_candidate(directory / "candidate")
                package_identity(directory / "task-package", purpose=purpose)
                directory.rename(destination)
        verify_candidate(destination / "candidate")
        package_identity(destination / "task-package", purpose=purpose)
        # Content address must match bytes even when reusing an existing stage.
        if (
            transfer_payload(
                destination / "candidate", destination / "task-package", purpose=purpose
            )
            != payload
        ):
            raise ValueError("staged benchmark bytes differ")
    return {
        "candidate": str(destination / "candidate"),
        "task_package": str(destination / "task-package"),
    }


class RemoteBenchmarkSpectre(RemoteSessionTransport):
    """Explicit operator final evaluation; not a public Agent tool or sandbox."""

    purpose = "final"

    def __init__(self, config: dict, evidence: Path):
        for name in ("profile", "run_root", "archive_root", "upload_root"):
            if not isinstance(config.get(name), str) or not Path(config[name]).is_absolute():
                raise ValueError(f"remote {name} must be absolute")
        super().__init__({**config, "session": config["upload_root"]}, evidence)
        private_root(self.evidence)

    def _record(self, job_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job_id):
            raise ValueError("invalid benchmark job ID")
        return self.evidence / job_id

    def submit(self, candidate: Path, task_package: Path, job_id: str) -> dict:
        payload = transfer_payload(candidate, task_package, purpose=self.purpose)
        record = self._record(job_id)
        record.mkdir(mode=0o700, exist_ok=True)
        identity = {
            "transfer_sha256": digest(json.dumps(payload, sort_keys=True).encode()),
            "remote": self.config,
        }
        request = record / "submission.json"
        if request.exists() and json.loads(request.read_text()) != identity:
            raise ValueError("local job ID already belongs to another benchmark request")
        atomic_json(request, identity)
        staged = self.cli(
            "stage-benchmark",
            "--root",
            self.config["upload_root"],
            "--request",
            "-",
            "--purpose",
            self.purpose,
            payload=payload,
        )
        reply = self.cli(
            "submit-benchmark-spectre",
            "--candidate",
            staged["candidate"],
            "--task-package",
            staged["task_package"],
            "--profile",
            self.config["profile"],
            "--job-id",
            job_id,
            "--purpose",
            self.purpose,
            *(("--isolated-public",) if self.purpose == "public" else ()),
        )
        atomic_json(record / "ack.json", reply)
        return reply

    def query(self, job_id: str) -> dict:
        self._record(job_id)
        reply = self.cli("job-status", str(Path(self.config["run_root"]) / job_id))
        if reply.get("state") == "missing":
            reply = self.cli("archive-status", str(Path(self.config["archive_root"]) / job_id))
        return reply

    def retrieve(self, job_id: str) -> dict:
        """Download private receipt and package; independently validate every member."""
        record = self._record(job_id)
        state = self.query(job_id)
        if state.get("state") != "finished":
            raise ValueError("benchmark job is not finished; query this same ID")
        remote = Path(self.config["archive_root"]) / job_id
        self.cli("verify-archive", str(remote))
        archive = record / "archive"
        private_root(archive)
        for name in ("receipt.json", "job.tar.gz"):
            self.download(remote / name, archive / name)
        receipt = verify_archive(archive)
        identity = receipt["request"]["identity"]
        local = json.loads((record / "submission.json").read_text())
        transferred = {
            kind: {name: base64.b64encode(data).decode("ascii") for name, data in files.items()}
            for kind, files in _archive_inputs(archive / "job.tar.gz").items()
        }
        if local["transfer_sha256"] != digest(json.dumps(transferred, sort_keys=True).encode()):
            raise ValueError("retrieved archive differs from submitted frozen bytes")
        request = receipt["request"]
        configuration = identity["profile"]["configuration"]
        if (
            request["job_id"] != job_id
            or identity["profile"]["path"] != self.config["profile"]
            or any(
                configuration[name] != self.config[name] for name in ("run_root", "archive_root")
            )
            or request.get("archive_directory") != str(remote)
        ):
            raise ValueError("retrieved job differs from configured remote identity")
        if self.purpose == "public" and configuration.get("backend") not in {
            "docker",
            "spectre_namespace",
        }:
            raise ValueError("public job did not use an isolated profile")
        if identity.get("purpose") != self.purpose:
            raise ValueError("retrieved job is not independent final evaluation")
        return receipt["completion"]["result"]


def _archive_inputs(package):
    import tarfile

    result = {"candidate": {}, "task-package": {}}
    with tarfile.open(package, "r:gz") as archive:
        for member in archive:
            kind, _, name = member.name.partition("/")
            if kind in result:
                result[kind][name] = archive.extractfile(member).read()
    return result
