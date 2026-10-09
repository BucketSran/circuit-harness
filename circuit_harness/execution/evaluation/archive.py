"""Sealed job archives: local packing, verified publication, independent retry.

Only receipt-listed evidence is archived. Live worker/archiver diagnostics stay
outside the sealed receipt. These operator-owned archives can contain hidden
benchmark answers and must never be exposed as an agent's public task input.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

from circuit_harness.execution.runtime.journal import atomic_json, file_digest


def private_root(path: Path) -> Path:
    path = Path(path).resolve()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError("storage root must be owned by this user with mode 0700")
    return path


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path: Path, value: dict) -> None:
    atomic_json(path, value)
    sync_directory(path.parent)


def archive_status(record: Path) -> dict:
    """Read recorded state; verification is an explicit operation, not a status side effect."""
    status = record / "archive-status.json"
    return json.loads(status.read_text()) if status.exists() else {"state": "pending"}


def reserve_archive(record: Path, directory: Path, identity: dict) -> bool:
    """Persist an ID before launch so expired scratch can never trigger a retry run."""
    registration = {"schema_version": 1, "identity": identity, "work_directory": str(directory)}
    with (record.parent / ".register.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if record.exists():
            if (
                record.is_symlink()
                or json.loads((record / "registration.json").read_text()) != registration
            ):
                raise ValueError("job ID already belongs to a different task or work directory")
            return False
        record.mkdir(mode=0o700)
        durable_json(record / "registration.json", registration)
        sync_directory(record.parent)
        return True


def archived_job_status(record: Path) -> dict:
    reply = {"job_id": record.name, "state": "unknown", "archive": archive_status(record)}
    receipt = record / "receipt.json"
    if receipt.exists():
        reply.update(state="finished", **json.loads(receipt.read_text())["completion"])
    return reply


def verify_archive(record: Path) -> dict:
    """Verify the package and all members offline, then use the original job verifier."""
    # All receipts and extracted JSON here are untrusted downloaded structures.
    # Normalize structural failures only, without hiding execution/program errors.
    try:
        receipt = json.loads((record / "receipt.json").read_text())
        if not isinstance(receipt, dict):
            raise ValueError("archive receipt must be an object")
        package = record / "job.tar.gz"
        if (
            package.is_symlink()
            or package.stat().st_size != receipt["package"]["bytes"]
            or file_digest(package) != receipt["package"]["sha256"]
        ):
            raise ValueError("archive package missing or modified")
        _verify_package(package, receipt)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("invalid archive receipt structure") from error
    return receipt


def _verify_package(package: Path, receipt: dict, *, temporary_root: Path | None = None) -> None:
    # Extraction is to a new private temporary directory, not directly into a
    # caller-selected existing path. Validate each member before writing it.
    from circuit_harness.execution.runtime.jobs import verify_job

    expected = receipt["members"]
    with tempfile.TemporaryDirectory(prefix="chips-verify-", dir=temporary_root) as temporary:
        destination = Path(temporary)
        seen = set()
        try:
            with tarfile.open(package, "r:gz") as tar:
                for member in tar:
                    name = member.name
                    if (
                        name not in expected
                        or name in seen
                        or not member.isfile()
                        or Path(name).is_absolute()
                        or ".." in Path(name).parts
                        or str(Path(name)) != name
                        or member.size != expected[name]["bytes"]
                    ):
                        raise ValueError(f"unsafe or unexpected archive member: {name}")
                    seen.add(name)
                    path = destination / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with tar.extractfile(member) as source, path.open("xb") as target:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            target.write(chunk)
                    if digest.hexdigest() != expected[name]["sha256"]:
                        raise ValueError(f"archive member modified: {name}")
        except tarfile.TarError as error:
            raise ValueError(f"invalid archive: {error}") from error
        if seen != expected.keys():
            raise ValueError("archive is missing members")
        completion = verify_job(destination)
        request = json.loads((destination / "request.json").read_text())
        if completion != receipt["completion"] or request != receipt["request"]:
            raise ValueError("archive receipt differs from sealed job")
        if set(expected) != set(completion["artifacts"]) | {"completion.json"}:
            raise ValueError("archive member list differs from job receipt")


def archive_job(directory: Path) -> dict:
    """Called under the job's worker lock; never runs a simulator, even on retry."""
    from circuit_harness.execution.runtime.jobs import verify_job

    request = json.loads((directory / "request.json").read_text())
    record = Path(request["archive_directory"])
    started = time.monotonic()
    timings = {}

    def status(state: str, **extra) -> dict:
        value = {"state": state, "updated_at": time.time(), "timings_s": timings, **extra}
        atomic_json(directory / "archive-status.json", value)
        durable_json(record / "archive-status.json", value)
        return value

    try:
        registration = json.loads((record / "registration.json").read_text())
        if registration["identity"] != request["identity"] or registration["work_directory"] != str(
            directory
        ):
            raise ValueError("archive registration differs from job")
        status("running")
        if (record / "receipt.json").exists():
            receipt = verify_archive(record)
            if receipt["request"] != request:
                raise ValueError("archive belongs to a different job")
            timings.update(receipt["timings_s"])
            return status("verified", package=receipt["package"])
        stage = time.monotonic()
        completion = verify_job(directory)
        timings["verify_source"] = time.monotonic() - stage
        members = {
            **completion["artifacts"],
            "completion.json": {
                "sha256": file_digest(directory / "completion.json"),
                "bytes": (directory / "completion.json").stat().st_size,
            },
        }
        receipt = {
            "schema_version": 1,
            "request": request,
            "completion": completion,
            "members": members,
        }
        stage = time.monotonic()
        packed = directory / "archive.tar.gz.partial"
        with tarfile.open(packed, "w:gz") as tar:
            for name in sorted(members):
                tar.add(directory / name, arcname=name, recursive=False)
        timings["pack"] = time.monotonic() - stage
        stage = time.monotonic()
        partial = record / "job.tar.gz.partial"
        with packed.open("rb") as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        timings["copy_fsync"] = time.monotonic() - stage
        stage = time.monotonic()
        package = {"sha256": file_digest(packed), "bytes": packed.stat().st_size}
        if partial.stat().st_size != package["bytes"] or file_digest(partial) != package["sha256"]:
            raise ValueError("copied archive differs from local package")
        _verify_package(partial, receipt, temporary_root=directory)
        timings["verify_copy"] = time.monotonic() - stage
        stage = time.monotonic()
        os.replace(partial, record / "job.tar.gz")
        sync_directory(record)
        timings["publish_package"] = time.monotonic() - stage
        timings["attempt_elapsed"] = time.monotonic() - started
        receipt.update(package=package, timings_s=dict(timings), archived_at=time.time())
        durable_json(record / "receipt.json", receipt)
        packed.unlink()
        return status("verified", package=package)
    except Exception as error:
        # Preserve terminal simulation state and local evidence if NFS goes away.
        timings["attempt_elapsed"] = time.monotonic() - started
        value = {
            "state": "failed",
            "error": f"{type(error).__name__}: {error}",
            "updated_at": time.time(),
            "timings_s": timings,
        }
        atomic_json(directory / "archive-status.json", value)
        try:
            durable_json(record / "archive-status.json", value)
        except OSError:
            pass
        return value
