"""Exact frozen deliverables; syntax and correctness belong to the benchmark."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

from circuit_harness.execution.runtime.journal import atomic_json, file_digest

MAX_CANDIDATE_BYTES = 16 * 1024 * 1024
MAX_FILES = 1000


def declared_files(names: list[str]) -> list[str]:
    """Validate portable, nonoverlapping relative regular-file names."""
    if not isinstance(names, list) or not 1 <= len(names) <= MAX_FILES:
        raise ValueError("declare 1–1000 files")
    for name in names:
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise ValueError("invalid declared file")
        path = PurePosixPath(name)
        if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
            raise ValueError("file must be a canonical relative path")
    ordered = sorted(names)
    if len(set(ordered)) != len(ordered):
        raise ValueError("duplicate declared file")
    for index, name in enumerate(ordered):
        if any(other.startswith(name + "/") for other in ordered[index + 1 :]):
            raise ValueError("overlapping declared files")
    return ordered


def regular_file(root: Path, name: str) -> Path:
    """Reject symlinks at every level, including the declared root."""
    declared_files([name])
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("file root must be a regular directory")
    path = root
    for part in PurePosixPath(name).parts:
        path /= part
        if path.is_symlink():
            raise ValueError("symlinks are not candidate files")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"missing declared file: {name}")
    return path


def _inventory(root: Path, names: list[str]) -> dict:
    records = {}
    total = 0
    for name in declared_files(names):
        path = regular_file(root, name)
        size = path.stat().st_size
        total += size
        if total > MAX_CANDIDATE_BYTES:
            raise ValueError("candidate exceeds 16 MiB")
        records[name] = {"sha256": file_digest(path), "bytes": size}
    return records


def _identity(task_id: str, task_version: str, records: dict) -> str:
    value = {"task_id": task_id, "task_version": task_version, "files": records}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze_candidate(
    source: Path,
    destination: Path,
    files: list[str],
    *,
    task_id: str,
    task_version: str,
    reason: str,
) -> dict:
    """Freeze declared bytes into a new bundle, never selecting a historical version."""
    if any(not isinstance(value, str) or not value for value in (task_id, task_version, reason)):
        raise ValueError("task identity and reason must be nonempty strings")
    records = _inventory(Path(source), files)
    destination = Path(destination).absolute()
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    root = destination / "files"
    root.mkdir(mode=0o700)
    for name in records:
        target = root / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(regular_file(Path(source), name), target)
    if _inventory(root, files) != records:
        raise ValueError("candidate changed during freeze")
    result = {
        "schema_version": 1,
        "task_id": task_id,
        "task_version": task_version,
        "reason": reason,
        "files": records,
        "candidate_sha256": _identity(task_id, task_version, records),
    }
    atomic_json(destination / "manifest.json", result)
    return result


def verify_candidate(directory: Path) -> dict:
    """Rehash all declared bytes and reject undeclared inventory or altered identity."""
    directory = Path(directory).absolute()
    manifest = regular_file(directory, "manifest.json")
    try:
        record = json.loads(manifest.read_text())
        if (
            set(record)
            != {"schema_version", "task_id", "task_version", "reason", "files", "candidate_sha256"}
            or record["schema_version"] != 1
        ):
            raise ValueError("unsupported candidate manifest")
        if any(
            not isinstance(record[key], str) or not record[key]
            for key in ("task_id", "task_version", "reason")
        ):
            raise ValueError("invalid candidate identity")
        root = directory / "files"
        actual = _inventory(root, list(record["files"]))
        names = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("symlink in frozen candidate")
            if not path.is_dir():
                names.add(path.relative_to(root).as_posix())
        if (
            names != set(actual)
            or actual != record["files"]
            or _identity(record["task_id"], record["task_version"], actual)
            != record["candidate_sha256"]
        ):
            raise ValueError("frozen candidate changed")
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("malformed candidate manifest") from error
    return record
