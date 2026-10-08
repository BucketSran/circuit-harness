"""Durable execution facts and read-only status projection for chip runs."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

_SOURCE_SHA256 = hashlib.sha256(files(__package__).joinpath("journal.py").read_bytes()).hexdigest()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_events(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """A torn final append is visible; malformed complete records are errors."""
    if not path.exists():
        return [], False
    raw = path.read_bytes()
    complete, _, tail = raw.rpartition(b"\n")
    rows = []
    for line in complete.splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("sequence") != len(rows) + 1:
            raise ValueError("invalid chips event sequence")
        rows.append(row)
    return rows, bool(tail)


class Journal:
    """One writer per locked run directory; threads share this instance."""

    def __init__(self, directory: Path, *, call_id: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "events.jsonl"
        rows, torn = read_events(self.path)
        self._sequence = len(rows)
        self.call_id = call_id
        self._lock = threading.Lock()
        if torn:
            # Keep the original bytes as evidence before repairing append framing.
            raw = self.path.read_bytes()
            backup = directory / f"events-torn-{time.time_ns()}.bin"
            backup.write_bytes(raw)
            self.path.write_bytes(raw[: raw.rfind(b"\n") + 1])
            self.emit("journal_recovered", backup=backup.name)

    @contextmanager
    def stage(self, name: str):
        """Measure a synchronous phase, including a phase ending in an exception."""
        started = time.monotonic()
        completed = False
        try:
            yield
            completed = True
        finally:
            self.emit(
                "stage_finished",
                stage=name,
                elapsed_s=time.monotonic() - started,
                completed=completed,
            )

    def emit(self, event: str, **fields: Any) -> None:
        with self._lock:
            row = {
                "schema_version": 1,
                "sequence": self._sequence + 1,
                "time": time.time(),
                "call_id": self.call_id,
                "event": event,
                **fields,
            }
            encoded = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._sequence += 1


def status(directory: Path) -> dict[str, Any]:
    rows, torn = read_events(directory / "events.jsonl")
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    last = rows[-1] if rows else None
    # A previous attempt's result must not hide an active resumed attempt.
    terminal = last is not None and last["event"] in {"run_finished", "result_reused"}
    state = "finished" if terminal else "incomplete_or_running"
    if (
        terminal
        and result
        and (
            result.get("execution") == "unknown"
            or result.get("remote_cancellation") in {"unknown", "cancellation_requested"}
        )
    ):
        state = "attention_required"
    return {
        "events": len(rows),
        "torn_tail": torn,
        "last_event": last,
        "seconds_since_event": None if last is None else max(0, time.time() - last["time"]),
        "result": result if terminal else None,
        "state": state,
    }
