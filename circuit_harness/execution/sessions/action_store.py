"""Internal durable action records shared by public task sessions.

Callers retain admission policy and decide when execution can leave the lock.
A request without a response is uncertain and must never be replayed by this store.
"""

from __future__ import annotations

import fcntl
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from circuit_harness.execution.runtime.journal import atomic_json


@dataclass(frozen=True)
class ActionRecord:
    path: Path

    def cached_response(self) -> dict:
        response = self.path / "response.json"
        if response.exists():
            return json.loads(response.read_text())
        return {"ok": False, "error": "unknown_execution", "retry_safe": False}

    def complete(self, response: dict, *, started: float | None = None) -> dict:
        if started is not None:
            atomic_json(self.path / "measurement.json", {"elapsed_s": time.monotonic() - started})
        atomic_json(self.path / "response.json", response)
        return response


class ActionStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.actions = directory / "actions"

    @contextmanager
    def lock(self):
        # Closing the descriptor releases the lock, including on exceptions.
        with (self.directory / ".session.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield lock

    def lookup(self, action_id: str, request: dict) -> ActionRecord | None:
        action = self.actions / action_id
        if not action.exists():
            return None
        if json.loads((action / "request.json").read_text()) != request:
            raise ValueError("action ID belongs to another request")
        return ActionRecord(action)

    def begin(self, action_id: str, request: dict) -> ActionRecord:
        action = self.actions / action_id
        action.mkdir(mode=0o700)
        atomic_json(action / "request.json", request)
        return ActionRecord(action)
