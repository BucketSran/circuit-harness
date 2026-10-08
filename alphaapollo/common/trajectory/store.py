"""Append-only JSONL trajectory log with ArtifactStore-backed blob offload."""

from __future__ import annotations

import copy
import fcntl
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from alphaapollo.common.artifacts.schemas import ArtifactRef, ProvenanceRef
from alphaapollo.common.execution.workspace import ArtifactStore
from alphaapollo.common.trajectory.episode import TrajectoryQuery
from alphaapollo.common.trajectory.schemas import (
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryEvent,
    TrajectoryRef,
    require_supported_trajectory_event_schema,
)

logger = logging.getLogger(__name__)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TrajectoryStoreError(Exception):
    pass


class TrajectoryStore:
    """Single-writer append log; file order is authoritative replay order."""

    def __init__(self, *, jsonl_path: Path, artifacts: ArtifactStore) -> None:
        self._jsonl_path = Path(jsonl_path)
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifacts = artifacts
        self._closed = False
        self._writer_pid = os.getpid()
        self._mutex = threading.RLock()
        self._lock_path = self._jsonl_path.with_suffix(self._jsonl_path.suffix + ".lock")
        self._lock_handle = self._lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_handle.seek(0)
            owner = self._lock_handle.read().strip() or "unknown"
            self._lock_handle.close()
            raise TrajectoryStoreError(
                f"trajectory store already has a writer (pid {owner})"
            ) from exc
        self._lock_handle.seek(0)
        self._lock_handle.truncate()
        self._lock_handle.write(str(os.getpid()))
        self._lock_handle.flush()
        self._cache_key: tuple[int, int] | None = None
        self._cache_events: tuple[TrajectoryEvent, ...] = ()
        try:
            if self._jsonl_path.exists():
                self._jsonl_path.chmod(0o600)
            existing = self._read_unfiltered(repair_trailing=True)
            self._resync(existing)
        except Exception:
            self.close()
            raise

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    def _resync(self, events: list[TrajectoryEvent]) -> None:
        event_ids = {event.event_id for event in events}
        if len(event_ids) != len(events):
            raise TrajectoryStoreError("trajectory contains duplicate event_id")
        for expected, event in enumerate(events):
            if event.sequence is not None and event.sequence != expected:
                raise TrajectoryStoreError(
                    "trajectory sequence does not match append order: "
                    f"event {event.event_id!r} has {event.sequence}, expected {expected}"
                )
        self._event_ids = event_ids
        self._next_sequence = len(events)

    def append(self, event: TrajectoryEvent) -> TrajectoryEvent:
        with self._mutex:
            self._ensure_owner_process()
            return self._append_unlocked(event)

    def _append_unlocked(self, event: TrajectoryEvent) -> TrajectoryEvent:
        if self._closed:
            raise TrajectoryStoreError("trajectory store is closed")
        if not isinstance(event, TrajectoryEvent):
            raise TrajectoryStoreError(
                f"append expects TrajectoryEvent, got {type(event).__name__}"
            )
        if event.event_id in self._event_ids:
            raise TrajectoryStoreError(f"duplicate event_id {event.event_id!r}")
        version = event.trajectory_schema_version
        if version not in (None, TRAJECTORY_SCHEMA_VERSION):
            raise TrajectoryStoreError(
                "event trajectory_schema_version must be "
                f"{TRAJECTORY_SCHEMA_VERSION}, got {version}"
            )
        updates = (
            {"trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION} if version is None else {}
        )
        if event.sequence is None:
            stored = event.model_copy(update={"sequence": self._next_sequence, **updates})
        elif event.sequence != self._next_sequence:
            raise TrajectoryStoreError(
                f"event sequence must be {self._next_sequence}, got {event.sequence}"
            )
        else:
            stored = event.model_copy(update=updates) if updates else event

        self._validate_schema_version(stored)
        line = stored.model_dump_json() + "\n"
        file_existed = self._jsonl_path.exists()
        try:
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                if not file_existed:
                    os.fchmod(handle.fileno(), 0o600)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            if not file_existed:
                _fsync_directory(self._jsonl_path.parent)
        except OSError as exc:
            self._invalidate_cache()
            try:
                self._resync(self._read_unfiltered(repair_trailing=True))
            except Exception as recovery_error:
                raise TrajectoryStoreError(
                    "append failed and trajectory recovery also failed: "
                    f"append={exc}; recovery={recovery_error}"
                ) from exc
            raise TrajectoryStoreError(
                "append durability was not confirmed; the store was resynchronized "
                f"from disk: {exc}"
            ) from exc

        self._event_ids.add(stored.event_id)
        self._next_sequence += 1
        self._invalidate_cache()
        return stored

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            self._closed = True
            try:
                if os.getpid() == self._writer_pid:
                    fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_handle.close()

    def __enter__(self) -> TrajectoryStore:
        return self

    def __exit__(self, *args: Any) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _invalidate_cache(self) -> None:
        self._cache_key = None
        self._cache_events = ()

    def _ensure_owner_process(self) -> None:
        if os.getpid() != self._writer_pid:
            raise TrajectoryStoreError(
                "trajectory writer cannot be used after fork; create a new store in the child"
            )

    def _file_key(self) -> tuple[int, int] | None:
        if not self._jsonl_path.exists():
            return None
        stat = self._jsonl_path.stat()
        return stat.st_mtime_ns, stat.st_size

    @staticmethod
    def _validate_schema_version(event: TrajectoryEvent) -> None:
        try:
            require_supported_trajectory_event_schema(event.schema_version)
        except ValueError as exc:
            raise TrajectoryStoreError(
                f"unsupported trajectory schema_version {event.schema_version!r}: {exc}"
            ) from exc

    def _read_unfiltered(self, *, repair_trailing: bool = False) -> list[TrajectoryEvent]:
        key = self._file_key()
        if key is None:
            return []
        if not repair_trailing and key == self._cache_key:
            return [event.model_copy(deep=True) for event in self._cache_events]

        content = self._jsonl_path.read_bytes()
        lines = content.splitlines(keepends=True)
        events: list[TrajectoryEvent] = []
        valid_bytes = 0
        for index, raw in enumerate(lines):
            line_number = index + 1
            is_last = index == len(lines) - 1
            if not raw.strip():
                valid_bytes += len(raw)
                continue
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as exc:
                incomplete_tail = is_last and not raw.endswith(b"\n")
                if repair_trailing and incomplete_tail:
                    logger.warning("dropping incomplete trailing trajectory line %s", line_number)
                    with self._jsonl_path.open("r+b") as handle:
                        handle.truncate(valid_bytes)
                        handle.flush()
                        os.fsync(handle.fileno())
                    self._invalidate_cache()
                    break
                raise TrajectoryStoreError(
                    f"malformed trajectory JSONL at line {line_number}: {exc}"
                ) from exc
            try:
                event = TrajectoryEvent.model_validate(decoded)
                self._validate_schema_version(event)
            except Exception as exc:
                raise TrajectoryStoreError(
                    f"malformed trajectory JSONL at line {line_number}: {exc}"
                ) from exc
            events.append(event)
            valid_bytes += len(raw)

        complete_unterminated_tail = (
            repair_trailing
            and content
            and not content.endswith(b"\n")
            and valid_bytes == len(content)
        )
        if complete_unterminated_tail:
            with self._jsonl_path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._invalidate_cache()

        repaired_key = self._file_key()
        self._cache_key = repaired_key
        self._cache_events = tuple(events)
        return [event.model_copy(deep=True) for event in events]

    def read_events(self, query: TrajectoryQuery | None = None) -> list[TrajectoryEvent]:
        with self._mutex:
            self._ensure_owner_process()
            events = self._read_unfiltered()
        if query is None:
            return events
        allowed_types = set(query.event_types)
        selected: list[TrajectoryEvent] = []
        for event in events:
            if query.session_id is not None and event.session_id != query.session_id:
                continue
            if query.branch_id is not None and event.branch_id != query.branch_id:
                continue
            if allowed_types and event.type not in allowed_types:
                continue
            flags = event.payload.get("contamination_flags", [])
            artifact_is_contaminated = any(ref.contamination_flags for ref in event.artifact_refs)
            if query.exclude_contaminated and (flags or artifact_is_contaminated):
                continue
            if (
                query.training_eligible_only
                and event.payload.get("training_eligibility", False) is not True
            ):
                continue
            selected.append(event)
        return selected

    def export(self, query: TrajectoryQuery | None = None) -> list[TrajectoryEvent]:
        """Return neutral semantic records for an owner adapter to transform."""

        return self.read_events(query)

    def export_json(self, query: TrajectoryQuery | None = None) -> list[dict[str, Any]]:
        return [event.model_dump(mode="json") for event in self.export(query)]

    def put_blob(
        self,
        content: bytes,
        *,
        type_: str,
        created_by: str,
        provenance: ProvenanceRef | None = None,
        contamination_flags: list[str] | None = None,
    ) -> ArtifactRef:
        artifact = self._artifacts.put(
            content,
            type_=type_,
            created_by=created_by,
            provenance=provenance,
            contamination_flags=contamination_flags,
        )
        return self._artifacts.ref(artifact)

    def get_blob(self, ref: ArtifactRef) -> bytes:
        return self._artifacts.get(ref)

    def trajectory_ref(self) -> TrajectoryRef:
        return TrajectoryRef(
            id=self._jsonl_path.stem,
            location=self._jsonl_path.name,
            version="1.0",
        )

    @staticmethod
    def stamp_payload(
        payload: dict[str, Any],
        *,
        source_flags: list[str],
        redact_paths: tuple[tuple[str, ...], ...] = (),
    ) -> dict[str, Any]:
        """Copy provenance labels and perform explicit, auditable redactions.

        Common records contamination facts but does not decide training
        eligibility. Every removed field must be named by the emitter as a key
        path, for example ``(("evaluation", "ground_truth"),)``.
        """

        stamped = copy.deepcopy(payload)
        stamped["contamination_flags"] = sorted(set(source_flags))
        redacted: list[str] = []
        for path in redact_paths:
            if not path or any(not part for part in path):
                raise ValueError("redact paths must contain non-empty key names")
            target: Any = stamped
            for part in path[:-1]:
                if not isinstance(target, dict) or part not in target:
                    target = None
                    break
                target = target[part]
            if isinstance(target, dict) and path[-1] in target:
                del target[path[-1]]
                redacted.append(".".join(path))
        if redacted:
            stamped["_redactions"] = sorted(redacted)
        return stamped


__all__ = ["TrajectoryStore", "TrajectoryStoreError"]
