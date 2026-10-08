"""Deterministic Mem0 persistence for immutable reasoning entries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .interface import (
    MemoryEntry,
    MemoryLifetime,
    MemoryScope,
    canonical_entry_id,
)

_PERSISTENT_RUN_ID = "__persistent__"
_SCHEMA_VERSION = 1


class Mem0MemoryError(RuntimeError):
    """Raised when Mem0 cannot prove that an operation completed correctly."""


class Mem0MemoryAdapter:
    """Map AlphaApollo scopes onto Mem0's user/agent/run isolation fields.

    ``infer=False`` is mandatory: AlphaApollo has already decided what the
    verified memory is, so a second LLM extraction would be lossy and
    non-replayable.
    """

    def __init__(self, client: Any) -> None:
        if not callable(getattr(client, "add", None)) or not callable(
            getattr(client, "search", None)
        ):
            raise TypeError("Mem0 client must provide add and search")
        self._client = client

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Mem0MemoryAdapter:
        try:
            from mem0 import Memory
        except ImportError as exc:  # pragma: no cover - exercised without optional extra
            raise Mem0MemoryError(
                "Mem0 is unavailable; install AlphaApollo with the memory extra"
            ) from exc
        return cls(Memory.from_config(dict(config)))

    def write(self, entry: MemoryEntry) -> str:
        entry_id = canonical_entry_id(entry)
        result = self._client.add(
            entry.model_dump_json(),
            user_id=entry.scope.namespace_id,
            agent_id=entry.scope.task_id,
            run_id=_run_filter(entry),
            metadata={
                "alphaapollo_schema_version": _SCHEMA_VERSION,
                "alphaapollo_entry_id": entry_id,
                "alphaapollo_lifetime": entry.lifetime.value,
            },
            infer=False,
        )
        if not _write_was_acknowledged(result):
            raise Mem0MemoryError("Mem0 add did not acknowledge an ADD or UPDATE")
        return entry_id

    def query(self, query: str, *, reader: MemoryScope, top_k: int = 5) -> tuple[MemoryEntry, ...]:
        if not query.strip():
            raise ValueError("memory query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        records: list[tuple[float, str, MemoryEntry]] = []
        for run_id in (reader.run_id, _PERSISTENT_RUN_ID):
            result = self._client.search(
                query,
                filters={
                    "user_id": reader.namespace_id,
                    "agent_id": reader.task_id,
                    "run_id": run_id,
                },
                top_k=top_k,
            )
            for record in _search_records(result):
                entry = _decode_entry(record)
                if entry is None or not entry.visible_to(reader):
                    continue
                records.append((float(record.get("score", 0.0)), canonical_entry_id(entry), entry))
        best: dict[str, tuple[float, MemoryEntry]] = {}
        for score, entry_id, entry in records:
            if entry_id not in best or score > best[entry_id][0]:
                best[entry_id] = (score, entry)
        ranked = sorted(best.items(), key=lambda item: (-item[1][0], item[0]))
        return tuple(item[1][1] for item in ranked[:top_k])

    def dedup(self, entry: MemoryEntry, *, threshold: float = 0.95) -> bool:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")
        entry_id = canonical_entry_id(entry)
        return any(
            canonical_entry_id(found) == entry_id or _same_kind_content(found, entry)
            for found in self.query(entry.content, reader=entry.scope, top_k=10)
        )


def _run_filter(entry: MemoryEntry) -> str:
    return entry.scope.run_id if entry.lifetime is MemoryLifetime.RUN else _PERSISTENT_RUN_ID


def _write_was_acknowledged(result: Any) -> bool:
    if isinstance(result, Mapping):
        values = result.get("results", result.get("data", []))
    else:
        values = result
    return any(
        isinstance(item, Mapping) and item.get("event") in {"ADD", "UPDATE"}
        for item in values or []
    )


def _search_records(result: Any) -> list[Mapping[str, Any]]:
    if isinstance(result, Mapping):
        result = result.get("results", result.get("data", []))
    return [item for item in result or [] if isinstance(item, Mapping)]


def _decode_entry(record: Mapping[str, Any]) -> MemoryEntry | None:
    raw = record.get("memory")
    if not isinstance(raw, str):
        return None
    try:
        entry = MemoryEntry.model_validate_json(raw)
    except (ValueError, TypeError):
        return None
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    if metadata.get("alphaapollo_schema_version") != _SCHEMA_VERSION:
        return None
    if metadata.get("alphaapollo_entry_id") != canonical_entry_id(entry):
        return None
    if metadata.get("alphaapollo_lifetime") != entry.lifetime.value:
        return None
    return entry


def _same_kind_content(left: MemoryEntry, right: MemoryEntry) -> bool:
    return (
        left.kind is right.kind and left.content == right.content and left.visible_to(right.scope)
    )
