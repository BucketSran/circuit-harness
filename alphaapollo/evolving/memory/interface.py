"""Backend-independent contracts for reasoning memory.

Memory entries are model context, not an operation log. Callers must store
compact, source-attributed conclusions and keep raw transcripts and telemetry
elsewhere.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator

_FROZEN = ConfigDict(extra="forbid", frozen=True, strict=True)


class MemoryKind(str, Enum):
    CLAIM = "claim"
    RESULT = "result"
    FAILURE = "failure"


class MemoryLifetime(str, Enum):
    RUN = "run"
    PERSISTENT = "persistent"


class MemoryMode(str, Enum):
    """Runtime topology; independent from the kind of content being stored."""

    OFF = "off"
    WORKING = "working"
    PERSISTENT = "persistent"


class MemoryScope(BaseModel):
    model_config = _FROZEN

    namespace_id: str
    task_id: str
    run_id: str

    @field_validator("namespace_id", "task_id", "run_id")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("memory scope identifiers must be non-empty")
        if any(character.isspace() for character in value):
            raise ValueError("memory scope identifiers must not contain whitespace")
        return value

    def allows(self, reader: MemoryScope, lifetime: MemoryLifetime) -> bool:
        if self.namespace_id != reader.namespace_id or self.task_id != reader.task_id:
            return False
        return lifetime is MemoryLifetime.PERSISTENT or self.run_id == reader.run_id


class MemoryProvenance(BaseModel):
    model_config = _FROZEN

    actor_id: str
    source_event_id: str
    artifact_ids: tuple[str, ...] = ()

    @field_validator("actor_id", "source_event_id")
    @classmethod
    def _required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("memory provenance identifiers must be non-empty")
        return value


class MemoryEntry(BaseModel):
    model_config = _FROZEN

    kind: MemoryKind
    content: str
    scope: MemoryScope
    provenance: MemoryProvenance
    lifetime: MemoryLifetime = MemoryLifetime.RUN

    @field_validator("content")
    @classmethod
    def _content_is_useful(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("memory content must be non-empty")
        return value

    def visible_to(self, reader: MemoryScope) -> bool:
        return self.scope.allows(reader, self.lifetime)


def canonical_entry_id(entry: MemoryEntry) -> str:
    """Return a stable content address for an immutable memory entry."""

    return hashlib.sha256(entry.model_dump_json().encode("utf-8")).hexdigest()


@runtime_checkable
class SharedMemory(Protocol):
    def write(self, entry: MemoryEntry) -> str: ...

    def query(
        self, query: str, *, reader: MemoryScope, top_k: int = 5
    ) -> tuple[MemoryEntry, ...]: ...

    def dedup(self, entry: MemoryEntry, *, threshold: float = 0.95) -> bool: ...
