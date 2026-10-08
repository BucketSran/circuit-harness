"""Scoped reasoning memory with optional persistent backends."""

from .backends import DEFAULT_MEMORY_BACKENDS, MemoryBackendRegistry
from .interface import (
    MemoryEntry,
    MemoryKind,
    MemoryLifetime,
    MemoryMode,
    MemoryProvenance,
    MemoryScope,
    SharedMemory,
    canonical_entry_id,
)
from .persistent import Mem0MemoryAdapter, Mem0MemoryError
from .tiered import (
    MemoryPolicyError,
    NullProceduralMemory,
    ProceduralMemory,
    Tier,
    TieredMemory,
)
from .working_memory import (
    HashingSentenceEmbedder,
    SentenceEmbedder,
    SentenceTransformerEmbedder,
    WorkingMemory,
    WorkingMemoryError,
)

__all__ = [
    "HashingSentenceEmbedder",
    "DEFAULT_MEMORY_BACKENDS",
    "Mem0MemoryAdapter",
    "Mem0MemoryError",
    "MemoryEntry",
    "MemoryKind",
    "MemoryLifetime",
    "MemoryMode",
    "MemoryPolicyError",
    "MemoryBackendRegistry",
    "MemoryProvenance",
    "MemoryScope",
    "NullProceduralMemory",
    "ProceduralMemory",
    "SentenceEmbedder",
    "SentenceTransformerEmbedder",
    "SharedMemory",
    "Tier",
    "TieredMemory",
    "WorkingMemory",
    "WorkingMemoryError",
    "canonical_entry_id",
]
