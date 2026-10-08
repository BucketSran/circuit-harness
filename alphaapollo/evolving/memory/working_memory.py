"""Thread-safe process-local semantic memory.

The default embedder is deterministic and dependency-light.  Semantic quality
runs may inject :class:`SentenceTransformerEmbedder`; both paths share the same
normalization, scope filtering, ordering, and deduplication contract.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .interface import MemoryEntry, MemoryScope, canonical_entry_id

_TOKEN = re.compile(r"[A-Za-z0-9_.:+-]+")
FloatVector = NDArray[np.float64]


class WorkingMemoryError(ValueError):
    """Raised before an invalid embedding can corrupt the derived index."""


@runtime_checkable
class SentenceEmbedder(Protocol):
    """Injectable batch embedder used by semantic retrieval and deduplication."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]: ...


class SentenceTransformerEmbedder:
    """Lazy optional adapter around ``sentence-transformers``."""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise ImportError(
                "SentenceTransformerEmbedder requires the memory extra: "
                "python -m pip install -e '.[memory]'"
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]:
        return np.asarray(
            self._model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        )


class HashingSentenceEmbedder:
    """Deterministic local embeddings suitable for bounded reasoning records."""

    def __init__(self, dimensions: int = 384) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 32:
            raise ValueError("dimensions must be an integer of at least 32")
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float64)
        for row, text in enumerate(texts):
            tokens = [token.casefold() for token in _TOKEN.findall(str(text))]
            features = (
                *tokens,
                *(f"{left}::{right}" for left, right in zip(tokens, tokens[1:], strict=False)),
            )
            if not features:
                features = (str(text),)
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                matrix[row, bucket] += 1.0 if digest[4] & 1 else -1.0
        return matrix


class WorkingMemory:
    """In-process memory that enforces scope before similarity ranking."""

    def __init__(
        self,
        embedder: SentenceEmbedder | None = None,
        *,
        dedup_threshold: float = 0.95,
    ) -> None:
        self._embedder = embedder or HashingSentenceEmbedder()
        if not isinstance(self._embedder, SentenceEmbedder):
            raise TypeError("embedder must implement SentenceEmbedder.encode")
        if (
            isinstance(dedup_threshold, bool)
            or not isinstance(dedup_threshold, (int, float))
            or not math.isfinite(float(dedup_threshold))
            or not 0.0 <= float(dedup_threshold) <= 1.0
        ):
            raise WorkingMemoryError("dedup_threshold must be finite and within [0, 1]")
        self._dedup_threshold = float(dedup_threshold)
        self._entries: dict[str, MemoryEntry] = {}
        self._vectors: dict[str, FloatVector] = {}
        self._dimension: int | None = None
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def write(self, entry: MemoryEntry) -> str:
        if not isinstance(entry, MemoryEntry):
            entry = MemoryEntry.model_validate(entry, strict=True)
        entry_id = canonical_entry_id(entry)
        with self._lock:
            if entry_id in self._entries:
                return entry_id
        vector = self._encode_one(entry.content)
        with self._lock:
            if entry_id in self._entries:
                return entry_id
            self._validate_dimension(vector)
            self._entries[entry_id] = entry
            self._vectors[entry_id] = vector
            return entry_id

    def query(self, query: str, *, reader: MemoryScope, top_k: int = 5) -> tuple[MemoryEntry, ...]:
        if not query.strip():
            raise ValueError("memory query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        with self._lock:
            visible = [
                (entry_id, entry)
                for entry_id, entry in self._entries.items()
                if entry.visible_to(reader)
            ]
            expected_dimension = self._dimension
        if not visible:
            return ()
        query_vector = self._encode_one(query)
        if query_vector.size != expected_dimension:
            raise WorkingMemoryError(
                f"embedder dimension changed: expected {expected_dimension}, "
                f"got {query_vector.size}"
            )
        with self._lock:
            visible = [
                (entry_id, entry)
                for entry_id, entry in self._entries.items()
                if entry.visible_to(reader)
            ]
            scored = [
                (float(query_vector @ self._vectors[entry_id]), entry_id, entry)
                for entry_id, entry in visible
            ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in scored[:top_k])

    def dedup(self, entry: MemoryEntry, *, threshold: float | None = None) -> bool:
        effective_threshold = self._dedup_threshold if threshold is None else threshold
        if not 0.0 <= effective_threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")
        with self._lock:
            if canonical_entry_id(entry) in self._entries:
                return True
            visible_ids = [
                entry_id
                for entry_id, stored in self._entries.items()
                if stored.visible_to(entry.scope)
            ]
            expected_dimension = self._dimension
        if not visible_ids:
            return False
        vector = self._encode_one(entry.content)
        if vector.size != expected_dimension:
            raise WorkingMemoryError(
                f"embedder dimension changed: expected {expected_dimension}, got {vector.size}"
            )
        with self._lock:
            return any(
                float(vector @ self._vectors[entry_id]) >= effective_threshold
                for entry_id in visible_ids
                if entry_id in self._vectors
            )

    def snapshot(self, *, reader: MemoryScope) -> tuple[MemoryEntry, ...]:
        with self._lock:
            return tuple(entry for entry in self._entries.values() if entry.visible_to(reader))

    def _encode_one(self, text: str) -> FloatVector:
        try:
            raw = np.asarray(self._embedder.encode([text]), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - normalize optional provider failures
            raise WorkingMemoryError(f"embedding failed: {exc}") from exc
        if raw.ndim != 2 or raw.shape[0] != 1 or raw.shape[1] == 0:
            raise WorkingMemoryError(
                f"embedder must return shape (n_texts, dimension); got {raw.shape}"
            )
        vector = raw[0]
        if not np.all(np.isfinite(vector)):
            raise WorkingMemoryError("embedding contains NaN or infinity")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise WorkingMemoryError("embedding must have non-zero norm")
        return vector / norm

    def _validate_dimension(self, vector: FloatVector) -> None:
        if self._dimension is None:
            self._dimension = int(vector.size)
        elif vector.size != self._dimension:
            raise WorkingMemoryError(
                f"embedder dimension changed: expected {self._dimension}, got {vector.size}"
            )
