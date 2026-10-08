"""Compose semantic, lexical, procedural, and durable memory channels."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from .grep_stm import MemorySource, grep
from .interface import MemoryEntry, MemoryScope, SharedMemory, canonical_entry_id
from .working_memory import WorkingMemory

_WORD = re.compile(r"\w+", re.UNICODE)
_RRF_K = 60
_LEXICAL_PATTERN_CAP = 512


class Tier(str, Enum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    PROCEDURAL = "procedural"


@runtime_checkable
class ProceduralMemory(Protocol):
    """Seam for the skill registry; procedures are not stored as result entries."""

    def query(self, query: str, *, reader: MemoryScope, top_k: int) -> tuple[MemoryEntry, ...]: ...


class NullProceduralMemory:
    def query(self, query: str, *, reader: MemoryScope, top_k: int) -> tuple[MemoryEntry, ...]:
        del query, reader, top_k
        return ()


class MemoryPolicyError(ValueError):
    """Raised when the composition-owned contamination policy rejects an entry."""


class TieredMemory:
    """One ``SharedMemory`` over fast STM and an optional durable provider.

    Writes always enter semantic working memory and, when configured, the
    persistent provider.  Lexical recall scans the same already scope-filtered
    working snapshot.  The procedural channel is an explicit read-only seam for
    the skills subsystem instead of a second procedure store.
    """

    def __init__(
        self,
        *,
        working: WorkingMemory | None = None,
        persistent: SharedMemory | None = None,
        procedural: ProceduralMemory | None = None,
        default_tiers: Sequence[Tier | str] = (Tier.SEMANTIC,),
        entry_gate: Callable[[MemoryEntry], bool] | None = None,
    ) -> None:
        self._working = working or WorkingMemory()
        self._persistent = persistent
        self._procedural = procedural or NullProceduralMemory()
        self._default_tiers = _coerce_tiers(default_tiers)
        if entry_gate is not None and not callable(entry_gate):
            raise TypeError("entry_gate must be callable when provided")
        self._entry_gate = entry_gate or (lambda _entry: True)

    def write(self, entry: MemoryEntry) -> str:
        if not self._entry_gate(entry):
            raise MemoryPolicyError("memory entry rejected by contamination policy")
        entry_id = self._working.write(entry)
        if self._persistent is not None:
            persisted_id = self._persistent.write(entry)
            if persisted_id != entry_id:
                raise RuntimeError("persistent memory returned a non-canonical entry id")
        return entry_id

    def warm(self, entry: MemoryEntry) -> str:
        """Hydrate the process cache without repeating a durable write."""

        if not self._entry_gate(entry):
            raise MemoryPolicyError("memory entry rejected by contamination policy")
        return self._working.write(entry)

    def snapshot(self, *, reader: MemoryScope) -> tuple[MemoryEntry, ...]:
        return tuple(
            entry for entry in self._working.snapshot(reader=reader) if self._entry_gate(entry)
        )

    def query(
        self,
        query: str,
        *,
        reader: MemoryScope,
        top_k: int = 5,
        tiers: Sequence[Tier | str] | None = None,
    ) -> tuple[MemoryEntry, ...]:
        if not query.strip():
            raise ValueError("memory query must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        selected = self._default_tiers if tiers is None else _coerce_tiers(tiers)
        persistent_candidates = (
            self._persistent.query(query, reader=reader, top_k=top_k)
            if self._persistent is not None
            and (Tier.SEMANTIC in selected or Tier.LEXICAL in selected)
            else ()
        )
        rankings: list[tuple[MemoryEntry, ...]] = []
        if Tier.SEMANTIC in selected:
            rankings.append(
                self._semantic(
                    query,
                    reader=reader,
                    top_k=top_k,
                    persistent_candidates=persistent_candidates,
                )
            )
        if Tier.LEXICAL in selected:
            rankings.append(
                self._lexical(
                    query,
                    reader=reader,
                    persistent_candidates=persistent_candidates,
                )
            )
        if Tier.PROCEDURAL in selected:
            rankings.append(self._procedural.query(query, reader=reader, top_k=top_k))
        write_order = {
            canonical_entry_id(entry): index
            for index, entry in enumerate(self._working.snapshot(reader=reader))
        }
        return _fuse_rankings(
            rankings,
            reader=reader,
            top_k=top_k,
            write_order=write_order,
            entry_gate=self._entry_gate,
        )

    def dedup(self, entry: MemoryEntry, *, threshold: float | None = None) -> bool:
        if not self._entry_gate(entry):
            raise MemoryPolicyError("memory entry rejected by contamination policy")
        persistent_threshold = 0.95 if threshold is None else threshold
        return self._working.dedup(entry, threshold=threshold) or (
            self._persistent is not None
            and self._persistent.dedup(entry, threshold=persistent_threshold)
        )

    def _semantic(
        self,
        query: str,
        *,
        reader: MemoryScope,
        top_k: int,
        persistent_candidates: Sequence[MemoryEntry],
    ) -> tuple[MemoryEntry, ...]:
        # Fusion needs each local ranking beyond the final cutoff; otherwise a
        # semantic top-1 receives two-channel credit while the lexical top-1 is
        # absent from the semantic ranking entirely.
        visible_count = len(self._working.snapshot(reader=reader))
        candidates = list(
            self._working.query(
                query,
                reader=reader,
                top_k=max(top_k, visible_count),
            )
        )
        candidates.extend(persistent_candidates)
        unique: dict[str, MemoryEntry] = {}
        for entry in candidates:
            if entry.visible_to(reader):
                unique.setdefault(canonical_entry_id(entry), entry)
        return tuple(unique.values())

    def _lexical(
        self,
        query: str,
        *,
        reader: MemoryScope,
        persistent_candidates: Sequence[MemoryEntry],
    ) -> tuple[MemoryEntry, ...]:
        candidates = list(self._working.snapshot(reader=reader))
        candidates.extend(persistent_candidates)
        entries_by_id: dict[str, MemoryEntry] = {}
        for entry in candidates:
            if entry.visible_to(reader) and self._entry_gate(entry):
                entries_by_id.setdefault(canonical_entry_id(entry), entry)
        entries = tuple(entries_by_id.values())
        if not entries:
            return ()
        tokens = _bounded_lexical_tokens(query)
        if not tokens:
            return ()
        result = grep(
            "|".join(re.escape(token) for token in tokens),
            (MemorySource(entries),),
            ignore_case=True,
            output_mode="count",
        )
        query_phrase = query.casefold().strip()
        scored: list[tuple[int, int, int, int, MemoryEntry]] = []
        for locator, _count in result.counts:
            index = int(locator.split(":")[1])
            entry = entries[index]
            entry_tokens = Counter(token.casefold() for token in _WORD.findall(entry.content))
            coverage = sum(token in entry_tokens for token in tokens)
            occurrences = sum(entry_tokens[token] for token in tokens)
            phrase = int(bool(query_phrase) and query_phrase in entry.content.casefold())
            scored.append((-phrase, -coverage, -occurrences, -index, entry))
        scored.sort()
        return tuple(item[-1] for item in scored)


def _coerce_tiers(values: Sequence[Tier | str]) -> tuple[Tier, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("tiers must be a sequence")
    selected = tuple(Tier(value) for value in values)
    return tuple(dict.fromkeys(selected))


def _fuse_rankings(
    rankings: Sequence[Sequence[MemoryEntry]],
    *,
    reader: MemoryScope,
    top_k: int,
    write_order: dict[str, int],
    entry_gate: Callable[[MemoryEntry], bool],
) -> tuple[MemoryEntry, ...]:
    """Merge rankings by reciprocal rank, de-duplicating immutable entries."""

    scores: dict[str, float] = {}
    entries: dict[str, MemoryEntry] = {}
    first_seen: dict[str, int] = {}
    sequence = 0
    for ranking in rankings:
        visible_rank = 0
        seen_in_tier: set[str] = set()
        for entry in ranking:
            if not entry.visible_to(reader) or not entry_gate(entry):
                continue
            entry_id = canonical_entry_id(entry)
            if entry_id in seen_in_tier:
                continue
            seen_in_tier.add(entry_id)
            entries[entry_id] = entry
            first_seen.setdefault(entry_id, sequence)
            sequence += 1
            scores[entry_id] = scores.get(entry_id, 0.0) + 1.0 / (_RRF_K + visible_rank + 1)
            visible_rank += 1
    ordered = sorted(
        scores,
        key=lambda entry_id: (
            -scores[entry_id],
            -write_order.get(entry_id, -1),
            first_seen[entry_id],
            entry_id,
        ),
    )
    return tuple(entries[entry_id] for entry_id in ordered[:top_k])


def _bounded_lexical_tokens(query: str) -> tuple[str, ...]:
    """Keep a deterministic query prefix within the grep engine's pattern cap."""

    selected: list[str] = []
    seen: set[str] = set()
    encoded_length = 0
    for raw in _WORD.findall(query):
        token = raw.casefold()
        if not token or token in seen:
            continue
        escaped_length = len(re.escape(token))
        next_length = encoded_length + escaped_length + int(bool(selected))
        if next_length > _LEXICAL_PATTERN_CAP:
            break
        selected.append(token)
        seen.add(token)
        encoded_length = next_length
    return tuple(selected)


__all__ = [
    "NullProceduralMemory",
    "MemoryPolicyError",
    "ProceduralMemory",
    "Tier",
    "TieredMemory",
]
