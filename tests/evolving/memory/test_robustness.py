"""Within-run robustness gates retained from shared-working-memory."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from alphaapollo.evolving.memory import (
    MemoryEntry,
    MemoryKind,
    MemoryProvenance,
    MemoryScope,
    Tier,
    TieredMemory,
    WorkingMemory,
)


class _MappingEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._vectors[text] for text in texts], dtype=np.float64)


def _entry(content: str, *, kind: MemoryKind = MemoryKind.RESULT, run: str = "run-1"):
    return MemoryEntry(
        kind=kind,
        content=content,
        scope=MemoryScope(namespace_id="robust", task_id="task", run_id=run),
        provenance=MemoryProvenance(actor_id="test", source_event_id=f"event:{content}"),
    )


def test_negation_is_distinct_in_lexical_and_semantic_recall() -> None:
    embedder = _MappingEmbedder(
        {
            "the sum converges": [1.0, 0.0],
            "the sum does not converge": [0.0, 1.0],
            "does not converge": [0.0, 1.0],
        }
    )
    memory = TieredMemory(
        working=WorkingMemory(embedder),
        default_tiers=(Tier.SEMANTIC, Tier.LEXICAL),
    )
    negated = _entry("the sum does not converge", kind=MemoryKind.CLAIM)
    affirmed = _entry("the sum converges", kind=MemoryKind.CLAIM)
    memory.write(negated)
    memory.write(affirmed)

    lexical = memory.query("does not converge", reader=negated.scope, tiers=(Tier.LEXICAL,))
    fused = memory.query("does not converge", reader=negated.scope, top_k=1)

    assert lexical[0] == negated
    assert affirmed in lexical
    assert fused == (negated,)


def test_correction_wins_a_semantic_lexical_rank_tie_by_recency() -> None:
    embedder = _MappingEmbedder(
        {
            "root": [1.0, 0.0],
            "root is 2": [1.0, 0.0],
            "root is 5 corrected": [0.0, 1.0],
        }
    )
    memory = TieredMemory(
        working=WorkingMemory(embedder),
        default_tiers=(Tier.SEMANTIC, Tier.LEXICAL),
    )
    memory.write(_entry("root is 2", kind=MemoryKind.CLAIM))
    correction = _entry("root is 5 corrected")
    memory.write(correction)

    assert memory.query("root", reader=correction.scope, top_k=1) == (correction,)


def test_false_dedup_and_cross_run_leak_rates_are_zero() -> None:
    near = [0.6, math.sqrt(1 - 0.6**2)]
    memory = WorkingMemory(
        _MappingEmbedder(
            {
                "x is prime": [1.0, 0.0],
                "x is not prime": near,
                "query": [1.0, 0.0],
            }
        ),
        dedup_threshold=0.9,
    )
    stored = _entry("x is prime", kind=MemoryKind.CLAIM)
    candidate = _entry("x is not prime", kind=MemoryKind.CLAIM)
    hidden = _entry("x is prime", kind=MemoryKind.CLAIM, run="run-2")
    memory.write(stored)
    memory.write(hidden)

    assert not memory.dedup(candidate)
    assert memory.query("query", reader=stored.scope, top_k=10) == (stored,)
