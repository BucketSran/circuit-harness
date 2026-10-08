"""Scoped working and Mem0 memory contract tests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from alphaapollo.evolving.memory import (
    Mem0MemoryAdapter,
    Mem0MemoryError,
    MemoryEntry,
    MemoryKind,
    MemoryLifetime,
    MemoryProvenance,
    MemoryScope,
    WorkingMemory,
    canonical_entry_id,
)


def _entry(
    content: str,
    *,
    namespace: str = "bio",
    task: str = "gene-task",
    run: str = "run-1",
    lifetime: MemoryLifetime = MemoryLifetime.RUN,
) -> MemoryEntry:
    return MemoryEntry(
        kind=MemoryKind.RESULT,
        content=content,
        scope=MemoryScope(namespace_id=namespace, task_id=task, run_id=run),
        provenance=MemoryProvenance(actor_id="coordinator", source_event_id="event-1"),
        lifetime=lifetime,
    )


def test_working_memory_filters_scope_before_semantic_ranking() -> None:
    memory = WorkingMemory()
    visible = _entry("rejected ATP6V0B prefix after Recall@FDR fell")
    other_run = _entry("rejected ATP6V0B prefix", run="run-2")
    other_task = _entry("rejected ATP6V0B prefix", task="drug-task")
    persistent = _entry(
        "stable task instruction about ATP6V0B",
        run="authoring-run",
        lifetime=MemoryLifetime.PERSISTENT,
    )
    for item in (visible, other_run, other_task, persistent):
        memory.write(item)

    found = memory.query("ATP6V0B rejected prefix", reader=visible.scope, top_k=10)

    assert visible in found
    assert persistent in found
    assert other_run not in found
    assert other_task not in found
    assert memory.dedup(visible)


def test_memory_scope_rejects_whitespace_before_calling_mem0() -> None:
    with pytest.raises(ValidationError, match="must not contain whitespace"):
        MemoryScope(namespace_id="aime demo", task_id="task", run_id="run-1")


def test_working_memory_rejects_negative_dedup_threshold_at_construction() -> None:
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        WorkingMemory(dedup_threshold=-0.1)


class _FakeMem0:
    def __init__(
        self,
        *,
        empty_add: bool = False,
        add_event: str = "ADD",
    ) -> None:
        self.empty_add = empty_add
        self.add_event = add_event
        self.records: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []

    def add(self, memory: str, **kwargs: Any) -> list[dict[str, Any]]:
        _validate_entity_ids(kwargs)
        self.add_calls.append({"memory": memory, **kwargs})
        if self.empty_add:
            return []
        record = {
            "id": f"mem-{len(self.records)}",
            "memory": memory,
            "metadata": kwargs["metadata"],
            "user_id": kwargs["user_id"],
            "agent_id": kwargs["agent_id"],
            "run_id": kwargs["run_id"],
            "score": 0.9,
        }
        self.records.append(record)
        return [{"id": record["id"], "event": self.add_event}]

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        del query
        filters = kwargs["filters"]
        _validate_entity_ids(filters)
        threshold = float(kwargs.get("threshold", 0.1))
        top_k = int(kwargs.get("top_k", 100))
        return {
            "results": [
                item
                for item in self.records
                if item["score"] >= threshold
                and all(item[key] == filters[key] for key in ("user_id", "agent_id", "run_id"))
            ][:top_k]
        }


def _validate_entity_ids(values: dict[str, Any]) -> None:
    for key in ("user_id", "agent_id", "run_id"):
        value = values[key]
        if (
            not isinstance(value, str)
            or not value
            or any(character.isspace() for character in value)
        ):
            raise ValueError(f"invalid {key}")


def test_mem0_adapter_uses_exact_content_and_all_scope_filters() -> None:
    client = _FakeMem0()
    adapter = Mem0MemoryAdapter(client)
    entry = _entry('{"decision":"rejected","tested_budget_prefix":["G1"]}')

    assert adapter.write(entry) == canonical_entry_id(entry)
    call = client.add_calls[0]
    assert call["infer"] is False
    assert call["user_id"] == "bio"
    assert call["agent_id"] == "gene-task"
    assert call["run_id"] == "run-1"
    assert call["memory"] == entry.model_dump_json()
    assert adapter.query("rejected G1", reader=entry.scope) == (entry,)
    assert (
        adapter.query(
            "rejected G1",
            reader=MemoryScope(namespace_id="bio", task_id="gene-task", run_id="run-2"),
        )
        == ()
    )


def test_mem0_fake_models_real_threshold_top_k_and_bare_add_result() -> None:
    client = _FakeMem0()
    first = _entry("first", run="run-1")
    second = _entry("second", run="run-1")

    first_result = client.add(
        first.model_dump_json(),
        user_id="bio",
        agent_id="gene-task",
        run_id="run-1",
        metadata={},
        infer=False,
    )
    client.add(
        second.model_dump_json(),
        user_id="bio",
        agent_id="gene-task",
        run_id="run-1",
        metadata={},
        infer=False,
    )
    client.records[0]["score"] = 0.05

    found = client.search(
        "query",
        filters={"user_id": "bio", "agent_id": "gene-task", "run_id": "run-1"},
        threshold=0.1,
        top_k=1,
    )

    assert isinstance(first_result, list)
    assert [record["memory"] for record in found["results"]] == [second.model_dump_json()]


def test_mem0_adapter_fails_closed_when_write_is_not_acknowledged() -> None:
    with pytest.raises(Mem0MemoryError, match="did not acknowledge"):
        Mem0MemoryAdapter(_FakeMem0(empty_add=True)).write(_entry("verified result"))


def test_mem0_adapter_rejects_non_write_result_as_acknowledgement() -> None:
    with pytest.raises(Mem0MemoryError, match="did not acknowledge"):
        Mem0MemoryAdapter(_FakeMem0(add_event="ERROR")).write(_entry("verified result"))


def test_mem0_adapter_rejects_tampered_entry_metadata() -> None:
    client = _FakeMem0()
    adapter = Mem0MemoryAdapter(client)
    entry = _entry("verified result")
    adapter.write(entry)
    client.records[0]["metadata"]["alphaapollo_entry_id"] = "tampered"

    assert adapter.query("verified result", reader=entry.scope) == ()


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {
            "alphaapollo_schema_version": 2,
            "alphaapollo_entry_id": "unused",
            "alphaapollo_lifetime": "run",
        },
    ],
)
def test_mem0_adapter_rejects_records_without_complete_trusted_metadata(metadata) -> None:
    client = _FakeMem0()
    adapter = Mem0MemoryAdapter(client)
    entry = _entry("verified result")
    adapter.write(entry)
    if metadata is None:
        client.records[0].pop("metadata")
    else:
        client.records[0]["metadata"] = metadata

    assert adapter.query("verified result", reader=entry.scope) == ()


def test_mem0_adapter_rejects_lifetime_metadata_mismatch() -> None:
    client = _FakeMem0()
    adapter = Mem0MemoryAdapter(client)
    entry = _entry("verified result")
    adapter.write(entry)
    client.records[0]["metadata"]["alphaapollo_lifetime"] = "persistent"

    assert adapter.query("verified result", reader=entry.scope) == ()


def test_two_run_repeated_writes_are_distinct_persistent_records() -> None:
    """#340 contract (c): per-run evidence, no cross-run dedup, both retrievable.

    canonical_entry_id hashes the full entry including scope.run_id, so the
    same logical outcome written in two runs stores two records; a later run
    sees both through the PERSISTENT visibility exemption.
    """

    first = _entry("accepted G2-first prefix", run="run-1", lifetime=MemoryLifetime.PERSISTENT)
    second = _entry("accepted G2-first prefix", run="run-2", lifetime=MemoryLifetime.PERSISTENT)
    assert canonical_entry_id(first) != canonical_entry_id(second)

    reader = MemoryScope(namespace_id="bio", task_id="gene-task", run_id="run-3")

    # Working store as the durable stand-in.
    store = WorkingMemory()
    store.write(first)
    store.write(second)
    assert len(store) == 2
    found = store.query("accepted G2-first prefix", reader=reader, top_k=5)
    assert {canonical_entry_id(entry) for entry in found} == {
        canonical_entry_id(first),
        canonical_entry_id(second),
    }

    # Real backend shape via the Mem0 fake: persistent-channel routing included.
    adapter = Mem0MemoryAdapter(_FakeMem0())
    adapter.write(first)
    adapter.write(second)
    recalled = adapter.query("accepted G2-first prefix", reader=reader, top_k=5)
    assert {canonical_entry_id(entry) for entry in recalled} == {
        canonical_entry_id(first),
        canonical_entry_id(second),
    }
