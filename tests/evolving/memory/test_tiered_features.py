"""Regression coverage for the ported shared-working-memory channels."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphaapollo.common.execution.tools import ExecutionContext, ToolRequest
from alphaapollo.evolving.memory import (
    MemoryEntry,
    MemoryKind,
    MemoryPolicyError,
    MemoryProvenance,
    MemoryScope,
    Tier,
    TieredMemory,
    WorkingMemory,
)
from alphaapollo.evolving.memory.grep_stm import (
    FileSource,
    GrepError,
    MemorySource,
    TranscriptSource,
    grep,
)
from alphaapollo.evolving.memory.scratchpad import (
    bounded_scratchpad,
    update_scratchpad,
)
from alphaapollo.evolving.memory.tool_adapters import (
    ScratchpadToolAdapter,
    serve_transcript_grep,
)
from alphaapollo.evolving.memory_profile import MEMORY_FULL, NAMED_PROFILES


def _entry(content: str, *, run: str = "run-1") -> MemoryEntry:
    return MemoryEntry(
        kind=MemoryKind.RESULT,
        content=content,
        scope=MemoryScope(namespace_id="workflow:test", task_id="problem:branch-0", run_id=run),
        provenance=MemoryProvenance(actor_id="verifier", source_event_id=f"event:{content}"),
    )


def test_grep_searches_transcript_memory_and_files_deterministically(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("first\nERR42 from file\n", encoding="utf-8")
    sources = (
        TranscriptSource(({"role": "assistant", "content": "ERR42 from transcript"},)),
        MemorySource((_entry("ERR42 from verified memory"),)),
        FileSource(tmp_path),
    )

    result = grep("ERR42", sources, fixed_string=True, output_mode="content")

    assert result.total == 3
    assert [(hit.source, hit.locator) for hit in result.hits] == [
        ("transcript", "turn:0:assistant"),
        ("memory", "entry:0:result"),
        ("files", "notes.txt"),
    ]


def test_grep_supports_context_count_and_bounded_regex(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("before\nvalue 42\nafter\nvalue 43\n", encoding="utf-8")
    source = (FileSource(tmp_path),)

    content = grep(r"value \d+", source, context=1)
    counts = grep(r"value \d+", source, output_mode="count", head_limit=1)

    assert content.hits[0].before == ("before",)
    assert content.hits[0].after == ("after",)
    assert counts.counts == (("notes.txt", 1),)
    assert counts.total == 2 and counts.truncated
    with pytest.raises(GrepError, match="backreferences"):
        grep(r"(a)\1", source)
    with pytest.raises(GrepError, match="nested variable repeats"):
        grep(r"(a+)+", source)


def test_tiered_memory_fuses_semantic_lexical_and_procedural_channels() -> None:
    procedural = _entry("procedure: isolate the failing invariant first")

    class Procedures:
        def query(self, query: str, *, reader: MemoryScope, top_k: int):
            del query, reader, top_k
            return (procedural,)

    memory = TieredMemory(
        procedural=Procedures(),
        default_tiers=(Tier.SEMANTIC, Tier.LEXICAL, Tier.PROCEDURAL),
    )
    exact = _entry("failure signature ERR42 occurred after normalization")
    other = _entry("a general result about normalization")
    memory.write(other)
    memory.write(exact)

    found = memory.query("ERR42", reader=exact.scope, top_k=3)

    assert exact in found
    assert procedural in found
    assert len(found) == 3


def test_tiered_lexical_channel_never_crosses_scope() -> None:
    memory = TieredMemory(default_tiers=(Tier.LEXICAL,))
    visible = _entry("ERR42 visible", run="run-1")
    hidden = _entry("ERR42 hidden", run="run-2")
    memory.write(visible)
    memory.write(hidden)

    assert memory.query("ERR42", reader=visible.scope) == (visible,)


def test_tiered_lexical_channel_queries_persistent_memory_after_restart() -> None:
    persistent = WorkingMemory()
    visible = _entry("durable lexical marker ERR42")
    persistent.write(visible)
    restarted = TieredMemory(persistent=persistent, default_tiers=(Tier.LEXICAL,))

    assert restarted.snapshot(reader=visible.scope) == ()
    assert restarted.query("ERR42", reader=visible.scope) == (visible,)


def test_tiered_lexical_channel_bounds_large_model_context() -> None:
    memory = TieredMemory(default_tiers=(Tier.LEXICAL,))
    visible = _entry("needle is preserved")
    memory.write(visible)
    query = "needle " + " ".join(f"token{index}" for index in range(2_000))

    assert memory.query(query, reader=visible.scope, top_k=1) == (visible,)


def test_tiered_memory_contamination_gate_fails_closed() -> None:
    memory = TieredMemory(entry_gate=lambda entry: "PRIVATE-GOLD" not in entry.content)

    with pytest.raises(MemoryPolicyError, match="contamination policy"):
        memory.write(_entry("PRIVATE-GOLD must never enter model memory"))


def test_memory_profiles_keep_channels_independent() -> None:
    assert MEMORY_FULL.tier_names == ("semantic", "lexical")
    assert MEMORY_FULL.scratchpad
    assert NAMED_PROFILES["scratchpad"].tier_names == ()
    assert not NAMED_PROFILES["semantic_lexical"].scratchpad


def test_scratchpad_append_replace_and_bounded_read() -> None:
    text = update_scratchpad("", op="append", content="## Plan\nTry A")
    text = update_scratchpad(text, op="replace", section="Plan", content="Try B")

    assert text == "## Plan\nTry B\n"
    rendered, truncated = bounded_scratchpad("x" * 7_000)
    assert truncated and rendered.startswith("[truncated]\n") and len(rendered) < 7_000


class _Sandbox:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def read_file(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content


def test_scratchpad_tool_and_transcript_grep_use_current_tool_contract() -> None:
    sandbox = _Sandbox()
    adapter = ScratchpadToolAdapter(sandbox)
    context = ExecutionContext(session_id="session", branch_id="branch")
    append = adapter.execute(
        ToolRequest(
            call_id="call-1",
            tool_id="scratchpad",
            arguments={"op": "append", "content": "## Facts\nx=7"},
        ),
        context,
    )
    read = adapter.execute(
        ToolRequest(call_id="call-2", tool_id="scratchpad", arguments={"op": "read"}),
        context,
    )
    grep_result = serve_transcript_grep(
        ToolRequest(
            call_id="call-3",
            tool_id="grep_workspace",
            arguments={"source": "transcript", "pattern": "x=7", "fixed_string": True},
        ),
        [{"role": "assistant", "content": "remember x=7"}],
    )

    assert append.exit_code == 0
    assert read.stdout == "## Facts\nx=7"
    assert grep_result is not None and "remember x=7" in grep_result.stdout
