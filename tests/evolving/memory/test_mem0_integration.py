"""Opt-in real Mem0/Chroma persistence test with no remote LLM calls."""

from __future__ import annotations

import json
import os

import pytest

from alphaapollo.evolving.memory import DEFAULT_MEMORY_BACKENDS, MemoryScope
from alphaapollo.reasoning.verification import VerificationResult
from alphaapollo.workflows.config import MemoryConfig
from alphaapollo.workflows.memory import WorkflowMemorySession
from alphaapollo.workflows.records import WorkflowInput


def test_real_mem0_persists_exact_entry_across_instances(tmp_path) -> None:
    pytest.importorskip("mem0")
    model = os.environ.get("ALPHAAPOLLO_MEM0_EMBED_MODEL")
    if not model:
        pytest.skip("set ALPHAAPOLLO_MEM0_EMBED_MODEL to run the real Mem0 test")
    config = {
        "llm": {
            "provider": "openai",
            "config": {"model": "unused-infer-false", "api_key": "not-used"},
        },
        "embedder": {"provider": "huggingface", "config": {"model": model}},
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "alphaapollo_workflow_test",
                "path": str(tmp_path / "chroma"),
            },
        },
        "history_db_path": str(tmp_path / "history.db"),
        "version": "v1.1",
    }
    session = WorkflowMemorySession(
        MemoryConfig(mode="persistent", persistent_backend="mem0", persistent_config=config),
        workflow_name="memory_smoke",
        run_id="run-1",
        journal_path=tmp_path / "memory.jsonl",
    )
    session.observe_output(
        WorkflowInput(input_id="rc-task", problem="RC waveform ERR42"),
        branch_index=0,
        step_id="verify",
        output=VerificationResult(
            request_id="verify-1",
            verdict="fail",
            candidate="RC candidate ERR42",
            candidate_ref="candidate-artifact",
            feedback="Repair ERR42 before submission.",
        ),
    )
    fresh = DEFAULT_MEMORY_BACKENDS.build("mem0", config)
    scope = MemoryScope(
        namespace_id="workflow:memory_smoke", task_id="rc-task:branch-0", run_id="run-1"
    )
    entries = fresh.query("RC waveform ERR42", reader=scope)
    assert json.loads(entries[0].content)["candidate_ref"] == "candidate-artifact"
    assert json.loads(entries[0].content)["feedback"] == "Repair ERR42 before submission."
    assert (
        fresh.query("RC waveform ERR42", reader=scope.model_copy(update={"run_id": "run-2"})) == ()
    )
