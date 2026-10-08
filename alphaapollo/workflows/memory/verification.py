"""Verifier-specific producer for generic Workflow memory entries."""

from __future__ import annotations

import json

from alphaapollo.evolving.memory import (
    MemoryEntry,
    MemoryKind,
    MemoryLifetime,
    MemoryProvenance,
    MemoryScope,
)
from alphaapollo.reasoning.verification import VerificationResult
from alphaapollo.workflows.memory.adapter import WorkflowMemoryContext


class VerificationMemoryAdapter:
    """Convert candidate-bound verifier judgments into ``MemoryEntry`` values.

    With ``cross_run`` (#340) the entries become explicit per-run evidence
    that outlives the run: lifetime turns ``PERSISTENT`` and the task id drops
    its branch suffix so every run of the same input shares one scope. Branch
    isolation is deliberately traded away for these entries; the default
    keeps today's scope, lifetime, and isolation byte-identically.
    """

    def __init__(self, *, cross_run: bool = False) -> None:
        self._cross_run = bool(cross_run)

    def scope(self, *, context: WorkflowMemoryContext) -> MemoryScope:
        task_id = (
            context.workflow_input.input_id
            if self._cross_run
            else f"{context.workflow_input.input_id}:branch-{context.branch_index}"
        )
        return MemoryScope(
            namespace_id=f"workflow:{context.workflow_name}",
            task_id=task_id,
            run_id=context.run_id,
        )

    def retrieval_query(
        self,
        *,
        context: WorkflowMemoryContext,
        prompt: str,
    ) -> str:
        return (
            f"{context.workflow_input.problem}\nworkflow step {context.step_id}\n{prompt[-4_000:]}"
        )

    def project_output(
        self,
        output: object,
        *,
        context: WorkflowMemoryContext,
        scope: MemoryScope,
    ) -> MemoryEntry | None:
        if not isinstance(output, VerificationResult) or output.candidate_ref is None:
            return None
        record = {
            "schema_version": 1,
            "workflow_name": context.workflow_name,
            "input_id": context.workflow_input.input_id,
            "branch_index": context.branch_index,
            "verifier_step": context.step_id,
            "candidate_ref": output.candidate_ref,
            "candidate": output.candidate[:12_000],
            "verdict": output.verdict,
            "feedback": output.feedback[:4_000],
            "candidate_sha256": output.candidate_sha256,
        }
        return MemoryEntry(
            kind=(MemoryKind.RESULT if output.verdict == "pass" else MemoryKind.FAILURE),
            content=json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            scope=scope,
            provenance=MemoryProvenance(
                actor_id="workflow-verifier",
                source_event_id=output.request_id,
                artifact_ids=(output.candidate_ref,),
            ),
            lifetime=MemoryLifetime.PERSISTENT if self._cross_run else MemoryLifetime.RUN,
        )


__all__ = ["VerificationMemoryAdapter"]
