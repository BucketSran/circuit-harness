"""Robotics terminal-outcome producer for generic Workflow memory."""

from __future__ import annotations

import json
from collections.abc import Mapping

from alphaapollo.evolving.memory import (
    MemoryEntry,
    MemoryKind,
    MemoryLifetime,
    MemoryProvenance,
    MemoryScope,
)
from alphaapollo.reasoning.runtime import AgentResult
from alphaapollo.workflows.memory.adapter import WorkflowMemoryContext


class RoboticsMemoryAdapter:
    """Store only explicit terminal results from a Robotics environment episode.

    With ``cross_run`` (#340) the terminal outcomes become explicit per-run
    evidence that outlives the run: lifetime turns ``PERSISTENT`` and the task
    id drops its branch suffix so every run of the same task shares one scope.
    The default keeps today's scope, lifetime, and isolation byte-identically.
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
            namespace_id=f"robotics:{context.workflow_name}",
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
            f"{context.workflow_input.problem}\nrobotics step {context.step_id}\n{prompt[-4_000:]}"
        )

    def project_output(
        self,
        output: object,
        *,
        context: WorkflowMemoryContext,
        scope: MemoryScope,
    ) -> MemoryEntry | None:
        if not isinstance(output, AgentResult) or not output.turns:
            return None
        # Environment identity, not terminal shape, is this projector's
        # authority gate: the default tool environment also ends episodes with
        # done=True and a boolean success (budget exhaustion returns
        # success=False), so accept only results whose runtime recorded a
        # robotics initialization — the validated task payload's benchmark
        # under metadata['environment_init'].
        metadata = output.metadata if isinstance(output.metadata, Mapping) else {}
        initialization = metadata.get("environment_init")
        if not isinstance(initialization, Mapping) or not initialization.get("benchmark"):
            return None
        transition = output.turns[-1].environment_transition
        if not transition.done or not isinstance(transition.success, bool):
            return None

        metadata = _scalar_terminal_metadata(transition.metadata)
        event_id = metadata.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            event_id = output.task_id
        record = {
            "schema_version": 1,
            "workflow_name": context.workflow_name,
            "input_id": context.workflow_input.input_id,
            "branch_index": context.branch_index,
            "step_id": context.step_id,
            "success": transition.success,
            "termination_reason": transition.termination_reason,
            "reward": transition.reward,
            "agent_summary": output.final_text[:12_000],
            "agent_summary_trust": "untrusted",
            "terminal_metadata": metadata,
        }
        return MemoryEntry(
            kind=MemoryKind.RESULT if transition.success else MemoryKind.FAILURE,
            content=json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            scope=scope,
            provenance=MemoryProvenance(
                actor_id="robotics-environment",
                source_event_id=event_id,
                artifact_ids=(output.task_id,),
            ),
            lifetime=MemoryLifetime.PERSISTENT if self._cross_run else MemoryLifetime.RUN,
        )


def _scalar_terminal_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Keep provenance-sized terminal metadata without copying observations or images."""

    allowed = ("event_id", "episode_id", "steps_used", "turns_used", "backend")
    return {
        key: value
        for key in allowed
        if isinstance(value := metadata.get(key), (bool, float, int, str))
    }


__all__ = ["RoboticsMemoryAdapter"]
