"""Single configured producer composing the domain projectors (#347)."""

from __future__ import annotations

from alphaapollo.evolving.memory import MemoryEntry, MemoryScope
from alphaapollo.workflows.memory.adapter import WorkflowMemoryContext
from alphaapollo.workflows.memory.robotics import RoboticsMemoryAdapter
from alphaapollo.workflows.memory.verification import VerificationMemoryAdapter


class UnifiedMemoryAdapter:
    """First-match composition of the domain projectors.

    This class owns no domain knowledge: it never imports or classifies domain
    result types. Each projector keeps its own authority gate — verifier
    verdicts with bound candidates, environment-identified terminal episodes,
    coordinator-verified evaluations — and the composite offers every output
    to each projector in turn; the first projected entry wins and anything no
    projector claims fails closed. A new domain composes in by constructing
    its projector here, not by adding classification branches.

    What the composite does own is the shared scope scheme: ``scope()`` also
    serves retrieval, where no output exists to consult a projector, so all
    entries live under the workflow-generic ``workflow:<name>`` namespace —
    Robotics entries move off their domain namespace when a run opts into
    ``unified``.
    """

    def __init__(
        self,
        *,
        cross_run: bool = False,
    ) -> None:
        self._cross_run = bool(cross_run)
        projectors = [
            VerificationMemoryAdapter(cross_run=cross_run),
            RoboticsMemoryAdapter(cross_run=cross_run),
        ]
        self._projectors = tuple(projectors)

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
        for projector in self._projectors:
            entry = projector.project_output(output, context=context, scope=scope)
            if entry is not None:
                return entry
        return None


__all__ = ["UnifiedMemoryAdapter"]
