"""Domain adapter contracts for Workflow memory ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from alphaapollo.evolving.memory import MemoryEntry, MemoryScope
from alphaapollo.workflows.records import WorkflowInput


@dataclass(frozen=True, slots=True)
class WorkflowMemoryContext:
    """Stable Workflow coordinates supplied to one domain memory adapter."""

    workflow_name: str
    run_id: str
    workflow_input: WorkflowInput
    branch_index: int
    step_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.workflow_name, str)
            or not isinstance(self.run_id, str)
            or not isinstance(self.step_id, str)
            or not self.workflow_name.strip()
            or not self.run_id.strip()
            or not self.step_id.strip()
        ):
            raise ValueError("workflow_name, run_id, and step_id must be non-empty")
        if not isinstance(self.workflow_input, WorkflowInput):
            raise TypeError("workflow_input must be a WorkflowInput")
        if (
            isinstance(self.branch_index, bool)
            or not isinstance(self.branch_index, int)
            or self.branch_index < 0
        ):
            raise ValueError("branch_index must be a non-negative integer")


@runtime_checkable
class WorkflowMemoryAdapter(Protocol):
    """Define domain scope, retrieval, and source-attributed entry projection."""

    def scope(self, *, context: WorkflowMemoryContext) -> MemoryScope: ...

    def retrieval_query(
        self,
        *,
        context: WorkflowMemoryContext,
        prompt: str,
    ) -> str: ...

    def project_output(
        self,
        output: object,
        *,
        context: WorkflowMemoryContext,
        scope: MemoryScope,
    ) -> MemoryEntry | None: ...


def build_workflow_memory_adapter(name: str, *, cross_run: bool = False) -> WorkflowMemoryAdapter:
    """Build a configured built-in adapter without importing domain modules eagerly."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("workflow memory adapter name must be non-empty")
    normalized = name.strip().lower()
    if normalized == "verification":
        from alphaapollo.workflows.memory.verification import VerificationMemoryAdapter

        return VerificationMemoryAdapter(cross_run=cross_run)
    if normalized == "robotics":
        from alphaapollo.workflows.memory.robotics import RoboticsMemoryAdapter

        return RoboticsMemoryAdapter(cross_run=cross_run)
    if normalized == "unified":
        from alphaapollo.workflows.memory.unified import UnifiedMemoryAdapter

        return UnifiedMemoryAdapter(cross_run=cross_run)
    raise ValueError(
        f"unknown workflow memory adapter {name!r}; "
        "expected 'verification', 'robotics', or 'unified'"
    )


__all__ = [
    "WorkflowMemoryAdapter",
    "WorkflowMemoryContext",
    "build_workflow_memory_adapter",
]
