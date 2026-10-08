# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic batched interpreter for the strict Workflow configuration DSL."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from alphaapollo.reasoning.runtime import AgentResult, AgentRuntime, AgentTask
from alphaapollo.reasoning.verification import (
    VerificationContractError,
    VerificationRequest,
    VerificationResult,
    Verifier,
)
from alphaapollo.reasoning.verification.base import _validate_result_batch
from alphaapollo.workflows.config import _thaw_json
from alphaapollo.workflows.records import (
    StepResult,
    Workflow,
    WorkflowInput,
    WorkflowResult,
    WorkflowResumeStep,
    WorkflowStepOutput,
)
from alphaapollo.workflows.selection import select_result

if TYPE_CHECKING:
    from alphaapollo.workflows.memory import WorkflowMemorySession

__all__ = ["WorkflowExecutionError", "WorkflowExecutor"]


class WorkflowExecutionError(RuntimeError):
    """A configured resource violated its ordered batch contract."""


@dataclass(slots=True)
class _ExecutionSlot:
    input_index: int
    workflow_input: WorkflowInput
    branch_index: int
    current_step: str
    history: list[StepResult] = field(default_factory=list)
    step_counts: dict[str, int] = field(default_factory=dict)
    transition_counts: dict[int, int] = field(default_factory=dict)
    executions: int = 0
    output: WorkflowStepOutput | None = None
    completed: bool = False

    @property
    def key(self) -> tuple[int, int]:
        """The slot's identity in a batch: one branch of one input.

        Scheduling matches a group's outputs back to its slots by this value
        rather than by object identity, so the match survives a slot being
        rebuilt (a frozen record replaced rather than mutated) instead of
        depending on every slot staying alive for the whole run.
        """

        return (self.input_index, self.branch_index)


class WorkflowExecutor:
    """Interpret one immutable plan over injected Runtime and Verifier registries.

    Every scheduling round groups slots by the exact same step and resource
    target.  Each group is sent through exactly one ``run_batch`` or
    ``verify_batch`` call, and the returned positions are matched back without
    sorting.  Workflow input order is restored when results are assembled.
    """

    def __init__(
        self,
        workflow: Workflow,
        runtimes: Mapping[str, AgentRuntime],
        verifiers: Mapping[str, Verifier],
        memory: WorkflowMemorySession | None = None,
        step_observer: Callable[[StepResult], None] | None = None,
    ) -> None:
        if not isinstance(workflow, Workflow):
            raise TypeError("workflow must be a Workflow")
        self.workflow = workflow
        self._runtimes = _validate_registry(runtimes, "runtimes", "run_batch")
        self._verifiers = _validate_registry(verifiers, "verifiers", "verify_batch")
        self._memory = memory
        if step_observer is not None and not callable(step_observer):
            raise TypeError("step_observer must be callable when provided")
        self._step_observer = step_observer
        self._validate_targets()

    def run(self, workflow_input: WorkflowInput) -> WorkflowResult:
        """Execute one input through the same batch-first code path."""

        if not isinstance(workflow_input, WorkflowInput):
            raise TypeError("workflow_input must be a WorkflowInput")
        return self.run_batch((workflow_input,))[0]

    def run_batch(self, inputs: Sequence[WorkflowInput]) -> list[WorkflowResult]:
        """Execute inputs while batching slots at identical Workflow steps."""

        materialized = _coerce_inputs(inputs)
        if not materialized:
            return []
        _validate_unique_input_ids(materialized)
        all_slots = tuple(
            _ExecutionSlot(
                input_index=input_index,
                workflow_input=workflow_input,
                branch_index=branch_index,
                current_step=self.workflow.entry_step,
            )
            for input_index, workflow_input in enumerate(materialized)
            for branch_index in range(self.workflow.ensemble_replicas)
        )
        if self._memory is not None:
            for slot in all_slots:
                self._restore_memory_execution(slot)
        active = [slot for slot in all_slots if not slot.completed]

        while active:
            grouped: dict[tuple[str, str, str], list[_ExecutionSlot]] = {}
            for slot in active:
                step = self.workflow.get_step(slot.current_step)
                role = self.workflow.get_role(step.role)
                grouped.setdefault((step.kind, step.id, role.target), []).append(slot)

            produced: dict[tuple[int, int], tuple[WorkflowStepOutput, StepResult]] = {}
            for (kind, _step_id, target), slots in grouped.items():
                if kind == "agent":
                    outputs = self._run_agent_group(target, slots)
                else:
                    outputs = self._run_verifier_group(target, slots)
                for slot, output in zip(slots, outputs, strict=True):
                    record = self._next_step_result(slot, output)
                    if self._step_observer is not None:
                        self._step_observer(record)
                    produced[slot.key] = (output, record)

            next_active: list[_ExecutionSlot] = []
            for slot in active:
                output, record = produced[slot.key]
                self._record_and_advance(slot, output, record=record)
                if not slot.completed:
                    next_active.append(slot)
            active = next_active

        return self._assemble_results(materialized, all_slots)

    def _validate_targets(self) -> None:
        for step in self.workflow.config.steps:
            role = self.workflow.get_role(step.role)
            registry = self._runtimes if step.kind == "agent" else self._verifiers
            noun = "runtime" if step.kind == "agent" else "verifier"
            if role.target not in registry:
                raise ValueError(f"step {step.id!r} targets missing {noun} {role.target!r}")

    def _run_agent_group(
        self,
        target: str,
        slots: Sequence[_ExecutionSlot],
    ) -> list[AgentResult]:
        runtime = self._runtimes[target]
        tasks = [self._agent_task(slot) for slot in slots]
        outputs = runtime.run_batch(tasks)
        if not isinstance(outputs, list):
            raise WorkflowExecutionError(f"runtime {target!r}.run_batch() must return a list")
        if len(outputs) != len(tasks):
            raise WorkflowExecutionError(
                f"runtime {target!r}.run_batch() returned {len(outputs)} results for "
                f"{len(tasks)} tasks"
            )
        for index, (task, output) in enumerate(zip(tasks, outputs, strict=True)):
            if not isinstance(output, AgentResult):
                raise WorkflowExecutionError(
                    f"runtime {target!r} result {index} is not an AgentResult"
                )
            if output.task_id != task.task_id:
                raise WorkflowExecutionError(
                    f"runtime {target!r} changed task identity at index {index}: "
                    f"expected {task.task_id!r}, got {output.task_id!r}"
                )
        return outputs

    def _run_verifier_group(
        self,
        target: str,
        slots: Sequence[_ExecutionSlot],
    ) -> list[VerificationResult]:
        verifier = self._verifiers[target]
        requests = [
            _select_verification_candidate(
                slot,
                self._verification_request(slot),
                verifier=verifier,
            )
            for slot in slots
        ]
        outputs = verifier.verify_batch(requests)
        try:
            return _validate_result_batch(
                requests,
                outputs,
                verifier=f"verifier {target!r}",
            )
        except VerificationContractError as exc:
            raise WorkflowExecutionError(str(exc)) from exc

    def _agent_task(self, slot: _ExecutionSlot) -> AgentTask:
        step = self.workflow.get_step(slot.current_step)
        role = self.workflow.get_role(step.role)
        iteration = slot.step_counts.get(step.id, 0) + 1
        task_id = _invocation_id(slot, step.id, iteration)
        metadata = _invocation_metadata(slot, step.id, iteration, self.workflow.name)
        metadata.update(
            {
                "actor": "solver",
                "role": role.id,
            }
        )
        template = step.input_template
        if template is None:
            template = role.input_template
        if template is None:
            template = "{problem}"
        rendered_prompt = _render_template(
            template,
            slot=slot,
            candidate_from=step.candidate_from,
            iteration=iteration,
            request_id=task_id,
            role_id=role.id,
            model=role.model,
            tools=role.tools,
        )
        # Memory is an agent-facing context layer.  Environments that validate
        # the public task instruction (for example Robotics) must still see
        # the unaugmented prompt when they initialize their episode.
        metadata["environment_user_prompt"] = rendered_prompt
        if self._memory is not None:
            memory_context = self._memory.render_context(
                slot.workflow_input,
                branch_index=slot.branch_index,
                step_id=step.id,
                prompt=rendered_prompt,
            )
            memory_metadata = self._memory.task_metadata(
                slot.workflow_input,
                branch_index=slot.branch_index,
                step_id=step.id,
            )
            if not isinstance(memory_metadata, Mapping):
                raise WorkflowExecutionError(
                    "workflow memory task_metadata() must return a mapping"
                )
            conflicts = sorted(set(metadata).intersection(memory_metadata))
            if conflicts:
                raise WorkflowExecutionError(
                    f"workflow memory task metadata conflicts with executor fields: {conflicts}"
                )
            metadata.update(memory_metadata)
            self._record_memory_provenance(slot)
            if memory_context:
                # Environment-backed runtimes replace `AgentTask.prompt` with the
                # environment-authored observation, so the context also travels
                # through metadata for the runtime to attach to the first model
                # message. `invoke_environment_init` strips it before the
                # environment sees metadata.
                metadata["agent_memory_context"] = memory_context
                rendered_prompt = f"{rendered_prompt}\n\n{memory_context}"
        if role.prompt_ref is not None:
            from alphaapollo.common.prompts import prompt_digest, resolve_prompt

            prompt = resolve_prompt(role.prompt_ref)
            metadata.update(
                prompt_ref=prompt.prompt_ref,
                prompt_version=prompt.version,
                prompt_digest=prompt_digest(role.system_prompt, rendered_prompt),
            )
        return AgentTask(
            task_id=task_id,
            system=role.system_prompt,
            prompt=rendered_prompt,
            model=role.model,
            routing_key=(
                f"{slot.workflow_input.input_id}:{self.workflow.name}:"
                f"{step.id}:iteration-{iteration}"
            ),
            tools=role.tools,
            metadata=metadata,
            task_payload=_thaw_json(slot.workflow_input.task_payload),
            branch_id=f"branch-{slot.branch_index}",
            round_index=iteration - 1,
            sample_id=slot.branch_index,
        )

    def _record_memory_provenance(self, slot: _ExecutionSlot) -> None:
        """Add newly exposed memory-owned executions to the branch history."""

        assert self._memory is not None
        provenance = self._memory.provenance_steps(
            slot.workflow_input,
            branch_index=slot.branch_index,
        )
        slot.history.extend(
            _validated_memory_provenance(
                provenance,
                history=slot.history,
                workflow=self.workflow,
                workflow_input=slot.workflow_input,
                branch_index=slot.branch_index,
            )
        )

    def _restore_memory_execution(self, slot: _ExecutionSlot) -> None:
        """Replay memory-owned durable steps through the configured graph."""

        assert self._memory is not None
        resume_steps = getattr(self._memory, "resume_steps", None)
        if resume_steps is None:
            return
        if not callable(resume_steps):
            raise WorkflowExecutionError("workflow memory resume_steps must be callable")
        recovered = resume_steps(
            slot.workflow_input,
            branch_index=slot.branch_index,
        )
        if not isinstance(recovered, tuple) or not all(
            isinstance(item, WorkflowResumeStep) for item in recovered
        ):
            raise WorkflowExecutionError(
                "workflow memory resume_steps() must return a tuple of WorkflowResumeStep records"
            )
        if not recovered:
            return

        self._record_memory_provenance(slot)
        for index, item in enumerate(recovered):
            if slot.completed:
                raise WorkflowExecutionError(
                    "workflow memory resume steps continue after a terminal Workflow state"
                )
            step = self.workflow.get_step(slot.current_step)
            if item.step_id != step.id:
                raise WorkflowExecutionError(
                    "workflow memory resume step does not match the configured Workflow cursor"
                )
            iteration = slot.step_counts.get(step.id, 0) + 1
            if item.iteration != iteration:
                raise WorkflowExecutionError(
                    "workflow memory resume step has a non-contiguous iteration"
                )
            if step.kind != "agent":
                raise WorkflowExecutionError(
                    "workflow memory may resume only configured agent steps"
                )
            expected_id = _invocation_id(slot, step.id, iteration)
            if item.output.task_id != expected_id:
                raise WorkflowExecutionError(
                    "workflow memory resume output identity does not match its configured step"
                )

            slot.step_counts[step.id] = iteration
            slot.executions += 1
            if slot.executions > self.workflow.execution_limit:
                raise WorkflowExecutionError(
                    "workflow memory resume steps exceed the Workflow execution bound"
                )
            slot.history.append(
                StepResult(
                    input_id=slot.workflow_input.input_id,
                    step_id=step.id,
                    role=step.role,
                    iteration=iteration,
                    branch_index=slot.branch_index,
                    output=item.output,
                )
            )

            if step.output:
                slot.output = item.output
                slot.completed = True
                continue
            if index == len(recovered) - 1 and self._memory_complete(slot):
                slot.output = item.output
                slot.completed = True
                continue
            transition = self._select_transition(slot, item.output)
            if transition is None:
                slot.completed = True
                continue
            transition_index, target = transition
            slot.transition_counts[transition_index] = (
                slot.transition_counts.get(transition_index, 0) + 1
            )
            if target is None:
                slot.completed = True
            else:
                slot.current_step = target

    def _memory_complete(self, slot: _ExecutionSlot) -> bool:
        assert self._memory is not None
        complete = self._memory.is_complete(
            slot.workflow_input,
            branch_index=slot.branch_index,
        )
        if not isinstance(complete, bool):
            raise WorkflowExecutionError("workflow memory is_complete() must return a bool")
        return complete

    def _verification_request(self, slot: _ExecutionSlot) -> VerificationRequest:
        step = self.workflow.get_step(slot.current_step)
        iteration = slot.step_counts.get(step.id, 0) + 1
        request_id = _invocation_id(slot, step.id, iteration)
        candidate_result = _candidate_result(slot, step.candidate_from)
        metadata = _invocation_metadata(slot, step.id, iteration, self.workflow.name)
        if candidate_result is not None:
            runtime_seconds = candidate_result.metadata.get("runtime_seconds")
            if (
                isinstance(runtime_seconds, (int, float))
                and not isinstance(runtime_seconds, bool)
                and runtime_seconds >= 0
            ):
                metadata["candidate_runtime_seconds"] = float(runtime_seconds)
        return VerificationRequest(
            request_id=request_id,
            problem=slot.workflow_input.problem,
            candidate=(
                candidate_result.final_text
                if candidate_result is not None
                else slot.workflow_input.problem
            ),
            metadata=metadata,
            candidate_ref=(candidate_result.task_id if candidate_result is not None else None),
            branch_id=f"branch-{slot.branch_index}",
            round_index=iteration - 1,
            sample_id=slot.branch_index,
            routing_key=(
                f"{slot.workflow_input.input_id}:{self.workflow.name}:"
                f"{step.id}:iteration-{iteration}"
            ),
        )

    def _next_step_result(
        self,
        slot: _ExecutionSlot,
        output: WorkflowStepOutput,
    ) -> StepResult:
        step = self.workflow.get_step(slot.current_step)
        iteration = slot.step_counts.get(step.id, 0) + 1
        return StepResult(
            input_id=slot.workflow_input.input_id,
            step_id=step.id,
            role=step.role,
            iteration=iteration,
            branch_index=slot.branch_index,
            output=output,
        )

    def _record_and_advance(
        self,
        slot: _ExecutionSlot,
        output: WorkflowStepOutput,
        *,
        record: StepResult,
    ) -> None:
        step = self.workflow.get_step(slot.current_step)
        iteration = slot.step_counts.get(step.id, 0) + 1
        slot.step_counts[step.id] = iteration
        slot.executions += 1
        if slot.executions > self.workflow.execution_limit:
            raise WorkflowExecutionError(
                f"workflow execution exceeded its static bound for input "
                f"{slot.workflow_input.input_id!r}, branch {slot.branch_index}"
            )
        slot.history.append(record)
        if self._memory is not None:
            self._memory.observe_output(
                slot.workflow_input,
                branch_index=slot.branch_index,
                step_id=step.id,
                output=output,
            )

        if step.output:
            slot.output = output
            slot.completed = True
            return

        if self._memory is not None:
            if self._memory_complete(slot):
                slot.output = output
                slot.completed = True
                return

        transition = self._select_transition(slot, output)
        if transition is None:
            slot.completed = True
            return
        transition_index, target = transition
        slot.transition_counts[transition_index] = (
            slot.transition_counts.get(transition_index, 0) + 1
        )
        if target is None:
            slot.completed = True
            return
        slot.current_step = target

    def _select_transition(
        self,
        slot: _ExecutionSlot,
        output: WorkflowStepOutput,
    ) -> tuple[int, str | None] | None:
        for index, transition in self.workflow.transitions_from(slot.current_step):
            if transition.max_iterations is not None and (
                slot.transition_counts.get(index, 0) >= transition.max_iterations
            ):
                continue
            if _condition_matches(transition.condition, output):
                return index, transition.target
        return None

    def _assemble_results(
        self,
        inputs: Sequence[WorkflowInput],
        all_slots: Sequence[_ExecutionSlot],
    ) -> list[WorkflowResult]:
        by_input: list[list[_ExecutionSlot]] = [[] for _ in inputs]
        for slot in all_slots:
            by_input[slot.input_index].append(slot)

        assembled: list[WorkflowResult] = []
        for workflow_input, slots in zip(inputs, by_input, strict=True):
            slots.sort(key=lambda slot: slot.branch_index)
            finalized: list[tuple[_ExecutionSlot, WorkflowStepOutput]] = []
            for slot in slots:
                if slot.output is None:
                    continue
                output: object = slot.output
                if self._memory is not None:
                    output = self._memory.finalize_output(
                        workflow_input,
                        branch_index=slot.branch_index,
                        output=output,
                    )
                if not isinstance(output, (AgentResult, VerificationResult)):
                    raise WorkflowExecutionError(
                        "workflow memory finalizer must return an AgentResult or VerificationResult"
                    )
                finalized.append((slot, output))
            outputs = [output for _slot, output in finalized]
            selected: WorkflowStepOutput | None = None
            selected_branch: int | None = None
            selected_step_id: str | None = None
            if outputs:
                strategy = (
                    self.workflow.config.ensemble.strategy
                    if self.workflow.config.ensemble is not None
                    else "first"
                )
                selection_key = (
                    self.workflow.config.ensemble.selection_key
                    if self.workflow.config.ensemble is not None
                    else "exact_text"
                )
                try:
                    selected = select_result(
                        outputs,
                        strategy=strategy,
                        selection_key=selection_key,
                    )
                except ValueError as exc:
                    raise WorkflowExecutionError(
                        f"workflow {self.workflow.name!r} could not select a result for "
                        f"input {workflow_input.input_id!r}: {exc}"
                    ) from exc
                selected_slot = next(slot for slot, output in finalized if output is selected)
                selected_branch = selected_slot.branch_index
                selected_step_id = _source_step_id(
                    selected_slot.history,
                    selected,
                )
            steps = tuple(step for slot in slots for step in slot.history)
            completed = len(outputs) == len(slots)
            assembled.append(
                WorkflowResult(
                    input_id=workflow_input.input_id,
                    workflow_name=self.workflow.name,
                    steps=steps,
                    output=selected,
                    selected_step_id=selected_step_id,
                    selected_branch_index=selected_branch,
                    status="completed" if completed else "incomplete",
                )
            )
        return assembled


def _validated_memory_provenance(
    value: object,
    *,
    history: Sequence[StepResult],
    workflow: Workflow,
    workflow_input: WorkflowInput,
    branch_index: int,
) -> tuple[StepResult, ...]:
    """Validate memory-owned execution evidence before adding it to history."""

    if not isinstance(value, tuple):
        raise WorkflowExecutionError(
            "workflow memory provenance_steps() must return a tuple of StepResult records"
        )
    if not all(isinstance(step, StepResult) for step in value):
        raise WorkflowExecutionError(
            "workflow memory provenance_steps() must return only StepResult records"
        )

    configured_step_ids = {step.id for step in workflow.config.steps}
    steps_by_coordinate = {(step.step_id, step.iteration): step for step in history}
    execution_ids = {_step_execution_identity(step) for step in history}
    new_steps: list[StepResult] = []
    for step in value:
        if step.input_id != workflow_input.input_id:
            raise WorkflowExecutionError(
                "workflow memory provenance input identity does not match its execution slot"
            )
        if step.branch_index != branch_index:
            raise WorkflowExecutionError(
                "workflow memory provenance branch identity does not match its execution slot"
            )
        if step.step_id in configured_step_ids:
            raise WorkflowExecutionError(
                "workflow memory provenance must not impersonate a configured Workflow step"
            )
        coordinate = (step.step_id, step.iteration)
        existing = steps_by_coordinate.get(coordinate)
        if existing == step:
            continue
        if existing is not None:
            raise WorkflowExecutionError(
                "workflow memory provenance contains a duplicate step execution"
            )
        execution_id = _step_execution_identity(step)
        if execution_id in execution_ids:
            raise WorkflowExecutionError(
                "workflow memory provenance contains a duplicate output identity"
            )
        steps_by_coordinate[coordinate] = step
        execution_ids.add(execution_id)
        new_steps.append(step)
    return tuple(new_steps)


def _step_execution_identity(step: StepResult) -> tuple[str, str]:
    if isinstance(step.output, AgentResult):
        return ("agent", step.output.task_id)
    return ("verification", step.output.request_id)


def _source_step_id(
    history: Sequence[StepResult],
    selected: WorkflowStepOutput,
) -> str:
    """Resolve a finalized output back to its original Workflow step.

    Memory finalizers may replace a frozen result to expose an
    Environment-projected value or select a validated memory-owned provenance
    step. Prefer object identity, then use the typed execution identity that
    survives replacement.
    """

    for step in history:
        if step.output is selected:
            return step.step_id
    if isinstance(selected, AgentResult):
        matches = [
            step.step_id
            for step in history
            if isinstance(step.output, AgentResult) and step.output.task_id == selected.task_id
        ]
    else:
        matches = [
            step.step_id
            for step in history
            if isinstance(step.output, VerificationResult)
            and step.output.request_id == selected.request_id
        ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise WorkflowExecutionError(
            "workflow memory finalizer returned an output with ambiguous source identity"
        )
    raise WorkflowExecutionError(
        "workflow memory finalizer returned an output that does not match Workflow history"
    )


def _validate_registry(
    value: Mapping[str, object],
    where: str,
    method: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be a mapping")
    copied: dict[str, object] = {}
    for name, resource in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where} keys must be non-empty strings")
        operation = getattr(resource, method, None)
        if operation is None or not callable(operation):
            raise TypeError(f"{where}[{name!r}] must provide {method}()")
        copied[name] = resource
    return MappingProxyType(copied)


def _coerce_inputs(inputs: Sequence[WorkflowInput]) -> tuple[WorkflowInput, ...]:
    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise TypeError("inputs must be a sequence of WorkflowInput objects")
    materialized = tuple(inputs)
    for index, workflow_input in enumerate(materialized):
        if not isinstance(workflow_input, WorkflowInput):
            raise TypeError(f"inputs[{index}] must be a WorkflowInput")
    return materialized


def _validate_unique_input_ids(inputs: Sequence[WorkflowInput]) -> None:
    seen: set[str] = set()
    for workflow_input in inputs:
        if workflow_input.input_id in seen:
            raise ValueError(f"duplicate WorkflowInput id {workflow_input.input_id!r}")
        seen.add(workflow_input.input_id)


def _invocation_id(slot: _ExecutionSlot, step_id: str, iteration: int) -> str:
    return (
        f"{slot.workflow_input.input_id}:branch-{slot.branch_index}:{step_id}:iteration-{iteration}"
    )


def _invocation_metadata(
    slot: _ExecutionSlot,
    step_id: str,
    iteration: int,
    workflow_name: str,
) -> dict[str, object]:
    return {
        **_thaw_json(slot.workflow_input.metadata),
        "workflow_input_id": slot.workflow_input.input_id,
        "workflow_name": workflow_name,
        "workflow_step_id": step_id,
        "branch_index": slot.branch_index,
        "iteration": iteration,
    }


def _render_template(
    template: str,
    *,
    slot: _ExecutionSlot,
    candidate_from: str | None,
    iteration: int,
    request_id: str,
    role_id: str,
    model: str | None,
    tools: Sequence[str],
) -> str:
    previous = slot.history[-1].output if slot.history else None
    values = {
        "branch_index": str(slot.branch_index),
        "candidate": _candidate_text(slot, candidate_from),
        "feedback": _latest_feedback(slot),
        "input_id": slot.workflow_input.input_id,
        "iteration": str(iteration),
        "metadata": json.dumps(
            dict(slot.workflow_input.metadata), ensure_ascii=False, sort_keys=True
        ),
        "model": model or "",
        "previous_output": _output_text(previous),
        "problem": slot.workflow_input.problem,
        "request_id": request_id,
        "role": role_id,
        "tools": json.dumps(tuple(tools), ensure_ascii=False),
    }
    try:
        return template.format_map(values)
    except (KeyError, ValueError) as exc:  # Config validates templates; keep a clear guard.
        raise WorkflowExecutionError(f"could not render step template: {exc}") from exc


def _candidate_text(slot: _ExecutionSlot, candidate_from: str | None) -> str:
    result = _candidate_result(slot, candidate_from)
    return result.final_text if result is not None else slot.workflow_input.problem


def _candidate_result(
    slot: _ExecutionSlot,
    candidate_from: str | None,
) -> AgentResult | None:
    if candidate_from is not None:
        for result in reversed(slot.history):
            if result.step_id == candidate_from:
                if not isinstance(result.output, AgentResult):
                    raise WorkflowExecutionError(
                        f"candidate source {candidate_from!r} did not produce AgentResult"
                    )
                return result.output
        raise WorkflowExecutionError(
            f"candidate source {candidate_from!r} has not run on input "
            f"{slot.workflow_input.input_id!r}, branch {slot.branch_index}"
        )
    for result in reversed(slot.history):
        if isinstance(result.output, AgentResult):
            return result.output
    return None


def _select_verification_candidate(
    slot: _ExecutionSlot,
    request: VerificationRequest,
    *,
    verifier: Verifier,
) -> VerificationRequest:
    """Resolve a stateful Verifier's candidate choice from this branch history."""

    selected_ref = verifier.select_candidate_ref(request)
    if selected_ref == request.candidate_ref:
        return request
    if not isinstance(selected_ref, str) or not selected_ref.strip():
        raise WorkflowExecutionError(
            "verifier select_candidate_ref() must preserve the current reference "
            "or return a non-empty prior AgentResult task_id"
        )
    for item in reversed(slot.history):
        output = item.output
        if isinstance(output, AgentResult) and output.task_id == selected_ref:
            return replace(
                request,
                candidate=output.final_text,
                candidate_ref=output.task_id,
            )
    raise WorkflowExecutionError(
        "verifier select_candidate_ref() returned a reference outside the current "
        f"Workflow branch history: {selected_ref!r}"
    )


def _latest_feedback(slot: _ExecutionSlot) -> str:
    for result in reversed(slot.history):
        if isinstance(result.output, VerificationResult):
            return result.output.feedback
    return ""


def _output_text(output: WorkflowStepOutput | None) -> str:
    if output is None:
        return ""
    if isinstance(output, AgentResult):
        return output.final_text
    return output.feedback


def _condition_matches(condition: str, output: WorkflowStepOutput) -> bool:
    if condition == "always":
        return True
    if not isinstance(output, VerificationResult):
        return False
    verdict = output.verdict
    if condition == "passed":
        return verdict == "pass"
    if condition == "failed":
        return verdict == "fail"
    if condition == "inconclusive":
        return verdict == "inconclusive"
    if condition == "not_passed":
        return verdict != "pass"
    return False
