---
name: chips-dev
description: Implement or validate Circuit Harness simulator adapters, task sessions and evidence collection. Also select checks or reanalyze saved runs without expanding them into live execution.
---

# Develop and validate Harness behavior

Read the affected component contract, [SOP](../../../docs/chips/DEVELOPMENT_SOP.md#development-workflow)
and [test entry](../../../tests/chips/README.md). Preserve the task's operation: selecting
checks does not run them, and reanalysis consumes saved evidence without a new model or simulator run.

## Implement

Use shared `tdd` for changed behavior at the highest useful existing public entry. Reuse
already-agreed test boundaries; ask only for unresolved circuit or execution decisions.
Apply [write-clear-code](../write-clear-code/SKILL.md) when ownership or execution structure changes.
Keep public APIs, serialized evidence and the pinned VABench path compatible.

Test actual process/file/session boundaries: snapshot identity, timeouts, cleanup, repeated
actions, uncertain remote state, frozen candidates and artifact integrity as relevant.
A constructed simulator or network fixture proves the protocol it exercises, not lab operation.
Keep module test paths and use `tests/chips` for acceptance navigation. Documentation changes
need document checks; adding regression coverage to existing code is not a historical Red phase.

## Validate the requested claim

Use the [case catalog](../../../tests/chips/CASES.md) and
[record template](../../../tests/chips/RECORD_TEMPLATE.md) for lab acceptance or comparisons.
Choose the smallest relevant checks and expand for actual shared risk. Default CI is local;
a case's existence or an available credential does not authorize a live run.

Record source, candidate, configuration, task/checker and actual backend identities.
Distinguish host observations, protocol fixtures, real simulator execution, real Agent flow
and independently graded task success. An exit code of zero does not establish all of them.
Candidate errors follow the benchmark contract; infrastructure failure is not a model zero score.

Harness owns execution and evidence. EVAS algorithms/semantics and benchmark grading belong
to vaEVAS; course materials stay in their project. For cross-repository changes, use the
[SOP](../../../docs/chips/DEVELOPMENT_SOP.md#shared-backend-workflow) and verify the combined call path.

## Deliver

Report actual checks, failures, skips, evidence limitations and pending external acceptance.
Use [review-pr](../review-pr/SKILL.md) for the required review and
[prepare-contribution](../prepare-contribution/SKILL.md) for publication.
Follow the [delivery policy](../../../docs/chips/DEVELOPMENT_SOP.md#delivery-and-review);
reuse review evidence for the same revision instead of stacking complete review workflows.
