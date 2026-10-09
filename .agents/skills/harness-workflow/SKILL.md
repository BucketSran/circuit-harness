---
name: harness-workflow
description: Select or resume Circuit Harness work across development stages, simulators or repositories. Route to project contracts and shared Matt/pstack skills, preserving task scope, evidence and the PR review boundary.
---

# Coordinate Harness work

Use this entry when the next stage or owner needs coordination. A known component task
can enter its skill directly.

## Recover the task

Read the current request, accepted decisions, Git status and relevant Issue/PR. Find the
owning component through [AGENTS](../../../AGENTS.md) and [domain docs](../../../docs/agents/domain.md).
Distinguish discussion, implementation, review, check selection, execution and evidence reanalysis.
Reuse agreed test boundaries and prior authorization within the same task and resource limits.

Check the actual checkout, branch and unrelated changes using the
[workspace policy](../../../docs/development/DEVELOPMENT_SOP.md#workspace-handoff).
For cross-repository work, follow the [shared backend contract](../../../docs/development/DEVELOPMENT_SOP.md#shared-backend-workflow).

## Choose the next stage

| Need | Entry |
| --- | --- |
| Goals or acceptance remain unsettled | Requested `grill-with-docs`, using `grilling` and `domain-modeling` |
| Record an agreed outcome or split dependent work | Requested `to-spec` / `to-tickets`; use existing [tracker conventions](../../../docs/agents/issue-tracker.md) |
| Implement a known behavior | [chips-dev](../chips-dev/SKILL.md), shared `implement` and `tdd` when selected |
| Coordinate a ticket graph | Selected `implement-spec`, adapted to the SOP's PR units, actual dependencies and workspace ownership |
| Choose/run checks or reanalyze saved evidence | [chips-dev](../chips-dev/SKILL.md); retain the requested operation |
| Review | [review-pr](../review-pr/SKILL.md), shared `code-review` |
| Prepare/update an issue or PR | [prepare-contribution](../prepare-contribution/SKILL.md); every PR uses shared `pr` |

Load shared skills from the actual client catalog when applicable, respecting invocation settings.
A missing optional skill is not a reason to initialize an unrelated tracker or restart settled
design. Apply available project contracts and report any method that could not be used.
Never claim an unavailable skill ran. The [SOP](../../../docs/development/DEVELOPMENT_SOP.md#development-workflow)
defines the local adaptations to shared defaults.

Use pstack methods for their specific need: `blast-radius` for consequential shared changes,
`benchmark-checklist` for measured performance, `correct` for evidenced recurrence,
`show-me-your-work` for substantial multi-stage/unattended work, and `unslop` for prose.
For documentation, locate its owner through the [documentation index](../../../docs/README.md).
Use `writing-for-agents` for instructions, `technical-writing` for human-facing docs,
and `diagnosing-bugs` for a debugging task. These are conditional entries, not a checklist
to invoke for every change.

## Finish the requested stage

The [delivery policy](../../../docs/development/DEVELOPMENT_SOP.md#delivery-and-review) governs PR,
human review, merging and daily-checkout synchronization. Normal development ends at a
reviewable PR pending human review; explicit autonomous-merge authorization continues through integration.
Report completed checks, unresolved findings, external acceptance still pending, and where
the result is visible. For authorized publication, update the existing Issue/PR; for a
development handoff, update the temporary-worktree navigation when one was used.
A merged child ticket does not complete a parent spec with outstanding acceptance.
