---
name: review-fix-loop
description: Repeatedly harden an existing PR with three independent clean reviews of one final revision, only when explicitly requested.
---

# Repeated PR hardening

This is an explicit special mode. Ordinary implementation, PR preparation or autonomous
completion uses the normal [review](../review-pr/SKILL.md) and
[delivery policy](../../../docs/development/DEVELOPMENT_SOP.md#delivery-and-review).

Resolve the requested PR, base/head, workspace, checks and authorized actions. Use shared
`show-me-your-work` for the decision trail in the project's ignored planning storage.
Reuse the project review contract and shared methods rather than creating a second ledger format.

## Review and repair

Review the complete change from its base. Consider correctness/failure handling, integration
and compatibility, then operational/artifact readiness. Require concrete actionable findings,
not new speculative requirements. Repair within scope, run affected checks, update the PR
when authorized and reset the clean streak.

Three independent substantive clean rounds must inspect the same final head. Distinct lenses
do not mean repeating identical reviews. Reset when code or relevant evidence changes,
a finding or failed check appears, or a base change affects the result. A test rerun is not
an independent review. If a concrete finding cannot be resolved within scope, report it;
do not count it as clean.

Before handoff, verify that the reviewed revision equals the remote PR head and that required
checks still apply. Reuse already-run checks on that exact revision unless new evidence warrants
rerunning them. Account for unrelated workspace changes without staging or deleting them.
Report review coverage, repairs, validation and remaining external limits.

The three clean rounds do not authorize merging. Follow the task's human-review or explicit
autonomous-merge policy. When asked for a goal prompt instead of execution, describe this
scope and completion condition without starting the loop.
