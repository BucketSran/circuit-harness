---
name: review-pr
description: Review Circuit Harness changes against their spec and repository contracts using shared code-review. Resolve actual revisions, validate concrete findings and keep review-only requests read-only.
---

# Review Harness changes

Use shared `code-review` for the Standards and Spec axes, with the local adaptations below.
Read the [review checklist](references/review-checklist.md) for affected contracts, not as
a requirement to run every listed check.

## Fix the review scope

Read the task, PR/Issue and repository instructions. Resolve the actual base/head from
available Git/PR metadata before asking for missing information; the default integration
target is `main` in `BucketSran/circuit-harness`. Use the [tracker configuration](../../../docs/agents/issue-tracker.md).
Review a fixed revision or explicitly identified local diff. A commit-only diff omits
unstaged and untracked work; include intended local files when reviewing a workspace.

For substantive behavior changes, provide independent fresh reviewers with the Standards
and Spec inputs, exact revisions/diff and relevant evidence. Do not give them the implementer's
conclusions as facts. The shared `code-review` skill owns the two-axis method. Small document
edits use proportional factual/link/diff review; this is not a mandatory multi-agent exercise.

## Check the evidence

Read the changed paths and relevant callers yourself. Validate findings against the base,
requirements and a reachable failure scenario. The checklist's false-positive filter applies.
Distinguish documented contract violations and defects from labelled design suggestions;
a generic code-smell heuristic alone is not a merge blocker.

Run affected checks when they help resolve a finding. Preserve the actual host/model/budget
authorization and report external checks not run. Reuse existing evidence for the same
revision and conditions; new changes, failures or unresolved findings justify additional review.
Include the `show-me-your-work` trail in substantial unattended review when one exists.

## Report or repair within scope

Present Standards and Spec findings separately, with concrete locations, impacts and evidence.
State what was inspected, checks actually run and coverage limits; no findings is not proof
of untested lab/model behavior. The coordinator validates and prioritizes actionable findings.

A review request inspects and reports. Repair, publication and merging follow the current
task's authorization and [delivery policy](../../../docs/development/DEVELOPMENT_SOP.md#delivery-and-review).
Implementation tasks include fixing actionable findings, checking the affected behavior and
refreshing the reviewed revision. Ordinary completion does not trigger the three-round hardening skill.
