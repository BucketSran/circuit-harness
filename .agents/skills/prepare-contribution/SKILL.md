---
name: prepare-contribution
description: Prepare, check, create or update Circuit Harness issues and PRs using repository templates, shared pr guidance and actual validation evidence. Preserve the requested stage and the human merge-review boundary.
---

# Prepare a Harness contribution

Read the current task, [tracker conventions](../../../docs/agents/issue-tracker.md) and the
relevant [.github template](../../../.github). For updates, read the current artifact,
discussion, labels and related work. Reuse related-work searches for the same unchanged
scope; search open and closed issues/PRs when proposing new scope or publishing a new contribution.

## Scope the artifact

Use the actual base/head and intended diff, including relevant uncommitted/untracked files
when preparing local work. Check dependencies and ownership before staging.
A ticket normally maps to one independently verifiable PR; small specs need no duplicate ticket.
Review fixes continue in the existing PR. Link a parent spec without closing it unless all
of its acceptance is complete. See the [PR policy](../../../docs/development/DEVELOPMENT_SOP.md#delivery-and-review).

Preserve discussion and review-only scope. For an authorized development delivery, prepare,
commit, push and create/update its PR under that policy; do not stop at a local draft.
For an issue/PR writing request, perform only the requested draft, check or publication action.
Existing authorization remains valid within the task's target and resource limits.

## Write the PR with shared pr

Every PR creation or body update must load and use shared `pr`, then apply `unslop`.
The project template supplies the structure; map the shared skill's content into it:

- Summary and Changes explain the concrete problem and resulting behavior. Use a small
  diagram/diff sketch only when it helps the reader.
- Validation carries before/after evidence and commands actually run. Document-only work
  uses document checks instead of invented failing tests.
- Risk and compatibility explains affected contracts, reversibility and remaining limits.
- Related work and Scope identify the governing task, dependencies and explicit non-goals.
- Check a template item only when its statement is true.

Scale detail to the change. A substantial multi-stage task links its `show-me-your-work`
trail when the evidence is accessible and safe to share; raw logs and private paths remain local.
Do not copy a second template into the body or require meaningless diagrams for small changes.
If the required `pr` source is unavailable, preserve the prepared work and report that
specific missing dependency rather than claiming it was used.

## Publish and hand off

Verify the target repository `BucketSran/circuit-harness`, PR base `main`, remote ownership,
branch identity and relevant checks. Never push private development history into the public repository. Use
structured bodies or a body file. Read back the published title/body/base and attach the PR
to the current chat when the host provides that capability.

Return the PR link, the reviewable revision, key checks and pending acceptance. By default
leave it open for human review. After confirmation, or under explicit autonomous-merge
authorization, verify the current head/checks and continue integration and
[workspace handoff](../../../docs/development/DEVELOPMENT_SOP.md#workspace-handoff).
New material changes after human approval require review of the affected change; do not
use approval of an older result to merge a different result silently.
