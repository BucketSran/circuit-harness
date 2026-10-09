# Issue tracker

Use GitHub Issues and PRs in the public repository
[BucketSran/circuit-harness](https://github.com/BucketSran/circuit-harness).
The integration branch is `main`; verify the actual remote and base before publication.
PRs as an external request surface: off.

## Work and completion

Start with the [roadmap](../development/ROADMAP.md), [next work](../development/NEXT_WORK.md) and the
current public issue for the task. Reuse accepted scope instead of creating duplicate trackers.
Pre-publication issues and PRs remain in `circuit-harness-private` and require its access
permission. Their numbers are not public issue IDs; never use them as unqualified closing references.

- A spec defines a capability and its acceptance; a ticket defines an independently
  verifiable delivery unit with actual blocking dependencies.
- Normally one ticket maps to one PR. A small spec can be the delivery unit without another issue.
- Inseparable changes may share an integration PR with explicit dependencies. Dependent work
  need not become one large spec PR merely because `implement-spec` supports that mode.
- Keep review fixes in the existing PR. Close only the work whose acceptance is complete.
  A merged child PR does not close its parent spec or outstanding real-environment acceptance.

Use `ready-for-agent` for accepted specs/tickets once the label is configured. Read blockers and
acceptance before starting; the label alone does not mean external dependencies are available.
Preserve existing labels. Other triage roles are not configured by this workflow; inspect
and agree their mapping if a later task invokes automatic triage.

## Operations

Use the repository's issue/PR templates and
[prepare-contribution](../../.agents/skills/prepare-contribution/SKILL.md).
Search both open and closed work for a new contribution; reuse that search for the same scope.
Read full bodies and relevant discussion before updating. Every PR body creation/update uses
shared `pr`, mapped into the project template.

Publication and human review follow the [delivery policy](../development/DEVELOPMENT_SOP.md#delivery-and-review).
Use exact issue references in closing keywords only when completion is justified. Preserve
raw evidence and machine-specific information in private storage; publish safe summaries.

Keep unpublished drafts and `show-me-your-work` logs in ignored `.planning/chips/`.
An execution log links decisions to evidence; it is not another scope/progress tracker.
