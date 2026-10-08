---
name: write-clear-code
description: Apply Circuit Harness ownership and compatibility rules when changing modules, public interfaces, configuration or execution lifecycles. Use shared codebase-design for general module design.
---

# Keep Harness contracts clear

Use shared `codebase-design` for module boundaries and `blast-radius` when a shared contract
or lifecycle changes. This skill adds the repository-specific constraints; concrete examples
are in [patterns](references/patterns.md).

- Keep execution, task grading and Agent orchestration with their actual owners. Lower-level
  common modules must not import product workflows to reach shared behavior.
- Preserve public import paths, CLI/config defaults and serialized records unless the task
  explicitly includes migration. A facade can preserve imports while implementation moves.
- Before moving execution code, inspect cancellation, exception timing, process groups,
  event loops, signal/thread restrictions, repeated calls and monkeypatch/import consumers.
- Resolve simulator/config/resource identity once and record what was actually consumed.
  Unknown or unsupported options must not look successfully honored.
- Freeze inputs before remote or asynchronous execution. Keep stable action/job identities;
  client disconnection is not proof of cancellation, and unknown state is not permission to retry.
- Keep missing results, infrastructure errors and valid task failure distinguishable. The
  benchmark owns the score; evidence formatting must not turn absence into success or zero.
- Inspect every consumer before consolidating shared code/resources. Preserve behavior when
  two paths differ; changing semantics is an explicit change rather than incidental cleanup.
- Preserve the pinned training submodule and optional-import boundaries. Derive supported
  Python/dependency conditions from current manifests and CI instead of old examples.

Verify the affected behavior using [chips-dev](../chips-dev/SKILL.md). Local process tests,
safe fixtures and authorized real runs prove different claims. Choose the evidence required
by the task; mentioning a live server here does not authorize contacting it.
