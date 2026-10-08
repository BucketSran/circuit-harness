# Circuit Harness agent guide

This repository develops circuit-harness, a circuit experiment platform derived from AlphaApollo.
Harness owns simulator access, public task sessions, evaluation integration and evidence
collection. vaEVAS owns EVAS algorithms and benchmark tasks/checkers; course content stays
in its course project. Preserve circuit protocols, serialized evidence and the pinned VABench path.
Apollo runtime integration is retired; use Harbor for Agent/Job/Trial ownership.

## Start here

Read the current request, accepted decisions, Git status and the affected component entry.
For work spanning stages or repositories, use [harness-workflow](.agents/skills/harness-workflow/SKILL.md).
Known implementation, validation and review tasks can use their entry directly.
Small fixes and document edits do not need a new grill/spec/ticket sequence.

The [development SOP](docs/chips/DEVELOPMENT_SOP.md#development-workflow) defines stage selection,
review and delivery. Default development delivery is a reviewable PR; merge and synchronize
the daily checkout after the user confirms. An explicit request to complete and merge
authorizes that full path without another confirmation. Discussion and review retain their scope.

## Agent skills

Shared Matt and pstack skills provide general methods; project skills supply local contracts.
Resolve shared skills through the client's installed catalog and read the applicable source.
Follow explicit invocation settings and the user's selected workflow; a reference is not
proof that a skill was loaded or that a remote action happened.

| Work | Project entry |
| --- | --- |
| Select/resume a stage, coordinate dependencies or cross-repository work | [harness-workflow](.agents/skills/harness-workflow/SKILL.md) |
| Implement, select checks, validate or reanalyze execution evidence | [chips-dev](.agents/skills/chips-dev/SKILL.md) |
| Prepare or publish an issue/PR; use shared `pr` for every PR body/update | [prepare-contribution](.agents/skills/prepare-contribution/SKILL.md) |
| Review against requirements and repository rules | [review-pr](.agents/skills/review-pr/SKILL.md) |
| Change module ownership, public interfaces or execution structure | [write-clear-code](.agents/skills/write-clear-code/SKILL.md) |

Use [review-fix-loop](.agents/skills/review-fix-loop/SKILL.md) only for explicitly requested
repeated hardening. Ordinary autonomous delivery does not select that loop.
Planning/review skills use the [issue tracker](docs/agents/issue-tracker.md) and
[domain documents](docs/agents/domain.md). Reuse existing scope and evidence.

## Ownership and compatibility

- `circuit_harness/benchmarks/`: benchmark-specific preparation, replay and operator CLIs.
- `circuit_harness/execution/`: simulators, public sessions, transport and evidence.
- `circuit_harness/harbor/`: Harbor environment/verifier plugins, Agent/Model configuration and independent verification.
- `examples/analogbench/`, `examples/evas-va07/`: complete benchmark recipes and their runtime assets.
- `docs/guides/integration.md`: developer integration guide.
- `circuit_harness/data/`: ATIF preparation and external-trainer dataset interfaces.
- `circuit_harness/reporting/`: offline saved-episode reports.
- `circuit_harness/cli.py`, `public_mcp.py`, `task_authoring.py`: operator and public protocol entries.
- `tests/`, `docs/`, `examples/`: maintained checks, contracts and safe templates.

The public `main` branch contains the independent `circuit_harness` package. Apollo's
runtime, generic Workflow, Robotics, Memory and built-in trainers are retired.
Use the [migration guide](docs/chips/MIGRATION.md) when adapting old callers; new interfaces
must preserve existing circuit semantics unless explicitly changed. Training engines stay
external, with no pinned trainer submodule in this repository.

## Checks and evidence

Use [chips-dev](.agents/skills/chips-dev/SKILL.md) and the [test entry](tests/chips/README.md)
to select affected checks. Behavior changes use shared `tdd` at agreed public boundaries;
document-only changes need factual, link and diff checks. Expand testing for actual shared
risks, not unrelated Bio/Robotics/Math domain acceptance. Schema changes include regenerated
adjacent JSON Schema artifacts. Report actual checks, skipped checks and their limits.

Local fixtures do not certify a real simulator, license, model or independently graded circuit.
Use the task's actual host, model and budget authorization for live work. Preserve native-tool
restrictions; preserve the native Harbor path without forcing a tool-loop migration.

## Workspaces and records

Follow the SOP's [workspace](docs/chips/DEVELOPMENT_SOP.md#workspace-handoff) and
[cross-repository](docs/chips/DEVELOPMENT_SOP.md#shared-backend-workflow) rules.
Keep the daily checkout discoverable, give temporary worktrees a visible entry, and preserve
unrelated work. Target PRs at `BucketSran/circuit-harness` with base `main`; verify whether
`origin` is this repository or a contributor fork. Never push to the original AlphaApollo upstream.
The old `circuit-harness-private` checkout contains private history: do not retarget its remote
or merge its history into this public repository. Transfer reviewed source changes only.

Scope/dependencies live in the existing Issue/PR. Drafts and decision logs stay in ignored
`.planning/chips/`; raw prompts, trajectories, simulator output and host configuration stay
in private run storage. Follow the [documentation policy](docs/chips/DEVELOPMENT_SOP.md#documentation-policy).
Commit maintained contracts and safe fixtures only. Verify template checkboxes, avoid private
data in publication, and preserve needed ignored material before retiring a worktree.
For public-source preparation, follow the [publication boundary](docs/chips/REPOSITORY_SCOPE.md#publication-boundary).
Prepare a reviewed source snapshot while keeping development history private; a cleanup PR
does not authorize a visibility change, history rewrite or external benchmark/data release.
