# Circuit Harness agent guide

Circuit Harness is a circuit experiment platform. Harbor owns Agent installation,
Job scheduling and Trial execution. This repository owns benchmark adapters,
simulator access, public task sessions, independent verification and evidence.
vaEVAS owns EVAS algorithms and its benchmark tasks/checkers; course materials
stay in their course project. Preserve circuit protocols and serialized evidence.

## Start here

Read the request, accepted decisions, Git status and the affected component entry.
For stage selection or cross-repository work, load
[harness-workflow](.agents/skills/harness-workflow/SKILL.md).
Known component tasks can enter their skill directly. Small fixes and document
edits do not need a new grill/spec/ticket sequence.

Use the [documentation index](docs/README.md) to locate current contracts and the
[development SOP](docs/development/DEVELOPMENT_SOP.md#development-workflow) for delivery.
Default delivery is a reviewable PR. User approval authorizes merge and daily
checkout synchronization. An explicit request to complete and merge authorizes
that full path. Discussion, review and saved-evidence analysis retain their scope.

## Skills

Shared Matt and pstack skills supply general methods; project skills add local
contracts. Read applicable sources from the client's actual catalog. A reference
alone does not establish that a skill ran or a remote operation happened.

| Work | Entry |
| --- | --- |
| Select a stage or coordinate dependencies | [harness-workflow](.agents/skills/harness-workflow/SKILL.md) |
| Implement, choose checks, validate or reanalyze evidence | [chips-dev](.agents/skills/chips-dev/SKILL.md) |
| Prepare or publish an issue/PR | [prepare-contribution](.agents/skills/prepare-contribution/SKILL.md), shared `pr` for every PR body |
| Review requirements and repository contracts | [review-pr](.agents/skills/review-pr/SKILL.md) |
| Change ownership, interfaces or execution structure | [write-clear-code](.agents/skills/write-clear-code/SKILL.md) |

Use [review-fix-loop](.agents/skills/review-fix-loop/SKILL.md) for explicitly
requested repeated hardening. Reuse settled scope and evidence from the
[tracker](docs/agents/issue-tracker.md) and [domain contracts](docs/agents/domain.md).

## Ownership

- `circuit_harness/benchmarks/`: benchmark preparation, replay and operator CLIs.
- `circuit_harness/execution/`: simulators, sessions, transport, freezing and evidence.
- `circuit_harness/harbor/`: Agent/Model configuration and Harbor environment/verifier plugins.
- `circuit_harness/data/`: ATIF preparation and external-trainer dataset interfaces.
- `circuit_harness/reporting/`: offline saved-episode reports.
- `circuit_harness/cli.py`, `public_mcp.py`, `task_authoring.py`: operator and public protocol entries.
- `examples/analogbench/`, `examples/evas-va07/`: benchmark recipes and runtime assets.
- `docs/`, `tests/`, `examples/chips/`: contracts, checks and safe templates.

Preserve public APIs, CLI behavior, configuration defaults, fixed VABench r53
protocols and saved evidence unless the task explicitly changes them. Training
engines remain external. Select native Harbor tasks or Harness public sessions
using the [integration guide](docs/guides/integration.md#2-选择任务执行方式).

## Checks and evidence

Choose affected checks through [chips-dev](.agents/skills/chips-dev/SKILL.md) and
[tests/chips](tests/chips/README.md). Behavior changes use shared `tdd` at agreed
public boundaries; documentation uses factual, link and diff checks. Schema
changes regenerate adjacent JSON Schema artifacts.

Distinguish fixtures, real simulator execution, model Trials and independent
scores. Report actual checks, skips and limitations. Live execution requires the
task's actual host, model and budget authorization. Preserve native Agent tool
restrictions and the original Harbor path without forcing a tool-loop migration.

## Workspaces and publication

Follow the SOP's [workspace](docs/development/DEVELOPMENT_SOP.md#workspace-handoff) and
[cross-repository](docs/development/DEVELOPMENT_SOP.md#shared-backend-workflow) rules.
Verify the checkout, remote and unrelated changes before editing. Target
`BucketSran/circuit-harness`, PR base `main`; distinguish contributor forks.
Transfer reviewed source changes from private checkouts without merging private history.

Drafts and decision logs stay in ignored `.planning/chips/`. Raw prompts, trajectories,
simulator output and actual host configuration stay in private run storage.
Commit maintained contracts and safe fixtures under the
[documentation policy](docs/development/DEVELOPMENT_SOP.md#documentation-policy).
Verify contribution checkboxes and the [publication boundary](docs/development/REPOSITORY_SCOPE.md#publication-boundary).
Source cleanup does not authorize visibility changes, history rewrites or external data release.
