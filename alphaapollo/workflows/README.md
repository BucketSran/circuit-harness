# Workflows

This is the shared Apollo runtime documentation. Its AIME and other domain
recipes remain in private history and backups; commands
pointing to those examples are not runnable from this Chips-only example tree.
For a runnable entry on this branch, use [Chips examples](../../examples/chips/README.md).

This package owns configuration-driven orchestration above Reasoning. A
`WorkflowConfig` defines roles, steps, and transitions; a `RunConfig` binds that
preset to prepared data, runtimes, verifiers, an Environment, and persistence.

## Package layout

```text
alphaapollo/workflows/
├── __init__.py       # Small public API: config, records, executor, and run_workflow.
├── main.py           # Canonical CLI shim; delegates parsing and execution to run.py.
├── run.py            # Application runner: one run or repeated/resumable scored cells.
├── config.py         # Immutable config models, Hydra composition, and static graph checks.
├── resources.py      # Public resource API, cross-resource checks, and composition.
├── _resources/       # Shared implementation; each family validates what it constructs.
│   ├── runtime.py    # Runtime options, external sessions, tool grants, and model backends.
│   ├── verifier.py   # Verifier options, role binding, and construction.
│   ├── environment.py # Generic Environment validation, construction, and response adaptation.
│   └── lifecycle.py  # ExecutionResources, ownership, and ordered shutdown/rollback.
├── data.py           # Public input loading, private-gold alignment, and durable projections.
├── records.py        # Workflow input/result records and repeated-run scoring records.
├── executor.py       # Executes one Workflow graph over public WorkflowInput values.
├── memory/           # The memory subsystem behind one package boundary.
│   ├── __init__.py   # Core working/persistent memory; the public entry point (#311).
│   ├── adapter.py    # The adapter protocol every domain projection implements.
│   ├── unified.py    # One adapter over verification and Robotics projections (#352).
│   ├── robotics.py   # Robotics projection.
│   ├── verification.py # Verification memory.
│   └── _journal.py   # Crash-tolerant JSONL parsing shared by Workflow memory.
├── scoring.py        # Runs one seeded cell, then grades and reduces metrics out of loop.
├── selection.py      # Declared-answer extraction, equivalence, and ensemble selection.
├── report.py         # Aggregates RunRecord values into JSON and Markdown reports.
├── visualize.py      # Renders run directories as a self-contained HTML trajectory viewer.
├── chips.py                  # Direct simulator CLI; included in the offline server bundle.
├── chips_episode_report.py   # Offline Chips evidence projection and private report CLI.
├── _chips_episode_report/    # Internal implementation of the Chips report CLI.
│   ├── evidence.py           # Candidate/action/archive linkage for the Chips report.
│   └── view.py               # Chips rendering using the shared viewer primitives.
├── chips_experiment.py       # Stable one-episode CLI and run/collect/finalize/archive entry.
├── chips_evaluate.py         # Fixed batch preparation, serial run/reconcile, and task/condition summary.
├── _chips_experiment/        # Internal implementation of the Chips experiment CLI.
│   ├── core.py               # Plan identity, output manifest and result persistence.
│   └── tasks.py              # VABench/Analog capabilities and task-owned lifecycle rules.
├── chips_experiment_settings.py # Shared experiment settings snapshot for task runners.
├── chips_pi_runtime.py       # Pi launch and optional Apollo Runtime bridge shared by tasks.
├── chips_vabench_agent.py    # Public VABench Pi/Codex tools and episode entry.
├── chips_vabench_record.py   # VABench one-cell result ledger.
├── chips_vabench_deployment.py # Private VABench operator configuration entry.
├── chips_analog_agent.py     # Public Analog RLC Pi tools and episode entry.
├── chips_task_authoring.py   # Review and confirmation of task drafts.
├── chips_task_authoring_agent.py # Bounded gain-task authoring Agent entry.
├── chips_public_mcp.py       # Public EVAS MCP broker outside the native Agent sandbox.
├── harbor_chips/             # Optional Harbor trial, native Agent and final verifier adapters.
```

The Chips entry points are separate on purpose: `chips.py` runs direct simulator
commands and is packaged for the server; `chips_experiment.py` runs or resumes
one configured Agent experiment; `chips_episode_report.py` reads saved evidence
without starting an experiment. Task-specific Agent entries choose their public
tools and prompts. The private `_chips_*` packages implement their adjacent
entry points and are not separate command or import contracts. Simulator jobs,
task sessions, and private scoring live in
[`common/execution/chips/`](../common/execution/chips/README.md); runnable task
definitions and operator examples live in [`examples/chips/`](../../examples/chips/README.md).

The optional [Harbor adapter](../../docs/chips/HARBOR.md) uses Harbor's trial
lifecycle to run one native Codex attempt, freeze the public candidate and collect
an independent final result. It does not run a second agent loop. Public session
tools are served by `chips_public_mcp.py`; final evaluation remains operator-owned.

Reusable topology presets live in `alphaapollo/configs/preset/` beside the
other Hydra config groups, rather than inside the execution package.

`config.py` owns the recipe schemas and primitive configuration guards.
`data.py` holds both ends of the public/private boundary: the metadata allowlist
stays beside the loader that produces the row it narrows. Their module docstrings
and numbered section markers provide a navigation map.

### Resource composition and downstream synchronization

Import `ExecutionResources`, `validate_composition_config`, and
`compose_resources` from `alphaapollo.workflows.resources`. The public entry
point validates the complete configuration before acquiring resources, resolves
Verifier roles and Runtime tool grants, then constructs the Workflow, Runtimes,
and Verifiers. An Environment factory creates an owned Environment per task;
`ExecutionResources.environment_factory` is available to application adapters.
Partial construction rolls back owned resources in reverse order. Closing the
returned context is idempotent and preserves the original exception on failure.

The four internal modules depend on configuration, records, and lower-level
implementations, never on the public resource entry point. Runtime and Verifier
builders use the shared lifecycle helpers. Composition passes existing configs,
resolved role specifications, and tool catalogs across these boundaries.

For Chips and Robotics branch synchronization, port changes to the owner
shown below and patch that module directly in tests:

| Previous implementation in `resources.py` | Owner |
| --- | --- |
| Runtime validation/builders, external session options, fake/OpenAI backends, tool grants | `_resources/runtime.py` |
| Verifier validation/builders and role resolution | `_resources/verifier.py` |
| Generic Environment validation/builders, response adaptation, tool schema projection | `_resources/environment.py` |
| `ExecutionResources`, shutdown, termination, ownership cleanup | `_resources/lifecycle.py` |
| Cross-resource checks, tool catalog selection and Robotics composition | `resources.py` |

External session setup and model propagation live in `_resources/runtime.py`.
Explicit tools are validated against the selected Common catalog; native
`python_execute` construction lives in `_resources/environment.py`. Built-in
Verifier types are `agent` and application-registered `custom`. The branch does
not include MathMemory, Math presets, optional Math engines or Lean4.
Chips task sessions and their experiment entry remain in `workflows/chips/`;
they reuse these shared Runtime contracts.

### Reserved fields

`RunRecord.retries_exhausted` is written by nothing today. The reader in
`report.py` renders "answers produced without ever passing a verifier are not
clean solves"; the producer was #160's DAG engine, which was retired, and the
canonical `WorkflowExecutor` has not been given a replacement even though
`_select_transition` is exactly where a `not_passed` budget runs out. It is kept
rather than deleted so restoring the signal does not also have to restore the
reader, and both halves are pinned by
`tests/workflows/test_report.py::test_the_retries_exhausted_section_has_a_reader_but_no_producer`.

The main dependency paths are:

- CLI: `main.main()` → `run.main()` → the one-run or repeated-run path.
- One execution: `run_workflow()` → `compose_resources()` → `WorkflowExecutor`
  → `persist_results()`.
- Repeated or scored execution: `run()` → `run_batch()` → `scoring.run_cell()`
  → `run_workflow()` → post-execution grading → `report.build_report()`.

Render one or more completed or in-flight run directories as a self-contained
HTML trajectory viewer with the workflow-owned module entry point:

```bash
python -m alphaapollo.workflows.visualize RUN_DIR [RUN_DIR ...] -o OUT.html
```

The viewer reads each cell's canonical trajectory, workflow-step results, and
scored result when present. Missing or malformed cell artifacts are shown as
visible notices so one bad or still-running cell does not hide the rest of the
run.

Private gold used for scored evaluation is joined in the repeated-run data layer
and consumed by `scoring.py` only after `run_workflow()` returns; `report.py`
only reads the resulting `RunRecord` values.

An opt-in, separate channel exists for state that a Runtime-owned Environment
must consume during an episode. `dataset.task_payload_path` names one concrete
private split file, such as `private/test.parquet`; it is never filtered with
the public `dataset.split`, because canonical private rows have no `split`
column. Its format inherits `dataset.format` unless `task_payload_format` is set.
Rows join to public inputs by `id_key`. Under the `direct`
`task_payload_envelope`, `task_payload_key` must contain an object or a
JSON-encoded object, and the loader passes that object through unchanged; the
`robot_task` envelope decodes the same wire shape and then reshapes it into the
canonical robotics payload. The discriminator remains explicit so a further
envelope transforms at this one audited loader boundary instead of guessing from
private record shape.
The resulting payload uses dedicated `WorkflowInput`, `AgentTask`, and
`EnvironmentContext` fields; it is excluded from prompts, attribution, reprs,
and persisted Workflow results.

The packaged Hydra root `alphaapollo/configs/workflow.yaml` composes generic
configuration groups. Standalone JSON/YAML recipes remain supported. Chips
operator examples use `chips_experiment` as described in the
[unified experiment guide](../../docs/chips/UNIFIED_EXPERIMENT.md).

The unified Workflow runner executes once by default; `execution` adds samples,
concurrency, limits and resume policy. Optional `scoring` joins private gold
only after execution. Results and trajectories remain isolated per cell.

Optional `memory: {adapter: verification, mode: working, profile: full,
top_k: 3}` stores entries projected by a Workflow memory adapter and injects
same-input, same-branch recalls into later agent steps. `verification` is the
default and stores normalized verifier judgments. Robotics workflows can use
`adapter: robotics`; that adapter accepts only a terminal
`EnvironmentTransition` with an explicit boolean `success`, while any
model-authored summary is retained as untrusted context. `adapter: unified`
selects one class that dispatches on the output's own type and delegates to
the single-domain projections (verifier verdicts, terminal environment
results); its entries live under the
workflow-generic `workflow:<name>` namespace, and everything unrecognized
fails closed. `cross_run: true` (valid only with `mode: persistent`) turns the
built-in projections into explicit per-run evidence that outlives the run:
entries become `PERSISTENT`, the task id drops its branch suffix so every run
of the same input shares one scope, and a later run recalls earlier runs'
outcomes through the persistent channel — each run's outcome remains a
distinct attributed record, bounded at recall by `top_k`. Applications with a
domain-specific adapter can pass it directly to
`run_workflow(..., memory_adapter=adapter)`, keeping memory independent from
the verifier interface. Memory is agent-facing only: the runtime preserves the
unaugmented public instruction for Environment initialization, so Robotics'
task-payload consistency check remains intact. Profiles independently enable semantic recall, lexical
grep recall, and the per-branch scratchpad; `semantic` preserves the original
thin-memory path. Omitting the block preserves the existing execution and
config digest. See `memory/` for journal and adapter contracts.

Role prompts are referenced by stable IDs such as `roles.proposer` and resolved
from `alphaapollo.common.prompts`. Inline `system_prompt` and `input_template`
remain supported for compatibility, but a role cannot combine them with
`prompt_ref`, and a step cannot override a referenced role's user template. Use
`roles.model_default` to send no system message; it does not guarantee that the
provider supplies a semantic default instruction. Native tool descriptions
remain separate structured schemas under `alphaapollo.common.execution.tools`.

## Persistent external-agent workspaces

External runtimes create a separate workspace for each invocation by default.
Set `reuse_workspace: true` when later Workflow steps must revise files created
by an earlier step:

```yaml
runtimes:
  solver:
    type: external
    options:
      agent: codex
      model: Codex CLI default
      workspace_root: runs/workspaces
      reuse_workspace: true
      keep_workspaces: false
```

Reuse is scoped to one Workflow name, input ID, and ensemble branch. Different
inputs and branches never share a directory, and concurrent use of the same
branch is rejected. `reuse_workspace` controls reuse during the run;
`keep_workspaces` independently controls whether those directories remain after
the Runtime closes. Both options default to `false`.

## Application-registered Verifiers

`type: custom` lets an application select a Verifier factory it registered in
Python before composing resources:

```python
from alphaapollo.reasoning.verification import register_custom_verifier

register_custom_verifier("my.project.score", create_project_verifier)
```

```yaml
verifiers:
  project_verifier:
    type: custom
    options:
      name: my.project.score
      config:
        threshold: 0.95
```

The `config` mapping is passed to the factory, which must return a fresh
`Verifier`. Names must already be registered: YAML cannot import a module or
execute a Python path. This keeps executable-code selection at the application
boundary while allowing a complete `RunConfig` to configure the registered
implementation.
