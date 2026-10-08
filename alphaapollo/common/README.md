# AlphaApollo Common

`alphaapollo.common` contains policy-free infrastructure shared by Reasoning,
Learning, and Evolving.

## Ownership

- `artifacts/`: serializable artifact references and array descriptors.
- `generation/`: one model generation and backend adapters (Issue #185).
- `environment/`: task action projection and Environment transitions.
- `execution/`: typed tools, isolated backends, sessions, and workspaces.
- `grader/`: policy-free answer grading plus protected domain ranking graders.
- `prompts/`: canonical prompt resources and policy-free rendering shared by
  Learning and Reasoning.
- `trajectory/`: complete episode capture, read-back, and persistence.

## Module map

Some established modules intentionally expose more than one closely related
primitive. Their filenames are kept stable for import compatibility:

- `environment/default/environment.py` is the stable default Environment facade.
  Tool bridges, the single-episode lifecycle, and outcome projection live in
  focused sibling modules under `environment/default/`.
- `environment/chips.py` adapts public Chips task sessions to the shared
  Environment contract; independent final scoring stays with the task owner.
- `environment/robotics/` retains the Robotics reference adapter and its
  action/observation projection.
- `environment/provider.py` retains pool lifecycle, seed and slot ownership.
  Applications explicitly register their own pool builders; no other benchmark
  pool implementation is bundled in this branch.
- `execution/sandbox/docker.py` owns the Docker CLI environment and backend.
  `sandbox/local.py` owns host-local subprocess execution,
  `sandbox/manager.py` selects a backend, and
  `execution/sandbox/podman.py` owns the stable rootless Podman facade.
  Podman subprocess runners, CLI lifecycle,
  and backend adaptation live under `sandbox/_podman/`. The former imports
  from `docker.py` and all established `execution/backends/` paths remain
  available as compatibility aliases.
- `execution/session.py` is the stable session import facade. Backend borrowing,
  runtime sessions, the Environment-facing tool gateway, and the legacy session
  live in focused `execution/_session/` modules. `execution/tools/gateway.py`
  owns pre-execution authorization.
- `execution/workspace.py` is the stable workspace import facade. Its internal
  implementations are separated under `execution/_workspace/` into artifact
  storage, leases, snapshots, host exports, and disposable verifier staging.
- `execution/output.py` owns backend-neutral bounded output capture and rendering.
  Podman runners and shell tools depend on this lower-level primitive instead
  of sandbox backends importing tool builtins.
- `execution/sandbox/local.py` bounds wall clock, CPU time and open files, and
  bounds address space where the platform enforces `RLIMIT_AS` (macOS does not,
  and says so). It does **not** bound process count at any setting: `RLIMIT_NPROC`
  is enforced per real uid rather than per sandbox, so the OS's per-user limit is
  the only backstop and `max_processes` is refused rather than applied as
  something else. A per-sandbox process cap means the Podman backend, whose
  `--pids-limit` is a per-container cgroup limit.
- `execution/process_group.py` owns group-wide signalling of sandboxed children,
  so a timeout stops the whole subtree instead of orphaning what the command
  spawned. Every backend that spawns one starts it with `start_new_session=True`
  and stops it through this primitive; the two halves are one contract.
- `execution/tools/builtins/shell.py` is the stable shell-tool facade. Bash
  adaptation and compatibility Python execution live under
  `execution/tools/builtins/_shell/`.
- `execution/tools/builtins/files.py` is the stable workspace-tool facade.
  Worker source, command construction, and host-side adapters live separately
  under `execution/tools/builtins/_files/`.
- `execution/tools/python.py` owns structured `python_execute` over the shared
  sandbox; the internal `python` ID and `<python_code>` protocol remain available.
- `execution/tools/chips.py` and `execution/tools/robotics/` own their public
  domain ToolSpecs and executors.
- `grader/answer.py` reads declared conclusions; `grader/declared.py` supplies
  generic vote keys without a symbolic solver or unit-stripping score policy.
  Only `exact_match` and `environment_success` are built-in graders here.
- `trajectory/recorder.py` is the stable recording facade. Episode assembly,
  capture contracts, sinks, bounded sanitization, and typed replay live in
  focused `trajectory/_recorder/` modules.

Within these modules, section comments make contract and lifecycle boundaries
explicit, while private helpers stay close to the code they support. Canonical
ownership follows the package named above (`common.execution`,
`common.trajectory`, and so on); cross-package re-exports are convenience or
compatibility surfaces rather than ownership declarations.

Persistent records live in the owning component's `schemas.py`; Common has no
global schema package.

## Forbidden

- consumer-specific prompt selection, revise/stop policy, or verifier policy;
- Learning runtime `DataProto`, reward, advantage, or trainer policy;
- Evolving memory, skill-selection, or agent policy;
- scheduler, branch, global-budget, or product workflow policy;
- a universal registry for unrelated extension points.

`common` must not import implementations from `reasoning`, `learning`, or
`evolving`. Product-facing composition belongs in `workflows`.

Historical shared-architecture design and merge notes remain in private backups.
The maintained [development SOP](../../docs/chips/DEVELOPMENT_SOP.md) and
[repository scope](../../docs/chips/REPOSITORY_SCOPE.md) define current integration rules.
