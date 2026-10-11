# Current EVAS public sessions

`circuit_harness.execution.sessions.current_evas_session` owns public candidate
editing, bounded diagnostic execution and final collection. It does not grade
benchmark tasks. A benchmark owner must explicitly supply a public task mapping;
there is no inferred va07 mapping or hidden checker mount.

## Create and use a session

Call `create_session(task=..., materials=..., checkout=..., kernel=...,
directory=..., image=...)`. The directory must be new and private to the broker.
The `task` declaration requires exactly these fields:

```python
{
    "task_id": "synthetic-integrator",
    "task_version": "fixture-v1",
    "public_files": ["instruction.md"],
    "candidate_files": ["idt.va"],
    "feedback_fields": ["diagnostics", "observations"],
    "manifest": {
        "models": ["idt.va"],
        "instances": [{"name": "integrator", "module": "integrated_ramp",
                       "connections": {"vin": "input", "vout": "output", "vref": "0"}}],
        "transient": {"sources": {"input": [[0, 0.2], [4e-6, -0.2]]},
                      "output_times": [0, 1e-6, 2e-6, 3e-6, 4e-6],
                      "stop": 4e-6, "max_step": 4e-6},
        "tolerances": {"vabstol": 1e-8, "reltol": 0.0}
    }
}
```

Paths must be unique canonical relative paths without symlinks. Manifest models
must exactly match candidate files. Feedback fields must be explicitly declared;
only `diagnostics`, `diagnostic` and `observations` are supported, including an empty selection.
`diagnostic` is an opt-in machine field for failed fixed EVAS simulations; existing tasks keep their feedback selection.
Public material bytes, source inventory and kernel identity are frozen at creation.

`session_info(directory)` returns tool instructions, a summary of the fixed manifest, declared
file names, remaining budgets, backend/config version and broker workspace paths.
Those paths are for the broker. Do not grant the Agent direct access to the private
session tree: that would bypass action budgets and immutable inputs.

`session_action(directory, request)` accepts a request with `action_id`, `tool`
and `arguments`. `tool_schemas()` returns OpenAI function wrappers. The tools are:

- `evas_read(path)`: read a declared public or candidate file; empty path lists files.
- `evas_write(path, content)`: atomically replace one complete candidate file.
  Empty or invalid source is accepted; benchmark correctness is a separate decision.
- `evas_simulate()`: freeze all candidate files and run the task's fixed manifest.
- `evas_submit()`: freeze the last complete candidate, without a syntax gate.
- `evas_observe(artifact_id, signals?, start?, end?, max_points?)`: inspect saved public
  EVAS waveforms, without another simulation.
- `evas_read_artifact(artifact_id, offset?, limit?)`: read the complete public JSON
  in bounded pages. `manifest` identifies the fixed public configuration.

## Compact observations and exact artifacts

New sessions pin `observation_view_version=1` in `session.json`, public info and
simulation responses. Sessions without this field keep their original tool catalog,
full manifest and full feedback. Completed action responses are replayed unchanged.
Existing trajectories are never shortened or rewritten by this feature.

For a successful EVAS simulation, `observations` contains a `waveform_summary`:
engine, point count, time range, event count and up to 32 signal names with minima
and maxima over all saved points. This is descriptive data, not a correctness score.
The summary is limited to 4096 UTF-8 bytes; oversized signal names cause the signal
list to be omitted with `signals_truncated=true`. Numerical execution and sampling
requests do not change. Small Python measurements remain inline; observations larger
than 4096 bytes become a `json_artifact` notice. Remote public Spectre observations use
this generic JSON policy, without interpreting them as EVAS waveforms.

`observation_artifact` identifies the complete public observation JSON by action,
SHA-256 and byte count. It also carries the frozen candidate identity, authority and,
for measurements, experiment identity. The broker stores this public JSON separately
from the unchanged execution stdout and `result.json`. Only task-declared observations
receive an artifact. Artifact IDs do not grant access to private logs or final grading.

For example, after action `sim-2`, send these arguments to `evas_observe`:

```json
{"artifact_id":"observation:sim-2","signals":["z","count"],"start":0.49,"end":0.51,"max_points":64}
```

Time bounds are inclusive. Omitted bounds select the full saved range. The default
selects the first eight signals and at most 64 points; limits are eight signals and
128 points, with a minimum point limit of two. Replies include exact saved `times`
and `values`, `matched_points`, and `sampled`. When sampling is needed, uniformly
spaced row indices retain both endpoints. There is no interpolation. This preview
can miss events and extrema; narrow the window or read the full JSON for exact
measurements. Replies over 24 KiB are rejected; request fewer signals or points.

`evas_read_artifact` uses Unicode character offsets, not byte offsets. The default
page is 2048 characters, with a maximum of 4096. Concatenate `content` pages using
`next_offset` until it is null, then parse JSON. The SHA-256 covers the UTF-8 bytes of
the concatenated text. `info.manifest_artifact` uses the same API for the complete
manifest, including stimulus arrays and observation times.

Both read tools spend an action but no simulation budget. Repeating an action ID
returns its saved response. New reads stop after submission, just like other actions.
Reads verify the artifact digest and reject missing, changed or symlinked files.
Existing action budgets still apply; use signal windows or compute compact measurements
instead of paging a large waveform through every remaining action.

## Action and collection lifecycle

An action ID has one immutable request and durable response. Reusing it with another
request fails. An unfinished action returns `unknown_execution`, `retry_safe=False`;
its budget is not reset. An unfinished simulation permits edits and collection but
blocks another simulation with `unresolved_previous_simulation`. After its worker
stores the response, the next simulation may run if budget remains. The final
submission action has a reserved budget slot. Defaults are 24 actions, 4 simulations,
120 seconds per execution and 16 MiB output; maximums are 1000/1000, 300 seconds and
16 MiB. Candidate/public inputs each have a 16 MiB aggregate bound.

`close_session(directory, reason)` returns a stable receipt with `state`,
`candidate_directory` and `candidate_sha256` when complete. Repeat close returns
that same frozen package. The caller must stop the Agent before collecting it.
Collection can run while a simulation executes its older snapshot. Missing files
produce `missing_candidate`; changed immutable material produces an integrity error.

## Execution backends

Configuration has `backend_config_version=1` and an explicit backend:

- `docker` (default): requires `image="sha256:<immutable-local-image-id>"` and a
  Linux ELF kernel built for that image's architecture. A host Mach-O kernel is
  rejected. The image must already exist locally and contain Python 3. Frozen
  engine/kernel/inputs are read-only; only execution artifacts are writable.
  Network is disabled, capabilities are dropped, privileges cannot increase,
  container root is read-only, and limits are 32 PIDs, 512 MiB memory with no extra
  swap, one CPU by default, file size/output quota and a 16 MiB no-exec temporary filesystem.
  Python uses `-I -S -B`. An independent in-container watchdog bounds execution
  even if the broker disappears; normal cancellation/error cleanup runs `docker rm -f`.
- `podman`: uses the same public container constraints and watchdog through the
  local Podman CLI. `cpu_limit=1` remains the default; explicit `cpu_limit=None`
  records that no CPU hard quota is requested. Unsupported requested quotas fail,
  with no host fallback or quota-free retry. Images and kernel must match the
  Podman host architecture and namespace UID mapping. Both container backends
  accept declared Python measurements. Harbor operators use `public_cpu_limit`
  for this setting and the [private Podman runner](../guides/HARBOR_DEPLOYMENT.md#使用-rootless-podman).
- `native_codex_sandbox`: requires `image=None`, the actual native `codex` executable
  path and a Python executable via `codex=...`, `python=...`. This uses the outer
  Codex OS sandbox with minimal reads, explicit read-only source/kernel/runtime
  grants, one writable execution workspace and no network. An access probe must pass.
  Executable hashes are checked. Wall clock, CPU, file size, descriptor count and
  recursive output are bounded. **Memory and PID counts are not limited** on this
  backend. Its scope is fixed EVAS engine/manifest execution on candidate text;
  it does not accept arbitrary candidate Python experiments. Native Python and
  kernel executables must match the host platform.

Docker Desktop must permit bind mounts from the execution directory. A rejected
mount is an infrastructure failure; the implementation never falls back to a host
process. Local Docker tests use a directory inside the checkout rather than macOS's
private temporary directory when Desktop file sharing requires it.

Successful simulation returns task-declared public observations/diagnostics plus
execution/backend/image/cleanup metadata, frozen candidate identity and
`task_correctness="not_evaluated"`. It never returns reward or hidden acceptance.
Errors distinguish exhausted action/simulation budgets, unresolved actions, missing
candidate, infrastructure failure, timeout, output limit, backend failure, invalid
result and unconfirmed cleanup. Inspect private action receipts for detailed evidence;
do not expose their paths or logs wholesale to the Agent.

### Versioned EVAS failure metadata

For a failed fixed EVAS simulation, `diagnostic` contains `diagnostic_version`, `code`,
`category`, `stage` and `capability`. It is read from the complete private stderr before
text truncation. Only well-formed v1 metadata is projected; missing, malformed, oversized
or future-version payloads have `category="unknown"`. A valid integer version is retained.
The original log and its artifact hash remain private. No category is inferred from a
message or kind prefix. Metadata is not a circuit correctness verdict.

Messages, locations, raw nested payloads and arbitrary extra keys are excluded from this
new public field. The existing `diagnostics` string keeps its current truncation/redaction
contract. Python measurement experiments and remote Spectre do not emit EVAS metadata.
`execution`, task grading and replay classification are unchanged. A missing machine
field never implies success. Parsing is limited to 1 MiB; larger raw logs remain subject
to the existing output limit and are not discarded from accounting.

## Evidence and limits

`tests/chips/test_current_evas_session.py` covers complete writes, exact bundles,
action replay, budgets, pending simulation exclusion and collection during execution.
`test_current_evas_public.py` has opt-in real Docker and native OS boundary tests.
Set `CHIPS_TEST_DOCKER_IMAGE` to a local immutable image ID or
`CHIPS_TEST_NATIVE_CODEX` to the native CLI binary to run those probes. Their engines
are synthetic fixtures; they establish platform access boundaries, not circuit scores.
`test_current_evas_real.py` additionally executes a selected real EVAS checkout through
both the trusted runner and an OS-isolated public session. Set `CHIPS_TEST_EVAS_CHECKOUT`
and `CHIPS_TEST_NATIVE_CODEX` to run the five fixed compile/runtime/success controls.
They verify metadata, retained failures and action replay, not benchmark replay or scores.
The earlier macOS-only acceptance selected EVAS `ab0df35b` / IR18 and kernel
`0de37e97dca3e52b871a8895872a39994bd012b51c9d5d5086f8bc26ead6667b`.
It completed five controls, but did not establish a Linux build or Docker benchmark replay.

On 2026-10-11, Harness `bdcdde2` and clean EVAS source
`989e7f53ad825e7393167feef67e0df1e2039a1f` completed the six Docker transport
controls below with IR18 and diagnostic v1. Both native and Linux kernels were
built from the same frozen source inventory. The four EVAS failures retained
`backend_error` in the runner and public session, with three `unsupported` reasons
and one `numerical` reason. Replay kept all four failures unscored and `unevaluable`.
The successful control was `not_compared`; the actual missing-kernel or missing-local-image
control was `infrastructure_error` and counted as `infra`. All six sealed replay
receipts verified, and the aggregate retained six records and six candidates.
These complete the fixed external consumer controls under vaEVAS #64. They
establish no circuit correctness score, full diagnostic-exit coverage or Spectre alignment.

A separate Linux OS probe used the same selected kernel and a two-sample static
request with `EVAS_STATIC_THREADS=2`. Docker's inspected `pids-limit=1` caused
actual exit 2 with `worker_start` and OS error 11. The same request under
`pids-limit=8` completed with y=2 and y=4. The frozen EVAS Python adapter retained
the raw payload and projected v1, `kernel.worker_start`, kernel and infrastructure.
Both containers used no network, a read-only root, 512 MiB memory and one CPU,
with confirmed cleanup. This probe verifies the OS thread-start failure and Python
adaptation, separately from the six replay records. It did not pass through a
public session or benchmark replay and does not replace their infrastructure control.
Worker panic and VCO advice are producer-owned EVAS tests, outside this consumer
replay denominator.

The selected immutable local image is
`sha256:3e7c9ffd3018d384a089688b8eece2d95b4d0b4b056fc733513b375b13ba0d44`.
Kernel SHA-256 values are:

| Build | SHA-256 |
| --- | --- |
| Native debug | `b1bb58a2bb4d59f060eda0d8e788ef5a4f5e9c52e38c74a4349214f2ee30d86f` |
| Linux release | `86c6d470a39075f7648b431e5d315ae29cb2d54871687d048ea06bb70382ee8e` |

The 107-file build inventory has SHA-256
`2a4e80076e1428ed8c4eafb1870d346be626364e120a15d1b6acdd84607ad2a8`,
computed from the sorted file inventory as compact JSON. Complete raw runs, build
logs, receipts and failed preparation attempts remain private; they are not a
public dataset. The maintained test below supplies the rerunnable acceptance path.

`test_current_evas_docker_replay_real.py` accepts `CHIPS_TEST_EVAS_CHECKOUT`,
`CHIPS_TEST_EVAS_LINUX_KERNEL` and `CHIPS_TEST_DOCKER_IMAGE` for the real Docker
transport controls. The selected checkout also needs its built native debug kernel
for the trusted runner. Use an immutable local image ID containing Python 3 and a
Linux kernel built from the same frozen source bytes. The test runs the four EVAS
failures, a successful arithmetic control and a real missing-kernel or missing-local-image
infrastructure failure through the runner, public session and sealed benchmark replay.
The event DAE must pass static lint before failing during transient execution.

These are diagnostic transport controls, not benchmark circuit scores. The probe
package preserves complete EVAS stderr inside the sealed replay inventory; after
`verify_replay`, the existing metadata reader consumes those original bytes. Failed
simulations remain unscored `unclassified_failure` records classified `unevaluable`.
The success control is `not_compared`, and the infrastructure control is `infra`.
`summarize_replays` counts all six records and candidates. Public feedback retains
only the declared machine projection; original messages, locations and sample identity
remain in the private logs. The run identity records source inventory, image, native
and Linux kernel hashes, IR and diagnostic versions, acceptance file hash and sealed
replay receipt hashes. Missing, malformed and future payloads remain covered by the
constructed negative controls in `test_current_evas.py`, rather than pretending that
a current EVAS engine emits a future protocol.

The trusted resource runner `backends/current_evas.py` is not a sandbox and is not a fallback.

The public adapter does not yet establish real va07 benchmark circuit acceptance,
arbitrary native script isolation or native memory/PID enforcement. These require
separate owned acceptance evidence. The earlier Linux-build and Docker-replay gaps
are closed only for the fixed diagnostic transport controls described above.

## Opt-in public Python measurements

Only Docker tasks may add the optional declaration below. Omit `experiments` to
disable the capability; an explicit `null` value is rejected before creating a session.
Fixed tasks cannot write or execute experiment files. Native sessions reject
this declaration, since native execution has no memory or PID isolation.

```python
"experiments": {
    "version": "measurement-v1",
    "files": ["probe.py", "stimulus.json", "testbench.va"],
    "analyses": ["python_measurement"]
}
```

Files are an exact, disjoint allowlist with `.py`, `.json`, `.va` or `.txt` suffixes.
All declared experiment files must exist before execution. They use the same
`evas_read` and `evas_write` actions. `session_info()['tools']` adds
`evas_experiment(analysis="python_measurement", script="probe.py")` only for this
capability; a broker should advertise that per-session list.

Each experiment spends the shared action/simulation budget and freezes every
candidate, experiment and public material file. Its `experiment_sha256` identifies
that complete snapshot, separately from the formal `candidate_sha256`. The frozen
request also records the fixed manifest, source, kernel and analysis/script choice.
Scripts run with isolated Python flags inside the same Docker boundary: `/inputs`
is read-only, `/engine` contains pinned EVAS source, `/kernel` is pinned and `/output`
is the writable working directory. Scripts may construct their own testbench and
invoke the declared engine; no host commands, network, hidden checker or reward
service is mounted. Python standard library is available, with no package installs.

The script must print one JSON object on stdout. Top-level `reward`, `score`,
`authority` and `task_correctness` are rejected. Results carry
`authority="agent_measurement"` and `task_correctness="not_evaluated"`; selected
observations still require `feedback_fields`. Invalid JSON, timeouts, output limits,
backend/cleanup failures remain unscored diagnostics. Collection freezes only
`candidate_files`, never incidental experiment files. Operators can replay the
private complete snapshot through `run_public(..., experiment=..., script=...)`
using its original pinned session configuration and a new action directory containing a new execution directory.

Real Docker fixtures cover a task-created Python measurement, all-input identity,
formal-only collection, access restrictions, timeout, malformed output, attempted
reward output and excessive output. They do not establish a real Agent-authored
benchmark trial; that acceptance still requires a benchmark-owned opt-in task.

## Configuration B: task-declared remote public Spectre

Select `backend="remote_spectre"`, `image=None`, `public_task_package=...` and
`public_remote=...` when creating the same public session. `checkout` and `kernel`
may be omitted for this backend. The public task uses the same file, feedback and
budget declarations, but its `manifest` is exactly `{"condition_id": "..."}`.
That condition must match the inventoried public package. It is a benchmark-owned
Spectre condition, not a translation of EVAS solver options. Only the existing
`diagnostics` and `observations` feedback selections are accepted by the session.
The package's purpose must be `public`, and its task identity and feedback fields
must match the session. Public Python experiments are unsupported on this path.

`public_remote` declares `host`, `python`, `bundle`, `profile`, `run_root`,
`archive_root` and `upload_root`, with the same private transport contract as final
jobs. The frozen public package and candidate are transferred separately from the
final package. The Agent receives no profile, SSH identity, artifact path, raw logs,
final report or reward. Remote public requests require the server's Docker profile;
a host operator profile is rejected even if the request claims isolation.

Each simulation reserves its existing session action and quota before network
execution and persists a randomly unique public job ID before submission. A broken
handoff returns `unknown_execution`, `retry_safe=False`. Repeating the exact action
queries the same ID without submitting again or spending another action. Missing
or unknown remote state blocks new simulations. It can remain unknown if a crash
occurred before the server accepted the submission; an operator must reconcile it,
rather than automatically relaunching. A completed corrupt archive returns
`invalid_result` and never a score. License/dependency failures retain their structured
execution classifications. Only a verified completed public archive can supply
package-declared feedback.

Every CLI call and archive download shares the remaining per-call wait deadline.
Waiting consumes Harbor's solve wall clock; `wait_elapsed_s` reports the current
call's wait and private `execution/wait.json` accumulates waits across recovery calls.
Collection does not await a remote simulation. Its older result can only update that
action's private receipt, never the frozen candidate or another session's result.
Stopping the client does not cancel the detached remote job; explicit job cancellation
and the server's finite execution limit are independent operations.

The server boundary and its deployment limits are described in
[BENCHMARK_EVALUATION](BENCHMARK_EVALUATION.md#isolated-public-spectre-jobs).
Local protocol, real Docker and controlled Harbor Trial tests prove their respective
boundaries; they do not establish a real native Agent, SSH server, Spectre license or
independently graded configuration B trial.
