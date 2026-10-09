# Circuit execution module map

This package owns simulator execution, bounded public task sessions, durable
jobs, and operator-only verification. Agent orchestration and experiment scheduling
use [Harbor](../harbor/README.md); runnable task inputs live
in [examples/chips](../../examples/chips/README.md).

| Owning directory | Responsibility and modules |
| --- | --- |
| [runtime/](runtime/) | Processes, events, digests, detached jobs and native isolation: `process.py`, `journal.py`, `jobs.py`, `simulator.py`, `native_sandbox.py`. `bundle.py` builds the fixed standard-library server zipapp. |
| [backends/](backends/) | Simulator execution and source identity: `ngspice.py`, `spectre.py`, `emx.py`, `spectre_testbench.py`, `current_evas.py`. Fixed VABench replay and pinned interpreter workers: `vabench.py`, `vabench_worker.py`, `vabench_public_worker.py`. |
| [sessions/](sessions/) | Public candidate editing, diagnostic actions and final collection: `analog_session.py`, `current_evas_session.py`, `vabench_session.py`, `authoring_session.py`. Public execution and feedback: `analog_public.py`, `current_evas_public.py`, `public_observations.py`. Action limits and sourced drafts: `session_budget.py`, `task_authoring.py`. |
| [evaluation/](evaluation/) | Frozen byte identity, sealed evidence and operator verification: `candidate_bundle.py`, `archive.py`, `analog_episode.py`, `analog_design_bench.py`, `benchmark_spectre.py`, `benchmark_replay.py`, `rc_validation.py`, `vabench_spectre_parity.py`. |
| [transport/](transport/) | Local and SSH action identity, recovery and verified retrieval: `session_transport.py`, `remote_public.py`, `ssh_worker.py`, `analog_remote.py`, `vabench_remote.py`, `benchmark_remote.py`. |

The Python modules at the package root preserve existing imports, `python -m`
commands and worker entry paths. They are compatibility facades. Internal code
imports the implementation from its owning directory; new development belongs
there. The operator command entry remains `circuit_harness/cli.py`.

Public sessions use [sessions/action_store.py](sessions/action_store.py) for the
action lock, request consistency, durable reservation and cached responses. A
request without a response remains unknown and is never replayed by the store.
Task callers own action budgets, remote
snapshot recovery and candidate freezing policies. Keep public feedback in task
sessions and final scoring in operator verification modules; a successful public
simulation is not an independent final score.

Not every module is shipped in the server zipapp. When changing a server-side
import, update [runtime/bundle.py](runtime/bundle.py)'s explicit source list and
run the bundle tests. The package contains both owning implementations and the
compatibility entries required by retained commands and workers.

The [current EVAS session contract](../../docs/reference/CURRENT_EVAS_PUBLIC_SESSION.md)
defines the public task manifest and execution limits. The
[benchmark evaluation guide](../../docs/reference/BENCHMARK_EVALUATION.md)
defines frozen inputs, checker results and replay. Harbor owns trial scheduling
through the [Harbor adapter](../../docs/reference/HARBOR.md); it does not change
the retained VABench or Analog operator/session protocols.
