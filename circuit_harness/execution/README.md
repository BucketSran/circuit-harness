# Chips execution module map

This package owns simulator execution, bounded public task sessions, durable
jobs, and operator-only verification. Agent orchestration and experiment scheduling
use [Harbor](../harbor/README.md); runnable task inputs live
in [examples/chips](../../examples/chips/README.md).

| Responsibility | Modules |
| --- | --- |
| Durable execution and evidence | `journal.py` (events and digests), `process.py` (owned processes), `jobs.py` (detached jobs), `archive.py` (sealed outputs), `ssh_worker.py` (remote job protocol) |
| Local/SSH task actions | `session_transport.py` (action IDs, retries and evidence), `session_budget.py` (public action limits) |
| Direct simulator paths | `simulator.py` (RTL), `emx.py` (JSON to GDS and EMX), `ngspice.py` and `rc_validation.py` (RC run and independent check), `spectre.py` and `spectre_testbench.py` (bounded Spectre runs and gain checks) |
| VABench | `vabench_session.py` (public actions), `vabench.py` (operator replay), `vabench_worker.py` and `vabench_public_worker.py` (pinned interpreter/sandbox workers), `vabench_remote.py` (transport compatibility), `vabench_spectre_parity.py` (separate waveform comparison) |
| Analog Design Bench | `analog_public.py` (candidate rules), `analog_session.py` (public actions and freeze), `analog_design_bench.py` (operator verification), `analog_episode.py` (private archive), `analog_remote.py` (transport adapter) |
| Current EVAS | `current_evas.py` (source and kernel identity), `current_evas_session.py` (public actions and freeze), `current_evas_public.py` (bounded public execution), `native_sandbox.py` (outer native process isolation) |
| Frozen benchmark evaluation | `candidate_bundle.py` (byte identity), `benchmark_spectre.py` (task-owned checker and private profile), `benchmark_remote.py` (SSH submission and verified retrieval), `benchmark_replay.py` (saved candidate replay and comparison) |
| Task authoring | `task_authoring.py` (sourced drafts and confirmation), `authoring_session.py` (bounded public gain session) |
| Server packaging | `bundle.py` builds the fixed standard-library zipapp and explicitly lists its source files; it also includes `circuit_harness/cli.py` as the command entry |

Not every module is shipped in the server zipapp. When changing a server-side
import, update `bundle.py`'s explicit source list and run the bundle tests.
Keep public feedback in task sessions and final scoring in the operator-owned
modules; a successful public simulation is not an independent final score.

The [current EVAS session contract](../../docs/chips/CURRENT_EVAS_PUBLIC_SESSION.md)
defines the public task manifest and execution limits. The
[benchmark evaluation guide](../../docs/chips/BENCHMARK_EVALUATION.md)
defines frozen inputs, checker results and replay. Harbor owns trial scheduling
through the [Harbor adapter](../../docs/chips/HARBOR.md); it does not change
the retained VABench or Analog operator/session protocols.
