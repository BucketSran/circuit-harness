# Cadence / Spectre connection reference

For the current server-local, detached and independently graded PDK-free RC-001
path, use the [Spectre RC harness guide](../../../docs/reference/SPECTRE_RC.md) and
its private-profile [template](spectre-rc.profile.example.json). The labctl
commands below remain a manual connection reference for other netlists.

This example follows vaEVAs' `docs/LABCTL_SPECTRE_WORKFLOW.md` and
`runners/run_gold_dual_suite.py`: local argv → labctl transfer → remote bash →
Cadence csh setup → Spectre → download. The source was inspected locally;
no vaEVAs source, credentials, PDKs or evaluation datasets are copied here.

Keep these stages distinct from EMX: Spectre runs a netlist and optionally
compiles Verilog-A, while the EMX path takes a GDS layout and a process/port
configuration. Neither successful exit establishes circuit correctness.

The local machine used for implementation does not have `labctl` on PATH.
Install/configure your lab's existing labctl distribution before this example.
Use an SSH key/agent and verified host key. Place host, username, jump-host
and Cadence setup paths in your private labctl profile; never commit them.

```bash
# Use the labctl profile already maintained with vaEVAs.
labctl --profile YOUR_LAB_PROFILE check

# Pick a NEW directory for each simulation. Do not reuse an existing job ID.
labctl --profile YOUR_LAB_PROFILE up examples/chips/cadence/input \
  /your/private/chips-cadence/JOB_ID
labctl --profile YOUR_LAB_PROFILE sh examples/chips/cadence/remote_spectre.sh \
  /your/private/chips-cadence/JOB_ID /your/cadence/setup.csh
labctl --profile YOUR_LAB_PROFILE down /your/private/chips-cadence/JOB_ID \
  runs/chips/cadence/JOB_ID
```

This is a manual connection example, not an EMX substitute or a second
agent tool. If execution loses its connection, inspect that job's existing
logs before launching anything else. Download `spectre.log` and `psf/` even
after a tool error. Spectre version/options must be confirmed against the
lab installation. For model-based tasks add the lab's `.va` file and matching
`ahdl_include` to the staged input, preserving any support-file dependencies.

`input/rc.scs` is a small PDK-free transient smoke input. It has not been run
on the lab server as part of this delivery; it does not use private models.
