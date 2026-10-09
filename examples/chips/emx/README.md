# EMX smoke example

This example is **layout JSON → local GDS converter → SSH → lab EMX →
download logs/results**. Server, converter, process files and EMX command are operator-owned settings.

Install with Python 3.12+:

```bash
python -m pip install -e '.[chips]'
```

The sample converter uses [gdstk](https://heitzmann.github.io/gdstk/gettingstarted.html)
1.0.0 to write polygons and labels, with coordinates in micrometers and 1 nm
precision. `layout.json` is a transport fixture, not a foundry-qualified layout.
Replace it and the converter with the lab's real format/generator when available.

## Operator setup

Copy `emx.config.example.json` to `emx.local.json` (ignored by Git). Replace all
placeholder paths and `lab-emx` with a configured SSH alias. Use absolute local
converter paths, a dedicated private remote root, remote Python 3.12+, and an
executable lab wrapper. SSH uses BatchMode and strict host key checking; no
passwords, host-key bypass, PDK data or credentials go in the config/repository.

The converter receives `{input_json}` and `{output_gds}` and must produce a
nonempty GDS file. The remote wrapper receives `{gds}` and `{job_dir}` and
runs in that job directory. It must load the lab's EMX/license environment,
declare the real cell/ports/frequency/process options, and return EMX's exit code.
List required result basenames in `outputs`; stdout/stderr are always collected.
List all process files and wrapper scripts in `remote_input_files` so their
content hashes are recorded. The harness does not infer dependencies or EMX flags.

```bash
python -m circuit_harness.cli emx \
  --config examples/chips/emx/emx.local.json --input examples/chips/emx/layout.json \
  --output runs/chips/emx-demo
python -m circuit_harness.cli watch runs/chips/emx-demo
python -m circuit_harness.cli report runs/chips/emx-demo
# After interruption, same directory + identical input and configuration:
python -m circuit_harness.cli emx \
  --config examples/chips/emx/emx.local.json --input examples/chips/emx/layout.json \
  --output runs/chips/emx-demo --resume
```

Run watch in another terminal. SIGINT/SIGTERM request cancellation; a remote
cancellation request is recorded separately from confirmed termination. A lost
connection returns `unknown` and a job ID. Resume queries that ID. Duplicate
submit requests never relaunch a receipt-bearing job; a missing previously
submitted job requires operator investigation. Download retries reuse the job
and verify byte length and SHA-256 before publishing each local artifact.
Authentication/host-key failures and EMX errors are not retried as new jobs.
Retry count, per-command timeout and total deadline are bounded. Cleanup gets
at most an additional 5 seconds; remote timeout remains active after disconnection.

The remote worker is stdlib-only and leaves job evidence on the server.
Operators should remove old job directories when no longer needed. It uses
process groups and output-size polling for resource control, not a security
sandbox or a hard disk quota. Jobs that deliberately detach from their group
require a scheduler/container backend, outside this phase.

## Checks

```bash
pytest -q tests/chips/test_simulator.py tests/test_cli.py
```

The commands above are operator interfaces. To connect an Agent, define a Harbor
task with explicit public feedback and an independent verifier under the
[integration guide](../../../docs/guides/integration.md). A successful EMX process
does not establish circuit correctness.
