# EMX smoke example

This example is **layout JSON → local GDS converter → SSH → lab EMX →
download logs/results**. `emx_simulate` is the single agent-visible tool;
server, converter, process files and EMX command are operator-owned settings.

This example still selects Codex; Pi/EMX lab acceptance is pending.

Install with Python 3.10+:

```bash
python -m pip install -e '.[chips,mcp]'
```

The sample converter uses [gdstk](https://heitzmann.github.io/gdstk/gettingstarted.html)
1.0.0 to write polygons and labels, with coordinates in micrometers and 1 nm
precision. `layout.json` is a transport fixture, not a foundry-qualified layout.
Replace it and the converter with the lab's real format/generator when available.

## Operator setup

Copy `emx.config.example.json` to `emx.local.json` (ignored by Git). Replace all
placeholder paths and `lab-emx` with a configured SSH alias. Use absolute local
converter paths, a dedicated private remote root, remote Python 3.10+, and an
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
python -m alphaapollo.workflows.chips emx \
  --config examples/chips/emx/emx.local.json --input examples/chips/emx/layout.json \
  --output runs/chips/emx-demo
python -m alphaapollo.workflows.chips watch runs/chips/emx-demo
python -m alphaapollo.workflows.chips report runs/chips/emx-demo
# After interruption, same directory + identical input and configuration:
python -m alphaapollo.workflows.chips emx \
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

## Existing AlphaApollo agent integration

```bash
export ALPHAAPOLLO_CHIPS_CONFIG=/absolute/path/to/emx.local.json
python -m alphaapollo.workflows.main --config examples/chips/emx/config.yaml
```

The supplied config uses the existing Codex external runtime, preserves workspaces,
and grants only `emx_simulate`. Change the model in YAML to your configured model.
Its MCP tool timeout is 1900 seconds, above the example job's 1800-second budget
and cleanup allowance, with a 2100-second agent session. Keep these limits aligned
when changing the lab budget; Codex otherwise has a
[60-second MCP tool default](https://developers.openai.com/zh-Hans/docs/extend/mcp).
The existing pi external runtime can use the same MCP bridge. The library rejects
native/default-environment routing for this host/SSH tool. A successful harness
run is not a verifier pass; task/model ranking and independent circuit evaluation
belong to the next phase. LLM SDK retries/costs are not inferred from SSH attempts.

For Cadence/Spectre connection reuse, see [Cadence example](../cadence/README.md).
That example records the existing vaEVAs labctl pattern and is separate from EMX.

## Checks

```bash
pytest -q tests/common/execution/test_chips_harness.py tests/workflows/test_chips.py
```

See [validation](../../../docs/chips/VALIDATION.md) for measured checks and
the remaining lab-specific acceptance requirements.
