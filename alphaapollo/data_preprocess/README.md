# Dataset preprocessing

The preprocessing package keeps general sources flat and groups robotics
sources by benchmark:

- `prepare_custom_data.py` maps user-selected columns from local or Hub data.
- `preference_dataset.py` owns offline preference rows, pair-preserving collation,
  and validation of frozen-reference cache metadata.
- `prepare_dpo_reference.py` computes the reference log-probability columns and
  writes the content-bound `*.refmeta.json` sidecar consumed by DPO training.
  The tokenizer fingerprint follows vocabulary, special-token, and chat-template
  semantics but ignores the local or Hub path from which equivalent files load.
- `robotics/prepare_libero.py` normalizes metadata-only LIBERO task records.
- `robotics/prepare_liberopro.py` prepares the pinned LIBERO-Pro perturbation suites.
- `robotics/prepare_robocasa.py` builds the vendored RoboCasa365 task-suite
  snapshot into a prepared dataset with the scene triplet in `env_payload`.
- `core.py` owns stable identity, validation, public/private separation, and builds.
- `io.py` owns source loading, Parquet/JSON/JSONL I/O, manifests, and atomic commits.

The branch keeps generic/custom data and Robotics metadata preparation.
Math and Bio preparers remain in private history and backups, outside this source tree.

## Prepare LIBERO task metadata

Each input row names a task instruction, task index and name, problem folder,
BDDL file, initial-state file, and explicit `environment_version`. Optional
`demonstration_id`/`demonstration_path` fields identify provenance when a source
contains demonstrations. An optional positive `max_episode_steps` sets a
per-task simulator horizon. Select one of `libero_spatial`, `libero_object`,
`libero_goal`, `libero_90`, or `libero_10`:

```bash
python -m alphaapollo.data_preprocess.robotics.prepare_libero \
  --data-source ./libero_spatial.jsonl \
  --suite libero_spatial \
  --environment-version libero-<commit-or-release> \
  --checkout-path /opt/LIBERO \
  --output-root ./data
```

`--checkout-path` names the official checkout those rows are validated against.
Omit it to validate against an installed `libero` package instead. LIBERO is not
a declared extra and cannot be pip-installed in a form that carries its BDDL and
initial-state files, so a checkout is the supported form; `pyproject.toml`
records the measurements behind that. The checkout is imported only when a row
is normalized, so `--if-exists reuse` on an up-to-date build and `--if-exists
error` on an existing destination are answered without it.

The selected suite is also the default dataset name. `libero_90` defaults to a
`train` split and the other suites default to `test`; `--splits` can override
that source label. The public projection can be loaded through the ordinary
Workflow dataset mapping using `statement` as the input and `task_uid` as the
id. Simulator-facing metadata remains in the private `env_payload` for the
robotics integration layer.

For executable LIBERO data, prefer the canonical checkout path instead of
hand-authored metadata. It reads task name, language, BDDL, initial-state
count, and demonstration path from the selected checkout's
`get_benchmark_dict()` API, creates one row per task and initial state, and
always performs checkout validation before the shared atomic writer publishes
the build:

```python
from alphaapollo.data_preprocess.robotics.prepare_libero import prepare_from_checkout

prepared = prepare_from_checkout(
    checkout_path="/opt/LIBERO",
    suite="libero_spatial",
    environment_version="libero-0.1",
    source_revision="<git-commit-or-explicit-checkout-revision>",
    output_root="./data",
)
```

`source_revision` identifies the metadata checkout; it is distinct from
`environment_version`, which identifies the simulator/runtime contract. If
`checkout_path` is a Git worktree, the source revision may be omitted and is
resolved from `HEAD`. The adapter configures the selected checkout
non-interactively and never starts MuJoCo, reads demonstration HDF5 contents,
or mutates the checkout. Rows generated from the checkout are not then
compared back to it: every field would be re-read from the object it came from,
at the cost of a second full benchmark load per task. The `prepare()` entry
point remains available for metadata-only snapshots, where the comparison is a
real cross-check; pass its `checkout_path` to name the checkout those rows are
validated against.

The private payload has an explicit boundary with the shared `RobotTask`
contract. It contains `benchmark="libero"`, the required `environment_version`,
and the LIBERO task spec: `suite`, `task_order_index`, `task_index`, `task_name`,
`problem_folder`, `bddl_file`, `init_states_file`, and `initial_state_index`.
`demonstration_id`, `demonstration_path`, and `max_episode_steps` are included
only when supplied by the source. The integration layer uses the two named
envelope fields for `RobotTask.benchmark` and `RobotTask.environment_version`,
and passes the remaining task-spec fields as `RobotTask.backend_metadata`.

`--environment-version` is a required build input and one authority for the
whole prepared dataset. A row may repeat the same value for compatibility with
an exported schema, but a different row value is rejected; one build can never
mix tasks from different simulator checkouts. Suite, environment version, and
the explicit provenance claims are included in the build fingerprint so
`--if-exists reuse` cannot return an artifact built under different options.

`source_record_id`, when supplied, is the authoritative row-level id and must be
globally unique within the build. Task-level aliases (`task_id`, `task_uid`, or
`id`) are combined with the demonstration identity when present and
`initial_state_index`, so multiple demonstrations or initial states for one
task remain distinct. With no explicit id, the suite/task fields form the base
instead. Canonical checkout rows state `source_record_id` themselves from suite,
task order, task index, task name, and `initial_state_index`, so a checkout
whose demonstration HDF5s are not downloaded -- LIBERO ships them separately --
still produces the same `task_uid` for the same task and initial state. A path-only demonstration contributes an opaque identity token: the
raw path stays private and no `demonstration_id` is invented. Composed identity
components are escaped; explicit demonstration ids and path-derived tokens use
separate tags so they cannot alias one another.

If more than one instruction alias is present, the values must agree. Official
LIBERO `Task` records use `language` for the instruction and `problem="Libero"`
as a benchmark label, so that official `problem` value remains source metadata.
For compatibility with older exports, a non-`Libero` `problem` value may be
used as the instruction only when no higher-confidence instruction alias is
present. Conflicting `language`/`instruction`/`question`/`prompt` fields fail
rather than silently selecting one. Other alias groups (suite, task name, BDDL,
initial-state file, demonstration, and task id) follow the same rule: matching
values are accepted and conflicting values fail.

LIBERO metadata preparation imports an official checkout -- `--checkout-path`
when given, otherwise the installed `libero` package -- and checks task fields,
exact instruction wording, and initial-state bounds against it. The checkout is
mandatory for any build that normalizes a row: a local or Hub metadata row that
disagrees with the official benchmark is rejected during preparation rather
than at the first runtime reset. The canonical checkout path generates its rows
from that same API and records `checkout_validation:
"rows-generated-from-checkout"` instead. Validation mode is part of the build
identity, and the builder version is bumped when this contract changes.

`source_id` and `source_revision` remain provenance for the metadata dataset.
They are not aliases for the robot benchmark or simulator version. In
particular, a local metadata file may have `source_revision="local"` while its
required `environment_version` identifies the official LIBERO checkout used to
execute the task. Row-level `source_revision` and `source_uri` values are
retained under `extra.source_metadata` as upstream claims but cannot override
build-resolved provenance: the recorded `source_uri` is always the `file://` URI
or Hub dataset URL this build read, and the recorded `source_revision` is always
the resolved Hub revision or the `local` marker.

The private execution fields have one authority: `env_payload`. Diagnostic
source-column mappings and unmapped provenance stay in `extra`; execution
metadata is not duplicated there.

License defaults are deliberately `unverified`. LIBERO's code license,
task/demonstration data license, and the license of a metadata export are
different claims, and upstream surfaces do not state them consistently. Use
`--dataset-license`, `--original-content-owner`, and
`--original-content-license` only when the selected source supports those
claims; see `robotics/UPSTREAM.md` for the evidence boundary.

LIBERO rows declare `grader_id="environment_success"`, which is an
environment-outcome scoring contract rather than a text grader. It must be
resolved from the terminal `EnvironmentTransition.success` by the robotics
integration path; registering a candidate/gold text grader under that name
would let model prose impersonate simulator truth. The shared integration PR
#256 owns that outcome-scoring connection.

## Prepare LIBERO-Pro task metadata

`prepare_liberopro.py` uses the same shared public/private rows and manifest as
`prepare_libero.py`, but pins the executable identity to
`rpent-liberopro==0.1.1`. It supports exactly these packaged perturbations:

```text
libero_{spatial,object,goal,10}_{swap,task,lan,object}
```

For an existing metadata export, provide one task and initial-state selection
per row. The preparer validates suite identity, task fields, exact instruction
wording, package identity, and initial-state bounds against the installed pinned
package before publishing the build:

```bash
python -m alphaapollo.data_preprocess.robotics.prepare_liberopro \
  --data-source ./libero_goal_swap.jsonl \
  --suite libero_goal_swap \
  --output-root ./data
```

For the canonical path, generate rows directly from the package registry:

```bash
python -m alphaapollo.data_preprocess.robotics.prepare_liberopro \
  --from-installed-package \
  --suite libero_goal_swap \
  --task-order-index 0 \
  --output-root ./data
```

The Python entry point is `prepare_from_package()`. It creates one row per task
and initial-state index, records the exact package version as source revision,
and never starts MuJoCo or reads demonstration trajectories. Importing the
preparer itself does not import LIBERO-Pro; the optional package is loaded only
when a build actually needs to validate or generate rows, so an up-to-date
`--if-exists reuse` remains available without the simulator package installed.

The private payload contains `benchmark="liberopro"`,
`environment_version="rpent-liberopro==0.1.1"`, package distribution/version,
and the suite/task/BDDL/initial-state metadata. `LiberoBackend` executes that
payload directly: it resolves the `liberopro` label to the `rpent-liberopro`
package, refuses a LIBERO-Pro suite under `benchmark="libero"` and the reverse,
and cross-checks the declared package distribution and version at reset. An optional positive
`max_episode_steps` is accepted only when the caller states it. The preparer
does not infer Issue 269's 300/520 evaluation budgets, copy BDDL or initial-state
assets, generate checksums, or include case ids, evaluation seeds, Pi0.5 model
identity, Agent configuration, backend code, or runnable examples.

The fixed 16-suite allowlist excludes the five base LIBERO suites, older
trigger/OOD experiment registries, and dynamic `*_temp` suites. Expanding that
allowlist or accepting another package version requires a measured compatibility
change and builder-version bump rather than silently treating a future registry
as equivalent.

The published `0.1.1` wheel registers all 16 suites, but four task entries have
empty packaged initial-state sequences: `libero_spatial_task` indexes 3 and 7,
`libero_10_task` index 2, and `libero_10_object` index 4. Canonical preparation
fails closed when it reaches one of these tasks instead of silently omitting it;
metadata preparation can still build valid rows from the unaffected tasks in
those suites. A future package release must be measured before this exception
can be removed.

`rpent-liberopro` is intentionally not part of AlphaApollo's base or `data`
extras because it brings the full simulator dependency stack. Install the exact
package in the environment used to build LIBERO-Pro metadata. License and
provenance defaults remain `unverified`; see `robotics/UPSTREAM.md` before
asserting stronger claims.

## Prepare RoboCasa

RoboCasa tasks have no text answer: the simulator's success predicate decides
the outcome. `prepare_robocasa` therefore records a placeholder `answer`,
declares `grader_id="environment_success"`, and fills the reserved
`env_payload` column — the `(task_name, layout_id, style_id, scene_seed)`
scene triplet plus the task's official `max_episode_steps` budget that a
training-side loader needs to rebuild and bound the exact scene (see
[Private extension fields](#private-extension-fields)).

```bash
# build from the committed snapshot (370 tasks); the environment version is
# required and names the robocasa+robosuite checkout that sampled the scenes
python -m alphaapollo.data_preprocess.robotics.prepare_robocasa \
  --environment-version robocasa-921c9a5+robosuite-5ce6643

# or from a locally regenerated snapshot (requires the robocasa runtime)
python -m alphaapollo.data_preprocess.robotics.generate_robocasa_snapshot --output my.jsonl
python -m alphaapollo.data_preprocess.robotics.prepare_robocasa \
  --environment-version <robocasa-commit>+robosuite-<commit> --data-source my.jsonl

# verify the committed snapshot against the runtime (rebuilds every scene
# through the backend's exact factory call and refuses on any mismatch)
python -m alphaapollo.data_preprocess.robotics.generate_robocasa_snapshot --check
```

The private payload follows the robotics envelope contract:
`benchmark="robocasa"` and the required `environment_version` are named
envelope fields feeding `RobotTask.benchmark` and
`RobotTask.environment_version`, and everything else in the payload — the
`(task_name, layout_id, style_id, scene_seed)` triplet and
`max_episode_steps` — is backend task metadata. The backend refuses a scene whose sampled instruction differs
from the prepared one, so the version must be the checkout that generated
the rows — it is task identity, not provenance.

The snapshot rows are generated deterministically (generator seed 42, matching
the ENPIRE arXiv 2606.19980 matched-evaluation protocol) and instructions are
recorded from each built scene, so they embed the sampled object name. The
snapshot is an AlphaApollo-derived task-suite manifest, not the official
RoboCasa demonstration datasets (LeRobot trajectory data for VLA training,
out of scope for this slice). Per-task horizons come from the official
`dataset_registry` of the pinned checkout where the task has an entry; the
env constructor's default applies to the tasks without one, named together
with the protocol, seed, and upstream pins in
`robotics/snapshots/robocasa365_tasks.meta.json`. The prepared `env_payload`
carries each task's horizon as `max_episode_steps`, so episodes terminate on
the task's own budget instead of the env-wide constructor default. Three
instruction texts repeat across task pairs (CoffeeServeMug/PickPlaceCoffee,
ManipulateSinkFaucet/TurnOnSinkFaucet, MicrowavePressButton/TurnOnMicrowave):
upstream `ep_meta` wording collides while the success predicates differ, so the
`statement` alone does not distinguish those six rows — disambiguate by
`task_uid`/`env_payload`.

Regenerating the snapshot requires the robocasa runtime — a source install of
RoboCasa365 v1.0.1 with robosuite **master** (the 1.5.2 wheel lacks kwargs
robocasa calls) plus the ~10 GB kitchen assets; the exact pins, the
network-restricted install routes, and the asset mirror procedure are in
`robotics/UPSTREAM.md`. There is deliberately no pip extra for it: the
source install carries its own robosuite/mujoco/numpy pins, and an extra
naming those would install something that still cannot run the generator.
Preparation itself never starts a simulator, and `--limit` or a non-default
`--seed` must name its own `--output` rather than replace the committed
suite.

## Prepare custom data

Take an input file `questions.jsonl`:

```json
{"id":"net-001","question":"In the netlist line R1 in out 10k, what is the resistance? Reply with the SPICE value.","final_answer":"10k","topic":"netlist","difficulty":"easy"}
```

Build it with the generic preparer and exact text grader:

```bash
python -m alphaapollo.data_preprocess.prepare_custom_data \
  --data-source ./questions.jsonl \
  --question-key question \
  --answer-key final_answer \
  --id-key id \
  --metadata-keys topic,difficulty \
  --dataset-name netlist_reading \
  --domain chips \
  --grader-id exact_match \
  --splits eval \
  --format jsonl \
  --output-root ./data
```

The command prints the build root and writes:

```text
data/netlist_reading/v1/
├── public/
│   └── eval.jsonl
├── private/
│   └── eval.jsonl
└── manifest.json
```

The public row contains the task statement and its stable identity. The private
row holds `answer="10k"`, `grader_id="exact_match"`, metadata and provenance.
`env_payload` and `extra` are canonical JSON strings on disk;
`read_prepared()` decodes them back into mappings. This example checks data
preparation and exact text grading; it does not invoke a circuit simulator.

Read it back in Python:

```python
from alphaapollo.data_preprocess import read_prepared

examples = read_prepared("netlist_reading", "v1", "./data", split="eval")
print(examples[0].statement, examples[0].answer, examples[0].extra)
```

### Arguments

Required:

| Flag | Meaning |
| --- | --- |
| `--data-source` | Hub repo id (`owner/name`), a local `.parquet`/`.jsonl`/`.json` file, or a directory holding one `<split>.<ext>` file per requested split. |
| `--dataset-name` | Name the build is stored and referenced under; also the `source_id` hashed into `task_uid`. It selects the output directory, so a default would let two unrelated builds land on one path. |
| `--question-key` | Column holding the problem statement. Published as `statement`. |
| `--answer-key` | Column holding the gold answer. Kept private. |

Optional:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--id-key` | row position | Column holding a stable per-row id, published as `source_record_id` and hashed into `task_uid`. Strongly recommended: without it, ids are row positions, so reordering the source changes every `task_uid`. |
| `--metadata-keys` | none | Comma-separated columns copied verbatim into the private `extra` map. Values must be JSON-serializable. |
| `--output-root` | `$ALPHAAPOLLO_DATA_DIR` or `~/.alphaapollo/data_preprocess` | Root the build is written under. |
| `--dataset-version` | `v1` | Version directory under the name. |
| `--revision` | resolved | Pin a Hub commit. Refused for a local source, which is fingerprinted by content and has no commit to name. |
| `--splits` | `train` | Comma-separated split names to build. |
| `--format` | `parquet` | `parquet`, `jsonl`, or `json`. |
| `--if-exists` | `reuse` | `reuse`, `refresh`, or `error`. |
| `--domain` | `custom` | Public `domain` tag on every row, e.g. `chips` or `code`. |
| `--grader-id` | `exact_match` | Private `grader_id` recorded on every row, naming the grader an evaluation run should score these answers with. Registered text grader: `exact_match`. `environment_success` is registered too, but it is not a text grader: the execution backend's success predicate grades those rows at rollout, and the offline scorer refuses them with that explanation. Nothing validates arbitrary ids at build time, so a typo surfaces only when a run tries to grade. |
| `--dataset-license` | `unverified` | License recorded in the private provenance columns. |

### Refused combinations

A metadata key may not repeat a question, answer, or id column. The build is
refused before anything is read:

```text
ValueError: metadata_keys repeats protected columns: ['final_answer']
```

This exists because `extra` is a verbatim copy of the named columns. It is
private, so a repeated answer column would not leak, but it would silently store
the gold answer twice under two names and let a later `extra`-forwarding step
resurface it. Name the column once, in the role it belongs to.

The answer column may also not be reused as the question or the id column,
because both of those are published in the public row:

```text
ValueError: answer_key 'final_answer' is also used as ['id_key']; those columns are published in the public row and must not carry the answer
```

A `--revision` may not accompany a local `--data-source`. Both sides are named,
before the source is read:

```text
ValueError: revision '0000...' was given for the local source './questions.jsonl'; a local file has no Hub commit and is fingerprinted by its content, so recording the revision would claim provenance the build does not have
```

Finally, `--id-key` is required as soon as `--splits` names more than one split,
because the fallback id is the row's position *within its split* and would
collide across splits:

```text
ValueError: id_key is required when preparing more than one split; without it record ids fall back to per-split row positions, which collide across splits
```

Duplicate ids in the source are caught later by the build itself, as
`duplicate task_uid '<hash>'`.

## Row digests are evidence, file digests are enforced

A prepared build carries two kinds of SHA-256, and only one of them is checked.

| Digest | Covers | Verified? |
| --- | --- | --- |
| `manifest.json` → `splits.<split>.public_sha256` / `private_sha256` | The bytes of each written `public/` and `private/` split file | **Yes.** `open_dataset()` rehashes both files on every open and raises `prepared dataset digest mismatch for <path>` before returning. Every read path goes through it, so a truncated, corrupted, or hand-edited split file cannot be read silently. |
| private row column `raw_digest` | The canonical JSON of the single source row that example was normalized from | **No, deliberately.** Nothing in the codebase reads it back. |

`raw_digest` looks like a safeguard and is not one. It is written on every
private row and never verified, on purpose — do not "fix" that by adding a
check, and do not delete it as a dead field.

Automatic verification is impossible without re-fetching the source. A prepared
build keeps the digest but not the row it was computed from, so there is nothing
local to compare against; the only way to recompute it is to pull the upstream
data again, normalize it again, and hash it again — which is a rebuild. An
"integrity check" that rebuilds the dataset is just the rebuild, so there is no
cheap check available to add.

What the field is for is forensics after a suspicion, not prevention before one.
The failure it addresses is an upstream dataset silently editing rows while
keeping the same revision — the one case `source_revision`, `source_fingerprint`,
and `build_id` cannot catch, because all three would be unchanged. The workflow
is manual and deliberate:

1. Keep the existing build.
2. Rebuild the same dataset from the same revision into a different
   `--output-root` (or a new `--dataset-version`).
3. Diff the `raw_digest` column of the two private splits, joined on `task_uid`.
4. Every row whose digest differs is a row upstream changed, named exactly.

Without the column, step 4 could only report that some answer somewhere now
differs; with it, the changed source rows are identified directly. That is worth
the 64 bytes per row, which is why the field stays.

## Schema versions are recorded and enforced

A prepared build states two versions in `manifest.json`, and both are checked on
every read:

| Field | Describes | Checked by |
| --- | --- | --- |
| `schema_version` | The manifest's own field set | `read_manifest()`, before the file is parsed against the model |
| `row_schema_version` | The shape of the rows in `public/` and `private/` | `open_dataset()`, beside the file digests |

`build_id` already folds the row schema in, so bumping it does produce a
different fingerprint — but a reader holding a prepared directory learns that
only by recomputing the fingerprint from a source it no longer has. Recording
the version lets any open answer the question directly: do these rows match the
code about to read them?

A mismatch is refused, naming both versions:

```text
ValueError: prepared dataset /data/geometry_drills/v1 was written under row schema version 1, but this code reads row schema version 2; rebuild the dataset (prepare with if_exists='refresh')
```

Refusing is deliberate rather than adapting: there is no migration path, no
in-repository consumer pins a row schema, and `read_prepared()` is the only
reader. Rebuilding is cheap, and `--if-exists refresh` rebuilds over a stale
directory without opening it, so the refusal never blocks the fix it asks for.

A manifest written before `row_schema_version` existed is refused the same way,
by its `schema_version`. Such a build cannot say which row shape its files hold,
so nothing can establish that they match the current code; every dataset
prepared before this field existed must be rebuilt.

## Private extension fields

Two extension columns are written on every private row. Only the LIBERO
preparer fills one of them today; the rest leave both empty, but they are
reserved, not forgotten, so do not remove them while cleaning up dead fields. They survived the removal of `answer_space` and
`grader_params` because each has a named consumer.

| Field | For | Who would fill it | Who would read it | Today |
| --- | --- | --- | --- | --- |
| `env_payload` | Environment-private state: gold needed for in-loop RL reward, and the construction data an execution backend needs | A preparer for a dataset that needs in-loop environment state, such as `robotics/prepare_libero.py` | The Workflow private-sidecar loader; `DefaultEnvironment._gold_from_payload` is one consumer and reads `context.task_payload["answer"]` when a grader is configured, and the robotics integration layer is another, mapping the payload into its robot task contract rather than inferring the benchmark or simulator version from provenance fields | LIBERO fills it with explicit benchmark and environment-version fields plus the suite, BDDL, initial-state, and demonstration task spec. The other shipped preparers write `{}`. The transport is wired, but a run must explicitly point `dataset.task_payload_path` at one concrete private split file. |
| `reference_solution` | The worked gold derivation, as distinct from the final `answer` | A preparer for a dataset that ships solutions alongside answers | The SFT export, which skipped tasks whose reference solution was empty | Always `""`. That export has been removed, so nothing reads it. |

If you are building the training path, fill `env_payload` in the preparer's
normalizer and configure `dataset.task_payload_path` as the exact
`private/<split>.<suffix>` file. Private rows deliberately contain no `split`
column: the path selects the split, while `dataset.split` applies only to the
public input. The `direct` envelope decodes `env_payload` as an object and joins
it to public rows by the configured `id_key`; `task_payload_format` is needed
only when the sidecar format differs from the public file. The generic
scored-evaluation path does not use this channel — it grades after the fact from
the private `answer` column and never configures a grader on the environment.

## ATIF trajectories for SFT

The [offline Pi exporter](../../docs/chips/HARBOR.md#offline-pi-trajectory-export)
creates the initial supported source. Use a private selection file to name every
trajectory, split and task family. Paths are relative to that file:

```json
[
  {"trajectory": "task-trial/trajectory.json", "split": "train", "task_group": "triangle-family"}
]
```

```bash
python -m alphaapollo.data_preprocess.prepare_atif \
  --selection /private/exports/selection.json --output /private/datasets/circuits-v1 \
  --reasoning exclude --allow-reconstructed-context
```

Install `.[data,harbor]` for this conversion. Select `include` to prepend recorded
reasoning as a `<think>` block, or `exclude` to omit it. The choice is recorded in
the manifest; no reasoning is invented. Reconstructed native context needs the
explicit acknowledgement above. Multimodal, continued, subagent and copied or
deterministic steps are unsupported and rejected.

Selection includes every assistant step in each chosen trajectory, including
unsuccessful attempts followed by repairs. A high final reward does not label
every earlier action as optimal. Reward, candidate identity and source hashes
are metadata, never training messages. Scores must carry independently verified
metadata from a trusted export, including valid zero rewards. Explicit selection
may also include complete unscored trajectories with a null reward. The converter
does not authenticate arbitrary edited ATIF files. Choose task families so
related variants share one group. Groups cannot span train/validation/test;
repeated task IDs cannot be assigned different groups, even across versions.
Duplicate trajectory/session/Trial identities are rejected.

Each split has structured `messages`, `tools`, `extra_info` in JSONL and the same
values encoded as `messages_json`, `tools_json`, `extra_info_json` in Parquet.
JSON columns preserve heterogeneous tool arguments and genuine nulls without
Arrow adding keys from other rows. `manifest.json` records file hashes,
selection policy and provenance. Its packaged JSON Schema lives in
`json/atif_dataset_manifest.json`. Outputs are private and refuse overwrite.

Use the existing pinned verl custom dataset hook:

```yaml
data:
  custom_cls:
    path: pkg://alphaapollo.data_preprocess.atif_dataset
    name: AtifSFTDataset
  truncation: error
  max_length: 65536
  pad_mode: right
```

Set the normal trainer train/validation paths to the exported Parquet files.
The adapter requires a fast text tokenizer whose template supports the selected
tools. It renders the full conversation once, checks stable assistant prefixes,
and maps character spans through tokenizer offsets to `loss_mask`. System,
user, tool observations and padding receive zero loss; assistant completions,
tool calls and their ending markers receive loss. Templates that rewrite earlier
context or merge a token across a context/assistant boundary fail visibly.
Overlength sequences fail instead of silently losing tool results or answers.
The context limit above is an example; select one supported by the training model.

Before training, validate with tokenizer files already present locally:

```bash
PYTHONPATH=third_party/verl python -m alphaapollo.data_preprocess.validate_atif \
  --dataset /private/datasets/circuits-v1 --tokenizer /private/tokenizers/target \
  --max-length 65536 --output /private/validation/circuits-v1.json
```

This requires the learning dependencies and initialized pinned verl checkout.
It loads the class through verl's import hook and iterates a PyTorch DataLoader,
checks manifest hashes and JSONL/Parquet fidelity, compares exact token IDs with
the full chat template, and records tokenizer/template fingerprint and per-row
context/assistant token counts. It does not load model weights or start the verl
trainer. Validate again after changing the tokenizer or template. Successful
loading is evidence of usable training inputs, not evidence of learning quality.

## Install

Install preprocessing dependencies with:

```bash
python -m pip install -e ".[data]"
```

## When the Hub is unreachable

A host that cannot reach `huggingface.co` fails with a bare
`ConnectionError: ... Network is unreachable`, which does not say what to do.
Point the client at a mirror instead:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

The mirror serves the same commits, so the resulting `build_id` is identical to
one built against the Hub directly — the fingerprint covers the source, revision,
splits, format, and builder version, not the route taken to fetch them.

Behind a SOCKS proxy, `huggingface_hub` also needs `pip install "httpx[socks]"`;
without it the failure names `socksio` rather than the proxy.
