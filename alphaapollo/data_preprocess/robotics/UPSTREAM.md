# Upstream provenance for the robotics preparers

Each benchmark under this package records its own upstream pins, license
boundary, and asset story below. The sections are independent: a preparer
documents only the upstream it reads, and neither vendors upstream source.

## LIBERO upstream and provenance boundary

`prepare_libero.py` has two explicit source modes. The metadata compatibility
path normalizes caller-supplied rows; the canonical checkout path generates
rows from the selected official `get_benchmark_dict()` API. Neither mode
vendors LIBERO source code, BDDL assets, initial-state files, or demonstration
HDF5 files, and neither downloads or starts the simulator.

Upstream surfaces describe different artifacts under different licenses:

- Official project: https://github.com/Lifelong-Robot-Learning/LIBERO
- The repository `LICENSE` covers the LIBERO code under MIT.
- The repository README states that the released LIBERO datasets use
  CC BY 4.0.
- The Hugging Face card at
  https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets declares
  Apache-2.0 in its metadata.

Those statements do not establish the license of every local or Hub metadata
export accepted by this preparer, nor do they establish that one license applies
equally to code, task descriptions, demonstrations, and a downstream metadata
table. The preparer therefore records `unverified` for dataset and original
content provenance by default. Callers may set explicit values only when their
selected source supplies the corresponding evidence.

`environment_version` is separate from source provenance. It is a required
build-time identity for the official LIBERO checkout or release used to execute
the prepared tasks. `source_revision` instead identifies the metadata source:
the resolved Hub revision for a Hub dataset, `local` for a local metadata file,
or the explicit/resolved Git revision for a canonical checkout. `source_uri`
is resolved the same way: the `file://` URI of a local source, the Hub dataset
URL, or the official project URL for canonical rows. Row-level `source_revision`
and `source_uri` claims are retained in private `source_metadata` but cannot
override those build-level identities. Canonical rows do not accept
caller-supplied task fields; they are generated from the checkout rather than
compared back to it, while the metadata path validates its rows against an
official checkout named by `checkout_path` or installed as `libero`.

The official benchmark `Task` record uses `language` for the natural-language
instruction and a separate constant `problem="Libero"` field. The preparer
therefore preserves that official `problem` value as source metadata instead of
treating it as an instruction alias. For compatibility with older exports, a
non-`Libero` `problem` value is accepted as a fallback instruction only when no
higher-confidence instruction field is present. `get_task_demonstration()`
exposes a relative path rather than a distinct demonstration id; path-only rows
keep that path private and use a tagged opaque identity component without
fabricating provenance. Demonstration datasets are a separate download from the
code checkout, so canonical rows keep demonstrations out of their identity
entirely: a checkout without them prepares the same `task_uid`s as one with
them.

LIBERO itself is not a declared package dependency. Its `setup.py` declares no
requirements while its benchmark module imports `torch`, and its BDDL,
initial-state, and asset directories are not Python packages and ship no
`MANIFEST.in`, so a pip install carries neither the imports nor the data files
this preparer reads. Clone the official repository and pass `checkout_path`
(`--checkout-path`) instead.

## LIBERO-Pro distribution boundary

`prepare_liberopro.py` targets the RLinf distribution `rpent-liberopro==0.1.1`
and records `https://github.com/RLinf/LIBERO-PRO` as the canonical source URI.
The supported surface is intentionally narrower than the package registry: only
the 16 static perturbation suites formed by `spatial`/`object`/`goal`/`10`
crossed with `swap`/`task`/`lan`/`object` are accepted. Base LIBERO suites,
historical trigger/OOD suites, and dynamic `*_temp` suites are different
contracts and are refused.

The distribution metadata labels the package MIT and declares the simulator
stack as dependencies. That package license does not by itself establish the
license of every BDDL task, initial-state file, generated metadata export, or
downstream prepared dataset. The preparer therefore leaves dataset and original
content provenance as `unverified` unless the caller has separate evidence.

The canonical package mode reads task records and initial-state counts through
`liberopro.liberopro.benchmark.get_benchmark_dict()`. It does not start an
environment, copy packaged assets, or hash selected state tensors. Metadata mode
cross-checks caller-supplied task identity against the same pinned API. Both
modes keep package version and environment version explicit so a row prepared
for another distribution cannot be accepted as executable LIBERO-Pro data.

The `0.1.1` wheel's registry is a rename fork of LIBERO's: it registers the
five official base suites (`libero_spatial`, `libero_object`, `libero_goal`,
`libero_90`, `libero_10`) alongside the perturbation suites, and its base-suite
tasks carry the same `name`, `bddl_file`, and `init_states_file` values as the
official ones while resolving against its own asset tree (both packages build
every task with `problem_folder=<suite name>`, and `get_libero_path` reads
`~/.liberopro/config.yaml` rather than `~/.libero/config.yaml`). A base-suite
row therefore cross-checks cleanly against the wrong package. That is why
`benchmark` — not the suite name and not the task fields — is what selects the
package in `common/execution/robotics/backends/libero.py`, and why the
`package_distribution`/`package_version` fields this preparer writes are read
back at reset.

The published `0.1.1` wheel was measured on August 18, 2026. Four registered
tasks deserialize to zero initial states: `libero_spatial_task` indexes 3 and 7,
`libero_10_task` index 2, and `libero_10_object` index 4. The preparer records
that defect beside its suite allowlist and refuses those rows. It does not skip
them or publish a canonical suite with a silently smaller task set.

## RoboCasa upstream and provenance boundary

This file records what generated
`snapshots/robocasa365_tasks.jsonl` and what it takes to regenerate it; the
execution backend that consumes the triplets is owned by the execution
slice of issue #252.

- Upstream project: RoboCasa — https://github.com/robocasa/robocasa (MIT license)
- The derived snapshot retains upstream task instructions and registry horizon
  fields. Their RoboCasa MIT attribution and the full pinned license are
  preserved in [third_party/robocasa/LICENSE](../../../third_party/robocasa/LICENSE).
  This does not extend the source-code license to downloaded datasets or assets.
- Version pinned for snapshot generation: **robocasa@921c9a5** (the
  checkout the execution backend validates against; the committed snapshot
  was regenerated on it) + **robosuite@5ce6643**. An earlier build on the
  v1.0.1 `main` tarball (sha256
  `dd7e31d53e2fca5bb11eda08ef873ae994dca03703b51c33676fd15280390f74`) is
  what the `robocasa` conda env first installed; same asset overlay.
- Backend stack: **robosuite master** (the 1.5.2 PyPI wheel is insufficient —
  robocasa 1.0.1 calls `load_model_on_init` and other master-only kwargs, so
  the tarball above's asserts are a lower bound only), **mujoco 3.3.1**,
  **numpy 2.2.5** (the latter two asserted by robocasa at import).
- Environment creation goes through `robocasa.utils.env_utils.create_env`
  (robosuite `make` with `layout_ids`/`style_ids`/`seed`); the
  `gym.make("robocasa/<task>")` gymnasium wrapper is PandaOmron/GR00T-oriented
  and is intentionally not used.
- Assets: kitchen scene/object assets (~10 GB) are downloaded at runtime via
  `python -m robocasa.scripts.download_kitchen_assets` (source: utexas.box.com);
  keep them on durable storage, never a wiped-on-reboot temp directory.
- Headless rendering on GPU servers runs with `MUJOCO_GL=egl` (see
  `configure_render_backend`).
- This package is original AlphaApollo code written against the upstream
  public APIs (scene `layout_id`/`style_id`/seed config, `ep_meta["lang"]`
  instructions, `_check_success` predicates). No upstream source is vendored
  or adapted here.
- Evaluation protocol alignment: the task-suite snapshot (see
  `generate_robocasa_snapshot.py` in this package) uses fixed
  (scene_seed, layout_id, style_id) triplets derived from generator seed 42,
  matching the ENPIRE (arXiv 2606.19980) RoboCasa365 evaluation protocol.
  The committed rows are prepared with
  `environment_version="robocasa-921c9a5+robosuite-5ce6643"`, and the
  generator's phase-two build replicates the backend factory call so the
  recorded instructions validate against it by construction.
- Derived manifest, not the official datasets: the snapshot is an
  AlphaApollo task-suite manifest for evaluation; the official RoboCasa
  demonstration datasets (LeRobot trajectory data for VLA training —
  pretraining 300 tasks, target 50 tasks) are a different artifact and out
  of scope for this slice. Provenance (protocol, seed, pins, horizon
  sourcing) is recorded in `snapshots/robocasa365_tasks.meta.json` beside
  the rows.
- Per-task horizons come from the official `dataset_registry` that ships
  with the pinned robocasa checkout (offline, no dataset download): 317
  tasks have an official entry; the remaining 53 (composite tasks without
  a dataset, named in the sidecar meta) keep the env constructor default.
  `generate_robocasa_snapshot.py --check` re-derives every committed row
  through the factory call and refuses on any mismatch.
