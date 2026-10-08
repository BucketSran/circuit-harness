# Trajectory data

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
python -m circuit_harness.data.prepare_atif \
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

Load exported Parquet with `circuit_harness.data.atif_dataset.AtifSFTDataset`
in an external trainer. Its constructor accepts Parquet paths, tokenizer and a
configuration with `max_length`, `truncation: error` and `pad_mode: right` or
`no_padding`. The constructor remains compatible with external custom dataset
hooks, but Circuit Harness does not install or pin a trainer.

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
python -m circuit_harness.data.validate_atif \
  --dataset /private/datasets/circuits-v1 --tokenizer /private/tokenizers/target \
  --max-length 65536 --output /private/validation/circuits-v1.json
```

Install `.[harbor,sft]` for validation. It loads AtifSFTDataset directly and
iterates a PyTorch DataLoader,
checks manifest hashes and JSONL/Parquet fidelity, compares exact token IDs with
the full chat template, and records tokenizer/template fingerprint and per-row
context/assistant token counts. It does not load model weights or start a
trainer. Validate again after changing the tokenizer or template. Successful
loading is evidence of usable training inputs, not evidence of learning quality.
