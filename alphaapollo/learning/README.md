# Learning runtime

Owner: Learning group. This package owns the optional verl-facing training
runtime while preserving the dependency direction `learning -> common -> schema`.

- `adapters/` restores durable Common inference manifests into verl `DataProto`.
- `common.environment.provider` owns reusable benchmark resources and vends one
  `BaseEnvironment` per trajectory; `on_policy/environment_providers.py` sizes
  the train/validation providers from Learning config; Reasoning owns batching
  and scheduling. Applications explicitly register their pool factories with
  `register_environment_provider`; this branch bundles no task-specific pools.
- `on_policy/` contains the multi-turn rollout, episode reward attribution, trainer
  extension hooks, and algorithm plugin boundary. For architectures vLLM cannot load,
  `on_policy/` also carries the HF-generate rollout path: `hf_rollout_worker.py`
  (standalone `HFServerActor` generation server + chunked CUDA-IPC weight push),
  `hf_task_runner.py`, `main_on_policy_hf.py`, and the `configs/ppo_trainer_hf.yaml`
  overlay. Runbook: `docs/runbooks/pretrain-usage.md`.
- `off_policy/` contains the verl SFT/DPO entry and preference-optimization extension
  seam. Its shared overlays are `configs/sft_trainer.yaml` and
  `configs/dpo_trainer.yaml`. `off_policy/algorithms/dpo.py` owns offline DPO loss and trainer wiring; its
  paired dataset and frozen-reference cache live in `data_preprocess/`. The Megatron
  pretrain subpackage `off_policy/pretrain/` is a verl-free leaf (see its module map
  below).
- This branch keeps the shared model-parameter learning runtime. The separate
  AlphaHebe artifact-search application remains in private history and backups.
- `export/` reserves an owner-local extension point.

## `off_policy/pretrain/` module map

From-scratch / continued pretraining on Megatron-LM (mcore 0.18 + Transformer Engine).
The only artifact crossing to RL/SFT is the HuggingFace checkpoint its converters write;
the package imports no verl. Entry: `main_pretrain.py` (Megatron args → `adapters.py`
wires the registries). `checkpoint_bridge.py` is the documented runbook shim.

- `base/` — the ABCs and the `Registry` every component family registers into.
- `archs/` — model builders: `gpt` (Qwen-arch TE/local), `bert`, `causal_mlp`
  (attention-free conv LM), `example_custom`/`myarch` (templates for your own).
- `forward/` — forward+loss steps: `causal_lm`, `masked_diffusion` (pairs with `bert`).
- `datasets/` — corpus providers: `gpt_mmap` (real), `mock` (smoke).
- `tokenizers/` — `huggingface`, `null` (vocab-space smokes).
- `converters/` — mcore→HF export: `qwen2` (default), `decoder` (llama/qwen2/mistral),
  `mapping` (declarative JSON), `causal_mlp`, `bridge` (megatron-bridge route), plus
  `parity_check.py` (numerical gates). `hf_models/` ships the modeling file the
  CausalMLP `auto_map` names (packaged in the wheel); `examples/*.json` are
  repo-checkout reference templates and do not ship.

Adding an architecture is one module under `archs/` (or `custom:<module>:<Class>`) plus,
for HF export, a converter and an HF modeling file — see `docs/runbooks/pretrain-usage.md`
scenario C. The packaged registry surface is pinned by
`tests/learning/off_policy/test_registry_completeness.py`.

The migrated runtime comes from `branch/learning/file_example` at
`1eaa9d1fdd38bc0edba40673bff5b2fa33ebc92f`; the rollout and episode reward source
was the same branch's former `alphaapollo/core/generation/multi_turn_rollout/` and
`alphaapollo/core/reward_manager/`. Those capabilities now live wholly inside
Learning. Algorithm implementations are deliberately excluded from this shared
base and land in their own PRs.

Generation no longer owns rollout capture. The on-policy collector therefore
returns its live `DataProto` directly; durable training capture is deferred to a
separate Learning-owned boundary. `DataProtoAdapter` continues to restore
existing schema-valid `InferenceCaptureRecord` archives without requiring the
retired Common recorder API.
