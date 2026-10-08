"""AlphaApollo HF-rollout PPO entry — RL for custom architectures vLLM can't load.

Thin hydra @main that invokes verl's `run_ppo` with an injected `AlphaApolloHFTaskRunner`
(which registers `AAHFRolloutWorker` — an `ActorRolloutRefWorker` whose rollout is the in-repo
`HFServerAdapter` driving a standalone `HFServerActor` over Ray). The trainer is verl's
native `RayPPOTrainer` (single-turn GRPO) — no env block, no trainer swap.

Launch (custom HF model, e.g. the CausalMLP conv LM):

    python -m alphaapollo.learning.on_policy.main_on_policy_hf \\
        actor_rollout_ref.model.path=/tmp/causal_mlp_hf \\
        actor_rollout_ref.model.trust_remote_code=true \\
        +actor_rollout_ref.model.override_config.attn_implementation=sdpa \\
        reward.custom_reward_function.path=/tmp/pipeline_reward.py \\
        data.train_files=/tmp/pipeline_prompts.parquet \\
        trainer.total_training_steps=2 trainer.n_gpus_per_node=2 ...

`rollout.name=hf`, `use_remove_padding=false`, and the GRPO defaults come from the
`ppo_trainer_hf` overlay (see `configs/ppo_trainer_hf.yaml`).

Heavy imports (ray, run_ppo, the task runner) live inside `main_entry` so the module stays
import-safe for `--cfg job` introspection.
"""

from __future__ import annotations

import hydra


def main_entry(config):
    """Hydra entry: run verl PPO with AlphaApolloHFTaskRunner (HF rollout for custom arches)."""
    import os

    import ray
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.device import auto_set_device

    from alphaapollo.learning.on_policy.hf_task_runner import AlphaApolloHFTaskRunner

    # Multi-GPU WAR: verl's global process-group init (engine_workers.py:88, called with
    # timeout_second=None so it NEVER times out) deadlocks on NCCL 2.27's P2P transport probe (one
    # rank's ncclCommInitRankConfig stalls mid-transport negotiation while the other completes — a
    # pure race). NCCL_P2P_DISABLE=1 forces SHM transport and sidesteps it. Auto-enable for >1 GPU
    # (the HF-rollout path targets custom/small models where SHM-vs-NVLink bandwidth is moot) unless
    # the user already set NCCL_P2P_DISABLE. Driver os.environ propagates to Ray workers (verified).
    n_gpus = int(getattr(config.trainer, "n_gpus_per_node", 1)) * int(
        getattr(config.trainer, "nnodes", 1)
    )
    if n_gpus > 1 and os.environ.get("NCCL_P2P_DISABLE") is None:
        os.environ["NCCL_P2P_DISABLE"] = "1"

    # Mirrors verl's parent main(): device + reward-config migration must run before run_ppo.
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)

    runner_cls = ray.remote(num_cpus=1)(AlphaApolloHFTaskRunner)
    run_ppo(config, task_runner_class=runner_cls)


# Single-turn GRPO overlay for custom architectures: rollout.name=hf +
# use_remove_padding=false + HF engine knobs (see configs/ppo_trainer_hf.yaml,
# layered on the shared alphaapollo_on_policy base).
@hydra.main(config_path="configs", config_name="ppo_trainer_hf", version_base=None)
def main(config):
    main_entry(config)


if __name__ == "__main__":
    main()
