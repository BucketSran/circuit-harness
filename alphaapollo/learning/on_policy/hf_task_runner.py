"""AlphaApolloHFTaskRunner: verl TaskRunner that registers AAHFRolloutWorker.

For custom architectures vLLM can't load; generation runs on the in-repo HF rollout
(`rollout.name='hf'`, see `hf_rollout_worker.py`). Mirrors the
parent `add_actor_rollout_worker` (`verl/trainer/main_ppo.py:126-146`) exactly, except it swaps
`ActorRolloutRefWorker` -> `AAHFRolloutWorker`.

`run()` is inherited verbatim from verl's TaskRunner — it builds the **native RayPPOTrainer**
(single-turn GRPO). No trainer swap is needed: set
`actor_rollout_ref.model.use_remove_padding=false`
so the engine feeds the model padded `[bs,seq]` batches (the nested/jagged path is incompatible
with non-transformer arches), and `rollout.name=hf` for generation.
"""

from __future__ import annotations

from verl.trainer.main_ppo import TaskRunner


class AlphaApolloHFTaskRunner(TaskRunner):
    """TaskRunner that uses AAHFRolloutWorker (enables rollout.name='hf')."""

    def add_actor_rollout_worker(self, config):
        import ray
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        from verl.trainer.ppo.utils import need_reference_policy

        from alphaapollo.learning.on_policy.hf_rollout_worker import AAHFRolloutWorker

        actor_rollout_cls = AAHFRolloutWorker
        ray_worker_group_cls = RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = (
            lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )
        # GRPO without KL -> no reference policy -> Role.ActorRollout (not ActorRolloutRef).
        if need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls
