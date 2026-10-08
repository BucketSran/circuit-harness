# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AlphaApollo TaskRunner: subclasses verl TaskRunner, injects envs/traj_collector.

Inherits verl 0.9 TaskRunner and reuses its worker registration (add_actor_rollout_worker /
add_critic_worker / add_ref_policy_worker / init_resource_pool_mgr). Only run() is overridden to
swap the trainer construction for AlphaApolloRayPPOTrainer and inject envs/traj_collector.

Design rules:
  - Adaptation 1: AlphaApolloTrainer -> AlphaApolloRayPPOTrainer (canonical trainer name)
  - Adaptation 2: reuse parent fragments (add_*/init_resource_pool_mgr); do not
    wholesale-copy verl run()
  - RL main chain is pure episode rule reward, no model RM -> do NOT register RewardModel role;
    do NOT pass reward_fn (built inside trainer __init__)
  - verl rollout.n MUST be 1; GRPO grouping is in env.rollout.n
"""

from __future__ import annotations

from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler


class AlphaApolloTaskRunner(TaskRunner):
    def run(self, config):
        from omegaconf import OmegaConf
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        OmegaConf.resolve(config)

        # --- Reuse parent worker registration (actor/critic/ref — do not copy verl run() body) ---
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        # pure rule reward — do NOT register RewardModel role
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # Resource-pool registration helpers (gated no-ops on the normal AA path).
        # Parity with verl parent main_ppo.py:247-249 — ensures Role.RewardModel / TeacherModel
        # are registered in the pool mapping if the corresponding capabilities are enabled.
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)

        # Fail-fast on unsupported capabilities.
        # AA's pure-rule-reward main loop does not build a Role.RewardModel worker or a
        # Role.TeacherModel worker, so launching with these flags enabled would silently
        # produce incorrect training results.  Raise loudly instead.
        from verl.trainer.distillation import is_distillation_enabled

        if config.reward.reward_model.enable:
            raise NotImplementedError(
                "reward_model.enable=True is not supported by AlphaApolloTaskRunner. "
                "AA uses a pure rule-based reward (EpisodeRewardManager) and does not "
                "build a RewardModel worker.  Either disable reward_model.enable or use "
                "the verl parent entry point (verl.trainer.main_ppo) directly."
            )
        if is_distillation_enabled(config.get("distillation")):
            raise NotImplementedError(
                "Distillation is not supported by AlphaApolloTaskRunner. "
                "AA does not build a Role.TeacherModel worker.  Either disable "
                "distillation or use the verl parent entry point (verl.trainer.main_ppo) directly."
            )

        # Validate shared GPU/batch/ref/critic config invariants via the
        # parent validator.  Parity with verl parent main_ppo.py:254-259.
        # Kept independent of the AA-specific rollout.n==1 assert below.
        #
        # verl's validate_config() checks config.data.train_batch_size (a prompt
        # count) against actor.ppo_mini_batch_size (sized in trajectories), and
        # applies an actor_rollout_ref.rollout.n expansion for its own
        # divisibility check. AA pins actor_rollout_ref.rollout.n to 1 (see the
        # assert below) and instead expands the per-step trajectory count via
        # env.rollout.n — TrajectoryCollector repeats each prompt env.rollout.n
        # times before the actor update. So the prompt count verl's validator
        # must compare against ppo_mini_batch_size is
        # train_batch_size * env.rollout.n, not the raw prompt count.
        #
        # Validate against a throwaway copy of the config with that AA-effective
        # batch size substituted in. The real config object passed to dataset /
        # trainer construction below is untouched, so training math is unchanged.
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config

        group_size = config.get("env", {}).get("rollout", {}).get("n", 1)
        validation_config = config
        if group_size != 1:
            validation_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            validation_config.data.train_batch_size = config.data.train_batch_size * group_size

        validate_config(
            config=validation_config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        # --- tokenizer / processor ---
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust)
        processor = hf_processor(local_path, trust_remote_code=trust, use_fast=True)

        # --- AlphaApollo injection: Common per-episode Environment providers ---
        from alphaapollo.learning.on_policy.environment_providers import (
            make_environment_providers,
        )
        from alphaapollo.learning.on_policy.registry import trainer_class_for
        from alphaapollo.learning.on_policy.rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(
            config=config,
            tokenizer=tokenizer,
        )

        # GRPO grouping is via env.rollout.n; verl rollout.n MUST be 1
        assert config.actor_rollout_ref.rollout.n == 1, (
            "GRPO 通过 env.rollout.n 实现，verl rollout.n 必须为 1"
        )

        # --- datasets ---
        # Pass max_samples from config — parity with verl parent main_ppo.py:280-295.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # --- build the trainer (per-algorithm subclass, selected by adv_estimator) ---
        # do NOT pass reward_fn — EpisodeRewardManager is constructed inside trainer __init__
        TrainerCls = trainer_class_for(config.algorithm.adv_estimator)
        envs, val_envs = make_environment_providers(config)
        try:
            trainer = TrainerCls(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
                # AlphaApollo multi-turn injection
                traj_collector=traj_collector,
                envs=envs,
                val_envs=val_envs,
            )
            # super().init_workers() internally wires LLMServerManager -> HYBRID.
            # AlphaApolloRayPPOTrainer.init_workers() then injects the backend into collector.
            trainer.init_workers()
            trainer.fit()
        finally:
            # Learning owns the long-lived Common worker pools; Reasoning owns
            # only the per-trajectory Environment wrappers created from them.
            envs.close()
            val_envs.close()
