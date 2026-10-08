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

"""AlphaApollo's fit() loop: every-step-one-row data model, reward pre-baked by the
rollout, advantage left on verl's native compute_advantage path.

The only structural change relative to verl's parent fit() is the rollout/union
step: instead of `batch.repeat(n).union(gen_batch_output)`, the multi-turn rollout
already returns one row per (trajectory, step), so `batch` is replaced outright by
`gen_batch_output`. Advantage computation stays verl-native and, for the default
estimators, uses the same ``compute_advantage`` call as verl's own fit loop.
"""

from __future__ import annotations

from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from . import metrics as aa_metrics
from .reward_manager import penalty as algos
from .rollout.provenance import synchronize_rollout_weights, tokenizer_fingerprints


class AlphaApolloRayPPOTrainer(RayPPOTrainer):
    # ------------------------------------------------------------------ __init__
    def __init__(self, *args, traj_collector=None, envs=None, val_envs=None, **kwargs):
        self.traj_collector = traj_collector
        self.envs = envs
        self.val_envs = val_envs
        super().__init__(*args, **kwargs)

        # verl 0.9's base trainer does not accept reward_fn, so build the
        # rule-reward EpisodeRewardManager here (no model RM in the RL main loop).
        from alphaapollo.learning.on_policy.reward_manager.episode import (
            EpisodeRewardManager,
        )

        aa = self.config.get("alphaapollo", {})
        self.reward_fn = EpisodeRewardManager(
            self.tokenizer, num_examine=0, normalize_by_length=aa.get("normalize_by_length", False)
        )

    # ---------------------------------------------------------------- algo hooks
    # Template-method seams so per-algorithm trainers (algorithms/<algo>/) plug in
    # without forking the parity-critical fit() loop. Base defaults = vanilla GRPO:
    # _before_rollout is a pass-through, _compute_advantage is a byte-identical call
    # to verl's native compute_advantage. Subclasses override only these hooks.
    def _before_rollout(self, batch: DataProto, batch_idx: int) -> DataProto:
        """Transform the PRE-rollout batch (default: identity). batch_idx is the raw
        dataloader-yield counter (0-based, persists across epochs)."""
        return batch

    def _postprocess_trajectories(self, trajectories, *, is_train: bool):
        """Algorithm hook over the stable Learning trajectory contract."""

        return trajectories

    def _compute_advantage(self, batch: DataProto, batch_idx: int, metrics: dict) -> DataProto:
        """Compute advantages (default: verl-native compute_advantage).

        Runs after token_level_scores / penalty / kl are set in fit(). `metrics` is
        passed so algorithm hooks can emit selection/diagnostic metrics.
        """
        return compute_advantage(
            batch,
            adv_estimator=str(self.config.algorithm.adv_estimator),
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

    # ------------------------------------------------------------------ init_workers
    def init_workers(self):
        super().init_workers()
        # Learning owns the HYBRID rollout workers, server manager, and weight
        # synchronization. Common only wraps the already-running client; the
        # resulting backend is injected into Reasoning through Learning's single
        # rollout-consumer boundary.
        from alphaapollo.common.generation import VerlLLMServerGenerationBackend

        model_name = str(self.config.actor_rollout_ref.model.path)
        tokenizer_id = str(getattr(self.tokenizer, "name_or_path", model_name))
        tokenizer_fingerprint, chat_template_fingerprint = tokenizer_fingerprints(
            self.tokenizer,
            tokenizer_id,
        )
        backend = VerlLLMServerGenerationBackend(
            client=self.llm_server_manager.get_client(),
            tokenizer=self.tokenizer,
            model_name=model_name,
            tokenizer_id=tokenizer_id,
            tokenizer_fingerprint=tokenizer_fingerprint,
            chat_template_fingerprint=chat_template_fingerprint,
            actor="policy",
            max_prompt_tokens=int(self.config.data.max_prompt_length),
            prompt_truncation=str(self.config.data.get("truncation", "error")),
        )
        self._generation_backend = backend
        self._rollout_sync_count = 0
        self.traj_collector.set_generation_backend(backend)
        self.traj_collector.set_trajectory_postprocessor(self._postprocess_trajectories)

    def _sync_rollout_weights(self) -> None:
        """Synchronize rollout replicas, then publish the version they now serve."""

        self._rollout_sync_count = synchronize_rollout_weights(
            checkpoint_manager=self.checkpoint_manager,
            generation_backend=self._generation_backend,
            global_step=self.global_steps,
            previous_sync_count=self._rollout_sync_count,
        )

    # ------------------------------------------------------------------ fit
    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        self._load_checkpoint()
        self._sync_rollout_weights()

        if self.config.trainer.get("val_before_train", True):
            logger.log(self._validate(), step=self.global_steps)
        self.global_steps += 1
        last_val_metrics = None

        # Raw dataloader-yield counter (0-based, persists across epochs). Passed to
        # the algorithm hooks; algorithms may use it to coordinate adjacent yields.
        batch_idx = 0

        for _epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics, timing_raw = {}, {}
                cur_idx = batch_idx
                batch_idx += 1
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(__import__("uuid").uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                # Live pre-rollout uid, consumed by GRPO advantage grouping (and by
                # algorithm hooks below). Regenerated every iteration -- any uid
                # reasoning must key off the uid VALUE, not row position (verl
                # reorders rows via _balance_batch).

                # Algorithm hook: transform the PRE-rollout batch before
                # _get_env_gen_batch pops env_kwargs for rollout. Default = identity;
                # Environment kwargs are still intact at this hook boundary.
                batch = self._before_rollout(batch, cur_idx)

                gen_batch = self._get_env_gen_batch(batch)
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # ============ rollout seam ============
                    self._sync_rollout_weights()
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            envs=self.envs,
                            is_train=True,
                        )  # one row per (trajectory, step): uid/traj_uid/rm_scores/
                        # input_ids/attn/resp_mask/pos_ids/multi_modal_inputs (may be empty)
                    self.checkpoint_manager.sleep_replicas()
                    # ============ replace batch with the rollout output ============
                    batch = (
                        gen_batch_output  # replaces verl's batch.repeat(n).union(gen_batch_output)
                    )

                    # The remaining verl helpers are row-agnostic.
                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Every-step-one-row means a variable row count per batch; verl's
                    # dp dispatch chunks by dp_size and update_actor slices by
                    # ppo_mini_batch_size, both of which require divisibility. Pad to
                    # the next multiple of ppo_mini_batch_size (itself a multiple of
                    # dp_size), and zero the response_mask of the padding rows so
                    # they don't contribute to loss/advantage.
                    _pad_divisor = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    batch, _pad_size = pad_dataproto_to_divisor(batch, _pad_divisor)
                    if _pad_size > 0:
                        batch.batch["response_mask"][-_pad_size:] = 0
                        # Neutralize pad rows so they don't bias GRPO groups.
                        # pad_dataproto_to_divisor copies data[:take_size], so each pad
                        # row duplicates a real row's uid AND rm_scores. Zero rm_scores
                        # before token_level_scores=reward_tensor below so
                        # token_level_scores inherits the zeros, and give each pad row
                        # a unique deterministic sentinel uid so it forms its own
                        # singleton GRPO group (verl core_algos.py: len==1 ->
                        # mean=0,std=1 -> advantage=0). The sentinel is deterministic
                        # (not uuid4) so frozen artifacts stay reproducible.
                        if "rm_scores" in batch.batch:
                            batch.batch["rm_scores"][-_pad_size:] = 0
                        for i in range(_pad_size):
                            batch.non_tensor_batch["uid"][-(i + 1)] = (
                                f"__pad_{self.global_steps}_{i}__"
                            )

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    # The actor's compute_log_prob scales logits by 1/temperature to
                    # match rollout sampling (same as verl's native fit()). The
                    # env-driven batch's meta_info doesn't carry temperature, so set
                    # it explicitly here.
                    batch.meta_info["temperature"] = (
                        self.config.actor_rollout_ref.rollout.temperature
                    )

                    # Rollout normally pre-bakes rm_scores.
                    # If multi_turn_loop did not pre-bake rm_scores, fall back to reward_fn here.
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if "rm_scores" in batch.batch:
                            reward_tensor = batch.batch["rm_scores"]
                        else:
                            reward_tensor = self.reward_fn(batch, return_dict=True)["reward_tensor"]

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob, _ = self._compute_old_log_prob(batch)
                        old_log_prob.batch.pop("entropys", None)
                        batch = batch.union(old_log_prob)
                    if self.use_reference_policy:
                        batch = batch.union(self._compute_ref_log_prob(batch))
                    if self.use_critic:
                        batch = batch.union(self._compute_values(batch))

                    # ============ advantage ============
                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["token_level_scores"] = reward_tensor
                        # token_level_scores may be mutated in place by the invalid-action
                        # penalty below; algorithm hooks that need an unshaped difficulty
                        # Algorithm hooks can read the unshaped episode outcome.
                        aa = self.config.get("alphaapollo", {})
                        if aa.get("use_invalid_action_penalty", False):
                            batch, m = algos.apply_invalid_action_penalty(
                                batch, aa.get("invalid_action_penalty_coef", 0.1)
                            )
                            metrics.update(m)
                        if self.config.algorithm.use_kl_in_reward:
                            batch, m = apply_kl_penalty(
                                batch, self.kl_ctrl_in_reward, self.config.algorithm.kl_penalty
                            )
                            metrics.update(m)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Algorithm hook: advantage computation. Default is a
                        # byte-identical call to verl's native compute_advantage (grpo
                        # etc. unaffected); an algorithm may override it to inject `batch` into
                        # its estimator and run even-step top-K selection.
                        batch = self._compute_advantage(batch, cur_idx, metrics)

                        # Episode-level metrics, computed on the UNPADDED view: pad
                        # rows are row-copies that would otherwise inflate per-row
                        # broadcast metrics (e.g. success_rate). Training/advantage/
                        # actor/critic still consume the full padded `batch` (padding
                        # is required for dp/mini-batch divisibility).
                        #
                        # _balance_batch (above) calls batch.reorder() when
                        # balance_batch=true (set in all AA configs), so pad rows are
                        # not guaranteed to be at the tail here -- a tail-slice would
                        # leak scattered pad rows in and drop real tail rows instead.
                        # Filter by the pad sentinel uid (written above, before
                        # balancing, and stable across reorder) instead.
                        if _pad_size > 0:
                            _real_idx = np.array(
                                [
                                    i
                                    for i, u in enumerate(batch.non_tensor_batch["uid"])
                                    if not str(u).startswith("__pad_")
                                ]
                            )
                            _unpadded = batch[_real_idx]
                        else:
                            _unpadded = batch
                        metrics.update(aa_metrics.compute_episode_metrics(_unpadded))

                    # ============ critic / actor update: reuse base class ============
                    if self.use_critic:
                        metrics.update(
                            reduce_metrics(self._update_critic(batch).meta_info["metrics"])
                        )
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            a_out = self._update_actor(batch)
                        metrics.update(reduce_metrics(a_out.meta_info["metrics"]))
                    self._sync_rollout_weights()

                # checkpoint / validate
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    val_metrics = self._validate()
                    last_val_metrics = val_metrics if is_last_step else last_val_metrics
                    metrics.update(val_metrics)

                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    return

    # ------------------------------------------------------ _get_env_gen_batch
    def _get_env_gen_batch(self, batch: DataProto) -> DataProto:
        """Build the gen_batch for the env-driven rollout (verl 0.9 contract).

        verl 0.9 moved tokenization into the rollout/AgentLoop, so RLHFDataset yields
        raw_prompt + a placeholder `dummy_tensor` instead of input_ids. The base
        `_get_gen_batch()` pops zero tensor keys, leaving `gen_batch.batch` empty/None —
        but the AlphaApollo rollout sizes the batch with `len(gen_batch.batch)`
        (multi_turn_rollout/rollout_loop.py). So we pop the sizing tensor
        (`dummy_tensor`) into gen_batch and carry every non-reward non_tensor key
        (env_kwargs, raw_prompt, ...) the env reset / rollout need, re-attaching the
        reward keys afterwards (same key policy as the base helper).
        """
        reward_keys = {"data_source", "reward_model", "extra_info", "uid"} & set(
            batch.non_tensor_batch.keys()
        )
        non_tensor_pop = list(set(batch.non_tensor_batch.keys()) - reward_keys)
        batch_pop = (
            ["dummy_tensor"] if "dummy_tensor" in batch.batch.keys() else list(batch.batch.keys())
        )
        gen_batch = batch.pop(batch_keys=batch_pop, non_tensor_batch_keys=non_tensor_pop)
        # re-attach reward keys (data_source/reward_model/extra_info/uid) for scoring/grouping
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        return gen_batch

    # ------------------------------------------------------------------ _validate
    def _validate(self, merged: bool = False):
        """Validate on val_envs using multi_turn_loop (is_train=False).

        Two AlphaApollo-specific edits relative to verl's parent _validate():
          (1) rollout seam: multi_turn_loop(val_envs, is_train=False) replaces generate_sequences.
          (2) metric aggregation: tool_calling/traj_uid/success_rate dicts +
              compute_episode_metrics(episode/* keys, absent-safe).

        Scoring: the new trainer has no val_reward_fn; self.reward_fn (EpisodeRewardManager)
        is used for rule-reward val scoring when rm_scores is absent from the output batch.
        When rm_scores is already present in the multi_turn_loop output (the normal path, since
        multi_turn_loop pre-bakes them), reward_fn is not called — same pattern as fit().
        """
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch.get("reward_model", {}).get("style") == "model"
            ):
                return {}

            # prompt uid — GRPO grouping relies on it; mirror fit() (verl 0.9 contract)
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(__import__("uuid").uuid4()) for _ in range(len(test_batch.batch))],
                    dtype=object,
                )

            # Store original inputs for the val-generation table. verl 0.9 moved
            # tokenization into the rollout/AgentLoop, so the dataset yields raw_prompt
            # (non_tensor) instead of input_ids in batch — derive input text from raw_prompt.
            for item in test_batch:
                rp = item.non_tensor_batch.get("raw_prompt", None)
                sample_inputs.append(str(rp) if rp is not None else "")

            # Build test_gen_batch exactly as fit() does (verl 0.9 env-driven contract):
            # _get_env_gen_batch pops the dummy_tensor for sizing + env_kwargs/raw_prompt,
            # so multi_turn_loop's len(gen_batch.batch) works.
            test_gen_batch = self._get_env_gen_batch(test_batch)

            # meta_info for the validate=True sampling branch (+ global_steps)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # ============ AlphaApollo rollout seam ============
            # actor_rollout_wg is optional (default None); is_train=False and
            # envs=self.val_envs are mandatory for the validate=True branch.
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                gen_batch=test_gen_batch,
                envs=self.val_envs,
                is_train=False,
            )
            print("validation generation end")

            del test_batch
            test_batch = test_output_gen_batch

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            # Scoring: reward_fn as val_reward_fn (no val_reward_fn in new trainer).
            # multi_turn_loop pre-bakes rm_scores — use them directly; fall back to reward_fn.
            if "rm_scores" in test_batch.batch:
                reward_tensor = test_batch.batch["rm_scores"]  # pre-baked path (normal)
            else:
                reward_tensor = self.reward_fn(test_batch, return_dict=True)["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )
            # .get fallback: an env that emits no tool calls would otherwise raise
            # KeyError here. Default = zero-count array matching the batch row count
            # so the np.concatenate below keeps shapes aligned (mirrors metrics.py's
            # dual-spelling absent-safe tool-call read).
            _tool_callings_default = np.zeros(reward_tensor.shape[0], dtype=np.int64)
            tool_calling_list.append(
                test_output_gen_batch.non_tensor_batch.get("tool_callings", _tool_callings_default)
            )
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch["traj_uid"])

            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if "success_rate" in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        first_value = test_batch.non_tensor_batch[k][0]
                        current_value = test_batch.non_tensor_batch[k][i]
                        assert first_value == current_value, (
                            f"success rate differs: index 0={first_value}, "
                            f"index {i}={current_value}"
                        )

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)

        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source — unique-traj dedup
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/{data_source}/test_score"] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f"val/{data_source}/tool_call_count/mean"] = np.mean(tool_calls)
            metric_dict[f"val/{data_source}/tool_call_count/max"] = np.max(tool_calls)
            metric_dict[f"val/{data_source}/tool_call_count/min"] = np.min(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f"val/{k}"] = v

        # ============ AlphaApollo metric augmentation (absent-safe) ============
        # Merge episode/* keys from compute_episode_metrics into the returned dict.
        # compute_episode_metrics reads traj_uid/tool_call_count/success from non_tensor_batch;
        # build a synthetic DataProto from the concatenated rollout output to feed it.
        if traj_uid_list:
            import numpy as _np

            _ntb: dict = {"traj_uid": traj_uids}
            if tool_callings is not None:
                _ntb["tool_call_count"] = tool_callings
            # success field: collect per-step success flags if emitted by multi_turn_loop
            # (absent-safe — compute_episode_metrics silently skips missing keys)
            import torch as _torch
            from verl import DataProto as _DP

            _dummy_tensor = _torch.zeros(len(traj_uids), 1)
            from tensordict import TensorDict as _TD

            _ep_batch = _DP(
                batch=_TD({"_dummy": _dummy_tensor}, batch_size=[len(traj_uids)]),
                non_tensor_batch={
                    k: _np.array(v) if not isinstance(v, _np.ndarray) else v
                    for k, v in _ntb.items()
                },
            )
            metric_dict.update(aa_metrics.compute_episode_metrics(_ep_batch))

        return metric_dict


# ======================================================================
# verl-native fit() features intentionally not ported here. Revisit the
# corresponding lines in verl's ray_trainer.py if any of these need to be
# enabled during a future upgrade:
#   - REMAX baseline branch (only used by adv_estimator=REMAX)
#   - rollout correction and bypass mode (off-policy importance sampling)
#   - speculative decoding, GDPO, and variance-proxy metrics
#   - global profiler hooks
#   - ESI checkpoint, dump executor, and multimodal image-sequence edge cases
# ======================================================================
