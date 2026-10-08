# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Learning-owned bridge from Reasoning trajectories to rollout rows.

Reasoning owns decoding and Environment interaction.  This module is the only
on-policy Learning boundary that knows about ``AgentResult``: it injects a
Common vector Environment batch into Reasoning's Runtime and projects returned
trajectories into Learning's every-step-one-row representation.  Algorithm
implementations never depend on Reasoning classes directly.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from alphaapollo.common.environment.provider import EnvironmentProvider
from alphaapollo.learning.adapters.rollout_tensors import build_rollout_row
from alphaapollo.learning.rollout_contract import (
    LearningTrajectory,
    LearningTrajectoryBatch,
    LearningTurn,
)
from alphaapollo.reasoning.runtime import (
    AgentResult,
    AgentTask,
    AlphaApolloAgentRuntime,
)


class ReasoningRolloutConsumer:
    """Run an injected Generation backend through Reasoning for Learning."""

    def __init__(self, *, backend: Any, tokenizer: Any, config: Any) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.config = config

    def collect(
        self,
        gen_batch: Any,
        envs: EnvironmentProvider,
        *,
        group_size: int,
        validate: bool,
    ) -> LearningTrajectoryBatch:
        batch_size = len(gen_batch.batch)
        trajectory_ids = tuple(uuid.uuid4().hex for _ in range(batch_size))
        group_ids = _group_ids(batch_size, group_size)
        data_sources = tuple(
            str(value)
            for value in gen_batch.non_tensor_batch.get(
                "data_source", np.asarray(["unknown"] * batch_size, dtype=object)
            )
        )
        reset_kwargs = gen_batch.non_tensor_batch.get("env_kwargs")
        payloads = (
            [dict(value) for value in reset_kwargs]
            if reset_kwargs is not None
            else [{} for _ in range(batch_size)]
        )
        environment_seeds = envs.episode_seeds(payloads)
        tasks = [
            AgentTask(
                task_id=trajectory_ids[index],
                system="",
                prompt="",
                model=str(self.config.actor_rollout_ref.model.path),
                routing_key=trajectory_ids[index],
                branch_id="learning",
                sample_id=index % max(group_size, 1),
                environment_slot=index,
                environment_seed=environment_seeds[index],
                task_payload=payloads[index],
                metadata={"data_source": data_sources[index]},
            )
            for index in range(batch_size)
        ]
        sampling = _runtime_sampling(self.config, validate=validate)
        runtime = AlphaApolloAgentRuntime(
            self.backend,
            model=str(self.config.actor_rollout_ref.model.path),
            environment_factory=envs.environment_factory,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            max_tokens=int(self.config.actor_rollout_ref.rollout.response_length),
            max_turns=int(self.config.env.max_steps),
            seed=None,
            provider_options=sampling["provider_options"],
        )
        expected_provenance = getattr(self.backend, "current_provenance", None)
        results = tuple(runtime.run_batch(tasks))
        trajectories = self.consume_results(
            results,
            group_ids=group_ids,
            trajectory_ids=trajectory_ids,
            data_sources=data_sources,
            expected_provenance=expected_provenance,
        )
        return trajectories

    def consume_results(
        self,
        results: Sequence[AgentResult],
        *,
        group_ids: Sequence[str],
        trajectory_ids: Sequence[str],
        data_sources: Sequence[str],
        expected_provenance: Any | None = None,
    ) -> LearningTrajectoryBatch:
        """Convert public Reasoning records into the stable Learning contract."""

        trajectories: list[LearningTrajectory] = []
        provenance_identity: tuple[Any, ...] | None = None
        for sample_id, (result, group_id, trajectory_id, data_source) in enumerate(
            zip(results, group_ids, trajectory_ids, data_sources, strict=True)
        ):
            if result.task_id != trajectory_id:
                raise ValueError(
                    f"Reasoning result {result.task_id!r} does not match trajectory "
                    f"identity {trajectory_id!r}"
                )
            result_metadata = dict(result.metadata)
            initial_environment_metadata = result_metadata.get("environment_init", {})
            if not isinstance(initial_environment_metadata, Mapping):
                raise ValueError("AgentResult environment_init metadata must be a mapping")
            initial_environment_metadata = dict(initial_environment_metadata)
            learning_turns: list[LearningTurn] = []
            for turn_position, turn in enumerate(result.turns):
                response = turn.generation_response
                transition = turn.environment_transition
                is_last_turn = turn_position == len(result.turns) - 1
                runtime_terminated = is_last_turn and not transition.done
                episode_done = transition.done or runtime_terminated
                termination_reason = (
                    transition.termination_reason
                    if transition.done
                    else result.termination_reason
                    if runtime_terminated
                    else None
                )
                terminal_success = (
                    transition.success if transition.done else False if runtime_terminated else None
                )
                if (
                    response.prompt_token_ids is None
                    or response.response_token_ids is None
                    or response.response_logprobs is None
                    or response.provenance is None
                ):
                    raise ValueError(
                        f"trajectory {trajectory_id!r} turn {turn.index} lacks token-native facts"
                    )
                _validate_provenance(
                    response.provenance,
                    expected_model=str(self.config.actor_rollout_ref.model.path),
                    expected=expected_provenance,
                )
                current_identity = tuple(
                    getattr(response.provenance, field)
                    for field in (
                        "policy_model",
                        "tokenizer_id",
                        "weights_version",
                        "tokenizer_fingerprint",
                        "chat_template_fingerprint",
                        "actor",
                    )
                )
                if provenance_identity is None:
                    provenance_identity = current_identity
                elif current_identity != provenance_identity:
                    raise ValueError("rollout batch mixes incompatible Generation provenance")
                learning_turns.append(
                    LearningTurn(
                        original_step_index=turn.index,
                        generated_text=response.content,
                        reasoning_text=response.reasoning_content,
                        finish_reason=response.finish_reason,
                        tool_calls=tuple(response.tool_calls),
                        generation_usage=dict(response.usage),
                        generation_metadata=dict(response.backend_metadata),
                        prompt_token_ids=tuple(response.prompt_token_ids),
                        response_token_ids=tuple(response.response_token_ids),
                        rollout_logprobs=tuple(response.response_logprobs),
                        executed_action=transition.executed_action,
                        pre_action_observation=transition.previous_observation,
                        raw_observation=transition.raw_observation,
                        next_prompt=transition.observation,
                        raw_env_reward=float(transition.reward),
                        environment_done=transition.done,
                        done=episode_done,
                        terminal_success=terminal_success,
                        termination_reason=termination_reason,
                        response_format_valid=transition.response_format_valid,
                        env_action_valid=transition.env_action_valid,
                        environment_metadata=dict(transition.metadata),
                        provenance=response.provenance,
                    )
                )
            trajectories.append(
                LearningTrajectory(
                    task_id=result.task_id,
                    trajectory_id=trajectory_id,
                    group_id=group_id,
                    sample_id=(
                        int(result.turns[0].generation_request.sample_id)
                        if result.turns
                        else sample_id
                    ),
                    data_source=data_source,
                    initial_observation=result.initial_observation,
                    turns=tuple(learning_turns),
                    termination_reason=result.termination_reason,
                    metadata=result_metadata,
                    initial_environment_metadata=initial_environment_metadata,
                )
            )
        return LearningTrajectoryBatch(tuple(trajectories))

    def to_step_rows(self, trajectories: LearningTrajectoryBatch) -> list[list[dict[str, Any]]]:
        rc = self.config.actor_rollout_ref.rollout
        all_rows: list[list[dict[str, Any]]] = []
        for trajectory in trajectories.trajectories:
            rows: list[dict[str, Any]] = []
            for turn in trajectory.turns:
                tensor_row = build_rollout_row(
                    prompt_ids=turn.prompt_token_ids,
                    response_ids=turn.response_token_ids,
                    response_mask=[1] * len(turn.response_token_ids),
                    prompt_length=int(rc.prompt_length),
                    response_length=int(rc.response_length),
                    pad_token_id=int(self.tokenizer.pad_token_id),
                    response_logprobs=turn.rollout_logprobs,
                )
                row = {
                    key: torch.tensor(
                        value,
                        dtype=torch.long
                        if key
                        in {
                            "prompts",
                            "responses",
                            "response_mask",
                            "input_ids",
                            "attention_mask",
                            "position_ids",
                        }
                        else torch.float,
                    )
                    for key, value in tensor_row.items()
                }
                metadata = dict(turn.environment_metadata)
                row.update(
                    uid=trajectory.group_id,
                    traj_uid=trajectory.trajectory_id,
                    data_source=trajectory.data_source,
                    sample_id=trajectory.sample_id,
                    original_step_index=turn.original_step_index,
                    generated_action=turn.generated_text,
                    reasoning_content=turn.reasoning_text,
                    generation_finish_reason=turn.finish_reason,
                    generation_tool_calls=turn.tool_calls,
                    generation_usage=dict(turn.generation_usage),
                    generation_metadata=dict(turn.generation_metadata),
                    executed_action=turn.executed_action,
                    observation=turn.pre_action_observation,
                    raw_next_observation=turn.raw_observation,
                    next_prompt=turn.next_prompt,
                    rewards=turn.raw_env_reward,
                    raw_env_reward=turn.raw_env_reward,
                    environment_done=turn.environment_done,
                    done=turn.done,
                    active_masks=True,
                    is_success_terminal=turn.terminal_success,
                    termination_reason=turn.termination_reason,
                    is_response_format_valid=turn.response_format_valid,
                    is_env_action_valid=turn.env_action_valid,
                    is_action_valid=turn.response_format_valid and turn.env_action_valid,
                    anchor_obs=turn.pre_action_observation,
                    tool_callings=float(metadata.get("tool_calling", False)),
                    environment_metadata=metadata,
                    provenance=turn.provenance,
                    trajectory_metadata=dict(trajectory.metadata),
                    initial_environment_metadata=dict(trajectory.initial_environment_metadata),
                )
                rows.append(row)
            all_rows.append(rows)
        return all_rows


def _group_ids(batch_size: int, group_size: int) -> tuple[str, ...]:
    ids: list[str] = []
    current = ""
    for index in range(batch_size):
        if index % max(group_size, 1) == 0:
            current = uuid.uuid4().hex
        ids.append(current)
    return tuple(ids)


def _runtime_sampling(config: Any, *, validate: bool) -> dict[str, Any]:
    rollout = config.actor_rollout_ref.rollout
    if validate:
        source = rollout.val_kwargs
    else:
        source = rollout
    do_sample = bool(getattr(source, "do_sample", True))
    provider_options: dict[str, Any] = {
        "top_k": int(getattr(source, "top_k", -1)),
        "repetition_penalty": float(getattr(source, "repetition_penalty", 1.0)),
    }
    seed = getattr(source, "seed", None)
    if seed is not None:
        provider_options["seed"] = int(seed)
    return {
        "temperature": float(getattr(source, "temperature", 1.0)) if do_sample else 0.0,
        "top_p": float(getattr(source, "top_p", 1.0)) if do_sample else 1.0,
        "provider_options": provider_options,
    }


def _validate_provenance(
    value: Any,
    *,
    expected_model: str,
    expected: Any | None = None,
) -> None:
    required = (
        "policy_model",
        "tokenizer_id",
        "weights_version",
        "tokenizer_fingerprint",
        "chat_template_fingerprint",
        "actor",
    )
    if value is None or any(not hasattr(value, field) for field in required):
        raise ValueError("trainable trajectory requires complete Generation provenance fields")
    if value.policy_model != expected_model:
        raise ValueError(
            f"rollout policy model {value.policy_model!r} does not match {expected_model!r}"
        )
    for field in (
        "tokenizer_id",
        "weights_version",
        "tokenizer_fingerprint",
        "chat_template_fingerprint",
    ):
        field_value = getattr(value, field)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"rollout provenance requires a non-empty {field}")
    if expected is not None and value != expected:
        raise ValueError("rollout provenance does not match the synchronized policy identity")


__all__ = [
    "ReasoningRolloutConsumer",
]
