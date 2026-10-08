# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from alphaapollo.common.environment import EnvironmentInitResult, EnvironmentTransition
from alphaapollo.common.environment.provider import EnvironmentProvider
from alphaapollo.common.generation import (
    BackendCapabilities,
    GenerationBackend,
    GenerationResponse,
    Provenance,
    ToolCall,
)
from alphaapollo.learning.on_policy.rollout.reasoning_consumer import (
    ReasoningRolloutConsumer,
)
from alphaapollo.reasoning.runtime import AgentResult


class _Backend(GenerationBackend):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.seeds: list[list[int | None]] = []

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            token_native=True,
            logprobs=True,
            tool_calls=True,
            policy_source="remote_engine",
        )

    def _generate_batch(self, requests):
        self.batch_sizes.append(len(requests))
        self.seeds.append([request.provider_options.get("seed") for request in requests])
        return [
            GenerationResponse(
                request_id=request.request_id,
                group_id=request.group_id,
                sample_id=request.sample_id,
                content=f"action-{request.request_id}",
                reasoning_content="private reasoning trace",
                finish_reason="stop",
                tool_calls=(ToolCall(id="call-1", name="search", arguments='{"q":"x"}'),),
                usage={"prompt_tokens": 2, "completion_tokens": 2},
                backend_metadata={"engine": "fake"},
                prompt_token_ids=(10, 11),
                response_token_ids=(20, 21),
                response_logprobs=(-0.1, -0.2),
                provenance=Provenance(
                    policy_model="model",
                    tokenizer_id="tokenizer",
                    weights_version="step-3",
                    tokenizer_fingerprint="tok-fp",
                    chat_template_fingerprint="chat-fp",
                    actor="policy",
                ),
            )
            for request in requests
        ]


class _Pool:
    batch_size = 2

    def close(self) -> None:
        pass


class _Environment:
    def __init__(self, slot: int, payload) -> None:
        self.slot = slot
        self.payload = payload
        self.steps = 0
        self.raw_observation = f"raw-task-{slot}"

    def init(self, context):
        assert self.payload == {"case": self.slot + 1}
        return EnvironmentInitResult(
            f"task {self.slot}",
            {"source": "test"},
            raw_observation=self.raw_observation,
        )

    def project_response(self, response):
        return response.content

    def continuation_messages(self, response, transition):
        return (
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": transition.observation},
        )

    def step(self, action):
        assert action
        previous = self.raw_observation
        self.steps += 1
        done = self.steps >= self.slot + 1
        self.raw_observation = f"raw-obs-{self.slot}-{self.steps}"
        return EnvironmentTransition(
            observation=f"next-prompt-{self.slot}-{self.steps}",
            reward=float(done),
            done=done,
            success=done if done else None,
            response_format_valid=True,
            env_action_valid=self.slot == 0,
            termination_reason="success" if done else None,
            metadata={"won": done},
            previous_observation=previous,
            raw_observation=self.raw_observation,
            executed_action=f"executed:{action}",
        )

    def terminate(self, reason):
        pass

    def close(self):
        pass


def _config():
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            model=SimpleNamespace(path="model"),
            rollout=SimpleNamespace(
                prompt_length=4,
                response_length=3,
                temperature=0.7,
                top_p=0.9,
                top_k=-1,
                repetition_penalty=1.0,
                seed=7,
                val_kwargs=SimpleNamespace(
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=-1,
                ),
            ),
        ),
        env=SimpleNamespace(max_steps=3, seed=5),
    )


def test_reasoning_consumer_batches_active_trajectories_and_preserves_signals():
    backend = _Backend()
    consumer = ReasoningRolloutConsumer(
        backend=backend,
        tokenizer=SimpleNamespace(pad_token_id=0),
        config=_config(),
    )
    env_kwargs = np.empty(2, dtype=object)
    env_kwargs[:] = [{"case": 1}, {"case": 2}]
    gen_batch = SimpleNamespace(
        batch=np.zeros((2, 1)),
        non_tensor_batch={
            "data_source": np.asarray(["one", "two"], dtype=object),
            "env_kwargs": env_kwargs,
        },
    )

    trajectories = consumer.collect(
        gen_batch,
        EnvironmentProvider(_Pool(), lambda slot, payload: _Environment(slot, payload)),
        group_size=2,
        validate=False,
    )
    rows = consumer.to_step_rows(trajectories)

    assert backend.batch_sizes == [2, 1]
    assert [len(item) for item in rows] == [1, 2]
    assert trajectories.trajectories[0].group_id == trajectories.trajectories[1].group_id
    assert rows[0][0]["raw_env_reward"] == 1.0
    assert rows[1][0]["is_env_action_valid"] is False
    assert rows[1][1]["done"] is True
    assert rows[0][0]["provenance"].weights_version == "step-3"
    assert rows[0][0]["response_mask"].tolist() == [1, 1, 0]
    assert rows[1][1]["original_step_index"] == 1
    assert trajectories.trajectories[0].initial_observation == "raw-task-0"
    assert trajectories.trajectories[0].initial_environment_metadata == {"source": "test"}
    assert rows[0][0]["initial_environment_metadata"] == {"source": "test"}
    assert rows[0][0]["generated_action"].startswith("action-")
    assert rows[0][0]["reasoning_content"] == "private reasoning trace"
    assert rows[0][0]["generation_finish_reason"] == "stop"
    assert rows[0][0]["generation_tool_calls"][0].name == "search"
    assert rows[0][0]["generation_usage"] == {"prompt_tokens": 2, "completion_tokens": 2}
    assert rows[0][0]["generation_metadata"] == {"engine": "fake"}
    assert rows[0][0]["executed_action"].startswith("executed:action-")
    assert rows[0][0]["observation"] == "raw-task-0"
    assert rows[0][0]["anchor_obs"] == "raw-task-0"
    assert rows[0][0]["raw_next_observation"] == "raw-obs-0-1"
    assert rows[0][0]["next_prompt"] == "next-prompt-0-1"
    assert "rm_scores" not in rows[0][0]


def test_model_and_environment_seed_streams_are_independent() -> None:
    backend = _Backend()
    consumer = ReasoningRolloutConsumer(
        backend=backend,
        tokenizer=SimpleNamespace(pad_token_id=0),
        config=_config(),
    )
    environment_seeds: list[int | None] = []
    next_environment_seed = 40

    class Environment(_Environment):
        def init(self, context):
            environment_seeds.append(context.seed)
            return super().init(context)

    def prepare(active_size: int):
        nonlocal next_environment_seed
        next_environment_seed += 1
        return [next_environment_seed] * active_size

    provider = EnvironmentProvider(
        _Pool(),
        lambda slot, payload: Environment(slot, payload),
        prepare_batch=prepare,
    )
    env_kwargs = np.asarray([{"case": 1}, {"case": 2}], dtype=object)
    batch = SimpleNamespace(
        batch=np.zeros((2, 1)),
        non_tensor_batch={"env_kwargs": env_kwargs},
    )

    consumer.collect(batch, provider, group_size=2, validate=False)
    consumer.collect(batch, provider, group_size=2, validate=False)

    assert environment_seeds == [41, 41, 42, 42]
    assert backend.seeds == [[7, 7], [7], [7, 7], [7]]


def test_runtime_termination_is_projected_onto_the_last_learning_turn() -> None:
    config = _config()
    config.env.max_steps = 1

    class NeverDone(_Environment):
        def step(self, action):
            transition = super().step(action)
            return EnvironmentTransition(
                observation=transition.observation,
                reward=transition.reward,
                done=False,
                success=None,
                metadata=transition.metadata,
                previous_observation=transition.previous_observation,
                raw_observation=transition.raw_observation,
                executed_action=transition.executed_action,
            )

    consumer = ReasoningRolloutConsumer(
        backend=_Backend(),
        tokenizer=SimpleNamespace(pad_token_id=0),
        config=config,
    )
    batch = SimpleNamespace(
        batch=np.zeros((1, 1)),
        non_tensor_batch={"env_kwargs": np.asarray([{"case": 1}], dtype=object)},
    )

    trajectories = consumer.collect(
        batch,
        EnvironmentProvider(_Pool(), lambda slot, payload: NeverDone(slot, payload)),
        group_size=1,
        validate=False,
    )
    row = consumer.to_step_rows(trajectories)[0][-1]

    assert row["environment_done"] is False
    assert row["done"] is True
    assert row["termination_reason"] == "max_turns"
    assert row["is_success_terminal"] is False


def test_consumer_rejects_incomplete_provenance() -> None:
    class IncompleteBackend(_Backend):
        def _generate_batch(self, requests):
            responses = super()._generate_batch(requests)
            return [
                GenerationResponse(
                    request_id=response.request_id,
                    group_id=response.group_id,
                    sample_id=response.sample_id,
                    content=response.content,
                    prompt_token_ids=response.prompt_token_ids,
                    response_token_ids=response.response_token_ids,
                    response_logprobs=response.response_logprobs,
                    provenance=Provenance(policy_model="model", tokenizer_id="tokenizer"),
                )
                for response in responses
            ]

    consumer = ReasoningRolloutConsumer(
        backend=IncompleteBackend(),
        tokenizer=SimpleNamespace(pad_token_id=0),
        config=_config(),
    )
    batch = SimpleNamespace(
        batch=np.zeros((1, 1)),
        non_tensor_batch={"env_kwargs": np.asarray([{"case": 1}], dtype=object)},
    )

    with pytest.raises(ValueError, match="weights_version"):
        consumer.collect(
            batch,
            EnvironmentProvider(_Pool(), lambda slot, payload: _Environment(slot, payload)),
            group_size=1,
            validate=False,
        )


def test_consumer_rejects_mismatched_reasoning_result_identity() -> None:
    consumer = ReasoningRolloutConsumer(
        backend=_Backend(),
        tokenizer=SimpleNamespace(pad_token_id=0),
        config=_config(),
    )

    with pytest.raises(ValueError, match="does not match trajectory identity"):
        consumer.consume_results(
            [AgentResult(task_id="wrong", final_text="", termination_reason="final")],
            group_ids=["group"],
            trajectory_ids=["expected"],
            data_sources=["test"],
        )
