# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared single-task Environment initialization for Reasoning runtimes."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from typing import Any

from alphaapollo.common.environment.base import EnvironmentContext
from alphaapollo.reasoning._immutable import _freeze_json_mapping, _thaw_json
from alphaapollo.reasoning.runtime.agent_runtime import AgentTask

__all__ = [
    "EnvironmentInitialization",
    "effective_task_seed",
    "initialize_environment",
    "invoke_environment_init",
    "normalize_environment_init",
]


@dataclass(frozen=True, slots=True)
class EnvironmentInitialization:
    """Validated initial observation and immutable public metadata."""

    observation: Any
    metadata: Mapping[str, Any]
    raw_observation: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_mapping(self.metadata, where="Environment init metadata"),
        )


def effective_task_seed(task: AgentTask, runtime_seed: int | None) -> int | None:
    """Resolve the task override before a Runtime-level deterministic seed."""

    if task.sampling_seed is not None:
        return task.sampling_seed
    if runtime_seed is None:
        return None
    return runtime_seed + task.sample_id


def initialize_environment(
    environment: Any,
    task: AgentTask,
    *,
    seed: int | None,
) -> EnvironmentInitialization:
    """Initialize one Environment with the same context under every Runtime."""

    return normalize_environment_init(invoke_environment_init(environment, task, seed=seed))


def invoke_environment_init(
    environment: Any,
    task: AgentTask,
    *,
    seed: int | None,
) -> Any:
    """Invoke ``Environment.init`` without conflating startup with validation."""

    init = getattr(environment, "init", None)
    if not callable(init):
        raise TypeError("Environment must expose init()")
    try:
        signature = inspect.signature(init)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is inspect.Parameter.empty
        ]
    except (TypeError, ValueError):
        required = []
    if len(required) >= 2:
        if task.task_payload:
            raise TypeError(
                "task_payload requires Environment.init(EnvironmentContext); "
                "the legacy init(system, prompt) contract cannot receive it"
            )
        return normalize_environment_init(init(task.system, task.prompt))

    metadata = _thaw_json(task.metadata)
    task_payload = _thaw_json(task.task_payload)
    environment_config = metadata.pop("environment_config", {})
    if not isinstance(environment_config, Mapping):
        raise TypeError("environment_config metadata must be a mapping")
    environment_user_prompt = metadata.pop("environment_user_prompt", task.prompt)
    if not isinstance(environment_user_prompt, str) or not environment_user_prompt.strip():
        raise TypeError("environment_user_prompt metadata must be a non-empty string")
    # Agent-facing memory context: consumed by the trajectory slot when it
    # builds the first model message, never part of the environment's view.
    metadata.pop("agent_memory_context", None)
    result = init(
        EnvironmentContext(
            session_id=str(metadata.pop("session_id", task.task_id)),
            actor=str(metadata.pop("actor", metadata.pop("role", "agent"))),
            task_id=task.task_id,
            branch_id=task.branch_id,
            round_index=task.round_index,
            sample_id=task.sample_id,
            seed=(
                task.environment_seed
                if task.environment_seed is not None
                else effective_task_seed(task, seed)
            ),
            routing_key=task.routing_key,
            system_prompt=task.system,
            user_prompt=environment_user_prompt,
            task_payload=copy.deepcopy(dict(task_payload)),
            environment_config=copy.deepcopy(dict(environment_config)),
            metadata=metadata,
        )
    )
    return result


def normalize_environment_init(result: Any) -> EnvironmentInitialization:
    """Validate every supported init shape into one Runtime-owned record."""

    if isinstance(result, tuple) and len(result) == 2:
        observation, metadata = result
        raw_observation = None
    else:
        if isinstance(result, type) or not is_dataclass(result):
            raise TypeError(
                "Environment init must return a typed init record or (observation, metadata) tuple"
            )
        missing = [name for name in ("observation", "metadata") if not hasattr(result, name)]
        if missing:
            raise TypeError("Environment init record is missing field(s): " + ", ".join(missing))
        observation = result.observation
        metadata = result.metadata
        raw_observation = getattr(result, "raw_observation", None)
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, Mapping):
        raise TypeError("Environment init metadata must be a mapping")
    return EnvironmentInitialization(
        observation,
        copy.deepcopy(dict(metadata)),
        copy.deepcopy(raw_observation),
    )
