"""Canonical trajectory batch consumed by Learning.

This module intentionally has no torch or verl dependency. Reasoning may produce
these fields using any runtime; the Learning adapter owns conversion to DataProto.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

ROLLOUT_TENSOR_FIELDS = frozenset(
    {
        "prompts",
        "responses",
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
        "rollout_log_probs",
        "rm_scores",
        "step_rewards",
        "token_level_scores",
    }
)

REQUIRED_ROLLOUT_TENSOR_FIELDS = frozenset(
    {
        "prompts",
        "responses",
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
        "rollout_log_probs",
    }
)


class RolloutContractError(ValueError):
    """Raised when a producer returns a malformed Learning rollout batch."""


@dataclass(frozen=True, slots=True)
class LearningTurn:
    """One framework-neutral step exposed to Learning algorithms."""

    original_step_index: int
    generated_text: str
    reasoning_text: str | None
    finish_reason: str | None
    tool_calls: tuple[Any, ...]
    generation_usage: Mapping[str, Any]
    generation_metadata: Mapping[str, Any]
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    rollout_logprobs: tuple[float, ...]
    executed_action: Any
    pre_action_observation: Any
    raw_observation: Any
    next_prompt: Any
    raw_env_reward: float | None
    environment_done: bool
    done: bool
    terminal_success: bool | None
    termination_reason: str | None
    response_format_valid: bool
    env_action_valid: bool
    environment_metadata: Mapping[str, Any]
    provenance: Any


@dataclass(frozen=True, slots=True)
class LearningTrajectory:
    """One complete trajectory independent of Reasoning implementation types."""

    task_id: str
    trajectory_id: str
    group_id: str
    sample_id: int
    data_source: str
    initial_observation: Any
    turns: tuple[LearningTurn, ...]
    termination_reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    initial_environment_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LearningTrajectoryBatch:
    """Stable algorithm-facing batch before tensor/DataProto conversion."""

    trajectories: tuple[LearningTrajectory, ...]

    def __post_init__(self) -> None:
        identities = [item.trajectory_id for item in self.trajectories]
        if len(identities) != len(set(identities)):
            raise ValueError("Learning trajectory identities must be unique")


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """Framework-neutral batch of trajectory steps.

    Every tensor and non-tensor field is row-aligned. ``meta_info`` contains
    batch-level information, including reproducibility metadata supplied by the
    rollout producer.
    """

    tensors: Mapping[str, Any]
    non_tensors: Mapping[str, np.ndarray]
    meta_info: Mapping[str, Any] = field(default_factory=dict)
    batch_size: int = field(init=False)

    def __post_init__(self) -> None:
        tensors = dict(self.tensors)
        non_tensors = {key: _as_non_tensor_array(value) for key, value in self.non_tensors.items()}
        meta_info = dict(self.meta_info)

        missing = sorted(REQUIRED_ROLLOUT_TENSOR_FIELDS.difference(tensors))
        if missing:
            raise RolloutContractError("missing rollout tensor fields: " + ", ".join(missing))

        observed: dict[str, int] = {}
        for key, value in tensors.items():
            observed[f"tensor.{key}"] = _leading_dimension(value, key)
        for key, value in non_tensors.items():
            observed[f"non_tensor.{key}"] = _leading_dimension(value, key)

        sizes = set(observed.values())
        if len(sizes) != 1:
            detail = ", ".join(f"{key}={size}" for key, size in sorted(observed.items()))
            raise RolloutContractError("rollout fields are not row-aligned: " + detail)

        object.__setattr__(self, "tensors", MappingProxyType(tensors))
        object.__setattr__(self, "non_tensors", MappingProxyType(non_tensors))
        object.__setattr__(self, "meta_info", MappingProxyType(meta_info))
        object.__setattr__(self, "batch_size", sizes.pop())

    @classmethod
    def from_collated(
        cls,
        fields: Mapping[str, Any],
        *,
        meta_info: Mapping[str, Any] | None = None,
        tensor_fields: frozenset[str] = ROLLOUT_TENSOR_FIELDS,
    ) -> RolloutBatch:
        """Split a collated legacy mapping without importing torch or verl."""

        tensors = {key: value for key, value in fields.items() if key in tensor_fields}
        non_tensors = {key: value for key, value in fields.items() if key not in tensor_fields}
        return cls(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info or {})


def _as_non_tensor_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _leading_dimension(value: Any, key: str) -> int:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        raise RolloutContractError(f"rollout field {key!r} must have a batch dimension")
    return int(shape[0])


__all__ = [
    "LearningTrajectory",
    "LearningTrajectoryBatch",
    "LearningTurn",
    "REQUIRED_ROLLOUT_TENSOR_FIELDS",
    "ROLLOUT_TENSOR_FIELDS",
    "RolloutBatch",
    "RolloutContractError",
]
