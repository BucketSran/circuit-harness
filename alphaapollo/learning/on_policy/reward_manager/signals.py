"""Canonical rollout-signal readers with migration fallbacks."""

from __future__ import annotations

from typing import Any

import numpy as np


def episode_reward(item: Any) -> float:
    """Read the canonical episode outcome, falling back to the legacy plural name."""

    return _scalar(item.non_tensor_batch, "episode_reward", "episode_rewards")


def episode_length(item: Any) -> float:
    """Read canonical episode length, falling back to the legacy plural name."""

    return _scalar(item.non_tensor_batch, "episode_length", "episode_lengths")


def action_validity(non_tensor_batch: dict[str, Any]) -> np.ndarray:
    """Read combined validity or derive it from the two canonical checks."""

    if "is_action_valid" in non_tensor_batch:
        return np.asarray(non_tensor_batch["is_action_valid"], dtype=bool)
    detail_keys = ("is_response_format_valid", "is_env_action_valid")
    if all(key in non_tensor_batch for key in detail_keys):
        return np.logical_and(
            np.asarray(non_tensor_batch[detail_keys[0]], dtype=bool),
            np.asarray(non_tensor_batch[detail_keys[1]], dtype=bool),
        )
    raise KeyError("rollout must provide is_action_valid or both split action-validity fields")


def _scalar(values: dict[str, Any], canonical: str, legacy: str) -> float:
    key = canonical if canonical in values else legacy
    return float(np.asarray(values[key]).item())


__all__ = ["action_validity", "episode_length", "episode_reward"]
