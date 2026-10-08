"""CPU-only checks for Learning package and ownership boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from alphaapollo.learning import on_policy
from alphaapollo.learning.on_policy import metrics


def test_on_policy_package_is_importable_without_optional_runtime() -> None:
    assert "AlphaApolloRayPPOTrainer" in on_policy.__all__
    assert on_policy.metrics is metrics


def test_episode_metrics_are_duck_typed_and_absent_safe() -> None:
    batch = SimpleNamespace(
        non_tensor_batch={
            "traj_uid": np.array(["a", "a", "b"], dtype=object),
            "tool_callings": np.array([0, 2, 1]),
            "success_rate": np.array([1.0, 1.0, 0.0]),
        }
    )

    assert metrics.compute_episode_metrics(batch) == {
        "episode/num_trajectories": 2,
        "episode/tool_call_count": 1.0,
        "episode/success_rate": 2.0 / 3.0,
    }
    assert metrics.compute_episode_metrics(SimpleNamespace(non_tensor_batch={})) == {}


def test_learning_runtime_has_no_legacy_core_imports() -> None:
    package_root = Path(__file__).parents[3] / "alphaapollo" / "learning"
    stale = []
    for path in package_root.rglob("*.py"):
        if "alphaapollo.core" in path.read_text(encoding="utf-8"):
            stale.append(path.relative_to(package_root).as_posix())
    assert stale == []
