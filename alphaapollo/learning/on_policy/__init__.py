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

"""On-policy training: sample, evaluate, and update through verl.

The public re-exports are resolved lazily so importing ``alphaapollo.learning``
does not require the optional torch/verl runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AlphaApolloRayPPOTrainer",
    "apply_invalid_action_penalty",
    "compute_episode_metrics",
    "metrics",
    "penalty",
]


def __getattr__(name: str) -> Any:
    if name == "metrics":
        return import_module("alphaapollo.learning.on_policy.metrics")
    if name == "penalty":
        return import_module("alphaapollo.learning.on_policy.reward_manager.penalty")
    if name == "compute_episode_metrics":
        module = import_module("alphaapollo.learning.on_policy.metrics")
        return module.compute_episode_metrics
    if name == "apply_invalid_action_penalty":
        module = import_module("alphaapollo.learning.on_policy.reward_manager.penalty")
        return module.apply_invalid_action_penalty
    if name == "AlphaApolloRayPPOTrainer":
        module = import_module("alphaapollo.learning.on_policy.trainer")
        return module.AlphaApolloRayPPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
