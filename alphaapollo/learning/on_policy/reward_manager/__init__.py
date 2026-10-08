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

"""Reward-signal shaping applied inside the on-policy update.

These are NOT training algorithms (no advantage estimator / policy loss, no verl
registry seam) — they are adjustments to the reward/score tensors that the fit loop
applies before advantage computation:

- ``episode`` — maps an episode outcome to the final valid response token
- ``penalty`` — invalid-action penalty on token-level scores

Future shaping ops (e.g. length normalization for the ``normalize_by_length`` flag)
land here too. Algorithms live in ``../algorithms``.

Both raw episode reward attribution and on-policy shaping are Learning-owned. Public
symbols are resolved lazily so either component can be tested in isolation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["EpisodeRewardManager", "apply_invalid_action_penalty"]


def __getattr__(name: str) -> Any:
    if name == "EpisodeRewardManager":
        module = import_module("alphaapollo.learning.on_policy.reward_manager.episode")
        return module.EpisodeRewardManager
    if name == "apply_invalid_action_penalty":
        module = import_module("alphaapollo.learning.on_policy.reward_manager.penalty")
        return module.apply_invalid_action_penalty
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
