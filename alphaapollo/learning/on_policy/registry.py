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

"""Trainer factory for algorithms that need fit-loop hooks.

The base ``AlphaApolloRayPPOTrainer`` runs verl-native GRPO (and any estimator that
needs no fit()-loop hooks). Algorithms that need loop hooks ship a trainer subclass in
``algorithms/<algo>/`` and register here. Adding a new algorithm = new subclass + one
line below; the fit() loop and task_runner never change.

Algorithm PRs add their own lazy route here. The shared fallback remains verl-native.
"""

from __future__ import annotations

from typing import Any


def trainer_class_for(adv_estimator) -> type[Any]:
    """Return the trainer class for the given advantage estimator."""
    del adv_estimator
    from alphaapollo.learning.on_policy.trainer import AlphaApolloRayPPOTrainer

    return AlphaApolloRayPPOTrainer
