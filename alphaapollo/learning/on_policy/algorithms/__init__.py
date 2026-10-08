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

"""On-policy algorithm plugins: one module or package per algorithm.

Each advantage-estimator / policy-loss variant is self-contained, registered through
verl's public seams (``@register_adv_est`` / ``register_policy_loss``) and selected by
config (``algorithm.adv_estimator`` / ``...policy_loss.loss_mode``).

An algorithm that needs no fit()-loop hooks can be a single ``<algo>.py`` module
(registration only). An algorithm that needs loop hooks ships a ``<algo>/`` package with
a trainer subclass and is wired into ``../registry.py`` (``trainer_class_for``). The
verl-native baseline (GRPO) has neither — it is the base trainer's default path.
Registration is a pure import side-effect. (Reward-signal shaping such as the
invalid-action penalty is NOT an algorithm — it lives in ``../reward_manager``.)
"""
