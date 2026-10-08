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

"""Off-policy training: static data -> forward -> loss -> update.

No sampling loop. Two off-policy *engines* live here:

- **verl engine** — entry ``main_off_policy`` (hydra) dispatches the verl-native SFT
  passthrough or an AlphaApollo preference trainer selected by
  ``alphaapollo.off_policy_algo``. Preference-optimization algorithms (DPO / SimPO /
  KTO) live one-file-each under ``algorithms`` and attach their loss through a thin
  ``SFTTrainer`` subclass. The verl-native baseline (SFT) has no algorithm file,
  mirroring GRPO on the on-policy side.
- **Megatron engine** — the ``pretrain`` subpackage: from-scratch pretraining on
  standalone Megatron-LM (megatron-core 0.18 + Transformer Engine), a **verl-free**
  leaf with its own entry ``alphaapollo.learning.off_policy.pretrain.main_pretrain``
  and its own conda env (``alphaapollo-pretrain``). Its only crossing to the RL/SFT
  side is a HuggingFace checkpoint (``pretrain.checkpoint_bridge``). It does NOT plug
  into ``main_off_policy`` / the verl SFT engine.
"""
