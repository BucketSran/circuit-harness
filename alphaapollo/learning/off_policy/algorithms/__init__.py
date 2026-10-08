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

"""Off-policy algorithm plugins — one module per preference-optimization algorithm.

DPO / SimPO / KTO each land here as ``<algo>.py``. Start from ``template.py`` — it
is an annotated, copy-to-create skeleton covering the three verl public seams
(paired dataset via ``data.custom_cls``, a custom ``loss_fn``, and a thin
``SFTTrainer`` subclass that swaps the loss). Their distinguishing feature is the
loss form. SFT is a pure passthrough and has no file (mirrors GRPO being verl-native
on the on-policy side).

Design note: for two-forward losses (DPO/KTO), precompute reference log-probs at
preprocessing time and read them as parquet columns, rather than threading a second
frozen model through verl's single-model engine. (SimPO needs no reference model.)
"""
