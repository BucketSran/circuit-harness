# Copyright 2026 TMLR Group
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

"""AlphaApollo off-policy training entry point (SFT and offline DPO).

Thin hydra @main entry that dispatches to verl-native SFT or AlphaApollo offline DPO.
All heavy imports live inside ``main_entry()`` to keep the module import-safe for
``--cfg job`` introspection and unit tests.

The configs are loaded from the Learning-owned off-policy config package and inherit
verl's SFT trainer engine through a Hydra search-path overlay. DPO selects its paired
dataset and trainer with ``alphaapollo.off_policy_algo=dpo``.

Note: run_sft calls initialize_global_process_group() so real training must be launched
under torchrun (the SFT launcher YAMLs already set launcher: torchrun).

Usage:
    # GPU-free config resolve gate:
    python -m alphaapollo.learning.off_policy.main_off_policy --cfg job

    python -m alphaapollo.learning.off_policy.main_off_policy \
        --config-name dpo_trainer --cfg job

    # Real SFT training launch (GPU required — via alphaapollo.workflows.sft):
    torchrun --standalone --nnodes=1 --nproc-per-node=2 \\
        -m alphaapollo.learning.off_policy.main_off_policy \\
        data.train_files=/path/to/train.parquet \\
        data.val_files=/path/to/val.parquet \\
        model.path=Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import hydra


def main_entry(config):
    """Hydra entry point for AlphaApollo off-policy training.

    SFT remains a pure verl passthrough; DPO selects AlphaApollo's thin SFT-engine
    adapter. All heavy imports are inside this function so the module stays import-safe.

    Args:
        config: Hydra OmegaConf config loaded from an off-policy config overlay.
    """
    try:
        from verl.trainer.sft_trainer import auto_set_device

        auto_set_device(config)
    except (ImportError, AttributeError):
        # auto_set_device is NPU autodetect — harmless to skip on CUDA
        pass

    algorithm = str(config.get("alphaapollo", {}).get("off_policy_algo", "sft"))
    if algorithm == "sft":
        from verl.trainer.sft_trainer import run_sft

        run_sft(config)
    elif algorithm == "dpo":
        from alphaapollo.learning.off_policy.algorithms.dpo import run_dpo

        run_dpo(config)
    else:
        raise ValueError(f"Unsupported off-policy algorithm: {algorithm!r}")


@hydra.main(config_path="configs", config_name="sft_trainer", version_base=None)
def main(config):
    main_entry(config)


if __name__ == "__main__":
    main()
