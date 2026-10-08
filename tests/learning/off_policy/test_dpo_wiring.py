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

"""Entry dispatch + config-namespace gates for offline DPO (no GPU, no distributed)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from alphaapollo.learning.off_policy import main_off_policy

_CONFIG_DIR = Path(main_off_policy.__file__).resolve().parent / "configs"


def _resolve(config_name: str):
    pytest.importorskip("verl", reason="config composition requires the optional Learning runtime")
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        return compose(config_name=config_name)


def test_dpo_config_resolves_without_touching_verl_dataclass_namespaces():
    """AA knobs must resolve under alphaapollo.*, never in a verl namespace."""
    config = _resolve("dpo_trainer")

    assert config.alphaapollo.off_policy_algo == "dpo"
    assert config.alphaapollo.dpo.loss_type == "sigmoid"
    assert pytest.approx(config.alphaapollo.dpo.beta) == 0.1
    # v1 batching invariants are config-level, not just runtime asserts
    assert config.data.use_dynamic_bsz is False
    assert config.data.micro_batch_size_per_gpu % 2 == 0
    assert config.data.pad_mode == "no_padding"
    # no stray codapo-style knob leaked into a strict verl dataclass namespace
    assert "dpo" not in config.data
    OmegaConf.resolve(config)  # interpolations (e.g. reference_model_path) must not dangle


def test_main_entry_dispatches_dpo(monkeypatch):
    pytest.importorskip("torch", reason="DPO requires the optional Learning runtime")
    from alphaapollo.learning.off_policy.algorithms import dpo

    called = {}
    monkeypatch.setattr(dpo, "run_dpo", lambda config: called.setdefault("dpo", config))
    config = OmegaConf.create({"alphaapollo": {"off_policy_algo": "dpo"}})

    main_off_policy.main_entry(config)

    assert called["dpo"] is config


def test_main_entry_defaults_to_verl_native_sft(monkeypatch):
    pytest.importorskip("verl", reason="SFT requires the optional Learning runtime")
    import verl.trainer.sft_trainer as sft_trainer

    called = {}
    monkeypatch.setattr(sft_trainer, "run_sft", lambda config: called.setdefault("sft", config))
    config = OmegaConf.create({"alphaapollo": {}})  # no off_policy_algo → SFT passthrough

    main_off_policy.main_entry(config)

    assert called["sft"] is config


def test_main_entry_rejects_unknown_algorithm():
    config = OmegaConf.create({"alphaapollo": {"off_policy_algo": "kto"}})

    with pytest.raises(ValueError, match="Unsupported off-policy algorithm"):
        main_off_policy.main_entry(config)
