from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from alphaapollo.data_preprocess.preference_dataset import (
    REF_CHOSEN_COLUMN,
    REF_REJECTED_COLUMN,
    PairedPreferenceCollator,
    PairedPreferenceDataset,
    build_reference_metadata,
    reference_metadata_path,
    tokenizer_fingerprint,
)
from alphaapollo.data_preprocess.prepare_dpo_reference import completion_logps_from_logits


class FakeChatTokenizer:
    name_or_path = "fake-chat-tokenizer"
    chat_template = "fake-template"
    vocab_size = 64
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    @staticmethod
    def _text_ids(text):
        return [10 + (ord(char) % 40) for char in str(text)]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize
        prompt = [self.bos_token_id]
        assistant = None
        for message in messages:
            if message["role"] == "assistant":
                assistant = message["content"]
                break
            prompt.extend(self._text_ids(message["role"]))
            prompt.extend(self._text_ids(message["content"]))
        prompt.append(8)  # assistant generation prefix
        if add_generation_prompt:
            return prompt
        return [*prompt, *self._text_ids(assistant), self.eos_token_id]


def _write_preferences(path):
    pd.DataFrame(
        {
            "sample_id": ["p0", "p1"],
            "prompt": [
                [{"role": "user", "content": "q0"}],
                [{"role": "user", "content": "q1"}],
            ],
            "chosen": ["good", "yes"],
            "rejected": ["bad", "no"],
            REF_CHOSEN_COLUMN: [-3.0, -2.0],
            REF_REJECTED_COLUMN: [-4.0, -3.0],
        }
    ).to_parquet(path, index=False)


def _write_reference_metadata(
    parquet,
    tokenizer,
    *,
    reference_model_revision=None,
    max_length=128,
    apply_chat_template_kwargs=None,
):
    metadata = build_reference_metadata(
        pd.read_parquet(parquet),
        tokenizer=tokenizer,
        reference_model_path="fake-model",
        reference_model_revision=reference_model_revision,
        max_length=max_length,
        truncation="right",
        prompt_key="prompt",
        chosen_key="chosen",
        rejected_key="rejected",
        apply_chat_template_kwargs=apply_chat_template_kwargs or {},
    )
    reference_metadata_path(parquet).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_tokenizer_fingerprint_ignores_load_path_but_tracks_semantics():
    tokenizer = SimpleNamespace(
        name_or_path="original/model",
        chat_template="template-a",
        vocab_size=3,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        get_vocab=lambda: {"<bos>": 0, "<eos>": 1, "<pad>": 2},
    )
    relocated = SimpleNamespace(**{**vars(tokenizer), "name_or_path": "checkpoint/model"})
    changed_template = SimpleNamespace(**{**vars(tokenizer), "chat_template": "template-b"})
    changed_vocab = SimpleNamespace(
        **{
            **vars(tokenizer),
            "get_vocab": lambda: {"<bos>": 0, "<eos>": 2, "<pad>": 1},
        }
    )

    assert tokenizer_fingerprint(relocated) == tokenizer_fingerprint(tokenizer)
    assert tokenizer_fingerprint(changed_template) != tokenizer_fingerprint(tokenizer)
    assert tokenizer_fingerprint(changed_vocab) != tokenizer_fingerprint(tokenizer)


def test_preference_dataset_and_collator_preserve_adjacent_pairs(tmp_path):
    parquet = tmp_path / "preferences.parquet"
    _write_preferences(parquet)
    dataset = PairedPreferenceDataset(
        parquet_files=str(parquet),
        tokenizer=FakeChatTokenizer(),
        config={
            "pad_mode": "no_padding",
            "max_length": 128,
            "truncation": "error",
            "require_reference_logps": True,
        },
    )

    first = dataset[0]
    assert first["chosen"]["loss_mask"].sum() > 0
    assert first["chosen"]["loss_mask"][0] == 0

    batch = PairedPreferenceCollator()([dataset[0], dataset[1]])
    assert batch["input_ids"].is_nested
    assert batch["input_ids"].shape[0] == 4
    assert batch["pair_id"].tolist() == [0, 0, 1, 1]
    assert batch["is_chosen"].tolist() == [True, False, True, False]
    torch.testing.assert_close(batch["reference_logp"], torch.tensor([-3.0, -4.0, -2.0, -3.0]))


def test_reference_logp_oracle_uses_completion_tokens_only():
    logits = torch.zeros(1, 4, 3)
    input_ids = torch.tensor([[0, 1, 2, 1]])
    loss_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)

    actual = completion_logps_from_logits(logits, input_ids, loss_mask)

    # Uniform logits and two completion labels -> 2 * log(1/3).
    expected = 2.0 * torch.log(torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(actual, expected.unsqueeze(0))


def test_reference_metadata_mismatch_fails_loud(tmp_path):
    parquet = tmp_path / "preferences.parquet"
    _write_preferences(parquet)
    tokenizer = FakeChatTokenizer()
    _write_reference_metadata(parquet, tokenizer, max_length=64)

    with pytest.raises(ValueError, match="max_length"):
        PairedPreferenceDataset(
            parquet_files=str(parquet),
            tokenizer=tokenizer,
            config={
                "pad_mode": "no_padding",
                "max_length": 128,
                "truncation": "right",
                "require_reference_logps": True,
                "validate_reference_metadata": True,
                "reference_model_path": "fake-model",
            },
        )


def test_matching_reference_metadata_is_accepted(tmp_path):
    parquet = tmp_path / "preferences.parquet"
    _write_preferences(parquet)
    tokenizer = FakeChatTokenizer()
    _write_reference_metadata(parquet, tokenizer, reference_model_revision="rev-a")

    dataset = PairedPreferenceDataset(
        parquet_files=str(parquet),
        tokenizer=tokenizer,
        config={
            "pad_mode": "no_padding",
            "max_length": 128,
            "truncation": "right",
            "require_reference_logps": True,
            "validate_reference_metadata": True,
            "reference_model_path": "fake-model",
            "reference_model_revision": "rev-a",
        },
    )

    assert len(dataset) == 2


def test_stale_reference_metadata_schema_is_rejected(tmp_path):
    parquet = tmp_path / "preferences.parquet"
    _write_preferences(parquet)
    tokenizer = FakeChatTokenizer()
    _write_reference_metadata(parquet, tokenizer)
    metadata_path = reference_metadata_path(parquet)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] -= 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        PairedPreferenceDataset(
            parquet_files=str(parquet),
            tokenizer=tokenizer,
            config={
                "pad_mode": "no_padding",
                "max_length": 128,
                "truncation": "right",
                "require_reference_logps": True,
                "validate_reference_metadata": True,
                "reference_model_path": "fake-model",
            },
        )


@pytest.mark.parametrize(
    ("changed_column", "changed_value"),
    [
        ("chosen", "changed-after-reference-precompute"),
        (REF_CHOSEN_COLUMN, -999.0),
    ],
)
def test_reference_metadata_rejects_changed_cache_content(
    tmp_path,
    changed_column,
    changed_value,
):
    parquet = tmp_path / "preferences.parquet"
    _write_preferences(parquet)
    tokenizer = FakeChatTokenizer()
    _write_reference_metadata(parquet, tokenizer)
    dataframe = pd.read_parquet(parquet)
    dataframe.loc[0, changed_column] = changed_value
    dataframe.to_parquet(parquet, index=False)

    with pytest.raises(ValueError, match="dataset_fingerprint"):
        PairedPreferenceDataset(
            parquet_files=str(parquet),
            tokenizer=tokenizer,
            config={
                "pad_mode": "no_padding",
                "max_length": 128,
                "truncation": "right",
                "require_reference_logps": True,
                "validate_reference_metadata": True,
                "reference_model_path": "fake-model",
            },
        )


@pytest.mark.parametrize(
    ("config_override", "error_key"),
    [
        ({"reference_model_revision": "rev-b"}, "reference_model_revision"),
        ({"apply_chat_template_kwargs": {"enable_thinking": False}}, "apply_chat_template_kwargs"),
    ],
)
def test_reference_metadata_rejects_changed_preprocessing_contract(
    tmp_path,
    config_override,
    error_key,
):
    parquet = tmp_path / "preferences.parquet"
    _write_preferences(parquet)
    tokenizer = FakeChatTokenizer()
    _write_reference_metadata(parquet, tokenizer, reference_model_revision="rev-a")
    config = {
        "pad_mode": "no_padding",
        "max_length": 128,
        "truncation": "right",
        "require_reference_logps": True,
        "validate_reference_metadata": True,
        "reference_model_path": "fake-model",
        "reference_model_revision": "rev-a",
    }
    config.update(config_override)

    with pytest.raises(ValueError, match=error_key):
        PairedPreferenceDataset(
            parquet_files=str(parquet),
            tokenizer=tokenizer,
            config=config,
        )
