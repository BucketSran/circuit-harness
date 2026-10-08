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

"""Static chosen/rejected data for offline preference optimization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from verl.utils import tensordict_utils as tu
from verl.utils.fs import copy_local_path_from_hdfs

REF_CHOSEN_COLUMN = "ref_logprob_chosen"
REF_REJECTED_COLUMN = "ref_logprob_rejected"
REFERENCE_METADATA_SCHEMA_VERSION = 2


def _to_python(value: Any) -> Any:
    """Convert parquet/Arrow nested values into ordinary Python containers."""

    if isinstance(value, np.ndarray):
        return [_to_python(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _to_python(value.item())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_python(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _to_python(item) for key, item in value.items()}
    if hasattr(value, "as_py"):
        return _to_python(value.as_py())
    if value is pd.NA or isinstance(value, float) and math.isnan(value):
        return None
    return value


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash tokenization semantics while ignoring where the tokenizer was loaded from."""

    payload = {
        "chat_template": str(getattr(tokenizer, "chat_template", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None and hasattr(backend_tokenizer, "to_str"):
        payload["backend_tokenizer"] = backend_tokenizer.to_str()
    elif callable(getattr(tokenizer, "get_vocab", None)):
        payload["vocab"] = sorted(
            (str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items()
        )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reference_metadata_path(parquet_path: str | Path) -> Path:
    return Path(f"{parquet_path}.refmeta.json")


def preference_dataset_fingerprint(
    dataframe: pd.DataFrame,
    *,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
) -> str:
    """Hash preference inputs and cached reference log-probabilities in row order."""

    columns = (
        prompt_key,
        chosen_key,
        rejected_key,
        REF_CHOSEN_COLUMN,
        REF_REJECTED_COLUMN,
    )
    missing = sorted(set(columns).difference(dataframe.columns))
    if missing:
        raise ValueError(f"Preference dataframe is missing fingerprint columns: {missing}")

    digest = hashlib.sha256()
    header = json.dumps(
        {"format_version": 1, "columns": columns, "num_preferences": len(dataframe)},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    for values in dataframe.loc[:, list(columns)].itertuples(index=False, name=None):
        encoded = json.dumps(
            [_to_python(value) for value in values],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def build_reference_metadata(
    dataframe: pd.DataFrame,
    *,
    tokenizer: Any,
    reference_model_path: str,
    reference_model_revision: str | None,
    max_length: int,
    truncation: str,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
    apply_chat_template_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the cache contract shared by preprocessing and training."""

    return {
        "schema_version": REFERENCE_METADATA_SCHEMA_VERSION,
        "reference_model_path": str(reference_model_path),
        "reference_model_revision": (
            None if reference_model_revision is None else str(reference_model_revision)
        ),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "max_length": int(max_length),
        "truncation": str(truncation),
        "prompt_key": str(prompt_key),
        "chosen_key": str(chosen_key),
        "rejected_key": str(rejected_key),
        "apply_chat_template_kwargs": _to_python(apply_chat_template_kwargs),
        "dataset_fingerprint": preference_dataset_fingerprint(
            dataframe,
            prompt_key=prompt_key,
            chosen_key=chosen_key,
            rejected_key=rejected_key,
        ),
        "logp_normalization": "sum",
        "num_preferences": len(dataframe),
    }


class PairedPreferenceDataset(Dataset):
    """Read pairwise preferences and tokenize both completions identically.

    Constructor arguments intentionally match verl's ``create_sft_dataset``
    custom-class contract.
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
    ):
        if processor is not None:
            raise NotImplementedError("Offline DPO v1 supports text-only preference data")
        if str(config.get("pad_mode", "no_padding")) != "no_padding":
            raise ValueError("PairedPreferenceDataset requires data.pad_mode=no_padding")

        self.tokenizer = tokenizer
        self.max_length = int(config.get("max_length", 1024))
        self.truncation = str(config.get("truncation", "error"))
        if self.truncation not in {"error", "left", "right"}:
            raise ValueError(f"Unsupported DPO truncation mode: {self.truncation!r}")

        self.prompt_key = str(config.get("prompt_key", "prompt"))
        self.chosen_key = str(config.get("chosen_key", "chosen"))
        self.rejected_key = str(config.get("rejected_key", "rejected"))
        self.sample_id_key = str(config.get("sample_id_key", "sample_id"))
        self.require_reference_logps = bool(config.get("require_reference_logps", True))
        self.validate_reference_metadata = bool(config.get("validate_reference_metadata", False))
        self.reference_model_path = config.get("reference_model_path", None)
        self.reference_model_revision = config.get("reference_model_revision", None)
        self.apply_chat_template_kwargs = dict(config.get("apply_chat_template_kwargs", {}))

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        self.parquet_files = [
            copy_local_path_from_hdfs(path, verbose=True) for path in parquet_files
        ]
        if not self.parquet_files:
            raise ValueError("At least one preference parquet file is required")

        frames = [pd.read_parquet(path, dtype_backend="pyarrow") for path in self.parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        if max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples].reset_index(drop=True)

        required = {self.prompt_key, self.chosen_key, self.rejected_key}
        if self.require_reference_logps:
            required.update({REF_CHOSEN_COLUMN, REF_REJECTED_COLUMN})
        missing = sorted(required.difference(self.dataframe.columns))
        if missing:
            raise ValueError(f"Preference parquet is missing required columns: {missing}")
        if self.dataframe.empty:
            raise ValueError("Preference dataset is empty")

        if self.validate_reference_metadata:
            self._validate_metadata(frames)

    def _validate_metadata(self, frames: list[pd.DataFrame]) -> None:
        if self.reference_model_path is None:
            raise ValueError(
                "data.reference_model_path is required when reference metadata "
                "validation is enabled"
            )
        for parquet_file, dataframe in zip(self.parquet_files, frames, strict=True):
            metadata_file = reference_metadata_path(parquet_file)
            if not metadata_file.is_file():
                raise ValueError(f"Reference metadata sidecar is missing: {metadata_file}")
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            expected_metadata = build_reference_metadata(
                dataframe,
                tokenizer=self.tokenizer,
                reference_model_path=self.reference_model_path,
                reference_model_revision=self.reference_model_revision,
                max_length=self.max_length,
                truncation=self.truncation,
                prompt_key=self.prompt_key,
                chosen_key=self.chosen_key,
                rejected_key=self.rejected_key,
                apply_chat_template_kwargs=self.apply_chat_template_kwargs,
            )
            for key, expected in expected_metadata.items():
                if key not in metadata:
                    raise ValueError(f"Reference metadata for {parquet_file} is missing {key!r}")
                actual = metadata.get(key)
                if actual != expected:
                    raise ValueError(
                        f"Reference metadata mismatch for {parquet_file}: "
                        f"{key}={actual!r}, expected {expected!r}"
                    )

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def _response_text(value: Any, *, field_name: str) -> str:
        value = _to_python(value)
        if isinstance(value, dict):
            if value.get("role", "assistant") != "assistant":
                raise ValueError(f"{field_name} message must have role='assistant'")
            value = value.get("content", "")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} must be a non-empty string or assistant message")
        return value

    @staticmethod
    def _prompt_messages(value: Any) -> list[dict[str, Any]]:
        value = _to_python(value)
        if isinstance(value, str):
            value = [{"role": "user", "content": value}]
        if not isinstance(value, list) or not value:
            raise ValueError("prompt must be a non-empty string or message list")
        messages = []
        for message in value:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError("Every prompt message must contain role and content")
            if message["role"] == "assistant":
                raise ValueError("prompt must not already contain an assistant response")
            messages.append({"role": str(message["role"]), "content": message["content"]})
        return messages

    @staticmethod
    def _token_ids(value: Any) -> list[int]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if isinstance(value, dict):
            value = value["input_ids"]
        if value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError("Tokenizer returned an unexpected batched chat template result")
            value = value[0]
        return [int(token_id) for token_id in value]

    def _tokenize(self, prompt: list[dict[str, Any]], response: str) -> dict[str, torch.Tensor]:
        prompt_ids = self._token_ids(
            self.tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                **self.apply_chat_template_kwargs,
            )
        )
        full_ids = self._token_ids(
            self.tokenizer.apply_chat_template(
                [*prompt, {"role": "assistant", "content": response}],
                tokenize=True,
                add_generation_prompt=False,
                **self.apply_chat_template_kwargs,
            )
        )
        if not full_ids[: len(prompt_ids)] == prompt_ids:
            raise ValueError(
                "Tokenizer chat template does not preserve prompt+generation-prefix "
                "as a prefix of the full response"
            )

        loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
        if len(full_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError(
                    f"DPO sequence length {len(full_ids)} exceeds max_length={self.max_length}"
                )
            if self.truncation == "right":
                full_ids = full_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            else:
                full_ids = full_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]

        if not full_ids:
            raise ValueError("DPO tokenization produced an empty sequence")
        if sum(loss_mask) == 0:
            raise ValueError("DPO tokenization/truncation removed every completion token")

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "position_ids": torch.arange(input_ids.shape[0], dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        prompt = self._prompt_messages(row[self.prompt_key])
        chosen = self._response_text(row[self.chosen_key], field_name=self.chosen_key)
        rejected = self._response_text(row[self.rejected_key], field_name=self.rejected_key)

        if self.require_reference_logps:
            ref_chosen = float(row[REF_CHOSEN_COLUMN])
            ref_rejected = float(row[REF_REJECTED_COLUMN])
            if not np.isfinite(ref_chosen) or not np.isfinite(ref_rejected):
                raise ValueError(
                    f"Reference log-probabilities must be finite at preference row {index}"
                )
        else:
            ref_chosen = 0.0
            ref_rejected = 0.0

        sample_id = (
            row[self.sample_id_key] if self.sample_id_key in self.dataframe.columns else index
        )
        return {
            "pair_id": int(index),
            "sample_id": str(sample_id),
            "chosen": self._tokenize(prompt, chosen),
            "rejected": self._tokenize(prompt, rejected),
            REF_CHOSEN_COLUMN: ref_chosen,
            REF_REJECTED_COLUMN: ref_rejected,
        }


class PairedPreferenceCollator:
    """Flatten pair items into adjacent no-padding model sequences."""

    _SEQUENCE_KEYS = ("input_ids", "position_ids", "loss_mask")

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Cannot collate an empty DPO preference batch")

        rows: list[dict[str, Any]] = []
        for item in batch:
            pair_id = int(item["pair_id"])
            rows.append(
                {
                    **item["chosen"],
                    "reference_logp": float(item[REF_CHOSEN_COLUMN]),
                    "pair_id": pair_id,
                    "is_chosen": True,
                }
            )
            rows.append(
                {
                    **item["rejected"],
                    "reference_logp": float(item[REF_REJECTED_COLUMN]),
                    "pair_id": pair_id,
                    "is_chosen": False,
                }
            )

        output = {
            key: tu.nested_tensor_from_tensor_list([row[key] for row in rows])
            for key in self._SEQUENCE_KEYS
        }
        output.update(
            {
                "reference_logp": torch.tensor(
                    [row["reference_logp"] for row in rows], dtype=torch.float32
                ),
                "pair_id": torch.tensor([row["pair_id"] for row in rows], dtype=torch.long),
                "is_chosen": torch.tensor([row["is_chosen"] for row in rows], dtype=torch.bool),
            }
        )
        return output


__all__ = [
    "PairedPreferenceCollator",
    "PairedPreferenceDataset",
    "REF_CHOSEN_COLUMN",
    "REF_REJECTED_COLUMN",
    "build_reference_metadata",
    "preference_dataset_fingerprint",
    "reference_metadata_path",
    "tokenizer_fingerprint",
]
