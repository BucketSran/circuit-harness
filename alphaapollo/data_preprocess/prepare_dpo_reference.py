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

"""Precompute frozen-reference completion log-probabilities for offline DPO.

This is deliberately a separate process from policy training: after it exits,
the DPO FSDP job holds only the trainable policy model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from alphaapollo.data_preprocess.preference_dataset import (
    REF_CHOSEN_COLUMN,
    REF_REJECTED_COLUMN,
    PairedPreferenceDataset,
    build_reference_metadata,
    reference_metadata_path,
)


def _pad_sequences(
    sequences: list[dict[str, torch.Tensor]],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(sequence["input_ids"].shape[0] for sequence in sequences)
    batch_size = len(sequences)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        length = sequence["input_ids"].shape[0]
        input_ids[index, :length] = sequence["input_ids"]
        attention_mask[index, :length] = 1
        loss_mask[index, :length] = sequence["loss_mask"].to(torch.bool)
    return input_ids, attention_mask, loss_mask


def completion_logps_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Independent padded-forward oracle used to create the reference cache."""

    if logits.shape[:2] != input_ids.shape or input_ids.shape != loss_mask.shape:
        raise ValueError("logits, input_ids, and loss_mask have incompatible shapes")
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:].to(logits.dtype)
    valid_tokens = shift_mask.sum(dim=-1)
    if torch.any(valid_tokens <= 0):
        raise ValueError("Every DPO sequence must contain at least one completion token")
    token_logps = -F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape_as(shift_labels)
    return (token_logps * shift_mask).sum(dim=-1)


def _json_object(value: str) -> dict:
    """Parse a command-line JSON object for chat-template options."""

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute reference log-probabilities for offline DPO"
    )
    parser.add_argument("--input", required=True, help="Input prompt/chosen/rejected parquet")
    parser.add_argument(
        "--output", required=True, help="Output parquet with reference log-probability columns"
    )
    parser.add_argument(
        "--model", required=True, help="Frozen HuggingFace reference model path or ID"
    )
    parser.add_argument(
        "--revision", default=None, help="Optional HuggingFace model/tokenizer revision"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Preference pairs per forward batch"
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--truncation", choices=["error", "left", "right"], default="right")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--chosen-key", default="chosen")
    parser.add_argument("--rejected-key", default="rejected")
    parser.add_argument(
        "--apply-chat-template-kwargs",
        type=_json_object,
        default={},
        help="JSON object forwarded to tokenizer.apply_chat_template",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--device", default=None, help="Default: cuda when available, otherwise cpu"
    )
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Reference tokenizer must define either pad_token_id or eos_token_id")

    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()

    data_config = {
        "pad_mode": "no_padding",
        "max_length": args.max_length,
        "truncation": args.truncation,
        "prompt_key": args.prompt_key,
        "chosen_key": args.chosen_key,
        "rejected_key": args.rejected_key,
        "apply_chat_template_kwargs": args.apply_chat_template_kwargs,
        "require_reference_logps": False,
        "validate_reference_metadata": False,
    }
    dataset = PairedPreferenceDataset(
        parquet_files=args.input,
        tokenizer=tokenizer,
        config=data_config,
        processor=None,
    )

    chosen_logps: list[float] = []
    rejected_logps: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            items = [
                dataset[index] for index in range(start, min(start + args.batch_size, len(dataset)))
            ]
            sequences = [
                sequence for item in items for sequence in (item["chosen"], item["rejected"])
            ]
            input_ids, attention_mask, loss_mask = _pad_sequences(
                sequences, pad_token_id=pad_token_id
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            loss_mask = loss_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            sequence_logps = (
                completion_logps_from_logits(outputs.logits, input_ids, loss_mask).float().cpu()
            )
            chosen_logps.extend(sequence_logps[0::2].tolist())
            rejected_logps.extend(sequence_logps[1::2].tolist())

    dataframe = pd.read_parquet(args.input)
    if len(dataframe) != len(chosen_logps):
        raise RuntimeError(
            f"Reference output size mismatch: {len(chosen_logps)} != {len(dataframe)}"
        )
    dataframe[REF_CHOSEN_COLUMN] = chosen_logps
    dataframe[REF_REJECTED_COLUMN] = rejected_logps

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    metadata = build_reference_metadata(
        dataframe,
        tokenizer=tokenizer,
        reference_model_path=args.model,
        reference_model_revision=args.revision,
        max_length=args.max_length,
        truncation=args.truncation,
        prompt_key=args.prompt_key,
        chosen_key=args.chosen_key,
        rejected_key=args.rejected_key,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
    )
    reference_metadata_path(output_path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
