"""Loss-preserving text trajectory dataset for external trainers.

Render the whole conversation once. Prefix-checked character spans mark assistant
completions, and fast-tokenizer offsets project those spans onto the exact full
sequence. Context-dependent templates that rewrite earlier text fail visibly.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class AtifSFTDataset(Dataset):
    """Read JSON-column Parquet without Arrow's heterogeneous-struct coercion.

    Supports text, right padding or no padding, and truncation='error' only.
    The constructor supports external custom dataset hooks; no trainer is bundled.
    """

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        import pyarrow.parquet as pq

        if processor is not None or not getattr(tokenizer, "is_fast", False):
            raise ValueError("text ATIF requires a fast tokenizer and no multimodal processor")
        if config.get("truncation", "error") != "error":
            raise ValueError("ATIF SFT requires truncation=error")
        self.max_length = int(config.get("max_length", 2048))
        self.pad_mode = config.get("pad_mode", "right")
        if self.max_length < 1 or self.pad_mode not in {"right", "no_padding"}:
            raise ValueError("invalid max_length or pad_mode")
        self.tokenizer = tokenizer
        self.template_kwargs = dict(config.get("apply_chat_template_kwargs", {}))
        if set(self.template_kwargs).intersection(
            {"messages", "tools", "tokenize", "add_generation_prompt"}
        ):
            raise ValueError("template kwargs cannot override conversation rendering")
        paths = [parquet_files] if isinstance(parquet_files, (str, Path)) else parquet_files
        self.rows = []
        for path in paths:
            for encoded in pq.read_table(path).to_pylist():
                row = {key: json.loads(encoded[key + "_json"]) for key in ("messages", "tools")}
                if not isinstance(row["messages"], list) or not isinstance(row["tools"], list):
                    raise ValueError("messages and tools must be arrays")
                self.rows.append(row)
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError("empty SFT dataset")

    def __len__(self):
        return len(self.rows)

    def _render(self, messages, tools, *, generation=False):
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=generation,
            **self.template_kwargs,
        )

    def __getitem__(self, index):
        row = self.rows[index]
        messages, tools = row["messages"], row["tools"]
        if (
            not messages
            or messages[0].get("role") != "system"
            or any(
                m.get("role") not in {"system", "user", "assistant", "tool"}
                or not isinstance(m.get("content"), str)
                for m in messages
            )
        ):
            raise ValueError("invalid text conversation")
        rendered = self._render(messages, tools)
        spans = []
        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            before = self._render(messages[:i], tools, generation=True)
            after = self._render(messages[: i + 1], tools)
            if not after.startswith(before) or not rendered.startswith(after):
                raise ValueError(
                    "chat template rewrites prior context; cannot establish assistant loss spans"
                )
            if len(after) == len(before):
                raise ValueError("assistant completion is empty under the selected template")
            spans.append((len(before), len(after)))
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        if len(ids) > self.max_length:
            raise ValueError(
                f"sequence has {len(ids)} tokens, exceeds max_length={self.max_length}"
            )
        mask, span_index = [], 0
        for start, end in encoded["offset_mapping"]:
            while span_index < len(spans) and spans[span_index][1] <= start:
                span_index += 1
            if span_index == len(spans) or end <= spans[span_index][0]:
                mask.append(0)
            else:
                begin, stop = spans[span_index]
                if start < begin or end > stop:
                    raise ValueError("a token crosses a context/assistant boundary")
                mask.append(int(end > start))
        if not any(mask):
            raise ValueError("no assistant training tokens")
        length = len(ids)
        attention = [1] * length
        positions = list(range(length))
        if self.pad_mode == "right":
            if self.tokenizer.pad_token_id is None:
                raise ValueError("right padding requires a pad token")
            padding = self.max_length - length
            ids += [self.tokenizer.pad_token_id] * padding
            mask += [0] * padding
            attention += [0] * padding
            positions += [0] * padding
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "position_ids": torch.tensor(positions, dtype=torch.long),
        }
