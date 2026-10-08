"""The real tokenizer and training loader must preserve tool data and loss boundaries."""

import importlib.abc
import json
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pyarrow")

from circuit_harness.data.atif_dataset import AtifSFTDataset


def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    # Deliberately tiny, offline fixture. Real Qwen acceptance is a separate check.
    core = Tokenizer(
        models.WordLevel({"[UNK]": 0, "[PAD]": 1, "answer": 2, "public": 3}, unk_token="[UNK]")
    )
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    result = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]")
    result.chat_template = (
        r"{% if tools %}TOOLS {{ tools | tojson }}{{ '\n' }}{% endif %}"
        "{% for m in messages %}{{ m.role }}: "
        "{% if m.content %}{{ m.content }}{% endif %}"
        "{% if m.tool_calls %}{{ m.tool_calls | tojson }}{% endif %}"
        r"{{ '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    return result


def parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    row = {
        "messages": [
            {"role": "system", "content": "public policy"},
            {"role": "user", "content": "public request"},
            {
                "role": "assistant",
                "content": "answer",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": {"command": "public", "optional": None},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "public observation"},
            {"role": "assistant", "content": "answer final"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "bash", "parameters": {"type": "object", "properties": {}}},
            }
        ],
    }
    path = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist([{key + "_json": json.dumps(value) for key, value in row.items()}]),
        path,
    )
    return path, row


def test_loader_preserves_structures_and_masks_only_assistant_spans(tmp_path):
    import torch

    path, row = parquet(tmp_path)
    tok = tokenizer()
    dataset = AtifSFTDataset(
        str(path), tok, {"max_length": 1000, "truncation": "error", "pad_mode": "no_padding"}
    )
    batch = dataset[0]
    assert batch["input_ids"].tolist() == tok.apply_chat_template(
        row["messages"], tools=row["tools"]
    )
    text = tok.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
    offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    mask = batch["loss_mask"].tolist()
    answer_starts = [text.index("answer"), text.rindex("answer")]
    for begin, end in offsets:
        if any(begin <= start < end for start in answer_starts):
            assert mask[offsets.index((begin, end))] == 1
        if (
            begin < text.index("assistant:")
            or begin >= text.index("tool:")
            and end <= text.rindex("assistant:")
        ):
            assert mask[offsets.index((begin, end))] == 0
    assert int(torch.sum(batch["loss_mask"])) > 0
    assert dataset.rows[0]["messages"] == row["messages"]
    assert dataset.rows[0]["tools"] == row["tools"]
    with pytest.raises(ValueError, match="max_length"):
        AtifSFTDataset(str(path), tok, {"max_length": 2})[0]


def test_padding_is_excluded_and_rewritten_context_is_rejected(tmp_path):
    path, row = parquet(tmp_path)
    tok = tokenizer()
    length = len(tok.apply_chat_template(row["messages"], tools=row["tools"]))
    batch = AtifSFTDataset(str(path), tok, {"max_length": length + 5})[0]
    assert batch["attention_mask"].tolist() == [1] * length + [0] * 5
    assert batch["loss_mask"][-5:].tolist() == [0] * 5
    assert batch["input_ids"][-5:].tolist() == [tok.pad_token_id] * 5
    tok.chat_template = "{{ messages | length }} " + tok.chat_template
    with pytest.raises(ValueError, match="rewrites prior context"):
        AtifSFTDataset(str(path), tok, {"max_length": 1000})[0]


def test_offline_validation_without_verl_rejects_changed_data(tmp_path, monkeypatch):
    class BlockTrainer(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"verl", "alphaapollo"}:
                raise AssertionError("retired training dependency: " + fullname)

    monkeypatch.setattr(sys, "meta_path", [BlockTrainer(), *sys.meta_path])
    from circuit_harness.data.io import file_sha256
    from circuit_harness.data.prepare_atif import AtifDatasetManifest
    from circuit_harness.data.validate_atif import validate_dataset

    path, row = parquet(tmp_path)
    structured = tmp_path / "train.jsonl"
    structured.write_text(json.dumps(row) + "\n")
    manifest = AtifDatasetManifest(
        reasoning="exclude",
        allow_reconstructed_context=True,
        sources=[{"session_id": "fixture"}],
        splits={"train": 1},
        files={p.name: file_sha256(p) for p in (path, structured)},
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    tok_dir = tmp_path / "tokenizer"
    tokenizer().save_pretrained(tok_dir)
    result = validate_dataset(tmp_path, tok_dir, max_length=1000)
    assert result["status"] == "passed"
    assert result["training_executed"] is False
    assert len(result["tokenizer_sha256"]) == 64
    assert 0 < result["rows"][0]["assistant_tokens"] < result["rows"][0]["tokens"]
    with pytest.raises(ValueError, match="max_length"):
        validate_dataset(tmp_path, tok_dir, max_length=2)
    structured.write_text("{}\n")
    with pytest.raises(ValueError, match="integrity"):
        validate_dataset(tmp_path, tok_dir, max_length=1000)
