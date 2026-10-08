"""Check exported SFT files through verl's custom dataset hook, without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphaapollo.data_preprocess.io import file_sha256
from alphaapollo.data_preprocess.prepare_atif import checked_manifest


def validate_dataset(directory, tokenizer_path, *, max_length):
    """Load every row with a local tokenizer and return a content-bound receipt."""
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from verl.utils.import_utils import load_extern_object

    from alphaapollo.data_preprocess.preference_dataset import tokenizer_fingerprint

    directory, tokenizer_path = Path(directory), Path(tokenizer_path)
    if not tokenizer_path.is_dir():
        raise ValueError("tokenizer must be an existing local directory")
    manifest = checked_manifest(directory)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    dataset_type = load_extern_object(
        "pkg://alphaapollo.data_preprocess.atif_dataset", "AtifSFTDataset"
    )
    config = {"max_length": max_length, "truncation": "error", "pad_mode": "no_padding"}
    rows = []
    for split, expected in manifest.splits.items():
        dataset = dataset_type(str(directory / f"{split}.parquet"), tokenizer, config)
        structured = [
            json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines()
        ]
        if len(dataset) != expected or len(structured) != expected:
            raise ValueError("dataset row count differs from manifest")
        for i, batch in enumerate(DataLoader(dataset, batch_size=None, num_workers=0)):
            source = structured[i]
            if dataset.rows[i] != {key: source[key] for key in ("messages", "tools")}:
                raise ValueError("Parquet changed the structured messages/tools")
            ids = tokenizer.apply_chat_template(
                source["messages"], tools=source["tools"], tokenize=True
            )
            if batch["input_ids"].tolist() != ids:
                raise ValueError("dataset tokens differ from the full chat template")
            rows.append(
                {
                    "split": split,
                    "row": i,
                    "tokens": len(ids),
                    "assistant_tokens": int(batch["loss_mask"].sum()),
                    "context_tokens": len(ids) - int(batch["loss_mask"].sum()),
                }
            )
    return {
        "format": "atif-sft-validation-v1",
        "status": "passed",
        "training_executed": False,
        "manifest_sha256": file_sha256(directory / "manifest.json"),
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "dataset_class": "alphaapollo.data_preprocess.atif_dataset.AtifSFTDataset",
        "config": config,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    receipt = validate_dataset(args.dataset, args.tokenizer, max_length=args.max_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        args.output.chmod(0o600)
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
