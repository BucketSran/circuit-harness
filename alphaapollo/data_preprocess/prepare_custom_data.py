"""Prepare user-supplied Parquet, JSONL, or JSON data with explicit column mapping."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from alphaapollo.data_preprocess.core import (
    BuildRequest,
    ExistingPolicy,
    NormalizeContext,
    OutputFormat,
    PreparedDataset,
    PreparedExample,
    prepare_dataset,
    raw_digest,
    task_uid,
)
from alphaapollo.data_preprocess.io import default_root, resolve_hub_revision

__all__ = ["prepare"]


BUILDER_NAME = "custom_data"
BUILDER_VERSION = "1"


def prepare(
    *,
    data_source: str,
    dataset_name: str,
    question_key: str,
    answer_key: str,
    output_root: Path | None = None,
    dataset_version: str = "v1",
    revision: str | None = None,
    splits: Sequence[str] = ("train",),
    id_key: str | None = None,
    metadata_keys: Sequence[str] = (),
    output_format: OutputFormat | str = OutputFormat.PARQUET,
    if_exists: ExistingPolicy | str = ExistingPolicy.REUSE,
    domain: str = "custom",
    grader_id: str = "exact_match",
    dataset_license: str = "unverified",
) -> PreparedDataset:
    """Prepare custom rows without allowing answer columns into public output."""

    split_names = tuple(splits)
    # ``statement`` and ``source_record_id`` are public projections, so pointing
    # either of them at the answer column would publish the gold answer no matter
    # how careful the rest of the mapping is.
    public_aliases = sorted(
        role
        for role, name in (("question_key", question_key), ("id_key", id_key))
        if name == answer_key
    )
    if public_aliases:
        raise ValueError(
            f"answer_key {answer_key!r} is also used as {public_aliases}; those columns are "
            "published in the public row and must not carry the answer"
        )
    if id_key is None and len(split_names) > 1:
        raise ValueError(
            "id_key is required when preparing more than one split; without it record ids "
            "fall back to per-split row positions, which collide across splits"
        )
    required_keys = {question_key, answer_key}
    if id_key is not None:
        required_keys.add(id_key)
    overlap = required_keys.intersection(metadata_keys)
    if overlap:
        raise ValueError(f"metadata_keys repeats protected columns: {sorted(overlap)}")
    local_source = Path(data_source).expanduser().exists()
    if revision is not None and local_source:
        raise ValueError(
            f"revision {revision!r} was given for the local source {data_source!r}; a local "
            "file has no Hub commit and is fingerprinted by its content, so recording the "
            "revision would claim provenance the build does not have"
        )
    # Build the request first: everything the caller stated is validated here,
    # before resolving a Hub commit spends a network round trip on a request
    # that cannot produce a build.
    request = BuildRequest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        data_source=data_source,
        revision=revision,
        splits=split_names,
        output_root=output_root or default_root(),
        output_format=OutputFormat(output_format),
        if_exists=ExistingPolicy(if_exists),
    )
    if revision is None and not local_source:
        # Resolve now so the manifest names the exact commit that was read, even
        # though the caller did not have to look it up.
        revision = resolve_hub_revision(data_source)
        request = replace(request, revision=revision)
    source_uri = (
        Path(data_source).expanduser().resolve().as_uri()
        if local_source
        else f"https://huggingface.co/datasets/{data_source}"
    )

    def normalize(raw: Mapping[str, Any], context: NormalizeContext) -> PreparedExample:
        missing = sorted(key for key in required_keys if key not in raw)
        if missing:
            raise ValueError(f"custom row is missing required columns: {missing}")
        missing_metadata = sorted(key for key in metadata_keys if key not in raw)
        if missing_metadata:
            raise ValueError(f"custom row is missing metadata columns: {missing_metadata}")
        record_id = str(raw[id_key]).strip() if id_key is not None else str(context.source_order)
        statement = str(raw[question_key]).strip()
        answer = str(raw[answer_key]).strip()
        if not record_id or not statement or not answer:
            raise ValueError("custom id, question, and answer values must be non-empty")
        extra = {key: raw[key] for key in metadata_keys}
        return PreparedExample(
            task_uid=task_uid(context.source_id, record_id),
            source_id=context.source_id,
            source_record_id=record_id,
            source_order=context.source_order,
            split=context.split,
            statement=statement,
            domain=domain,
            answer=answer,
            extra=extra,
            grader_id=grader_id,
            source_uri=source_uri,
            source_revision=revision or "local",
            dataset_license=dataset_license,
            raw_digest=raw_digest(raw),
            builder_version=context.builder_version,
        )

    return prepare_dataset(
        request,
        normalize,
        builder_name=BUILDER_NAME,
        builder_version=BUILDER_VERSION,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        required=True,
        help="Hub repo id (owner/name), a local .parquet/.jsonl/.json file, "
        "or a directory holding one <split>.<ext> file per requested split.",
    )
    parser.add_argument(
        "--question-key",
        required=True,
        help="Column holding the problem statement; published as `statement`.",
    )
    parser.add_argument(
        "--answer-key",
        required=True,
        help="Column holding the gold answer; kept private. It must not also be "
        "passed to --question-key, --id-key, or --metadata-keys.",
    )
    parser.add_argument(
        "--id-key",
        help="Column holding a stable per-row id; published as `source_record_id`. "
        "Omitted, the row position within the split is used, which is only safe "
        "for a single-split build. Required when --splits names more than one split.",
    )
    parser.add_argument(
        "--metadata-keys",
        default="",
        help="Comma-separated columns copied verbatim into the private `extra` map "
        "(values must be JSON-serializable). Reusing a question/answer/id column is refused.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_root(),
        help="Root the build is written under; defaults to $ALPHAAPOLLO_DATA_DIR "
        "or ~/.alphaapollo/data_preprocess.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Name the prepared build is stored and referenced under, e.g. geometry_drills. "
        "It is also the `source_id` hashed into every `task_uid`, and the output "
        "directory, so a default would let two unrelated builds collide on one path.",
    )
    parser.add_argument("--dataset-version", default="v1", help="Version directory under the name.")
    parser.add_argument(
        "--revision",
        help="Pin a Hub commit. Omitted, the current one is resolved and recorded. "
        "Refused for a local source, which is fingerprinted by content and has no "
        "commit to name.",
    )
    parser.add_argument(
        "--splits", default="train", help="Comma-separated split names to build, e.g. train,test."
    )
    parser.add_argument(
        "--format",
        choices=[item.value for item in OutputFormat],
        default="parquet",
        help="Output serialization format.",
    )
    parser.add_argument(
        "--if-exists",
        choices=[item.value for item in ExistingPolicy],
        default="reuse",
        help="What to do when the target build directory already exists.",
    )
    parser.add_argument(
        "--domain",
        default="custom",
        help="Public `domain` tag on every row, e.g. math or code.",
    )
    parser.add_argument(
        "--grader-id",
        default="exact_match",
        help="Private `grader_id` recorded for every row; names the grader an "
        "evaluation run should score these answers with.",
    )
    parser.add_argument(
        "--dataset-license",
        default="unverified",
        help="License recorded in the private provenance columns.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = prepare(
        data_source=args.data_source,
        question_key=args.question_key,
        answer_key=args.answer_key,
        id_key=args.id_key,
        metadata_keys=tuple(key.strip() for key in args.metadata_keys.split(",") if key.strip()),
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        revision=args.revision,
        splits=tuple(split.strip() for split in args.splits.split(",") if split.strip()),
        output_format=args.format,
        if_exists=args.if_exists,
        domain=args.domain,
        grader_id=args.grader_id,
        dataset_license=args.dataset_license,
    )
    print(prepared.root)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())
