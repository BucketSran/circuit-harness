"""Prepare the vendored RoboCasa365 task-suite snapshot.

RoboCasa tasks are not Q&A records: correctness is decided by the
simulator's success predicate, not by comparing an answer string. The row
records the contract's environment-grading shape — ``answer="environment"``
with ``answer_type="none"`` — while the real payload, the
``(task_name, layout_id, style_id, scene_seed)`` scene triplet, travels in
``env_payload``, the reserved private channel for in-loop environment
reward. ``grader_id`` is ``environment_success``: the environment, not a
text scorer, decides the outcome.

The private payload follows the shared robotics envelope contract:
``benchmark`` and ``environment_version`` are explicit named fields (they
feed ``RobotTask.benchmark`` / ``RobotTask.environment_version`` in the
integration layer), and everything else in the payload is backend task
metadata. The row's official per-task horizon — the upstream
``dataset_registry`` value the snapshot carries — rides as
``max_episode_steps`` so episodes terminate on the task's own budget
instead of the env-wide constructor default. The
environment version is a REQUIRED build-time input, not provenance: it
names the robocasa/robosuite checkout that sampled the scenes, which is
exactly what decides the instructions' wording — a dataset prepared
against one checkout will not validate on another (#256 review).

The snapshot is regenerated deterministically from the installed
RoboCasa365 runtime (``generate_robocasa_snapshot``), following the ENPIRE
(arXiv 2606.19980) matched-evaluation protocol of task-stable seeds
derived from generator seed 42; instructions are recorded from each built
scene's ``ep_meta`` lang, so they embed the sampled object name.

The snapshot is an AlphaApollo-derived task-suite manifest, not the
official RoboCasa demonstration datasets (LeRobot trajectory data for VLA
training, out of scope for this slice); its provenance — protocol, seed,
upstream pins, horizon sourcing — is recorded beside it in
``snapshots/robocasa365_tasks.meta.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
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
from alphaapollo.data_preprocess.io import default_root

__all__ = ["SNAPSHOT_PATH", "prepare"]

BUILDER_NAME = "robocasa"
BUILDER_VERSION = "1"
# The vendored snapshot ships as package data beside this module; io.py
# fingerprints a local path by content, which is exactly the provenance a
# machine-generated snapshot has (no Hub commit to name).
SNAPSHOT_PATH = Path(__file__).resolve().parent / "snapshots" / "robocasa365_tasks.jsonl"
_TRIPLET_FIELDS = ("task_name", "layout_id", "style_id", "scene_seed")
_MAPPED_FIELDS = frozenset({"id", "instruction", "horizon", *_TRIPLET_FIELDS})


def _integer_field(raw: Mapping[str, Any], field: str, record_id: str, *, minimum: int) -> int:
    """Validate one integer row field, refusing silent truncation."""

    if field not in raw:
        raise ValueError(f"robocasa row {record_id!r} is missing {field}")
    value = raw[field]
    if isinstance(value, bool):
        # int(True) == 1 would silently pass validation (same stance
        # prepare_aime takes on boolean answers).
        raise ValueError(f"robocasa row {record_id!r} has a boolean {field}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"robocasa row {record_id!r} has a non-integer {field}: {value!r}")
    number = int(value)
    if number < minimum:
        raise ValueError(
            f"robocasa row {record_id!r} has {field}={number}; robocasa expects >= {minimum}"
        )
    return number


def _snapshot_path(data_source: str | None) -> Path:
    return Path(data_source).expanduser().resolve() if data_source is not None else SNAPSHOT_PATH


def _snapshot_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"robocasa snapshot is empty: {path}")
    return rows


def prepare(
    *,
    environment_version: str,
    dataset_name: str = "robocasa365_tasks",
    data_source: str | None = None,
    output_root: Path | None = None,
    dataset_version: str = "v1",
    splits: Sequence[str] = ("train",),
    output_format: OutputFormat | str = OutputFormat.PARQUET,
    if_exists: ExistingPolicy | str = ExistingPolicy.REUSE,
) -> PreparedDataset:
    """Prepare RoboCasa task-suite rows with the scene triplet in ``env_payload``.

    ``environment_version`` is required and names the robocasa+robosuite
    checkout that generated the rows (e.g.
    ``robocasa-921c9a5+robosuite-5ce6643``); the backend refuses a scene
    whose sampled instruction differs from the prepared one, so a wrong
    version here fails closed at integration time.

    ``data_source`` overrides the vendored snapshot with a local jsonl path
    (regenerated via ``generate_robocasa_snapshot``); a local file has no Hub
    commit, so its content fingerprint is the provenance.

    ``environment_version`` participates in the ``build_id`` via
    ``build_options``, so reuse under a different version is refused by the
    shared pipeline's own spec check, and a ``refresh`` rebuild gets its own
    id instead of overwriting the first dataset's identity.
    """
    environment_version = environment_version.strip()
    if not environment_version:
        raise ValueError("environment_version must be a non-empty string")

    source_path = _snapshot_path(data_source)
    # Fail fast on a missing/empty snapshot before the build machinery runs.
    _snapshot_rows(source_path)
    request = BuildRequest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        data_source=str(source_path),
        revision=None,
        splits=tuple(splits),
        output_root=output_root or default_root(),
        output_format=OutputFormat(output_format),
        if_exists=ExistingPolicy(if_exists),
    )

    def normalize(raw: Mapping[str, Any], context: NormalizeContext) -> PreparedExample:
        for field in ("id", "instruction", "task_name"):
            if field not in raw:
                raise ValueError(f"robocasa row is missing {field}")
        record_id = str(raw["id"]).strip()
        statement = str(raw["instruction"]).strip()
        if not record_id or not statement:
            raise ValueError("robocasa row needs a non-empty id and instruction")
        task_name = str(raw["task_name"]).strip()
        if not task_name:
            raise ValueError(f"robocasa row {record_id!r} has an empty task_name")
        env_payload: dict[str, Any] = {
            "benchmark": "robocasa",
            "environment_version": environment_version,
            "task_name": task_name,
        }
        for field in ("layout_id", "style_id", "scene_seed"):
            # RoboCasa resolves layout/style ids through its registry; id 0
            # fails there with a cryptic KeyError, so refuse it here.
            minimum = 0 if field == "scene_seed" else 1
            env_payload[field] = _integer_field(raw, field, record_id, minimum=minimum)
        # The row's official step budget must reach the execution side; left
        # in extra it would be ignored for the env-wide constructor default.
        env_payload["max_episode_steps"] = _integer_field(raw, "horizon", record_id, minimum=1)
        # The envelope mirror goes last: a stray raw column of the same name
        # must not override the task identity the payload carries.
        extra = {
            **{key: value for key, value in raw.items() if key not in _MAPPED_FIELDS},
            "benchmark": "robocasa",
            "environment_version": environment_version,
        }
        return PreparedExample(
            task_uid=task_uid(context.source_id, record_id),
            source_id=context.source_id,
            source_record_id=record_id,
            source_order=context.source_order,
            split=context.split,
            statement=statement,
            domain="robotics",
            answer="environment",
            answer_type="none",
            env_payload=env_payload,
            extra=extra,
            grader_id="environment_success",
            source_uri=source_path.as_uri(),
            source_revision="local",
            dataset_license="MIT",
            original_content_owner="RoboCasa contributors",
            original_content_license="MIT",
            raw_digest=raw_digest(raw),
            builder_version=context.builder_version,
        )

    return prepare_dataset(
        request,
        normalize,
        builder_name=BUILDER_NAME,
        builder_version=BUILDER_VERSION,
        # The version is normalizer input, so it rides in build_options: the
        # build id separates two checkouts and core's own reuse check refuses
        # a mismatch (the #256 follow-up; the interim row-reading guard this
        # replaces covered only the reuse path).
        build_options={"environment_version": environment_version},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        help=f"Local jsonl snapshot path. Omitted, the vendored {SNAPSHOT_PATH.name} is used.",
    )
    parser.add_argument(
        "--environment-version",
        required=True,
        help="The robocasa+robosuite checkout that generated the rows, e.g. "
        "robocasa-921c9a5+robosuite-5ce6643; recorded in every private row.",
    )
    parser.add_argument("--output-root", type=Path, default=default_root())
    parser.add_argument("--dataset-name", default="robocasa365_tasks")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--splits", default="train")
    parser.add_argument(
        "--format", choices=[item.value for item in OutputFormat], default="parquet"
    )
    parser.add_argument(
        "--if-exists", choices=[item.value for item in ExistingPolicy], default="reuse"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepared = prepare(
        environment_version=args.environment_version,
        dataset_name=args.dataset_name,
        data_source=args.data_source,
        output_root=args.output_root,
        dataset_version=args.dataset_version,
        splits=tuple(split.strip() for split in args.splits.split(",") if split.strip()),
        output_format=args.format,
        if_exists=args.if_exists,
    )
    print(f"wrote {prepared.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
