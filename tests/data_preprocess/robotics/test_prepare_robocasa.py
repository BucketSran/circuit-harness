"""RoboCasa task-suite preparation: triplet privacy, env_payload, determinism."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphaapollo.data_preprocess import OutputFormat, read_manifest, read_prepared, read_records
from alphaapollo.data_preprocess.robotics.prepare_robocasa import prepare

ENV_VERSION = "robocasa-921c9a5+robosuite-5ce6643"

_ROWS = [
    {
        "id": "coffee-setup-000",
        "task_name": "CoffeeSetup",
        "layout_id": 2,
        "style_id": 5,
        "scene_seed": 42,
        "instruction": "set up the coffee machine on the counter",
        "horizon": 120,
    },
    {
        "id": "coffee-setup-001",
        "task_name": "CoffeeSetup",
        "layout_id": 1,
        "style_id": 3,
        "scene_seed": 43,
        "instruction": "set up the coffee machine on the counter",
        "horizon": 120,
    },
]


def _write_snapshot(path: Path) -> Path:
    path.write_text("".join(f"{json.dumps(row)}\n" for row in _ROWS), encoding="utf-8")
    return path


def _prepare(tmp_path: Path, *, dataset_name: str = "robocasa_test"):
    return prepare(
        environment_version=ENV_VERSION,
        dataset_name=dataset_name,
        data_source=str(_write_snapshot(tmp_path / "snapshot.jsonl")),
        output_root=tmp_path / "prepared",
        output_format=OutputFormat.JSONL,
    )


def test_prepare_separates_public_and_private_rows(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)

    public = read_records(prepared.public_splits["train"])
    private = read_records(prepared.private_splits["train"])
    assert len(public) == len(private) == 2
    assert public[0]["statement"] == "set up the coffee machine on the counter"
    assert public[0]["domain"] == "robotics"
    # The scene triplet and the placeholder answer never reach the public face.
    for key in ("answer", "env_payload", "scene_seed", "layout_id", "style_id"):
        assert key not in public[0]
    assert public[0]["task_uid"] == private[0]["task_uid"]


def test_env_payload_carries_envelope_triplet_and_budget(tmp_path: Path) -> None:
    _prepare(tmp_path)

    example = read_prepared("robocasa_test", "v1", tmp_path / "prepared", split="train")[0]
    assert example.grader_id == "environment_success"
    assert example.env_payload == {
        "benchmark": "robocasa",
        "environment_version": ENV_VERSION,
        "task_name": "CoffeeSetup",
        "layout_id": 2,
        "style_id": 5,
        "scene_seed": 42,
        "max_episode_steps": 120,
    }
    # The envelope is mirrored into extra; mapped fields (incl. horizon) do
    # not duplicate there — the payload is the single owner.
    assert example.extra == {
        "benchmark": "robocasa",
        "environment_version": ENV_VERSION,
    }


def test_build_is_deterministic(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snapshot.jsonl")
    first = prepare(
        environment_version=ENV_VERSION,
        dataset_name="robocasa_det",
        data_source=str(snapshot),
        output_root=tmp_path / "a",
        output_format=OutputFormat.JSONL,
    )
    second = prepare(
        environment_version=ENV_VERSION,
        dataset_name="robocasa_det",
        data_source=str(snapshot),
        output_root=tmp_path / "b",
        output_format=OutputFormat.JSONL,
    )
    for split in first.public_splits:
        assert first.public_splits[split].read_bytes() == second.public_splits[split].read_bytes()
        assert first.private_splits[split].read_bytes() == second.private_splits[split].read_bytes()


def test_malformed_rows_fail_with_actionable_errors(tmp_path: Path) -> None:
    complete = {"layout_id": 1, "style_id": 1, "scene_seed": 1, "horizon": 100}
    bad_rows = [
        {"task_name": "T", **complete},  # no id, no instruction -> ValueError
        {"id": "x-000", "instruction": "do it", **complete},  # no task_name
        {"id": "x-001", "task_name": "T", **complete},  # no instruction
        # empty task_name slips past nothing else
        {"id": "x-002", "task_name": "  ", "instruction": "do it", **complete},
        # layout id 0 breaks robocasa's registry with a cryptic KeyError
        {
            "id": "x-003",
            "task_name": "T",
            "instruction": "do it",
            "layout_id": 0,
            "style_id": 1,
            "scene_seed": 1,
            "horizon": 100,
        },
        # fractional ids truncate silently if not refused
        {
            "id": "x-004",
            "task_name": "T",
            "instruction": "do it",
            "layout_id": 1.5,
            "style_id": 1,
            "scene_seed": 1,
            "horizon": 100,
        },
        # booleans pass int() as 1; refuse them like prepare_aime does
        {
            "id": "x-005",
            "task_name": "T",
            "instruction": "do it",
            "layout_id": True,
            "style_id": 1,
            "scene_seed": 1,
            "horizon": 100,
        },
        # without the budget the termination contract falls back to the
        # env-wide default; a row that cannot set it fails the build
        {
            "id": "x-006",
            "task_name": "T",
            "instruction": "do it",
            "layout_id": 1,
            "style_id": 1,
            "scene_seed": 1,
        },
        # and a fake budget (zero, boolean, fractional) must not reach the
        # execution payload either
        {
            "id": "x-007",
            "task_name": "T",
            "instruction": "do it",
            "layout_id": 1,
            "style_id": 1,
            "scene_seed": 1,
            "horizon": 0,
        },
        {
            "id": "x-008",
            "task_name": "T",
            "instruction": "do it",
            "layout_id": 1,
            "style_id": 1,
            "scene_seed": 1,
            "horizon": True,
        },
        {
            "id": "x-009",
            "task_name": "T",
            "instruction": "do it",
            "layout_id": 1,
            "style_id": 1,
            "scene_seed": 1,
            "horizon": 1.5,
        },
    ]
    for row in bad_rows:
        source = tmp_path / "bad.jsonl"
        source.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            prepare(
                environment_version=ENV_VERSION,
                dataset_name="robocasa_bad",
                data_source=str(source),
                output_root=tmp_path / "prepared",
                output_format=OutputFormat.JSONL,
            )


def test_empty_snapshot_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    source.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        prepare(
            environment_version=ENV_VERSION,
            dataset_name="robocasa_empty",
            data_source=str(source),
            output_root=tmp_path / "prepared",
        )


def test_environment_version_is_required() -> None:
    with pytest.raises(TypeError, match="environment_version"):
        prepare(output_root=Path("/tmp/never"))


def test_blank_environment_version_is_refused(tmp_path: Path) -> None:
    source = _write_snapshot(tmp_path / "snapshot.jsonl")
    with pytest.raises(ValueError, match="non-empty"):
        prepare(
            environment_version="   ",
            data_source=str(source),
            output_root=tmp_path / "prepared",
        )


def test_reuse_refuses_a_build_from_another_environment_version(tmp_path: Path) -> None:
    """Two checkouts must not silently share one build directory.

    ``environment_version`` rides in ``build_options`` (#256 follow-up), so
    the shared ``build_id`` separates the two datasets and core's own reuse
    check refuses the mismatch. A ``refresh`` rebuild under the new version
    gets its own id instead of overwriting the first dataset's identity —
    the residue the interim row-reading guard could not cover.
    """

    snapshot = str(_write_snapshot(tmp_path / "snapshot.jsonl"))
    first = prepare(
        environment_version=ENV_VERSION,
        dataset_name="robocasa_reuse",
        data_source=snapshot,
        output_root=tmp_path / "prepared",
        output_format=OutputFormat.JSONL,
    )

    with pytest.raises(ValueError, match="different spec"):
        prepare(
            environment_version="robocasa-deadbee+robosuite-f00d123",
            dataset_name="robocasa_reuse",
            data_source=snapshot,
            output_root=tmp_path / "prepared",
            output_format=OutputFormat.JSONL,
        )

    # The stored rows are untouched, and the same version still reuses.
    again = prepare(
        environment_version=ENV_VERSION,
        dataset_name="robocasa_reuse",
        data_source=snapshot,
        output_root=tmp_path / "prepared",
        output_format=OutputFormat.JSONL,
    )
    assert again.build_id == first.build_id
    stored = read_prepared("robocasa_reuse", "v1", tmp_path / "prepared", split="train")
    assert {example.env_payload["environment_version"] for example in stored} == {ENV_VERSION}

    # Refresh under the other version rebuilds under its own build id, so
    # downstream artifacts can tell the two datasets apart.
    refreshed = prepare(
        environment_version="robocasa-deadbee+robosuite-f00d123",
        dataset_name="robocasa_reuse",
        data_source=snapshot,
        output_root=tmp_path / "prepared",
        output_format=OutputFormat.JSONL,
        if_exists="refresh",
    )
    assert refreshed.build_id != first.build_id
    rebuilt = read_prepared("robocasa_reuse", "v1", tmp_path / "prepared", split="train")
    assert {example.env_payload["environment_version"] for example in rebuilt} == {
        "robocasa-deadbee+robosuite-f00d123"
    }


def test_refresh_rewrites_the_rows_for_the_new_environment_version(tmp_path: Path) -> None:
    """``refresh`` is the caller stating the intent the guard refuses to guess."""

    snapshot = str(_write_snapshot(tmp_path / "snapshot.jsonl"))
    prepare(
        environment_version=ENV_VERSION,
        dataset_name="robocasa_refresh",
        data_source=snapshot,
        output_root=tmp_path / "prepared",
        output_format=OutputFormat.JSONL,
    )
    prepare(
        environment_version="robocasa-deadbee+robosuite-f00d123",
        dataset_name="robocasa_refresh",
        data_source=snapshot,
        output_root=tmp_path / "prepared",
        output_format=OutputFormat.JSONL,
        if_exists="refresh",
    )

    stored = read_prepared("robocasa_refresh", "v1", tmp_path / "prepared", split="train")
    assert {example.env_payload["environment_version"] for example in stored} == {
        "robocasa-deadbee+robosuite-f00d123"
    }


def test_vendored_snapshot_builds(tmp_path: Path) -> None:
    prepared = prepare(
        environment_version=ENV_VERSION,
        output_root=tmp_path / "prepared",
        output_format=OutputFormat.JSONL,
    )

    manifest = read_manifest(prepared.manifest_path)
    assert manifest.splits["train"].rows == 370
    examples = read_prepared("robocasa365_tasks", "v1", tmp_path / "prepared", split="train")
    assert all(example.grader_id == "environment_success" for example in examples)
    assert all(
        set(example.env_payload)
        == {
            "benchmark",
            "environment_version",
            "task_name",
            "layout_id",
            "style_id",
            "scene_seed",
            "max_episode_steps",
        }
        for example in examples
    )
    assert all(example.env_payload["benchmark"] == "robocasa" for example in examples)
    assert all(example.env_payload["environment_version"] == ENV_VERSION for example in examples)
    # The official registry budget reaches the execution payload verbatim;
    # this is the regression gate for the round-4 review finding (#256).
    by_task = {example.env_payload["task_name"]: example for example in examples}
    assert by_task["AddIceCubes"].env_payload["max_episode_steps"] == 1650
