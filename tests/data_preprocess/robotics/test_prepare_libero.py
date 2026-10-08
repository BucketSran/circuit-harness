"""Contract tests for the benchmark-specific LIBERO preparer."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from alphaapollo.data_preprocess import read_prepared, read_records
from alphaapollo.data_preprocess.robotics.prepare_libero import (
    LIBERO_SUITES,
    _benchmark_api,
    main,
    prepare,
    prepare_from_checkout,
)
from alphaapollo.workflows.config import DatasetConfig
from alphaapollo.workflows.data import load_prepared_inputs

_ENVIRONMENT_VERSION = "libero-fixture-revision"


def _row(*, suite: str = "libero_spatial", **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": "open_drawer",
        "language": "open the top drawer",
        "suite": suite,
        "task_order_index": 0,
        "task_index": 3,
        "task_name": "open_drawer",
        "problem_folder": suite,
        "bddl_file": "open_drawer.bddl",
        "init_states_file": "open_drawer.pruned_init",
        "demonstration_id": "open_drawer_demo",
        "demonstration_path": f"{suite}/open_drawer_demo.hdf5",
        "initial_state_index": 0,
        "source_uri": "https://github.com/Lifelong-Robot-Learning/LIBERO",
        "source_revision": "metadata-fixture-revision",
        "scene_category": "kitchen",
    }
    row.update(updates)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize("suite", LIBERO_SUITES)
def test_all_official_libero_suites_use_the_shared_prepared_layout(
    tmp_path: Path,
    suite: str,
) -> None:
    source = tmp_path / f"{suite}.jsonl"
    _write_jsonl(source, [_row(suite=suite)])

    prepared = prepare(
        data_source=str(source),
        suite=suite,
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    expected_split = "train" if suite == "libero_90" else "test"
    assert set(prepared.public_splits) == {expected_split}
    assert prepared.root == tmp_path / "prepared" / suite / "v1"
    example = read_prepared(suite, "v1", tmp_path / "prepared")[0]
    assert example.statement == "open the top drawer"
    assert example.domain == "robotics"
    assert example.group_id == suite
    assert example.answer == "environment"
    assert example.answer_type == "none"
    assert example.grader_id == "environment_success"
    assert example.builder_version == "7"
    assert example.env_payload == {
        "benchmark": "libero",
        "environment_version": _ENVIRONMENT_VERSION,
        "suite": suite,
        "task_order_index": 0,
        "task_index": 3,
        "task_name": "open_drawer",
        "problem_folder": suite,
        "bddl_file": "open_drawer.bddl",
        "init_states_file": "open_drawer.pruned_init",
        "demonstration_id": "open_drawer_demo",
        "demonstration_path": f"{suite}/open_drawer_demo.hdf5",
        "initial_state_index": 0,
    }
    assert example.extra["source_metadata"] == {
        "scene_category": "kitchen",
        "source_revision": "metadata-fixture-revision",
        "source_uri": "https://github.com/Lifelong-Robot-Learning/LIBERO",
    }
    assert "benchmark" not in example.extra
    assert "environment_version" not in example.extra
    assert "task_name" not in example.extra
    assert example.source_uri == source.resolve().as_uri()
    assert example.source_revision == "local"
    assert example.dataset_license == "unverified"
    assert example.original_content_owner == "unverified"
    assert example.original_content_license == "unverified"


def test_libero_environment_version_is_not_inferred_from_source_revision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "libero.jsonl"
    row = _row()
    row.pop("source_revision")
    _write_jsonl(source, [row])

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_spatial", "v1", tmp_path / "prepared")[0]
    assert example.source_revision == "local"
    assert example.env_payload["environment_version"] == _ENVIRONMENT_VERSION


def test_libero_identity_is_stable_and_can_be_derived_from_suite_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "libero.jsonl"
    row = _row()
    row.pop("task_id")
    _write_jsonl(source, [row])

    first = prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "first",
        output_format="jsonl",
    )
    second = prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "second",
        output_format="jsonl",
    )

    first_row = read_prepared("libero_spatial", "v1", tmp_path / "first")[0]
    second_row = read_prepared("libero_spatial", "v1", tmp_path / "second")[0]
    assert first.build_id == second.build_id
    assert first_row.source_record_id == "libero_spatial:0:3:open_drawer:demo-id-open_drawer_demo:0"
    assert first_row.task_uid == second_row.task_uid
    assert first_row.raw_digest == second_row.raw_digest


@pytest.mark.parametrize(
    ("removed", "message"),
    [
        ("language", "task instruction"),
        ("task_name", "task name"),
        ("task_index", "task_index"),
        ("bddl_file", "BDDL file"),
        ("init_states_file", "initial-states file"),
    ],
)
def test_libero_malformed_rows_fail_with_the_missing_field(
    tmp_path: Path,
    removed: str,
    message: str,
) -> None:
    source = tmp_path / f"missing-{removed}.jsonl"
    row = _row()
    row.pop(removed)
    _write_jsonl(source, [row])

    with pytest.raises(ValueError, match=message):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )

    assert not (tmp_path / "prepared" / "libero_spatial" / "v1").exists()


def test_libero_rejects_a_row_from_another_suite(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(suite="libero_object")])

    with pytest.raises(ValueError, match="does not match configured suite"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_rejects_conflicting_instruction_aliases(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(instruction="open the bottom drawer")])

    with pytest.raises(ValueError, match="instruction is ambiguous"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_accepts_the_official_problem_metadata_field(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    row = _row(problem="Libero", name="open_drawer")
    row.pop("task_id")
    row.pop("task_name")
    row.pop("demonstration_id")
    _write_jsonl(source, [row])

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_spatial", "v1", tmp_path / "prepared")[0]
    assert example.statement == "open the top drawer"
    assert example.extra["source_metadata"]["problem"] == "Libero"
    assert "demonstration_id" not in example.env_payload


def test_libero_accepts_a_legacy_problem_instruction_without_primary_alias(
    tmp_path: Path,
) -> None:
    source = tmp_path / "libero.jsonl"
    row = _row(problem="open the top drawer")
    row.pop("language")
    _write_jsonl(source, [row])

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_spatial", "v1", tmp_path / "prepared")[0]
    assert example.statement == "open the top drawer"
    assert example.extra["source_instruction_key"] == "problem"
    assert "problem" not in example.extra["source_metadata"]


def test_libero_does_not_use_the_official_problem_label_as_instruction(
    tmp_path: Path,
) -> None:
    source = tmp_path / "libero.jsonl"
    row = _row(problem="Libero")
    row.pop("language")
    _write_jsonl(source, [row])

    with pytest.raises(ValueError, match="missing task instruction"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_rejects_a_nonofficial_problem_conflicting_with_language(
    tmp_path: Path,
) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(problem="open the bottom drawer")])

    with pytest.raises(ValueError, match="instruction is ambiguous"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_accepts_matching_instruction_aliases(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(instruction="open the top drawer")])

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_spatial", "v1", tmp_path / "prepared")[0]
    assert example.statement == "open the top drawer"
    assert example.extra["source_instruction_key"] == "language"


@pytest.mark.parametrize(
    "updates",
    [
        {"benchmark_suite": "libero_object"},
        {"name": "close_drawer"},
        {"bddl_file_name": "other.bddl"},
        {"initial_states_file": "other.pruned_init"},
        {"demo_path": "libero_spatial/other_demo.hdf5"},
        {"demo_id": "other_demo"},
        {"id": "other_task"},
        {"id": "other_task", "source_record_id": "row-1"},
    ],
)
def test_libero_rejects_conflicting_non_instruction_aliases(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(**updates)])

    with pytest.raises(ValueError, match="ambiguous"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_accepts_matching_non_instruction_aliases(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(
        source,
        [
            _row(
                benchmark_suite="libero_spatial",
                name="open_drawer",
                bddl_file_name="open_drawer.bddl",
                initial_states_file="open_drawer.pruned_init",
                demo_path="libero_spatial/open_drawer_demo.hdf5",
                demo_id="open_drawer_demo",
                id="open_drawer",
            )
        ],
    )

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_spatial", "v1", tmp_path / "prepared")[0]
    assert example.env_payload["task_name"] == "open_drawer"
    assert example.source_record_id == "open_drawer:demo-id-open_drawer_demo:0"
    assert example.extra["source_metadata"] == {
        "scene_category": "kitchen",
        "source_revision": "metadata-fixture-revision",
        "source_uri": "https://github.com/Lifelong-Robot-Learning/LIBERO",
    }


def test_libero_identity_escapes_delimiters_in_task_and_demo_ids(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(
        source,
        [
            _row(task_id="a", demonstration_id="b:c"),
            _row(task_id="a:b", demonstration_id="c"),
        ],
    )

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    examples = read_prepared("libero_spatial", "v1", tmp_path / "prepared")
    assert [example.source_record_id for example in examples] == [
        "a:demo-id-b%3Ac:0",
        "a%3Ab:demo-id-c:0",
    ]


def test_libero_identity_separates_explicit_demo_ids_from_path_tokens(
    tmp_path: Path,
) -> None:
    source = tmp_path / "libero.jsonl"
    private_path = "/mnt/private/alice/run-17/demo.hdf5"
    path_token = "demo-path-" + hashlib.sha256(private_path.encode("utf-8")).hexdigest()
    first = _row(demonstration_path=private_path)
    first.pop("demonstration_id")
    second = _row(demonstration_id=path_token)
    _write_jsonl(source, [first, second])

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    examples = read_prepared("libero_spatial", "v1", tmp_path / "prepared")
    assert len({example.source_record_id for example in examples}) == 2
    assert examples[0].source_record_id.startswith("open_drawer:demo-path-")
    assert examples[1].source_record_id == f"open_drawer:demo-id-{path_token}:0"


def test_libero_public_output_is_a_generic_workflow_input(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row()])
    prepared = prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )
    public_path = prepared.public_splits["test"]
    public = read_records(public_path)[0]
    assert "env_payload" not in public
    assert "answer" not in public

    config = SimpleNamespace(
        dataset=DatasetConfig(
            path=public_path,
            format="jsonl",
            input_key="statement",
            id_key="task_uid",
            metadata_keys=("source_record_id", "group_id", "domain"),
        )
    )
    workflow_input = load_prepared_inputs(config)[0]
    assert workflow_input.problem == "open the top drawer"
    assert workflow_input.metadata == {
        "source_record_id": "open_drawer:demo-id-open_drawer_demo:0",
        "group_id": "libero_spatial",
        "domain": "robotics",
    }


def test_libero_requires_a_nonempty_build_environment_version(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row()])

    with pytest.raises(ValueError, match="environment_version"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version="  ",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )

    with pytest.raises(SystemExit):
        main(["--data-source", str(source), "--suite", "libero_spatial"])


def test_libero_rejects_a_conflicting_row_environment_version(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(environment_version="another-libero-checkout")])

    with pytest.raises(ValueError, match="does not match configured environment_version"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_accepts_a_matching_row_environment_version(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(environment_version=_ENVIRONMENT_VERSION)])

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_spatial", "v1", tmp_path / "prepared")[0]
    assert example.env_payload["environment_version"] == _ENVIRONMENT_VERSION
    assert "environment_version" not in example.extra["source_metadata"]


def test_libero_carries_a_per_task_episode_horizon(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(max_episode_steps=123)])

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_spatial", "v1", tmp_path / "prepared")[0]
    assert example.env_payload["max_episode_steps"] == 123
    assert "max_episode_steps" not in example.extra["source_metadata"]


@pytest.mark.parametrize("value", [0, -1, True, ""])
def test_libero_rejects_an_invalid_per_task_episode_horizon(
    tmp_path: Path,
    value: object,
) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(max_episode_steps=value)])

    with pytest.raises(ValueError, match="max_episode_steps"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_preserves_row_provenance_without_overriding_build_claims(
    tmp_path: Path,
) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(
        source,
        [
            _row(
                source_revision="deadbeef",
                dataset_license="row-license",
                original_content_owner="row-owner",
                original_content_license="row-content-license",
            )
        ],
    )

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        dataset_license="build-license",
        original_content_owner="build-owner",
        original_content_license="build-content-license",
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_spatial", "v1", tmp_path / "prepared")[0]
    assert example.source_revision == "local"
    assert example.dataset_license == "build-license"
    assert example.original_content_owner == "build-owner"
    assert example.original_content_license == "build-content-license"
    assert example.extra["source_metadata"] == {
        "dataset_license": "row-license",
        "original_content_license": "row-content-license",
        "original_content_owner": "row-owner",
        "scene_category": "kitchen",
        "source_revision": "deadbeef",
        "source_uri": "https://github.com/Lifelong-Robot-Learning/LIBERO",
    }
    assert example.source_uri == source.resolve().as_uri()


def test_libero_derived_identity_distinguishes_demonstrations_and_initial_states(
    tmp_path: Path,
) -> None:
    source = tmp_path / "libero.jsonl"
    first = _row()
    first.pop("task_id")
    second = _row(
        demonstration_id="open_drawer_demo_2",
        demonstration_path="libero_spatial/open_drawer_demo_2.hdf5",
        initial_state_index=1,
    )
    second.pop("task_id")
    _write_jsonl(source, [first, second])

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    examples = read_prepared("libero_spatial", "v1", tmp_path / "prepared")
    assert len({example.task_uid for example in examples}) == 2
    assert [example.source_record_id for example in examples] == [
        "libero_spatial:0:3:open_drawer:demo-id-open_drawer_demo:0",
        "libero_spatial:0:3:open_drawer:demo-id-open_drawer_demo_2:1",
    ]


def test_libero_explicit_task_id_distinguishes_initial_states(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(initial_state_index=0), _row(initial_state_index=1)])

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    examples = read_prepared("libero_spatial", "v1", tmp_path / "prepared")
    assert [example.source_record_id for example in examples] == [
        "open_drawer:demo-id-open_drawer_demo:0",
        "open_drawer:demo-id-open_drawer_demo:1",
    ]
    assert len({example.task_uid for example in examples}) == 2


def test_libero_explicit_source_record_id_must_be_row_unique(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(
        source,
        [
            _row(source_record_id="open_drawer_row", initial_state_index=0),
            _row(source_record_id="open_drawer_row", initial_state_index=1),
        ],
    )

    with pytest.raises(ValueError, match="duplicate task_uid"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_build_options_prevent_stale_reuse(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    row = _row()
    row.pop("suite")
    _write_jsonl(source, [row])
    output_root = tmp_path / "prepared"

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        dataset_name="shared-name",
        output_root=output_root,
        output_format="jsonl",
    )

    with pytest.raises(ValueError, match="different spec"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version="another-libero-checkout",
            dataset_name="shared-name",
            output_root=output_root,
            output_format="jsonl",
        )
    with pytest.raises(ValueError, match="different spec"):
        prepare(
            data_source=str(source),
            suite="libero_object",
            environment_version=_ENVIRONMENT_VERSION,
            dataset_name="shared-name",
            output_root=output_root,
            output_format="jsonl",
        )

    assert read_prepared("shared-name", "v1", output_root)[0].group_id == "libero_spatial"


def test_libero_accepts_an_evaluation_row_without_demonstrations(tmp_path: Path) -> None:
    """Demonstrations are provenance, not an execution requirement.

    An evaluation-only source (task x initial state) has no demonstration to
    name; the backend treats both demonstration fields as optional strings.
    """

    source = tmp_path / "libero.jsonl"
    row = _row()
    row.pop("demonstration_path")
    row.pop("demonstration_id")
    _write_jsonl(source, [row])

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_spatial", "v1", tmp_path / "prepared")[0]
    assert "demonstration_id" not in example.env_payload
    assert "demonstration_path" not in example.env_payload
    assert example.env_payload["task_name"] == "open_drawer"


@pytest.mark.parametrize("explicit_task_id", [False, True])
def test_libero_keeps_a_path_only_demonstration_private(
    tmp_path: Path,
    explicit_task_id: bool,
) -> None:
    source = tmp_path / "libero.jsonl"
    private_path = "/mnt/private/alice/run-17/demo.hdf5"
    row = _row(demonstration_path=private_path)
    if not explicit_task_id:
        row.pop("task_id")
    row.pop("demonstration_id")
    _write_jsonl(source, [row])

    prepared = prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_spatial", "v1", tmp_path / "prepared")[0]
    assert example.env_payload["demonstration_path"] == private_path
    assert "demonstration_id" not in example.env_payload
    assert private_path not in example.source_record_id
    assert "demo-path-" in example.source_record_id
    assert private_path not in json.dumps(read_records(prepared.public_splits["test"]))


def _fake_libero_checkout(monkeypatch: pytest.MonkeyPatch, *, language: str) -> None:
    """Install a stub official checkout exposing every supported suite."""

    benchmarks: dict[str, type] = {}
    for suite in LIBERO_SUITES:
        task = type(
            "_Task",
            (),
            {
                "name": "open_drawer",
                "problem": "Libero",
                "problem_folder": suite,
                "bddl_file": "open_drawer.bddl",
                "init_states_file": "open_drawer.pruned_init",
                "language": language,
            },
        )

        class _Suite:
            def __init__(self, task_order_index: int) -> None:
                if task_order_index != 0:
                    raise AssertionError(f"unsupported task order {task_order_index}")

            def get_task(self, index: int, _task=task):
                if index != 3:
                    raise IndexError(index)
                return _task()

            def get_num_tasks(self) -> int:
                return 1

            def get_task_init_states(self, index: int) -> list[list[float]]:
                return [[0.0], [1.0]]

            def get_task_demonstration(self, index: int, _suite=suite) -> str:
                return f"{_suite}/open_drawer_demo.hdf5"

        benchmarks[suite] = _Suite

    libero_pkg = ModuleType("libero")
    libero_inner = ModuleType("libero.libero")
    benchmark = ModuleType("libero.libero.benchmark")
    benchmark.get_benchmark_dict = lambda: benchmarks  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_inner)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark)


@pytest.fixture(autouse=True)
def _official_libero_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_libero_checkout(monkeypatch, language="open the top drawer")


def _write_fake_libero_checkout(root: Path) -> Path:
    benchmark = root / "libero" / "libero" / "benchmark"
    benchmark.mkdir(parents=True)
    (root / "libero" / "__init__.py").write_text("", encoding="utf-8")
    (root / "libero" / "libero" / "__init__.py").write_text("", encoding="utf-8")
    (benchmark / "__init__.py").write_text(
        """
class _Task:
    name = "open_drawer"
    language = "open the top drawer"
    problem = "Libero"
    problem_folder = "libero_spatial"
    bddl_file = "open_drawer.bddl"
    init_states_file = "open_drawer.pruned_init"


class _Suite:
    def __init__(self, task_order_index):
        if task_order_index != 0:
            raise ValueError(task_order_index)

    def get_num_tasks(self):
        return 1

    def get_task(self, index):
        if index != 0:
            raise IndexError(index)
        return _Task()

    def get_task_init_states(self, index):
        return [[0.0], [1.0]]

    def get_task_demonstration(self, index):
        return "libero_spatial/open_drawer_demo.hdf5"


def get_benchmark_dict():
    return {"libero_spatial": _Suite}
""".lstrip(),
        encoding="utf-8",
    )
    return root


def test_libero_checkout_validation_accepts_a_matching_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_libero_checkout(monkeypatch, language="open the top drawer")
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row()])

    prepared = prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    assert (prepared.root / "private" / "test.jsonl").exists()


def test_libero_checkout_validation_rejects_instruction_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #256 failure class fails at preparation, not at the first episode.

    A metadata source exported from a different checkout samples different
    wording; the backend would refuse 100% of resets, so the drift must
    surface while the dataset is being built.
    """

    _fake_libero_checkout(monkeypatch, language="open the bottom drawer")
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row()])

    with pytest.raises(ValueError, match="does not match the checkout task language"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_checkout_validation_rejects_an_out_of_range_initial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_libero_checkout(monkeypatch, language="open the top drawer")
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(initial_state_index=9)])

    with pytest.raises(ValueError, match="out of range"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_checkout_validation_contextualizes_an_invalid_task_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_libero_checkout(monkeypatch, language="open the top drawer")
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(task_order_index=1)])

    with pytest.raises(ValueError, match="task_order_index 1 is not available"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_requires_the_official_checkout_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in ("libero", "libero.libero", "libero.libero.benchmark"):
        monkeypatch.setitem(sys.modules, module_name, None)
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row()])

    with pytest.raises(RuntimeError, match="requires the optional 'libero' dependency"):
        prepare(
            data_source=str(source),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_canonical_path_generates_rows_from_the_official_api(
    tmp_path: Path,
) -> None:
    checkout = _write_fake_libero_checkout(tmp_path / "checkout")

    prepared = prepare_from_checkout(
        checkout_path=checkout,
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        source_revision="official-fixture-revision",
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    examples = read_prepared("libero_spatial", "v1", tmp_path / "prepared")
    assert len(examples) == 2
    assert [example.env_payload["initial_state_index"] for example in examples] == [0, 1]
    assert all(example.statement == "open the top drawer" for example in examples)
    assert all(example.env_payload["task_name"] == "open_drawer" for example in examples)
    assert all(
        example.env_payload["demonstration_path"] == "libero_spatial/open_drawer_demo.hdf5"
        for example in examples
    )
    assert all(example.source_revision == "official-fixture-revision" for example in examples)
    assert all(
        example.source_uri == "https://github.com/Lifelong-Robot-Learning/LIBERO"
        for example in examples
    )
    assert all(example.extra["source_metadata"]["problem"] == "Libero" for example in examples)

    manifest = json.loads((prepared.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_source"].startswith("libero://checkout/")
    assert manifest["source_revision"] == "official-fixture-revision"
    assert not list(checkout.rglob("__pycache__"))


def test_libero_canonical_path_ignores_stale_checkout_bytecode(tmp_path: Path) -> None:
    checkout = _write_fake_libero_checkout(tmp_path / "checkout")
    benchmark = checkout / "libero" / "libero" / "benchmark" / "__init__.py"
    original_stat = benchmark.stat()
    py_compile.compile(str(benchmark), doraise=True)
    benchmark.write_text(
        benchmark.read_text(encoding="utf-8").replace(
            "open the top drawer",
            "open the bot drawer",
        ),
        encoding="utf-8",
    )
    os.utime(
        benchmark,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert list(checkout.rglob("*.pyc"))

    prepare_from_checkout(
        checkout_path=checkout,
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        source_revision="official-fixture-revision",
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    examples = read_prepared("libero_spatial", "v1", tmp_path / "prepared")
    assert all(example.statement == "open the bot drawer" for example in examples)


def test_libero_checkout_imports_are_serialized_and_restore_process_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_checkout = _write_fake_libero_checkout(tmp_path / "first")
    second_checkout = _write_fake_libero_checkout(tmp_path / "second")
    second_benchmark = second_checkout / "libero" / "libero" / "benchmark" / "__init__.py"
    second_benchmark.write_text(
        second_benchmark.read_text(encoding="utf-8").replace(
            "open the top drawer",
            "open the bot drawer",
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("LIBERO_CONFIG_PATH", "alphaapollo-original-config")
    original_path = tuple(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    original_pycache_prefix = sys.pycache_prefix
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "libero" or name.startswith("libero.")
    }
    second_attempting = threading.Event()
    second_entered = threading.Event()
    results: dict[str, str] = {}
    errors: list[BaseException] = []

    def load_second_checkout() -> None:
        second_attempting.set()
        try:
            with _benchmark_api(second_checkout) as get_benchmark_dict:
                benchmark = get_benchmark_dict()["libero_spatial"](0)
                results["second"] = benchmark.get_task(0).language
                second_entered.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    with _benchmark_api(first_checkout) as get_benchmark_dict:
        benchmark = get_benchmark_dict()["libero_spatial"](0)
        results["first"] = benchmark.get_task(0).language
        thread = threading.Thread(target=load_second_checkout)
        thread.start()
        assert second_attempting.wait(timeout=5)
        assert not second_entered.wait(timeout=0.25)

    assert second_entered.wait(timeout=5)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert results == {
        "first": "open the top drawer",
        "second": "open the bot drawer",
    }
    assert tuple(sys.path) == original_path
    assert sys.dont_write_bytecode is original_dont_write_bytecode
    assert sys.pycache_prefix == original_pycache_prefix
    assert os.environ["LIBERO_CONFIG_PATH"] == "alphaapollo-original-config"
    assert {
        name: module
        for name, module in sys.modules.items()
        if name == "libero" or name.startswith("libero.")
    } == original_modules


def test_libero_canonical_rows_prevent_stale_reuse(tmp_path: Path) -> None:
    checkout = _write_fake_libero_checkout(tmp_path / "checkout")
    output_root = tmp_path / "prepared"

    prepare_from_checkout(
        checkout_path=checkout,
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        source_revision="same-declared-revision",
        output_root=output_root,
        output_format="jsonl",
    )
    benchmark = checkout / "libero" / "libero" / "benchmark" / "__init__.py"
    benchmark.write_text(
        benchmark.read_text(encoding="utf-8").replace(
            "open the top drawer",
            "open the bottom drawer",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different spec"):
        prepare_from_checkout(
            checkout_path=checkout,
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            source_revision="same-declared-revision",
            output_root=output_root,
            output_format="jsonl",
        )

    assert read_prepared("libero_spatial", "v1", output_root)[0].statement == (
        "open the top drawer"
    )


def test_libero_canonical_path_requires_source_revision_without_git_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_libero_checkout(monkeypatch, language="open the top drawer")

    with pytest.raises(ValueError, match="source_revision"):
        prepare_from_checkout(
            checkout_path=None,
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
        )


def test_libero_cli_can_use_the_canonical_checkout_path(tmp_path: Path) -> None:
    checkout = _write_fake_libero_checkout(tmp_path / "checkout")

    assert (
        main(
            [
                "--checkout-path",
                str(checkout),
                "--suite",
                "libero_spatial",
                "--environment-version",
                _ENVIRONMENT_VERSION,
                "--source-revision",
                "official-fixture-revision",
                "--format",
                "jsonl",
                "--output-root",
                str(tmp_path / "prepared"),
            ]
        )
        == 0
    )
    assert (tmp_path / "prepared" / "libero_spatial" / "v1" / "manifest.json").exists()


def test_libero_cli_rejects_canonical_options_in_metadata_mode(tmp_path: Path) -> None:
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row()])

    with pytest.raises(SystemExit):
        main(
            [
                "--data-source",
                str(source),
                "--suite",
                "libero_spatial",
                "--environment-version",
                _ENVIRONMENT_VERSION,
                "--task-order-index",
                "0",
            ]
        )


def _uninstall_libero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the autouse stub so the real optional-dependency surface is exercised."""

    for module_name in ("libero", "libero.libero", "libero.libero.benchmark"):
        monkeypatch.setitem(sys.modules, module_name, None)


def test_libero_metadata_mode_answers_if_exists_without_the_optional_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`reuse` and `error` are answerable from the destination, so LIBERO is not needed.

    `libero` is an optional dependency with no packaged form that carries the
    benchmark's BDDL and initial-state files, so importing it before the shared
    short-circuits would make every documented `--data-source` call require a
    configured checkout.
    """

    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row()])
    output_root = tmp_path / "prepared"
    arguments: dict[str, object] = {
        "data_source": str(source),
        "suite": "libero_spatial",
        "environment_version": _ENVIRONMENT_VERSION,
        "output_root": output_root,
        "output_format": "jsonl",
    }
    built = prepare(**arguments)  # type: ignore[arg-type]

    _uninstall_libero(monkeypatch)

    reused = prepare(**arguments)  # type: ignore[arg-type]
    assert reused.build_id == built.build_id

    with pytest.raises(FileExistsError):
        prepare(if_exists="error", **arguments)  # type: ignore[arg-type]

    # A build that does normalize a row still needs the checkout, and says so.
    with pytest.raises(RuntimeError, match="checkout_path"):
        prepare(dataset_version="v2", **arguments)  # type: ignore[arg-type]


def test_libero_metadata_rows_can_be_validated_against_a_named_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`checkout_path` is the remedy the missing-dependency error names."""

    checkout = _write_fake_libero_checkout(tmp_path / "checkout")
    _uninstall_libero(monkeypatch)
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(task_index=0)])

    prepare(
        data_source=str(source),
        suite="libero_spatial",
        environment_version=_ENVIRONMENT_VERSION,
        checkout_path=checkout,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_spatial", "v1", tmp_path / "prepared")[0]
    assert example.statement == "open the top drawer"

    drifted = tmp_path / "drifted.jsonl"
    _write_jsonl(drifted, [_row(task_index=0, language="open the bottom drawer")])
    with pytest.raises(ValueError, match="does not match the checkout task language"):
        prepare(
            data_source=str(drifted),
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            checkout_path=checkout,
            dataset_name="drifted",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_libero_cli_validates_metadata_rows_against_a_named_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = _write_fake_libero_checkout(tmp_path / "checkout")
    _uninstall_libero(monkeypatch)
    source = tmp_path / "libero.jsonl"
    _write_jsonl(source, [_row(task_index=0)])

    assert (
        main(
            [
                "--data-source",
                str(source),
                "--checkout-path",
                str(checkout),
                "--suite",
                "libero_spatial",
                "--environment-version",
                _ENVIRONMENT_VERSION,
                "--format",
                "jsonl",
                "--output-root",
                str(tmp_path / "prepared"),
            ]
        )
        == 0
    )


def test_libero_cli_requires_a_source(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--suite", "libero_spatial", "--environment-version", _ENVIRONMENT_VERSION])


def test_libero_canonical_identity_ignores_demonstration_availability(tmp_path: Path) -> None:
    """LIBERO ships demonstration HDF5s separately, so their absence is normal.

    Folding the demonstration into `task_uid` would give the identical task and
    initial state different ids on a machine that has not downloaded them, and
    prepared builds would stop joining.
    """

    with_demonstrations = _write_fake_libero_checkout(tmp_path / "with")
    without_demonstrations = _write_fake_libero_checkout(tmp_path / "without")
    benchmark = without_demonstrations / "libero" / "libero" / "benchmark" / "__init__.py"
    benchmark.write_text(
        benchmark.read_text(encoding="utf-8").replace(
            '        return "libero_spatial/open_drawer_demo.hdf5"',
            "        raise FileNotFoundError(index)",
        ),
        encoding="utf-8",
    )

    def prepared_examples(checkout: Path, output_root: Path) -> list:
        prepare_from_checkout(
            checkout_path=checkout,
            suite="libero_spatial",
            environment_version=_ENVIRONMENT_VERSION,
            source_revision="official-fixture-revision",
            output_root=output_root,
            output_format="jsonl",
        )
        return read_prepared("libero_spatial", "v1", output_root)

    complete = prepared_examples(with_demonstrations, tmp_path / "prepared-with")
    partial = prepared_examples(without_demonstrations, tmp_path / "prepared-without")

    assert [example.task_uid for example in complete] == [example.task_uid for example in partial]
    assert all("demonstration_path" in example.env_payload for example in complete)
    assert all("demonstration_path" not in example.env_payload for example in partial)
