"""Contract tests for the benchmark-specific LIBERO-Pro preparer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from alphaapollo.data_preprocess import read_manifest, read_prepared
from alphaapollo.data_preprocess.robotics import prepare_liberopro as preparer
from alphaapollo.data_preprocess.robotics.prepare_liberopro import (
    LIBEROPRO_SUITES,
    PACKAGE_DISTRIBUTION,
    SUPPORTED_PACKAGE_VERSION,
    main,
    prepare,
    prepare_from_package,
)

_ENVIRONMENT_VERSION = f"{PACKAGE_DISTRIBUTION}=={SUPPORTED_PACKAGE_VERSION}"


class _Suite:
    def __init__(
        self,
        suite: str,
        task_order_index: int,
        *,
        instruction: str = "put the cream cheese in the bowl",
        initial_state_count: int = 2,
    ) -> None:
        self.suite = suite
        self.task_order_index = task_order_index
        self._task = SimpleNamespace(
            name="put_the_cream_cheese_in_the_bowl",
            language=instruction,
            problem="Libero",
            problem_folder=suite,
            bddl_file="put_the_cream_cheese_in_the_bowl.bddl",
            init_states_file="put_the_cream_cheese_in_the_bowl.pruned_init",
        )
        self._initial_states = [object() for _ in range(initial_state_count)]

    def get_num_tasks(self) -> int:
        return 1

    def get_task(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return self._task

    def get_task_init_states(self, index: int) -> list[object]:
        if index != 0:
            raise IndexError(index)
        return self._initial_states


def _install_fake_package(
    monkeypatch: pytest.MonkeyPatch,
    *,
    suites: tuple[str, ...] = LIBEROPRO_SUITES,
    installed_version: str = SUPPORTED_PACKAGE_VERSION,
    instruction: str = "put the cream cheese in the bowl",
    initial_state_count: int = 2,
) -> None:
    monkeypatch.setattr(
        preparer.importlib.metadata,
        "version",
        lambda distribution: (
            installed_version
            if distribution == PACKAGE_DISTRIBUTION
            else pytest.fail(f"unexpected distribution {distribution!r}")
        ),
    )
    module = ModuleType("liberopro.liberopro.benchmark")
    module.get_benchmark_dict = lambda: {
        suite: (
            lambda task_order_index=0, suite=suite: _Suite(
                suite,
                task_order_index,
                instruction=instruction,
                initial_state_count=initial_state_count,
            )
        )
        for suite in suites
    }
    monkeypatch.setattr(
        preparer.importlib,
        "import_module",
        lambda name: (
            module
            if name == "liberopro.liberopro.benchmark"
            else pytest.fail(f"unexpected import {name!r}")
        ),
    )


def _row(*, suite: str = "libero_goal_swap", **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": "cream-cheese-task",
        "language": "put the cream cheese in the bowl",
        "problem": "Libero",
        "suite": suite,
        "task_order_index": 0,
        "task_index": 0,
        "task_name": "put_the_cream_cheese_in_the_bowl",
        "problem_folder": suite,
        "bddl_file": "put_the_cream_cheese_in_the_bowl.bddl",
        "init_states_file": "put_the_cream_cheese_in_the_bowl.pruned_init",
        "initial_state_index": 0,
        "package_distribution": PACKAGE_DISTRIBUTION,
        "package_version": SUPPORTED_PACKAGE_VERSION,
        "environment_version": _ENVIRONMENT_VERSION,
        "source_uri": "https://github.com/RLinf/LIBERO-PRO",
        "source_revision": "upstream-row-claim",
        "perturbation": "position",
    }
    row.update(updates)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize("suite", LIBEROPRO_SUITES)
def test_all_supported_liberopro_suites_use_the_shared_prepared_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suite: str,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / f"{suite}.jsonl"
    _write_jsonl(source, [_row(suite=suite)])

    prepared = prepare(
        data_source=str(source),
        suite=suite,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    assert prepared.root == tmp_path / "prepared" / suite / "v1"
    example = read_prepared(suite, "v1", tmp_path / "prepared")[0]
    assert example.statement == "put the cream cheese in the bowl"
    assert example.domain == "robotics"
    assert example.group_id == suite
    assert example.answer == "environment"
    assert example.answer_type == "none"
    assert example.grader_id == "environment_success"
    assert example.builder_version == "1"
    assert example.env_payload == {
        "benchmark": "liberopro",
        "environment_version": _ENVIRONMENT_VERSION,
        "suite": suite,
        "task_order_index": 0,
        "task_index": 0,
        "task_name": "put_the_cream_cheese_in_the_bowl",
        "problem_folder": suite,
        "bddl_file": "put_the_cream_cheese_in_the_bowl.bddl",
        "init_states_file": "put_the_cream_cheese_in_the_bowl.pruned_init",
        "initial_state_index": 0,
        "package_distribution": PACKAGE_DISTRIBUTION,
        "package_version": SUPPORTED_PACKAGE_VERSION,
    }
    assert example.extra["source_metadata"] == {
        # ``problem`` is the official benchmark label, not an instruction, so
        # it is neither promoted into the payload nor dropped on the way past.
        "problem": "Libero",
        "perturbation": "position",
        "source_revision": "upstream-row-claim",
        "source_uri": "https://github.com/RLinf/LIBERO-PRO",
    }
    assert example.source_uri == source.resolve().as_uri()
    assert example.source_revision == "local"


@pytest.mark.parametrize(
    "suite",
    ["libero_goal", "libero_goal_temp", "libero_goal_relation_ood", "unknown"],
)
def test_liberopro_rejects_noncanonical_suite_scope(tmp_path: Path, suite: str) -> None:
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row()])

    with pytest.raises(ValueError, match="unsupported LIBERO-Pro suite"):
        prepare(
            data_source=str(source),
            suite=suite,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_liberopro_requires_the_supported_package_version(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row()])

    with pytest.raises(ValueError, match="unsupported rpent-liberopro version"):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            package_version="0.2.0",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("package_distribution", "other-package", "package_distribution"),
        ("package_version", "0.1.0", "package_version"),
        ("environment_version", "rpent-liberopro==0.1.0", "environment_version"),
    ],
)
def test_liberopro_rejects_conflicting_row_build_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row(**{field: value})])

    with pytest.raises(ValueError, match=message):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


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
def test_liberopro_rejects_missing_execution_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    removed: str,
    message: str,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    row = _row()
    row.pop(removed)
    _write_jsonl(source, [row])

    with pytest.raises(ValueError, match=message):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_liberopro_rejects_conflicting_instruction_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row(instruction="put the wine bottle in the bowl")])

    with pytest.raises(ValueError, match="instruction is ambiguous"):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_liberopro_validates_metadata_against_the_installed_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row(language="put the wine bottle in the bowl")])

    with pytest.raises(ValueError, match="does not match the installed package"):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


@pytest.mark.parametrize(
    "field",
    ["task_name", "problem_folder", "bddl_file", "init_states_file"],
)
def test_liberopro_rejects_task_metadata_that_drifts_from_the_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Drift here refuses every episode at reset time, so it must fail at build."""

    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row(**{field: "drifted_from_upstream"})])

    with pytest.raises(ValueError, match=f"LIBERO-Pro {field} mismatch"):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_liberopro_rejects_a_row_from_another_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row(suite="libero_object_lan")])

    with pytest.raises(ValueError, match="does not match configured suite"):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_liberopro_refuses_a_revision_for_a_local_source(tmp_path: Path) -> None:
    """A local file has no Hub commit; recording one would claim false provenance."""

    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row()])

    with pytest.raises(ValueError, match=r"revision 'deadbeef' was given for the local source"):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            revision="deadbeef",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_liberopro_refuses_a_row_horizon_conflicting_with_the_build_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row(max_episode_steps=520)])

    with pytest.raises(ValueError, match="does not match configured max_episode_steps 300"):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            max_episode_steps=300,
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_liberopro_records_unverified_license_provenance_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upstream MIT package license does not license an arbitrary export."""

    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row()])

    prepare(
        data_source=str(source),
        suite="libero_goal_swap",
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_goal_swap", "v1", tmp_path / "prepared")[0]
    assert example.dataset_license == "unverified"
    assert example.original_content_owner == "unverified"
    assert example.original_content_license == "unverified"


def test_liberopro_identity_escapes_delimiters_in_task_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row(task_id="cream:cheese")])

    prepare(
        data_source=str(source),
        suite="libero_goal_swap",
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_goal_swap", "v1", tmp_path / "prepared")[0]
    assert example.source_record_id == "cream%3Acheese:0"


def test_liberopro_rejects_an_out_of_range_initial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch, initial_state_count=2)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row(initial_state_index=2)])

    with pytest.raises(ValueError, match="out of range"):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_liberopro_identity_distinguishes_initial_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    first = _row()
    second = _row(initial_state_index=1)
    _write_jsonl(source, [first, second])

    prepare(
        data_source=str(source),
        suite="libero_goal_swap",
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    examples = read_prepared("libero_goal_swap", "v1", tmp_path / "prepared")
    assert [example.source_record_id for example in examples] == [
        "cream-cheese-task:0",
        "cream-cheese-task:1",
    ]
    assert examples[0].task_uid != examples[1].task_uid


@pytest.mark.parametrize("value", [0, -1, True, "not-an-int"])
def test_liberopro_rejects_invalid_episode_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row(max_episode_steps=value)])

    with pytest.raises(ValueError, match="max_episode_steps"):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_liberopro_carries_only_an_explicit_episode_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    row = _row()
    row.pop("task_id")
    _write_jsonl(source, [row])

    prepare(
        data_source=str(source),
        suite="libero_goal_swap",
        max_episode_steps=300,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    example = read_prepared("libero_goal_swap", "v1", tmp_path / "prepared")[0]
    assert example.env_payload["max_episode_steps"] == 300
    assert example.source_record_id == ("libero_goal_swap:0:0:put_the_cream_cheese_in_the_bowl:0")


def test_liberopro_canonical_package_path_generates_one_row_per_initial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch, initial_state_count=2)

    prepared = prepare_from_package(
        suite="libero_goal_swap",
        task_order_index=3,
        max_episode_steps=300,
        output_root=tmp_path / "prepared",
        output_format="jsonl",
    )

    manifest = read_manifest(prepared.manifest_path)
    assert manifest.data_source == ("pypi://rpent-liberopro/0.1.1/libero_goal_swap/3")
    assert manifest.source_revision == SUPPORTED_PACKAGE_VERSION
    examples = read_prepared("libero_goal_swap", "v1", tmp_path / "prepared")
    assert len(examples) == 2
    assert [example.env_payload["initial_state_index"] for example in examples] == [0, 1]
    assert all(example.env_payload["max_episode_steps"] == 300 for example in examples)
    assert all(example.source_uri == "https://github.com/RLinf/LIBERO-PRO" for example in examples)
    assert all(example.source_revision == SUPPORTED_PACKAGE_VERSION for example in examples)
    # The package's ``problem`` label is read from the task record, so it has to
    # arrive somewhere a reader can find it rather than being generated and lost.
    assert all(example.extra["source_metadata"] == {"problem": "Libero"} for example in examples)
    assert all("bddl_asset" not in example.env_payload for example in examples)
    assert all("selected_state_sha256" not in example.env_payload for example in examples)
    assert all("evaluation_seed" not in example.env_payload for example in examples)
    assert all("pi05_model" not in example.env_payload for example in examples)


def test_liberopro_canonical_path_fails_closed_for_an_empty_upstream_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch, initial_state_count=0)

    with pytest.raises(ValueError, match="has no initial states"):
        prepare_from_package(
            suite="libero_spatial_task",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )

    assert not (tmp_path / "prepared" / "libero_spatial_task" / "v1").exists()


def test_liberopro_canonical_path_requires_the_exact_installed_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch, installed_version="0.1.0")

    with pytest.raises(RuntimeError, match="expected '0.1.1'"):
        prepare_from_package(
            suite="libero_goal_swap",
            output_root=tmp_path / "prepared",
            output_format="jsonl",
        )


def test_liberopro_metadata_reuse_does_not_reimport_the_optional_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row()])
    arguments: dict[str, Any] = {
        "data_source": str(source),
        "suite": "libero_goal_swap",
        "output_root": tmp_path / "prepared",
        "output_format": "jsonl",
    }
    built = prepare(**arguments)

    def missing(_distribution: str) -> str:
        raise preparer.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(preparer.importlib.metadata, "version", missing)
    reused = prepare(**arguments)
    assert reused.build_id == built.build_id

    with pytest.raises(RuntimeError, match="requires rpent-liberopro"):
        prepare(dataset_version="v2", **arguments)


def test_liberopro_build_options_prevent_stale_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row()])
    output = tmp_path / "prepared"
    prepare(
        data_source=str(source),
        suite="libero_goal_swap",
        dataset_name="shared",
        output_root=output,
        output_format="jsonl",
    )

    with pytest.raises(ValueError, match="different spec"):
        prepare(
            data_source=str(source),
            suite="libero_goal_swap",
            dataset_name="shared",
            max_episode_steps=300,
            output_root=output,
            output_format="jsonl",
        )


def test_liberopro_module_import_does_not_require_the_optional_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import alphaapollo.data_preprocess.robotics.prepare_liberopro",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_liberopro_cli_supports_metadata_and_canonical_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_package(monkeypatch)
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row()])

    assert (
        main(
            [
                "--data-source",
                str(source),
                "--suite",
                "libero_goal_swap",
                "--output-root",
                str(tmp_path / "metadata"),
                "--format",
                "jsonl",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--from-installed-package",
                "--suite",
                "libero_goal_swap",
                "--task-order-index",
                "2",
                "--output-root",
                str(tmp_path / "canonical"),
                "--format",
                "jsonl",
            ]
        )
        == 0
    )


def test_liberopro_cli_rejects_mode_specific_options(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [_row()])

    with pytest.raises(SystemExit):
        main(
            [
                "--data-source",
                str(source),
                "--suite",
                "libero_goal_swap",
                "--task-order-index",
                "1",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--from-installed-package",
                "--suite",
                "libero_goal_swap",
                "--revision",
                "deadbeef",
            ]
        )
