"""Prepare LIBERO-Pro task metadata with AlphaApollo's shared contracts.

The metadata path normalizes caller-provided JSON, JSONL, Parquet, or Hugging
Face rows and validates them against an installed, exactly pinned LIBERO-Pro
package. The canonical package path generates rows directly from the package's
benchmark registry. One row identifies one task and one initial-state index.
Neither path starts a simulator, copies BDDL or initial-state assets, or reads
demonstration trajectories.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

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

__all__ = [
    "LIBEROPRO_SUITES",
    "PACKAGE_DISTRIBUTION",
    "SUPPORTED_PACKAGE_VERSION",
    "prepare",
    "prepare_from_package",
]


BUILDER_NAME = "liberopro"
BUILDER_VERSION = "1"
PACKAGE_DISTRIBUTION = "rpent-liberopro"
SUPPORTED_PACKAGE_VERSION = "0.1.1"
_LIBEROPRO_SOURCE_URI = "https://github.com/RLinf/LIBERO-PRO"
_BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
_PERTURBATIONS = ("swap", "task", "lan", "object")
LIBEROPRO_SUITES = tuple(
    f"{base}_{perturbation}" for base in _BASE_SUITES for perturbation in _PERTURBATIONS
)
# Measured from the published rpent-liberopro==0.1.1 wheel on 2026-08-18.
# These registry entries exist, but their packaged init-state files deserialize
# to empty sequences. Keep them in the suite contract and fail closed when a
# caller reaches the incomplete task; silently dropping tasks would publish a
# dataset that no longer represents the named upstream suite.
_KNOWN_EMPTY_INITIAL_STATE_TASKS = {
    "libero_spatial_task": (3, 7),
    "libero_10_task": (2,),
    "libero_10_object": (4,),
}

_PRIMARY_INSTRUCTION_KEYS = (
    "language",
    "task_language",
    "instruction",
    "language_instruction",
)
_SECONDARY_INSTRUCTION_KEYS = ("problem", "question", "prompt")
_TASK_ID_KEYS = ("task_id", "task_uid", "id")
_ROW_ID_KEYS = ("source_record_id",)
_SUITE_KEYS = ("suite", "benchmark_suite", "category")
_TASK_NAME_KEYS = ("task_name", "name")
_PROBLEM_FOLDER_KEYS = ("problem_folder",)
_BDDL_KEYS = ("bddl_file", "bddl_file_name")
_INIT_STATES_KEYS = ("init_states_file", "initial_states_file")
_MAX_EPISODE_STEPS_KEYS = ("max_episode_steps",)
_OFFICIAL_PROBLEM_LABEL = "Libero"


def _present_aliases(
    raw: Mapping[str, Any],
    names: Sequence[str],
) -> list[tuple[str, Any]]:
    return [(name, raw[name]) for name in names if name in raw and raw[name] is not None]


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LIBERO-Pro {field} must be a non-empty string")
    return value.strip()


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"LIBERO-Pro {field} must be a non-negative integer")
    if isinstance(value, int):
        converted = value
    elif isinstance(value, str) and value.strip().isdigit():
        converted = int(value.strip())
    else:
        raise ValueError(f"LIBERO-Pro {field} must be a non-negative integer")
    if converted < 0:
        raise ValueError(f"LIBERO-Pro {field} must be a non-negative integer")
    return converted


def _positive_integer(value: Any, *, field: str) -> int:
    converted = _nonnegative_integer(value, field=field)
    if converted == 0:
        raise ValueError(f"LIBERO-Pro {field} must be a positive integer")
    return converted


def _resolve_string_aliases(
    raw: Mapping[str, Any],
    names: Sequence[str],
    label: str,
) -> tuple[str, str]:
    present = _present_aliases(raw, names)
    if not present:
        raise ValueError(f"LIBERO-Pro row is missing {label}; tried {list(names)}")
    normalized = [(name, _nonempty_string(value, field=label)) for name, value in present]
    values = {value for _, value in normalized}
    if len(values) > 1:
        fields = ", ".join(name for name, _ in normalized)
        raise ValueError(
            f"LIBERO-Pro {label} is ambiguous across fields: {fields}; "
            "provide one value or matching aliases"
        )
    return normalized[0]


def _resolve_optional_string_aliases(
    raw: Mapping[str, Any],
    names: Sequence[str],
    label: str,
) -> tuple[str | None, str | None]:
    present = _present_aliases(raw, names)
    if not present:
        return None, None
    normalized = [(name, _nonempty_string(value, field=label)) for name, value in present]
    values = {value for _, value in normalized}
    if len(values) > 1:
        fields = ", ".join(name for name, _ in normalized)
        raise ValueError(
            f"LIBERO-Pro {label} is ambiguous across fields: {fields}; "
            "provide one value or matching aliases"
        )
    return normalized[0]


def _instruction(raw: Mapping[str, Any]) -> tuple[str, str, frozenset[str]]:
    """Resolve the instruction and report only the keys it actually consumed.

    Official task records carry ``language`` beside ``problem="Libero"``. That
    label is the benchmark's problem class, not an instruction, so it is never
    consumed here: an unconsumed key survives in ``extra["source_metadata"]``,
    while a key reported as consumed but never read would be dropped from both
    the payload and the metadata. ``prepare_libero`` keeps the same field for
    the same reason. Older exports used ``problem`` as a generic instruction
    column, so it stays a fallback when no primary alias is present.
    """

    primary = _present_aliases(raw, _PRIMARY_INSTRUCTION_KEYS)
    normalized_primary = [
        (name, _nonempty_string(value, field="task instruction")) for name, value in primary
    ]
    primary_values = {value for _, value in normalized_primary}
    if len(primary_values) > 1:
        fields = ", ".join(name for name, _ in normalized_primary)
        raise ValueError(
            "LIBERO-Pro task instruction is ambiguous across fields: "
            f"{fields}; provide one value or matching aliases"
        )

    secondary = _present_aliases(raw, _SECONDARY_INSTRUCTION_KEYS)
    normalized_secondary = [
        (name, _nonempty_string(value, field="task instruction")) for name, value in secondary
    ]
    meaningful_secondary = [
        (name, value)
        for name, value in normalized_secondary
        if not (name == "problem" and value == _OFFICIAL_PROBLEM_LABEL)
    ]
    if normalized_primary:
        instruction_key, instruction = normalized_primary[0]
        conflicting = [name for name, value in meaningful_secondary if value != instruction]
        if conflicting:
            raise ValueError(
                "LIBERO-Pro task instruction is ambiguous across fields: "
                f"{instruction_key}, {', '.join(conflicting)}; provide matching aliases"
            )
        consumed = {name for name, _ in normalized_primary}
        consumed.update(name for name, _ in meaningful_secondary)
        return instruction_key, instruction, frozenset(consumed)

    values = {value for _, value in meaningful_secondary}
    if not meaningful_secondary:
        raise ValueError(
            "LIBERO-Pro row is missing task instruction; tried "
            f"{list((*_PRIMARY_INSTRUCTION_KEYS, *_SECONDARY_INSTRUCTION_KEYS))}"
        )
    if len(values) > 1:
        fields = ", ".join(name for name, _ in meaningful_secondary)
        raise ValueError(
            "LIBERO-Pro task instruction is ambiguous across fields: "
            f"{fields}; provide one value or matching aliases"
        )
    instruction_key, instruction = meaningful_secondary[0]
    return instruction_key, instruction, frozenset(name for name, _ in meaningful_secondary)


def _normalize_id(value: Any) -> str:
    if (
        isinstance(value, bool)
        or isinstance(value, (Mapping, Sequence))
        and not isinstance(value, (str, bytes))
    ):
        raise ValueError("LIBERO-Pro task id must be a string or scalar")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("LIBERO-Pro task id must be non-empty")
    return normalized


def _resolve_id_aliases(
    raw: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[str | None, str | None]:
    present = _present_aliases(raw, names)
    if not present:
        return None, None
    normalized = [(name, _normalize_id(value)) for name, value in present]
    values = {value for _, value in normalized}
    if len(values) > 1:
        fields = ", ".join(name for name, _ in normalized)
        raise ValueError(
            f"LIBERO-Pro task id is ambiguous across fields: {fields}; "
            "provide one value or matching aliases"
        )
    return normalized[0]


def _suite_name(value: Any) -> str:
    suite = _nonempty_string(value, field="suite")
    if suite not in LIBEROPRO_SUITES:
        raise ValueError(
            f"unsupported LIBERO-Pro suite {suite!r}; expected one of {list(LIBEROPRO_SUITES)}"
        )
    return suite


def _package_version(value: Any) -> str:
    version = _nonempty_string(value, field="package_version")
    if version != SUPPORTED_PACKAGE_VERSION:
        raise ValueError(
            f"unsupported {PACKAGE_DISTRIBUTION} version {version!r}; "
            f"expected {SUPPORTED_PACKAGE_VERSION!r}"
        )
    return version


def _environment_version(package_version: str) -> str:
    return f"{PACKAGE_DISTRIBUTION}=={package_version}"


def _row_suite(raw: Mapping[str, Any], configured_suite: str) -> tuple[str, str | None]:
    source_key, value = _resolve_optional_string_aliases(raw, _SUITE_KEYS, "suite")
    if source_key is None or value is None:
        return configured_suite, None
    row_suite = _suite_name(value)
    if row_suite != configured_suite:
        raise ValueError(
            f"LIBERO-Pro row suite {row_suite!r} does not match configured suite "
            f"{configured_suite!r}"
        )
    return row_suite, source_key


def _source_record_id(
    raw: Mapping[str, Any],
    *,
    suite: str,
    task_order_index: int,
    task_index: int,
    task_name: str,
    initial_state_index: int,
) -> tuple[str, str]:
    row_id_key, row_id = _resolve_id_aliases(raw, _ROW_ID_KEYS)
    if row_id_key is not None and row_id is not None:
        return row_id, row_id_key
    task_id_key, task_id = _resolve_id_aliases(raw, _TASK_ID_KEYS)
    if task_id_key is not None and task_id is not None:
        return _compose_identity((task_id, str(initial_state_index))), task_id_key
    return (
        _compose_identity(
            (suite, str(task_order_index), str(task_index), task_name, str(initial_state_index))
        ),
        "derived",
    )


def _compose_identity(parts: Sequence[str]) -> str:
    return ":".join(quote(part, safe="") for part in parts)


def _validate_row_build_identity(
    raw: Mapping[str, Any],
    *,
    package_version: str,
    environment_version: str,
) -> None:
    row_distribution = raw.get("package_distribution")
    if row_distribution is not None:
        distribution = _nonempty_string(row_distribution, field="package_distribution")
        if distribution != PACKAGE_DISTRIBUTION:
            raise ValueError(
                f"LIBERO-Pro row package_distribution {distribution!r} does not match "
                f"{PACKAGE_DISTRIBUTION!r}"
            )
    row_package_version = raw.get("package_version")
    if row_package_version is not None:
        version = _nonempty_string(row_package_version, field="package_version")
        if version != package_version:
            raise ValueError(
                f"LIBERO-Pro row package_version {version!r} does not match configured "
                f"package_version {package_version!r}"
            )
    row_environment_version = raw.get("environment_version")
    if row_environment_version is not None:
        version = _nonempty_string(row_environment_version, field="environment_version")
        if version != environment_version:
            raise ValueError(
                f"LIBERO-Pro row environment_version {version!r} does not match configured "
                f"environment_version {environment_version!r}"
            )


def _benchmark_factory(benchmarks: Mapping[str, Any], suite: str) -> Any:
    for name, factory in benchmarks.items():
        if str(name).lower() == suite.lower():
            return factory
    raise ValueError(f"LIBERO-Pro suite {suite!r} is not registered by the installed package")


def _installed_benchmarks(package_version: str) -> Mapping[str, Any]:
    try:
        installed = importlib.metadata.version(PACKAGE_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"LIBERO-Pro preparation requires {PACKAGE_DISTRIBUTION}=={package_version}"
        ) from exc
    if installed != package_version:
        raise RuntimeError(
            f"installed {PACKAGE_DISTRIBUTION} version is {installed!r}, "
            f"expected {package_version!r}"
        )
    try:
        module = importlib.import_module("liberopro.liberopro.benchmark")
    except ImportError as exc:
        raise RuntimeError(
            f"{PACKAGE_DISTRIBUTION}=={package_version} does not expose the LIBERO-Pro "
            "benchmark API"
        ) from exc
    get_benchmark_dict = getattr(module, "get_benchmark_dict", None)
    if not callable(get_benchmark_dict):
        raise RuntimeError("the installed LIBERO-Pro package exposes no get_benchmark_dict API")
    benchmarks = get_benchmark_dict()
    if not isinstance(benchmarks, Mapping):
        raise RuntimeError("LIBERO-Pro get_benchmark_dict() must return a mapping")
    return benchmarks


def _suite_instance(benchmarks: Mapping[str, Any], suite: str, task_order_index: int) -> Any:
    factory = _benchmark_factory(benchmarks, suite)
    if not callable(factory):
        raise RuntimeError(f"LIBERO-Pro suite factory for {suite!r} is not callable")
    try:
        return factory(task_order_index)
    except Exception as exc:
        raise ValueError(
            f"LIBERO-Pro task_order_index {task_order_index} is not available in "
            f"suite {suite!r}: {exc}"
        ) from exc


def _task_count(suite_instance: Any, suite: str) -> int:
    getter = getattr(suite_instance, "get_num_tasks", None)
    if not callable(getter):
        raise RuntimeError(
            f"LIBERO-Pro suite {suite!r} does not expose the required get_num_tasks API"
        )
    count = getter()
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RuntimeError(f"LIBERO-Pro suite {suite!r} exposes no positive task count")
    return count


def _task_string(task: Any, attribute: str) -> str:
    return _nonempty_string(
        getattr(task, attribute, None),
        field=f"official task {attribute}",
    )


def _initial_state_count(suite_instance: Any, task_index: int, suite: str) -> int:
    try:
        states = suite_instance.get_task_init_states(task_index)
        count = len(states)
    except Exception as exc:
        raise ValueError(
            f"LIBERO-Pro initial states are unavailable for suite {suite!r}, "
            f"task_index {task_index}: {exc}"
        ) from exc
    if count <= 0:
        known = task_index in _KNOWN_EMPTY_INITIAL_STATE_TASKS.get(suite, ())
        detail = (
            f"; this is a known incomplete task in {PACKAGE_DISTRIBUTION}=="
            f"{SUPPORTED_PACKAGE_VERSION}"
            if known
            else ""
        )
        raise ValueError(
            f"LIBERO-Pro task_index {task_index} in suite {suite!r} has no initial states{detail}"
        )
    return count


def _canonical_rows(
    benchmarks: Mapping[str, Any],
    *,
    suite: str,
    task_order_index: int,
    package_version: str,
    max_episode_steps: int | None,
) -> list[dict[str, Any]]:
    instance = _suite_instance(benchmarks, suite, task_order_index)
    rows: list[dict[str, Any]] = []
    for task_index in range(_task_count(instance, suite)):
        try:
            task = instance.get_task(task_index)
        except Exception as exc:
            raise ValueError(
                f"LIBERO-Pro task_index {task_index} is unavailable in suite {suite!r}: {exc}"
            ) from exc
        task_fields = {
            "task_name": _task_string(task, "name"),
            "language": _task_string(task, "language"),
            "problem_folder": _task_string(task, "problem_folder"),
            "bddl_file": _task_string(task, "bddl_file"),
            "init_states_file": _task_string(task, "init_states_file"),
        }
        problem = getattr(task, "problem", None)
        if problem is not None:
            task_fields["problem"] = _nonempty_string(
                problem,
                field="official task problem",
            )
        for initial_state_index in range(_initial_state_count(instance, task_index, suite)):
            row: dict[str, Any] = {
                "source_record_id": _compose_identity(
                    (
                        suite,
                        str(task_order_index),
                        str(task_index),
                        task_fields["task_name"],
                        str(initial_state_index),
                    )
                ),
                "suite": suite,
                "task_order_index": task_order_index,
                "task_index": task_index,
                **task_fields,
                "initial_state_index": initial_state_index,
                "package_distribution": PACKAGE_DISTRIBUTION,
                "package_version": package_version,
                "environment_version": _environment_version(package_version),
            }
            if max_episode_steps is not None:
                row["max_episode_steps"] = max_episode_steps
            rows.append(row)
    return rows


def _package_validator(package_version: str):
    resolved: dict[str, Mapping[str, Any]] = {}
    suites: dict[tuple[str, int], Any] = {}
    state_counts: dict[tuple[str, int, int], int] = {}

    def benchmarks() -> Mapping[str, Any]:
        if "api" not in resolved:
            resolved["api"] = _installed_benchmarks(package_version)
        return resolved["api"]

    def validate(
        *,
        suite: str,
        task_order_index: int,
        task_index: int,
        task_name: str,
        problem_folder: str,
        bddl_file: str,
        init_states_file: str,
        initial_state_index: int,
        instruction: str,
    ) -> None:
        key = (suite, task_order_index)
        instance = suites.get(key)
        if instance is None:
            instance = _suite_instance(benchmarks(), suite, task_order_index)
            suites[key] = instance
        try:
            task = instance.get_task(task_index)
        except Exception as exc:
            raise ValueError(
                f"LIBERO-Pro task_index {task_index} is unavailable in suite {suite!r}: {exc}"
            ) from exc
        for field, expected in (
            ("task_name", task_name),
            ("problem_folder", problem_folder),
            ("bddl_file", bddl_file),
            ("init_states_file", init_states_file),
        ):
            actual = getattr(task, field if field != "task_name" else "name", None)
            if actual != expected:
                raise ValueError(
                    f"LIBERO-Pro {field} mismatch against the installed package: "
                    f"row has {expected!r}, package has {actual!r}"
                )
        language = getattr(task, "language", None)
        if not isinstance(language, str) or language.strip() != instruction.strip():
            raise ValueError(
                "LIBERO-Pro instruction does not match the installed package task language: "
                f"row has {instruction!r}, package has {language!r}"
            )
        count_key = (suite, task_order_index, task_index)
        count = state_counts.get(count_key)
        if count is None:
            count = _initial_state_count(instance, task_index, suite)
            state_counts[count_key] = count
        if initial_state_index >= count:
            raise ValueError(
                f"LIBERO-Pro initial_state_index {initial_state_index} is out of range for "
                f"{count} prepared initial states"
            )

    return validate


def _normalizer(
    *,
    suite: str,
    package_version: str,
    source_uri: str,
    source_revision: str,
    dataset_license: str,
    original_content_owner: str,
    original_content_license: str,
    max_episode_steps: int | None = None,
    validator=None,
):
    environment_version = _environment_version(package_version)

    def normalize(raw: Mapping[str, Any], context: NormalizeContext) -> PreparedExample:
        row_suite, suite_key = _row_suite(raw, suite)
        instruction_key, instruction, instruction_keys = _instruction(raw)
        _, task_name = _resolve_string_aliases(raw, _TASK_NAME_KEYS, "task name")
        _, problem_folder = _resolve_string_aliases(
            raw,
            _PROBLEM_FOLDER_KEYS,
            "problem folder",
        )
        _, bddl_file = _resolve_string_aliases(raw, _BDDL_KEYS, "BDDL file")
        _, init_states_file = _resolve_string_aliases(
            raw,
            _INIT_STATES_KEYS,
            "initial-states file",
        )
        _validate_row_build_identity(
            raw,
            package_version=package_version,
            environment_version=environment_version,
        )
        task_order_index = _nonnegative_integer(
            raw.get("task_order_index", 0),
            field="task_order_index",
        )
        task_index = _nonnegative_integer(raw.get("task_index"), field="task_index")
        initial_state_index = _nonnegative_integer(
            raw.get("initial_state_index", 0),
            field="initial_state_index",
        )
        row_horizon = raw.get("max_episode_steps")
        if row_horizon is None:
            resolved_horizon = max_episode_steps
        else:
            resolved_horizon = _positive_integer(row_horizon, field="max_episode_steps")
            if max_episode_steps is not None and resolved_horizon != max_episode_steps:
                raise ValueError(
                    f"LIBERO-Pro row max_episode_steps {resolved_horizon} does not match "
                    f"configured max_episode_steps {max_episode_steps}"
                )
        if validator is not None:
            validator(
                suite=row_suite,
                task_order_index=task_order_index,
                task_index=task_index,
                task_name=task_name,
                problem_folder=problem_folder,
                bddl_file=bddl_file,
                init_states_file=init_states_file,
                initial_state_index=initial_state_index,
                instruction=instruction,
            )
        record_id, id_key = _source_record_id(
            raw,
            suite=row_suite,
            task_order_index=task_order_index,
            task_index=task_index,
            task_name=task_name,
            initial_state_index=initial_state_index,
        )
        consumed_keys = {
            *instruction_keys,
            *_TASK_ID_KEYS,
            *_ROW_ID_KEYS,
            *_SUITE_KEYS,
            *_TASK_NAME_KEYS,
            *_PROBLEM_FOLDER_KEYS,
            *_BDDL_KEYS,
            *_INIT_STATES_KEYS,
            *_MAX_EPISODE_STEPS_KEYS,
            "task_order_index",
            "task_index",
            "initial_state_index",
            "package_distribution",
            "package_version",
            "environment_version",
        }
        source_metadata = {key: value for key, value in raw.items() if key not in consumed_keys}
        backend_metadata: dict[str, Any] = {
            "suite": row_suite,
            "task_order_index": task_order_index,
            "task_index": task_index,
            "task_name": task_name,
            "problem_folder": problem_folder,
            "bddl_file": bddl_file,
            "init_states_file": init_states_file,
            "initial_state_index": initial_state_index,
            "package_distribution": PACKAGE_DISTRIBUTION,
            "package_version": package_version,
        }
        if resolved_horizon is not None:
            backend_metadata["max_episode_steps"] = resolved_horizon
        return PreparedExample(
            task_uid=task_uid(context.source_id, record_id),
            source_id=context.source_id,
            source_record_id=record_id,
            source_order=context.source_order,
            split=context.split,
            group_id=row_suite,
            statement=instruction,
            domain="robotics",
            answer="environment",
            answer_type="none",
            env_payload={
                "benchmark": "liberopro",
                "environment_version": environment_version,
                **backend_metadata,
            },
            extra={
                "source_instruction_key": instruction_key,
                "source_id_key": id_key,
                "source_suite_key": suite_key or "configured",
                "source_fields": sorted(raw),
                "source_metadata": source_metadata,
            },
            grader_id="environment_success",
            source_uri=source_uri,
            source_revision=source_revision,
            dataset_license=dataset_license,
            original_content_owner=original_content_owner,
            original_content_license=original_content_license,
            raw_digest=raw_digest(raw),
            builder_version=context.builder_version,
        )

    return normalize


def prepare(
    *,
    data_source: str,
    suite: str,
    package_version: str = SUPPORTED_PACKAGE_VERSION,
    dataset_name: str | None = None,
    output_root: Path | None = None,
    dataset_version: str = "v1",
    revision: str | None = None,
    splits: Sequence[str] | None = None,
    output_format: OutputFormat | str = OutputFormat.PARQUET,
    if_exists: ExistingPolicy | str = ExistingPolicy.REUSE,
    max_episode_steps: int | None = None,
    dataset_license: str = "unverified",
    original_content_owner: str = "unverified",
    original_content_license: str = "unverified",
) -> PreparedDataset:
    """Normalize metadata rows and validate them against the installed package."""

    normalized_suite = _suite_name(suite)
    normalized_version = _package_version(package_version)
    if max_episode_steps is not None:
        max_episode_steps = _positive_integer(max_episode_steps, field="max_episode_steps")
    dataset_license = _nonempty_string(dataset_license, field="dataset_license")
    original_content_owner = _nonempty_string(
        original_content_owner,
        field="original_content_owner",
    )
    original_content_license = _nonempty_string(
        original_content_license,
        field="original_content_license",
    )
    split_names = tuple(splits) if splits is not None else ("test",)
    local_source = Path(data_source).expanduser().exists()
    if revision is not None and local_source:
        raise ValueError(
            f"revision {revision!r} was given for the local source {data_source!r}; a local "
            "file has no Hub commit and is fingerprinted by its content"
        )
    request = BuildRequest(
        dataset_name=dataset_name or normalized_suite,
        dataset_version=dataset_version,
        data_source=data_source,
        revision=revision,
        splits=split_names,
        output_root=output_root or default_root(),
        output_format=OutputFormat(output_format),
        if_exists=ExistingPolicy(if_exists),
    )
    if revision is None and not local_source:
        revision = resolve_hub_revision(data_source)
        request = replace(request, revision=revision)
    source_uri = (
        Path(data_source).expanduser().resolve().as_uri()
        if local_source
        else f"https://huggingface.co/datasets/{data_source}"
    )
    return prepare_dataset(
        request,
        _normalizer(
            suite=normalized_suite,
            package_version=normalized_version,
            source_uri=source_uri,
            source_revision=revision or "local",
            dataset_license=dataset_license,
            original_content_owner=original_content_owner,
            original_content_license=original_content_license,
            max_episode_steps=max_episode_steps,
            validator=_package_validator(normalized_version),
        ),
        builder_name=BUILDER_NAME,
        builder_version=BUILDER_VERSION,
        build_options={
            "source_mode": "metadata",
            "suite": normalized_suite,
            "package_distribution": PACKAGE_DISTRIBUTION,
            "package_version": normalized_version,
            "environment_version": _environment_version(normalized_version),
            "max_episode_steps": max_episode_steps,
            "package_validation": "installed-benchmark-api",
            "dataset_license": dataset_license,
            "original_content_owner": original_content_owner,
            "original_content_license": original_content_license,
        },
    )


def prepare_from_package(
    *,
    suite: str,
    package_version: str = SUPPORTED_PACKAGE_VERSION,
    task_order_index: int = 0,
    dataset_name: str | None = None,
    output_root: Path | None = None,
    dataset_version: str = "v1",
    split: str = "test",
    output_format: OutputFormat | str = OutputFormat.PARQUET,
    if_exists: ExistingPolicy | str = ExistingPolicy.REUSE,
    max_episode_steps: int | None = None,
    dataset_license: str = "unverified",
    original_content_owner: str = "unverified",
    original_content_license: str = "unverified",
) -> PreparedDataset:
    """Generate canonical metadata rows from the installed LIBERO-Pro package."""

    normalized_suite = _suite_name(suite)
    normalized_version = _package_version(package_version)
    task_order_index = _nonnegative_integer(task_order_index, field="task_order_index")
    split = _nonempty_string(split, field="split")
    if max_episode_steps is not None:
        max_episode_steps = _positive_integer(max_episode_steps, field="max_episode_steps")
    dataset_license = _nonempty_string(dataset_license, field="dataset_license")
    original_content_owner = _nonempty_string(
        original_content_owner,
        field="original_content_owner",
    )
    original_content_license = _nonempty_string(
        original_content_license,
        field="original_content_license",
    )
    request = BuildRequest(
        dataset_name=dataset_name or normalized_suite,
        dataset_version=dataset_version,
        data_source=(
            f"pypi://{PACKAGE_DISTRIBUTION}/{normalized_version}/"
            f"{quote(normalized_suite, safe='')}/{task_order_index}"
        ),
        revision=normalized_version,
        splits=(split,),
        output_root=output_root or default_root(),
        output_format=OutputFormat(output_format),
        if_exists=ExistingPolicy(if_exists),
    )
    rows = _canonical_rows(
        _installed_benchmarks(normalized_version),
        suite=normalized_suite,
        task_order_index=task_order_index,
        package_version=normalized_version,
        max_episode_steps=max_episode_steps,
    )
    return prepare_dataset(
        request,
        _normalizer(
            suite=normalized_suite,
            package_version=normalized_version,
            source_uri=_LIBEROPRO_SOURCE_URI,
            source_revision=normalized_version,
            dataset_license=dataset_license,
            original_content_owner=original_content_owner,
            original_content_license=original_content_license,
            max_episode_steps=max_episode_steps,
        ),
        builder_name=BUILDER_NAME,
        builder_version=BUILDER_VERSION,
        build_options={
            "source_mode": "canonical-package",
            "suite": normalized_suite,
            "task_order_index": task_order_index,
            "package_distribution": PACKAGE_DISTRIBUTION,
            "package_version": normalized_version,
            "environment_version": _environment_version(normalized_version),
            "max_episode_steps": max_episode_steps,
            "package_validation": "rows-generated-from-installed-package",
            "dataset_license": dataset_license,
            "original_content_owner": original_content_owner,
            "original_content_license": original_content_license,
        },
        source_rows={split: rows},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data-source",
        help="Hub repo id or local metadata-only .parquet/.jsonl/.json source.",
    )
    source.add_argument(
        "--from-installed-package",
        action="store_true",
        help="Generate canonical rows from the installed rpent-liberopro package.",
    )
    parser.add_argument("--suite", required=True, choices=LIBEROPRO_SUITES)
    parser.add_argument(
        "--package-version",
        default=SUPPORTED_PACKAGE_VERSION,
        choices=[SUPPORTED_PACKAGE_VERSION],
    )
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--revision", help="Pin a Hub commit; refused for local sources.")
    parser.add_argument(
        "--task-order-index",
        type=int,
        help="LIBERO-Pro task-order index; canonical package mode only.",
    )
    parser.add_argument(
        "--splits",
        help="Comma-separated metadata source splits; defaults to test.",
    )
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--output-root", type=Path, default=default_root())
    parser.add_argument(
        "--format",
        choices=[item.value for item in OutputFormat],
        default="parquet",
    )
    parser.add_argument(
        "--if-exists",
        choices=[item.value for item in ExistingPolicy],
        default="reuse",
    )
    parser.add_argument("--dataset-license", default="unverified")
    parser.add_argument("--original-content-owner", default="unverified")
    parser.add_argument("--original-content-license", default="unverified")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    splits = (
        None
        if args.splits is None
        else tuple(split.strip() for split in args.splits.split(",") if split.strip())
    )
    if args.from_installed_package:
        if args.revision is not None or args.splits is not None:
            parser.error("--revision/--splits are only valid with --data-source")
        prepared = prepare_from_package(
            suite=args.suite,
            package_version=args.package_version,
            task_order_index=0 if args.task_order_index is None else args.task_order_index,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            output_root=args.output_root,
            output_format=args.format,
            if_exists=args.if_exists,
            max_episode_steps=args.max_episode_steps,
            dataset_license=args.dataset_license,
            original_content_owner=args.original_content_owner,
            original_content_license=args.original_content_license,
        )
    else:
        if args.task_order_index is not None:
            parser.error("--task-order-index is only valid with --from-installed-package")
        prepared = prepare(
            data_source=args.data_source,
            suite=args.suite,
            package_version=args.package_version,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            revision=args.revision,
            splits=splits,
            output_root=args.output_root,
            output_format=args.format,
            if_exists=args.if_exists,
            max_episode_steps=args.max_episode_steps,
            dataset_license=args.dataset_license,
            original_content_owner=args.original_content_owner,
            original_content_license=args.original_content_license,
        )
    print(prepared.root)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())
