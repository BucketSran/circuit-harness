"""Prepare LIBERO task metadata with AlphaApollo's shared dataset contracts.

The metadata compatibility path accepts JSON, JSONL, Parquet, or Hugging Face
rows. The canonical checkout path generates rows from LIBERO's official
``get_benchmark_dict()`` API. One row identifies a task and initial-state
selection; demonstration fields are optional provenance. Neither path reads
HDF5 trajectories or starts a simulator. ``core.py`` and ``io.py`` remain the
owners of identity, validation, public/private separation, manifests, integrity
checks, and atomic writes.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
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
    canonical_json,
    prepare_dataset,
    raw_digest,
    sha256_hex,
    task_uid,
)
from alphaapollo.data_preprocess.io import default_root, resolve_hub_revision

__all__ = ["LIBERO_SUITES", "prepare", "prepare_from_checkout"]


BUILDER_NAME = "libero"
BUILDER_VERSION = "7"
_LIBERO_SOURCE_URI = "https://github.com/Lifelong-Robot-Learning/LIBERO"
_LIBERO_IMPORT_LOCK = threading.Lock()
LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
)

_DEFAULT_SPLIT = {
    "libero_spatial": "test",
    "libero_object": "test",
    "libero_goal": "test",
    "libero_90": "train",
    "libero_10": "test",
}
_PRIMARY_INSTRUCTION_KEYS = (
    "language",
    "task_language",
    "instruction",
    "language_instruction",
)
_SECONDARY_INSTRUCTION_KEYS = (
    "problem",
    "question",
    "prompt",
)
_INSTRUCTION_KEYS = _PRIMARY_INSTRUCTION_KEYS + _SECONDARY_INSTRUCTION_KEYS
_OFFICIAL_PROBLEM_LABEL = "Libero"
_TASK_ID_KEYS = ("task_id", "task_uid", "id")
_ROW_ID_KEYS = ("source_record_id",)
_SUITE_KEYS = ("suite", "benchmark_suite", "category")
_DEMONSTRATION_ID_KEYS = ("demonstration_id", "demo_id", "episode_id")
_DEMONSTRATION_PATH_KEYS = ("demonstration_path", "demo_path")
_TASK_NAME_KEYS = ("task_name", "name")
_PROBLEM_FOLDER_KEYS = ("problem_folder",)
_BDDL_KEYS = ("bddl_file", "bddl_file_name")
_INIT_STATES_KEYS = ("init_states_file", "initial_states_file")
_MAX_EPISODE_STEPS_KEYS = ("max_episode_steps",)


def _present_aliases(
    raw: Mapping[str, Any],
    names: Sequence[str],
) -> list[tuple[str, Any]]:
    return [(name, raw[name]) for name in names if name in raw and raw[name] is not None]


def _resolve_string_aliases(
    raw: Mapping[str, Any],
    names: Sequence[str],
    label: str,
) -> tuple[str, str]:
    present = _present_aliases(raw, names)
    if not present:
        raise ValueError(f"LIBERO row is missing {label}; tried {list(names)}")
    normalized = [(name, _nonempty_string(value, field=label)) for name, value in present]
    values = {value for _, value in normalized}
    if len(values) > 1:
        fields = ", ".join(name for name, _ in normalized)
        raise ValueError(
            f"LIBERO {label} is ambiguous across fields: {fields}; "
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
            f"LIBERO {label} is ambiguous across fields: {fields}; "
            "provide one value or matching aliases"
        )
    return normalized[0]


def _normalize_id(value: Any) -> str:
    if (
        isinstance(value, bool)
        or isinstance(value, (Mapping, Sequence))
        and not isinstance(value, (str, bytes))
    ):
        raise ValueError("LIBERO task id must be a string or scalar")
    record_id = str(value).strip()
    if not record_id:
        raise ValueError("LIBERO task id must be non-empty")
    return record_id


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
            "LIBERO task id is ambiguous across fields: "
            f"{fields}; provide one value or matching aliases"
        )
    return normalized[0]


def _instruction(raw: Mapping[str, Any]) -> tuple[str, str, frozenset[str]]:
    """Resolve an instruction while recognizing official LIBERO metadata.

    Official ``Task`` records carry ``language`` alongside
    ``problem="Libero"``.  The latter is a benchmark label, not an
    instruction.  Older exports used ``problem`` as a generic instruction
    column, so it remains a fallback only when no high-confidence instruction
    alias is present.
    """

    primary = _present_aliases(raw, _PRIMARY_INSTRUCTION_KEYS)
    secondary = _present_aliases(raw, _SECONDARY_INSTRUCTION_KEYS)
    consumed: set[str] = set()

    if primary:
        normalized_primary = [
            (name, _nonempty_string(value, field="task instruction")) for name, value in primary
        ]
        values = {value for _, value in normalized_primary}
        if len(values) > 1:
            fields = ", ".join(name for name, _ in normalized_primary)
            raise ValueError(
                "LIBERO task instruction is ambiguous across fields: "
                f"{fields}; provide one value or matching aliases"
            )
        instruction_key, instruction = normalized_primary[0]
        consumed.update(name for name, _ in normalized_primary)

        for name, value in secondary:
            normalized = _nonempty_string(value, field="task instruction")
            if name == "problem" and normalized == _OFFICIAL_PROBLEM_LABEL:
                # Preserve the official benchmark label in source metadata.
                continue
            if normalized != instruction:
                fields = ", ".join([instruction_key, name])
                raise ValueError(
                    "LIBERO task instruction is ambiguous across fields: "
                    f"{fields}; provide one value or matching aliases"
                )
            consumed.add(name)
        return instruction_key, instruction, frozenset(consumed)

    fallback: list[tuple[str, str]] = []
    for name, value in secondary:
        normalized = _nonempty_string(value, field="task instruction")
        if name == "problem" and normalized == _OFFICIAL_PROBLEM_LABEL:
            # This is metadata, not a usable instruction.  If no other
            # instruction alias exists, the missing-instruction error below
            # remains explicit rather than silently using the label.
            continue
        fallback.append((name, normalized))
    if not fallback:
        raise ValueError(f"LIBERO row is missing task instruction; tried {list(_INSTRUCTION_KEYS)}")
    values = {value for _, value in fallback}
    if len(values) > 1:
        fields = ", ".join(name for name, _ in fallback)
        raise ValueError(
            "LIBERO task instruction is ambiguous across fields: "
            f"{fields}; provide one value or matching aliases"
        )
    instruction_key, instruction = fallback[0]
    consumed.update(name for name, _ in fallback)
    return instruction_key, instruction, frozenset(consumed)


def _optional_present(
    raw: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[str | None, Any | None]:
    for name in names:
        if name in raw and raw[name] is not None:
            return name, raw[name]
    return None, None


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LIBERO {field} must be a non-empty string")
    return value.strip()


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"LIBERO {field} must be a non-negative integer")
    if isinstance(value, int):
        converted = value
    elif isinstance(value, str) and value.strip().isdigit():
        converted = int(value.strip())
    else:
        raise ValueError(f"LIBERO {field} must be a non-negative integer")
    if converted < 0:
        raise ValueError(f"LIBERO {field} must be a non-negative integer")
    return converted


def _positive_integer(value: Any, *, field: str) -> int:
    converted = _nonnegative_integer(value, field=field)
    if converted == 0:
        raise ValueError(f"LIBERO {field} must be a positive integer")
    return converted


def _suite_name(value: Any) -> str:
    suite = _nonempty_string(value, field="suite")
    if suite not in LIBERO_SUITES:
        raise ValueError(
            f"unsupported LIBERO suite {suite!r}; expected one of {list(LIBERO_SUITES)}"
        )
    return suite


def _row_suite(raw: Mapping[str, Any], configured_suite: str) -> tuple[str, str | None]:
    source_key, value = _resolve_optional_string_aliases(raw, _SUITE_KEYS, "suite")
    if source_key is None or value is None:
        return configured_suite, None
    row_suite = _suite_name(value)
    if row_suite != configured_suite:
        raise ValueError(
            f"LIBERO row suite {row_suite!r} does not match configured suite {configured_suite!r}"
        )
    return row_suite, source_key


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _libero_import_root(checkout_path: Path) -> Path:
    """Find the import root containing ``libero/libero/benchmark``."""

    for candidate in (checkout_path, *checkout_path.parents):
        if (candidate / "libero" / "libero" / "benchmark").is_dir():
            return candidate
    raise ValueError(
        f"checkout_path {str(checkout_path)!r} does not contain the official "
        "LIBERO package (expected libero/libero/benchmark)"
    )


@contextmanager
def _checkout_config(import_root: Path) -> Iterator[None]:
    """Give one LIBERO checkout a deterministic, non-interactive config."""

    package_root = import_root / "libero" / "libero"
    config = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(package_root.parent / "datasets"),
        "assets": str(package_root / "assets"),
    }
    previous = os.environ.get("LIBERO_CONFIG_PATH")
    with tempfile.TemporaryDirectory(prefix=".alphaapollo-libero-config-") as config_root:
        config_path = Path(config_root) / "config.yaml"
        config_path.write_text(canonical_json(config) + "\n", encoding="utf-8")
        os.environ["LIBERO_CONFIG_PATH"] = config_root
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("LIBERO_CONFIG_PATH", None)
            else:
                os.environ["LIBERO_CONFIG_PATH"] = previous


@contextmanager
def _benchmark_api(
    checkout_path: Path | str | None,
) -> Iterator[Any]:
    """Yield the official benchmark API, optionally isolated to one checkout.

    LIBERO is an optional dependency and is intentionally imported only for the
    canonical checkout path or explicit cross-validation. When a checkout path
    is supplied, any already-imported ``libero`` modules are temporarily
    removed so an installed package from a different checkout cannot satisfy
    the request by accident. The lock covers the complete import session:
    checkout selection temporarily changes process-global import and
    configuration state, so two sessions must not overlap.
    """

    with _LIBERO_IMPORT_LOCK:
        yield from _benchmark_api_locked(checkout_path)


def _benchmark_api_locked(
    checkout_path: Path | str | None,
) -> Iterator[Any]:
    """Run one benchmark import session while ``_LIBERO_IMPORT_LOCK`` is held."""

    if checkout_path is None:
        if "libero.libero.benchmark" not in sys.modules:
            config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser()
            if not (config_root / "config.yaml").is_file():
                raise RuntimeError(
                    "LIBERO preparation needs a configured official checkout; pass "
                    f"checkout_path (searched {config_root / 'config.yaml'}) or set "
                    "LIBERO_CONFIG_PATH to a directory containing config.yaml"
                )
        try:
            module = importlib.import_module("libero.libero.benchmark")
        except ImportError as exc:
            raise RuntimeError(
                "LIBERO preparation requires the optional 'libero' dependency; pass "
                "checkout_path to use an official checkout without installing it "
                "(see alphaapollo/data_preprocess/robotics/UPSTREAM.md)"
            ) from exc
        get_benchmark_dict = getattr(module, "get_benchmark_dict", None)
        if not callable(get_benchmark_dict):
            raise RuntimeError("the installed LIBERO checkout exposes no get_benchmark_dict API")
        yield get_benchmark_dict
        return

    root = Path(checkout_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"checkout_path must be an existing directory, got {root}")
    import_root = _libero_import_root(root)
    previous_path = list(sys.path)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_pycache_prefix = sys.pycache_prefix
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "libero" or name.startswith("libero.")
    }
    with tempfile.TemporaryDirectory(prefix=".alphaapollo-libero-pycache-") as pycache_root:
        try:
            # A clean cache prefix prevents both reading stale checkout bytecode
            # and writing new __pycache__ files into the selected checkout.
            sys.dont_write_bytecode = True
            sys.pycache_prefix = pycache_root
            for name in saved_modules:
                sys.modules.pop(name, None)
            sys.path.insert(0, str(import_root))
            with _checkout_config(import_root):
                try:
                    module = importlib.import_module("libero.libero.benchmark")
                except ImportError as exc:
                    raise RuntimeError(
                        f"could not import the official LIBERO benchmark API from {root}: {exc}"
                    ) from exc
                module_file = getattr(module, "__file__", None)
                if module_file is None or not _within(Path(module_file), import_root):
                    raise RuntimeError(
                        f"LIBERO benchmark API was not loaded from checkout_path {root}"
                    )
                get_benchmark_dict = getattr(module, "get_benchmark_dict", None)
                if not callable(get_benchmark_dict):
                    raise RuntimeError("the LIBERO checkout exposes no get_benchmark_dict API")
                yield get_benchmark_dict
        finally:
            for name in list(sys.modules):
                if name == "libero" or name.startswith("libero."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)
            sys.path[:] = previous_path
            sys.pycache_prefix = previous_pycache_prefix
            sys.dont_write_bytecode = previous_dont_write_bytecode


def _checkout_git_revision(checkout_path: Path | str | None) -> str | None:
    if checkout_path is None:
        return None
    root = Path(checkout_path).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _benchmark_factory(benchmarks: Mapping[str, Any], suite: str) -> Any:
    for name, factory in benchmarks.items():
        if str(name).lower() == suite.lower():
            return factory
    raise ValueError(f"LIBERO suite {suite!r} is not registered in this checkout")


def _official_task_string(task: Any, attribute: str) -> str:
    value = getattr(task, attribute, None)
    return _nonempty_string(value, field=f"official task {attribute}")


def _canonical_rows(
    get_benchmark_dict: Any,
    *,
    suite: str,
    task_order_index: int,
) -> list[dict[str, Any]]:
    """Generate input rows directly from official LIBERO task metadata."""

    benchmarks = get_benchmark_dict()
    if not isinstance(benchmarks, Mapping):
        raise RuntimeError("LIBERO get_benchmark_dict() must return a mapping")
    factory = _benchmark_factory(benchmarks, suite)
    if not callable(factory):
        raise RuntimeError(f"LIBERO suite factory for {suite!r} is not callable")
    try:
        benchmark = factory(task_order_index)
    except Exception as exc:
        raise ValueError(
            f"LIBERO task_order_index {task_order_index} is not available in suite {suite!r}: {exc}"
        ) from exc
    get_num_tasks = getattr(benchmark, "get_num_tasks", None)
    get_task = getattr(benchmark, "get_task", None)
    get_init_states = getattr(benchmark, "get_task_init_states", None)
    if not all(callable(item) for item in (get_num_tasks, get_task, get_init_states)):
        raise RuntimeError(
            f"LIBERO suite {suite!r} does not expose the required benchmark metadata API"
        )
    try:
        task_count = int(get_num_tasks())
    except Exception as exc:
        raise ValueError(f"LIBERO suite {suite!r} did not report a task count: {exc}") from exc
    if task_count <= 0:
        raise ValueError(f"LIBERO suite {suite!r} has no tasks")

    rows: list[dict[str, Any]] = []
    for task_index in range(task_count):
        try:
            task = get_task(task_index)
        except Exception as exc:
            raise ValueError(
                f"LIBERO task_index {task_index} is not available in suite {suite!r} "
                f"under task_order_index {task_order_index}: {exc}"
            ) from exc
        task_fields: dict[str, str] = {
            "task_name": _official_task_string(task, "name"),
            "language": _official_task_string(task, "language"),
            "problem_folder": _official_task_string(task, "problem_folder"),
            "bddl_file": _official_task_string(task, "bddl_file"),
            "init_states_file": _official_task_string(task, "init_states_file"),
        }
        problem = getattr(task, "problem", None)
        if problem is not None:
            task_fields["problem"] = _nonempty_string(problem, field="official task problem")
        try:
            initial_states = get_init_states(task_index)
            initial_state_count = len(initial_states)
        except Exception as exc:
            raise ValueError(
                f"LIBERO initial states are unavailable for suite {suite!r}, "
                f"task_index {task_index}: {exc}"
            ) from exc
        if initial_state_count <= 0:
            raise ValueError(
                f"LIBERO task_index {task_index} in suite {suite!r} has no initial states"
            )

        demonstration_path: str | None = None
        get_demonstration = getattr(benchmark, "get_task_demonstration", None)
        if callable(get_demonstration):
            try:
                demonstration_path = _nonempty_string(
                    get_demonstration(task_index),
                    field="official demonstration path",
                )
            except (AssertionError, FileNotFoundError, IndexError, OSError):
                # Demonstrations are provenance, not required execution input.
                demonstration_path = None

        for initial_state_index in range(initial_state_count):
            row: dict[str, Any] = {
                # A canonical row is one task and one initial state, so those
                # fields alone identify it. Stating the id here keeps identity
                # independent of whether the demonstration HDF5s happen to be
                # downloaded: LIBERO ships datasets separately, and folding an
                # optional provenance field into task_uid would stop prepared
                # builds from joining across machines.
                "source_record_id": _compose_identity(
                    [
                        suite,
                        str(task_order_index),
                        str(task_index),
                        task_fields["task_name"],
                        str(initial_state_index),
                    ]
                ),
                "suite": suite,
                "task_order_index": task_order_index,
                "task_index": task_index,
                "task_name": task_fields["task_name"],
                "language": task_fields["language"],
                "problem_folder": task_fields["problem_folder"],
                "bddl_file": task_fields["bddl_file"],
                "init_states_file": task_fields["init_states_file"],
                "initial_state_index": initial_state_index,
            }
            if "problem" in task_fields:
                row["problem"] = task_fields["problem"]
            if demonstration_path is not None:
                row["demonstration_path"] = demonstration_path
            rows.append(row)
    return rows


def _source_record_id(
    raw: Mapping[str, Any],
    *,
    suite: str,
    task_order_index: int,
    task_index: int,
    task_name: str,
    demonstration_id: str | None,
    demonstration_path: str | None,
    initial_state_index: int,
) -> tuple[str, str]:
    row_id_key, row_id = _resolve_id_aliases(raw, _ROW_ID_KEYS)
    task_id_key, task_id = _resolve_id_aliases(raw, _TASK_ID_KEYS)
    if row_id_key is not None and row_id is not None:
        # ``source_record_id`` is the explicit row-level contract. If it is
        # repeated, the shared build pipeline rejects the duplicate task_uid;
        # unlike task_id aliases, it must not be silently reinterpreted.
        return row_id, row_id_key

    if task_id_key is not None and task_id is not None:
        parts = [task_id]
        demo_component = _demo_identity_component(
            demonstration_id=demonstration_id,
            demonstration_path=demonstration_path,
        )
        if demo_component is not None:
            parts.append(demo_component)
        parts.append(str(initial_state_index))
        return _compose_identity(parts), task_id_key

    parts = [str(suite), str(task_order_index), str(task_index), task_name]
    demo_component = _demo_identity_component(
        demonstration_id=demonstration_id,
        demonstration_path=demonstration_path,
    )
    if demo_component is not None:
        parts.append(demo_component)
    parts.append(str(initial_state_index))
    return _compose_identity(parts), "derived"


def _compose_identity(parts: Sequence[str]) -> str:
    """Join identity components without allowing delimiter collisions."""

    return ":".join(quote(part, safe="") for part in parts)


def _demo_identity_component(
    *,
    demonstration_id: str | None,
    demonstration_path: str | None,
) -> str | None:
    if demonstration_id is not None:
        # Keep explicit ids and opaque path tokens in separate namespaces.
        return f"demo-id-{demonstration_id}"
    if demonstration_path is None:
        return None
    # Keep the raw path private. The opaque token still distinguishes
    # path-only demonstrations without putting a machine/user path in the
    # public source_record_id.
    return f"demo-path-{sha256_hex(demonstration_path)}"


def _validate_row_environment_version(
    raw: Mapping[str, Any],
    *,
    environment_version: str,
) -> None:
    raw_version = raw.get("environment_version")
    if raw_version is None:
        return
    row_version = _nonempty_string(raw_version, field="environment_version")
    if row_version != environment_version:
        raise ValueError(
            f"LIBERO row environment_version {row_version!r} does not match "
            f"configured environment_version {environment_version!r}"
        )


def _checkout_validator(*, benchmarks: Mapping[str, Any] | None = None):
    """Cross-check each row against the installed LIBERO checkout, fail-closed.

    This mirrors the reset-time validation the in-process backend performs
    (task_name/problem_folder/bddl_file/init_states_file per index, exact
    instruction wording, initial-state range): any drift between the metadata
    source and the checkout would otherwise pass preparation silently and
    refuse 100% of episodes at run time — the failure mode that cost the
    RoboCasa snapshot 133/370 wrong instructions in #256. The checkout cannot
    self-report a version, so the caller asserts that the installed checkout
    is the one --environment-version names.

    Without ``benchmarks`` the installed checkout is imported on the first row
    rather than here. ``prepare_dataset`` answers ``if_exists='error'`` and an
    up-to-date ``if_exists='reuse'`` from the destination alone, without
    normalizing a row; importing eagerly would make an optional dependency
    mandatory for calls that never read the source.
    """

    if benchmarks is not None and not isinstance(benchmarks, Mapping):
        raise RuntimeError("LIBERO get_benchmark_dict() must return a mapping")
    resolved: dict[str, Mapping[str, Any]] = {} if benchmarks is None else {"api": benchmarks}
    suites: dict[tuple[str, int], Any] = {}
    init_counts: dict[tuple[str, int, int], int] = {}

    def installed_benchmarks() -> Mapping[str, Any]:
        if "api" not in resolved:
            with _benchmark_api(None) as get_benchmark_dict:
                loaded = get_benchmark_dict()
            if not isinstance(loaded, Mapping):
                raise RuntimeError("LIBERO get_benchmark_dict() must return a mapping")
            resolved["api"] = loaded
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
        factory = _benchmark_factory(installed_benchmarks(), suite)
        suite_key = (suite, task_order_index)
        suite_obj = suites.get(suite_key)
        if suite_obj is None:
            try:
                suite_obj = factory(task_order_index)
            except Exception as exc:
                raise ValueError(
                    f"LIBERO task_order_index {task_order_index} is not available in "
                    f"suite {suite!r}: {exc}"
                ) from exc
            suites[suite_key] = suite_obj
        try:
            task = suite_obj.get_task(task_index)
        except Exception as exc:
            raise ValueError(
                f"LIBERO task_index {task_index} is not available in suite {suite!r} "
                f"under task_order_index {task_order_index}: {exc}"
            ) from exc
        for field, expected, attribute in (
            ("task_name", task_name, "name"),
            ("problem_folder", problem_folder, "problem_folder"),
            ("bddl_file", bddl_file, "bddl_file"),
            ("init_states_file", init_states_file, "init_states_file"),
        ):
            actual = getattr(task, attribute, None)
            if actual is None:
                raise ValueError(
                    f"checkout task exposes no {attribute!r}; cannot verify the row's {field}"
                )
            if actual != expected:
                raise ValueError(
                    f"LIBERO {field} mismatch against the installed checkout: "
                    f"row has {expected!r}, checkout has {actual!r}"
                )
        language = getattr(task, "language", None)
        if not isinstance(language, str) or not language.strip():
            raise ValueError(
                "checkout task exposes no language; cannot verify the row's instruction"
            )
        if language.strip() != instruction.strip():
            raise ValueError(
                "LIBERO instruction does not match the checkout task language: "
                f"row has {instruction!r}, checkout has {language!r}"
            )
        count_key = (suite, task_order_index, task_index)
        count = init_counts.get(count_key)
        if count is None:
            try:
                count = len(suite_obj.get_task_init_states(task_index))
            except Exception as exc:
                raise ValueError(
                    f"LIBERO initial states are unavailable for suite {suite!r}, "
                    f"task_order_index {task_order_index}, task_index {task_index}: {exc}"
                ) from exc
            init_counts[count_key] = count
        if initial_state_index >= count:
            raise ValueError(
                f"LIBERO initial_state_index {initial_state_index} is out of range for "
                f"{count} prepared initial states"
            )

    return validate


def _normalizer(
    *,
    suite: str,
    environment_version: str,
    source_uri: str,
    source_revision: str,
    dataset_license: str,
    original_content_owner: str,
    original_content_license: str,
    validator=None,
):
    def normalize(raw: Mapping[str, Any], context: NormalizeContext) -> PreparedExample:
        row_suite, suite_key = _row_suite(raw, suite)
        instruction_key, instruction, instruction_keys = _instruction(raw)
        _task_name_key, task_name_value = _resolve_string_aliases(raw, _TASK_NAME_KEYS, "task name")
        demo_path_key, demo_path_value = _resolve_optional_string_aliases(
            raw, _DEMONSTRATION_PATH_KEYS, "demonstration path"
        )
        _problem_folder_key, problem_folder_value = _resolve_string_aliases(
            raw, _PROBLEM_FOLDER_KEYS, "problem folder"
        )
        _bddl_key, bddl_value = _resolve_string_aliases(raw, _BDDL_KEYS, "BDDL file")
        _init_states_key, init_states_value = _resolve_string_aliases(
            raw, _INIT_STATES_KEYS, "initial-states file"
        )

        task_name = task_name_value
        # Demonstrations are provenance, not an execution requirement: an
        # evaluation-only source (task x initial state) has none, and the
        # backend treats both fields as optional strings.
        demonstration_path = demo_path_value
        problem_folder = problem_folder_value
        bddl_file = bddl_value
        init_states_file = init_states_value
        _validate_row_environment_version(raw, environment_version=environment_version)
        task_order_index = _nonnegative_integer(
            raw.get("task_order_index", 0), field="task_order_index"
        )
        task_index = _nonnegative_integer(raw.get("task_index"), field="task_index")
        initial_state_index = _nonnegative_integer(
            raw.get("initial_state_index", 0), field="initial_state_index"
        )
        max_episode_steps_key, max_episode_steps_value = _optional_present(
            raw, _MAX_EPISODE_STEPS_KEYS
        )
        max_episode_steps = (
            None
            if max_episode_steps_key is None
            else _positive_integer(max_episode_steps_value, field="max_episode_steps")
        )
        demo_id_key, demonstration_id = _resolve_optional_string_aliases(
            raw, _DEMONSTRATION_ID_KEYS, "demonstration id"
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
            demonstration_id=demonstration_id,
            demonstration_path=demonstration_path,
            initial_state_index=initial_state_index,
        )
        consumed_keys = {
            *instruction_keys,
            *_TASK_ID_KEYS,
            *_ROW_ID_KEYS,
            *_SUITE_KEYS,
            *_DEMONSTRATION_ID_KEYS,
            *_DEMONSTRATION_PATH_KEYS,
            *_TASK_NAME_KEYS,
            *_PROBLEM_FOLDER_KEYS,
            *_BDDL_KEYS,
            *_INIT_STATES_KEYS,
            *_MAX_EPISODE_STEPS_KEYS,
            "task_order_index",
            "task_index",
            "initial_state_index",
            "environment_version",
        }
        # ``source_uri`` and ``source_revision`` are deliberately absent: a row
        # cannot restate build-level provenance. Both stay in
        # ``extra["source_metadata"]`` as upstream claims while the recorded
        # provenance remains the URI this build actually resolved and read.
        source_metadata = {key: value for key, value in raw.items() if key not in consumed_keys}
        backend_metadata = {
            "suite": row_suite,
            "task_order_index": task_order_index,
            "task_index": task_index,
            "task_name": task_name,
            "problem_folder": problem_folder,
            "bddl_file": bddl_file,
            "init_states_file": init_states_file,
            "initial_state_index": initial_state_index,
        }
        if demonstration_id is not None:
            backend_metadata["demonstration_id"] = demonstration_id
        if demonstration_path is not None:
            backend_metadata["demonstration_path"] = demonstration_path
        if max_episode_steps is not None:
            backend_metadata["max_episode_steps"] = max_episode_steps

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
                "benchmark": "libero",
                "environment_version": environment_version,
                **backend_metadata,
            },
            extra={
                "source_instruction_key": instruction_key,
                "source_id_key": id_key,
                "source_suite_key": suite_key or "configured",
                "source_demo_id_key": demo_id_key or demo_path_key,
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
    environment_version: str,
    checkout_path: Path | str | None = None,
    dataset_name: str | None = None,
    output_root: Path | None = None,
    dataset_version: str = "v1",
    revision: str | None = None,
    splits: Sequence[str] | None = None,
    output_format: OutputFormat | str = OutputFormat.PARQUET,
    if_exists: ExistingPolicy | str = ExistingPolicy.REUSE,
    dataset_license: str = "unverified",
    original_content_owner: str = "unverified",
    original_content_license: str = "unverified",
) -> PreparedDataset:
    """Prepare one LIBERO suite without importing or mutating the simulator.

    Rows are cross-checked against an official checkout: ``checkout_path`` when
    given, otherwise the installed ``libero`` package. The installed package is
    imported only once a row is actually normalized, so an ``if_exists``
    decision answerable from the destination alone needs no LIBERO install.
    """

    normalized_suite = _suite_name(suite)
    environment_version = _nonempty_string(
        environment_version,
        field="environment_version",
    )
    dataset_license = _nonempty_string(dataset_license, field="dataset_license")
    original_content_owner = _nonempty_string(
        original_content_owner,
        field="original_content_owner",
    )
    original_content_license = _nonempty_string(
        original_content_license,
        field="original_content_license",
    )
    split_names = tuple(splits) if splits is not None else (_DEFAULT_SPLIT[normalized_suite],)
    local_source = Path(data_source).expanduser().exists()
    if revision is not None and local_source:
        raise ValueError(
            f"revision {revision!r} was given for the local source {data_source!r}; a local "
            "file has no Hub commit and is fingerprinted by its content, so recording the "
            "revision would claim provenance the build does not have"
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

    # An official checkout is the source of truth for executable task metadata.
    # Accepting unchecked local/Hub metadata would defer a mismatch until reset
    # time, where every episode can fail. A named checkout holds its import
    # session open for the whole build because its benchmark objects stay bound
    # to the temporarily selected sys.path; the installed package needs no
    # window and is therefore imported lazily, on the first row.
    with ExitStack() as import_session:
        if checkout_path is None:
            validator = _checkout_validator()
        else:
            get_benchmark_dict = import_session.enter_context(_benchmark_api(checkout_path))
            validator = _checkout_validator(benchmarks=get_benchmark_dict())
        return prepare_dataset(
            request,
            _normalizer(
                suite=normalized_suite,
                environment_version=environment_version,
                source_uri=source_uri,
                source_revision=revision or "local",
                dataset_license=dataset_license,
                original_content_owner=original_content_owner,
                original_content_license=original_content_license,
                validator=validator,
            ),
            builder_name=BUILDER_NAME,
            builder_version=BUILDER_VERSION,
            build_options={
                "suite": normalized_suite,
                "environment_version": environment_version,
                "checkout_validation": "official-libero-benchmark-api",
                "dataset_license": dataset_license,
                "original_content_owner": original_content_owner,
                "original_content_license": original_content_license,
            },
        )


def prepare_from_checkout(
    *,
    checkout_path: Path | str | None,
    suite: str,
    environment_version: str,
    source_revision: str | None = None,
    task_order_index: int = 0,
    dataset_name: str | None = None,
    output_root: Path | None = None,
    dataset_version: str = "v1",
    split: str | None = None,
    output_format: OutputFormat | str = OutputFormat.PARQUET,
    if_exists: ExistingPolicy | str = ExistingPolicy.REUSE,
    dataset_license: str = "unverified",
    original_content_owner: str = "unverified",
    original_content_license: str = "unverified",
) -> PreparedDataset:
    """Prepare rows generated directly from an official LIBERO checkout.

    This is the canonical path for executable LIBERO data. It does not accept
    caller-supplied task metadata: task name, language, BDDL, initial-state
    count, and demonstration path all come from ``get_benchmark_dict()``.
    Cross-checking those rows against the same API would compare each field to
    the object it was read from, so this path states its rows are generated
    rather than validated and skips the second benchmark load. The simulator is
    never started.
    """

    normalized_suite = _suite_name(suite)
    environment_version = _nonempty_string(
        environment_version,
        field="environment_version",
    )
    task_order_index = _nonnegative_integer(task_order_index, field="task_order_index")
    split_name = (
        _DEFAULT_SPLIT[normalized_suite]
        if split is None
        else _nonempty_string(split, field="split")
    )
    dataset_license = _nonempty_string(dataset_license, field="dataset_license")
    original_content_owner = _nonempty_string(
        original_content_owner,
        field="original_content_owner",
    )
    original_content_license = _nonempty_string(
        original_content_license,
        field="original_content_license",
    )
    resolved_revision = source_revision or _checkout_git_revision(checkout_path)
    if resolved_revision is None:
        raise ValueError(
            "canonical LIBERO preparation requires an explicit source_revision when "
            "checkout_path is not a Git checkout"
        )
    resolved_revision = _nonempty_string(resolved_revision, field="source_revision")
    source_id = dataset_name or normalized_suite
    source_name = (
        "libero://checkout/"
        f"{quote(resolved_revision, safe='')}/"
        f"{quote(normalized_suite, safe='')}/"
        f"{task_order_index}"
    )
    request = BuildRequest(
        dataset_name=source_id,
        dataset_version=dataset_version,
        data_source=source_name,
        revision=resolved_revision,
        splits=(split_name,),
        output_root=output_root or default_root(),
        output_format=OutputFormat(output_format),
        if_exists=ExistingPolicy(if_exists),
    )

    # The import session covers row generation only. Everything after it --
    # normalization, file writes, the manifest, the atomic commit, and the lazy
    # pyarrow/datasets imports -- runs with the ordinary sys.path and no
    # pycache prefix pointing at a directory this scope is about to delete.
    with _benchmark_api(checkout_path) as get_benchmark_dict:
        rows = _canonical_rows(
            get_benchmark_dict,
            suite=normalized_suite,
            task_order_index=task_order_index,
        )

    return prepare_dataset(
        request,
        _normalizer(
            suite=normalized_suite,
            environment_version=environment_version,
            source_uri=_LIBERO_SOURCE_URI,
            source_revision=resolved_revision,
            dataset_license=dataset_license,
            original_content_owner=original_content_owner,
            original_content_license=original_content_license,
        ),
        builder_name=BUILDER_NAME,
        builder_version=BUILDER_VERSION,
        build_options={
            "source_mode": "canonical",
            "suite": normalized_suite,
            "task_order_index": task_order_index,
            "environment_version": environment_version,
            "source_revision": resolved_revision,
            "checkout_validation": "rows-generated-from-checkout",
            "dataset_license": dataset_license,
            "original_content_owner": original_content_owner,
            "original_content_license": original_content_license,
        },
        source_rows={split_name: rows},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        help="Hub repo id or local metadata-only .parquet/.jsonl/.json source.",
    )
    parser.add_argument(
        "--checkout-path",
        type=Path,
        help=(
            "Official LIBERO checkout. On its own it is the canonical metadata "
            "source; with --data-source it is the checkout those rows are "
            "validated against instead of an installed libero package."
        ),
    )
    parser.add_argument("--suite", required=True, choices=LIBERO_SUITES)
    parser.add_argument(
        "--environment-version",
        required=True,
        help="Official LIBERO checkout or release identity used to execute these tasks.",
    )
    parser.add_argument(
        "--dataset-name",
        help="Prepared dataset name; defaults to the selected LIBERO suite.",
    )
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--revision", help="Pin a Hub commit; refused for local sources.")
    parser.add_argument(
        "--source-revision",
        help="Canonical checkout revision; omitted, a Git checkout's HEAD is used.",
    )
    parser.add_argument(
        "--task-order-index",
        type=int,
        help="Official LIBERO task-order index for canonical checkout mode; defaults to 0.",
    )
    parser.add_argument(
        "--splits",
        help="Comma-separated source splits. Defaults to train for libero_90 and test otherwise.",
    )
    parser.add_argument("--output-root", type=Path, default=default_root())
    parser.add_argument(
        "--format", choices=[item.value for item in OutputFormat], default="parquet"
    )
    parser.add_argument(
        "--if-exists", choices=[item.value for item in ExistingPolicy], default="reuse"
    )
    parser.add_argument("--dataset-license", default="unverified")
    parser.add_argument("--original-content-owner", default="unverified")
    parser.add_argument("--original-content-license", default="unverified")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    splits = (
        None
        if args.splits is None
        else tuple(split.strip() for split in args.splits.split(",") if split.strip())
    )
    if args.data_source is None and args.checkout_path is None:
        _parser().error("one of --data-source or --checkout-path is required")
    if args.data_source is None:
        if args.revision is not None:
            _parser().error("--revision is only valid with --data-source")
        if args.splits is not None and len(splits or ()) != 1:
            _parser().error("canonical checkout mode accepts exactly one --splits value")
        prepared = prepare_from_checkout(
            checkout_path=args.checkout_path,
            suite=args.suite,
            environment_version=args.environment_version,
            source_revision=args.source_revision,
            task_order_index=(0 if args.task_order_index is None else args.task_order_index),
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            split=(splits or (None,))[0],
            output_root=args.output_root,
            output_format=args.format,
            if_exists=args.if_exists,
            dataset_license=args.dataset_license,
            original_content_owner=args.original_content_owner,
            original_content_license=args.original_content_license,
        )
    else:
        if args.source_revision is not None or args.task_order_index is not None:
            _parser().error(
                "--source-revision/--task-order-index are only valid in canonical "
                "checkout mode: pass --checkout-path without --data-source"
            )
        prepared = prepare(
            data_source=args.data_source,
            suite=args.suite,
            environment_version=args.environment_version,
            checkout_path=args.checkout_path,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            revision=args.revision,
            splits=splits,
            output_root=args.output_root,
            output_format=args.format,
            if_exists=args.if_exists,
            dataset_license=args.dataset_license,
            original_content_owner=args.original_content_owner,
            original_content_license=args.original_content_license,
        )
    print(prepared.root)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())
