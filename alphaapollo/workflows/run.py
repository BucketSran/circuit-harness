# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-run and repeated/resumable orchestration for configured Workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from alphaapollo.workflows.config import (
    ConfigError,
    RunConfig,
    coerce_config,
    compose_config_mapping,
    config_digest,
    execution_label,
    parse_run_config,
    run_config_for_cell,
)
from alphaapollo.workflows.config import read_config_mapping as read_config_mapping
from alphaapollo.workflows.data import (
    Cell,
    IncrementalWorkflowRecorder,
    InputDataError,
    align_public_inputs_with_private_gold,
    expand_cells,
    load_prepared_inputs,
    persist_results,
)
from alphaapollo.workflows.data import validate_inputs as _validate_injected_inputs
from alphaapollo.workflows.executor import WorkflowExecutor
from alphaapollo.workflows.memory import WorkflowMemorySession
from alphaapollo.workflows.records import (
    RunRecord,
    ScoredTask,
    WorkflowInput,
    WorkflowResult,
)
from alphaapollo.workflows.report import (
    build_report,
    decoding_summary,
    every_run_failed,
    format_report_markdown,
    git_sha,
    model_label,
    resource_summary,
)
from alphaapollo.workflows.resources import compose_resources, validate_composition_config

if TYPE_CHECKING:
    from alphaapollo.workflows.memory.adapter import WorkflowMemoryAdapter

__all__ = [
    "load_prepared_inputs",
    "main",
    "read_config_mapping",
    "run",
    "run_workflow",
]


class _ArgumentExit(Exception):
    def __init__(self, status: int, message: str | None = None) -> None:
        self.status = status
        self.message = message
        super().__init__(message)


class _ReturningArgumentParser(argparse.ArgumentParser):
    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        raise _ArgumentExit(status, message)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one configured Workflow and return a stable process exit code."""

    parser = _argument_parser()
    try:
        namespace = parser.parse_args(argv)
    except _ArgumentExit as exc:
        if exc.message:
            stream = sys.stderr if exc.status else sys.stdout
            print(exc.message, end="", file=stream)
        return exc.status

    source = Path(namespace.config).resolve()
    try:
        raw = compose_config_mapping(source)
        for override in namespace.overrides:
            _apply_override(raw, override)
        config = parse_run_config(raw, base_dir=source.parent)
    except (ConfigError, InputDataError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        if _needs_repeated_run(config):
            shard = _parse_shard(namespace.shard)
            total_inputs = len(load_prepared_inputs(config))
            if config.execution.limit is not None:
                total_inputs = min(config.execution.limit, total_inputs)
            total = total_inputs * config.execution.samples
            completed = 0
            progress_lock = threading.Lock()

            def progress(record: RunRecord) -> None:
                nonlocal completed
                with progress_lock:
                    completed += 1
                    current = completed
                if not namespace.quiet:
                    tag = "resume" if record.resumed else "run"
                    mark = {True: "ok", False: "x", None: "?"}[record.correct]
                    print(
                        f"[{current}/{total}] {tag} {record.problem_id} "
                        f"seed={record.seed} correct={mark}",
                        file=sys.stderr,
                    )

            report = _run_repeated(
                config,
                workdir=namespace.workdir,
                resume=config.execution.resume and not namespace.no_resume,
                progress=progress,
                shard=shard,
                reverse=namespace.reverse,
            )
        else:
            report = None
            run(config)
    except (ConfigError, InputDataError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("workflow cancelled", file=sys.stderr)
        return 130
    except BaseException as exc:  # noqa: BLE001 - command boundary reports without traceback
        print(f"workflow failed: {exc}", file=sys.stderr)
        return 1

    # A run in which nothing ran is not a run that scored zero. Every cell ending
    # in a typed error is how an unreachable server or an unserved model name
    # arrives here, and exiting 0 beside a report of 0.000 hands that back as a
    # measurement. The report is still written: it holds the evidence.
    if report is not None and every_run_failed(report):
        print(
            "workflow failed: all runs ended in a typed error and none produced an "
            "answer, so the report's rates measure nothing; see the per-cell "
            "result.json error field",
            file=sys.stderr,
        )
        return 1
    return 0


def _needs_repeated_run(config: RunConfig) -> bool:
    """Whether this recipe takes the repeated/resumable path or one execution.

    The CLI and the programmatic entry point both have to answer this, and had to
    answer it identically: ``main`` needs it before it can size progress
    reporting, ``run`` needs it to pick the primitive. Written out twice, once as
    its own negation, they agreed only until someone added a fourth condition to
    one of them.
    """

    execution = config.execution
    return config.scoring is not None or execution.samples > 1 or execution.limit is not None


def run(
    config: RunConfig,
    *,
    inputs: Sequence[WorkflowInput] | None = None,
) -> object:
    """Run one configured pipeline, including optional scoring and reporting.

    ``run_workflow`` remains the execution primitive used by Learning and other
    programmatic consumers. This application-level function adds repetition,
    resume, optional post-execution scoring, and reporting.
    """

    if not _needs_repeated_run(config):
        return run_workflow(config, inputs=inputs)
    if inputs is not None:
        raise ConfigError("injected inputs are unsupported for repeated or scored runs")
    return _run_repeated(config)


def _run_repeated(
    config: RunConfig,
    *,
    workdir: str | Path | None = None,
    resume: bool | None = None,
    progress: ProgressFn | None = None,
    shard: tuple[int, int] | None = None,
    reverse: bool = False,
) -> dict[str, Any]:
    """Run repeated Workflow cells and write the aggregate JSON/Markdown report."""

    normalized = coerce_config(config)
    root = Path(workdir or config.output.directory)
    records = run_batch(
        normalized,
        workdir=root,
        resume=config.execution.resume if resume is None else resume,
        progress=progress,
        shard=shard,
        reverse=reverse,
    )
    solver_options, verifier_options, verifier_label = resource_summary(normalized)
    report = build_report(
        records,
        model=model_label(solver_options, verifier_label),
        config_digest=config_digest(normalized),
        seed_base=config.execution.seed,
        k=config.execution.samples,
        code_sha=git_sha(),
        decoding=decoding_summary(normalized, solver_options, verifier_options),
    )
    root.mkdir(parents=True, exist_ok=True)
    report_name = config.scoring.report_name if config.scoring is not None else "report.json"
    (root / report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "report.md").write_text(format_report_markdown(report), encoding="utf-8")
    return report


def run_workflow(
    config: RunConfig,
    *,
    inputs: Sequence[WorkflowInput] | None = None,
    memory_adapter: WorkflowMemoryAdapter | None = None,
) -> list[WorkflowResult]:
    """Compose, execute, persist, and clean up one validated Workflow run."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    if memory_adapter is not None and config.memory is None:
        raise ValueError("memory_adapter requires config.memory to be enabled")
    materialized_inputs = (
        load_prepared_inputs(config) if inputs is None else _validate_injected_inputs(inputs)
    )
    with compose_resources(config) as resources:
        incremental = IncrementalWorkflowRecorder(
            config.output.directory,
            resume=config.memory is not None and config.execution.resume,
        )
        memory: WorkflowMemorySession | None = None
        if config.memory is not None:
            memory_run_id = _workflow_memory_run_id(config, resources.workflow.name)
            memory = WorkflowMemorySession(
                config.memory,
                workflow_name=resources.workflow.name,
                run_id=memory_run_id,
                journal_path=config.output.directory / "workflow-memory.jsonl",
                adapter=memory_adapter,
            )
        if memory is not None and memory.profile.scratchpad:
            for runtime in resources.runtimes.values():
                configure = getattr(runtime, "configure_workspace_hooks", None)
                if callable(configure):
                    configure(
                        prepare=memory.prepare_agent_workspace,
                        observe=memory.observe_agent_workspace,
                    )
        executor = WorkflowExecutor(  # type: ignore[arg-type]
            resources.workflow,
            resources.runtimes,
            resources.verifiers,
            memory=memory,
            step_observer=incremental.record,
        )
        results = executor.run_batch(materialized_inputs)
        persist_results(config, results)
        return results


def _workflow_memory_run_id(config: RunConfig, workflow_name: str) -> str:
    """Reuse a durable memory identity when the same configured run resumes."""

    path = config.output.directory / "workflow-memory-run.json"
    digest = config_digest(config)
    if config.execution.resume and path.exists():
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid workflow memory run metadata: {path}") from exc
        if (
            isinstance(metadata, dict)
            and metadata.get("version") == 1
            and metadata.get("workflow_name") == workflow_name
            and metadata.get("config_digest") == digest
            and isinstance(metadata.get("run_id"), str)
            and metadata["run_id"].strip()
        ):
            return metadata["run_id"]

    run_id = uuid.uuid4().hex
    metadata = {
        "version": 1,
        "workflow_name": workflow_name,
        "config_digest": digest,
        "run_id": run_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return run_id


def _argument_parser() -> _ReturningArgumentParser:
    parser = _ReturningArgumentParser(
        prog="python -m alphaapollo.workflows.main",
        description="Run a validated AlphaApollo Reasoning Workflow.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True, help="JSON or YAML RunConfig path")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="A.B=VALUE",
        help="override one mapping value; may be repeated",
    )
    parser.add_argument("--workdir", default=None, help="override output directory for this run")
    parser.add_argument("--no-resume", action="store_true", help="ignore cached scored cells")
    parser.add_argument("--quiet", action="store_true", help="suppress per-cell progress")
    parser.add_argument("--shard", default=None, metavar="I/N", help="run cell shard I of N")
    parser.add_argument("--reverse", action="store_true", help="process cells tail-first")
    return parser


def _parse_shard(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        index_text, count_text = value.split("/", 1)
        shard = (int(index_text), int(count_text))
    except ValueError as exc:
        raise ConfigError(f"--shard expects I/N, got {value!r}") from exc
    index, count = shard
    if count < 1 or not 0 <= index < count:
        raise ConfigError(f"invalid shard {value!r}: need 0 <= I < N")
    return shard


def _apply_override(raw: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ConfigError(f"override {expression!r} must use a.b=value syntax")
    dotted, encoded = expression.split("=", 1)
    parts = dotted.split(".")
    if any(not part or not part.isidentifier() for part in parts):
        raise ConfigError(f"override path {dotted!r} must contain dotted identifiers")
    value = _parse_override_value(encoded)
    cursor: dict[str, Any] = raw
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise ConfigError(f"override path {dotted!r} crosses non-mapping key {part!r}")
        cursor = child
    cursor[parts[-1]] = value


def _parse_override_value(encoded: str) -> Any:
    try:
        import yaml
    except ImportError:
        try:
            return json.loads(encoded)
        except json.JSONDecodeError:
            return encoded
    try:
        return yaml.safe_load(encoded)
    except Exception as exc:
        raise ConfigError(f"could not parse override value {encoded!r}: {exc}") from exc


ProgressFn = Callable[["RunRecord"], None]


def supported_plan_modes() -> frozenset[str]:
    """Return plan labels representable by the canonical Workflow DSL."""

    from alphaapollo.workflows.scoring import supported_plan_modes as implementation

    return implementation()


def run_batch(
    config: object,
    *,
    workdir: str | Path,
    problems: list[ScoredTask] | None = None,
    tool_executor_factory: object | None = None,
    resume: bool = True,
    progress: ProgressFn | None = None,
    shard: tuple[int, int] | None = None,
    reverse: bool = False,
) -> list[RunRecord]:
    """Run ``inputs x samples`` through canonical Workflows, resumably."""

    from alphaapollo.workflows.scoring import run_cell

    normalized = coerce_config(config)
    validate_composition_config(normalized)
    execution = normalized.execution
    condition = execution_label(normalized)
    if tool_executor_factory is not None:
        raise ConfigError(
            "tool_executor_factory is a legacy execution injection; canonical tool "
            "composition is owned by #186"
        )
    run_config_for_cell(
        normalized,
        run_dir=Path(workdir) / ".preflight",
        condition=condition,
        seed=execution.seed,
    )

    task_rows: list[tuple[ScoredTask, WorkflowInput | None]]
    if problems is not None:
        task_rows = [(problem, None) for problem in problems]
    elif normalized.scoring is None:
        task_rows = [
            (
                ScoredTask(
                    id=item.input_id,
                    problem=item.problem,
                    gold_answer="",
                ),
                item,
            )
            for item in load_prepared_inputs(normalized)
        ]
    else:
        task_rows = align_public_inputs_with_private_gold(
            normalized,
            load_prepared_inputs(normalized),
        )
    if execution.limit is not None:
        task_rows = task_rows[: execution.limit]

    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    cells = expand_cells(
        task_rows,
        conditions=(condition,),
        k=execution.samples,
        seed_base=execution.seed,
        shard=shard,
        reverse=reverse,
    )

    expected_digest = config_digest(normalized)

    def execute(cell: Cell) -> RunRecord:
        record = run_cell(
            normalized,
            cell,
            root=root,
            tool_executor_factory=None,
            resume=resume,
            expected_digest=expected_digest,
        )
        if progress is not None:
            progress(record)
        return record

    workers = min(execution.concurrency, len(cells)) or 1
    if workers == 1:
        return [execute(cell) for cell in cells]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(execute, cells))
