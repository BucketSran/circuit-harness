"""Read-only projection of saved stock Harbor jobs and independently checked scores.

No Job, Agent, environment or transport is instantiated. The CLI writes a new private
report directory; incomplete attempts retain their place in the native plan.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from pydantic import BaseModel, ConfigDict, Field

from circuit_harness.execution.candidate_bundle import verify_candidate
from circuit_harness.execution.journal import file_digest

from .config import FinalEvaluationConfig, PublicSessionConfig, require_harbor_version
from .evidence import checked_evaluation as _checked_evaluation
from .evidence import read_json as _read


class AttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    plan_id: str
    attempt: int
    task: dict[str, Any]
    conditions: dict[str, Any]
    trial_id: str | None = None
    trial_name: str | None = None
    task_checksum: str | None = None
    status: Literal["not_run", "missing", "running", "failed", "unevaluable", "graded", "invalid"]
    started: bool | None = False
    agent_exception: str | None = None
    agent_info: dict[str, Any] | None = None
    collection_source: str | None = None
    termination_reason: str | None = None
    candidate_sha256: str | None = None
    evaluation: dict[str, Any] | None = None
    actual_conditions: dict[str, Any] = Field(default_factory=dict)
    score: float | None = Field(default=None, ge=0, le=1)
    usage: dict[str, int | float | None] = Field(
        default_factory=lambda: {
            "n_input_tokens": None,
            "n_cache_tokens": None,
            "n_output_tokens": None,
            "cost_usd": None,
        }
    )
    timings: dict[str, Any] = Field(default_factory=dict)
    sources: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class GroupSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    plan_id: str
    task: dict[str, Any]
    conditions: dict[str, Any]
    planned: int
    started: int
    started_unknown: int
    metric_valid: int
    missing: int
    coverage: float
    score_denominator: int
    mean_score: float | None
    statuses: dict[str, int]
    observed_conditions: dict[str, Any] | None = None


class ExperimentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    harbor_version: Literal["0.23.0"] = "0.23.0"
    job_id: str | None = None
    source_config_sha256: str
    source_result_sha256: str | None = None
    denominator: Literal["planned task × agent × n_attempts; score mean uses metric_valid only"] = (
        "planned task × agent × n_attempts; score mean uses metric_valid only"
    )
    records: list[AttemptRecord]
    groups: list[GroupSummary]
    errors: list[str] = Field(default_factory=list)


def _native(value: Any) -> Any:
    """Serialize data without Harbor's env serializers, which consult host keys."""
    if isinstance(value, BaseModel):
        return {name: _native(getattr(value, name)) for name in type(value).model_fields}
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (Path, UUID)):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _agent(agent) -> dict:
    # Never serialize env values. The digest still separates differing native options.
    return {
        "name": agent.kwargs.get("agent_name", agent.name or agent.import_path),
        "model": agent.model_name,
        "config_sha256": _digest(_native(agent)),
        "timeout_s": agent.override_timeout_sec,
        "max_timeout_s": agent.max_timeout_sec,
        "setup_timeout_s": agent.override_setup_timeout_sec,
    }


def _public(config, task, path=None) -> dict:
    kwargs = config.environment.kwargs
    path = path or kwargs.get("session_config")
    if path is None and config.environment.import_path == (
        "circuit_harness.harbor.environment:HarborChipsEnvironment"
    ):
        path = task.path / "environment/harness.json"
    if path is None:
        return {}
    raw = _read(Path(path))
    # Legacy native config extends PublicSessionConfig; project only declared public fields.
    public = PublicSessionConfig.model_validate(
        {key: value for key, value in raw.items() if key in PublicSessionConfig.model_fields}
    )
    if any(
        not isinstance(public.task.get(key), str) or not public.task[key]
        for key in ("task_id", "task_version")
    ):
        raise ValueError("public configuration requires nonempty task identity")
    return {
        "task_id": public.task["task_id"],
        "task_version": public.task["task_version"],
        "public_config_sha256": file_digest(Path(path)),
        "public_backend": public.public_backend,
        "public_image": public.image,
        # Runtime policy is pinned in session.json, not in this operator declaration.
        # Reanalyzing an old job must not infer today's tools from today's code.
        "public_tools": None,
        "public_budget": {
            "cpu_limit": public.public_cpu_limit,
            "max_actions": public.max_actions,
            "max_simulations": public.max_simulations,
            "timeout_s": public.simulation_timeout_s,
            "max_output_bytes": public.max_output_bytes,
        },
    }


def _final(config, task, path=None) -> dict:
    from circuit_harness.execution.benchmark_spectre import package_identity

    path = path or config.verifier.kwargs.get("config_path")
    if path is None:
        native = task.path / "environment/harness.json"
        if not native.exists():
            return {"final_backend": "spectre", "final_identity": None}
        raw = _read(native)
        values = {"task_package": raw["final_task_package"]}
        if "final_backend" in raw:
            values["backend"] = raw["final_backend"]
        for field in ("remote", "opensource"):
            if "final_" + field in raw:
                values[field] = raw["final_" + field]
        path = native
        raw = values
    else:
        raw = _read(Path(path))
    final = FinalEvaluationConfig.model_validate(raw)
    package = package_identity(final.task_package, purpose="final")
    manifest = package["manifest"]
    return {
        "final_backend": "opensource"
        if getattr(final, "backend", None) == "benchmark_opensource"
        else "spectre",
        "final_config_sha256": file_digest(Path(path)),
        "final_opensource": _native(final.opensource)
        if getattr(final, "opensource", None)
        else None,
        "final_identity": {
            "task_package_sha256": package["sha256"],
            **{
                key: manifest[key]
                for key in (
                    "task_id",
                    "task_version",
                    "criteria_sha256",
                    "condition_id",
                    "task_set",
                )
            },
        },
    }


def _task_key(task) -> str:
    return _digest(_native(task))


def _selection(config, task, tasks):
    env = config.environment.kwargs
    verifier = config.verifier.kwargs
    manifest = env.get("task_bindings")
    if not manifest and not verifier.get("task_bindings"):
        return None, None
    if (
        not manifest
        or manifest != verifier.get("task_bindings")
        or env.get("session_config") is not None
        or verifier.get("config_path") is not None
    ):
        raise ValueError("environment/verifier require the same exclusive task_bindings")
    from .task_bindings import load_task_bindings, resolve_task_binding

    roots = [item.path for item in tasks] + [config.jobs_dir]
    bindings = load_task_bindings(manifest, public_roots=roots)
    if {entry.task_path.resolve() for entry in bindings.tasks} != {
        item.path.resolve() for item in tasks
    }:
        raise ValueError("task binding coverage differs from native plan")
    entry = next(
        entry for entry in bindings.tasks if entry.task_path.resolve() == task.path.resolve()
    )
    if entry.support == "unsupported":
        return None, entry
    expected = env.get("task_binding_receipts", {}).get(str(task.path.resolve()))
    if expected != verifier.get("task_binding_receipts", {}).get(str(task.path.resolve())):
        raise ValueError("environment/verifier selected receipts differ")
    selected = resolve_task_binding(
        manifest,
        task.path,
        public_roots=roots,
        expected_manifest_sha256=env.get("task_bindings_sha256"),
        expected_receipt=expected,
    )
    return selected, entry


def _plan(config: JobConfig) -> list[tuple[Any, Any, dict, dict, str]]:
    if config.source_jobs:
        raise ValueError("offline reports do not resolve regrade source_jobs")
    tasks = list(config.tasks)
    for dataset in config.datasets:
        if not dataset.is_local():
            raise ValueError(
                "offline reports require local datasets; remote resolution is forbidden"
            )
        tasks.extend(
            asyncio.run(dataset.get_task_configs(disable_verification=config.verifier.disable))
        )
    result = []
    seen = set()
    for task in tasks:
        if task.git_url is not None or task.path is None:
            raise ValueError("offline reports require explicit local tasks")
        selection, binding = _selection(config, task, tasks)
        for agent in config.agents:
            key = (_task_key(task), _digest(_native(agent)))
            if key in seen:
                raise ValueError("duplicate planned task/agent configuration")
            seen.add(key)
            public = (
                _public(config, task, selection.session_config_path if selection else None)
                if binding is None or binding.support == "supported"
                else {"task_id": binding.task_id, "task_version": binding.task_version}
            )
            task_record = {
                "harbor_task": _native(task),
                "task_id": public.pop("task_id", None),
                "task_version": public.pop("task_version", None),
            }
            conditions = {
                "agent": _agent(agent),
                **public,
                "environment_config_sha256": _digest(_native(config.environment)),
                "verifier_config_sha256": _digest(_native(config.verifier)),
                **(
                    _final(config, task, selection.final_config_path if selection else None)
                    if binding is None or binding.support == "supported"
                    else {"final_backend": None, "unsupported_reason": binding.reason}
                ),
                "binding_receipt": selection.receipt if selection else None,
                "verifier_timeout_s": config.verifier.override_timeout_sec,
                "timeout_multiplier": config.timeout_multiplier,
                "agent_timeout_multiplier": config.agent_timeout_multiplier,
                "verifier_timeout_multiplier": config.verifier_timeout_multiplier,
                "agent_setup_timeout_multiplier": config.agent_setup_timeout_multiplier,
                "environment_build_timeout_multiplier": config.environment_build_timeout_multiplier,
                "install_only": config.install_only,
            }
            result.append(
                (task, agent, task_record, conditions, _digest([task_record, conditions]))
            )
    if not result or config.n_attempts < 1:
        raise ValueError("empty Harbor plan")
    return result


def _project(record: AttemptRecord, directory: Path, trial_config: TrialConfig) -> None:
    record.trial_name = directory.name
    record.sources["config.json"] = file_digest(directory / "config.json")
    result_path = directory / "result.json"
    if not result_path.exists():
        record.status = "missing"
        record.started = None
        return
    trial = TrialResult.model_validate(_read(result_path))
    if (
        _native(trial.config) != _native(trial_config)
        or trial.trial_name != directory.name
        or trial.config.job_id != trial_config.job_id
    ):
        raise ValueError("TrialResult differs from trial configuration")
    if trial.task_id != trial.config.task.get_task_id():
        raise ValueError("native trial task identity mismatch")
    expected_binding = record.conditions.get("binding_receipt")
    if expected_binding is not None:
        binding_path = directory / "task-binding.json"
        if _read(binding_path) != expected_binding:
            raise ValueError("actual task binding receipt differs from plan")
        record.sources["task-binding.json"] = file_digest(binding_path)
    elif record.conditions.get("unsupported_reason"):
        raise ValueError("unsupported planned task has a started Trial")
    record.sources["result.json"] = file_digest(result_path)
    record.trial_id = str(trial.id)
    record.task_checksum = trial.task_checksum
    record.agent_info = trial.agent_info.model_dump(mode="json")
    record.actual_conditions["agent"] = record.agent_info
    record.started = trial.started_at is not None
    record.agent_exception = trial.exception_info.exception_type if trial.exception_info else None
    record.status = (
        "failed"
        if trial.exception_info
        else ("running" if trial.finished_at is None else "unevaluable")
    )
    record.timings = {
        name: value.model_dump(mode="json")
        if isinstance(value, BaseModel)
        else value.isoformat()
        if value
        else None
        for name in (
            "started_at",
            "finished_at",
            "environment_setup",
            "agent_setup",
            "agent_execution",
            "verifier",
        )
        if (value := getattr(trial, name)) is not None
    }
    record.usage = dict(
        zip(
            ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd"),
            trial.compute_token_cost_totals(),
            strict=True,
        )
    )
    frozen_path = directory / "public-session/frozen.json"
    evaluation_path = directory / "verifier/evaluation.json"
    if not frozen_path.exists():
        if evaluation_path.exists() or trial.verifier_result is not None:
            raise ValueError("evaluation without frozen candidate evidence")
        return
    collected = _read(frozen_path)
    frozen = verify_candidate(directory / "public-session/candidate")
    if collected["candidate_sha256"] != frozen["candidate_sha256"]:
        raise ValueError("frozen collection candidate identity mismatch")
    session = _read(directory / "public-session/session.json")
    for key in ("task_id", "task_version"):
        if frozen[key] != session["task"][key] or frozen[key] != record.task[key]:
            raise ValueError("frozen/session/planned task identity mismatch")
    record.actual_conditions["public"] = {
        key: session.get(key)
        for key in ("backend", "image", "kernel_sha256", "codex_sha256", "python_sha256")
    }
    record.actual_conditions["public"]["source_sha256"] = _digest(session.get("source"))
    from circuit_harness.execution.current_evas_session import tool_schemas

    view_version = session.get("observation_view_version")
    if view_version not in (None, 1):
        raise ValueError("unsupported public observation view version")
    record.actual_conditions["public"].update(
        observation_view_version=view_version,
        public_tools=[
            schema["function"]["name"]
            for schema in tool_schemas(
                experiments=bool(session["task"].get("experiments")),
                observations=view_version == 1,
            )
        ],
    )
    # Sessions created before the configurable CPU policy always requested one core.
    record.actual_conditions["public"]["cpu_limit"] = session.get("cpu_limit", 1)
    record.actual_conditions["public"]["public_package_sha256"] = _digest(
        session.get("public_package")
    )
    if session.get("image") != record.conditions.get("public_image"):
        raise ValueError("actual public image differs from plan")
    if session.get("backend") != record.conditions.get("public_backend"):
        raise ValueError("actual public backend differs from plan")
    if any(
        session.get(key, 1 if key == "cpu_limit" else None) != value
        for key, value in record.conditions.get("public_budget", {}).items()
    ):
        raise ValueError("actual public budget differs from plan")
    episode_path = directory / "public-session/episode-end.json"
    if episode_path.exists():
        episode = _read(episode_path)
        if any(
            episode.get(key) != collected.get(key)
            for key in ("candidate_sha256", "collection_source", "state")
        ):
            raise ValueError("episode termination differs from frozen candidate collection")
        record.termination_reason = episode.get("termination_reason")
        record.sources["public-session/episode-end.json"] = file_digest(episode_path)
    record.collection_source = collected.get("collection_source")
    record.candidate_sha256 = frozen["candidate_sha256"]
    for path in (
        frozen_path,
        directory / "public-session/candidate/manifest.json",
        directory / "public-session/session.json",
    ):
        record.sources[str(path.relative_to(directory))] = file_digest(path)
    receipts = list((directory / "verifier/transport").glob("*/archive/receipt.json"))
    has_receipt = bool(receipts) or (directory / "verifier/replay/receipt.json").exists()
    if not evaluation_path.exists() and not has_receipt:
        if trial.verifier_result is not None:
            raise ValueError("native reward has no independent evaluation")
        return
    final, sources, identity = _checked_evaluation(directory, frozen, record.task)
    record.actual_conditions["final"] = {"backend": identity["backend"]}
    if "profile" in identity:
        record.actual_conditions["final"].update(
            profile_sha256=identity["profile"]["sha256"],
            tools_sha256=_digest(identity["profile"]["tools"]),
        )
    else:
        configuration = identity["configuration"]
        if configuration != record.conditions.get("final_opensource"):
            raise ValueError("actual replay configuration differs from plan")
        record.actual_conditions["final"]["configuration_sha256"] = _digest(configuration)
    record.sources.update(sources)
    if evaluation_path.exists():
        record.sources["verifier/evaluation.json"] = file_digest(evaluation_path)
    record.evaluation = {
        key: final.get(key)
        for key in (
            "execution",
            "verdict",
            "score",
            "backend",
            "purpose",
            "task_id",
            "task_version",
            "candidate_sha256",
            "task_package_sha256",
            "criteria_sha256",
            "condition_id",
            "task_set",
            "certified",
        )
    }
    if final.get("backend") != record.conditions["final_backend"]:
        raise ValueError("actual final backend differs from plan")
    expected_final = record.conditions.get("final_identity") or {}
    if any(final.get(key) != value for key, value in expected_final.items()):
        raise ValueError("actual final package/criteria/condition differs from plan")
    score = final.get("score")
    valid = (
        final.get("execution") == "ok"
        and final.get("verdict") in ("pass", "fail")
        and not isinstance(score, bool)
        and isinstance(score, (int, float))
        and math.isfinite(score)
        and 0 <= score <= 1
    )
    if valid:
        if trial.verifier_result is not None and trial.verifier_result.rewards != {"reward": score}:
            raise ValueError("native reward differs from independent final score")
        record.score = score
        if trial.exception_info is None:
            record.status = "graded"
    elif trial.verifier_result is not None:
        raise ValueError("native reward for unevaluable final execution")


def read_job(directory: Path) -> ExperimentReport:
    """Read a saved local Job without executing or modifying its plan or evidence."""
    require_harbor_version()
    directory = Path(directory).resolve()
    config = JobConfig.model_validate(_read(directory / "config.json"))
    planned = _plan(config)
    result_path = directory / "result.json"
    job = JobResult.model_validate(_read(result_path)) if result_path.exists() else None
    expected = len(planned) * config.n_attempts
    errors = []
    if job and job.n_total_trials != expected:
        errors.append("JobResult total differs from native plan")
    trial_directories = sorted(path.parent for path in directory.glob("*/config.json"))
    available = []
    orphaned = []
    for trial_dir in trial_directories:
        try:
            available.append(
                (trial_dir, TrialConfig.model_validate(_read(trial_dir / "config.json")))
            )
        except (OSError, ValueError) as error:
            errors.append(f"{trial_dir.name}: {error}")
            orphaned.append(
                AttemptRecord(
                    plan_id="unknown",
                    attempt=0,
                    task={},
                    conditions={},
                    trial_name=trial_dir.name,
                    status="invalid",
                    started=None,
                    errors=[str(error)],
                )
            )
    records = list(orphaned)
    consumed = set()
    ids = {}
    evaluated = {}
    for task, agent, task_record, conditions, plan_id in planned:
        matches = [
            (path, trial)
            for path, trial in available
            if _task_key(trial.task) == _task_key(task) and _native(trial.agent) == _native(agent)
        ]
        if len(matches) > config.n_attempts:
            errors.append("duplicate/excess Trial for planned task/agent")
        for attempt in range(max(config.n_attempts, len(matches))):
            record = AttemptRecord(
                plan_id=plan_id,
                attempt=attempt + 1,
                task=task_record,
                conditions=conditions,
                status="not_run",
            )
            if attempt < len(matches):
                path, trial = matches[attempt]
                consumed.add(path)
                try:
                    if job and trial.job_id != job.id:
                        raise ValueError("trial belongs to another Job")
                    if trial.environment != config.environment or trial.verifier != config.verifier:
                        raise ValueError("trial environment/verifier differs from Job plan")
                    shared_fields = (
                        "install_only",
                        "timeout_multiplier",
                        "agent_timeout_multiplier",
                        "verifier_timeout_multiplier",
                        "agent_setup_timeout_multiplier",
                        "environment_build_timeout_multiplier",
                        "user_agent",
                        "artifacts",
                        "extra_instruction_paths",
                        "extra_instructions",
                    )
                    if any(
                        _native(getattr(trial, key)) != _native(getattr(config, key))
                        for key in shared_fields
                    ):
                        raise ValueError("Trial settings/budgets differ from Job plan")
                    record.started = None
                    _project(record, path, trial)
                    if record.trial_id:
                        if record.trial_id in ids:
                            prior = ids[record.trial_id]
                            prior.status = "invalid"
                            prior.score = None
                            prior.errors.append("duplicate native Trial ID")
                            raise ValueError("duplicate native Trial ID")
                        ids[record.trial_id] = record
                    if record.evaluation is not None:
                        receipt_key = next(
                            value
                            for name, value in record.sources.items()
                            if name.endswith("/receipt.json") and name.startswith("verifier/")
                        )
                        if receipt_key in evaluated:
                            prior = evaluated[receipt_key]
                            prior.status = "invalid"
                            prior.score = None
                            prior.errors.append("duplicate independent evaluation receipt")
                            raise ValueError("duplicate independent evaluation receipt")
                        evaluated[receipt_key] = record
                    if len(matches) > config.n_attempts:
                        raise ValueError("duplicate/excess Trial for planned task/agent")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    record.status = "invalid"
                    record.score = None
                    record.errors.append(str(error))
            records.append(record)
    for path, _trial in available:
        if path not in consumed:
            errors.append(f"{path.name}: Trial not in native plan")
            records.append(
                AttemptRecord(
                    plan_id="unknown",
                    attempt=0,
                    task={"harbor_task": _native(_trial.task)},
                    conditions={"agent": _agent(_trial.agent)},
                    trial_name=path.name,
                    status="invalid",
                    started=None,
                    errors=["Trial not in native plan"],
                )
            )
    if job and job.trial_results:
        embedded_ids = [str(trial.id) for trial in job.trial_results]
        if len(embedded_ids) != len(set(embedded_ids)):
            errors.append("duplicate embedded JobResult Trial ID")
        for embedded in job.trial_results:
            path = directory / embedded.trial_name / "result.json"
            try:
                if not path.resolve().is_relative_to(directory):
                    raise ValueError("unsafe embedded trial name")
                native = TrialResult.model_validate(_read(path))
                if _native(native) != _native(embedded):
                    raise ValueError("embedded JobResult differs from TrialResult")
            except (OSError, ValueError) as error:
                errors.append(str(error))
    if errors:
        for record in records:
            if record.score is not None:
                record.status = "invalid"
                record.score = None
                record.errors.append("Job evidence is inconsistent; see report errors")
    groups = []
    for _task, _agent_config, task_record, conditions, plan_id in planned:
        rows = [record for record in records if record.plan_id == plan_id]
        observed = {_digest(row.actual_conditions) for row in rows if row.score is not None}
        if len(observed) > 1:
            for row in rows:
                if row.score is not None:
                    row.score = None
                    row.status = "invalid"
                    row.errors.append(
                        "conflicting actual Agent/Model or resource identities in one planned cell"
                    )
        valid = [record.score for record in rows if record.score is not None]
        statuses = {
            status: sum(record.status == status for record in rows)
            for status in sorted({record.status for record in rows})
        }
        groups.append(
            GroupSummary(
                plan_id=plan_id,
                task=task_record,
                conditions=conditions,
                planned=config.n_attempts,
                started=sum(row.started is True for row in rows),
                started_unknown=sum(row.started is None for row in rows),
                metric_valid=len(valid),
                missing=statuses.get("missing", 0) + statuses.get("not_run", 0),
                coverage=len(valid) / config.n_attempts,
                score_denominator=len(valid),
                mean_score=sum(valid) / len(valid) if valid else None,
                statuses=statuses,
                observed_conditions=next(
                    (row.actual_conditions for row in rows if row.score is not None), None
                ),
            )
        )
    return ExperimentReport(
        job_id=str(job.id) if job else None,
        source_config_sha256=file_digest(directory / "config.json"),
        source_result_sha256=file_digest(result_path) if job else None,
        records=records,
        groups=groups,
        errors=errors,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export a saved Harbor Chips Job offline")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.absolute()
        source = args.job.resolve()
        if output.resolve().is_relative_to(source):
            raise ValueError("report output must be outside the input Job")
        if output.exists() or output.is_symlink():
            raise ValueError("report output already exists; choose a new directory")
        report = read_job(source)
        serialized = report.model_dump_json(indent=2) + "\n"
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        path = output / "report.json"
        with path.open("x") as target:
            target.write(serialized)
        path.chmod(0o600)
        return 1 if report.errors or any(record.errors for record in report.records) else 0
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
