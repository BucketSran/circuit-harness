"""Task-owned Chips experiment policies; no simulator or score is shared here."""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from alphaapollo.common.execution.chips.journal import atomic_json
from alphaapollo.workflows import chips_analog_agent, chips_vabench_agent
from alphaapollo.workflows._chips_experiment.core import _write_result


def _project_vabench(output: Path) -> None:
    row = json.loads((output / "results.jsonl").read_text(encoding="utf-8"))
    inner = output / "agent/results.jsonl"
    if inner.is_file():
        result = json.loads(inner.read_text(encoding="utf-8"))
        row.update(
            status=result["status"],
            score=result.get("score"),
            benchmark_success=result.get("benchmark_success"),
            verdict=result.get("verdict"),
            error_type=result.get("error_type"),
        )
        report = output / "agent/report.json"
        if report.is_file():
            final = json.loads(report.read_text(encoding="utf-8")).get("result", {})
            row["final_execution"] = final.get("execution", "not_run")
            row["final_validity"] = "valid" if row["status"] == "completed" else "not_evaluated"
    else:
        row.update(status="error", error_type="agent_record_missing")
    _write_result(output, row)


def _project_analog(output: Path) -> None:
    row = json.loads((output / "results.jsonl").read_text(encoding="utf-8"))
    report = output / "agent/report.json"
    if report.is_file():
        detail = json.loads(report.read_text(encoding="utf-8"))
        state = detail.get("state")
        row.update(
            {
                key: detail[key]
                for key in (
                    "agent_submitted",
                    "collection_source",
                    "candidate_sha256",
                    "termination_reason",
                )
                if key in detail
            }
        )
        row["status"] = {
            "submitted": "awaiting_final",
            "collected": "awaiting_final",
            "missing_candidate": "missing_candidate",
            "awaiting_action_recovery": "awaiting_action_recovery",
            "unsubmitted": "unsubmitted",
        }.get(state, "error")
        row["error_type"] = "agent_report_state" if row["status"] == "error" else None
    else:
        row.update(status="error", error_type="agent_report_missing")
    _write_result(output, row)


def _archive_analog(remote, plan: dict, operator: dict, output: Path, candidate: str | None) -> str:
    receipt = remote.cli(
        "analog-archive",
        "--session",
        operator["session"],
        "--archive-root",
        plan["archive_root"],
        "--episode-id",
        plan["run_id"],
        "--agent-evidence",
        str(output / "agent"),
        *(["--final-output", plan["final_output"]] if candidate is not None else []),
        timeout=300,
    )
    digest = receipt.get("sha256")
    if receipt.get("candidate_sha256") != candidate or not isinstance(digest, str):
        raise ValueError("episode archive does not match the frozen candidate")
    record = str(Path(plan["archive_root"]) / "episodes" / plan["run_id"])
    verified = remote.cli("verify-analog-episode", record, timeout=120)
    if verified.get("state") != "verified" or verified.get("sha256") != digest:
        raise ValueError("episode archive verification failed")
    atomic_json(output / "archive-receipt.json", receipt)
    return digest


def _validate_v1_options(extra: set[str]) -> None:
    if extra not in (
        set(),
        {"key_file"},
        {"final_output", "archive_root"},
        {"key_file", "final_output", "archive_root"},
    ):
        raise ValueError("unsupported experiment config")


def _validate_vabench_plan(plan: dict, extra: set[str]) -> None:
    _validate_v1_options(extra)
    if "final_output" in plan:
        raise ValueError("final_output belongs only to Analog Design Bench")
    if "key_file" in plan:
        raise ValueError("key_file belongs only to Analog Design Bench")


def _validate_analog_plan(plan: dict, extra: set[str]) -> None:
    _validate_v1_options(extra)
    if "final_output" in plan:
        for name in ("final_output", "archive_root"):
            if not isinstance(plan[name], str) or not Path(plan[name]).is_absolute():
                raise ValueError(f"{name} must be an absolute server path")
    if "key_file" in plan:
        if not isinstance(plan["key_file"], str) or not Path(plan["key_file"]).is_absolute():
            raise ValueError("key_file must be an absolute private path")


def _validate_vabench_operator(plan: dict, operator: dict) -> None:
    validator = (
        chips_vabench_agent.validate_pilot
        if plan["agent"] == "pi"
        else chips_vabench_agent.validate_codex
    )
    validator(operator)
    if operator.get("task_id") != plan["task_id"] or operator.get("job_id") != plan["run_id"]:
        raise ValueError("experiment task_id and run_id must match the VABench operator")


def _validate_analog_operator(plan: dict, operator: dict) -> None:
    chips_analog_agent.validate_pilot(operator)
    if plan["task_id"] != chips_analog_agent.pilot_task_id(operator):
        raise ValueError("experiment task_id must match the Analog operator task_id")
    if operator["transport"] != "local":
        raise ValueError("the Analog experiment preflight currently requires local transport")


def _run_vabench(plan: dict, operator: dict, output: Path, row: dict, arguments: list[str]) -> int:
    row["status"] = "running"
    _write_result(output, row)
    try:
        return chips_vabench_agent.main(arguments)
    finally:
        _project_vabench(output)


def _run_analog(plan: dict, operator: dict, output: Path, row: dict, arguments: list[str]) -> int:
    agent_evidence = output / "agent"
    agent_evidence.mkdir(mode=0o700)
    try:
        preflight = chips_analog_agent.preflight_pilot(
            operator, key_file=Path(plan["key_file"]) if "key_file" in plan else None
        )
    except Exception as error:
        row.update(status="error", error_type=type(error).__name__)
        _write_result(output, row)
        raise
    atomic_json(agent_evidence / "preflight.json", preflight)
    if preflight["state"] != "ready":
        row.update(status="blocked", error_type="preflight_not_ready")
        _write_result(output, row)
        return 1
    if "key_file" in plan:
        arguments.extend(("--key-file", plan["key_file"]))
    row["status"] = "running"
    _write_result(output, row)
    try:
        return chips_analog_agent.main(arguments)
    finally:
        if "key_file" in plan:
            os.environ.pop("CHIPS_MODEL_KEY", None)
        _project_analog(output)


def _collect_vabench(plan: dict, operator: dict, output: Path, row: dict) -> int:
    if row["status"] != "pending":
        raise ValueError("only a pending VABench result can be collected")
    try:
        return chips_vabench_agent.main(
            [
                "collect",
                "--config",
                plan["operator_config"],
                "--evidence",
                str(output / "agent"),
            ]
        )
    finally:
        _project_vabench(output)


def _collect_analog(plan: dict, operator: dict, output: Path, row: dict) -> int:
    if row["status"] != "awaiting_action_recovery":
        raise ValueError("only an Analog episode awaiting recovery can be collected")
    evidence = output / "agent"
    outcome = json.loads((evidence / "pi-outcome.json").read_text())
    report = json.loads((evidence / "report.json").read_text())
    report.update(chips_analog_agent.finish_episode(operator, evidence, outcome))
    atomic_json(evidence / "report.json", report)
    _project_analog(output)
    return 0 if report["state"] in {"submitted", "collected", "missing_candidate"} else 1


def _finalize_analog(plan: dict, operator: dict, output: Path, row: dict) -> int:
    if "final_output" not in plan:
        raise ValueError("Analog finalization needs final_output and archive_root in the plan")
    if row["status"] != "awaiting_final":
        raise ValueError("only an acknowledged Analog submission can be finalized")
    submitted = []
    collection = output / "agent/collection.json"
    if collection.is_file():
        receipt = json.loads(collection.read_text())
        if (
            receipt.get("state") == "collected"
            and receipt.get("collection_source") == "episode_end"
            and receipt.get("agent_submitted") is False
        ):
            submitted.append(receipt.get("candidate_sha256"))
    for request_path in (output / "agent/tools").glob("*/request.json"):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response_path = request_path.with_name("response.json")
        if request.get("tool") == "analog_submit" and response_path.is_file():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if response.get("ok") and response.get("result", {}).get("state") == "submitted":
                submitted.append(response["result"].get("candidate_sha256"))
    if len(submitted) != 1 or not isinstance(submitted[0], str):
        raise ValueError("finalization needs one acknowledged candidate digest")
    intent = output / "finalize-intent.json"
    if intent.exists():
        raise ValueError("final scoring may already have started; inspect the original output")
    remote = chips_analog_agent.transport(operator, output / "agent/tools")
    atomic_json(
        intent,
        {
            "session": operator["session"],
            "final_output": plan["final_output"],
            "candidate_sha256": submitted[0],
        },
    )
    try:
        result = remote.cli(
            "analog-finalize",
            "--session",
            operator["session"],
            "--output",
            plan["final_output"],
            "--archive-root",
            plan["archive_root"],
            timeout=960,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        row.update(status="unknown_execution", error_type=type(error).__name__)
        _write_result(output, row)
        raise
    atomic_json(output / "final-result.json", result)
    if result.get("frozen_candidate_sha256") != submitted[0]:
        row.update(status="error", error_type="final_candidate_mismatch")
        _write_result(output, row)
        raise ValueError("final scorer candidate differs from the acknowledged submission")
    state = result.get("state")
    score = result.get("score")
    graded = (
        state == "graded"
        and type(score) in (int, float)
        and math.isfinite(score)
        and 0 <= score <= 1
    )
    row.update(
        status="final_graded" if graded else "error",
        error_type=None if graded else state or "invalid_final_result",
        final_execution="ok" if graded else state or "unknown",
        final_validity="valid" if graded else "not_evaluated",
        score=float(score) if graded else None,
        frozen_candidate_sha256=submitted[0],
    )
    _write_result(output, row)
    if not graded:
        return 1
    try:
        row["episode_archive_sha256"] = _archive_analog(
            remote, plan, operator, output, submitted[0]
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        row.update(status="archive_failed", error_type=type(error).__name__)
        _write_result(output, row)
        raise
    row["status"] = "completed"
    _write_result(output, row)
    return 0


def _resume_analog_archive(plan: dict, operator: dict, output: Path, row: dict) -> int:
    if row["status"] == "missing_candidate":
        if "archive_root" not in plan:
            raise ValueError("Analog archiving needs archive_root in the plan")
        receipt = json.loads((output / "agent/collection.json").read_text())
        if (
            receipt.get("state") != "missing_candidate"
            or receipt.get("candidate_sha256") is not None
        ):
            raise ValueError("missing candidate receipt cannot be verified")
        remote = chips_analog_agent.transport(operator, output / "agent/tools")
        row["episode_archive_sha256"] = _archive_analog(remote, plan, operator, output, None)
        row.update(
            status="completed",
            final_execution="not_applicable",
            final_validity="not_applicable",
            error_type=None,
        )
        _write_result(output, row)
        return 0
    if row["status"] not in {
        "final_graded",
        "archive_failed",
    }:
        raise ValueError("only a graded Analog experiment can resume archiving")
    final_path = output / "final-result.json"
    if final_path.is_symlink() or not final_path.is_file():
        raise ValueError("the original final result is unavailable")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    candidate = row.get("frozen_candidate_sha256")
    if (
        row.get("final_execution") != "ok"
        or final.get("state") != "graded"
        or final.get("frozen_candidate_sha256") != candidate
    ):
        raise ValueError("the original final result cannot be verified")
    remote = chips_analog_agent.transport(operator, output / "agent/tools")
    try:
        row["episode_archive_sha256"] = _archive_analog(remote, plan, operator, output, candidate)
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        row["error_type"] = type(error).__name__
        _write_result(output, row)
        raise
    row.update(status="completed", error_type=None)
    _write_result(output, row)
    return 0


def _unsupported_vabench_finalize(*_args) -> int:
    raise ValueError("finalize is only available for Analog Design Bench")


def _unsupported_vabench_archive(*_args) -> int:
    raise ValueError("only a graded Analog experiment can resume archiving")


@dataclass(frozen=True)
class ExperimentTask:
    agents: frozenset[str]
    validate_plan: Callable
    validate_operator: Callable
    run: Callable
    collect: Callable
    finalize: Callable
    archive: Callable


_TASKS = {
    "vabench": ExperimentTask(
        frozenset({"pi", "codex"}),
        _validate_vabench_plan,
        _validate_vabench_operator,
        _run_vabench,
        _collect_vabench,
        _unsupported_vabench_finalize,
        _unsupported_vabench_archive,
    ),
    "analog_design_bench": ExperimentTask(
        frozenset({"pi"}),
        _validate_analog_plan,
        _validate_analog_operator,
        _run_analog,
        _collect_analog,
        _finalize_analog,
        _resume_analog_archive,
    ),
}


def task_for(benchmark: object, agent: object) -> ExperimentTask:
    if not isinstance(benchmark, str) or not isinstance(agent, str):
        raise ValueError("unsupported Chips experiment condition")
    task = _TASKS.get(benchmark)
    if task is None or agent not in task.agents:
        raise ValueError("unsupported Chips experiment condition")
    return task
