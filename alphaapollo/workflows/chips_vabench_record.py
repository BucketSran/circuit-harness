"""One-cell evaluation ledger for a VABench Agent episode."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

from alphaapollo.common.execution.chips.journal import atomic_json


def _write_result(evidence: Path, row: dict) -> None:
    path = evidence / "results.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    completed = row["status"] == "completed"
    atomic_json(
        evidence / "summary.json",
        {
            "schema_version": 1,
            "expected_attempts": 1,
            "completed_attempts": int(completed),
            "status_counts": {row["status"]: 1},
            "success_rate": int(row["benchmark_success"]) if completed else None,
        },
    )


def _trajectory_refs(evidence: Path, kind: str) -> dict[str, str]:
    names = {
        "raw_agent_events": f"{kind}-events.jsonl",
        "agent_outcome": f"{kind}-outcome.json",
        "runtime_result": "runtime-result.json",
        "model_budget": "model-budget.jsonl" if kind == "pi" else None,
        "tool_actions": "tools",
    }
    return {
        key: name for key, name in names.items() if name is not None and (evidence / name).exists()
    }


def begin(evidence: Path, config: dict, kind: str, agent_input: dict) -> None:
    """Register the planned cell before preflight or model execution."""
    evidence = Path(evidence)
    if evidence.is_symlink():
        raise ValueError("private evidence directory must not be a symlink")
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = evidence.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("private evidence directory must be owned by this user and mode 0700")
    # detach creates these two launcher files before its child reaches this point.
    if {item.name for item in evidence.iterdir()} - {"launcher.json", "operator.log"}:
        raise ValueError("use a fresh evidence directory for each model attempt")
    atomic_json(
        evidence / "task_manifest.json",
        {
            "schema_version": 1,
            "benchmark": "vabench",
            "run_id": config.get("job_id"),
            "selected": [{"task_id": config.get("task_id"), "repeat": 1}],
            "task_identity": (
                "declared_by_operator_config"
                if config.get("task_id") is not None
                else "unspecified"
            ),
        },
    )
    atomic_json(
        evidence / "resolved_config.json",
        {"schema_version": 1, "agent_kind": kind, "operator": config},
    )
    atomic_json(evidence / "agent-input.json", agent_input)
    now = time.time()
    _write_result(
        evidence,
        {
            "schema_version": 1,
            "benchmark": "vabench",
            "task_id": config.get("task_id"),
            "repeat": 1,
            "agent_kind": kind,
            "model": config["model"],
            "status": "pending",
            "task_identity_status": "unverified",
            "server_task_id": None,
            "session_sha256": None,
            "error_type": None,
            "benchmark_success": None,
            "verdict": None,
            "score": None,
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "wall_elapsed_seconds": None,
            "tool_calls": 0,
            "trajectory_refs": {},
            "report_ref": None,
        },
    )


def confirm_task(evidence: Path, preflight: dict) -> bool:
    """Bind the declared task to a server-reported session before model use."""
    evidence = Path(evidence)
    row = json.loads((evidence / "results.jsonl").read_text(encoding="utf-8"))
    observed = preflight.get("task_id")
    row["server_task_id"] = observed
    if observed is not None:
        if row["task_id"] is None:
            row["task_identity_status"] = "server_reported"
        else:
            row["task_identity_status"] = "matched" if observed == row["task_id"] else "mismatch"
    row["session_sha256"] = preflight.get("session_sha256")
    row["updated_at"] = time.time()
    _write_result(evidence, row)
    return row["task_identity_status"] != "mismatch"


def require_operator(evidence: Path, config: dict) -> None:
    """A later collect/finalize must address the same immutable experiment."""
    resolved = json.loads((Path(evidence) / "resolved_config.json").read_text(encoding="utf-8"))
    if resolved["operator"] != config:
        raise ValueError("config does not match the recorded operator")


def update(
    evidence: Path, *, status: str, error_type: str | None = None, report: dict | None = None
) -> None:
    """Atomically replace the sole result row and its derived summary."""
    evidence = Path(evidence)
    row = json.loads((evidence / "results.jsonl").read_text(encoding="utf-8"))
    row["status"] = status
    row["error_type"] = error_type
    row["updated_at"] = time.time()
    row["finished_at"] = (
        row["updated_at"] if status in {"completed", "blocked", "unsubmitted", "error"} else None
    )
    row["wall_elapsed_seconds"] = (
        max(0.0, row["finished_at"] - row["started_at"]) if row["finished_at"] is not None else None
    )
    row["tool_calls"] = sum(1 for _ in (evidence / "tools").glob("*/request.json"))
    row["trajectory_refs"] = _trajectory_refs(evidence, row["agent_kind"])
    row["report_ref"] = "report.json" if (evidence / "report.json").is_file() else None
    if report is not None and report.get("state") == "verified":
        result = report.get("result", {})
        row["verdict"] = result.get("verdict")
        row["benchmark_success"] = row["verdict"] == "pass" if status == "completed" else None
        row["public_trace_sha256"] = report.get("public_trace_sha256")
        row["final_archive_sha256"] = report.get("final_archive_sha256")
    _write_result(evidence, row)


def reconcile(evidence: Path, *, exception_type: str | None = None) -> None:
    """Project the report after normal exit, collection, or a caught failure."""
    evidence = Path(evidence)
    row = json.loads((evidence / "results.jsonl").read_text(encoding="utf-8"))
    report_path = evidence / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        state = report.get("state")
        if state == "verified":
            result = report.get("result", {})
            if result.get("execution") == "ok" and result.get("verdict") in {"pass", "fail"}:
                update(evidence, status="completed", report=report)
            else:
                update(
                    evidence, status="error", error_type="final_result_unavailable", report=report
                )
        elif state == "unsubmitted":
            update(evidence, status="unsubmitted")
        elif state in {"pending", "archive_failed"}:
            update(evidence, status=state)
        else:
            update(evidence, status="error", error_type="unknown_report_state")
    elif exception_type and row["status"] != "blocked":
        update(evidence, status="error", error_type=exception_type)
