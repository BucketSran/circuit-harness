"""Shared plan, immutable experiment identity, and result persistence."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

from alphaapollo.common.execution.chips.journal import atomic_json, file_digest


def _read_plan(path: Path) -> tuple[dict, dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("experiment config must be a regular file")
    plan = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "benchmark",
        "task_id",
        "run_id",
        "agent",
        "operator_config",
        "output",
    }
    if not isinstance(plan, dict):
        raise ValueError("unsupported experiment config")
    if (
        not expected <= set(plan)
        or type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
    ):
        raise ValueError("unsupported experiment config")
    from alphaapollo.workflows._chips_experiment.tasks import task_for

    task = task_for(plan["benchmark"], plan["agent"])
    task.validate_plan(plan, set(plan) - expected)
    if not isinstance(plan["run_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", plan["run_id"]
    ):
        raise ValueError("invalid experiment run_id")
    if not isinstance(plan["task_id"], str) or not plan["task_id"]:
        raise ValueError("an explicit task_id is required")
    for name in ("operator_config", "output"):
        if not isinstance(plan[name], str) or not Path(plan[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    operator_path = Path(plan["operator_config"])
    if operator_path.is_symlink() or not operator_path.is_file():
        raise ValueError("operator config must be a regular file")
    operator = json.loads(operator_path.read_text(encoding="utf-8"))
    task.validate_operator(plan, operator)
    return plan, operator


def _write_result(output: Path, row: dict) -> None:
    facts = episode_facts(output)
    row.update(facts)
    path = output / "results.jsonl"
    temporary = output / "results.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    completed = row["status"] == "completed"
    atomic_json(
        output / "summary.json",
        {
            "schema_version": 1,
            "expected_attempts": 1,
            "completed_attempts": int(completed),
            "status_counts": {row["status"]: 1},
            "episode": facts,
            "success_rate": (
                int(row["benchmark_success"])
                if completed and row["benchmark_success"] is not None
                else None
            ),
        },
    )


def episode_facts(output: Path) -> dict:
    """Project observed closure and candidate feedback for either task or Agent."""
    from alphaapollo.workflows._chips_episode_report.evidence import candidate_hashes

    agent = output / "agent"

    def read(path):
        if not path.exists():
            return {}
        if path.is_symlink() or not path.is_file():
            raise ValueError("episode facts require regular evidence files")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("episode facts require JSON objects")
        return value

    outcomes = [
        read(agent / f"{kind}-outcome.json")
        for kind in ("pi", "codex", "native")
        if (agent / f"{kind}-outcome.json").exists()
    ]
    if len(outcomes) > 1:
        raise ValueError("ambiguous episode outcomes")
    outcome = outcomes[0] if outcomes else {}
    detail = read(agent / "report.json")
    collection = read(agent / "collection.json")
    reason = (
        outcome.get("harness_termination_reason")
        or collection.get("termination_reason")
        or detail.get("termination_reason")
        or outcome.get("termination_reason")
    )
    submitted = None
    acknowledged = candidate_hashes(collection) or candidate_hashes(detail)
    simulated = []
    missing_replies = False
    for path in (agent / "tools").glob("*/request.json"):
        request = read(path)
        reply = read(path.with_name("response.json"))
        missing_replies |= (
            not bool(reply)
            or reply.get("state") in {"pending", "unknown_execution"}
            or reply.get("error") in {"unknown_execution", "unresolved_previous_action"}
        )
        result = reply.get("result", {})
        state = result.get("status", result.get("state"))
        if reply.get("ok") is not True:
            continue
        if request.get("tool", "").endswith("_submit") and state == "submitted":
            submitted = True
            digest = candidate_hashes(result)
            if digest and acknowledged and digest != acknowledged:
                raise ValueError("episode candidate differs from acknowledged submission")
            acknowledged = digest or acknowledged
        if request.get("tool", "").endswith("_simulate") and state in {"succeeded", "simulated"}:
            digest = candidate_hashes(result)
            if digest:
                simulated.append(digest)
    if submitted and collection.get("agent_submitted") is False:
        raise ValueError("collection contradicts acknowledged submission")
    if submitted is None:
        submitted = collection.get("agent_submitted", detail.get("agent_submitted"))
    if submitted is None and not missing_replies and detail.get("state") == "unsubmitted":
        submitted = False
    if submitted is None and reason and not missing_replies and (agent / "tools").is_dir():
        submitted = False
    matched = acknowledged in simulated if acknowledged else None
    if matched is False and (missing_replies or not (agent / "tools").is_dir()):
        matched = None
    return {
        "termination_reason": reason,
        "raw_termination_reason": outcome.get("termination_reason"),
        "agent_submitted": submitted,
        "collection_source": "agent_submit"
        if submitted is True
        else collection.get("collection_source", detail.get("collection_source")),
        "final_candidate_publicly_simulated": matched,
    }


def _existing(path: Path) -> tuple[dict, dict, Path, dict]:
    plan, operator = _read_plan(path)
    output = Path(plan["output"])
    if output.is_symlink() or not output.is_dir():
        raise ValueError("experiment output is unavailable")
    metadata = output.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("experiment output directory must be owned and private (0700)")
    manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["plan_sha256"] != file_digest(path)
        or manifest["operator_sha256"] != file_digest(Path(plan["operator_config"]))
        or any(manifest[key] != plan[key] for key in ("benchmark", "task_id", "run_id", "agent"))
    ):
        raise ValueError("experiment config changed after run")
    row = json.loads((output / "results.jsonl").read_text(encoding="utf-8"))
    return plan, operator, output, row
