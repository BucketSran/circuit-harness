"""One configured Chips experiment, with task-owned simulation and scoring."""

from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path

from alphaapollo.common.execution.chips.journal import atomic_json, file_digest
from alphaapollo.workflows._chips_experiment.core import _existing, _read_plan, _write_result
from alphaapollo.workflows._chips_experiment.tasks import task_for
from alphaapollo.workflows.chips_experiment_settings import experiment_settings_snapshot


def validate(path: Path) -> int:
    """Validate plan/operator separation without creating a run or contacting services."""
    plan, operator = _read_plan(path)
    print(
        json.dumps(
            {
                "state": "config_valid",
                "benchmark": plan["benchmark"],
                "task_id": plan["task_id"],
                "run_id": plan["run_id"],
                "agent": plan["agent"],
                "experiment_settings": experiment_settings_snapshot(operator, plan["agent"]),
                "runtime_readiness": "not_checked",
            }
        )
    )
    return 0


def run(path: Path) -> int:
    plan, operator = _read_plan(path)
    output = Path(plan["output"])
    if output.exists():
        raise ValueError("use a fresh output directory for each experiment")
    output.mkdir(parents=True, mode=0o700)
    if stat.S_IMODE(output.stat().st_mode) != 0o700:
        raise ValueError("experiment output directory must be private (0700)")
    atomic_json(
        output / "experiment_manifest.json",
        {
            "schema_version": 1,
            "benchmark": plan["benchmark"],
            "task_id": plan["task_id"],
            "run_id": plan["run_id"],
            "agent": plan["agent"],
            "plan_sha256": file_digest(path),
            "operator_sha256": file_digest(Path(plan["operator_config"])),
            "experiment_settings": experiment_settings_snapshot(operator, plan["agent"]),
        },
    )
    row = {
        "schema_version": 1,
        "benchmark": plan["benchmark"],
        "task_id": plan["task_id"],
        "run_id": plan["run_id"],
        "agent": plan["agent"],
        "status": "pending",
        "score": None,
        "benchmark_success": None,
        "verdict": None,
        "error_type": None,
        "final_execution": "not_run",
        "final_validity": "not_evaluated",
        "evidence_refs": {"agent": "agent"},
    }
    _write_result(output, row)
    arguments = [
        plan["agent"],
        "--config",
        plan["operator_config"],
        "--evidence",
        str(output / "agent"),
    ]
    return task_for(plan["benchmark"], plan["agent"]).run(plan, operator, output, row, arguments)


def collect(path: Path) -> int:
    plan, operator, output, row = _existing(path)
    return task_for(plan["benchmark"], plan["agent"]).collect(plan, operator, output, row)


def finalize(path: Path) -> int:
    plan, operator, output, row = _existing(path)
    return task_for(plan["benchmark"], plan["agent"]).finalize(plan, operator, output, row)


def archive(path: Path) -> int:
    plan, operator, output, row = _existing(path)
    return task_for(plan["benchmark"], plan["agent"]).archive(plan, operator, output, row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run", "collect", "finalize", "archive"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    return {
        "validate": validate,
        "run": run,
        "collect": collect,
        "finalize": finalize,
        "archive": archive,
    }[args.command](args.config)


if __name__ == "__main__":
    raise SystemExit(main())
