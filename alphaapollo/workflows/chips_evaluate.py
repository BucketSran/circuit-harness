"""Prepare and review a fixed batch of existing one-cell Chips experiments."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import stat
import statistics
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib import error, request

from alphaapollo.common.execution.chips.journal import atomic_json, file_digest
from alphaapollo.workflows import chips_experiment
from alphaapollo.workflows._chips_experiment.core import _existing, _read_plan
from alphaapollo.workflows.chips_experiment_settings import experiment_settings_snapshot

_PIN_FIELDS = {
    "benchmark",
    "task_id",
    "condition_id",
    "agent_revision",
    "harness_revision",
    "task_revision",
    "prompt_revision",
    "toolset_revision",
    "simulator_revision",
    "scorer_revision",
    "memory_snapshot",
}


def _condition_pins(config: dict) -> dict[tuple[str, str, str], dict]:
    if config["schema_version"] == 1:
        return {}
    pins = {}
    entries = config["conditions"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("v2 batch needs condition pins")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _PIN_FIELDS:
            raise ValueError("unsupported condition pin")
        if any(
            not isinstance(entry[field], str) or not entry[field].strip() or len(entry[field]) > 200
            for field in _PIN_FIELDS - {"memory_snapshot"}
        ):
            raise ValueError("condition pin fields must be nonempty version identifiers")
        memory = entry["memory_snapshot"]
        if memory is not None and (
            not isinstance(memory, str) or not re.fullmatch(r"[0-9a-f]{64}", memory)
        ):
            raise ValueError("memory_snapshot must be null or a SHA-256 digest")
        key = entry["benchmark"], entry["task_id"], entry["condition_id"]
        if key in pins:
            raise ValueError("duplicate condition pin")
        pins[key] = entry
    return pins


def _load_batch(path: Path) -> tuple[Path, list[dict]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("batch config must be a regular file")
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or type(config.get("schema_version")) is not int
        or config["schema_version"] not in (1, 2)
        or set(config)
        != (
            {"schema_version", "output", "cells"}
            if config["schema_version"] == 1
            else {"schema_version", "output", "cells", "conditions"}
        )
        or not isinstance(config["output"], str)
        or not Path(config["output"]).is_absolute()
        or not isinstance(config["cells"], list)
        or not config["cells"]
    ):
        raise ValueError("unsupported batch config")
    root = Path(config["output"])
    pins = _condition_pins(config)
    cells = []
    run_ids, sessions, conditions, used_pins = set(), set(), {}, set()
    for entry in config["cells"]:
        if not isinstance(entry, dict) or set(entry) != {"condition_id", "experiment"}:
            raise ValueError("unsupported batch cell")
        condition = entry["condition_id"]
        if not isinstance(condition, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", condition
        ):
            raise ValueError("invalid condition_id")
        experiment = entry["experiment"]
        if not isinstance(experiment, str) or not Path(experiment).is_absolute():
            raise ValueError("experiment must be an absolute path")
        plan, operator = _read_plan(Path(experiment))
        run_id = plan["run_id"]
        if run_id in run_ids or Path(plan["output"]) != root / "cells" / run_id:
            raise ValueError("batch cells need unique run IDs under output/cells")
        session = operator.get("session")
        if not isinstance(session, str) or session in sessions:
            raise ValueError("batch cells need unique operator sessions")
        settings = experiment_settings_snapshot(operator, plan["agent"])
        pin_key = plan["benchmark"], plan["task_id"], condition
        pin = pins.get(pin_key)
        if config["schema_version"] == 2 and pin is None:
            raise ValueError("batch cell is missing a condition pin")
        if pin is not None:
            used_pins.add(pin_key)
        signature = (
            plan["agent"],
            operator.get("transport", "ssh"),
            operator.get("policy_kind"),
            operator.get("provider_kind"),
            operator.get("auth_kind"),
            json.dumps(settings, sort_keys=True),
            pin["agent_revision"] if pin is not None else None,
            pin["harness_revision"] if pin is not None else None,
            pin["memory_snapshot"] if pin is not None else None,
        )
        if condition in conditions and conditions[condition] != signature:
            raise ValueError("condition_id has inconsistent agent or model settings")
        conditions[condition] = signature
        run_ids.add(run_id)
        sessions.add(session)
        cell = {
            "condition_id": condition,
            "experiment": experiment,
            "experiment_sha256": file_digest(Path(experiment)),
            "operator_sha256": file_digest(Path(plan["operator_config"])),
            "run_id": run_id,
            "benchmark": plan["benchmark"],
            "task_id": plan["task_id"],
            "agent": plan["agent"],
            "transport": operator.get("transport", "ssh"),
            "model_settings": settings,
            "output": plan["output"],
        }
        if pin is not None:
            cell["condition_pin"] = pin
        cells.append(cell)
    if used_pins != pins.keys():
        raise ValueError("condition pin is not referenced by any batch cell")
    return root, cells


def _write_json(path: Path, value: object) -> None:
    atomic_json(path, value)
    path.chmod(0o600)


def _write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _not_run(cell: dict) -> dict:
    row = {
        "schema_version": 1,
        "condition_id": cell["condition_id"],
        "benchmark": cell["benchmark"],
        "task_id": cell["task_id"],
        "run_id": cell["run_id"],
        "agent": cell["agent"],
        "status": "not_run",
        "score": None,
        "benchmark_success": None,
    }
    if "condition_pin" in cell:
        row["condition_pin"] = cell["condition_pin"]
    return row


def _episode_counts(rows: list[dict]) -> dict:
    submission = {"submitted": 0, "not_submitted": 0, "unknown": 0}
    simulation = {"yes": 0, "no": 0, "unknown": 0}
    termination: dict[str, int] = {}
    collection: dict[str, int] = {}
    for row in rows:
        submitted = row.get("agent_submitted")
        submission[
            "submitted"
            if submitted is True
            else "not_submitted"
            if submitted is False
            else "unknown"
        ] += 1
        simulated = row.get("final_candidate_publicly_simulated")
        simulation["yes" if simulated is True else "no" if simulated is False else "unknown"] += 1
        reason = row.get("termination_reason") or "unavailable"
        termination[reason] = termination.get(reason, 0) + 1
        source = row.get("collection_source") or "unavailable"
        collection[source] = collection.get(source, 0) + 1
    return {
        "agent_submission_counts": submission,
        "collection_source_counts": collection,
        "termination_counts": termination,
        "final_candidate_publicly_simulated_counts": simulation,
    }


def _summary(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    groups: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
        key = row["benchmark"], row["task_id"], row["condition_id"]
        group = groups.setdefault(
            key,
            {
                "benchmark": key[0],
                "task_id": key[1],
                "condition_id": key[2],
                "condition_pin": row.get("condition_pin"),
                "planned": 0,
                "started": 0,
                "completed": 0,
                "metric_valid": 0,
                "binary_valid": 0,
                "binary_successes": 0,
                "score_count": 0,
                "score_values": [],
                "episode_rows": [],
            },
        )
        group["episode_rows"].append(row)
        group["planned"] += 1
        if status != "not_run":
            group["started"] += 1
        if status == "completed":
            group["completed"] += 1
            if row.get("final_validity") == "valid":
                has_metric = False
                if isinstance(row.get("benchmark_success"), bool):
                    group["binary_valid"] += 1
                    group["binary_successes"] += int(row["benchmark_success"])
                    has_metric = True
                score = row.get("score")
                if type(score) in (int, float) and math.isfinite(score):
                    group["score_count"] += 1
                    group["score_values"].append(score)
                    has_metric = True
                if has_metric:
                    group["metric_valid"] += 1
    summary_groups = []
    for group in groups.values():
        group.update(_episode_counts(group.pop("episode_rows")))
        values = group.pop("score_values")
        group["metric_coverage"] = group["metric_valid"] / group["planned"]
        group["binary_success_rate"] = (
            group["binary_successes"] / group["binary_valid"] if group["binary_valid"] else None
        )
        group["binary_successes_per_planned"] = (
            group["binary_successes"] / group["planned"] if group["binary_valid"] else None
        )
        group["score_mean"] = statistics.mean(values) if values else None
        group["score_median"] = statistics.median(values) if values else None
        group["score_sample_std"] = statistics.stdev(values) if len(values) > 1 else None
        group["score_min"] = min(values) if values else None
        group["score_max"] = max(values) if values else None
        summary_groups.append(group)
    return {
        "schema_version": 1,
        "planned": len(rows),
        "completed": counts.get("completed", 0),
        "status_counts": counts,
        **_episode_counts(rows),
        "groups": summary_groups,
    }


def prepare(path: Path) -> int:
    root, cells = _load_batch(path)
    if root.exists():
        raise ValueError("use a fresh batch output directory")
    root.mkdir(parents=True, mode=0o700)
    _write_json(
        root / "batch_manifest.json",
        {"schema_version": 1, "config_sha256": file_digest(path), "cells": cells},
    )
    rows = [_not_run(cell) for cell in cells]
    _write_rows(root / "results.jsonl", rows)
    _write_json(root / "summary.json", _summary(rows))
    return 0


def _prepared(path: Path) -> tuple[Path, list[dict]]:
    root, cells = _load_batch(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("batch output is unavailable")
    metadata = root.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("batch output directory must be owned and private (0700)")
    manifest_path = root / "batch_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("batch manifest is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("config_sha256") != file_digest(path)
        or manifest.get("cells") != cells
    ):
        raise ValueError("batch config or cell inputs changed after prepare")
    return root, cells


def _private_subdir(root: Path, name: str, *, create: bool = False) -> Path:
    directory = root / name
    if directory.is_symlink():
        raise ValueError(f"batch {name} directory must not be a symbolic link")
    if create:
        directory.mkdir(mode=0o700, exist_ok=True)
    if directory.exists():
        metadata = directory.stat()
        if (
            not directory.is_dir()
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ValueError(f"batch {name} directory must be owned and private (0700)")
    return directory


def _intent(root: Path, cell: dict) -> bool:
    path = _private_subdir(root, "intents") / f"{cell['run_id']}.json"
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or not path.is_file():
        raise ValueError("batch start intent is not a regular file")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded != {
        "run_id": cell["run_id"],
        "experiment_sha256": cell["experiment_sha256"],
    }:
        raise ValueError("batch start intent differs from the frozen cell")
    return True


def report(path: Path) -> int:
    root, cells = _prepared(path)
    rows = []
    damaged = False
    for cell in cells:
        output = Path(cell["output"])
        try:
            started = _intent(root, cell)
        except (OSError, ValueError, TypeError) as error:
            damaged = True
            rows.append(
                {**_not_run(cell), "status": "evidence_error", "error_type": type(error).__name__}
            )
            continue
        if not output.exists() and not output.is_symlink():
            if started:
                damaged = True
                rows.append({**_not_run(cell), "status": "unknown_execution"})
            else:
                rows.append(_not_run(cell))
            continue
        try:
            _, _, _, row = _existing(Path(cell["experiment"]))
            from alphaapollo.workflows._chips_experiment.core import episode_facts

            row.update(episode_facts(output))
            if any(
                row.get(key) != cell[key] for key in ("benchmark", "task_id", "run_id", "agent")
            ):
                raise ValueError("single-experiment result identity differs from batch cell")
            if not isinstance(row.get("status"), str):
                raise ValueError("single-experiment result has no status")
        except (OSError, ValueError, KeyError, TypeError) as error:
            damaged = True
            rows.append(
                {**_not_run(cell), "status": "evidence_error", "error_type": type(error).__name__}
            )
            continue
        row = {**row, "condition_id": cell["condition_id"]}
        if "condition_pin" in cell:
            row["condition_pin"] = cell["condition_pin"]
        try:
            phase = _phase_for(cell["benchmark"], row["status"])
        except ValueError:
            phase = None
        try:
            if phase is not None and _phase_history(root, cell, phase):
                row["single_status"] = row["status"]
                row["status"] = "unknown_execution"
                row["error_type"] = "batch_phase_unresolved"
                damaged = True
        except (OSError, ValueError, TypeError) as error:
            row["single_status"] = row["status"]
            row["status"] = "evidence_error"
            row["error_type"] = type(error).__name__
            damaged = True
        rows.append(row)
    _write_rows(root / "results.jsonl", rows)
    _write_json(root / "summary.json", _summary(rows))
    return 2 if damaged else 0


def _batch_preflight(root: Path, cells: list[dict]) -> int:
    """Check every pending real-model cell before launching any of them."""
    from alphaapollo.workflows import chips_analog_agent, chips_vabench_agent

    pending = [cell for cell in cells if not Path(cell["output"]).exists()]
    checks = []
    for cell in pending:
        plan, operator = _read_plan(Path(cell["experiment"]))
        if operator.get("policy_kind") != "remote_model":
            continue
        if cell["agent"] == "pi" and operator.get("provider_kind", "glm_coding") == "glm_coding":
            if "key_file" in plan and os.environ.get("CHIPS_MODEL_KEY"):
                checks.append(
                    {"run_id": cell["run_id"], "state": "blocked", "reason": "credential_conflict"}
                )
            elif "key_file" in plan:
                try:
                    chips_analog_agent.load_private_key(Path(plan["key_file"]))
                except (OSError, UnicodeError, ValueError):
                    checks.append(
                        {
                            "run_id": cell["run_id"],
                            "state": "blocked",
                            "reason": "credential_invalid",
                        }
                    )
            elif not os.environ.get("CHIPS_MODEL_KEY"):
                checks.append(
                    {"run_id": cell["run_id"], "state": "blocked", "reason": "credential_missing"}
                )
    if not checks:
        for cell in pending:
            plan, operator = _read_plan(Path(cell["experiment"]))
            if operator.get("policy_kind") != "remote_model":
                continue
            try:
                if cell["benchmark"] == "analog_design_bench":
                    result = chips_analog_agent.preflight_pilot(
                        operator,
                        key_file=Path(plan["key_file"]) if "key_file" in plan else None,
                    )
                    check = {
                        "run_id": cell["run_id"],
                        "state": result["state"],
                        **result["checks"],
                    }
                else:
                    launcher = _pi_launcher_checks(operator) if cell["agent"] == "pi" else {}
                    if all(value == "ready" for value in launcher.values()):
                        evidence = _private_subdir(root, "preflight_transport", create=True)
                        remote = chips_vabench_agent.transport(operator, evidence / cell["run_id"])
                        result = remote.cli(
                            "vabench-preflight", "--session", operator["session"], timeout=60
                        )
                    else:
                        result = {}
                    check = {
                        "run_id": cell["run_id"],
                        "state": (
                            "ready"
                            if result.get("state") == "ready"
                            and result.get("task_id") == cell["task_id"]
                            else "blocked"
                        ),
                        "public_session": (
                            "ready" if result.get("state") == "ready" else "unavailable"
                        ),
                        "task_identity": (
                            "ready"
                            if result.get("task_id") == cell["task_id"]
                            else "mismatch"
                            if result.get("task_id") is not None
                            else "not_checked"
                        ),
                        **launcher,
                    }
            except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
                check = {
                    "run_id": cell["run_id"],
                    "state": "blocked",
                    "reason": type(error).__name__,
                }
            checks.append(check)
    if any(check["state"] != "ready" for check in checks):
        _write_json(
            root / "preflight.json", {"schema_version": 1, "state": "blocked", "checks": checks}
        )
        return 2
    _write_json(root / "preflight.json", {"schema_version": 1, "state": "ready", "checks": checks})
    return 0


def _pi_launcher_checks(operator: dict) -> dict[str, str]:
    """Check the VABench Pi launcher's local dependencies and direct model route."""
    python = operator["python"] if operator.get("transport", "ssh") == "local" else sys.executable
    checks = {
        "python": ("ready" if Path(python).is_file() and os.access(python, os.X_OK) else "missing"),
        "pi_cli": (
            "ready"
            if Path(operator["pi_cli"]).is_file() and os.access(operator["pi_cli"], os.X_OK)
            else "missing"
        ),
    }
    if operator.get("transport", "ssh") == "local":
        checks["bundle"] = "ready" if zipfile.is_zipfile(operator["bundle"]) else "invalid"
    if checks["python"] == "ready":
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "HOME", "USER", "LOGNAME", "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME"}
        }
        try:
            process = subprocess.run(
                [python, "-c", "import mcp, anyio"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                env=environment,
                check=False,
            )
            checks["python_deps"] = "ready" if process.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            checks["python_deps"] = "unavailable"
    if (
        all(value == "ready" for value in checks.values())
        and operator.get("provider_kind", "glm_coding") == "glm_coding"
    ):
        try:
            opener = request.build_opener(request.ProxyHandler({}))
            with opener.open(request.Request(operator["base_url"], method="HEAD"), timeout=5):
                pass
            checks["model_https"] = "ready"
        except error.HTTPError as response:
            checks["model_https"] = "ready" if 400 <= response.code < 500 else "unavailable"
        except (OSError, ValueError):
            checks["model_https"] = "unavailable"
    return checks


def preflight(path: Path) -> int:
    root, cells = _prepared(path)
    with (root / "batch_manifest.json").open("r+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if report(path) != 0:
            return 2
        return _batch_preflight(root, cells)


def run(path: Path) -> int:
    root, cells = _prepared(path)
    with (root / "batch_manifest.json").open("r+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _private_subdir(root, "cells", create=True)
        if report(path) != 0:
            return 2
        if _batch_preflight(root, cells) != 0:
            return 2
        for cell in cells:
            output = Path(cell["output"])
            if output.exists() or output.is_symlink():
                continue
            intents = _private_subdir(root, "intents", create=True)
            _write_json(
                intents / f"{cell['run_id']}.json",
                {
                    "run_id": cell["run_id"],
                    "experiment_sha256": cell["experiment_sha256"],
                },
            )
            try:
                result = chips_experiment.run(Path(cell["experiment"]))
            finally:
                refreshed = report(path)
            if refreshed != 0:
                return refreshed
            if result != 0:
                return result
    return 0


def _phase_for(benchmark: str, status: str) -> str | None:
    if status in {"not_run", "completed", "unsubmitted", "blocked", "error"}:
        return None
    if benchmark == "vabench" and status == "pending":
        return "collect"
    if benchmark == "analog_design_bench":
        if status == "awaiting_action_recovery":
            return "collect"
        if status == "awaiting_final":
            return "finalize"
        if status in {"missing_candidate", "final_graded", "archive_failed"}:
            return "archive"
    raise ValueError(f"batch cell has an unresolved status: {status}")


def _phase_history(root: Path, cell: dict, phase: str) -> bool:
    actions = _private_subdir(root, "phase_actions")
    directory = _private_subdir(actions, cell["run_id"])
    if not directory.exists():
        return False
    previous = sorted(directory.glob(f"{phase}-*.json"))
    for index, path in enumerate(previous, start=1):
        if path.name != f"{phase}-{index:04d}.json" or path.is_symlink() or not path.is_file():
            raise ValueError("batch phase action record is damaged")
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if (
            recorded.get("run_id") != cell["run_id"]
            or recorded.get("phase") != phase
            or recorded.get("experiment_sha256") != cell["experiment_sha256"]
            or recorded.get("state") not in {"started", "finished"}
        ):
            raise ValueError("batch phase action record differs from the frozen cell")
        if index < len(previous) and recorded["state"] != "finished":
            raise ValueError("batch phase action history has an unfinished attempt")
        if index == len(previous) and recorded["state"] == "started":
            return True
    return False


def _phase_attempt(root: Path, cell: dict, phase: str) -> Path:
    if _phase_history(root, cell, phase):
        raise ValueError("batch phase may have executed; inspect its original output")
    actions = _private_subdir(root, "phase_actions", create=True)
    directory = _private_subdir(actions, cell["run_id"], create=True)
    previous = sorted(directory.glob(f"{phase}-*.json"))
    intent = directory / f"{phase}-{len(previous) + 1:04d}.json"
    _write_json(
        intent,
        {
            "run_id": cell["run_id"],
            "phase": phase,
            "experiment_sha256": cell["experiment_sha256"],
            "state": "started",
        },
    )
    return intent


def reconcile(path: Path) -> int:
    root, cells = _prepared(path)
    with (root / "batch_manifest.json").open("r+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if report(path) != 0:
            return 2
        for cell in cells:
            output = Path(cell["output"])
            if not output.exists():
                continue
            _, _, _, row = _existing(Path(cell["experiment"]))
            try:
                phase = _phase_for(cell["benchmark"], row["status"])
            except ValueError:
                return 2
            if phase is None:
                continue
            try:
                intent = _phase_attempt(root, cell, phase)
            except (OSError, ValueError, TypeError):
                return 2
            action = getattr(chips_experiment, phase)
            try:
                result = action(Path(cell["experiment"]))
                _write_json(
                    intent,
                    {
                        "run_id": cell["run_id"],
                        "phase": phase,
                        "experiment_sha256": cell["experiment_sha256"],
                        "state": "finished",
                        "exit_code": result,
                    },
                )
            finally:
                refreshed = report(path)
            if refreshed != 0:
                return refreshed
            if result != 0:
                return result
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run", "reconcile", "report"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    return {
        "prepare": prepare,
        "preflight": preflight,
        "run": run,
        "reconcile": reconcile,
        "report": report,
    }[args.command](args.config)


if __name__ == "__main__":
    raise SystemExit(main())
