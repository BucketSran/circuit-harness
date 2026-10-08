"""Independent saved-candidate replay; commercial comparison is offline only.

Mappings, solver options and unsupported analyses are benchmark declarations.
This module never starts or probes SSH, Spectre, licenses or a model.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import stat
from pathlib import Path

from .benchmark_spectre import package_identity, seal_result, stage_inputs, verify_benchmark
from .candidate_bundle import verify_candidate
from .journal import atomic_json, file_digest
from .vabench import candidate_files

_IDENTIFIERS = (
    "candidate_sha256",
    "task_id",
    "task_version",
    "criteria_sha256",
    "condition_id",
    "task_set",
)
_INFRA = {
    "infrastructure_error",
    "timeout",
    "cancelled",
    "output_limit",
    "cleanup_failed",
    "license_unavailable",
    "dependency_unavailable",
    "simulator_unavailable",
    "preflight_unknown",
    "invalid_preflight",
}
_CONFIG_KEYS = {
    "schema_version",
    "image",
    "task_package_sha256",
    "solver_options",
    "unsupported",
    "timeout_s",
    "max_output_bytes",
}


def compare_results(opensource: dict, spectre: dict | None = None) -> str:
    """Compare full task scores; partial-score disagreements remain unevaluable."""
    if spectre is not None:
        for key in _IDENTIFIERS:
            if not opensource.get(key) or opensource[key] != spectre.get(key):
                raise ValueError(f"comparison identity differs: {key}")
    records = [opensource] + ([spectre] if spectre is not None else [])
    if any(record.get("execution") in _INFRA for record in records):
        return "infra"
    if any(
        record.get("execution") != "ok"
        or type(record.get("score")) not in (int, float)
        or not math.isfinite(record["score"])
        or not 0 <= record["score"] <= 1
        for record in records
    ):
        return "unevaluable"
    if spectre is None:
        return "not_compared"
    if opensource["score"] == spectre["score"]:
        return "match"
    if opensource["score"] == 1:
        return "false_accept"
    if spectre["score"] == 1:
        return "false_reject"
    return "unevaluable"


def _configuration(config: dict, package: dict) -> dict:
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS or config["schema_version"] != 1:
        raise ValueError("unsupported open-source replay configuration")
    if not isinstance(config["image"], str) or not re.fullmatch(
        r"(?:[^\s]+@)?sha256:[a-f0-9]{64}", config["image"]
    ):
        raise ValueError("replay image must be digest-pinned")
    if config["task_package_sha256"] != package["sha256"]:
        raise ValueError("replay mapping differs from declared task package")
    if (
        not isinstance(config["solver_options"], dict)
        or not isinstance(config["unsupported"], list)
        or any(not isinstance(v, str) for v in config["unsupported"])
    ):
        raise ValueError("solver options and unsupported analyses must be explicit")
    value = config["timeout_s"]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 300:
        raise ValueError("invalid replay timeout")
    if (
        type(config["max_output_bytes"]) is not int
        or not 1 <= config["max_output_bytes"] <= 16 * 1024 * 1024
    ):
        raise ValueError("invalid replay output limit")
    return config


def _spectre_record(directory: Path) -> dict:
    # Imports are delayed so opensrc-only does not even load commercial dispatch.
    from .archive import verify_archive
    from .jobs import verify_job

    if (directory / "receipt.json").is_file():
        result = verify_archive(directory)["completion"]["result"]
    else:
        result = verify_job(directory)["result"]
    if result.get("backend") != "spectre" or result.get("purpose") != "final":
        raise ValueError("comparison requires independently verified Spectre final evidence")
    return result


def _save_spectre(source: Path, destination: Path) -> dict:
    _spectre_record(source)
    destination.mkdir(mode=0o700)
    if (source / "receipt.json").is_file():
        names = ["receipt.json", "job.tar.gz"]
    else:
        completion = json.loads((source / "completion.json").read_text())
        names = ["completion.json", *completion["artifacts"]]
    for name in names:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, path)
    return _spectre_record(destination)


def replay_candidate(
    candidate: Path,
    task_package: Path,
    config: dict,
    output: Path,
    *,
    spectre_record: Path | None = None,
    cancel=None,
) -> dict:
    """Run one benchmark-declared open-source checker without replacing old records."""
    frozen = verify_candidate(candidate)
    package = package_identity(task_package, purpose="final")
    config = _configuration(config, package)
    if any(frozen[k] != package["manifest"][k] for k in ("task_id", "task_version")):
        raise ValueError("replay candidate/task identity differs")
    candidate_file = package["manifest"]["candidate_file"]
    if candidate_file not in frozen["files"]:
        raise ValueError("replay candidate deliverable is missing")
    output = Path(output)
    # Never resume or overwrite a previous version's evaluation record.
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    identity = {
        "backend": "benchmark_opensource",
        "purpose": "final",
        "candidate_manifest": {k: v for k, v in frozen.items() if k != "reason"},
        "candidate": candidate_files(candidate),
        "package": package,
        "configuration": config,
    }
    stage_inputs(candidate, task_package, output, identity)
    atomic_json(output / "identity.json", identity)
    spectre = (
        _save_spectre(Path(spectre_record), output / "spectre-reference")
        if spectre_record
        else None
    )
    if spectre is not None:
        probe = {k: frozen[k] if k in frozen else package["manifest"][k] for k in _IDENTIFIERS}
        compare_results({**probe, "execution": "ok", "score": 1}, spectre)
    run = output / "run"
    run.mkdir(mode=0o700)
    work = run / "work"
    work.mkdir(mode=0o700)
    atomic_json(run / "identity.json", identity)
    if config["unsupported"]:
        process = {
            "execution": "unsupported_analysis",
            "returncode": None,
            "unsupported": config["unsupported"],
        }
    else:
        from .current_evas_public import run_isolated_docker

        report_parent = Path(package["manifest"]["report_path"]).parent.as_posix()
        command = [
            "/usr/bin/env",
            f"CANDIDATE=/candidate/{candidate_file}",
            f"VERIFY_OUTPUT=/output/{report_parent}",
            f"CHIPS_SOLVER_OPTIONS={json.dumps(config['solver_options'], sort_keys=True)}",
            "/bin/sh",
            f"/task/{package['manifest']['entrypoint']}",
        ]
        process = run_isolated_docker(
            image=config["image"],
            readonly={output / "candidate/files": "/candidate", output / "task-package": "/task"},
            writable={work: "/output"},
            command=command,
            directory=run,
            action_id="replay",
            timeout_s=config["timeout_s"],
            max_output_bytes=config["max_output_bytes"],
            cancel=cancel,
        )
    result = seal_result(run, identity, process)
    classification = compare_results(result, spectre)
    receipt = {
        "schema_version": 1,
        "result": result,
        "spectre": spectre,
        "classification": classification,
        "summary": {result["task_set"]: {classification: 1, "samples": 1}},
        "artifacts": _evidence_inventory(output),
    }
    atomic_json(output / "receipt.json", receipt)
    verify_replay(output)
    return receipt


def _evidence_inventory(directory: Path) -> dict:
    records = {}
    total = 0
    for path in sorted(directory.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("replay evidence must contain only regular files")
        total += info.st_size
        records[path.relative_to(directory).as_posix()] = {
            "bytes": info.st_size,
            "sha256": file_digest(path),
        }
        if len(records) > 20000 or total > 1024 * 1024 * 1024:
            raise ValueError("replay evidence exceeds inventory limits")
    return records


def verify_replay(output: Path) -> dict:
    """Verify saved replay evidence offline; do not start either backend."""
    output = Path(output)
    receipt = json.loads((output / "receipt.json").read_text())
    actual = _evidence_inventory(output)
    if {k: v for k, v in actual.items() if k != "receipt.json"} != receipt["artifacts"]:
        raise ValueError("replay evidence missing, modified or undeclared")
    identity = json.loads((output / "identity.json").read_text())
    if json.loads((output / "run/identity.json").read_text()) != identity:
        raise ValueError("replay run identity differs")
    _configuration(identity["configuration"], identity["package"])
    if verify_benchmark(output / "run") != receipt["result"]:
        raise ValueError("replay result differs from checker evidence")
    spectre = (
        _spectre_record(output / "spectre-reference") if receipt["spectre"] is not None else None
    )
    if (
        spectre != receipt["spectre"]
        or compare_results(receipt["result"], spectre) != receipt["classification"]
    ):
        raise ValueError("replay comparison differs from verified evidence")
    if receipt["summary"] != {
        receipt["result"]["task_set"]: {receipt["classification"]: 1, "samples": 1}
    }:
        raise ValueError("replay summary differs from individual results")
    return receipt


def summarize_replays(directories: list[Path]) -> dict:
    """Count all verified records, including failures, separately by task set."""
    categories = ("match", "false_accept", "false_reject", "infra", "unevaluable", "not_compared")
    summary = {
        task_set: {"records": 0, "candidates": 0, **dict.fromkeys(categories, 0)}
        for task_set in ("public", "extension")
    }
    candidates = {"public": set(), "extension": set()}
    seen = set()
    for directory in directories:
        directory = Path(directory).resolve()
        if directory in seen:
            raise ValueError("duplicate replay record in summary")
        seen.add(directory)
        receipt = verify_replay(directory)
        task_set = receipt["result"]["task_set"]
        summary[task_set]["records"] += 1
        summary[task_set][receipt["classification"]] += 1
        candidates[task_set].add(receipt["result"]["candidate_sha256"])
    for task_set in candidates:
        summary[task_set]["candidates"] = len(candidates[task_set])
    return summary
