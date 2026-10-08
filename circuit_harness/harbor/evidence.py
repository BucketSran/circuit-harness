"""Offline readers shared by reports and trajectory export."""

import json
from pathlib import Path

from circuit_harness.execution.archive import verify_archive
from circuit_harness.execution.journal import file_digest


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular evidence file: {path.name}")

    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in evidence")
            result[key] = item
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON value in evidence")

    value = json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"evidence must be a JSON object: {path.name}")
    return value


def checked_evaluation(directory: Path, frozen: dict, task: dict) -> tuple[dict, dict, dict]:
    archives = sorted((directory / "verifier/transport").glob("*/archive/receipt.json"))
    replay = directory / "verifier/replay/receipt.json"
    if len(archives) + int(replay.exists()) != 1:
        raise ValueError(
            "final evaluation requires exactly one independent archive or replay receipt"
        )
    if archives:
        receipt = verify_archive(archives[0].parent)
        identity = receipt["request"]["identity"]
        if identity.get("purpose") != "final":
            raise ValueError("evaluation receipt is not independent final evaluation")
        candidate = identity["candidate_manifest"]
        if any(
            candidate[key] != frozen[key]
            for key in ("task_id", "task_version", "candidate_sha256", "files")
        ):
            raise ValueError("archive candidate identity mismatch")
        final = receipt["completion"]["result"]
        sources = {
            str(archives[0].relative_to(directory)): file_digest(archives[0]),
            str((archives[0].parent / "job.tar.gz").relative_to(directory)): file_digest(
                archives[0].parent / "job.tar.gz"
            ),
        }
    else:
        from circuit_harness.execution.benchmark_replay import verify_replay

        receipt = verify_replay(replay.parent)
        final = receipt["result"]
        sources = {str(replay.relative_to(directory)): file_digest(replay)}
    for key in ("task_id", "task_version", "candidate_sha256"):
        if final.get(key) != frozen[key] or (
            key != "candidate_sha256" and final.get(key) != task.get(key)
        ):
            raise ValueError("final evaluation candidate/task identity mismatch")
    evaluation_path = directory / "verifier/evaluation.json"
    if evaluation_path.exists() and read_json(evaluation_path) != {"state": "completed", **final}:
        raise ValueError("evaluation.json differs from independently verified result")
    if final.get("purpose") != "final":
        raise ValueError("evaluation purpose mismatch")
    if replay.exists():
        identity = read_json(replay.parent / "identity.json")
    return final, sources, identity
