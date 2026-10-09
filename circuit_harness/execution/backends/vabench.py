"""Operator-only VABench replay; the upstream evaluator owns scoring semantics."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import threading
import time
from importlib.resources import files
from pathlib import Path

from circuit_harness.execution.runtime.journal import Journal, atomic_json, file_digest
from circuit_harness.execution.runtime.process import run_process

PACKAGE = "benchmark-vabench-release-v4"
RELEASE = f"{PACKAGE}/release/benchmarkv4-r53"


def replay_result(replay: dict, process: dict) -> dict:
    status = replay.get("status")
    if status not in {
        "passed",
        "behavior_failure",
        "compile_failure",
        "runtime_failure",
        "infrastructure_failure",
        "no_submission",
    }:
        raise ValueError(f"unknown VABench status: {status!r}")
    execution = process["execution"]
    if execution == "ok" and (process["returncode"] != 0 or status == "infrastructure_failure"):
        execution = "infrastructure_error"
    return {
        "execution": execution,
        "verdict": ("pass" if status == "passed" else "fail")
        if execution == "ok"
        else "not_evaluated",
        "backend": "evas",
        "evas_version": "0.8.7",
        "benchmark_status": status,
        "certified": False,
        "scope": "frozen_submission_replay",
        "score_authority": "development_only",
    }


def candidate_files(root: Path) -> dict:
    """Hash regular files only, rejecting links and oversized operator inputs."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate must be a directory without symlinks")
    records = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is forbidden: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"non-regular file: {path}")
        records[path.relative_to(root).as_posix()] = {
            "sha256": file_digest(path),
            "bytes": path.stat().st_size,
        }
    if (
        not records
        or len(records) > 1000
        or sum(r["bytes"] for r in records.values()) > 16 * 1024 * 1024
    ):
        raise ValueError("candidate must contain 1-1000 files and at most 16 MiB")
    return records


def _digest(paths, root):
    digest = hashlib.sha256()
    count = 0
    for path in sorted(set(paths)):
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError(f"source symlink or escape: {path}")
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(file_digest(path).encode() + b"\0")
        count += 1
    return {"sha256": digest.hexdigest(), "files": count}


def source_identity(source: Path, task_id: str) -> dict:
    release = source / RELEASE
    manifest = json.loads((release / "MANIFEST.json").read_text())
    if (
        manifest.get("release_revision") != "r53"
        or manifest.get("runtime_requirements", {}).get("evas_version") != "0.8.7"
    ):
        raise ValueError("requires VABench r53 and EVAS 0.8.7")
    rows = json.loads((release / "TASK_INDEX.json").read_text())["tasks"]
    matches = [row for row in rows if row["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous task: {task_id}")
    task_dir = matches[0]["task_dir"]
    task = release / task_dir
    if Path(task_dir).is_absolute() or ".." in Path(task_dir).parts or task.is_symlink():
        raise ValueError("unsafe task directory")
    if file_digest(task / "public_contract.json") != matches[0]["public_contract_sha256"]:
        raise ValueError("task contract hash mismatch")
    code = [
        path
        for prefix in (
            "runners",
            f"{PACKAGE}/runners",
            f"{PACKAGE}/scripts",
            f"{PACKAGE}/operations",
        )
        for path in (source / prefix).rglob("*.py")
    ]
    # Only this task's hidden/public data; all shared release assets and scorer code.
    assets = [
        p
        for p in release.rglob("*")
        if "tasks" not in p.relative_to(release).parts and "__pycache__" not in p.parts
    ]
    task_files = [p for p in task.rglob("*") if "__pycache__" not in p.parts]
    return {
        "task_dir": task_dir,
        "manifest_sha256": file_digest(release / "MANIFEST.json"),
        "tree": _digest(code + assets + task_files, source),
    }


_PROBE = """import hashlib, importlib.metadata, json, pathlib, evas
root = pathlib.Path(evas.__file__).parent
h = hashlib.sha256()
for p in sorted(root.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
  h.update(p.relative_to(root).as_posix().encode()+b'\\0'+hashlib.sha256(p.read_bytes()).digest())
names = ['evas-sim','numpy','matplotlib','pandas']
versions = {n: importlib.metadata.version(n) for n in names}
print(json.dumps({'evas_tree_sha256': h.hexdigest(), 'versions': versions}))
"""


def runtime_env(python: str, directory: Path | None = None) -> dict:
    env = {
        "PATH": str(Path(python).parent) + ":/usr/bin:/bin",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "EVAS_ENGINE": "evas2",
        "VAEVAS_DEFAULT_EVAS_ENGINE": "evas2",
        "VAEVAS_REQUIRE_EVAS2_EVIDENCE": "1",
        "VAEVAS_EVAS_PERSISTENT_WORKER": "0",
        "VABENCH_EVAS_PROFILE": "r53",
    }
    if directory is not None:
        env.update(
            HOME=str(directory),
            TMPDIR=str(directory / "scratch"),
            MPLCONFIGDIR=str(directory / "scratch/matplotlib"),
        )
    return env


def pin_vabench(source: Path, python: Path, task_id: str) -> dict:
    source = Path(source).resolve()
    # Preserve the venv path: resolving its Python symlink loses the environment.
    python = str(Path(python).absolute())
    observed = subprocess.run(
        [python, "-I", "-B", "-c", _PROBE],
        env=runtime_env(python),
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    simulator = json.loads(observed.stdout)
    if simulator["versions"]["evas-sim"] != "0.8.7":
        raise ValueError("requires installed evas-sim 0.8.7")
    evas = Path(python).parent / "evas"
    return {
        "schema_version": 1,
        "source": str(source),
        "python": python,
        "python_sha256": file_digest(Path(python)),
        "evas_sha256": file_digest(evas),
        "task_id": task_id,
        "source_identity": source_identity(source, task_id),
        "simulator": simulator,
    }


def verify_pin(pin: dict) -> None:
    if pin.get("schema_version") != 1:
        raise ValueError("unsupported VABench pin schema")
    if pin_vabench(Path(pin["source"]), Path(pin["python"]), pin["task_id"]) != pin:
        raise ValueError("VABench source or simulator changed since pinning")


def vabench_identity(pin: dict, submission: Path, timeout_s=300) -> dict:
    if (
        not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or not 0 < timeout_s <= 1800
    ):
        raise ValueError("timeout must be within (0, 1800] seconds")
    verify_pin(pin)
    return {
        "backend": "vabench",
        "pin": pin,
        "candidate": candidate_files(submission),
        "timeout_s": timeout_s,
    }


def freeze_candidate(source: Path, destination: Path, identity: dict) -> None:
    if candidate_files(source) != identity["candidate"]:
        raise ValueError("candidate changed before freezing")
    shutil.copytree(source, destination)
    if candidate_files(destination) != identity["candidate"]:
        raise ValueError("candidate changed while freezing")
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(0o444)


def run_vabench(identity: dict, candidate: Path, directory: Path, cancel: threading.Event) -> dict:
    directory.mkdir(mode=0o700)
    atomic_json(directory / "identity.json", identity)
    (directory / "scratch").mkdir(mode=0o700)
    bridge = directory / "vabench_worker.py"
    bridge.write_bytes(files(__package__).joinpath("vabench_worker.py").read_bytes())
    journal = Journal(directory, call_id=directory.parent.name)
    deadline = time.monotonic() + identity["timeout_s"]
    with journal.stage("preflight"):
        verify_pin(identity["pin"])
        if candidate_files(candidate) != identity["candidate"]:
            raise ValueError("frozen candidate changed")
    python = identity["pin"]["python"]
    argv = [
        "/usr/bin/env",
        "-i",
        *[f"{k}={v}" for k, v in runtime_env(python, directory).items()],
        python,
        "-I",
        "-B",
        str(bridge),
        "replay",
        str(directory / "identity.json"),
        str(candidate),
    ]
    process = run_process(
        argv, directory=directory, stage="replay", deadline=deadline, cancel=cancel, journal=journal
    )
    replay_path = directory / "replay.json"
    if process["execution"] == "ok" and process["returncode"] == 0 and replay_path.is_file():
        with journal.stage("postflight"):
            verify_pin(identity["pin"])
            result = replay_result(json.loads(replay_path.read_text()), process)
    else:
        result = {
            "execution": process["execution"]
            if process["execution"] != "ok"
            else "infrastructure_error",
            "verdict": "not_evaluated",
            "backend": "evas",
            "certified": False,
        }
    result.update(task_id=identity["pin"]["task_id"], process=process)
    journal.emit("run_finished", **result)
    atomic_json(directory / "result.json", result)
    return result


def verify_vabench(directory: Path) -> dict:
    """Offline receipt verification; this does not rerun EVAS on the client."""
    result = json.loads((directory / "result.json").read_text())
    if result["execution"] == "ok":
        replay = json.loads((directory / "replay.json").read_text())
        projected = replay_result(replay, result["process"])
        if any(result.get(key) != value for key, value in projected.items()):
            raise ValueError("VABench structured verdict differs from result")
        receipt = replay.get("score_sidecar_receipt")
        if replay["status"] != "no_submission":
            if (
                not receipt
                or Path(receipt["path"]).is_absolute()
                or ".." in Path(receipt["path"]).parts
            ):
                raise ValueError("missing or unsafe original score sidecar receipt")
            sidecar_path = directory / "runtime" / receipt["path"]
            if file_digest(sidecar_path) != receipt["sha256"]:
                raise ValueError("original score sidecar changed")
            sidecar = json.loads(sidecar_path.read_text())
            if (
                sidecar["submission_tree_sha256"] != replay["submission_tree_sha256"]
                or sidecar["structured_result"]["status"] != replay["status"]
            ):
                raise ValueError("sidecar submission or verdict mismatch")
    return result


def export_vabench(pin: dict, output: Path) -> None:
    """Export only the original public task surface; no judge or pin is exposed."""
    verify_pin(pin)
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("public export output must not already exist")
    with tempfile.TemporaryDirectory(prefix="chips-vabench-export-") as temporary:
        work = Path(temporary)
        (work / "scratch").mkdir()
        atomic_json(work / "identity.json", {"pin": pin})
        worker = work / "worker.py"
        worker.write_bytes(files(__package__).joinpath("vabench_worker.py").read_bytes())
        subprocess.run(
            [
                pin["python"],
                "-I",
                "-B",
                str(worker),
                "export",
                str(work / "identity.json"),
                str(output),
            ],
            env=runtime_env(pin["python"], work),
            check=True,
            timeout=30,
            capture_output=True,
            text=True,
        )
    verify_pin(pin)
