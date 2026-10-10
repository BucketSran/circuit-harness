"""Bounded execution of an explicitly selected EVAS checkout; no task grading."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import threading
import time
from importlib.resources import files
from pathlib import Path

from circuit_harness.execution.runtime.journal import _SOURCE_SHA256 as _JOURNAL_SOURCE_SHA256
from circuit_harness.execution.runtime.journal import Journal, atomic_json, digest, file_digest
from circuit_harness.execution.runtime.process import _SOURCE_SHA256 as _PROCESS_SOURCE_SHA256
from circuit_harness.execution.runtime.process import run_process

EVAS_SOURCE = (
    "evas/src",
    "evas/rust_core/src",
    "evas/rust_core/ir",
    "evas/rust_core/Cargo.toml",
    "evas/rust_core/Cargo.lock",
)
HARNESS_SOURCE = (
    "circuit_harness/__init__.py",
    "circuit_harness/execution/__init__.py",
    "circuit_harness/execution/current_evas.py",
    "circuit_harness/execution/process.py",
    "circuit_harness/execution/journal.py",
    "circuit_harness/execution/backends/__init__.py",
    "circuit_harness/execution/runtime/__init__.py",
    "circuit_harness/execution/backends/current_evas.py",
    "circuit_harness/execution/runtime/process.py",
    "circuit_harness/execution/runtime/journal.py",
)
_HARNESS_ROOT = Path(__file__).resolve().parents[3]
# Package resources read the actual distributed bytes in both checkouts and
# zipapps; __file__ paths inside a zip are not ordinary filesystem directories.
_LOADED_HARNESS = {
    name: digest(
        files("circuit_harness").joinpath(name.removeprefix("circuit_harness/")).read_bytes()
    )
    for name in HARNESS_SOURCE
}
_LOADED_HARNESS["circuit_harness/execution/runtime/journal.py"] = _JOURNAL_SOURCE_SHA256
_LOADED_HARNESS["circuit_harness/execution/runtime/process.py"] = _PROCESS_SOURCE_SHA256
# -I -S excludes ambient PYTHONPATH/site customizations; the selected source snapshot
# is inserted explicitly. The CLI remains EVAS's owner of compile/solve semantics.
_CLI = """
import json, os, pathlib, platform, runpy, sys
source, identity = sys.argv[1:3]
del sys.argv[1:3]
# Inherited engine controls must not change this explicit local request or write
# diagnostics outside its owned output directory.
for key in list(os.environ):
    if key.startswith('EVAS_'):
        del os.environ[key]
os.environ['EVAS_STATIC_THREADS'] = '1'
sys.path.insert(0, source)
import evas
pathlib.Path(identity).write_text(json.dumps(dict(
    python=sys.version, executable=sys.executable, platform=platform.platform(),
    evas_module=evas.__file__, environment={'EVAS_STATIC_THREADS':'1'}), indent=2)+'\\n')
runpy.run_module('evas', run_name='__main__')
"""


def snapshot_repository(root: Path, output: Path, paths: tuple[str, ...]) -> dict:
    """Save selected current source bytes and their Git context, including new files.

    Paths are repository-relative scopes chosen by the owner, not a whole-workspace
    archive. Unrelated dirty paths are named by status but their contents are not copied.
    """
    root, output = Path(root).resolve(), Path(output).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], timeout=15)

    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ValueError("checkout must be a Git repository root")
    for name in paths:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("source scope must be repository-relative")
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *paths)
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for name in sorted(set(filter(None, names.decode().split("\0")))):
        source = root / name
        if not source.exists():
            continue  # A tracked deletion is captured in the patch and status.
        if source.is_symlink() or root not in source.resolve().parents or not source.is_file():
            raise ValueError(f"source must be a regular file within checkout: {name}")
        target = output / "files" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records[name] = {"sha256": file_digest(target), "bytes": target.stat().st_size}
    (output / "changes.patch").write_bytes(git("diff", "HEAD", "--binary", "--", *paths))
    identity = {
        "checkout": str(root),
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "status": git("status", "--porcelain=v1", "--untracked-files=all").decode(),
        "scope": list(paths),
        "files": records,
        "patch_sha256": file_digest(output / "changes.patch"),
        "availability": "local-only",
    }
    atomic_json(output / "identity.json", identity)
    return identity


def _read_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=constant)


def read_diagnostic(path: Path) -> dict:
    """Project v1 metadata only; complete stderr remains a private artifact.

    Missing, malformed and future payloads stay unknown. Never infer a category
    from a kind prefix or message. This evidence cannot change execution/grading.
    """
    result = dict(
        diagnostic_version=None, code=None, category="unknown", stage=None, capability=None
    )
    try:
        # Bound parsing separately from retained log size. Large failures still
        # count and retain their original bytes, even when metadata is unknown.
        if path.stat().st_size > 1024 * 1024:
            return result
        payload = _read_json(path)
        if not isinstance(payload, dict):
            return result
        version = payload.get("diagnostic_version")
        if type(version) is int and 0 <= version <= 2**31 - 1:
            result["diagnostic_version"] = version
        if type(version) is not int or version != 1:
            return result
        categories = {
            "unknown",
            "invalid_input",
            "unsupported",
            "version",
            "numerical",
            "resource",
            "infrastructure",
            "internal",
            "protocol",
        }

        def token(value):
            return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", value)

        if (
            not token(payload.get("kind"))
            or not token(payload.get("code"))
            or not token(payload.get("stage"))
            or not isinstance(payload.get("message"), str)
            or not isinstance(payload.get("category"), str)
            or payload["category"] not in categories
            or not (payload.get("capability") is None or token(payload["capability"]))
        ):
            return result
        result.update(
            {key: payload.get(key) for key in ("code", "category", "stage", "capability")}
        )
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        pass
    return result


def _validate_manifest(config: dict) -> None:
    for label, value, fields in (
        ("manifest", config, {"models", "instances", "transient", "tolerances"}),
        ("transient", config.get("transient", {}), {"sources", "output_times", "stop", "max_step"}),
        ("tolerances", config.get("tolerances", {}), {"vabstol", "reltol"}),
    ):
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError(f"{label} requires exactly {sorted(fields)}; got {value}")
    if not isinstance(config["models"], list) or not config["models"]:
        raise ValueError("models must be a nonempty list")
    for name in config["models"]:
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or name in {"manifest.json", "evas-kernel"}
        ):
            raise ValueError("model paths must be relative and not overlap execution inputs")


def _validate_result(data: dict, manifest: dict) -> None:
    """Check artifact completeness; EVAS itself validates its full IR protocol."""
    nodes, solutions, trace = data["nodes"], data["solutions"], data["transient"]
    times = manifest["transient"]["output_times"]
    if (
        not isinstance(data["engine"], str)
        or not data["engine"]
        or not isinstance(nodes, list)
        or not nodes
        or any(not isinstance(n, str) for n in nodes)
        or len(set(nodes)) != len(nodes)
        or trace["times"] != times
        or len(solutions) != len(times)
        or not isinstance(trace["events"], list)
    ):
        raise ValueError("incomplete EVAS result or mismatched observation times")
    for row in solutions:
        voltages = row["voltages"]
        if (
            not isinstance(voltages, list)
            or len(voltages) != len(nodes)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in voltages)
        ):
            raise ValueError("invalid EVAS voltage row")


def run_evas(
    *,
    checkout: Path,
    kernel: Path,
    manifest: Path,
    output: Path,
    timeout_s: float,
    max_output_bytes: int,
    python: str | Path,
) -> dict:
    """Execute a transient manifest once in a new private directory.

    This trusted local operator API is resource control, not a candidate sandbox.
    The caller owns task mapping and grading; execution never assigns a score.
    """
    checkout, kernel, manifest, output = (
        Path(p).resolve() for p in (checkout, kernel, manifest, output)
    )
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    if type(max_output_bytes) is not int or max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be a positive integer")
    config = _read_json(manifest)
    if not isinstance(config, dict):
        raise ValueError("manifest must be an object")
    _validate_manifest(config)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in config["models"]:
        source = manifest.parent / name
        if (
            source.is_symlink()
            or manifest.parent not in source.resolve().parents
            or not source.is_file()
        ):
            raise ValueError(f"model must be a regular file: {name}")
        target = inputs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(inputs / "manifest.json", config)
    source = snapshot_repository(checkout, output / "source/evas", EVAS_SOURCE)
    harness = snapshot_repository(_HARNESS_ROOT, output / "source/harness", HARNESS_SOURCE)
    if {name: row["sha256"] for name, row in harness["files"].items()} != _LOADED_HARNESS:
        raise ValueError("harness source identity changed after module load; use a fresh process")
    frozen_kernel = inputs / "evas-kernel"
    kernel_record = {"path": str(kernel), "sha256": None}
    kernel_error = None
    try:
        if not kernel.is_file():
            raise FileNotFoundError(f"kernel is not a regular file: {kernel}")
        shutil.copy2(kernel, frozen_kernel)
        kernel_record["sha256"] = file_digest(frozen_kernel)
    except OSError as error:
        kernel_error = str(error)
    identity = {
        "schema_version": 1,
        "checkout": str(checkout),
        "source": source,
        "harness": harness,
        "kernel": kernel_record,
        "environment": {"EVAS_STATIC_THREADS": "1", "other_EVAS_variables": "removed"},
        "python": {"path": str(Path(python).absolute()), "sha256": file_digest(Path(python))},
        "manifest": config,
        "timeout_s": timeout_s,
        "max_output_bytes": max_output_bytes,
        "availability": "local-only",
        "scope": "local_transient_execution",
    }
    execution_dir = output / "execution"
    execution_dir.mkdir()
    argv = [
        str(Path(python).absolute()),
        "-I",
        "-S",
        "-B",
        "-c",
        _CLI,
        str(output / "source/evas/files/evas/src"),
        str(execution_dir / "runtime.json"),
        "transient",
        str(inputs / "manifest.json"),
        "--kernel",
        str(frozen_kernel),
        "--timeout",
        str(timeout_s),
    ]
    identity["argv"] = argv
    atomic_json(output / "request.json", identity)
    journal = Journal(execution_dir, call_id=output.name)
    if kernel_error is not None:
        (execution_dir / "evas.stderr.log").write_text(kernel_error + "\n")
        (execution_dir / "evas.stdout.log").touch()
        process = {
            "execution": "infrastructure_error",
            "returncode": None,
            "cleanup_confirmed": True,
            "elapsed_s": 0,
            "stdout": "evas.stdout.log",
            "stderr": "evas.stderr.log",
        }
        journal.emit("preflight_failed", reason=kernel_error)
    else:
        process = run_process(
            argv,
            directory=execution_dir,
            stage="evas",
            deadline=time.monotonic() + timeout_s,
            cancel=threading.Event(),
            journal=journal,
            max_output_bytes=max_output_bytes,
        )
    result = {
        "schema_version": 1,
        "execution": process["execution"],
        "verdict": "not_evaluated",
        "process": process,
        "raw_result": "execution/evas.stdout.log",
        "artifacts": {},
    }
    if result["execution"] == "ok" and process["returncode"] != 0:
        result["execution"] = "backend_error"
    if result["execution"] == "backend_error":
        result["diagnostic"] = read_diagnostic(execution_dir / "evas.stderr.log")
    if kernel_error:
        result["reason"] = kernel_error
    if result["execution"] == "ok":
        try:
            _validate_result(_read_json(output / result["raw_result"]), config)
        except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
            result.update(execution="invalid_result", reason=str(error))
    for path in sorted(execution_dir.iterdir()):
        if path.is_file():
            result["artifacts"][str(path.relative_to(output))] = {
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            }
    if sum(r["bytes"] for r in result["artifacts"].values()) > max_output_bytes:
        result["execution"] = "output_limit"
    atomic_json(output / "result.json", result)
    return result
