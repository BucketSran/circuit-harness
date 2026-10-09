"""Prepare and replay the external VA07 prototype without owning its grading."""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from circuit_harness.execution.evaluation.benchmark_replay import replay_candidate, verify_replay
from circuit_harness.execution.evaluation.benchmark_spectre import package_identity
from circuit_harness.execution.evaluation.candidate_bundle import freeze_candidate
from circuit_harness.execution.runtime.journal import atomic_json, file_digest

SOURCE_URL = "https://github.com/BucketSran/vaEVAS"
SOURCE_COMMIT = "f8b624f8887f5d55ce726373cff2a856c1487b79"
# Deterministic package exported by this commit's benchmark-owned builder.
TASK_VERSION = "dev-df643eaf9fd82a56"
CHECKER_PACKAGE_SHA256 = "f79cff373e45c5062357f58714dcefd2d77cc6115a17770b5b376cce844d0f32"
_ENGINE_PROBE = """
import hashlib,json,pathlib
root=pathlib.Path('/opt/evas')
record=json.loads((root/'source-identity.json').read_text())
actual=hashlib.sha256((root/'evas-kernel').read_bytes()).hexdigest()
files={str(p.relative_to(root/'src/evas')):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted((root/'src/evas').rglob('*.py'))}
if actual!=record['kernel_sha256'] or files!=record['evas_python']:
    raise ValueError('engine runtime differs from source identity')
if list((root/'src/evas').rglob('*.pyc')):
    raise ValueError('engine bytecode caches are forbidden')
print(json.dumps(record))
"""
SOURCE_PATHS = [
    "benchmark/checkers/build_triangle_evas_replay.py",
    "benchmark/checkers/triangle_evas_replay.py",
    "benchmark/checkers/triangle_oscillator.py",
    "benchmark/tasks/va07-triangle-repair",
    "evas/src",
    "evas/rust_core",
]


def prepare(checkout: Path, workspace: Path) -> dict:
    """Export exact published Git bytes; leave the supplied checkout untouched."""
    probe = subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-t", SOURCE_COMMIT],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if probe.returncode or probe.stdout.strip() != "commit":
        raise ValueError("checkout does not contain the pinned public commit " + SOURCE_COMMIT)
    workspace.mkdir(parents=True, mode=0o700, exist_ok=False)
    archive = workspace / "source.tar"
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "archive",
            "--format=tar",
            "-o",
            str(archive),
            SOURCE_COMMIT,
            *SOURCE_PATHS,
        ],
        check=True,
        timeout=30,
    )
    source = workspace / "source"
    source.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter="data")
    archive.unlink()
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(source / SOURCE_PATHS[0]),
            "--output",
            str(workspace / "checker"),
        ],
        check=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    task = source / "benchmark/tasks/va07-triangle-repair"
    candidate = workspace / "candidate"
    candidate.mkdir()
    shutil.copyfile(task / "solution/dut.va", candidate / "dut.va")
    shutil.copyfile(task / "instruction.md", workspace / "instruction.md")
    inventory = {
        str(path.relative_to(source)): file_digest(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    receipt = {
        "schema_version": 1,
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_files": inventory,
        "task_id": "va07-triangle-repair",
        "status": "development_prototype",
        "condition_count": 8,
        "checker": package_identity(workspace / "checker", purpose="final"),
        "candidate_origin": "benchmark reference solution; replace candidate/dut.va to submit",
    }
    atomic_json(workspace / "prepared.json", receipt)
    return {
        key: receipt[key]
        for key in ("source_url", "source_commit", "task_id", "status", "condition_count")
    }


def run(workspace: Path, image: str, output: Path) -> dict:
    """Freeze one saved submission and delegate all criteria to the external checker."""
    if output.exists():
        raise FileExistsError("output already exists: " + str(output))
    prepared = json.loads((workspace / "prepared.json").read_text())
    package = package_identity(workspace / "checker", purpose="final")
    if package != prepared["checker"]:
        raise ValueError("prepared checker changed")
    if prepared["source_commit"] != SOURCE_COMMIT:
        raise ValueError("prepared source differs from pinned public commit")
    for name, digest in prepared["source_files"].items():
        if file_digest(workspace / "source" / name) != digest:
            raise ValueError("prepared external source changed")
    # The runtime's source identity is separately declared and hashed. The checker
    # records actual Python package and kernel digests for every evaluation.
    inspected = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout.strip()
    if image != inspected:
        raise ValueError("runtime image must be an immutable local sha256 image ID")
    engine = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--entrypoint",
            "python3",
            image,
            "-B",
            "-c",
            _ENGINE_PROBE,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    engine_identity = json.loads(engine.stdout)
    if engine_identity.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("runtime engine source must match pinned public commit")
    expected_python = {
        name.removeprefix("evas/src/evas/"): digest
        for name, digest in prepared["source_files"].items()
        if name.startswith("evas/src/evas/") and name.endswith(".py")
    }
    if engine_identity["evas_python"] != expected_python:
        raise ValueError("runtime Python source differs from prepared public source")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    manifest = package["manifest"]
    frozen = freeze_candidate(
        workspace / "candidate",
        output / "candidate",
        ["dut.va"],
        task_id=manifest["task_id"],
        task_version=manifest["task_version"],
        reason="saved candidate VA07 quickstart replay",
    )
    atomic_json(
        output / "inputs.json",
        {
            "source_url": SOURCE_URL,
            "source_commit": SOURCE_COMMIT,
            "source_files": prepared["source_files"],
            "checker_sha256": package["sha256"],
            "candidate_sha256": frozen["candidate_sha256"],
            "image": image,
            "engine": engine_identity,
        },
    )
    config = {
        "schema_version": 1,
        "image": image,
        "task_package_sha256": package["sha256"],
        "solver_options": {"vabstol": 1e-8, "reltol": 0},
        "unsupported": [],
        "timeout_s": 300,
        "max_output_bytes": 16 * 1024 * 1024,
    }
    replay_candidate(output / "candidate", workspace / "checker", config, output / "replay")
    return result(output)


def result(output: Path) -> dict:
    """Read verified evidence; an unavailable score remains null."""
    receipt = verify_replay(output / "replay")
    identity = json.loads((output / "replay/identity.json").read_text())
    package = identity["package"]
    manifest = package["manifest"]
    if (
        manifest["task_id"] != "va07-triangle-repair"
        or manifest["task_version"] != TASK_VERSION
        or package["sha256"] != CHECKER_PACKAGE_SHA256
        or manifest["condition_id"] != "va07-original-eight-cases-df84f3123a91"
        or manifest["criteria_sha256"]
        != "df643eaf9fd82a569e276640d3a673fa0a632c50b7b56dcf009a83ab3ab77f10"
        or manifest["task_set"] != "extension"
    ):
        raise ValueError("result is not the pinned VA07 benchmark replay")
    inputs = json.loads((output / "inputs.json").read_text())
    if (
        inputs.get("source_url") != SOURCE_URL
        or inputs.get("source_commit") != SOURCE_COMMIT
        or inputs.get("candidate_sha256") != receipt["result"]["candidate_sha256"]
        or inputs.get("checker_sha256") != package["sha256"]
        or inputs.get("image") != identity["configuration"]["image"]
    ):
        raise ValueError("source provenance differs from sealed replay identity")
    for exported, member in (
        ("benchmark/checkers/triangle_evas_replay.py", "triangle_evas_replay.py"),
        ("benchmark/checkers/triangle_oscillator.py", "triangle_oscillator.py"),
        ("benchmark/tasks/va07-triangle-repair/tests/cases.json", "cases.json"),
    ):
        if inputs.get("source_files", {}).get(exported) != package["files"][member]["sha256"]:
            raise ValueError("source provenance differs from sealed checker files")
    report_path = output / "replay/run/work" / identity["package"]["manifest"]["report_path"]
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    runtime = report.get("runtime")
    if runtime is not None and (
        inputs.get("engine", {}).get("source_commit") != inputs["source_commit"]
        or inputs.get("engine", {}).get("kernel_sha256") != runtime.get("kernel_sha256")
        or inputs.get("engine", {}).get("evas_python") != runtime.get("evas_python")
    ):
        raise ValueError("engine provenance differs from sealed checker runtime")
    return {
        "task_id": receipt["result"]["task_id"],
        "status": "development_prototype",
        "execution": receipt["result"]["execution"],
        "score": receipt["result"]["score"],
        "classification": receipt["classification"],
        "source_url": inputs["source_url"],
        "source_commit": inputs["source_commit"],
        "candidate_bundle_sha256": receipt["result"]["candidate_sha256"],
        "candidate_file_sha256": report.get("candidate_sha256"),
        "checker_package_sha256": identity["package"]["sha256"],
        "checker_sha256": report.get("checker_sha256"),
        "kernel_sha256": report.get("runtime", {}).get("kernel_sha256"),
        "image": identity["configuration"]["image"],
        "cases": [
            {key: case.get(key) for key in ("name", "status", "passed", "reason")}
            for case in report.get("cases", [])
        ],
        "inputs": str(output / "inputs.json"),
        "receipt": str(output / "replay/receipt.json"),
        "note": "No Spectre comparison in this run. Null score is unevaluable, not zero.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--evas-checkout", type=Path, required=True)
    preparation.add_argument("--workspace", type=Path, required=True)
    execution = commands.add_parser("run")
    execution.add_argument("--workspace", type=Path, required=True)
    execution.add_argument("--image", required=True)
    execution.add_argument("--output", type=Path, required=True)
    reporting = commands.add_parser("result")
    reporting.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            value = prepare(args.evas_checkout.resolve(), args.workspace.resolve())
        elif args.command == "run":
            value = run(args.workspace.resolve(), args.image, args.output.resolve())
        else:
            value = result(args.output.resolve())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
