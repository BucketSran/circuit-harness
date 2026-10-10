"""Benchmark-owned Spectre packages executed inside existing durable jobs.

Only trusted operator packages may supply executable checker code. Resource limits
are not a security sandbox. Final reports stay in private job evidence; public jobs
use a separate package and return only explicitly declared measurement fields.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import sys
import threading
import time
from pathlib import Path

from circuit_harness.execution.backends.vabench import candidate_files
from circuit_harness.execution.runtime.journal import Journal, atomic_json, digest, file_digest
from circuit_harness.execution.runtime.process import run_process

_PACKAGE_KEYS = {
    "schema_version",
    "task_id",
    "task_version",
    "criteria_sha256",
    "condition_id",
    "task_set",
    "purpose",
    "entrypoint",
    "candidate_file",
    "report_path",
    "files",
    "feedback_fields",
}
_PROFILE_KEYS = {
    "schema_version",
    "shell",
    "setup_scripts",
    "spectre",
    "preflight_script",
    "run_root",
    "archive_root",
    "timeout_s",
    "max_output_bytes",
}
_PUBLIC_PROFILE_KEYS = {
    "schema_version",
    "backend",
    "image",
    "spectre",
    "preflight_script",
    "run_root",
    "archive_root",
    "timeout_s",
    "max_output_bytes",
}

_RESERVED_FEEDBACK = {"reward", "score", "verdict", "cases", "report", "artifacts", "logs"}
_MAX_PUBLIC_REPORT_BYTES = 16 * 1024 * 1024


def _relative(name):
    if (
        not isinstance(name, str)
        or not name
        or name != Path(name).as_posix()
        or name == "."
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError("invalid package relative path")
    path = Path(name)
    if path.is_absolute() or any(part in {"..", ".", ""} for part in name.split("/")):
        raise ValueError("invalid package relative path")
    return name


def package_identity(directory: Path, *, purpose: str) -> dict:
    """Verify an explicit benchmark package inventory, including its entrypoint."""
    directory = Path(directory)
    inventory = candidate_files(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict) or set(manifest) != _PACKAGE_KEYS:
        raise ValueError("benchmark package manifest fields differ")
    if manifest["schema_version"] != 1 or purpose not in {"final", "public"}:
        raise ValueError("unsupported package version or purpose")
    if manifest["purpose"] != purpose:
        raise ValueError("benchmark package purpose differs from execution purpose")
    for key in ("task_id", "task_version"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ValueError(f"missing benchmark {key}")
    if not isinstance(manifest["criteria_sha256"], str) or not re.fullmatch(
        r"[a-f0-9]{64}", manifest["criteria_sha256"]
    ):
        raise ValueError("criteria_sha256 must be a benchmark-declared SHA256")
    if manifest["task_set"] not in {"public", "extension"}:
        raise ValueError("task_set must be public or extension")
    if not isinstance(manifest["condition_id"], str) or not manifest["condition_id"]:
        raise ValueError("condition_id must be a benchmark-declared test condition")
    for key in ("entrypoint", "candidate_file", "report_path"):
        _relative(manifest[key])
    expected = manifest["files"]
    if not isinstance(expected, dict) or not expected or "manifest.json" in expected:
        raise ValueError("invalid task package file inventory")
    for name in expected:
        _relative(name)
    actual = {name: row for name, row in inventory.items() if name != "manifest.json"}
    if actual != expected or manifest["entrypoint"] not in expected:
        raise ValueError("task package files missing, modified or undeclared")
    if manifest["candidate_file"] in expected or any(
        manifest["candidate_file"].startswith(name + "/") for name in expected
    ):
        raise ValueError("task package must not contain the candidate")
    if any(name == "candidate" or name.startswith("candidate/") for name in expected):
        raise ValueError("task package overlaps frozen candidate workspace")
    report = manifest["report_path"]
    if (
        report == "candidate"
        or report.startswith("candidate/")
        or any(
            name == report or report.startswith(name + "/") or name.startswith(report + "/")
            for name in expected
        )
    ):
        raise ValueError("task report overlaps input files")
    fields = manifest["feedback_fields"]
    if (
        not isinstance(fields, list)
        or any(not isinstance(f, str) or not f for f in fields)
        or len(set(fields)) != len(fields)
        or set(fields) & _RESERVED_FEEDBACK
    ):
        raise ValueError("invalid public feedback fields")
    if purpose == "final" and fields:
        raise ValueError("final package cannot declare public feedback")
    return {
        "manifest": manifest,
        "files": inventory,
        "sha256": digest(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()),
    }


def _contains_private_field(value):
    if isinstance(value, dict):
        return bool(set(value) & _RESERVED_FEEDBACK) or any(
            _contains_private_field(v) for v in value.values()
        )
    if isinstance(value, list):
        return any(_contains_private_field(v) for v in value)
    return False


def project_report(report, *, purpose: str, feedback_fields: list[str]) -> dict:
    """Preserve task scores only when a structured report establishes grading."""
    result = {"execution": "invalid_result", "verdict": "not_evaluated", "score": None}
    if not isinstance(report, dict) or not isinstance(report.get("status"), str):
        return result
    status = report["status"]
    result["benchmark_status"] = status
    if status in {"infrastructure_error", "checker_error"}:
        return dict(result, execution="infrastructure_error")
    if purpose == "public":
        if status != "completed" or any(field not in report for field in feedback_fields):
            return result
        feedback = {f: report[f] for f in feedback_fields}
        if _contains_private_field(feedback):
            return result
        return dict(result, execution="ok", feedback=feedback)
    reward = report.get("reward")
    if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
        return result
    if status == "submission_contract_violation" and reward == 0:
        return dict(result, execution="ok", verdict="fail", score=reward)
    cases = report.get("cases")
    if status != "completed" or not isinstance(cases, list) or not cases:
        return result
    for case in cases:
        if not isinstance(case, dict) or type(case.get("passed")) is not bool:
            return result
        # The existing waveform checker identifies numerical grading explicitly.
        # Triangle's report identifies a completed simulation with raw waveform.
        graded = case.get("status") == "graded" or (
            case.get("returncode") == 0
            and case.get("timeout") is False
            and isinstance(case.get("waveform_sha256"), str)
        )
        if not graded:
            return dict(result, execution="unclassified_failure")
    return dict(result, execution="ok", verdict="pass" if reward == 1 else "fail", score=reward)


def _private_profile(path: Path) -> dict:
    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_uid != os.getuid()
        or path.stat().st_mode & 0o077
        or path.stat().st_size > 65536
    ):
        raise ValueError("benchmark profile must be an owned private regular file")
    profile = json.loads(path.read_text())
    if isinstance(profile, dict) and profile.get("backend") == "docker":
        if set(profile) != _PUBLIC_PROFILE_KEYS or profile["schema_version"] != 1:
            raise ValueError("invalid isolated Docker public profile")
        if not isinstance(profile["image"], str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", profile["image"]
        ):
            raise ValueError("declare immutable local image ID")
        for key in ("spectre", "preflight_script"):
            value = profile[key]
            if (
                not isinstance(value, str)
                or not Path(value).is_absolute()
                or ".." in Path(value).parts
            ):
                raise ValueError("declare absolute in-image runtime paths")
        for key in ("run_root", "archive_root"):
            root = Path(profile[key])
            if (
                not root.is_absolute()
                or root.is_symlink()
                or not root.is_dir()
                or root.stat().st_uid != os.getuid()
                or root.stat().st_mode & 0o077
            ):
                raise ValueError("job/archive roots must be owned private directories")
        a, b = (Path(profile[k]).resolve() for k in ("run_root", "archive_root"))
        if a == b or a in b.parents or b in a.parents:
            raise ValueError("job/archive roots must be disjoint")
        if (
            type(profile["timeout_s"]) not in (int, float)
            or not math.isfinite(profile["timeout_s"])
            or not 0 < profile["timeout_s"] <= 300
        ):
            raise ValueError("invalid Docker timeout")
        if (
            type(profile["max_output_bytes"]) is not int
            or not 1 <= profile["max_output_bytes"] <= _MAX_PUBLIC_REPORT_BYTES
        ):
            raise ValueError("invalid Docker output limit")
        return profile
    namespace = isinstance(profile, dict) and profile.get("backend") == "spectre_namespace"
    required = _PROFILE_KEYS | {"backend", "isolation_config"} if namespace else _PROFILE_KEYS
    if not isinstance(profile, dict) or set(profile) != required or profile["schema_version"] != 1:
        raise ValueError("unsupported benchmark Spectre profile fields")
    if profile["shell"] not in {"/bin/sh", "/bin/csh"}:
        raise ValueError("shell must be /bin/sh or /bin/csh")
    scripts = profile["setup_scripts"]
    if not isinstance(scripts, list) or len(scripts) > 4:
        raise ValueError("setup_scripts must contain at most four files")
    for name in [profile["shell"], profile["spectre"], profile["preflight_script"], *scripts]:
        if not isinstance(name, str) or not Path(name).is_absolute() or not Path(name).is_file():
            raise ValueError("profile executable/setup paths must be existing absolute files")
    if not os.access(profile["spectre"], os.X_OK):
        raise ValueError("configured Spectre executable is not executable")
    for key in ("run_root", "archive_root"):
        root = Path(profile[key])
        if (
            not root.is_absolute()
            or root.is_symlink()
            or not root.is_dir()
            or root.stat().st_uid != os.getuid()
            or root.stat().st_mode & 0o077
        ):
            raise ValueError("job/archive roots must be owned private directories")
    run, archive = (Path(profile[k]).resolve() for k in ("run_root", "archive_root"))
    if run == archive or run in archive.parents or archive in run.parents:
        raise ValueError("job/archive roots must be disjoint")
    value = profile["timeout_s"]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1800:
        raise ValueError("timeout_s must be finite within (0, 1800]")
    if (
        type(profile["max_output_bytes"]) is not int
        or not 1 <= profile["max_output_bytes"] <= 512 * 1024 * 1024
    ):
        raise ValueError("invalid max_output_bytes")
    if namespace:
        from circuit_harness.execution.backends.spectre_isolation import _configuration

        isolation, _ = _configuration(profile["isolation_config"])
        if isolation["spectre"] != profile["spectre"]:
            raise ValueError("namespace profile must declare the same Spectre binary")
    return profile


def _profile_identity(path: Path) -> dict:
    profile = _private_profile(path)
    isolation_tools = []
    if profile.get("backend") == "spectre_namespace":
        from circuit_harness.execution.backends.spectre_isolation import _configuration

        isolation, _ = _configuration(profile["isolation_config"])
        isolation_tools = [profile["isolation_config"], isolation["bubblewrap"], sys.executable]
    return {
        "path": str(Path(path).absolute()),
        "sha256": file_digest(path),
        "configuration": profile,
        "tools": {
            name: file_digest(Path(name))
            for name in (
                []
                if profile.get("backend") == "docker"
                else [
                    profile["shell"],
                    profile["spectre"],
                    profile["preflight_script"],
                    *profile["setup_scripts"],
                    *isolation_tools,
                ]
            )
        },
    }


def benchmark_identity(
    candidate: Path,
    task_package: Path,
    profile_path: Path,
    *,
    purpose="final",
    isolated_public=False,
) -> dict:
    from circuit_harness.execution.evaluation.candidate_bundle import verify_candidate

    frozen = verify_candidate(candidate)
    package = package_identity(task_package, purpose=purpose)
    manifest = package["manifest"]
    if any(frozen[k] != manifest[k] for k in ("task_id", "task_version")):
        raise ValueError("frozen candidate task identity differs from benchmark package")
    if manifest["candidate_file"] not in frozen["files"]:
        raise ValueError("declared checker candidate file is missing")
    profile = _profile_identity(profile_path)
    if isolated_public and (
        purpose != "public"
        or profile["configuration"].get("backend") not in {"docker", "spectre_namespace"}
    ):
        raise ValueError(
            "Agent public execution requires an isolated Docker or Spectre namespace profile"
        )
    if purpose == "final" and profile["configuration"].get("backend") == "docker":
        raise ValueError("Docker public profile cannot run final evaluation")
    return {
        "backend": "benchmark_spectre",
        "purpose": purpose,
        "candidate_manifest": {k: v for k, v in frozen.items() if k != "reason"},
        "candidate": candidate_files(candidate),
        "package": package,
        "profile": profile,
    }


def stage_inputs(candidate: Path, task_package: Path, directory: Path, identity: dict) -> None:
    """Copy and verify both immutable inventories before the worker is launched."""
    shutil.copytree(candidate, directory / "candidate")
    shutil.copytree(task_package, directory / "task-package")
    _verify_inputs(directory, identity)


def _verify_inputs(directory: Path, identity: dict) -> None:
    from circuit_harness.execution.evaluation.candidate_bundle import verify_candidate

    frozen = verify_candidate(directory / "candidate")
    if (
        {k: v for k, v in frozen.items() if k != "reason"} != identity["candidate_manifest"]
        or candidate_files(directory / "candidate") != identity["candidate"]
        or package_identity(directory / "task-package", purpose=identity["purpose"])
        != identity["package"]
    ):
        raise ValueError("frozen benchmark inputs changed")


def _read_report(directory: Path, identity: dict):
    manifest = identity["package"]["manifest"]
    path = directory / "work" / manifest["report_path"]
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(directory.resolve())
    ):
        raise ValueError("checker report missing or unsafe")
    max_report_bytes = (
        _MAX_PUBLIC_REPORT_BYTES if identity["purpose"] == "public" else 4 * 1024 * 1024
    )
    if path.stat().st_size > max_report_bytes:
        raise ValueError("checker report exceeds limit")
    report = json.loads(
        path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
    )
    if identity["backend"] == "benchmark_opensource" and (
        not isinstance(report, dict)
        or report.get("solver_options", {}) != identity["configuration"]["solver_options"]
    ):
        raise ValueError("checker report does not establish requested solver options")
    candidate = manifest["candidate_file"]
    expected = identity["candidate_manifest"]["files"][candidate]["sha256"]
    if not isinstance(report, dict) or report.get("candidate_sha256") != expected:
        raise ValueError("checker report candidate identity differs")
    for key in ("checker_sha256", "cases_sha256"):
        if key in report and report[key] not in {
            v["sha256"] for v in identity["package"]["files"].values()
        }:
            raise ValueError("checker report task identity differs")
    return report


def assess_preflight(directory: Path, process: dict | None) -> dict:
    """Consume deployment-owned license/dependency checks without guessing logs."""
    if process is None:
        return {"execution": "invalid_preflight", "reason": "preflight process receipt missing"}
    if process["execution"] != "ok":
        return {"execution": process["execution"]}
    path = directory / "preflight/report.json"
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(directory.resolve())
            or path.stat().st_size > 65536
        ):
            raise ValueError("preflight report missing, unsafe or oversized")
        report = json.loads(path.read_text())
        if (
            not isinstance(report, dict)
            or set(report) != {"schema_version", "checks"}
            or type(report["schema_version"]) is not int
            or report["schema_version"] != 1
        ):
            raise ValueError("invalid preflight report schema")
        checks = report["checks"]
        if not isinstance(checks, list) or not 2 <= len(checks) <= 64:
            raise ValueError("preflight must declare bounded license/dependency checks")
        seen = set()
        for check in checks:
            if (
                not isinstance(check, dict)
                or set(check) != {"kind", "name", "status", "detail"}
                or check["kind"] not in {"license", "dependency", "simulator"}
                or check["status"] not in {"passed", "failed", "unknown"}
                or not isinstance(check["name"], str)
                or not check["name"]
                or not isinstance(check["detail"], str)
            ):
                raise ValueError("invalid structured preflight check")
            key = check["kind"], check["name"]
            if key in seen:
                raise ValueError("duplicate preflight check")
            seen.add(key)
        if not {"license", "dependency"} <= {c["kind"] for c in checks}:
            raise ValueError("license and dependency checks are required")
        for kind in ("license", "dependency", "simulator"):
            if any(c["kind"] == kind and c["status"] == "failed" for c in checks):
                return {"execution": kind + "_unavailable", "checks": checks}
        if process["returncode"] != 0:
            return {
                "execution": "invalid_preflight",
                "reason": "preflight did not exit successfully",
                "checks": checks,
            }
        return {
            "execution": "preflight_unknown"
            if any(c["status"] == "unknown" for c in checks)
            else "ok",
            "checks": checks,
        }
    except (ValueError, OSError, TypeError) as error:
        return {"execution": "invalid_preflight", "reason": str(error)}


def _shell_argv(config: dict, directory: Path, arguments: list[str]) -> list[str]:
    source = "source" if config["shell"] == "/bin/csh" else "."
    command = "\n".join(f"{source} {shlex.quote(name)}" for name in config["setup_scripts"])
    command += "\nexec /usr/bin/env " + shlex.join(arguments)
    return [
        "/usr/bin/env",
        "-i",
        "PATH=/usr/bin:/bin",
        f"HOME={directory}",
        config["shell"],
        "-f",
        "-c",
        command,
    ]


def _assess(directory: Path, identity: dict, process: dict, preflight_process=None) -> dict:
    config = identity.get("profile", {}).get("configuration", {})
    if config.get("backend") == "docker":
        if (directory / "backend.json").exists():
            _verify_container_execution(directory, config)
        elif process["execution"] not in {"timeout", "cancelled"}:
            raise ValueError("public Docker execution receipt missing")
    result = {"execution": process["execution"], "verdict": "not_evaluated", "score": None}
    preflight = {}
    if "preflight_script" in identity.get("profile", {}).get("configuration", {}):
        preflight = assess_preflight(directory, preflight_process)
        if preflight["execution"] != "ok":
            result["execution"] = preflight["execution"]
    if result["execution"] == "ok":
        try:
            report = _read_report(directory, identity)
            result = project_report(
                report,
                purpose=identity["purpose"],
                feedback_fields=identity["package"]["manifest"]["feedback_fields"],
            )
        except (ValueError, OSError, TypeError) as error:
            result = {
                "execution": "invalid_result",
                "verdict": "not_evaluated",
                "score": None,
                "reason": str(error),
            }
    return {
        **result,
        "backend": "opensource" if identity["backend"] == "benchmark_opensource" else "spectre",
        "purpose": identity["purpose"],
        "task_id": identity["candidate_manifest"]["task_id"],
        "task_version": identity["candidate_manifest"]["task_version"],
        "candidate_sha256": identity["candidate_manifest"]["candidate_sha256"],
        "task_package_sha256": identity["package"]["sha256"],
        "criteria_sha256": identity["package"]["manifest"]["criteria_sha256"],
        "condition_id": identity["package"]["manifest"]["condition_id"],
        "task_set": identity["package"]["manifest"]["task_set"],
        "certified": False,
        "process": process,
        **({"preflight": preflight, "preflight_process": preflight_process} if preflight else {}),
    }


def run_benchmark(identity: dict, directory: Path, cancel=None) -> dict:
    """Run the original task entrypoint under the existing bounded process owner."""
    directory = Path(directory)
    directory.mkdir(mode=0o700)
    deadline = time.monotonic() + identity["profile"]["configuration"]["timeout_s"]
    _verify_inputs(directory.parent, identity)
    profile = identity["profile"]
    if _profile_identity(Path(profile["path"])) != profile:
        raise ValueError("Spectre profile/setup/tool changed after submission")
    config = profile["configuration"]
    manifest = identity["package"]["manifest"]
    if config.get("backend") == "docker":
        return _run_public_container(identity, directory, deadline, cancel)
    atomic_json(directory / "identity.json", identity)
    spectre = config["spectre"]
    if config.get("backend") == "spectre_namespace":
        # This trusted launcher, rather than arbitrary operator wrapper text,
        # selects the validated namespace for every candidate Spectre child.
        launcher = directory / "spectre-namespace-launcher"
        package_root = str(Path(__file__).resolve().parents[3])
        code = (
            "import sys;sys.path.insert(0," + repr(package_root) + ");"
            "from circuit_harness.execution.backends.spectre_isolation "
            "import exec_isolated_spectre;"
            "exec_isolated_spectre(" + repr(config["isolation_config"]) + ",sys.argv[1:])"
        )
        launcher.write_text(
            "#!/bin/sh\nexec " + shlex.join([sys.executable, "-c", code]) + ' "$@"\n'
        )
        launcher.chmod(0o700)
        spectre = str(launcher)
    work = directory / "work"
    shutil.copytree(directory.parent / "task-package", work)
    shutil.copytree(directory.parent / "candidate/files", work / "candidate")
    candidate = work / "candidate" / manifest["candidate_file"]
    output = work / Path(manifest["report_path"]).parent
    journal = Journal(directory, call_id=directory.parent.name)
    cancel = cancel or threading.Event()
    (directory / "preflight").mkdir(mode=0o700)
    preflight_process = run_process(
        _shell_argv(
            config,
            directory,
            [
                f"SPECTRE={spectre}",
                f"TASK_PACKAGE={directory.parent / 'task-package'}",
                f"PREFLIGHT_OUTPUT={directory / 'preflight/report.json'}",
                "/bin/sh",
                config["preflight_script"],
            ],
        ),
        directory=directory,
        stage="preflight",
        deadline=deadline,
        cancel=cancel,
        journal=journal,
        max_output_bytes=config["max_output_bytes"],
    )
    if assess_preflight(directory, preflight_process)["execution"] != "ok":
        return seal_result(
            directory, identity, preflight_process, preflight_process=preflight_process
        )
    # The original test.sh contract supplies CANDIDATE and VERIFY_OUTPUT.
    process = run_process(
        _shell_argv(
            config,
            directory,
            [
                f"SPECTRE={spectre}",
                f"CANDIDATE={candidate}",
                f"VERIFY_OUTPUT={output}",
                "/bin/sh",
                str(work / manifest["entrypoint"]),
            ],
        ),
        directory=directory,
        stage="checker",
        deadline=deadline,
        cancel=cancel,
        journal=journal,
        max_output_bytes=config["max_output_bytes"],
    )
    if candidate_files(work / "candidate") != identity["candidate_manifest"]["files"]:
        raise ValueError("checker changed frozen candidate bytes")
    return seal_result(directory, identity, process, preflight_process=preflight_process)


def seal_result(directory: Path, identity: dict, process: dict, *, preflight_process=None) -> dict:
    """Seal shared checker evidence for independently verifiable backend receipts."""
    _verify_inputs(directory.parent, identity)
    if (
        "profile" in identity
        and _profile_identity(Path(identity["profile"]["path"])) != identity["profile"]
    ):
        raise ValueError("Spectre profile/setup/preflight/tool changed during execution")
    result = _assess(directory, identity, process, preflight_process)
    artifacts = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("checker output contains symlink")
        if path.is_file():
            artifacts[path.relative_to(directory).as_posix()] = {
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            }
    result["artifacts"] = artifacts
    atomic_json(directory / "result.json", result)
    return result


def verify_benchmark(directory: Path) -> dict:
    """Verify frozen inputs, receipts and structured result without invoking tools."""
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    identity = json.loads((directory / "identity.json").read_text())
    _verify_inputs(directory.parent, identity)
    for name, receipt in result["artifacts"].items():
        _relative(name)
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or not path.is_file()
            or path.stat().st_size != receipt["bytes"]
            or file_digest(path) != receipt["sha256"]
        ):
            raise ValueError("benchmark artifact missing or modified")
    if _assess(directory, identity, result["process"], result.get("preflight_process")) != {
        k: v for k, v in result.items() if k != "artifacts"
    }:
        raise ValueError("benchmark receipt differs from checker report")
    return result


def public_feedback(result: dict) -> dict:
    """Return only package-declared public measurements, never final evidence."""
    if result.get("purpose") != "public":
        raise ValueError("final evaluation cannot be exposed as public feedback")
    return {
        "execution": result["execution"],
        "candidate_sha256": result["candidate_sha256"],
        "feedback": result.get("feedback", {}),
    }


def _run_public_container(identity, directory, deadline, cancel):
    """No profile, probe or candidate code executes on the host."""
    from circuit_harness.execution.sessions.current_evas_public import run_isolated_docker

    config = identity["profile"]["configuration"]
    manifest = identity["package"]["manifest"]
    atomic_json(directory / "identity.json", identity)
    work = directory / "work"
    work.mkdir(mode=0o700)
    preflight = directory / "preflight"
    preflight.mkdir(mode=0o700)
    readonly = {
        directory.parent / "candidate/files": "/candidate",
        directory.parent / "task-package": "/task",
    }
    writable = {work: "/output", preflight: "/preflight"}

    def execute(command, stage):
        # Monitor the complete writable result tree, not only subprocess logs.
        logs = directory
        if stage == "checker-process":
            saved = directory / "probe-process"
            saved.mkdir(mode=0o700)
            for name in (
                "evas.stdout.log",
                "evas.stderr.log",
                "evas.process.json",
                "backend.json",
                "cleanup.json",
                "events.jsonl",
            ):
                if (directory / name).exists():
                    shutil.copy2(directory / name, saved / name)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "execution": "timeout",
                "returncode": None,
                "elapsed_s": 0,
                "cleanup_confirmed": True,
            }
        return run_isolated_docker(
            image=config["image"],
            readonly=readonly,
            writable=writable,
            command=[
                "/usr/bin/env",
                "-i",
                "PATH=/usr/bin:/bin",
                f"SPECTRE={config['spectre']}",
                *command,
            ],
            directory=logs,
            action_id=directory.parent.name,
            timeout_s=remaining,
            max_output_bytes=config["max_output_bytes"],
            workdir="/task",
            cancel=cancel,
        )

    probe = execute(
        [
            "TASK_PACKAGE=/task",
            "PREFLIGHT_OUTPUT=/preflight/report.json",
            "/bin/sh",
            config["preflight_script"],
        ],
        "probe-process",
    )
    process = probe
    if assess_preflight(directory, probe)["execution"] == "ok":
        process = execute(
            [
                f"CANDIDATE=/candidate/{manifest['candidate_file']}",
                f"VERIFY_OUTPUT=/output/{Path(manifest['report_path']).parent}",
                "/bin/sh",
                "/task/" + manifest["entrypoint"],
            ],
            "checker-process",
        )
    return seal_result(directory, identity, process, preflight_process=probe)


def _verify_container_execution(directory, config):
    """Archive validation checks the executor's concrete mount/limit receipt."""
    paths = [directory / "backend.json"]
    if (directory / "probe-process/backend.json").exists():
        paths.append(directory / "probe-process/backend.json")
    for path in paths:
        record = json.loads(path.read_text())
        argv = record["argv"]
        required = {
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=32",
            "--memory=512m",
            "--memory-swap=512m",
            "--cpus=1",
            "--entrypoint=python3",
        }
        mounts = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--mount"]
        destinations = {}
        for mount in mounts:
            parts = mount.split(",")
            target = next(part[4:] for part in parts if part.startswith("dst="))
            destinations[target] = "readonly" in parts
        if (
            record["backend"] != "docker"
            or record["image"] != config["image"]
            or config["image"] not in argv
            or not required <= set(argv)
            or len(mounts) != 4
            or destinations
            != {"/candidate": True, "/task": True, "/output": False, "/preflight": False}
        ):
            raise ValueError("public Docker execution receipt differs from isolation contract")
