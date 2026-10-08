"""Prepare one private VABench experiment from a server-owned deployment profile."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
from pathlib import Path

from alphaapollo.common.execution.chips.journal import atomic_json, file_digest
from alphaapollo.workflows.chips_experiment_settings import experiment_settings_snapshot
from alphaapollo.workflows.chips_vabench_agent import validate_pilot

_REQUIRED = {
    "schema_version",
    "transport",
    "pin",
    "run_root",
    "python",
    "bundle",
    "task_id",
    "job_root",
    "archive_root",
    "policy_kind",
    "pi_cli",
    "base_url",
    "model",
}
_LIMITS = {
    "thinking",
    "episode_timeout_s",
    "max_model_calls",
    "max_request_bytes",
    "max_output_tokens",
}


def _private_directory(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"private directory must exist and not be a link: {path}")
    mode = path.stat()
    if mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) != 0o700:
        raise ValueError(f"private directory must be owned by this user and mode 0700: {path}")


def _bundle_reply(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("VABench bundle command failed; inspect the new run directory")
    try:
        reply = json.loads(result.stdout)
    except ValueError:
        raise RuntimeError("VABench bundle returned no structured reply") from None
    if reply.get("state") != "ready":
        raise RuntimeError("VABench public session or preflight is not ready")
    return reply


def prepare(profile_path, run_id):
    """Create one session and operator config; never launch or bill a model."""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run ID must use 1-80 letters, numbers, underscores or hyphens")
    profile_path = Path(profile_path)
    if profile_path.is_symlink() or not profile_path.is_file():
        raise ValueError("profile must be a private regular file")
    profile_stat = profile_path.stat()
    if profile_stat.st_uid != os.getuid() or stat.S_IMODE(profile_stat.st_mode) & 0o077:
        raise ValueError("profile must be private to the current server user")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or set(profile) - (_REQUIRED | _LIMITS):
        raise ValueError("unsupported deployment profile field; never store credentials here")
    if not _REQUIRED <= set(profile) or profile["schema_version"] != 1:
        raise ValueError("incomplete or unsupported deployment profile")
    if profile["transport"] != "local":
        raise ValueError("server deployment profile requires local transport")
    for name in ("pin", "run_root", "python", "bundle", "job_root", "archive_root", "pi_cli"):
        if not isinstance(profile[name], str) or not Path(profile[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute server path")
    for name in ("run_root", "job_root", "archive_root"):
        _private_directory(profile[name])
    scratch = Path(profile["run_root"]).resolve()
    jobs = Path(profile["job_root"]).resolve()
    archive = Path(profile["archive_root"]).resolve()
    if any(
        root == archive or root in archive.parents or archive in root.parents
        for root in (scratch, jobs)
    ):
        raise ValueError("scratch run/job roots and archive_root must be separate")
    for name in ("pin", "bundle", "python", "pi_cli"):
        if not Path(profile[name]).is_file():
            raise ValueError(f"missing deployment file: {name}")
    for name in ("python", "pi_cli"):
        if not os.access(profile[name], os.X_OK):
            raise ValueError(f"deployment executable is not runnable: {name}")
    pin = json.loads(Path(profile["pin"]).read_text(encoding="utf-8"))
    if not isinstance(pin, dict) or pin.get("task_id") != profile["task_id"]:
        raise ValueError("profile task_id does not match the pinned task")
    run_root = scratch / run_id
    session = run_root / "session"
    deployment_fields = {"schema_version", "pin", "run_root"}
    operator = {key: value for key, value in profile.items() if key not in deployment_fields}
    operator.update({"session": str(session), "job_id": run_id})
    validate_pilot(operator)
    old_umask = os.umask(0o077)
    try:
        run_root.mkdir(mode=0o700)
        command = [profile["python"], "-B", profile["bundle"]]
        _bundle_reply(
            [*command, "vabench-session", "--pin", profile["pin"], "--output", str(session)]
        )
        _bundle_reply([*command, "vabench-preflight", "--session", str(session)])
        atomic_json(run_root / "operator.json", operator)
        atomic_json(
            run_root / "deployment.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "profile_sha256": file_digest(profile_path),
                "pin_sha256": file_digest(Path(profile["pin"])),
                "bundle_sha256": file_digest(Path(profile["bundle"])),
            },
        )
    finally:
        os.umask(old_umask)
    return {
        "state": "prepared",
        "run_id": run_id,
        "config": str(run_root / "operator.json"),
        "evidence": str(run_root / "evidence"),
        "model_settings": {
            "model": operator["model"],
            "thinking": operator["thinking"],
            "max_model_calls": operator.get("max_model_calls", 12),
            "max_output_tokens": operator.get("max_output_tokens", 4096),
        },
        "experiment_settings": experiment_settings_snapshot(operator, "pi"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepare", choices=("prepare",))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.profile, args.run_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
