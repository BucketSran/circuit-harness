"""Server-side profile preparation; bundle responses are constructed fixtures."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from alphaapollo.workflows.chips_vabench_deployment import prepare


@pytest.mark.parametrize(
    "module",
    ("chips_evaluate", "chips_experiment", "chips_vabench_deployment", "chips_episode_report"),
)
def test_operator_cli_help_does_not_require_domain_or_model_dependencies(module):
    result = subprocess.run(
        [sys.executable, "-S", "-m", "alphaapollo.workflows." + module, "--help"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.fixture
def profile(tmp_path):
    for name in ("runs", "jobs", "archives"):
        (tmp_path / name).mkdir(mode=0o700)
    for name in ("python", "pi"):
        executable = tmp_path / name
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o700)
    bundle = tmp_path / "chips.pyz"
    bundle.write_bytes(b"bundle")
    pin = tmp_path / "task.pin.json"
    pin.write_text(json.dumps({"task_id": "v4-001"}))
    config = {
        "schema_version": 1,
        "transport": "local",
        "pin": str(pin),
        "run_root": str(tmp_path / "runs"),
        "python": str(tmp_path / "python"),
        "bundle": str(bundle),
        "task_id": "v4-001",
        "job_root": str(tmp_path / "jobs"),
        "archive_root": str(tmp_path / "archives"),
        "policy_kind": "remote_model",
        "pi_cli": str(tmp_path / "pi"),
        "base_url": "https://example.invalid/v1",
        "model": "test-model",
        "thinking": "low",
        "episode_timeout_s": 600,
        "max_model_calls": 12,
        "max_request_bytes": 64000,
        "max_output_tokens": 4096,
    }
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(config))
    path.chmod(0o600)
    return path


def test_prepare_generates_unique_secret_free_run_from_private_server_profile(
    profile, tmp_path, monkeypatch
):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[3] == "vabench-session":
            (tmp_path / "runs" / "run-001" / "session").mkdir()
            reply = {"state": "ready", "task_id": "v4-001"}
        else:
            reply = {"state": "ready"}
        return subprocess.CompletedProcess(argv, 0, json.dumps(reply), "")

    monkeypatch.setattr("alphaapollo.workflows.chips_vabench_deployment.subprocess.run", run)
    result = prepare(profile, "run-001")
    assert result["state"] == "prepared"
    assert result["model_settings"] == {
        "model": "test-model",
        "thinking": "low",
        "max_model_calls": 12,
        "max_output_tokens": 4096,
    }
    assert result["experiment_settings"] == {
        "schema_version": 1,
        "agent": "pi",
        "runtime": "direct",
        "model": "test-model",
        "reasoning": {"parameter": "thinking", "value": "low"},
        "budgets": {
            "max_model_calls": 12,
            "max_request_bytes": 64000,
            "max_output_tokens": 4096,
            "episode_timeout_s": 600,
        },
    }
    assert [args[3] for args in calls] == ["vabench-session", "vabench-preflight"]
    run_root = tmp_path / "runs" / "run-001"
    operator = json.loads((run_root / "operator.json").read_text())
    assert operator["transport"] == "local"
    assert operator["session"] == str(run_root / "session")
    assert operator["job_id"] == "run-001"
    assert "pin" not in operator and "run_root" not in operator
    assert not (run_root / "evidence").exists()
    assert (run_root / "deployment.json").is_file()
    with pytest.raises(FileExistsError):
        prepare(profile, "run-001")


def test_prepare_preserves_explicit_pi_thinking_level(profile, tmp_path, monkeypatch):
    config = json.loads(profile.read_text())
    config["thinking"] = "off"
    profile.write_text(json.dumps(config))

    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, json.dumps({"state": "ready"}), "")

    monkeypatch.setattr("alphaapollo.workflows.chips_vabench_deployment.subprocess.run", run)
    prepare(profile, "run-002")
    operator = json.loads((tmp_path / "runs/run-002/operator.json").read_text())
    assert operator["thinking"] == "off"


def test_prepare_rejects_missing_reasoning_setting_before_creating_run(profile, tmp_path):
    config = json.loads(profile.read_text())
    del config["thinking"]
    profile.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="thinking"):
        prepare(profile, "run-003")
    assert not (tmp_path / "runs/run-003").exists()


@pytest.mark.parametrize("change", [{"CHIPS_MODEL_KEY": "bad"}, {"transport": "ssh"}])
def test_invalid_profile_is_rejected_before_creating_run(profile, tmp_path, change):
    config = json.loads(profile.read_text())
    config.update(change)
    profile.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        prepare(profile, "run-001")
    assert not (tmp_path / "runs" / "run-001").exists()


def test_prepare_rejects_public_profile_or_unsafe_run_id(profile, tmp_path):
    profile.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        prepare(profile, "run-001")
    profile.chmod(0o600)
    with pytest.raises(ValueError, match="run ID"):
        prepare(profile, "../escape")
    assert not (tmp_path / "runs" / "run-001").exists()


def test_prepare_rejects_nonprivate_roots_and_invalid_pin(profile, tmp_path):
    jobs = tmp_path / "jobs"
    jobs.chmod(0o750)
    with pytest.raises(ValueError, match="mode 0700"):
        prepare(profile, "run-001")
    jobs.chmod(0o700)
    pin = tmp_path / "task.pin.json"
    pin.write_text("[]")
    with pytest.raises(ValueError, match="task_id"):
        prepare(profile, "run-001")
    assert not (tmp_path / "runs" / "run-001").exists()


def test_prepare_rejects_archive_nested_under_jobs(profile, tmp_path):
    config = json.loads(profile.read_text())
    nested_archive = tmp_path / "jobs" / "archives"
    nested_archive.mkdir(mode=0o700)
    config["archive_root"] = str(nested_archive)
    profile.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="separate"):
        prepare(profile, "run-001")
    assert not (tmp_path / "runs" / "run-001").exists()


def test_preflight_failure_preserves_session_without_publishing_operator(
    profile, tmp_path, monkeypatch
):
    def run(argv, **kwargs):
        if argv[3] == "vabench-session":
            (tmp_path / "runs" / "run-001" / "session").mkdir()
            reply = {"state": "ready"}
        else:
            reply = {"state": "unavailable"}
        return subprocess.CompletedProcess(argv, 0, json.dumps(reply), "")

    monkeypatch.setattr("alphaapollo.workflows.chips_vabench_deployment.subprocess.run", run)
    with pytest.raises(RuntimeError, match="not ready"):
        prepare(profile, "run-001")
    run = tmp_path / "runs" / "run-001"
    assert (run / "session").is_dir()
    assert not (run / "operator.json").exists()
