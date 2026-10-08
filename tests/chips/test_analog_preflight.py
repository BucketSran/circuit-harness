"""Constructed host boundaries for the read-only Analog pilot preflight."""

import json
import sys
from urllib import request as urllib_request
from urllib.error import HTTPError

import pytest

from alphaapollo.common.execution.chips import analog_design_bench as adb
from alphaapollo.common.execution.chips.analog_session import create_session
from alphaapollo.common.execution.chips.bundle import build_cli


@pytest.fixture(params=("rlc-rf-bandpass-100mhz", "rlc-broadband-50-to-200-match"))
def pilot(tmp_path, monkeypatch, request):
    task_id = request.param
    source = tmp_path / "source/tasks" / task_id
    bench = source / "environment/starter/testbench"
    bench.mkdir(parents=True)
    (source / "instruction.md").write_text("Public RLC task")
    (source / "environment/starter/circuit.spi").write_text("* starter\n")
    (bench / "tb_ac.spi").write_text("* ac\n")
    (bench / "tb_stopband.spi").write_text("* stopband\n")
    if adb.TASKS[task_id].public_rlc.analyzer:
        (bench / "analyze_broadband.py").write_text("# public analyzer fixture\n")
    monkeypatch.setitem(
        adb.TASKS,
        task_id,
        adb.Task(
            "fixture",
            adb.tree_digest(source),
            "fixture@sha256:" + "a" * 64,
            adb.TASKS[task_id].public_rlc,
        ),
    )
    podman = tmp_path / "podman"
    podman.write_text("#!/bin/sh\nexit 0\n")
    podman.chmod(0o700)
    session = tmp_path / "session"
    create_session(tmp_path / "source", session, task_id=task_id, podman=str(podman))
    cli = tmp_path / "pi"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o700)
    bundle = tmp_path / "chips.pyz"
    build_cli(bundle)
    config = {
        "task_id": task_id,
        "transport": "local",
        "python": sys.executable,
        "bundle": str(bundle),
        "session": str(session),
        "pi_cli": str(cli),
        "base_url": "https://example.invalid/model",
        "model": "glm-5.3-flash",
        "thinking": "low",
        "policy_kind": "remote_model",
    }
    config_path = tmp_path / "operator.json"
    config_path.write_text(json.dumps(config))

    class Reachable:
        def open(self, *_args, **_kwargs):
            raise HTTPError(config["base_url"], 401, "unauthorized", {}, None)

    def opener(*handlers):
        assert any(
            isinstance(handler, urllib_request.ProxyHandler) and not handler.proxies
            for handler in handlers
        )
        return Reachable()

    monkeypatch.setattr("urllib.request.build_opener", opener)
    monkeypatch.delenv("CHIPS_MODEL_KEY", raising=False)
    return config_path, session, podman


def test_preflight_reports_missing_key_without_launching_model(pilot, capsys):
    from alphaapollo.workflows.chips_analog_agent import main

    config_path, session, _ = pilot
    before = list(session.rglob("*"))
    assert main(["preflight", "--config", str(config_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["state"] == "blocked"
    assert report["checks"]["credential"] == "missing"
    assert all(value == "ready" for name, value in report["checks"].items() if name != "credential")
    assert report["model_auth"] == "not_tested"
    assert report["simulator_run"] == "not_tested"
    assert report["model_settings"] == {
        "model": "glm-5.3-flash",
        "thinking": "low",
        "max_model_calls": 12,
        "max_output_tokens": 4096,
    }
    assert report["experiment_settings"] == {
        "schema_version": 1,
        "agent": "pi",
        "runtime": "direct",
        "model": "glm-5.3-flash",
        "reasoning": {"parameter": "thinking", "value": "low"},
        "budgets": {
            "max_model_calls": 12,
            "max_request_bytes": 64000,
            "max_output_tokens": 4096,
            "episode_timeout_s": 600,
        },
    }
    assert list(session.rglob("*")) == before


def test_preflight_detects_public_tamper_and_unavailable_image(pilot, tmp_path, capsys):
    from alphaapollo.workflows.chips_analog_agent import main

    config_path, session, podman = pilot
    (session / "public/instruction.md").write_text("tampered")
    podman.write_text("#!/bin/sh\nexit 1\n")
    secrets = tmp_path / "secrets"
    secrets.mkdir(mode=0o700)
    key = secrets / "glm.key"
    key.write_text("CONSTRUCTED_SECRET\n")
    key.chmod(0o600)
    assert main(["preflight", "--config", str(config_path), "--key-file", str(key)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["checks"]["credential"] == "ready"
    assert report["checks"]["public_files"] == "changed"
    assert report["checks"]["podman_image"] == "unavailable"
    assert "CONSTRUCTED_SECRET" not in json.dumps(report)


def test_preflight_refuses_incomplete_public_manifest(pilot, capsys):
    from alphaapollo.workflows.chips_analog_agent import main

    config_path, session, _ = pilot
    manifest = session / "session.json"
    details = json.loads(manifest.read_text())
    details["public_files"] = {}
    manifest.write_text(json.dumps(details))
    assert main(["preflight", "--config", str(config_path)]) == 1
    assert json.loads(capsys.readouterr().out)["checks"]["public_files"] == "changed"


def test_preflight_refuses_a_task_mismatch_without_model_or_simulator(pilot):
    from alphaapollo.workflows.chips_analog_agent import preflight_pilot

    config_path, _, _ = pilot
    config = json.loads(config_path.read_text())
    config["task_id"] = (
        "rlc-broadband-50-to-200-match"
        if config["task_id"] == "rlc-rf-bandpass-100mhz"
        else "rlc-rf-bandpass-100mhz"
    )
    result = preflight_pilot(config)
    assert result["state"] == "blocked"
    assert result["checks"]["session"] == "unavailable"
    assert result["model_auth"] == result["simulator_run"] == "not_tested"


def test_preflight_distinguishes_model_service_failure_from_auth_denial(pilot, monkeypatch, capsys):
    from alphaapollo.workflows.chips_analog_agent import main

    config_path, _, _ = pilot

    class FailingService:
        def open(self, *_args, **_kwargs):
            raise HTTPError("https://example.invalid/model", 503, "unavailable", {}, None)

    monkeypatch.setattr("urllib.request.build_opener", lambda *_args: FailingService())
    assert main(["preflight", "--config", str(config_path)]) == 1
    assert json.loads(capsys.readouterr().out)["checks"]["model_https"] == "unavailable"
