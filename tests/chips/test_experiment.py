"""Constructed Agent/server replies for the shared Chips experiment entry."""

import json
import os
import subprocess

import pytest

from alphaapollo.common.execution.chips.journal import atomic_json
from alphaapollo.workflows import chips_analog_agent, chips_vabench_agent


def test_unsupported_task_agent_pair_fails_before_operator_or_output(tmp_path):
    from alphaapollo.workflows.chips_experiment import main

    plan = {
        "schema_version": 1,
        "benchmark": "analog_design_bench",
        "task_id": "rlc-rf-bandpass-100mhz",
        "run_id": "invalid-agent",
        "agent": "codex",
        "operator_config": str(tmp_path / "missing-operator.json"),
        "output": str(tmp_path / "output"),
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported Chips experiment condition"):
        main(["run", "--config", str(path)])
    assert not (tmp_path / "output").exists()


def test_validate_explains_misplaced_analog_archive_field_without_starting_a_run(tmp_path):
    from alphaapollo.workflows.chips_experiment import main

    operator = {
        "transport": "local",
        "python": "/private/python",
        "bundle": "/private/chips.pyz",
        "session": "/private/session",
        "pi_cli": "/private/pi",
        "model": "fixture",
        "base_url": "http://localhost:1/v1",
        "policy_kind": "scripted_http_fixture",
        "archive_root": "/private/archives",
    }
    atomic_json(tmp_path / "operator.json", operator)
    plan = {
        "schema_version": 1,
        "benchmark": "analog_design_bench",
        "task_id": "rlc-rf-bandpass-100mhz",
        "run_id": "bad-profile",
        "agent": "pi",
        "operator_config": str(tmp_path / "operator.json"),
        "output": str(tmp_path / "output"),
    }
    atomic_json(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="archive_root.*experiment plan"):
        main(["validate", "--config", str(tmp_path / "plan.json")])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("field", ("benchmark", "agent"))
def test_malformed_experiment_condition_is_rejected_before_start(tmp_path, field):
    from alphaapollo.workflows.chips_experiment import main

    plan = {
        "schema_version": 1,
        "benchmark": "vabench",
        "task_id": "v4-001",
        "run_id": "invalid-condition",
        "agent": "pi",
        "operator_config": str(tmp_path / "operator.json"),
        "output": str(tmp_path / "output"),
    }
    plan[field] = ["unhashable"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported Chips experiment condition"):
        main(["run", "--config", str(path)])
    assert not (tmp_path / "output").exists()


def test_vabench_unsubmitted_run_keeps_a_single_reviewable_cell(tmp_path, monkeypatch):
    from alphaapollo.workflows.chips_experiment import main

    operator = {
        "transport": "local",
        "python": "/private/python",
        "bundle": "/private/chips.pyz",
        "session": "/private/session",
        "task_id": "v4-001",
        "job_root": "/private/jobs",
        "job_id": "experiment-001",
        "archive_root": "/private/archives",
        "policy_kind": "scripted_http_fixture",
        "pi_cli": "/private/pi",
        "base_url": "http://localhost:1234/v1",
        "model": "fixture",
    }
    operator_path = tmp_path / "operator.json"
    operator_path.write_text(json.dumps(operator), encoding="utf-8")
    output = tmp_path / "experiment-001"
    plan = {
        "schema_version": 1,
        "benchmark": "vabench",
        "task_id": "v4-001",
        "run_id": "experiment-001",
        "agent": "pi",
        "operator_config": str(operator_path),
        "output": str(output),
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    calls = []

    class FixtureServer:
        def cli(self, command, *args, **kwargs):
            calls.append(command)
            assert command == "vabench-preflight"
            return {"state": "ready", "task_id": "v4-001", "session_sha256": "a" * 64}

    monkeypatch.setattr(chips_vabench_agent, "transport", lambda *_: FixtureServer())
    monkeypatch.setattr(
        chips_vabench_agent,
        "run_pi",
        lambda *_: {"termination_reason": "final"},
    )
    assert main(["validate", "--config", str(plan_path)]) == 0
    assert not output.exists() and calls == []
    assert main(["run", "--config", str(plan_path)]) == 1

    manifest = json.loads((output / "experiment_manifest.json").read_text())
    row = json.loads((output / "results.jsonl").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["benchmark"] == row["benchmark"] == "vabench"
    assert manifest["run_id"] == row["run_id"] == "experiment-001"
    assert manifest["task_id"] == row["task_id"] == "v4-001"
    assert manifest["experiment_settings"]["model"] == "fixture"
    assert manifest["experiment_settings"]["agent"] == "pi"
    assert row["status"] == "unsubmitted"
    assert row["termination_reason"] == "final"
    assert row["agent_submitted"] is False
    assert row["collection_source"] is None
    assert row["final_candidate_publicly_simulated"] is None
    assert row["score"] is None and row["benchmark_success"] is None
    assert row["evidence_refs"]["agent"] == "agent"
    assert summary["expected_attempts"] == 1
    assert summary["completed_attempts"] == 0
    assert calls == ["vabench-preflight"]
    operator_path.write_text(json.dumps({**operator, "model": "fixture-changed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="config changed"):
        main(["collect", "--config", str(plan_path)])
    assert calls == ["vabench-preflight"]


@pytest.mark.parametrize("task_id", ["rlc-rf-bandpass-100mhz", "rlc-broadband-50-to-200-match"])
def test_analog_submitted_run_waits_for_independent_final_score(tmp_path, monkeypatch, task_id):
    from alphaapollo.workflows.chips_experiment import main

    operator = {
        "task_id": task_id,
        "transport": "local",
        "python": "/private/python",
        "bundle": "/private/chips.pyz",
        "session": "/private/analog-session",
        "pi_cli": "/private/pi",
        "base_url": "http://localhost:1234/v1",
        "model": "fixture",
        "policy_kind": "scripted_http_fixture",
    }
    operator_path = tmp_path / "operator.json"
    operator_path.write_text(json.dumps(operator), encoding="utf-8")
    output = tmp_path / "analog-001"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "analog_design_bench",
                "task_id": task_id,
                "run_id": "analog-001",
                "agent": "pi",
                "operator_config": str(operator_path),
                "output": str(output),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        chips_analog_agent,
        "preflight_pilot",
        lambda *_a, **_k: {"state": "ready", "checks": {"fixture": "ready"}},
    )

    def fake_pi(_config, location):
        assert _config["task_id"] == task_id
        assert json.loads((output / "results.jsonl").read_text())["status"] == "running"
        action = location / "tools/submit-id"
        action.mkdir(parents=True)
        atomic_json(action / "request.json", {"tool": "analog_submit"})
        atomic_json(action / "response.json", {"ok": True, "result": {"state": "submitted"}})
        return {"termination_reason": "final"}

    monkeypatch.setattr(chips_analog_agent, "run_pi", fake_pi)
    assert main(["run", "--config", str(plan_path)]) == 0
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    row = json.loads((output / "results.jsonl").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["benchmark"] == row["benchmark"] == "analog_design_bench"
    assert manifest["task_id"] == row["task_id"] == task_id
    assert manifest["experiment_settings"]["model"] == "fixture"
    assert manifest["experiment_settings"]["agent"] == "pi"
    assert row["status"] == "awaiting_final"
    assert row["score"] is None and row["benchmark_success"] is None
    assert summary["completed_attempts"] == 0
    assert json.loads((output / "agent/preflight.json").read_text())["state"] == "ready"
    operator_path.write_text(json.dumps({**operator, "task_id": "rlc-rf-bandpass-100mhz"}))
    if task_id != "rlc-rf-bandpass-100mhz":
        with pytest.raises(ValueError, match="task_id.*match"):
            main(["validate", "--config", str(plan_path)])


def test_analog_preflight_error_is_not_left_as_a_pending_experiment(tmp_path, monkeypatch):
    from alphaapollo.workflows.chips_experiment import main

    operator = {
        "transport": "local",
        "python": "/private/python",
        "bundle": "/private/chips.pyz",
        "session": "/private/analog-session",
        "pi_cli": "/private/pi",
        "base_url": "http://localhost:1234/v1",
        "model": "fixture",
        "policy_kind": "scripted_http_fixture",
    }
    operator_path = tmp_path / "operator.json"
    operator_path.write_text(json.dumps(operator), encoding="utf-8")
    output = tmp_path / "analog-error"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "analog_design_bench",
                "task_id": "rlc-rf-bandpass-100mhz",
                "run_id": "analog-error",
                "agent": "pi",
                "operator_config": str(operator_path),
                "output": str(output),
            }
        ),
        encoding="utf-8",
    )

    def failed_preflight(*_a, **_k):
        raise ValueError("constructed preflight failure")

    monkeypatch.setattr(chips_analog_agent, "preflight_pilot", failed_preflight)
    monkeypatch.setattr(
        chips_analog_agent, "run_pi", lambda *_: pytest.fail("agent must not start")
    )
    with pytest.raises(ValueError, match="constructed preflight failure"):
        main(["run", "--config", str(plan_path)])
    row = json.loads((output / "results.jsonl").read_text())
    assert row["status"] == "error"
    assert row["error_type"] == "ValueError"


@pytest.mark.parametrize("verdict, success", (("pass", True), ("fail", False)))
def test_vabench_collect_reuses_the_same_run_and_updates_its_final_result(
    tmp_path, monkeypatch, verdict, success
):
    from alphaapollo.workflows.chips_experiment import main

    operator = {
        "transport": "local",
        "python": "/private/python",
        "bundle": "/private/chips.pyz",
        "session": "/private/session",
        "task_id": "v4-001",
        "job_root": "/private/jobs",
        "job_id": "collect-001",
        "archive_root": "/private/archives",
        "policy_kind": "scripted_http_fixture",
        "pi_cli": "/private/pi",
        "base_url": "http://localhost:1234/v1",
        "model": "fixture",
    }
    operator_path = tmp_path / "operator.json"
    operator_path.write_text(json.dumps(operator), encoding="utf-8")
    output = tmp_path / "collect-001"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "vabench",
                "task_id": "v4-001",
                "run_id": "collect-001",
                "agent": "pi",
                "operator_config": str(operator_path),
                "output": str(output),
            }
        ),
        encoding="utf-8",
    )

    class FixtureServer:
        def cli(self, command, *args, **kwargs):
            if command == "vabench-preflight":
                return {"state": "ready", "task_id": "v4-001"}
            assert command == "vabench-finalize"
            return {"state": "accepted"}

    def fake_pi(_config, location):
        assert json.loads((output / "results.jsonl").read_text())["status"] == "running"
        action = location / "tools/submit-id"
        action.mkdir(parents=True)
        atomic_json(action / "request.json", {"tool": "vabench_submit"})
        atomic_json(action / "response.json", {"ok": True, "result": {"status": "submitted"}})
        return {"termination_reason": "final"}

    collects = []

    def fake_collect(_remote, _config, location):
        collects.append(1)
        if len(collects) == 1:
            atomic_json(location / "report.json", {"state": "pending"})
            return False
        atomic_json(
            location / "report.json",
            {"state": "verified", "result": {"execution": "ok", "verdict": verdict}},
        )
        return True

    monkeypatch.setattr(chips_vabench_agent, "transport", lambda *_: FixtureServer())
    monkeypatch.setattr(chips_vabench_agent, "run_pi", fake_pi)
    monkeypatch.setattr(chips_vabench_agent, "collect_result", fake_collect)
    assert main(["run", "--config", str(plan_path)]) == 1
    assert json.loads((output / "results.jsonl").read_text())["status"] == "pending"
    assert main(["collect", "--config", str(plan_path)]) == 0
    row = json.loads((output / "results.jsonl").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert row["status"] == "completed" and row["benchmark_success"] is success
    assert row["verdict"] == verdict
    assert row["final_execution"] == "ok" and row["final_validity"] == "valid"
    assert summary["completed_attempts"] == 1 and summary["success_rate"] == int(success)
    assert len(collects) == 2


@pytest.mark.parametrize("collection_source", ("agent_submit", "episode_end"))
@pytest.mark.parametrize("archive_outcome", ("success", "failure", "interrupted"))
def test_analog_finalizes_the_frozen_candidate_and_keeps_zero_as_a_valid_score(
    tmp_path, monkeypatch, archive_outcome, collection_source
):
    from alphaapollo.workflows.chips_experiment import main

    operator = {
        "transport": "local",
        "python": "/private/python",
        "bundle": "/private/chips.pyz",
        "session": "/private/analog-session",
        "pi_cli": "/private/pi",
        "base_url": "http://localhost:1234/v1",
        "model": "fixture",
        "policy_kind": "scripted_http_fixture",
    }
    operator_path = tmp_path / "operator.json"
    operator_path.write_text(json.dumps(operator), encoding="utf-8")
    output = tmp_path / "analog-final-001"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "analog_design_bench",
                "task_id": "rlc-rf-bandpass-100mhz",
                "run_id": "analog-final-001",
                "agent": "pi",
                "operator_config": str(operator_path),
                "output": str(output),
                "final_output": str(tmp_path / "server-final"),
                "archive_root": str(tmp_path / "server-archive"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(chips_analog_agent, "preflight_pilot", lambda *_a, **_k: {"state": "ready"})

    def fake_pi(_config, location):
        if collection_source == "episode_end":
            return {"termination_reason": "truncated"}
        action = location / "tools/submit-id"
        action.mkdir(parents=True)
        atomic_json(action / "request.json", {"tool": "analog_submit"})
        atomic_json(
            action / "response.json",
            {
                "ok": True,
                "result": {"state": "submitted", "candidate_sha256": "a" * 64},
            },
        )
        return {"termination_reason": "final"}

    commands = []

    class FixtureServer:
        def cli(self, command, *args, **kwargs):
            commands.append(command)
            if command == "analog-close":
                return {
                    "state": "collected",
                    "candidate_sha256": "a" * 64,
                    "collection_source": "episode_end",
                    "agent_submitted": False,
                    "termination_reason": "output_token_limit",
                }
            if command == "analog-finalize":
                assert "--session" in args and "--output" in args
                return {
                    "state": "graded",
                    "score": 0.0,
                    "tests_total": 15,
                    "tests_passed": 0,
                    "frozen_candidate_sha256": "a" * 64,
                }
            if command == "analog-archive":
                if archive_outcome == "failure" and commands.count("analog-archive") == 1:
                    raise OSError("constructed archive failure")
                if archive_outcome == "interrupted" and commands.count("analog-archive") == 1:
                    raise KeyboardInterrupt("constructed interruption after grading")
                assert "--agent-evidence" in args and "--final-output" in args
                return {"sha256": "b" * 64, "candidate_sha256": "a" * 64}
            assert command == "verify-analog-episode"
            return {"state": "verified", "sha256": "b" * 64}

    monkeypatch.setattr(chips_analog_agent, "run_pi", fake_pi)
    monkeypatch.setattr(chips_analog_agent, "transport", lambda *_: FixtureServer())
    assert main(["run", "--config", str(plan_path)]) == 0
    if archive_outcome != "success":
        expected = OSError if archive_outcome == "failure" else KeyboardInterrupt
        with pytest.raises(expected):
            main(["finalize", "--config", str(plan_path)])
        interrupted = json.loads((output / "results.jsonl").read_text())
        assert interrupted["status"] == (
            "archive_failed" if archive_outcome == "failure" else "final_graded"
        )
        assert interrupted["score"] == 0.0
        assert main(["archive", "--config", str(plan_path)]) == 0
    else:
        assert main(["finalize", "--config", str(plan_path)]) == 0
    row = json.loads((output / "results.jsonl").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert row["status"] == "completed"
    assert row["score"] == 0.0
    assert row["benchmark_success"] is None
    assert row["final_execution"] == "ok"
    assert row["final_validity"] == "valid"
    assert row["frozen_candidate_sha256"] == "a" * 64
    assert row["episode_archive_sha256"] == "b" * 64
    assert row["agent_submitted"] == (collection_source == "agent_submit")
    assert row["collection_source"] == collection_source
    assert summary["completed_attempts"] == 1
    assert summary["success_rate"] is None
    assert commands.count("analog-finalize") == 1


def test_analog_experiment_reads_private_key_file_without_copying_it(tmp_path, monkeypatch):
    from alphaapollo.workflows.chips_experiment import main

    operator = {
        "transport": "local",
        "python": "/private/python",
        "bundle": "/private/chips.pyz",
        "session": "/private/analog-session",
        "pi_cli": "/private/pi",
        "base_url": "http://localhost:1234/v1",
        "model": "fixture",
        "policy_kind": "scripted_http_fixture",
    }
    operator_path = tmp_path / "operator.json"
    operator_path.write_text(json.dumps(operator), encoding="utf-8")
    secrets = tmp_path / "secrets"
    secrets.mkdir(mode=0o700)
    key_file = secrets / "glm.key"
    key_file.write_text("CONSTRUCTED_SECRET\n", encoding="utf-8")
    key_file.chmod(0o600)
    output = tmp_path / "analog-key"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "analog_design_bench",
                "task_id": "rlc-rf-bandpass-100mhz",
                "run_id": "analog-key",
                "agent": "pi",
                "operator_config": str(operator_path),
                "output": str(output),
                "key_file": str(key_file),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("CHIPS_MODEL_KEY", raising=False)

    def fake_preflight(_config, *, key_file=None):
        assert key_file == secrets / "glm.key"
        return {"state": "ready"}

    def fake_pi(_config, _evidence):
        assert os.environ["CHIPS_MODEL_KEY"] == "CONSTRUCTED_SECRET"
        _evidence.mkdir(exist_ok=True)
        return {"termination_reason": "final"}

    monkeypatch.setattr(chips_analog_agent, "preflight_pilot", fake_preflight)
    monkeypatch.setattr(chips_analog_agent, "run_pi", fake_pi)
    assert main(["run", "--config", str(plan_path)]) == 1
    assert (
        json.loads((output / "results.jsonl").read_text())["status"] == "awaiting_action_recovery"
    )
    for path in output.rglob("*"):
        if path.is_file():
            assert "CONSTRUCTED_SECRET" not in path.read_text(encoding="utf-8")


def test_analog_finalizer_timeout_is_unknown_and_never_retried_automatically(tmp_path, monkeypatch):
    from alphaapollo.workflows.chips_experiment import main

    operator = {
        "transport": "local",
        "python": "/private/python",
        "bundle": "/private/chips.pyz",
        "session": "/private/analog-session",
        "pi_cli": "/private/pi",
        "base_url": "http://localhost:1234/v1",
        "model": "fixture",
        "policy_kind": "scripted_http_fixture",
    }
    operator_path = tmp_path / "operator.json"
    operator_path.write_text(json.dumps(operator), encoding="utf-8")
    output = tmp_path / "analog-timeout"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "analog_design_bench",
                "task_id": "rlc-rf-bandpass-100mhz",
                "run_id": "analog-timeout",
                "agent": "pi",
                "operator_config": str(operator_path),
                "output": str(output),
                "final_output": str(tmp_path / "server-final"),
                "archive_root": str(tmp_path / "server-archive"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(chips_analog_agent, "preflight_pilot", lambda *_a, **_k: {"state": "ready"})

    def fake_pi(_config, location):
        action = location / "tools/submit-id"
        action.mkdir(parents=True)
        atomic_json(action / "request.json", {"tool": "analog_submit"})
        atomic_json(
            action / "response.json",
            {"ok": True, "result": {"state": "submitted", "candidate_sha256": "a" * 64}},
        )
        return {"termination_reason": "final"}

    calls = []

    class FixtureServer:
        def cli(self, command, *args, **kwargs):
            calls.append(command)
            raise subprocess.TimeoutExpired(command, 960)

    monkeypatch.setattr(chips_analog_agent, "run_pi", fake_pi)
    monkeypatch.setattr(chips_analog_agent, "transport", lambda *_: FixtureServer())
    assert main(["run", "--config", str(plan_path)]) == 0
    with pytest.raises(subprocess.TimeoutExpired):
        main(["finalize", "--config", str(plan_path)])
    row = json.loads((output / "results.jsonl").read_text())
    assert row["status"] == "unknown_execution"
    with pytest.raises(ValueError, match="only an acknowledged"):
        main(["finalize", "--config", str(plan_path)])
    assert calls == ["analog-finalize"]


def test_missing_candidate_archives_without_starting_a_final_scorer(tmp_path, monkeypatch):
    from alphaapollo.workflows.chips_experiment import main

    operator = {
        "transport": "local",
        "python": "/private/python",
        "bundle": "/private/chips.pyz",
        "session": "/private/session",
        "pi_cli": "/private/pi",
        "model": "fixture",
        "base_url": "http://localhost:1234/v1",
        "policy_kind": "scripted_http_fixture",
    }
    operator_path = tmp_path / "operator.json"
    atomic_json(operator_path, operator)
    output = tmp_path / "experiment"
    plan = {
        "schema_version": 1,
        "benchmark": "analog_design_bench",
        "task_id": "rlc-rf-bandpass-100mhz",
        "run_id": "missing",
        "agent": "pi",
        "operator_config": str(operator_path),
        "output": str(output),
        "final_output": str(tmp_path / "final"),
        "archive_root": str(tmp_path / "archives"),
    }
    path = tmp_path / "plan.json"
    atomic_json(path, plan)
    commands = []

    class Server:
        def cli(self, command, *args, **kwargs):
            commands.append(command)
            if command == "analog-close":
                return {
                    "state": "missing_candidate",
                    "agent_submitted": False,
                    "collection_source": "episode_end",
                    "candidate_sha256": None,
                    "termination_reason": "output_token_limit",
                }
            if command == "analog-archive":
                assert "--final-output" not in args
                return {"candidate_sha256": None, "sha256": "b" * 64}
            assert command == "verify-analog-episode"
            return {"state": "verified", "sha256": "b" * 64}

    monkeypatch.setattr(chips_analog_agent, "preflight_pilot", lambda *_a, **_k: {"state": "ready"})
    monkeypatch.setattr(
        chips_analog_agent, "run_pi", lambda *_: {"termination_reason": "truncated"}
    )
    monkeypatch.setattr(chips_analog_agent, "transport", lambda *_: Server())
    assert main(["run", "--config", str(path)]) == 1
    assert main(["archive", "--config", str(path)]) == 0
    row = json.loads((output / "results.jsonl").read_text())
    assert row["status"] == "completed"
    assert row["final_execution"] == row["final_validity"] == "not_applicable"
    assert row["score"] is None and row["agent_submitted"] is False
    assert commands == ["analog-close", "analog-archive", "verify-analog-episode"]
