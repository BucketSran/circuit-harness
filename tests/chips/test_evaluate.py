"""Constructed private plans for the Chips batch coordinator; no live model or simulator."""

import json
import stat
import sys
import zipfile

import pytest

from alphaapollo.common.execution.chips.journal import file_digest


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _batch(tmp_path):
    root = tmp_path / "batch"
    experiments = []
    for benchmark, task_id, run_id in (
        ("vabench", "v4-001", "va-001"),
        ("analog_design_bench", "rlc-rf-bandpass-100mhz", "analog-001"),
    ):
        operator = {
            "transport": "local",
            "python": "/private/python",
            "bundle": "/private/chips.pyz",
            "session": f"/private/{run_id}-session",
            "pi_cli": "/private/pi",
            "base_url": "http://localhost:1234/v1",
            "model": "fixture",
            "policy_kind": "scripted_http_fixture",
        }
        if benchmark == "vabench":
            operator.update(
                task_id=task_id,
                job_root="/private/jobs",
                job_id=run_id,
                archive_root="/private/archives",
            )
        operator_path = tmp_path / f"{run_id}-operator.json"
        _write(operator_path, operator)
        plan = {
            "schema_version": 1,
            "benchmark": benchmark,
            "task_id": task_id,
            "run_id": run_id,
            "agent": "pi",
            "operator_config": str(operator_path),
            "output": str(root / "cells" / run_id),
        }
        if benchmark == "analog_design_bench":
            plan.update(final_output=f"/private/{run_id}-final", archive_root="/private/archives")
        plan_path = tmp_path / f"{run_id}-experiment.json"
        _write(plan_path, plan)
        experiments.append({"condition_id": "pi-fixture", "experiment": str(plan_path)})
    batch_path = tmp_path / "batch.json"
    _write(batch_path, {"schema_version": 1, "output": str(root), "cells": experiments})
    return batch_path, root, experiments


def _single_result(experiment, *, status, score, benchmark_success):
    plan = json.loads(experiment.read_text())
    output = experiment.parent / "batch" / "cells" / plan["run_id"]
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    _write(
        output / "experiment_manifest.json",
        {
            "benchmark": plan["benchmark"],
            "task_id": plan["task_id"],
            "run_id": plan["run_id"],
            "agent": plan["agent"],
            "plan_sha256": file_digest(experiment),
            "operator_sha256": file_digest(experiment.parent / f"{plan['run_id']}-operator.json"),
        },
    )
    row = {
        "schema_version": 1,
        "benchmark": plan["benchmark"],
        "task_id": plan["task_id"],
        "run_id": plan["run_id"],
        "agent": plan["agent"],
        "status": status,
        "score": score,
        "benchmark_success": benchmark_success,
        "final_validity": "valid" if status == "completed" else "not_evaluated",
    }
    (output / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_prepare_records_every_cell_without_starting_an_agent(tmp_path):
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    manifest = json.loads((root / "batch_manifest.json").read_text())
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    summary = json.loads((root / "summary.json").read_text())
    assert [(row["benchmark"], row["status"]) for row in rows] == [
        ("vabench", "not_run"),
        ("analog_design_bench", "not_run"),
    ]
    assert manifest["cells"][0]["agent"] == "pi"
    assert manifest["cells"][0]["model_settings"]["model"] == "fixture"
    assert manifest["cells"][1]["task_id"] == "rlc-rf-bandpass-100mhz"
    assert summary["planned"] == 2 and summary["status_counts"] == {"not_run": 2}
    assert not (root / "cells").exists()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_report_counts_planned_cells_and_only_valid_binary_results(tmp_path):
    from alphaapollo.workflows.chips_evaluate import main

    config, root, cells = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    _single_result(
        tmp_path / "va-001-experiment.json", status="completed", score=None, benchmark_success=True
    )
    assert main(["report", "--config", str(config)]) == 0
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    summary = json.loads((root / "summary.json").read_text())
    assert [(row["run_id"], row["status"]) for row in rows] == [
        ("va-001", "completed"),
        ("analog-001", "not_run"),
    ]
    assert summary["planned"] == 2
    assert summary["completed"] == 1
    assert summary["status_counts"] == {"completed": 1, "not_run": 1}
    assert summary["groups"][0]["binary_success_rate"] == 1.0
    assert summary["groups"][0]["binary_valid"] == 1
    assert summary["groups"][0]["metric_coverage"] == 1.0
    assert summary["groups"][1]["binary_success_rate"] is None
    assert cells[0]["condition_id"] == rows[0]["condition_id"]


def test_report_keeps_zero_analog_reward_separate_from_vabench_pass_rate(tmp_path):
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    _single_result(
        tmp_path / "va-001-experiment.json", status="completed", score=None, benchmark_success=False
    )
    _single_result(
        tmp_path / "analog-001-experiment.json",
        status="completed",
        score=0.0,
        benchmark_success=None,
    )
    assert main(["report", "--config", str(config)]) == 0
    summary = json.loads((root / "summary.json").read_text())
    vabench, analog = summary["groups"]
    assert vabench["binary_valid"] == 1 and vabench["binary_success_rate"] == 0.0
    assert vabench["score_count"] == 0 and vabench["score_mean"] is None
    assert analog["binary_valid"] == 0 and analog["binary_success_rate"] is None
    assert analog["score_count"] == 1 and analog["score_mean"] == 0.0
    assert analog["metric_coverage"] == 1.0
    assert analog["score_median"] == 0.0
    assert analog["score_sample_std"] is None


def test_batch_report_counts_submission_and_budget_stop_separately_from_completed(tmp_path):
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    for ident, success, score in (("va-001", True, None), ("analog-001", None, 0.0)):
        _single_result(
            tmp_path / f"{ident}-experiment.json",
            status="completed",
            score=score,
            benchmark_success=success,
        )
        agent = root / "cells" / ident / "agent"
        _write(
            agent / "pi-outcome.json",
            {
                "termination_reason": "final" if success else "truncated",
                "harness_termination_reason": "final" if success else "output_token_limit",
            },
        )
        digest = "a" * 64
        _write(
            agent / "tools/sim/request.json",
            {"tool": "analog_simulate" if score is not None else "vabench_simulate"},
        )
        candidate = (
            {"candidate_sha256": digest}
            if score is not None
            else {"candidate": {"model.va": {"sha256": digest}}}
        )
        _write(
            agent / "tools/sim/response.json",
            {"ok": True, "result": {"state": "simulated", **candidate}},
        )
        if success:
            _write(agent / "tools/submit/request.json", {"tool": "vabench_submit"})
            _write(
                agent / "tools/submit/response.json",
                {"ok": True, "result": {"status": "submitted", **candidate}},
            )
        else:
            _write(
                agent / "collection.json",
                {
                    "state": "collected",
                    "agent_submitted": False,
                    "collection_source": "episode_end",
                    "candidate_sha256": digest,
                    "termination_reason": "output_token_limit",
                },
            )
    assert main(["report", "--config", str(config)]) == 0
    summary = json.loads((root / "summary.json").read_text())
    assert summary["completed"] == 2
    assert summary["agent_submission_counts"] == {"submitted": 1, "not_submitted": 1, "unknown": 0}
    assert summary["collection_source_counts"] == {"agent_submit": 1, "episode_end": 1}
    assert summary["termination_counts"] == {"final": 1, "output_token_limit": 1}
    assert summary["final_candidate_publicly_simulated_counts"] == {"yes": 2, "no": 0, "unknown": 0}
    assert summary["groups"][1]["score_mean"] == 0.0
    assert summary["groups"][1]["agent_submission_counts"]["not_submitted"] == 1


def test_report_separates_valid_result_rate_from_planned_sample_delivery(tmp_path):
    from alphaapollo.workflows.chips_evaluate import main

    config, root, cells = _batch(tmp_path)
    cells[1]["condition_id"] = "pi-fixture"
    second = json.loads((tmp_path / "analog-001-experiment.json").read_text())
    second["benchmark"] = "vabench"
    second["task_id"] = "v4-001"
    second.pop("final_output")
    second.pop("archive_root")
    _write(tmp_path / "analog-001-experiment.json", second)
    operator = json.loads((tmp_path / "analog-001-operator.json").read_text())
    operator.update(
        task_id="v4-001",
        job_root="/private/jobs",
        job_id="analog-001",
        archive_root="/private/archives",
    )
    _write(tmp_path / "analog-001-operator.json", operator)
    assert main(["prepare", "--config", str(config)]) == 0
    _single_result(
        tmp_path / "va-001-experiment.json", status="completed", score=None, benchmark_success=True
    )
    assert main(["report", "--config", str(config)]) == 0
    group = json.loads((root / "summary.json").read_text())["groups"][0]
    assert group["planned"] == 2
    assert group["binary_valid"] == 1
    assert group["binary_success_rate"] == 1.0
    assert group["metric_coverage"] == 0.5
    assert group["binary_successes_per_planned"] == 0.5


def test_batch_report_keeps_uncertain_submission_unknown(tmp_path):
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    _single_result(
        tmp_path / "va-001-experiment.json",
        status="unsubmitted",
        score=None,
        benchmark_success=None,
    )
    agent = root / "cells/va-001/agent"
    _write(agent / "pi-outcome.json", {"termination_reason": "deadline"})
    _write(agent / "report.json", {"state": "unsubmitted"})
    _write(agent / "tools/submit/request.json", {"tool": "vabench_submit"})
    _write(
        agent / "tools/submit/response.json",
        {"ok": False, "error": "unknown_execution", "retry_safe": False},
    )
    assert main(["report", "--config", str(config)]) == 0
    row = json.loads((root / "results.jsonl").read_text().splitlines()[0])
    assert row["agent_submitted"] is None
    counts = json.loads((root / "summary.json").read_text())["agent_submission_counts"]
    assert counts == {"submitted": 0, "not_submitted": 0, "unknown": 2}


def test_report_refuses_changed_operator_without_rewriting_batch_results(tmp_path):
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    before = (root / "results.jsonl").read_bytes()
    operator = tmp_path / "va-001-operator.json"
    value = json.loads(operator.read_text())
    _write(operator, {**value, "session": "/private/changed-session"})
    with pytest.raises(ValueError, match="changed after prepare"):
        main(["report", "--config", str(config)])
    assert (root / "results.jsonl").read_bytes() == before


def test_report_marks_a_damaged_started_cell_without_dropping_it(tmp_path):
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    damaged = root / "cells" / "va-001"
    damaged.mkdir(parents=True, mode=0o700)
    assert main(["report", "--config", str(config)]) == 2
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    summary = json.loads((root / "summary.json").read_text())
    assert [row["status"] for row in rows] == ["evidence_error", "not_run"]
    assert summary["planned"] == 2
    assert summary["status_counts"] == {"evidence_error": 1, "not_run": 1}


def test_prepare_refuses_one_condition_label_for_different_models(tmp_path):
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    operator = tmp_path / "analog-001-operator.json"
    value = json.loads(operator.read_text())
    _write(operator, {**value, "model": "other-fixture"})
    with pytest.raises(ValueError, match="condition_id has inconsistent agent or model settings"):
        main(["prepare", "--config", str(config)])
    assert not root.exists()


def test_v2_batch_freezes_reviewable_task_and_execution_pins(tmp_path):
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    value = json.loads(config.read_text())
    value["schema_version"] = 2
    value["conditions"] = [
        {
            "benchmark": benchmark,
            "task_id": task_id,
            "condition_id": "pi-fixture",
            "agent_revision": "pi-0.87.0",
            "harness_revision": "commit-abc",
            "task_revision": f"{benchmark}-pin",
            "prompt_revision": f"{benchmark}-prompt",
            "toolset_revision": f"{benchmark}-tools",
            "simulator_revision": f"{benchmark}-simulator",
            "scorer_revision": f"{benchmark}-scorer",
            "memory_snapshot": None,
        }
        for benchmark, task_id in (
            ("vabench", "v4-001"),
            ("analog_design_bench", "rlc-rf-bandpass-100mhz"),
        )
    ]
    _write(config, value)
    assert main(["prepare", "--config", str(config)]) == 0
    manifest = json.loads((root / "batch_manifest.json").read_text())
    assert manifest["cells"][0]["condition_pin"]["scorer_revision"] == "vabench-scorer"
    assert manifest["cells"][1]["condition_pin"]["memory_snapshot"] is None
    value["conditions"][0]["scorer_revision"] = "changed-scorer"
    _write(config, value)
    with pytest.raises(ValueError, match="changed after prepare"):
        main(["report", "--config", str(config)])


def test_v2_batch_rejects_unpinned_condition(tmp_path):
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    value = json.loads(config.read_text())
    value["schema_version"] = 2
    value["conditions"] = []
    _write(config, value)
    with pytest.raises(ValueError, match="condition pin"):
        main(["prepare", "--config", str(config)])
    assert not root.exists()


def test_run_stops_after_a_failed_cell_and_never_reruns_it(tmp_path, monkeypatch):
    from alphaapollo.workflows import chips_experiment
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    calls = []

    def fake_single_run(path):
        calls.append(path.name)
        if path.name.startswith("va-"):
            _single_result(path, status="unsubmitted", score=None, benchmark_success=None)
            return 1
        _single_result(path, status="completed", score=0.0, benchmark_success=None)
        return 0

    monkeypatch.setattr(chips_experiment, "run", fake_single_run)
    assert main(["run", "--config", str(config)]) == 1
    assert calls == ["va-001-experiment.json"]
    assert json.loads((root / "summary.json").read_text())["status_counts"] == {
        "unsubmitted": 1,
        "not_run": 1,
    }
    assert main(["run", "--config", str(config)]) == 0
    assert calls == ["va-001-experiment.json", "analog-001-experiment.json"]
    assert main(["run", "--config", str(config)]) == 0
    assert len(calls) == 2


def test_mixed_remote_batch_rejects_credential_conflict_before_first_agent(tmp_path, monkeypatch):
    from alphaapollo.workflows import chips_experiment
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    for run_id in ("va-001", "analog-001"):
        operator_path = tmp_path / f"{run_id}-operator.json"
        operator = json.loads(operator_path.read_text())
        operator.update(
            policy_kind="remote_model", thinking="low", base_url="https://example.invalid/v1"
        )
        _write(operator_path, operator)
    analog_path = tmp_path / "analog-001-experiment.json"
    analog = json.loads(analog_path.read_text())
    analog["key_file"] = str(tmp_path / "private.key")
    _write(analog_path, analog)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "constructed-secret")
    assert main(["prepare", "--config", str(config)]) == 0
    calls = []
    monkeypatch.setattr(chips_experiment, "run", lambda path: calls.append(path) or 0)

    assert main(["run", "--config", str(config)]) == 2
    assert calls == []
    assert not (root / "cells" / "va-001").exists()
    assert not (root / "intents").exists()
    receipt = json.loads((root / "preflight.json").read_text())
    assert receipt["state"] == "blocked"
    assert "constructed-secret" not in json.dumps(receipt)


def test_mixed_remote_batch_checks_later_simulator_before_first_agent(tmp_path, monkeypatch):
    from alphaapollo.workflows import chips_analog_agent, chips_experiment, chips_vabench_agent
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    for run_id in ("va-001", "analog-001"):
        operator_path = tmp_path / f"{run_id}-operator.json"
        operator = json.loads(operator_path.read_text())
        operator.update(
            policy_kind="remote_model", thinking="low", base_url="https://example.invalid/v1"
        )
        _write(operator_path, operator)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "constructed-secret")
    assert main(["prepare", "--config", str(config)]) == 0
    calls = []

    class ReadyVABench:
        def cli(self, *args, **kwargs):
            calls.append("vabench_preflight")
            return {"state": "ready", "task_id": "v4-001"}

    monkeypatch.setattr(chips_vabench_agent, "transport", lambda *_: ReadyVABench())
    monkeypatch.setattr(
        chips_analog_agent,
        "preflight_pilot",
        lambda *_a, **_k: {"state": "blocked", "checks": {"podman_image": "unavailable"}},
    )
    monkeypatch.setattr(chips_experiment, "run", lambda path: calls.append("agent") or 0)

    assert main(["run", "--config", str(config)]) == 2
    assert "agent" not in calls
    assert not (root / "intents").exists()
    receipt = json.loads((root / "preflight.json").read_text())
    assert receipt["state"] == "blocked"
    assert any(check.get("podman_image") == "unavailable" for check in receipt["checks"])


def test_remote_batch_rejects_missing_pi_launcher_before_agent(tmp_path, monkeypatch):
    from alphaapollo.workflows import chips_experiment
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    for run_id in ("va-001", "analog-001"):
        operator_path = tmp_path / f"{run_id}-operator.json"
        operator = json.loads(operator_path.read_text())
        operator.update(
            policy_kind="remote_model", thinking="low", base_url="https://example.invalid/v1"
        )
        _write(operator_path, operator)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "constructed-secret")
    assert main(["prepare", "--config", str(config)]) == 0
    calls = []
    monkeypatch.setattr(chips_experiment, "run", lambda path: calls.append(path) or 0)

    assert main(["run", "--config", str(config)]) == 2
    assert calls == []
    receipt = json.loads((root / "preflight.json").read_text())
    assert any(
        check["run_id"] == "va-001" and check.get("pi_cli") == "missing"
        for check in receipt["checks"]
    )


def test_preflight_command_checks_batch_without_starting_agent(tmp_path, monkeypatch):
    from alphaapollo.workflows import chips_experiment
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    monkeypatch.setattr(chips_experiment, "run", lambda path: pytest.fail("agent started"))

    assert main(["preflight", "--config", str(config)]) == 0
    assert json.loads((root / "preflight.json").read_text())["state"] == "ready"
    assert not (root / "intents").exists()


def test_mixed_remote_batch_records_ready_preflight_before_both_agents(tmp_path, monkeypatch):
    from alphaapollo.workflows import (
        chips_analog_agent,
        chips_evaluate,
        chips_experiment,
        chips_vabench_agent,
    )

    config, root, _ = _batch(tmp_path)
    pi_cli = tmp_path / "pi"
    pi_cli.write_text("#!/bin/sh\nexit 0\n")
    pi_cli.chmod(0o700)
    bundle = tmp_path / "chips.pyz"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("__main__.py", "pass")
    for run_id in ("va-001", "analog-001"):
        operator_path = tmp_path / f"{run_id}-operator.json"
        operator = json.loads(operator_path.read_text())
        operator.update(
            policy_kind="remote_model",
            thinking="low",
            base_url="https://example.invalid/v1",
            python=sys.executable,
            bundle=str(bundle),
            pi_cli=str(pi_cli),
        )
        _write(operator_path, operator)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "constructed-secret")
    assert chips_evaluate.main(["prepare", "--config", str(config)]) == 0
    calls = []

    class ReadyVABench:
        def cli(self, *args, **kwargs):
            calls.append("vabench_preflight")
            return {"state": "ready", "task_id": "v4-001"}

    class ReadyHTTP:
        def open(self, *args, **kwargs):
            calls.append("https_head")
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(chips_evaluate.request, "build_opener", lambda *_: ReadyHTTP())
    monkeypatch.setattr(chips_vabench_agent, "transport", lambda *_: ReadyVABench())
    monkeypatch.setattr(
        chips_analog_agent,
        "preflight_pilot",
        lambda *_a, **_k: {"state": "ready", "checks": {"podman_image": "ready"}},
    )

    def fake_run(path):
        calls.append(path.stem)
        _single_result(path, status="completed", score=0.0, benchmark_success=None)
        return 0

    monkeypatch.setattr(chips_experiment, "run", fake_run)
    assert chips_evaluate.main(["run", "--config", str(config)]) == 0
    assert calls[:2] == ["https_head", "vabench_preflight"]
    assert calls[2:] == ["va-001-experiment", "analog-001-experiment"]
    assert json.loads((root / "preflight.json").read_text())["state"] == "ready"


@pytest.mark.parametrize("command", ["run", "reconcile"])
def test_exclusive_batch_lock_uses_writable_descriptor(tmp_path, monkeypatch, command):
    from alphaapollo.workflows import chips_evaluate, chips_experiment

    config, root, _ = _batch(tmp_path)
    assert chips_evaluate.main(["prepare", "--config", str(config)]) == 0
    original_flock = chips_evaluate.fcntl.flock

    def require_writable(stream, operation):
        assert stream.writable(), "NFS requires a writable descriptor for an exclusive lock"
        return original_flock(stream, operation)

    def fake_run(path):
        _single_result(path, status="completed", score=0.0, benchmark_success=None)
        return 0

    monkeypatch.setattr(chips_evaluate.fcntl, "flock", require_writable)
    monkeypatch.setattr(chips_experiment, "run", fake_run)
    assert chips_evaluate.main([command, "--config", str(config)]) == 0
    assert (root / "batch_manifest.json").is_file()


def test_interrupted_start_is_unknown_and_cannot_be_launched_again(tmp_path, monkeypatch):
    from alphaapollo.workflows import chips_experiment
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    calls = []

    def interrupted(path):
        calls.append(path.name)
        raise RuntimeError("constructed control-process interruption")

    monkeypatch.setattr(chips_experiment, "run", interrupted)
    with pytest.raises(RuntimeError, match="constructed control-process interruption"):
        main(["run", "--config", str(config)])
    assert main(["report", "--config", str(config)]) == 2
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == ["unknown_execution", "not_run"]
    assert main(["run", "--config", str(config)]) == 2
    assert calls == ["va-001-experiment.json"]


@pytest.mark.parametrize("directory", ["intents", "cells"])
def test_run_refuses_linked_batch_work_directories(tmp_path, monkeypatch, directory):
    from alphaapollo.workflows import chips_experiment
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / directory).symlink_to(outside, target_is_directory=True)
    calls = []
    monkeypatch.setattr(chips_experiment, "run", lambda path: calls.append(path) or 0)
    if directory == "intents":
        assert main(["run", "--config", str(config)]) == 2
    else:
        with pytest.raises(ValueError, match="batch cells directory"):
            main(["run", "--config", str(config)])
    assert not calls
    assert list(outside.iterdir()) == []


def test_reconcile_collects_and_finalizes_existing_cells_once(tmp_path, monkeypatch):
    from alphaapollo.workflows import chips_experiment
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    _single_result(
        tmp_path / "va-001-experiment.json", status="pending", score=None, benchmark_success=None
    )
    _single_result(
        tmp_path / "analog-001-experiment.json",
        status="awaiting_final",
        score=None,
        benchmark_success=None,
    )
    calls = []

    def collect(path):
        calls.append(("collect", path.name))
        _single_result(path, status="completed", score=None, benchmark_success=True)
        return 0

    def finalize(path):
        calls.append(("finalize", path.name))
        _single_result(path, status="completed", score=0.0, benchmark_success=None)
        return 0

    monkeypatch.setattr(chips_experiment, "collect", collect)
    monkeypatch.setattr(chips_experiment, "finalize", finalize)
    assert main(["reconcile", "--config", str(config)]) == 0
    assert calls == [
        ("collect", "va-001-experiment.json"),
        ("finalize", "analog-001-experiment.json"),
    ]
    assert main(["reconcile", "--config", str(config)]) == 0
    assert len(calls) == 2
    assert json.loads((root / "summary.json").read_text())["completed"] == 2


def test_reconcile_does_not_repeat_interrupted_finalization(tmp_path, monkeypatch):
    from alphaapollo.workflows import chips_experiment
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    _single_result(
        tmp_path / "analog-001-experiment.json",
        status="awaiting_final",
        score=None,
        benchmark_success=None,
    )
    calls = []

    def interrupted(path):
        calls.append(path.name)
        raise RuntimeError("constructed finalization interruption")

    monkeypatch.setattr(chips_experiment, "finalize", interrupted)
    with pytest.raises(RuntimeError, match="constructed finalization interruption"):
        main(["reconcile", "--config", str(config)])
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    assert rows[1]["status"] == "unknown_execution"
    assert rows[1]["single_status"] == "awaiting_final"
    assert main(["reconcile", "--config", str(config)]) == 2
    assert calls == ["analog-001-experiment.json"]
    assert (root / "phase_actions" / "analog-001" / "finalize-0001.json").exists()


def test_reconcile_retries_completed_pending_poll_and_archives_graded_cell(tmp_path, monkeypatch):
    from alphaapollo.workflows import chips_experiment
    from alphaapollo.workflows.chips_evaluate import main

    config, root, _ = _batch(tmp_path)
    assert main(["prepare", "--config", str(config)]) == 0
    _single_result(
        tmp_path / "va-001-experiment.json",
        status="pending",
        score=None,
        benchmark_success=None,
    )
    _single_result(
        tmp_path / "analog-001-experiment.json",
        status="archive_failed",
        score=0.0,
        benchmark_success=None,
    )
    calls = []

    def collect(path):
        calls.append("collect")
        if calls.count("collect") == 2:
            _single_result(path, status="completed", score=None, benchmark_success=False)
        return 0

    def archive(path):
        calls.append("archive")
        _single_result(path, status="completed", score=0.0, benchmark_success=None)
        return 0

    monkeypatch.setattr(chips_experiment, "collect", collect)
    monkeypatch.setattr(chips_experiment, "archive", archive)
    monkeypatch.setattr(chips_experiment, "finalize", lambda path: pytest.fail("must not regrade"))
    assert main(["reconcile", "--config", str(config)]) == 0
    assert calls == ["collect", "archive"]
    assert main(["reconcile", "--config", str(config)]) == 0
    assert calls == ["collect", "archive", "collect"]
    assert json.loads((root / "summary.json").read_text())["completed"] == 2
