"""Adapter protocol tests. Constructed fixture; real r53 replay is opt-in."""

import json
from pathlib import Path

import pytest

from alphaapollo.common.execution.chips.vabench import replay_result


@pytest.mark.parametrize(
    ("status", "execution", "verdict"),
    [
        ("passed", "ok", "pass"),
        ("behavior_failure", "ok", "fail"),
        ("compile_failure", "ok", "fail"),
        ("runtime_failure", "ok", "fail"),
        ("infrastructure_failure", "infrastructure_error", "not_evaluated"),
    ],
)
def test_original_structured_verdict_controls_result(status, execution, verdict):
    result = replay_result({"status": status}, {"execution": "ok", "returncode": 0})
    assert (result["execution"], result["verdict"]) == (execution, verdict)
    assert result["backend"] == "evas"
    assert result["certified"] is False


def test_exit_zero_without_known_structured_verdict_is_not_success():
    with pytest.raises(ValueError, match="status"):
        replay_result({"status": "unknown"}, {"execution": "ok", "returncode": 0})


def test_pin_rejects_drift_and_symlink_candidates(tmp_path):
    from alphaapollo.common.execution.chips.vabench import candidate_files, verify_pin

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "dut.va").write_text("// fixed submission")
    assert candidate_files(candidate)["dut.va"]["bytes"] == 19
    (candidate / "escape.va").symlink_to(tmp_path / "outside.va")
    with pytest.raises(ValueError, match="symlink"):
        candidate_files(candidate)
    with pytest.raises(ValueError, match="pin"):
        verify_pin({"schema_version": 999})


def test_vabench_detached_submission_freezes_candidate(tmp_path, monkeypatch):
    from alphaapollo.common.execution.chips import jobs

    source = tmp_path / "submission"
    source.mkdir()
    (source / "dut.va").write_text("// accepted bytes")
    identity = {
        "backend": "vabench",
        "pin": {},
        "candidate": {"dut.va": {"sha256": "unused", "bytes": 17}},
    }
    from alphaapollo.common.execution.chips.vabench import candidate_files

    identity["candidate"] = candidate_files(source)
    monkeypatch.setattr(jobs, "vabench_identity", lambda *args: identity)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *args, **kwargs: None)
    monkeypatch.setattr(jobs, "inspect_job", lambda directory: {"state": "running"})
    root = tmp_path / "jobs"
    jobs.submit_vabench({}, source, root, "one")
    (source / "dut.va").write_text("// later edits")
    assert (root / "one/candidate/dut.va").read_text() == "// accepted bytes"
    assert json.loads((root / "one/request.json").read_text())["identity"] == identity


@pytest.fixture
def protocol_vendor(tmp_path):
    """Constructed upstream API double; only exercises Chips lifecycle, never EVAS."""
    import hashlib
    import sys

    source = tmp_path / "vendor"
    package = source / "benchmark-vabench-release-v4"
    release = package / "release/benchmarkv4-r53"
    task = release / "tasks/example"
    task.mkdir(parents=True)
    contract = task / "public_contract.json"
    contract.write_text("{}")
    (release / "MANIFEST.json").write_text(
        json.dumps({"release_revision": "r53", "runtime_requirements": {"evas_version": "0.8.7"}})
    )
    (release / "TASK_INDEX.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "v4-001",
                        "task_dir": "tasks/example",
                        "public_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )
    modules = {
        "operations/tri_form_derivation_prep/export_tri_form_runtime.py": """
def task_record(release, task_id):
 return {'form':'dut'}, release / 'tasks/example'
def install_public(task, public, form, mode):
 (public / 'task').mkdir(parents=True)
def install_evaluator(task, evaluator, record):
 evaluator.mkdir()
""",
        "operations/calibration_pilot/submission_contract.py": """
def submission_artifact_gate(runtime):
 return {'passed': True}
""",
        "operations/calibration_pilot/result_protocol.py": """
def snapshot_submission(runtime, gate):
 return {'tree_sha256': 'fixture-tree'}
def canonical_sha256(identity):
 return 'fixture-config'
""",
        "operations/calibration_pilot/final_replay.py": """
import hashlib, json, time

def build_final_test_profile(**kwargs):
 return {}
def run_trusted_replay(runtime, command, timeout, evas, frozen, **kwargs):
 content = (runtime / 'public/submission/dut.va').read_text()
 if content == 'sleep': time.sleep(20)
 status = 'behavior_failure' if content == 'bad' else 'passed'
 sidecar = runtime / 'evidence/score.json'
 sidecar.write_text(json.dumps({'submission_tree_sha256': 'fixture-tree',
                               'structured_result': {'status':status}}))
 return {'status':status, 'submission_tree_sha256':'fixture-tree',
         'score_sidecar_receipt': {'path':'evidence/score.json',
         'sha256':hashlib.sha256(sidecar.read_bytes()).hexdigest()}}
""",
    }
    for name, content in modules.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    runner = source / "runners/agent_harness.py"
    runner.parent.mkdir()
    runner.write_text("class EpisodeContext:\n def __init__(self, **kwargs): pass\n")
    executable = tmp_path / "env/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text(f"""#!{sys.executable}
import json, os, sys
if '-c' in sys.argv:
 print(json.dumps({{'evas_tree_sha256':'protocol-double', 'versions':{{'evas-sim':'0.8.7'}}}}))
else:
 os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
""")
    executable.chmod(0o755)
    (executable.parent / "evas").write_text("constructed EVAS protocol double")
    candidate = tmp_path / "submission"
    candidate.mkdir()
    (candidate / "dut.va").write_text("bad")
    from alphaapollo.common.execution.chips.vabench import pin_vabench

    return pin_vabench(source, executable, "v4-001"), candidate


def wait_finished(directory):
    import time

    from alphaapollo.common.execution.chips.jobs import inspect_job

    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        state = inspect_job(directory)
        if state["state"] == "finished":
            return state
        time.sleep(0.05)
    pytest.fail(f"job did not finish: {state}")


def test_detached_replay_deduplicates_and_verifies_nested_evidence(protocol_vendor, tmp_path):
    from alphaapollo.common.execution.chips.jobs import submit_vabench, verify_job

    pin, candidate = protocol_vendor
    root = tmp_path / "jobs"
    submit_vabench(pin, candidate, root, "replay")
    completed = wait_finished(root / "replay")
    assert completed["result"]["benchmark_status"] == "behavior_failure"
    assert submit_vabench(pin, candidate, root, "replay") == completed
    assert verify_job(root / "replay")["result"]["verdict"] == "fail"
    (root / "replay/run/runtime/evidence/score.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact"):
        verify_job(root / "replay")


def test_vabench_archives_negative_evidence_and_replay_timings(protocol_vendor, tmp_path):
    import time

    from alphaapollo.common.execution.chips.archive import verify_archive
    from alphaapollo.common.execution.chips.jobs import job_timings, submit_vabench

    pin, candidate = protocol_vendor
    root, archive = tmp_path / "jobs", tmp_path / "archives"
    submit_vabench(pin, candidate, root, "negative", archive_root=archive)
    wait_finished(root / "negative")
    deadline = time.monotonic() + 10
    while not (archive / "negative/receipt.json").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    receipt = verify_archive(archive / "negative")
    assert receipt["completion"]["result"]["verdict"] == "fail"
    timings = job_timings(archive / "negative")
    assert timings["backend_s"]["process:replay"] > 0
    assert timings["backend_s"]["preflight"] > 0
    assert timings["backend_s"]["postflight"] > 0
    assert "includes original simulator and scorer" in timings["note"]


def test_pinned_scorer_change_is_rejected_before_submission(protocol_vendor, tmp_path):
    from alphaapollo.common.execution.chips.jobs import submit_vabench

    pin, candidate = protocol_vendor
    scorer = Path(pin["source"]) / "runners/agent_harness.py"
    scorer.write_text("# changed scorer")
    with pytest.raises(ValueError, match="changed since pinning"):
        submit_vabench(pin, candidate, tmp_path / "jobs", "replay")
    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize("cancel", [False, True])
def test_vabench_timeout_and_cancel_are_not_candidate_failures(protocol_vendor, tmp_path, cancel):
    from alphaapollo.common.execution.chips.jobs import cancel_job, submit_vabench, verify_job

    pin, candidate = protocol_vendor
    (candidate / "dut.va").write_text("sleep")
    root = tmp_path / "jobs"
    submit_vabench(pin, candidate, root, "slow", timeout_s=20 if cancel else 1)
    if cancel:
        cancel_job(root / "slow")
    state = wait_finished(root / "slow")
    assert state["result"]["execution"] == ("cancelled" if cancel else "timeout")
    assert verify_job(root / "slow")["result"]["verdict"] == "not_evaluated"


def test_runtime_explicitly_selects_r53_without_inheriting_operator_secrets():
    from alphaapollo.common.execution.chips.vabench import runtime_env

    env = runtime_env("/private/venv/bin/python")
    assert env["VABENCH_EVAS_PROFILE"] == "r53"
    assert "OPENAI_API_KEY" not in env


def test_public_export_contains_no_evaluator_or_score_material(protocol_vendor, tmp_path):
    from alphaapollo.common.execution.chips.vabench import export_vabench

    pin, _ = protocol_vendor
    output = tmp_path / "model-public"
    export_vabench(pin, output)
    assert (output / "task/public_contract.json").is_file()
    assert not (output / "evaluator").exists()
    assert not (output / "evidence").exists()
    assert not (output / "pin.json").exists()
