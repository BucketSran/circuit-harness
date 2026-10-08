"""Synthetic public tasks test session behavior without benchmark grading."""

import subprocess
from pathlib import Path

import pytest

from circuit_harness.execution import current_evas_session as session


def make_session(
    tmp_path,
    *,
    main=None,
    image="sha256:" + "a" * 64,
    feedback_fields=None,
    experiments=None,
    manifest=None,
    **backend_options,
):
    source = tmp_path / "engine"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    module = source / "evas/src/evas"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("")
    if main is not None:
        (module / "__main__.py").write_text(main)
    kernel = tmp_path / "kernel"
    kernel.write_bytes(b"\x7fELF synthetic; not a real kernel")
    material = tmp_path / "materials"
    material.mkdir()
    (material / "instruction.md").write_text("synthetic protocol fixture")
    task = {
        "task_id": "synthetic",
        "task_version": "fixture-v1",
        "public_files": ["instruction.md"],
        "candidate_files": ["dut.va"],
        "feedback_fields": feedback_fields if feedback_fields is not None else ["diagnostics"],
        "manifest": {
            "models": ["dut.va"],
            "instances": [],
            "transient": {"sources": {}, "output_times": [0.0], "stop": 1.0, "max_step": 0.1},
            "tolerances": {"vabstol": 1e-8, "reltol": 0},
        },
    }
    if manifest is not None:
        task["manifest"] = manifest
    if experiments is not None:
        task["experiments"] = experiments
    directory = tmp_path / "session"
    session.create_session(
        task=task,
        materials=material,
        checkout=source,
        kernel=kernel,
        directory=directory,
        image=image,
        max_simulations=1,
        **backend_options,
    )
    return directory


def call(directory, action_id, tool, **arguments):
    return session.session_action(
        directory, {"action_id": action_id, "tool": tool, "arguments": arguments}
    )


def test_last_complete_candidate_freezes_even_when_empty(tmp_path):
    directory = make_session(tmp_path)
    assert call(directory, "a", "evas_write", path="dut.va", content="invalid syntax")["ok"]
    assert call(directory, "b", "evas_write", path="dut.va", content="")["ok"]
    receipt = session.close_session(directory, "timeout")
    assert receipt["state"] == "collected"
    assert (directory / "candidate/files/dut.va").read_bytes() == b""
    assert session.close_session(directory, "completed") == receipt
    assert (
        call(directory, "c", "evas_write", path="dut.va", content="later")["error"]
        == "submission_frozen"
    )


def test_snapshot_budget_and_duplicate_action(tmp_path, monkeypatch):
    directory = make_session(tmp_path)
    seen = []

    def run(**kwargs):
        seen.append((kwargs["candidate"] / "files/dut.va").read_text())
        return {"execution": "backend_error", "diagnostics": "synthetic failure"}

    monkeypatch.setattr(session, "run_public", run)
    call(directory, "write", "evas_write", path="dut.va", content="version one")
    first = call(directory, "sim", "evas_simulate")
    assert first == call(directory, "sim", "evas_simulate")
    assert seen == ["version one"]
    assert call(directory, "write2", "evas_write", path="dut.va", content="version two")["ok"]
    assert call(directory, "sim2", "evas_simulate")["error"] == "simulation_budget_exhausted"
    submitted = call(directory, "submit", "evas_submit")["result"]
    assert submitted["candidate_directory"] == str(directory / "candidate")
    assert (directory / "candidate/files/dut.va").read_text() == "version two"


def test_public_paths_and_unknown_actions_fail_closed(tmp_path):
    directory = make_session(tmp_path)
    assert call(directory, "read", "evas_read", path="instruction.md")["ok"]
    assert not call(directory, "private", "evas_read", path="../session.json")["ok"]
    assert not call(directory, "bench", "evas_write", path="instruction.md", content="bad")["ok"]
    with pytest.raises(ValueError):
        call(directory, "unknown", "shell", command="true")
    with pytest.raises(ValueError):
        call(directory, "read", "evas_read", path="other")


def test_simulation_snapshot_does_not_block_edit_or_deadline_collection(tmp_path, monkeypatch):
    import threading

    directory = make_session(tmp_path)
    config_path = directory / "session.json"
    import json

    config = json.loads(config_path.read_text())
    config["max_simulations"] = 2
    config_path.write_text(json.dumps(config))
    started, finish = threading.Event(), threading.Event()
    seen = []

    def run(**kwargs):
        started.set()
        assert finish.wait(3)
        seen.append((kwargs["candidate"] / "files/dut.va").read_text())
        return {"execution": "ok"}

    monkeypatch.setattr(session, "run_public", run)
    call(directory, "a", "evas_write", path="dut.va", content="old")
    worker = threading.Thread(target=lambda: call(directory, "sim", "evas_simulate"))
    worker.start()
    try:
        assert started.wait(2)
        assert call(directory, "sim", "evas_simulate")["error"] == "unknown_execution"
        assert call(directory, "sim2", "evas_simulate")["error"] == "unresolved_previous_simulation"
        assert not (directory / "actions/sim2").exists()
        assert call(directory, "b", "evas_write", path="dut.va", content="new")["ok"]
        assert session.close_session(directory, "deadline")["state"] == "collected"
        assert (directory / "candidate/files/dut.va").read_text() == "new"
    finally:
        finish.set()
        worker.join(timeout=3)
    assert seen == ["old"]


def test_feedback_is_only_task_declared_public_projection(tmp_path, monkeypatch):
    directory = make_session(tmp_path)
    monkeypatch.setattr(
        session,
        "run_public",
        lambda **_: {
            "execution": "ok",
            "diagnostics": "public",
            "observations": {"hidden": 1},
            "reward": 1,
        },
    )
    call(directory, "write", "evas_write", path="dut.va", content="candidate")
    result = call(directory, "simulate", "evas_simulate")["result"]
    assert result["diagnostics"] == "public"
    assert "observations" not in result
    assert "reward" not in result


def test_experiments_are_opt_in_and_final_collects_only_formal_files(tmp_path, monkeypatch):
    declaration = {
        "version": "measurement-v1",
        "files": ["probe.py", "stimulus.json"],
        "analyses": ["python_measurement"],
    }
    directory = make_session(tmp_path, experiments=declaration)
    monkeypatch.setattr(session, "run_public", lambda **kw: {"execution": "ok"})
    call(directory, "candidate", "evas_write", path="dut.va", content="complete candidate")
    call(directory, "script", "evas_write", path="probe.py", content="print('{}')")
    call(directory, "stimulus", "evas_write", path="stimulus.json", content="{}")
    result = call(
        directory, "experiment", "evas_experiment", analysis="python_measurement", script="probe.py"
    )
    assert result["ok"]
    assert result["result"]["authority"] == "agent_measurement"
    frozen = session.verify_candidate(directory / "actions/experiment/experiment")
    assert set(frozen["files"]) == {"dut.va", "probe.py", "stimulus.json", "instruction.md"}
    assert frozen["task_version"] == '["fixture-v1","measurement-v1"]'
    assert call(directory, "bad", "evas_write", path="hidden.py", content="bad")["ok"] is False
    final = session.close_session(directory, "completed")
    assert set(session.verify_candidate(Path(final["candidate_directory"]))["files"]) == {"dut.va"}


def test_fixed_task_does_not_authorize_experiments(tmp_path):
    directory = make_session(tmp_path)
    assert "evas_experiment" not in {
        tool["function"]["name"] for tool in session.session_info(directory)["tools"]
    }
    assert not call(directory, "script", "evas_write", path="probe.py", content="print(1)")["ok"]
    assert not call(
        directory, "execute", "evas_experiment", analysis="python_measurement", script="probe.py"
    )["ok"]


def test_unsupported_experiment_backend_and_analysis_fail_closed(tmp_path):
    declaration = {"version": "v1", "files": ["probe.py"], "analyses": ["python_measurement"]}
    with pytest.raises(ValueError, match="container-only"):
        make_session(tmp_path, experiments=declaration, backend="native_codex_sandbox", image=None)


def test_unenabled_measurement_reports_explicit_public_reason(tmp_path):
    directory = make_session(tmp_path)
    result = call(
        directory,
        "unsupported",
        "evas_experiment",
        analysis="python_measurement",
        script="probe.py",
    )
    assert result == {
        "ok": False,
        "error": "ValueError",
        "detail": "unsupported or undeclared public experiment",
    }


def test_explicit_null_experiment_declaration_is_rejected_before_session_creation(tmp_path):
    import json

    directory = make_session(tmp_path)
    task = json.loads((directory / "session.json").read_text())["task"]
    task["experiments"] = None
    rejected = tmp_path / "null-session"
    with pytest.raises(ValueError, match="declare a versioned container-only"):
        session.create_session(
            task=task,
            materials=tmp_path / "materials",
            checkout=tmp_path / "engine",
            kernel=tmp_path / "kernel",
            directory=rejected,
            image="sha256:" + "a" * 64,
        )
    assert not rejected.exists()
