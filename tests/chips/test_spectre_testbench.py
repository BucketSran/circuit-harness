"""Constructed AC fixtures matching the measured Spectre PSF subset."""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest

from circuit_harness.execution.journal import file_digest
from circuit_harness.execution.spectre_testbench import (
    REQUIREMENTS,
    grade_testbench,
    measure_gain,
    prepare_testbench,
    run_testbench,
    verify_testbench,
)
from circuit_harness.execution.spectre_testbench import (
    testbench_identity as gain_identity,
)
from circuit_harness.execution.task_authoring import confirm_draft


def candidate():
    return {
        "ports": ["vip", "vin", "out", "vdd", "0"],
        "positive_ac": {"magnitude": 0.5, "phase_deg": 0},
        "negative_ac": {"magnitude": 0.5, "phase_deg": 180},
        "measurement": {"numerator": ["out", "0"], "denominator": ["vip", "vin"]},
    }


def psf_fixture(path):
    path.write_text(
        'TRACE\n"vip" "V"\n"vin" "V"\n"out" "V"\n"vdd" "V"\nVALUE\n'
        '"freq" 10\n"vip" (0.5 0)\n"vin" (-0.5 0)\n'
        '"out" (990.099 -99.0099)\n"vdd" (0 0)\nEND\n'
    )


def test_gain_uses_differential_input_and_original_complex_psf(tmp_path):
    path = tmp_path / "ac1.ac"
    psf_fixture(path)
    measured = measure_gain(path, candidate(), 10)
    assert measured["gain_db"] == pytest.approx(59.95678626, abs=0.00001)
    assert measured["phase_deg"] == pytest.approx(-5.710593, abs=0.00001)
    wrong = candidate()
    wrong["measurement"]["denominator"] = ["vip", "0"]
    assert measure_gain(path, wrong, 10)["gain_db"] == pytest.approx(
        measured["gain_db"] + 20 * math.log10(2)
    )


def test_testbench_correctness_is_separate_from_dut_acceptance(tmp_path):
    path = tmp_path / "ac1.ac"
    psf_fixture(path)
    reference = {"gain0": 1000, "pole_hz": 100}
    result = grade_testbench(path, candidate(), 10, 55, reference)
    assert result["testbench_valid"] is True
    assert result["dut_pass"] is True
    path.write_text(path.read_text().replace("990.099 -99.0099", "99.0099 -9.90099"))
    reference["gain0"] = 100
    result = grade_testbench(path, candidate(), 10, 55, reference)
    assert result["testbench_valid"] is True
    assert result["dut_pass"] is False
    wrong = candidate()
    wrong["measurement"]["denominator"] = ["vip", "0"]
    assert grade_testbench(path, wrong, 10, 55, reference)["testbench_valid"] is False


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ('"out" (990.099 -99.0099)', '"out" (nan 0)'),
        ('"out" (990.099 -99.0099)', '"out" (inf 0)'),
        ('"freq" 10', '"freq" 20'),
        ('"out" "V"', '"other" "V"'),
        ('"vin" (-0.5 0)', '"vin" (0.5 0)'),
        ("END\n", ""),
    ],
)
def test_invalid_results_never_become_successful_measurements(tmp_path, before, after):
    path = tmp_path / "ac1.ac"
    psf_fixture(path)
    path.write_text(path.read_text().replace(before, after))
    with pytest.raises(ValueError):
        grade_testbench(path, candidate(), 10, 55, {"gain0": 1000, "pole_hz": 100})


def confirmed_task(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("Constructed operator fixture, values defined in this test.\n")
    values = {
        "supply_v": 1.8,
        "common_mode_v": 0.9,
        "temperature_c": 27,
        "load_ohm": 1e6,
        "frequency_hz": 10,
        "minimum_gain_db": 55,
        "dut_ports": ["vip", "vin", "out", "vdd", "0"],
    }
    draft = {
        "schema_version": 1,
        "task_id": "gain-d0",
        "sources": [{"path": path.name, "sha256": file_digest(path)}],
        "fields": {
            name: {
                "value": value,
                "unit": REQUIREMENTS[name],
                "status": "explicit",
                "evidence": [{"source": path.name, "page": 1, "region": "fixture"}],
            }
            for name, value in values.items()
        },
    }
    receipt = confirm_draft(
        draft, tmp_path, REQUIREMENTS, reviewer="fixture", kind="scripted_confirmation"
    )
    return draft, receipt


def test_preparation_requires_current_confirmation_and_emits_declared_measurement(tmp_path):
    draft, receipt = confirmed_task(tmp_path)
    with pytest.raises(ValueError, match="confirmation"):
        prepare_testbench(draft, tmp_path, None, candidate(), {"gain0": 1000, "pole_hz": 100})
    prepared = prepare_testbench(
        draft, tmp_path, receipt, candidate(), {"gain0": 1000, "pole_hz": 100}
    )
    assert 'ahdl_include "reference.va"' in prepared["netlist"]
    assert "ac1 ac values=[10]" in prepared["netlist"]
    assert prepared["candidate"]["measurement"] == candidate()["measurement"]
    draft["fields"]["frequency_hz"]["value"] = 20
    with pytest.raises(ValueError, match="confirmation"):
        prepare_testbench(draft, tmp_path, receipt, candidate(), {"gain0": 1000, "pole_hz": 100})


@pytest.mark.parametrize("case", ["port", "nan", "bool", "measurement", "unknown", "reference"])
def test_candidate_is_bounded_data_not_executable_code(tmp_path, case):
    draft, receipt = confirmed_task(tmp_path)
    bad = candidate()
    reference = {"gain0": 1000, "pole_hz": 100}
    if case == "port":
        bad["ports"][0] = 'vip)\nahdl_include "/untrusted.va"\n('
    elif case == "nan":
        bad["positive_ac"]["magnitude"] = float("nan")
    elif case == "bool":
        bad["negative_ac"]["magnitude"] = True
    elif case == "measurement":
        bad["measurement"]["denominator"] = ["out", "out"]
    elif case == "unknown":
        bad["script"] = "run arbitrary shell"
    else:
        reference["gain0"] = -1
    with pytest.raises(ValueError):
        prepare_testbench(draft, tmp_path, receipt, bad, reference)


@pytest.fixture
def gain_submission(tmp_path):
    draft, receipt = confirmed_task(tmp_path)
    fixture = tmp_path / "fixture.ac"
    psf_fixture(fixture)
    simulator = tmp_path / "spectre"
    simulator.write_text(
        f'#!/bin/sh\nmkdir psf\ncp "{fixture}" psf/ac1.ac\necho fixture > spectre.log\n'
    )
    simulator.chmod(0o700)
    setup = tmp_path / "setup.csh"
    setup.write_text("# constructed environment\n")
    for name in ("jobs", "archives"):
        (tmp_path / name).mkdir(mode=0o700)
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "OPAMP-GAIN-001",
                "shell": "/bin/csh",
                "setup_scripts": [str(setup)],
                "spectre": str(simulator),
                "run_root": str(tmp_path / "jobs"),
                "archive_root": str(tmp_path / "archives"),
                "timeout_s": 10,
                "license_queue_s": 1,
            }
        )
    )
    profile.chmod(0o600)
    model = (
        Path(__file__).resolve().parents[2]
        / "examples/chips/task_authoring/opamp_gain/reference.va"
    )
    payload = {
        "draft": draft,
        "confirmation": receipt,
        "materials": str(tmp_path),
        "candidate": candidate(),
        "reference": {"gain0": 1000, "pole_hz": 100},
        "model": str(model),
    }
    return payload, profile


def test_constructed_simulator_runs_confirmed_candidate_and_rechecks_archive(
    tmp_path, gain_submission
):
    payload, profile = gain_submission
    identity = gain_identity(payload, profile)
    result = run_testbench(identity, tmp_path / "run")
    assert result["execution"] == "ok"
    assert result["verdict"] == "pass"
    assert result["metrics"]["testbench_valid"] is True
    assert verify_testbench(tmp_path / "run") == result
    (tmp_path / "run/psf/ac1.ac").write_text("modified")
    with pytest.raises(ValueError, match="modified"):
        verify_testbench(tmp_path / "run")


def test_offline_cli_submits_detached_gain_job_and_archives(tmp_path, gain_submission):
    from circuit_harness.execution.bundle import build_cli
    from circuit_harness.execution.jobs import inspect_job, verify_job

    payload, profile = gain_submission
    bundle = tmp_path / "chips.pyz"
    build_cli(bundle)
    task = tmp_path / "task.json"
    task.write_text(json.dumps(payload))
    proc = subprocess.run(
        [
            sys.executable,
            str(bundle),
            "submit-spectre-gain",
            "--input",
            str(task),
            "--profile",
            str(profile),
            "--job-id",
            "gain-001",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    job = tmp_path / "jobs/gain-001"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = inspect_job(job)
        if state.get("archive", {}).get("state") == "verified":
            break
        time.sleep(0.05)
    assert state["state"] == "finished", state
    assert state["archive"]["state"] == "verified", state
    assert verify_job(job)["result"]["verdict"] == "pass"


def test_authoring_session_reserves_budget_deduplicates_and_freezes(tmp_path, gain_submission):
    from circuit_harness.execution.authoring_session import (
        action_response,
        create_session,
        request_action,
    )

    payload, profile = gain_submission
    session = tmp_path / "session"
    assert create_session(session, payload, profile, max_simulations=1)["status"] == "ready"
    request = {"id": "try-001", "tool": "gain_simulate", "arguments": {"candidate": candidate()}}
    assert request_action(session, request)["state"] == "accepted"
    assert request_action(session, request)["state"] == "accepted"
    with pytest.raises(ValueError, match="different"):
        request_action(session, {**request, "tool": "gain_submit"})
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = action_response(session, "try-001")
        if result.get("ok") is not None:
            break
        time.sleep(0.05)
    assert result["ok"] is True, result
    assert result["result"]["measurement"]["gain_db"] == pytest.approx(59.95678626)
    request_action(session, {**request, "id": "try-002"})
    assert action_response(session, "try-002")["error"] == "simulation_budget_exhausted"
    frozen = {"id": "submit-001", "tool": "gain_submit", "arguments": {"candidate": candidate()}}
    request_action(session, frozen)
    assert action_response(session, "submit-001")["result"]["status"] == "submitted"
    request_action(session, {**request, "id": "try-003"})
    assert action_response(session, "try-003")["error"] == "submission_frozen"


def test_session_refuses_unconfirmed_draft_and_changed_sources(tmp_path, gain_submission):
    from circuit_harness.execution.authoring_session import (
        action_response,
        create_session,
        request_action,
    )

    payload, profile = gain_submission
    original = payload["confirmation"]
    payload["confirmation"] = None
    with pytest.raises(ValueError, match="confirmation"):
        create_session(tmp_path / "rejected", payload, profile)
    assert not (tmp_path / "rejected").exists()
    payload["confirmation"] = original
    session = tmp_path / "session"
    create_session(session, payload, profile)
    (tmp_path / "source.txt").write_text("changed source")
    request_action(
        session, {"id": "changed", "tool": "gain_simulate", "arguments": {"candidate": candidate()}}
    )
    assert action_response(session, "changed")["ok"] is False
    assert not list((tmp_path / "jobs").iterdir())


def test_bundled_session_cli_accepts_and_returns_same_frozen_request(tmp_path, gain_submission):
    from circuit_harness.execution.bundle import build_cli

    bundle = tmp_path / "chips.pyz"
    build_cli(bundle)
    payload, profile = gain_submission
    task = tmp_path / "task.json"
    task.write_text(json.dumps(payload))
    session = tmp_path / "session"

    def cli(*args, payload=None):
        result = subprocess.run(
            [sys.executable, str(bundle), *map(str, args)],
            input=json.dumps(payload) if payload else None,
            text=True,
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        return json.loads(result.stdout)

    assert (
        cli("gain-session", "--session", session, "--input", task, "--profile", profile)["status"]
        == "ready"
    )
    request = {"id": "early-submit", "tool": "gain_submit", "arguments": {"candidate": candidate()}}
    assert (
        cli("gain-request", "--session", session, "--request", "-", payload=request)["state"]
        == "accepted"
    )
    assert (
        cli("gain-response", "--session", session, "--id", "early-submit")["error"]
        == "candidate_requires_public_simulation"
    )
