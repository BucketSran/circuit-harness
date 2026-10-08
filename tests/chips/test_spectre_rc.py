"""Constructed Spectre protocol fixtures; live simulator evidence is separate."""

import json
import math
import shutil
import time
from pathlib import Path

import pytest

from alphaapollo.common.execution.chips.archive import verify_archive
from alphaapollo.common.execution.chips.jobs import inspect_job, verify_job
from alphaapollo.common.execution.chips.spectre import (
    read_psf,
    run_spectre_rc,
    spectre_identity,
    verify_spectre_rc,
)
from alphaapollo.workflows.chips import main

TASK = {"resistance_ohm": 1000, "capacitance_f": 1e-9}


@pytest.fixture
def profile(tmp_path):
    for name in ("jobs", "archives"):
        (tmp_path / name).mkdir(mode=0o700)
    setup = tmp_path / "cadence.sui"
    setup.write_text("setenv CHIPS_SPECTRE_TEST ready\n")
    spectre = tmp_path / "spectre"
    spectre.write_text("#!/bin/sh\n")
    spectre.chmod(0o700)
    config = {
        "schema_version": 1,
        "task_id": "RC-001",
        "shell": "/bin/csh",
        "setup_scripts": [str(setup)],
        "spectre": str(spectre),
        "run_root": str(tmp_path / "jobs"),
        "archive_root": str(tmp_path / "archives"),
        "timeout_s": 60,
        "license_queue_s": 5,
    }
    path = tmp_path / "spectre.profile.json"
    path.write_text(json.dumps(config))
    path.chmod(0o600)
    return path


def test_identity_pins_private_profile_and_setup_content(profile, tmp_path):
    identity = spectre_identity(TASK, profile)
    assert identity["backend"] == "spectre_rc"
    assert identity["task"]["resistance_ohm"] == 1000
    assert identity["setup_scripts"][0]["sha256"]
    assert identity["license_queue_s"] == 5
    setup = tmp_path / "cadence.sui"
    setup.write_text("setenv CHIPS_SPECTRE_TEST changed\n")
    assert spectre_identity(TASK, profile) != identity


def test_profile_rejects_public_file_or_extra_commands(profile, tmp_path):
    profile.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        spectre_identity(TASK, profile)
    profile.chmod(0o600)
    config = json.loads(profile.read_text())
    config["setup_scripts"] = ["/tmp/unsafe;virtuoso"]
    profile.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="setup_scripts"):
        spectre_identity(TASK, profile)
    assert not (tmp_path / "jobs" / "new-job").exists()


def test_psf_reader_rejects_missing_trace_and_nonfinite_values(tmp_path):
    ac = tmp_path / "ac1.ac"
    ac.write_text(
        'TRACE\n"in" "V"\n"out" "V"\nVALUE\n"freq" 1000\n"in" (1 0)\n"out" (0.5 -0.5)\nEND\n'
    )
    assert read_psf(ac, axis="freq", complex_values=True) == [(1000.0, 0.5, -0.5)]
    ac.write_text(ac.read_text().replace('"out" (0.5 -0.5)', '"out" (nan 0)'))
    with pytest.raises(ValueError, match="nonfinite"):
        read_psf(ac, axis="freq", complex_values=True)
    ac.write_text('VALUE\n"freq" 1000\n"out" (0.5 -0.5)\nEND\n')
    with pytest.raises(ValueError, match="trace"):
        read_psf(ac, axis="freq", complex_values=True)


def test_transient_psf_reader_uses_out_voltage(tmp_path):
    tran = tmp_path / "tran1.tran.tran"
    tran.write_text(
        'TRACE\n"in" "V"\n"out" "V"\nVALUE\n'
        '"time" 0\n"in" 1\n"out" 0\n'
        '"time" 1e-6\n"in" 1\n"out" 0.6321205588\nEND\n'
    )
    assert read_psf(tran, axis="time", complex_values=False) == [
        (0.0, 0.0),
        (1e-6, pytest.approx(-math.expm1(-1))),
    ]


def test_psf_reader_rejects_wrong_source_voltage(tmp_path):
    ac = tmp_path / "ac1.ac"
    ac.write_text(
        'TRACE\n"in" "V"\n"out" "V"\nVALUE\n"freq" 1000\n"in" (0 0)\n"out" (0.5 -0.5)\nEND\n'
    )
    with pytest.raises(ValueError, match="input voltage"):
        read_psf(ac, axis="freq", complex_values=True, expected_input=1)
    tran = tmp_path / "tran1.tran.tran"
    tran.write_text('TRACE\n"in" "V"\n"out" "V"\nVALUE\n"time" 0\n"in" 1\n"out" 0\nEND\n')
    with pytest.raises(ValueError, match="input voltage"):
        read_psf(tran, axis="time", complex_values=False, expected_input=2)


def _fake_psf(tmp_path: Path) -> None:
    source = tmp_path / "fixture"
    source.mkdir()
    tau = 1e-6
    cutoff = 1 / (2 * math.pi * tau)
    ac = ["TRACE", '"in" "V"', '"out" "V"', "VALUE"]
    for index in range(121):
        frequency = cutoff * 10 ** (-3 + index / 20)
        value = 1 / complex(1, 2 * math.pi * frequency * tau)
        ac.extend(
            (f'"freq" {frequency:.15e}', '"in" (1 0)', f'"out" ({value.real:.6g} {value.imag:.6g})')
        )
    (source / "ac1.ac").write_text("\n".join([*ac, "END", ""]))
    tran = ["TRACE", '"in" "V"', '"out" "V"', "VALUE"]
    for index in range(1001):
        instant = index * tau / 100
        value = -math.expm1(-instant / tau)
        tran.extend((f'"time" {instant:.15e}', '"in" 1', f'"out" {value:.6g}'))
    (source / "tran1.tran.tran").write_text("\n".join([*tran, "END", ""]))
    simulator = tmp_path / "spectre"
    simulator.write_text(f'''#!/bin/sh
mkdir psf
cp "{source}/ac1.ac" psf/ac1.ac
cp "{source}/tran1.tran.tran" psf/tran1.tran.tran
echo 'Spectre fixture completed' > spectre.log
''')
    simulator.chmod(0o700)


def test_constructed_spectre_process_grades_raw_psf_and_detects_tamper(profile, tmp_path):
    _fake_psf(tmp_path)
    identity = spectre_identity(TASK, profile)
    result = run_spectre_rc(identity, tmp_path / "run")
    assert result["execution"] == "ok"
    assert result["verdict"] == "pass"
    assert result["metrics"]["ac_samples"] == 121
    assert result["metrics"]["transient_samples"] == 1001
    assert verify_spectre_rc(tmp_path / "run") == result
    shutil.move(str(tmp_path / "run" / "psf"), str(tmp_path / "outside-psf"))
    (tmp_path / "run" / "psf").symlink_to(tmp_path / "outside-psf")
    with pytest.raises(ValueError, match="modified"):
        verify_spectre_rc(tmp_path / "run")
    (tmp_path / "run" / "psf").unlink()
    shutil.move(str(tmp_path / "outside-psf"), str(tmp_path / "run" / "psf"))
    (tmp_path / "run" / "psf" / "ac1.ac").write_text("tampered")
    with pytest.raises(ValueError, match="modified"):
        verify_spectre_rc(tmp_path / "run")


def test_constructed_license_failure_is_not_a_circuit_failure(profile, tmp_path):
    simulator = tmp_path / "spectre"
    simulator.write_text("#!/bin/sh\necho SPECTRE-209 > spectre.log\nexit 1\n")
    simulator.chmod(0o700)
    result = run_spectre_rc(spectre_identity(TASK, profile), tmp_path / "run")
    assert result["execution"] == "infrastructure_error"
    assert result["verdict"] == "not_evaluated"
    assert result["reason"] == "license_checkout_failed"


def test_cli_detaches_spectre_job_and_archives_verified_result(profile, tmp_path, capsys):
    _fake_psf(tmp_path)
    task = tmp_path / "task.json"
    task.write_text(json.dumps(TASK))
    args = [
        "submit-spectre-rc",
        "--input",
        str(task),
        "--profile",
        str(profile),
        "--job-id",
        "spectre-fixture-001",
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["state"] in {"running", "finished"}
    root = tmp_path / "jobs" / "spectre-fixture-001"
    deadline = time.monotonic() + 15
    while (
        inspect_job(root).get("archive", {}).get("state") != "verified"
        and time.monotonic() < deadline
    ):
        time.sleep(0.1)
    assert inspect_job(root)["archive"]["state"] == "verified"
    assert verify_job(root)["result"]["verdict"] == "pass"
    assert (
        verify_archive(tmp_path / "archives" / "spectre-fixture-001")["completion"]["result"][
            "verdict"
        ]
        == "pass"
    )
    assert main(args) == 0
    assert inspect_job(root)["archive"]["state"] == "verified"
