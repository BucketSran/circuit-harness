"""EVAS onboarding uses benchmark-owned grading and immutable external source."""

import subprocess
import sys


def test_prepare_rejects_wrong_source_before_creating_workspace(tmp_path):
    checkout = tmp_path / "source"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    workspace = tmp_path / "state"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "circuit_harness.harbor.evas_example",
            "prepare",
            "--evas-checkout",
            str(checkout),
            "--workspace",
            str(workspace),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2
    assert "pinned public commit" in run.stderr
    assert not workspace.exists()


def test_run_preserves_existing_output_before_probing_runtime(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("previous evidence")
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "circuit_harness.harbor.evas_example",
            "run",
            "--workspace",
            str(tmp_path / "absent"),
            "--image",
            "invalid-image",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2
    assert "output already exists" in run.stderr
    assert (output / "keep.txt").read_text() == "previous evidence"


def test_result_rejects_valid_replay_for_another_task(tmp_path):
    import json

    from circuit_harness.execution.benchmark_replay import replay_candidate
    from circuit_harness.execution.benchmark_spectre import package_identity
    from circuit_harness.execution.candidate_bundle import freeze_candidate
    from circuit_harness.execution.journal import file_digest

    source = tmp_path / "source"
    source.mkdir()
    (source / "dut.va").write_text("module other; endmodule\n")
    output = tmp_path / "output"
    output.mkdir()
    freeze_candidate(
        source,
        output / "candidate",
        ["dut.va"],
        task_id="other-task",
        task_version="v1",
        reason="protocol fixture",
    )
    package = tmp_path / "package"
    package.mkdir()
    script = package / "test.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "other-task",
                "task_version": "v1",
                "criteria_sha256": "a" * 64,
                "condition_id": "fixture",
                "task_set": "public",
                "purpose": "final",
                "entrypoint": "test.sh",
                "candidate_file": "dut.va",
                "report_path": "verifier/report.json",
                "feedback_fields": [],
                "files": {
                    "test.sh": {"sha256": file_digest(script), "bytes": script.stat().st_size}
                },
            }
        )
    )
    replay_candidate(
        output / "candidate",
        package,
        {
            "schema_version": 1,
            "image": "sha256:" + "a" * 64,
            "task_package_sha256": package_identity(package, purpose="final")["sha256"],
            "solver_options": {},
            "unsupported": ["fixture does not run a simulator"],
            "timeout_s": 1,
            "max_output_bytes": 1024,
        },
        output / "replay",
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "circuit_harness.harbor.evas_example",
            "result",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2
    assert "pinned VA07" in run.stderr
    assert "source_commit" not in run.stdout
