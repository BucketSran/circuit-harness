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
