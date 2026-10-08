"""The public diagnostic is reachable from the deployed standard-library bundle."""

import subprocess
import sys

import pytest

from circuit_harness import cli as chips
from circuit_harness.execution.bundle import build_cli


@pytest.mark.parametrize("task_id", [None, "rlc-broadband-50-to-200-match"])
def test_public_cli_dispatches_without_scoring(tmp_path, monkeypatch, capsys, task_id) -> None:
    received = {}

    def fake_run(source_root, candidate, output, **options):
        received.update(source_root=source_root, candidate=candidate, output=output, **options)
        return {
            "state": "simulated",
            "authority": "public_diagnostic",
            "task_correctness": "not_evaluated",
        }

    monkeypatch.setattr(chips, "run_public_rlc", fake_run, raising=False)
    assert (
        chips.main(
            [
                "analog-public",
                "--source-root",
                str(tmp_path / "source"),
                "--candidate",
                str(tmp_path / "candidate.spi"),
                "--output",
                str(tmp_path / "run"),
                "--timeout-s",
                "45",
                *(["--task-id", task_id] if task_id else []),
            ]
        )
        == 0
    )
    assert received["timeout_s"] == 45
    assert received["task_id"] == (task_id or "rlc-rf-bandpass-100mhz")
    assert received["candidate"] == tmp_path / "candidate.spi"
    assert '"task_correctness": "not_evaluated"' in capsys.readouterr().out


def test_public_cli_is_packaged_in_offline_bundle(tmp_path) -> None:
    bundle = tmp_path / "chips.pyz"
    build_cli(bundle)
    completed = subprocess.run(
        [sys.executable, str(bundle), "analog-public", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--source-root" in completed.stdout
