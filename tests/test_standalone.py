"""Exercise the installed public boundary without the retired Apollo runtime."""

import json
import subprocess
import sys


def test_operator_cli_is_standalone():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import sys; sys.path.insert(0, '.'); "
            "from circuit_harness.cli import main; main(['--help'])",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "vabench" in result.stdout and "analog" in result.stdout


def test_harbor_plugins_load_without_apollo():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class BlockApollo(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'alphaapollo', 'verl'}:
            raise AssertionError('Retired dependency: ' + fullname)
sys.meta_path.insert(0, BlockApollo())
from circuit_harness.harbor.installed_agent import CircuitAgent
from circuit_harness.harbor.environment import HarborChipsEnvironment
from circuit_harness.harbor.verifier import FrozenCandidateVerifier
from circuit_harness.data.prepare_atif import AtifDatasetManifest
assert CircuitAgent and HarborChipsEnvironment and FrozenCandidateVerifier and AtifDatasetManifest
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_ssh_worker_module_reports_missing_job(tmp_path):
    """The published worker module stays callable without optional dependencies."""
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "circuit_harness.execution.ssh_worker",
            "status",
            str(tmp_path),
            "1" * 32,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"job_id": "1" * 32, "state": "missing"}


def test_legacy_parity_module_cli_remains_standalone():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "circuit_harness.execution.vabench_spectre_parity", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--evas-csv" in result.stdout and "--spectre-psf" in result.stdout


def test_pinned_workers_can_execute_by_file_outside_the_checkout(tmp_path):
    from importlib.resources import files

    missing = tmp_path / "missing-session"
    for worker, arguments in (
        ("vabench_worker.py", ["export", str(missing), str(tmp_path / "candidate")]),
        ("vabench_public_worker.py", [str(missing), str(tmp_path / "action")]),
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(files("circuit_harness.execution") / worker),
                *arguments,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert "FileNotFoundError" in result.stderr, result.stderr
        assert str(missing) in result.stderr
