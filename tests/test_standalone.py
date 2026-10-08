"""Exercise the installed public boundary without the retired Apollo runtime."""

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
