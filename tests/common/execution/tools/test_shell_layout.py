"""Regression coverage for output ownership and the stable shell facade."""

from pathlib import Path

from alphaapollo.common.execution import output
from alphaapollo.common.execution.sandbox._podman import backend as podman_backend
from alphaapollo.common.execution.sandbox._podman import runner as podman_runner
from alphaapollo.common.execution.tools.builtins import shell
from alphaapollo.common.execution.tools.builtins._files import command as files_command
from alphaapollo.common.execution.tools.builtins._files import tools as files_tools
from alphaapollo.common.execution.tools.builtins._shell.bash import BashTool
from alphaapollo.common.execution.tools.builtins._shell.python import PythonTool


def test_shell_facade_reexports_canonical_implementations() -> None:
    assert shell.BashTool is BashTool
    assert shell.PythonTool is PythonTool
    assert shell.OutputAccumulator is output.OutputAccumulator
    assert shell.TruncationResult is output.TruncationResult


def test_podman_uses_backend_neutral_output_primitives() -> None:
    assert podman_runner.OutputAccumulator is output.OutputAccumulator
    assert podman_runner.TruncationResult is output.TruncationResult
    assert podman_backend.format_truncated_output is output.format_truncated_output
    for module in (podman_runner, podman_backend):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "execution.tools.builtins.shell" not in source


def test_implementations_reach_the_owner_not_the_facade() -> None:
    """A private package imports the canonical module, never the stable facade.

    The facade exists for callers outside the split. An implementation that goes
    through it reads as though `shell.py` owned `BashTool`, and would keep
    working if the facade grew a definition of its own.
    """

    assert files_tools.BashTool is BashTool
    for module in (files_tools, files_command):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "execution.tools.builtins.shell" not in source
