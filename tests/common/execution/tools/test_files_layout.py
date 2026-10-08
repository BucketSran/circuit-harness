"""Compatibility tests for the split workspace file tools."""

from alphaapollo.common.execution.tools.builtins._files.tools import ReadTool as SplitReadTool
from alphaapollo.common.execution.tools.builtins.files import ReadTool


def test_files_facade_reexports_split_implementation() -> None:
    assert ReadTool is SplitReadTool
