"""Compatibility import for :mod:`circuit_harness.execution.evaluation.vabench_spectre_parity`."""

import runpy
import sys
from importlib import import_module

if __name__ == "__main__":
    runpy.run_module(
        "circuit_harness.execution.evaluation.vabench_spectre_parity", run_name="__main__"
    )
else:
    sys.modules[__name__] = import_module(
        "circuit_harness.execution.evaluation.vabench_spectre_parity"
    )
