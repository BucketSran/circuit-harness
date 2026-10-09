"""Compatibility import for :mod:`circuit_harness.execution.backends.vabench_public_worker`."""

import runpy
import sys
from importlib import import_module

if __name__ == "__main__":
    if not __package__:
        from pathlib import Path

        runpy.run_path(
            str(Path(__file__).parent / "backends/vabench_public_worker.py"), run_name="__main__"
        )
    else:
        runpy.run_module(
            "circuit_harness.execution.backends.vabench_public_worker", run_name="__main__"
        )
else:
    sys.modules[__name__] = import_module(
        "circuit_harness.execution.backends.vabench_public_worker"
    )
