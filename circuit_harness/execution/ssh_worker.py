"""Compatibility import for :mod:`circuit_harness.execution.transport.ssh_worker`."""

import runpy
import sys
from importlib import import_module

if __name__ == "__main__":
    runpy.run_module("circuit_harness.execution.transport.ssh_worker", run_name="__main__")
else:
    sys.modules[__name__] = import_module("circuit_harness.execution.transport.ssh_worker")
