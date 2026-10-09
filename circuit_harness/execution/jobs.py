"""Compatibility import for :mod:`circuit_harness.execution.runtime.jobs`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("circuit_harness.execution.runtime.jobs")
