"""Compatibility import for :mod:`circuit_harness.execution.evaluation.analog_design_bench`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("circuit_harness.execution.evaluation.analog_design_bench")
