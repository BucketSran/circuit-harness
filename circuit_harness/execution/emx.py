"""Compatibility import for :mod:`circuit_harness.execution.backends.emx`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("circuit_harness.execution.backends.emx")
