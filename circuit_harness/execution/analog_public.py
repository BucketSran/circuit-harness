"""Compatibility import for :mod:`circuit_harness.execution.sessions.analog_public`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("circuit_harness.execution.sessions.analog_public")
