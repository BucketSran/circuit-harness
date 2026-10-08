"""Compatibility entrypoint and plugin imports for Analog Design Bench."""

from circuit_harness.benchmarks.analogbench import EXAMPLES, REPOSITORY, main, prepare
from circuit_harness.harbor.analogbench import (
    AnalogDockerEnvironment,
    OriginalAnalogVerifier,
    validate_grading_evidence,
)

__all__ = [
    "EXAMPLES",
    "REPOSITORY",
    "main",
    "prepare",
    "AnalogDockerEnvironment",
    "OriginalAnalogVerifier",
    "validate_grading_evidence",
]

if __name__ == "__main__":
    raise SystemExit(main())
