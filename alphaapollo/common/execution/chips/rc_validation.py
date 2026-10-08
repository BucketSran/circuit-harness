"""Independent analytical oracle for ideal RC data, not arbitrary circuit signoff.

Consumes numerical files and task parameters only, not simulator pass markers
or expected parameters inferred from the generated netlist.
"""

from __future__ import annotations

import math
from pathlib import Path

AC_ABSOLUTE_TOLERANCE = 1e-5
TRANSIENT_NORMALIZED_TOLERANCE = 5e-5


def _table(path: Path, columns: int) -> list[list[float]]:
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = [float(value) for value in line.split()]
            except ValueError as error:
                raise ValueError(f"non-numeric {path.name}") from error
            if len(row) != columns or not all(math.isfinite(v) for v in row):
                raise ValueError(f"invalid columns/nonfinite values in {path.name}")
            if rows and row[0] <= rows[-1][0]:
                raise ValueError(f"non-increasing coordinate in {path.name}")
            rows.append(row)
            if len(rows) > 100_000:
                raise ValueError(f"too many samples in {path.name}")
    if not rows:
        raise ValueError(f"empty {path.name}")
    return rows


def grade_rc(task: dict, ac_path: Path, transient_path: Path) -> dict:
    """Require the full AC grid and dense transient coverage before comparing values."""
    tau = task["resistance_ohm"] * task["capacitance_f"]
    voltage = task["voltage_v"]
    cutoff = 1 / (2 * math.pi * tau)
    ac = _table(ac_path, 3)
    if len(ac) != 121 or any(
        not math.isclose(row[0], cutoff * 10 ** (-3 + i / 20), rel_tol=1e-8)
        for i, row in enumerate(ac)
    ):
        raise ValueError("incomplete or incorrect AC frequency grid")
    ac_error = max(
        abs(complex(real, imag) - 1 / complex(1, 2 * math.pi * frequency * tau))
        for frequency, real, imag in ac
    )
    transient = _table(transient_path, 2)
    if (
        len(transient) < 1000
        or not 0 <= transient[0][0] <= tau * 0.001
        or not math.isclose(transient[-1][0], tau * 10, rel_tol=1e-8)
        or any(
            b[0] - a[0] > tau * 0.01001
            for a, b in zip(transient, transient[1:])  # noqa: B905 - Python 3.9 host
        )
    ):
        raise ValueError("incomplete or too sparse transient time grid")
    transient_error = max(
        abs(value / voltage - (-math.expm1(-time / tau))) for time, value in transient
    )
    passed = ac_error <= AC_ABSOLUTE_TOLERANCE and transient_error <= TRANSIENT_NORMALIZED_TOLERANCE
    return {
        "grader": "ideal_rc_v1",
        "verdict": "pass" if passed else "fail",
        "tau_s": tau,
        "cutoff_hz": cutoff,
        "ac_samples": len(ac),
        "transient_samples": len(transient),
        "ac_max_abs_error": ac_error,
        "ac_absolute_tolerance": AC_ABSOLUTE_TOLERANCE,
        "transient_max_normalized_error": transient_error,
        "transient_normalized_tolerance": TRANSIENT_NORMALIZED_TOLERANCE,
    }
