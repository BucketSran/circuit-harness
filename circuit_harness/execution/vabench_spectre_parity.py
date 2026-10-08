"""Compare VABench's visible transient CSV with a separate Spectre PSFASCII run.

This is an operator analysis of public waveforms, not a VABench final scorer.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
from pathlib import Path

from .journal import atomic_json, file_digest

SIGNALS = ("data", "clk", "retimed_data", "up", "down")
_PSF_VALUE = re.compile(r'^"([a-z_]+)" ([-+\d.eE]+)$')


def _safe_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("missing or unsafe waveform file")
    return path.read_text(encoding="utf-8", errors="strict")


def _psf_rows(path: Path) -> list[dict[str, float]]:
    lines = _safe_text(path).splitlines()
    try:
        trace = lines.index("TRACE")
        value = lines.index("VALUE", trace + 1)
    except ValueError as error:
        raise ValueError("missing PSF trace/value section") from error
    if lines[trace + 1 : value] != [f'"{name}" "V"' for name in SIGNALS]:
        raise ValueError("unexpected PSF trace set")
    fields = ("time", *SIGNALS)
    body = lines[value + 1 :]
    if not body or body[-1] != "END" or (len(body) - 1) % len(fields):
        raise ValueError("truncated or unterminated PSF")
    rows = []
    for offset in range(0, len(body) - 1, len(fields)):
        row = {}
        for index, name in enumerate(fields):
            line = body[offset + index]
            match = _PSF_VALUE.fullmatch(line)
            if not match or match[1] != name:
                raise ValueError("invalid PSF value order")
            row[name] = float(match[2])
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError("nonfinite PSF value")
        if rows and row["time"] <= rows[-1]["time"]:
            raise ValueError("non-increasing PSF time")
        rows.append(row)
        if len(rows) > 100_000:
            raise ValueError("too many PSF samples")
    if len(rows) < 2:
        raise ValueError("too few PSF samples")
    return rows


def _csv_rows(path: Path) -> list[dict[str, float]]:
    text = _safe_text(path)
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames != ["time", *SIGNALS]:
        raise ValueError("unexpected visible CSV columns")
    rows = []
    for raw in reader:
        row = {name: float(raw[name]) for name in ("time", *SIGNALS)}
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError("nonfinite visible CSV value")
        if rows and row["time"] <= rows[-1]["time"]:
            raise ValueError("non-increasing visible CSV time")
        rows.append(row)
        if len(rows) > 100_000:
            raise ValueError("too many visible CSV samples")
    if len(rows) < 2:
        raise ValueError("too few visible CSV samples")
    return rows


def _crossings(rows: list[dict[str, float]]) -> list[float]:
    times = []
    for index in range(1, len(rows)):
        previous, current = rows[index - 1], rows[index]
        for name in SIGNALS[:3]:
            first, last = previous[name], current[name]
            if first != last and min(first, last) <= 0.45 <= max(first, last):
                times.append(
                    previous["time"]
                    + (0.45 - first) * (current["time"] - previous["time"]) / (last - first)
                )
    return times


def compare_visible_waveforms(evas_csv: Path, spectre_psf: Path) -> dict:
    """Interpolate Spectre onto EVAS times and report raw and quiet-window errors."""
    evas_csv, spectre_psf = Path(evas_csv), Path(spectre_psf)
    evas = _csv_rows(evas_csv)
    spectre = _psf_rows(spectre_psf)
    times = [row["time"] for row in spectre]
    if evas[0]["time"] < times[0] or evas[-1]["time"] > times[-1]:
        raise ValueError("visible CSV extends beyond Spectre PSF")
    crossings = _crossings(evas)
    quiet_margin_s = 2e-10
    result = {
        "method": "linear_interpolation_to_visible_csv_times",
        "quiet_margin_s": quiet_margin_s,
        "evas_csv_sha256": file_digest(evas_csv),
        "spectre_psf_sha256": file_digest(spectre_psf),
        "evas_points": len(evas),
        "spectre_points": len(spectre),
        "signals": {},
    }
    for name in SIGNALS:
        errors = []
        quiet_errors = []
        mismatches = 0
        quiet_mismatches = 0
        for row in evas:
            time = row["time"]
            position = bisect.bisect_left(times, time)
            if position < len(times) and times[position] == time:
                value = spectre[position][name]
            else:
                left, right = spectre[position - 1], spectre[position]
                weight = (time - left["time"]) / (right["time"] - left["time"])
                value = left[name] + weight * (right[name] - left[name])
            error = abs(row[name] - value)
            mismatch = (row[name] >= 0.45) != (value >= 0.45)
            errors.append(error)
            mismatches += mismatch
            if all(abs(time - transition) >= quiet_margin_s for transition in crossings):
                quiet_errors.append(error)
                quiet_mismatches += mismatch
        result["signals"][name] = {
            "max_abs_error_v": max(errors),
            "mean_abs_error_v": sum(errors) / len(errors),
            "quiet_max_abs_error_v": max(quiet_errors) if quiet_errors else None,
            "logic_mismatch_samples": mismatches,
            "quiet_logic_mismatch_samples": quiet_mismatches,
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evas-csv", type=Path, required=True)
    parser.add_argument("--spectre-psf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_visible_waveforms(args.evas_csv, args.spectre_psf)
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "signals": report["signals"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
