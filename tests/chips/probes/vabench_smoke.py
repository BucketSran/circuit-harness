#!/usr/bin/env python3
"""Explicit, no-model acceptance using PRIVATE r53 reference/fault fixtures.

Run on the simulator host with --cli pointing to the standard-library bundle.
Only the operator may access this directory; never mount it into an agent.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

TASKS = {
    "dut": ("v4-001", "001-bang-bang-phase-detector"),
    "bugfix": ("v4-1001", "1001-bang-bang-phase-detector-bugfix"),
    "testbench": ("v4-501", "501-bang-bang-phase-detector-testbench"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "check"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sim-python", type=Path)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    def cli(*argv):
        result = subprocess.run(
            [sys.executable, str(args.cli), *map(str, argv)],
            text=True,
            capture_output=True,
            timeout=120,
        )
        if result.returncode not in {0, 1}:
            raise RuntimeError(result.stdout + result.stderr)
        return json.loads(result.stdout)

    if args.action == "prepare":
        if not args.source or not args.sim_python:
            parser.error("prepare requires --source and --sim-python")
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        release = args.source / "benchmark-vabench-release-v4/release/benchmarkv4-r53/tasks"
        expectations = {}
        for form, (task_id, slug) in TASKS.items():
            pin = root / f"{form}.pin.json"
            cli(
                "pin-vabench",
                "--source",
                args.source,
                "--python",
                args.sim_python,
                "--task-id",
                task_id,
                "--output",
                pin,
            )
            for kind in ("gold", "negative"):
                name = f"{form}-{kind}"
                candidate = root / f"input-{name}"
                if form == "testbench":
                    candidate.mkdir()
                    shutil.copy2(
                        release / slug / "evaluator/reference_tb.scs", candidate / "testbench.scs"
                    )
                    if kind == "negative":
                        (candidate / "testbench.scs").write_text(
                            "this is not a valid Spectre testbench\n"
                        )
                    expected = "passed" if kind == "gold" else "compile_failure"
                else:
                    fixture = (
                        release / slug / "evaluator/solution"
                        if kind == "gold"
                        else release / TASKS["bugfix"][1] / "public/buggy_bundle"
                    )
                    shutil.copytree(fixture, candidate)
                    expected = "passed" if kind == "gold" else "behavior_failure"
                expectations[name] = expected
                state = cli(
                    "submit-vabench",
                    "--pin",
                    pin,
                    "--submission",
                    candidate,
                    "--root",
                    root / "jobs",
                    "--job-id",
                    name,
                    "--timeout",
                    "300",
                )
                print(json.dumps({"job": name, "state": state["state"]}), flush=True)
        (root / "expectations.json").write_text(json.dumps(expectations, indent=2))
        print(json.dumps({"prepared": str(root)}), flush=True)
        return 0
    expected = json.loads((root / "expectations.json").read_text())
    passed = True
    for name, verdict in expected.items():
        state = cli("job-status", root / "jobs" / name)
        if state["state"] != "finished":
            print(json.dumps({"job": name, "state": state["state"]}))
            passed = False
            continue
        result = cli("verify-job", root / "jobs" / name)
        matches = result.get("benchmark_status") == verdict
        passed &= matches
        print(json.dumps({"job": name, "expected": verdict, "result": result, "matches": matches}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
