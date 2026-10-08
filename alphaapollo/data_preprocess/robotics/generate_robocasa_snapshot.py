"""Deterministic RoboCasa365 task-suite snapshot generator.

The snapshot is an AlphaApollo-derived task-suite manifest, not the official
RoboCasa demonstration datasets (LeRobot trajectory data for VLA training;
out of scope for this slice): every field is derived from the official
runtime and metadata of one pinned checkout. The scene triplet and the
instruction are recorded from a scene built through the official
``robocasa.utils.env_utils.create_env`` API (``ep_meta["lang"]`` embeds the
sampled object name), and the per-task horizon comes from the official
``dataset_registry`` where the task has an entry (the env constructor's
default otherwise — those tasks are named in the sidecar meta).

Protocol: for every registered kitchen task the generator derives a
task-stable seed from the generator seed (42, matching the ENPIRE
matched-evaluation protocol), samples the ``(layout_id, style_id)`` pair
from a plain seeded build, then replays the scene through the backend's
exact factory call (pinned ``layout_and_style_ids``, no RNG overrides) and
records THAT build's instruction — validating by construction. ``--check``
re-derives every committed row the same way and refuses on any mismatch.

Provenance (protocol, seed, upstream pins, horizon sourcing) lives in the
``.meta.json`` sidecar beside the snapshot, recording the seed this run was
given and the upstream checkouts it actually imported; the pinned checkout
and the asset story are in ``UPSTREAM.md`` beside this module. The
generator runs only inside an environment with the robocasa runtime and is
deliberately NOT imported by ``prepare_robocasa``, which only reads the
committed snapshot.

The committed snapshot is the default output, so only a full sweep on the
default seed may write it: ``--limit`` or a non-default ``--seed`` produce a
different suite and require an explicit ``--output``.

Usage:
    python -m alphaapollo.data_preprocess.robotics.generate_robocasa_snapshot
    python -m alphaapollo.data_preprocess.robotics.generate_robocasa_snapshot \
        [--seed N] [--limit N] --output <jsonl>
    python -m alphaapollo.data_preprocess.robotics.generate_robocasa_snapshot \
        --check [--snapshot <jsonl>]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from alphaapollo.data_preprocess.robotics.prepare_robocasa import SNAPSHOT_PATH

GENERATOR_SEED = 42
# One owner for the location: the preparer's path is authoritative, so a
# regeneration can never write a stray copy one directory away from the
# committed snapshot (review finding on #256).
DEFAULT_OUTPUT = SNAPSHOT_PATH

# Registry entries that cannot construct at all: they are non-instantiable
# base classes, not tasks. Anything ELSE failing to build refuses the whole
# run — a partially-truncated suite must never look like a success.
KNOWN_UNBUILDABLE = frozenset(
    {"ManipulateLowerDoor", "OpenDropDownDoor", "CloseDropDownDoor", "Kitchen"}
)
# The upstream checkouts the sidecar names. Resolved per run, never pinned in
# source: a constant here would keep claiming one checkout while the generator
# imported whatever is on sys.path (#256 review).
UPSTREAM_PACKAGES = ("robocasa", "robosuite")
UNRESOLVED_PIN = "unresolved"


def _task_seed(generator_seed: int, task_name: str) -> int:
    digest = hashlib.sha256(f"{generator_seed}\x00{task_name}".encode()).hexdigest()
    return int(digest[:8], 16) % 2**31


def _canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def _package_version(module_name: str) -> str | None:
    try:
        return importlib.metadata.version(module_name)
    except Exception:
        pass
    try:
        return str(importlib.import_module(module_name).__version__)
    except Exception:
        return None


def _package_commit(module_name: str) -> str | None:
    """The git sha of the installed package root, for a source checkout."""

    try:
        module = importlib.import_module(module_name)
        root = Path(module.__file__).resolve().parent
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _resolved_pin(module_name: str) -> str:
    """Name the checkout this run actually imported.

    RoboCasa365 and the robosuite master it requires are source installs
    (neither is on PyPI in the form the generator needs), so the declared
    version does not move between commits and the git sha of the package
    root is the identifying half. Either half may be unavailable; the
    sidecar then records ``"unresolved"`` rather than a stale guess, so a
    reader can tell "not recorded" from "recorded as this".
    """

    parts = [part for part in (_package_version(module_name), _package_commit(module_name)) if part]
    return "+".join(parts) if parts else UNRESOLVED_PIN


def _resolved_upstream_pins() -> dict[str, str]:
    """Resolve every upstream pin the sidecar records."""

    return {name: _resolved_pin(name) for name in UPSTREAM_PACKAGES}


def _official_horizon(task_name: str) -> int | None:
    """The official per-task horizon, or None when the task has no entry.

    The dataset registry ships with the runtime (no dataset download), so
    it is pinned by the same commit as the scene sampling; the env
    constructor's horizon is a task-agnostic default, not metadata.
    """

    from robocasa.utils.dataset_registry import (
        ATOMIC_TASK_DATASETS,
        COMPOSITE_TASK_DATASETS,
    )

    for registry in (ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS):
        if task_name in registry:
            return int(registry[task_name]["horizon"])
    return None


def _replay_row(task_name: str, layout_id: int, style_id: int, scene_seed: int) -> dict[str, Any]:
    """Rebuild one scene through the backend's exact factory call.

    The instruction is read from THIS build only — it is the one the
    backend will reproduce. Pinning the pair also changes constructor RNG
    consumption, so only this replay can validate (see #256: the two ice
    cubes).
    """

    from robocasa.utils.env_utils import create_env

    env = create_env(
        env_name=task_name,
        layout_and_style_ids=[(layout_id, style_id)],
        seed=scene_seed,
        render_onscreen=False,
    )
    try:
        env.reset()
        ep_meta = env.get_ep_meta()
        instruction = str(ep_meta.get("lang", "")).strip()
        if not instruction:
            raise ValueError(f"{task_name} produced an empty language instruction")
        horizon = _official_horizon(task_name)
        if horizon is None:
            horizon = int(getattr(env, "horizon", 0))
        return {
            "id": f"{task_name}-000",
            "task_name": task_name,
            "layout_id": int(env.layout_id),
            "style_id": int(env.style_id),
            "scene_seed": int(scene_seed),
            "instruction": instruction,
            "horizon": horizon,
        }
    finally:
        env.close()


def _build_row(task_name: str, scene_seed: int) -> dict[str, Any]:
    from robocasa.utils.env_utils import create_env

    # Phase one: sample the scene pair from a plain seeded build.
    sampler = create_env(env_name=task_name, seed=scene_seed, render_onscreen=False)
    try:
        sampler.reset()
        layout_id, style_id = int(sampler.layout_id), int(sampler.style_id)
    finally:
        sampler.close()
    # Phase two: the backend's exact factory call.
    return _replay_row(task_name, layout_id, style_id, scene_seed)


def generate_rows(*, seed: int = GENERATOR_SEED, limit: int | None = None) -> list[dict[str, Any]]:
    """Build every registered kitchen task once and collect its scene triplet."""

    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

    task_names = sorted(ALL_KITCHEN_ENVIRONMENTS)
    if limit is not None:
        task_names = task_names[:limit]
    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for index, task_name in enumerate(task_names, start=1):
        scene_seed = _task_seed(seed, task_name)
        print(f"[{index}/{len(task_names)}] {task_name} (scene_seed={scene_seed})", flush=True)
        try:
            rows.append(_build_row(task_name, scene_seed))
        except Exception as exc:  # noqa: BLE001 — one broken scene must not void the suite
            failures.append((task_name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED: {failures[-1][1][:160]}", file=sys.stderr, flush=True)
    unexpected = [item for item in failures if item[0] not in KNOWN_UNBUILDABLE]
    if unexpected:
        for task_name, reason in unexpected:
            print(f"UNEXPECTED FAILURE {task_name}: {reason[:200]}", file=sys.stderr)
        raise RuntimeError(
            f"{len(unexpected)} task(s) failed outside the known-unbuildable set "
            f"{sorted(KNOWN_UNBUILDABLE)}; refusing to write a truncated suite"
        )
    print(
        f"built {len(rows)}/{len(task_names)} tasks; "
        f"{len(failures)} known-unbuildable skipped: {sorted(name for name, _ in failures)}",
        file=sys.stderr,
    )
    if not rows:
        raise RuntimeError("every task failed to build; see stderr")
    return rows


def _write_meta(
    meta_path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    upstream_pins: Mapping[str, str],
    env_default_tasks: Sequence[str],
) -> None:
    """Record the run's own provenance beside the snapshot it describes.

    ``seed`` and ``upstream_pins`` come from the caller because they describe
    THIS run. Recording the module defaults instead made the sidecar's own
    documented seed rule fail to reproduce the rows it sits beside whenever
    ``--seed`` was passed (#256 review).
    """

    meta = {
        "artifact": (
            "AlphaApollo-derived task-suite manifest for RoboCasa365 "
            "(ENPIRE matched-evaluation protocol); not the official RoboCasa "
            "demonstration datasets"
        ),
        "protocol": {
            "generator_seed": seed,
            "task_seed_rule": (
                'int(sha256(f"{generator_seed}\\x00{task_name}").hexdigest()[:8], 16) % 2**31'
            ),
        },
        "upstream_pins": dict(upstream_pins),
        "procedure": (
            "sample (layout_id, style_id) from a plain seeded create_env build, "
            "then rebuild via create_env(layout_and_style_ids=[(l, s)], seed) — "
            "the backend's exact factory call — and record that build's "
            "ep_meta['lang']"
        ),
        "horizon_source": (
            "official robocasa dataset_registry of the pinned checkout where the "
            "task has an entry; the env constructor default for the tasks listed "
            "in horizon_env_default"
        ),
        "horizon_env_default": sorted(env_default_tasks),
        "row_count": len(rows),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check(snapshot: Path) -> int:
    """Re-derive every committed row; refuse on any field mismatch."""

    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

    lines = [line for line in snapshot.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    if not rows:
        print(f"snapshot is empty: {snapshot}", file=sys.stderr)
        return 1
    expected_tasks = sorted(set(ALL_KITCHEN_ENVIRONMENTS) - KNOWN_UNBUILDABLE)
    present_tasks = sorted(row["task_name"] for row in rows)
    if present_tasks != expected_tasks:
        present = set(present_tasks)
        print(
            f"task coverage mismatch: missing={sorted(set(expected_tasks) - present)} "
            f"unexpected={sorted(present - set(expected_tasks))}",
            file=sys.stderr,
        )
        return 1
    mismatches: list[str] = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['task_name']}", flush=True)
        fresh = _replay_row(row["task_name"], row["layout_id"], row["style_id"], row["scene_seed"])
        differing = sorted(key for key in fresh if fresh[key] != row.get(key))
        if differing:
            mismatches.append(f"{row['task_name']}: {differing}")
            print(f"  MISMATCH {mismatches[-1]}", file=sys.stderr, flush=True)
    if mismatches:
        print(f"{len(mismatches)}/{len(rows)} rows failed reconstruction", file=sys.stderr)
        return 1
    print(
        f"all {len(rows)} rows reconstruct identically through the backend's "
        "factory call; task coverage matches the pinned registry",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=GENERATOR_SEED)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke: first N tasks only; requires --output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"where to write the suite; defaults to the committed {DEFAULT_OUTPUT.name}, "
        "which only a full default-seed sweep may write",
    )
    parser.add_argument(
        "--check", action="store_true", help="re-derive committed rows and refuse on mismatch"
    )
    parser.add_argument(
        "--snapshot", type=Path, default=DEFAULT_OUTPUT, help="snapshot to check (with --check)"
    )
    args = parser.parse_args(argv)

    if args.check:
        return _check(args.snapshot)

    if args.output == DEFAULT_OUTPUT and (args.limit is not None or args.seed != GENERATOR_SEED):
        # A smoke run used to replace the vendored 370-row suite in the working
        # tree and rewrite its sidecar with truncated counts, unprompted (#256
        # review). Only a full sweep on the protocol seed describes that file.
        parser.error(
            f"--limit or a non-default --seed builds a suite that is not the committed "
            f"{DEFAULT_OUTPUT.name} (a full sweep on seed {GENERATOR_SEED}); pass "
            "--output to write it elsewhere"
        )

    rows = generate_rows(seed=args.seed, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{_canonical_row(row)}\n" for row in rows), encoding="utf-8")
    env_default = [row["task_name"] for row in rows if _official_horizon(row["task_name"]) is None]
    _write_meta(
        args.output.with_suffix(".meta.json"),
        rows,
        seed=args.seed,
        upstream_pins=_resolved_upstream_pins(),
        env_default_tasks=env_default,
    )
    print(f"wrote {len(rows)} tasks to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
