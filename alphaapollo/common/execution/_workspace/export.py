# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Host export bundles and retention for completed workspaces."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import uuid
from dataclasses import dataclass
from typing import Any

from alphaapollo.common.execution._workspace.snapshot import WorkspaceSnapshotter

logger = logging.getLogger(__name__)

#: Env var overriding the host runs root.
RUNS_DIR_ENV = "ALPHAAPOLLO_RUNS_DIR"
#: Env var overriding the retention cap (int; empty/unset -> default).
RUNS_MAX_ENV = "ALPHAAPOLLO_RUNS_MAX"
_DEFAULT_MAX_RUNS = 50
_DEFAULT_CONTAINER_WORKSPACE = "/workspace"


# Host exports ---------------------------------------------------------------
class WorkspaceExportError(Exception):
    """Raised when a workspace export fails (fail-loud, never a partial export)."""


@dataclass(frozen=True)
class ExportResult:
    """The outcome of one export.

    The full debug bundle for one solve: the copied workspace, the trajectory log
    (if a store was given), the result-package-referenced artifacts (if a store +
    package were given), and the metadata file. ``traj_path`` / ``artifact_hashes``
    are empty when those inputs were not supplied.
    """

    path: pathlib.Path  # the run dir: runs/<session>/<problem>/<branch>
    workspace_path: pathlib.Path  # the copied-out workspace tree
    snapshot_ref: str  # deterministic sha256 tar ref of the exported tree
    export_json_path: pathlib.Path  # the written metadata file
    traj_path: pathlib.Path | None = None  # copied trajectory jsonl, if any
    artifact_hashes: tuple[str, ...] = ()  # content hashes of copied artifacts


def _default_runs_root() -> pathlib.Path:
    """Resolve the runs root: env override, else ``<cwd>/alphaapollo/eval/runs``."""
    env = os.environ.get(RUNS_DIR_ENV)
    if env:
        return pathlib.Path(env)
    return pathlib.Path.cwd() / "alphaapollo" / "eval" / "runs"


def _default_max_runs() -> int | None:
    """Resolve the retention cap from the env, else the module default."""
    raw = os.environ.get(RUNS_MAX_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_RUNS
    try:
        value = int(raw)
    except ValueError:
        raise WorkspaceExportError(f"{RUNS_MAX_ENV} must be an integer, got {raw!r}") from None
    return None if value <= 0 else value


def _safe_segment(name: str, *, field: str) -> str:
    """Validate an id used as a single path segment (no traversal / separators)."""
    if not name or not isinstance(name, str) or not name.strip():
        raise WorkspaceExportError(f"{field} must be a non-empty string")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise WorkspaceExportError(f"{field} must be a single path segment, got {name!r}")
    return name


class WorkspaceExporter:
    """Copy a finished sandbox workspace to the host runs directory and record it.

    ``backend`` need only expose ``copy_out(container_path, host_dest) -> None``
    (``PodmanBackend`` / ``DockerBackend`` do). ``snapshotter`` (optional) supplies
    the deterministic tar ref; a default in-memory one is used when omitted.
    """

    def __init__(
        self,
        *,
        runs_root: str | pathlib.Path | None = None,
        max_runs: int | None = -1,
        snapshotter: Any | None = None,
    ) -> None:
        self._runs_root = pathlib.Path(runs_root) if runs_root is not None else _default_runs_root()
        # Sentinel -1 == "not passed" -> fall back to the env/default; an explicit
        # None means "keep everything", an explicit int is the cap.
        self._max_runs = _default_max_runs() if max_runs == -1 else max_runs
        if self._max_runs is not None and (
            isinstance(self._max_runs, bool)
            or not isinstance(self._max_runs, int)
            or self._max_runs <= 0
        ):
            raise WorkspaceExportError("max_runs must be a positive int or None")
        self._snapshotter = snapshotter or WorkspaceSnapshotter()
        self._seq = 0

    @property
    def runs_root(self) -> pathlib.Path:
        """The host directory under which run exports are written."""
        return self._runs_root

    def export(
        self,
        backend: Any,
        *,
        session_id: str,
        problem_id: str,
        branch_id: str,
        container_workspace: str = _DEFAULT_CONTAINER_WORKSPACE,
        metadata: dict[str, Any] | None = None,
        trajectory_store: Any | None = None,
        artifact_store: Any | None = None,
        solution_package: Any | None = None,
    ) -> ExportResult:
        """Export the full debug bundle for one solve into the runs dir.

        Layout::

            runs/<session>/<problem>/<branch>/
                workspace/        # copied out of the container (always)
                traj.jsonl        # trajectory log       (if trajectory_store given)
                artifacts/<hash>  # result-package-referenced evidence
                                  #                      (if artifact_store+package given)
                export.json       # metadata + result summary

        Only ``backend`` + the ids are required; ``trajectory_store`` /
        ``artifact_store`` / ``solution_package`` are optional — each present input
        adds its slice to the bundle, each absent one is skipped. The workspace copy
        fails loud (no partial export left behind); trajectory/artifact copies that
        fail are recorded in ``export.json`` under ``export_warnings`` rather than
        aborting the whole bundle (a missing artifact must not lose the trajectory).
        """
        session = _safe_segment(session_id, field="session_id")
        problem = _safe_segment(problem_id, field="problem_id")
        branch = _safe_segment(branch_id, field="branch_id")

        run_dir = self._runs_root / session / problem / branch
        parent_dir = run_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        # Build the complete bundle beside the final path.  A failed copy,
        # snapshot, or metadata write therefore leaves the previous valid export
        # untouched instead of deleting it first.
        staging_dir = parent_dir / f".{branch}.exporting-{uuid.uuid4().hex}"
        staging_dir.mkdir()
        workspace_dir = staging_dir / "workspace"
        try:
            # (1) Persistence — copy the tree OUT of the container.
            try:
                backend.copy_out(container_workspace, str(workspace_dir))
            except Exception as exc:
                raise WorkspaceExportError(
                    f"copy_out({container_workspace!r}) failed for "
                    f"{session}/{problem}/{branch}: {exc}"
                ) from exc
            if not workspace_dir.is_dir():
                raise WorkspaceExportError(
                    f"copy_out produced no workspace directory at {workspace_dir}"
                )

            # (4) Snapshot format — deterministic tar ref over the exported tree.
            snapshot_ref = self._snapshotter.snapshot(workspace_dir)
            warnings: list[str] = []

            # (2) Trajectory — copy the append-only jsonl log (replay/debug core).
            traj_path: pathlib.Path | None = None
            if trajectory_store is not None:
                traj_path = self._copy_trajectory(trajectory_store, staging_dir, warnings)

            # (3) Artifacts — copy only the result-package-referenced content-
            # addressed evidence, never the whole store.
            artifact_hashes: tuple[str, ...] = ()
            package_summary: dict[str, Any] | None = None
            if solution_package is not None:
                package_summary = self._summarize_package(solution_package)
            if artifact_store is not None and solution_package is not None:
                artifact_hashes = self._copy_artifacts(
                    artifact_store, solution_package, staging_dir, warnings
                )

            # (3) Layout — record metadata alongside the copied tree.
            self._seq += 1
            export_payload = {
                "session_id": session,
                "problem_id": problem,
                "branch_id": branch,
                "container_workspace": container_workspace,
                "snapshot_ref": snapshot_ref,
                "exported_at": self._seq,
                "trajectory": traj_path.name if traj_path else None,
                "artifact_hashes": list(artifact_hashes),
                "solution_package": package_summary,
                "export_warnings": warnings,
                "metadata": metadata or {},
            }
            export_json = staging_dir / "export.json"
            export_json.write_text(
                json.dumps(export_payload, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )

            self._replace_run_dir(staging_dir, run_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        workspace_dir = run_dir / "workspace"
        export_json = run_dir / "export.json"
        traj_path = run_dir / traj_path.name if traj_path is not None else None
        logger.debug("exported workspace -> %s (snapshot=%s)", run_dir, snapshot_ref)

        # (5/6) Retention — evict oldest run dirs beyond the cap (best-effort).
        self._enforce_retention()

        return ExportResult(
            path=run_dir,
            workspace_path=workspace_dir,
            snapshot_ref=snapshot_ref,
            export_json_path=export_json,
            traj_path=traj_path,
            artifact_hashes=artifact_hashes,
        )

    @staticmethod
    def _replace_run_dir(staging_dir: pathlib.Path, run_dir: pathlib.Path) -> None:
        """Atomically publish a complete staging bundle, preserving old data on error."""
        backup_dir = run_dir.parent / f".{run_dir.name}.previous-{uuid.uuid4().hex}"
        had_old = run_dir.exists() or run_dir.is_symlink()
        try:
            if had_old:
                os.replace(run_dir, backup_dir)
            os.replace(staging_dir, run_dir)
        except Exception:
            if not run_dir.exists() and backup_dir.exists():
                os.replace(backup_dir, run_dir)
            raise
        finally:
            if backup_dir.exists() or backup_dir.is_symlink():
                if backup_dir.is_dir() and not backup_dir.is_symlink():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                else:
                    backup_dir.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # debug-bundle slices (trajectory + artifacts + package summary)
    # ------------------------------------------------------------------
    @staticmethod
    def _copy_trajectory(
        trajectory_store: Any, run_dir: pathlib.Path, warnings: list[str]
    ) -> pathlib.Path | None:
        """Copy the trajectory jsonl into the run dir; warn (not fail) on error.

        Prefer the store's live ``jsonl_path`` when available. Durable trajectory
        references may contain a location relative to their owning store, so treating
        that value as process-cwd-relative can silently omit the log. A missing or
        unreadable log is recorded as a warning so the rest of the bundle still lands.
        """
        try:
            location = getattr(trajectory_store, "jsonl_path", None)
            if location is None:
                location = trajectory_store.trajectory_ref().location
            if location is None:
                warnings.append("trajectory log location is unavailable")
                return None
            src = pathlib.Path(location)
            if not src.is_file():
                warnings.append(f"trajectory log not found at {src}")
                return None
            dest = run_dir / "traj.jsonl"
            shutil.copyfile(src, dest)
            return dest
        except Exception as exc:  # noqa: BLE001 — best-effort slice
            warnings.append(f"trajectory export failed: {exc}")
            return None

    @staticmethod
    def _package_artifact_refs(solution_package: Any) -> list[Any]:
        """Collect every ArtifactRef reachable from a result package (deduped)."""
        refs: list[Any] = []
        seen: set[str] = set()

        def _add(ref: Any) -> None:
            h = getattr(ref, "hash", None)
            if h and h not in seen:
                seen.add(h)
                refs.append(ref)

        # Single-ref fields + list fields on a duck-typed result package.
        for attr in ("code", "formal_proof", "experiment"):
            ref = getattr(solution_package, attr, None)
            if ref is not None:
                _add(ref)
        for attr in ("artifacts", "executable_evidence"):
            for ref in getattr(solution_package, attr, None) or ():
                _add(ref)
        return refs

    def _copy_artifacts(
        self,
        artifact_store: Any,
        solution_package: Any,
        run_dir: pathlib.Path,
        warnings: list[str],
    ) -> tuple[str, ...]:
        """Copy result-package artifacts into ``run_dir/artifacts/<hash>``.

        Only the package-referenced content-addressed evidence is copied (never the
        whole store). A ref that cannot be fetched is warned, not fatal.
        """
        refs = self._package_artifact_refs(solution_package)
        if not refs:
            return ()
        art_dir = run_dir / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for ref in refs:
            digest = getattr(ref, "hash", None)
            if not digest:
                continue
            try:
                content = artifact_store.get(ref)
                (art_dir / digest).write_bytes(content)
                copied.append(digest)
            except Exception as exc:  # noqa: BLE001 — best-effort per-artifact
                warnings.append(f"artifact {digest[:12]}… export failed: {exc}")
        return tuple(copied)

    @staticmethod
    def _summarize_package(solution_package: Any) -> dict[str, Any]:
        """Machine-readable result summary: what the solve concluded.

        Pulls the debug-relevant scalar fields off the result package; missing
        fields degrade to None so this works on partial/duck-typed packages too.
        """

        def _g(name: str) -> Any:
            return getattr(solution_package, name, None)

        verifier_results = _g("verifier_results") or []
        return {
            "final_answer": _g("final_answer"),
            "trust_level": _g("trust_level"),
            "confidence": _g("confidence"),
            "reasoning_summary": _g("reasoning_summary"),
            "verifier_count": len(verifier_results),
        }

    def _enforce_retention(self) -> None:
        """Keep at most ``max_runs`` leaf run dirs; evict oldest by mtime.

        A "run" is a ``<session>/<problem>/<branch>`` leaf (one export). Eviction is
        best-effort — a removal error is logged, never raised (never breaks a solve).
        """
        if self._max_runs is None:
            return
        runs = self._all_run_dirs()
        excess = len(runs) - self._max_runs
        if excess <= 0:
            return
        # Oldest first. Primary key is fs mtime (correct across time / restarts);
        # ``exported_at`` from export.json breaks same-mtime-tick ties so rapid
        # exports still evict in creation order.
        runs.sort(key=self._recency_key)
        for stale in runs[:excess]:
            try:
                shutil.rmtree(stale, ignore_errors=True)
                logger.debug("retention: evicted %s", stale)
            except OSError as exc:  # pragma: no cover — best-effort
                logger.warning("retention: failed to evict %s: %s", stale, exc)

    @staticmethod
    def _recency_key(run_dir: pathlib.Path) -> tuple[float, int]:
        """(mtime, exported_at) recency key; missing/unreadable export.json -> 0."""
        try:
            mtime = run_dir.stat().st_mtime
        except OSError:  # pragma: no cover — defensive
            mtime = 0.0
        seq = 0
        export_json = run_dir / "export.json"
        try:
            seq = int(json.loads(export_json.read_text(encoding="utf-8")).get("exported_at", 0))
        except (OSError, ValueError, TypeError):
            seq = 0
        return (mtime, seq)

    def _all_run_dirs(self) -> list[pathlib.Path]:
        """Return every leaf run dir: ``runs/<session>/<problem>/<branch>``."""
        if not self._runs_root.is_dir():
            return []
        leaves: list[pathlib.Path] = []
        for session in self._runs_root.iterdir():
            if not session.is_dir():
                continue
            for problem in session.iterdir():
                if not problem.is_dir():
                    continue
                for branch in problem.iterdir():
                    if branch.is_dir():
                        leaves.append(branch)
        return leaves
