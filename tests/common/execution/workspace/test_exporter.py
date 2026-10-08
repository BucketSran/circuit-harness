"""GPU/Podman-free tests for WorkspaceExporter (workspace -> host persistence).

A FAKE backend implements ``copy_out`` by writing a canned tree into the host dest,
so these tests exercise the export mechanism (layout, snapshot ref, metadata,
retention, fail-loud) with no Podman/Docker present.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from alphaapollo.common.execution.workspace import (
    ExportResult,
    WorkspaceExporter,
    WorkspaceExportError,
)


class _FakeBackend:
    """Fake backend whose ``copy_out`` materializes a canned tree at host_dest."""

    def __init__(self, files: dict[str, str] | None = None, fail: bool = False) -> None:
        self._files = files if files is not None else {"solution.py": "print(1)\n"}
        self._fail = fail
        self.calls: list[tuple[str, str]] = []

    def copy_out(self, container_path: str, host_dest: str) -> None:
        self.calls.append((container_path, host_dest))
        if self._fail:
            raise RuntimeError("podman cp: no such file or directory")
        dest = pathlib.Path(host_dest)
        dest.mkdir(parents=True, exist_ok=True)
        for rel, content in self._files.items():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")


def test_export_creates_session_problem_branch_layout(tmp_path):
    exporter = WorkspaceExporter(runs_root=tmp_path)
    result = exporter.export(
        _FakeBackend(),
        session_id="sess1",
        problem_id="prob1",
        branch_id="branch0",
    )
    assert isinstance(result, ExportResult)
    assert result.path == tmp_path / "sess1" / "prob1" / "branch0"
    assert result.workspace_path.is_dir()
    assert (result.workspace_path / "solution.py").read_text() == "print(1)\n"
    assert result.export_json_path.exists()


def test_export_copies_default_container_workspace(tmp_path):
    backend = _FakeBackend()
    exporter = WorkspaceExporter(runs_root=tmp_path)
    exporter.export(backend, session_id="s", problem_id="p", branch_id="b")
    # default container path is /workspace, copied to the run's workspace/ dir
    assert backend.calls[0][0] == "/workspace"
    assert backend.calls[0][1].endswith("workspace")


def test_export_writes_metadata_and_snapshot_ref(tmp_path):
    exporter = WorkspaceExporter(runs_root=tmp_path)
    result = exporter.export(
        _FakeBackend(),
        session_id="s",
        problem_id="p",
        branch_id="b",
        metadata={"trust_level": 3, "final_answer": "391"},
    )
    payload = json.loads(result.export_json_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "s"
    assert payload["problem_id"] == "p"
    assert payload["branch_id"] == "b"
    assert payload["metadata"] == {"trust_level": 3, "final_answer": "391"}
    assert payload["snapshot_ref"] == result.snapshot_ref
    # sha256 hexdigest shape
    assert len(result.snapshot_ref) == 64


def test_snapshot_ref_is_deterministic_for_same_tree(tmp_path):
    files = {"a.py": "x = 1\n", "sub/b.txt": "hello"}
    r1 = WorkspaceExporter(runs_root=tmp_path / "r1").export(
        _FakeBackend(files), session_id="s", problem_id="p", branch_id="b"
    )
    r2 = WorkspaceExporter(runs_root=tmp_path / "r2").export(
        _FakeBackend(files), session_id="s", problem_id="p", branch_id="b"
    )
    assert r1.snapshot_ref == r2.snapshot_ref


def test_copy_out_failure_is_fail_loud_and_leaves_no_partial(tmp_path):
    exporter = WorkspaceExporter(runs_root=tmp_path)
    with pytest.raises(WorkspaceExportError):
        exporter.export(
            _FakeBackend(fail=True),
            session_id="s",
            problem_id="p",
            branch_id="b",
        )
    # no workspace tree left behind
    assert not (tmp_path / "s" / "p" / "b" / "workspace").exists()


def test_failed_reexport_preserves_previous_valid_export(tmp_path):
    exporter = WorkspaceExporter(runs_root=tmp_path)
    exporter.export(
        _FakeBackend({"old.py": "1"}),
        session_id="s",
        problem_id="p",
        branch_id="b",
    )
    with pytest.raises(WorkspaceExportError):
        exporter.export(
            _FakeBackend(fail=True),
            session_id="s",
            problem_id="p",
            branch_id="b",
        )
    final_workspace = tmp_path / "s" / "p" / "b" / "workspace"
    assert (final_workspace / "old.py").read_text() == "1"
    assert (tmp_path / "s" / "p" / "b" / "export.json").exists()


def test_retention_cap_must_be_positive_or_none(tmp_path):
    with pytest.raises(WorkspaceExportError, match="max_runs"):
        WorkspaceExporter(runs_root=tmp_path, max_runs=0)


def test_reexport_replaces_stale_workspace(tmp_path):
    exporter = WorkspaceExporter(runs_root=tmp_path)
    exporter.export(
        _FakeBackend({"old.py": "1"}),
        session_id="s",
        problem_id="p",
        branch_id="b",
    )
    result = exporter.export(
        _FakeBackend({"new.py": "2"}),
        session_id="s",
        problem_id="p",
        branch_id="b",
    )
    assert (result.workspace_path / "new.py").exists()
    assert not (result.workspace_path / "old.py").exists()


def test_retention_evicts_oldest_beyond_cap(tmp_path):
    exporter = WorkspaceExporter(runs_root=tmp_path, max_runs=2)
    for i in range(4):
        exporter.export(
            _FakeBackend(),
            session_id="s",
            problem_id="p",
            branch_id=f"b{i}",
        )
    live = sorted(p.name for p in (tmp_path / "s" / "p").iterdir())
    # only the 2 most recent branch dirs survive
    assert len(live) == 2
    assert "b2" in live and "b3" in live


def test_retention_none_keeps_everything(tmp_path):
    exporter = WorkspaceExporter(runs_root=tmp_path, max_runs=None)
    for i in range(5):
        exporter.export(
            _FakeBackend(),
            session_id="s",
            problem_id="p",
            branch_id=f"b{i}",
        )
    live = list((tmp_path / "s" / "p").iterdir())
    assert len(live) == 5


def test_runs_dir_env_override(tmp_path, monkeypatch):
    from alphaapollo.common.execution.workspace import RUNS_DIR_ENV

    monkeypatch.setenv(RUNS_DIR_ENV, str(tmp_path / "from_env"))
    exporter = WorkspaceExporter()
    assert exporter.runs_root == tmp_path / "from_env"


@pytest.mark.parametrize("bad", ["", "a/b", "..", "."])
def test_rejects_path_traversal_ids(tmp_path, bad):
    exporter = WorkspaceExporter(runs_root=tmp_path)
    with pytest.raises(WorkspaceExportError):
        exporter.export(_FakeBackend(), session_id=bad, problem_id="p", branch_id="b")


# --- full debug bundle: trajectory + artifacts + package summary -------------
class _FakeRef:
    def __init__(self, h):
        self.hash = h


class _FakeTrajRef:
    def __init__(self, location):
        self.location = location


class _FakeTrajStore:
    def __init__(self, jsonl_path):
        self._path = jsonl_path

    @property
    def jsonl_path(self):
        return self._path

    def trajectory_ref(self):
        return _FakeTrajRef(self._path.name)


class _FakeArtifactStore:
    def __init__(self, blobs):
        self._blobs = blobs  # {hash: bytes}

    def get(self, ref):
        return self._blobs[ref.hash]


class _FakePackage:
    def __init__(self, artifacts=(), code=None, **scalars):
        self.artifacts = list(artifacts)
        self.executable_evidence = []
        self.code = code
        self.formal_proof = None
        self.experiment = None
        self.verifier_results = scalars.pop("verifier_results", [])
        self.final_answer = scalars.get("final_answer", "391")
        self.trust_level = scalars.get("trust_level", 2)
        self.confidence = scalars.get("confidence", 0.8)
        self.reasoning_summary = scalars.get("reasoning_summary", "computed via tool")


def test_export_copies_trajectory(tmp_path):
    traj = tmp_path / "src_traj.jsonl"
    traj.write_text('{"type":"PROPOSE"}\n{"type":"TOOL_CALLED"}\n', encoding="utf-8")
    exporter = WorkspaceExporter(runs_root=tmp_path / "runs")
    result = exporter.export(
        _FakeBackend(),
        session_id="s",
        problem_id="p",
        branch_id="b",
        trajectory_store=_FakeTrajStore(traj),
    )
    assert result.traj_path is not None
    assert result.traj_path.read_text() == traj.read_text()
    payload = json.loads(result.export_json_path.read_text())
    assert payload["trajectory"] == "traj.jsonl"


def test_export_copies_only_package_referenced_artifacts(tmp_path):
    blobs = {"a" * 64: b"report-A", "b" * 64: b"witness-B", "c" * 64: b"unused-C"}
    pkg = _FakePackage(artifacts=[_FakeRef("a" * 64)], code=_FakeRef("b" * 64))
    exporter = WorkspaceExporter(runs_root=tmp_path / "runs")
    result = exporter.export(
        _FakeBackend(),
        session_id="s",
        problem_id="p",
        branch_id="b",
        artifact_store=_FakeArtifactStore(blobs),
        solution_package=pkg,
    )
    # only the 2 referenced artifacts copied, not the unused third
    assert set(result.artifact_hashes) == {"a" * 64, "b" * 64}
    art_dir = result.path / "artifacts"
    assert (art_dir / ("a" * 64)).read_bytes() == b"report-A"
    assert (art_dir / ("b" * 64)).read_bytes() == b"witness-B"
    assert not (art_dir / ("c" * 64)).exists()


def test_export_writes_package_summary(tmp_path):
    pkg = _FakePackage(final_answer="391", trust_level=3, confidence=0.9)
    exporter = WorkspaceExporter(runs_root=tmp_path / "runs")
    result = exporter.export(
        _FakeBackend(),
        session_id="s",
        problem_id="p",
        branch_id="b",
        solution_package=pkg,
    )
    summary = json.loads(result.export_json_path.read_text())["solution_package"]
    assert summary["final_answer"] == "391"
    assert summary["trust_level"] == 3
    assert summary["confidence"] == 0.9


def test_missing_trajectory_is_warning_not_fatal(tmp_path):
    exporter = WorkspaceExporter(runs_root=tmp_path / "runs")
    result = exporter.export(
        _FakeBackend(),
        session_id="s",
        problem_id="p",
        branch_id="b",
        trajectory_store=_FakeTrajStore(tmp_path / "does_not_exist.jsonl"),
    )
    # workspace still exported; the missing traj is a recorded warning
    assert result.workspace_path.is_dir()
    warnings = json.loads(result.export_json_path.read_text())["export_warnings"]
    assert any("trajectory" in w for w in warnings)


def test_full_bundle_all_three_slices(tmp_path):
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"type":"VERIFY"}\n', encoding="utf-8")
    blobs = {"d" * 64: b"evidence"}
    pkg = _FakePackage(artifacts=[_FakeRef("d" * 64)], trust_level=4)
    exporter = WorkspaceExporter(runs_root=tmp_path / "runs")
    result = exporter.export(
        _FakeBackend({"solution.py": "print(391)\n"}),
        session_id="s",
        problem_id="p",
        branch_id="b",
        trajectory_store=_FakeTrajStore(traj),
        artifact_store=_FakeArtifactStore(blobs),
        solution_package=pkg,
    )
    # all four pieces present
    assert (result.workspace_path / "solution.py").exists()
    assert result.traj_path.exists()
    assert (result.path / "artifacts" / ("d" * 64)).read_bytes() == b"evidence"
    payload = json.loads(result.export_json_path.read_text())
    assert payload["solution_package"]["trust_level"] == 4
    assert payload["trajectory"] == "traj.jsonl"
    assert payload["artifact_hashes"] == ["d" * 64]
