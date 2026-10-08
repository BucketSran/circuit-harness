"""Tests for the workflow trajectory visualizer."""

from __future__ import annotations

import json
from pathlib import Path

from alphaapollo.workflows.visualize import main


def test_visualize_cli_renders_a_workflow_run(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    cell_dir = run_dir / "vanilla" / "p0000" / "sample-000"
    canonical_dir = cell_dir / "canonical"
    canonical_dir.mkdir(parents=True)

    trajectory = {
        "task_id": "p0000:branch-0:solve:iteration-1",
        "turns": [],
        "termination_reason": "completed",
    }
    (canonical_dir / "trajectories.jsonl").write_text(
        json.dumps(trajectory) + "\n",
        encoding="utf-8",
    )
    (cell_dir / "result.json").write_text(
        json.dumps({"correct": True, "final_answer": "42"}),
        encoding="utf-8",
    )

    output = tmp_path / "viewer.html"

    assert main([str(run_dir), "-o", str(output)]) == 0

    rendered = output.read_text(encoding="utf-8")
    assert "AlphaApollo &mdash; Complete Interaction Trajectories" in rendered
    assert "p0000" in rendered
    assert "solve" in rendered
    assert "42" in rendered
    assert "No trajectories on disk" not in rendered
    assert "wrote" in capsys.readouterr().out


def test_visualize_does_not_invent_diagnostics_for_missing_evidence(tmp_path: Path) -> None:
    """A constructed incomplete run must not inherit another experiment's findings."""
    cell = tmp_path / "run" / "plain" / "circuit" / "sample-000"
    canonical = cell / "canonical"
    canonical.mkdir(parents=True)
    (canonical / "trajectories.jsonl").write_text(
        json.dumps({"task_id": "solve", "turns": [{"index": 0}]}) + "\n"
    )
    (cell / "result.json").write_text(json.dumps({"verdict": "inconclusive"}))
    output = tmp_path / "report.html"

    assert main([str(tmp_path / "run"), "-o", str(output)]) == 0
    rendered = output.read_text()
    assert "p0002" not in rendered
    assert "silently discarded" not in rendered
    assert "model_visible=False" not in rendered
    assert "tools offered:</span> <code>unavailable" in rendered
    assert "calls:</span> <code>unavailable" in rendered
    assert "Recorded outcomes" in rendered
