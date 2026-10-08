"""Operator final scoring requires an acknowledged, unchanged frozen candidate."""

import json

import pytest

from alphaapollo.common.execution.chips import analog_session
from alphaapollo.common.execution.chips.journal import file_digest


def test_finalizer_refuses_unfrozen_and_changed_candidate(tmp_path, monkeypatch):
    session = tmp_path / "session"
    (session / "frozen").mkdir(parents=True)
    candidate = session / "frozen/circuit.spi"
    candidate.write_text(
        ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    )
    (session / "session.json").write_text(json.dumps({"source_root": str(tmp_path / "source")}))
    called = []
    monkeypatch.setattr(analog_session, "run_case", lambda *a, **k: called.append((a, k)))
    with pytest.raises(ValueError, match="acknowledged submission"):
        analog_session.finalize_session(session, tmp_path / "score")
    (session / "frozen.json").write_text(
        json.dumps({"candidate_sha256": file_digest(candidate), "action_id": "submit"})
    )
    action = session / "actions/submit"
    action.mkdir(parents=True)
    (action / "request.json").write_text(json.dumps({"tool": "analog_submit"}))
    (action / "response.json").write_text(
        json.dumps(
            {
                "ok": True,
                "result": {"state": "submitted", "candidate_sha256": file_digest(candidate)},
            }
        )
    )
    candidate.write_text(candidate.read_text() + "* CHANGED\n")
    with pytest.raises(ValueError, match="frozen candidate changed"):
        analog_session.finalize_session(session, tmp_path / "score")
    assert not called
    assert not (tmp_path / "score").exists()


def test_finalizer_links_upstream_score_to_frozen_candidate(tmp_path, monkeypatch):
    session = tmp_path / "session"
    (session / "frozen").mkdir(parents=True)
    candidate = session / "frozen/circuit.spi"
    candidate.write_text(
        ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    )
    digest = file_digest(candidate)
    (session / "frozen.json").write_text(
        json.dumps({"candidate_sha256": digest, "action_id": "submit"})
    )
    (session / "session.json").write_text(
        json.dumps(
            {
                "source_root": str(tmp_path / "source"),
                "runtime_image": None,
                "offline_image_archive": None,
                "podman_root": None,
                "podman_runroot": None,
                "podman_single_id": False,
                "podman_no_cpu_limit": False,
            }
        )
    )
    action = session / "actions/submit"
    action.mkdir(parents=True)
    (action / "request.json").write_text(
        json.dumps({"id": "submit", "tool": "analog_submit", "arguments": {}})
    )
    (action / "response.json").write_text(
        json.dumps({"ok": True, "result": {"state": "submitted", "candidate_sha256": digest}})
    )
    captured = {}

    def fake_run(task_id, source_root, actual_candidate, output, **options):
        captured.update(
            task_id=task_id,
            source_root=source_root,
            candidate=actual_candidate,
            output=output,
            **options,
        )
        (output / "inputs").mkdir(parents=True)
        (output / "inputs/circuit.spi").write_bytes(actual_candidate.read_bytes())
        return {"state": "graded", "score": 0.0, "candidate_sha256": digest}

    monkeypatch.setattr(analog_session, "run_case", fake_run)
    result = analog_session.finalize_session(session, tmp_path / "score")
    assert result["state"] == "graded" and result["score"] == 0.0
    assert captured["candidate"] == candidate
    assert captured["task_id"] == "rlc-rf-bandpass-100mhz"
    assert captured["backend"] == "podman"
