"""Private Analog episode packages remain verifiable after scratch is removed."""

import json

import pytest

from circuit_harness.execution.analog_episode import archive_episode, verify_episode
from circuit_harness.execution.journal import file_digest


def test_episode_archive_links_actions_model_trace_and_final_candidate(tmp_path):
    session = tmp_path / "session"
    (session / "frozen").mkdir(parents=True)
    (session / "actions/one").mkdir(parents=True)
    (session / "public").mkdir()
    (session / "simulations/one").mkdir(parents=True)
    candidate = session / "frozen/circuit.spi"
    candidate.write_text("R1 IN OUT 50\n")
    digest = file_digest(candidate)
    (session / "candidates").mkdir()
    (session / "candidates" / (digest + ".spi")).write_bytes(candidate.read_bytes())
    (session / "session.json").write_text(json.dumps({"task_id": "fixture"}))
    (session / "frozen.json").write_text(json.dumps({"candidate_sha256": digest}))
    (session / "actions/one/request.json").write_text(json.dumps({"tool": "analog_write"}))
    (session / "actions/one/response.json").write_text(json.dumps({"ok": True}))
    (session / "public/instruction.md").write_text("PUBLIC")
    (session / "simulations/one/public.log").write_text("gain = -2")
    agent = tmp_path / "agent"
    (agent / "tools/one").mkdir(parents=True)
    (agent / "pi-outcome.json").write_text(json.dumps({"events": [{"kind": "tool_call"}]}))
    (agent / "pi-events.jsonl").write_text('{"type":"message_end"}\n')
    (agent / "runtime-result.json").write_text('{"termination_reason":"submitted"}')
    (agent / ".alphaapollo-environment.json").write_text('{"token":"NEVER_ARCHIVE"}')
    (agent / "model-budget.jsonl").write_text('{"event":"request"}\n')
    (agent / "tools/one/request.json").write_text(json.dumps({"tool": "analog_write"}))
    (agent / "private.key").write_text("NEVER_ARCHIVE")
    final = tmp_path / "final"
    (final / "inputs").mkdir(parents=True)
    (final / "inputs/circuit.spi").write_bytes(candidate.read_bytes())
    (final / "result.json").write_text(json.dumps({"state": "graded", "score": 0.0}))
    archive_root = tmp_path / "archives"
    receipt = archive_episode(session, archive_root, "episode-1", agent=agent, final=final)
    record = archive_root / "episodes/episode-1"
    assert verify_episode(record) == receipt
    assert receipt["candidate_sha256"] == digest
    assert "agent/pi-outcome.json" in receipt["members"]
    assert "agent/pi-events.jsonl" in receipt["members"]
    assert "agent/runtime-result.json" in receipt["members"]
    assert "session/actions/one/request.json" in receipt["members"]
    assert receipt["members"]["session/candidates/" + digest + ".spi"]["sha256"] == digest
    assert "final/result.json" in receipt["members"]
    assert all("private.key" not in name for name in receipt["members"])
    assert all(".alphaapollo-environment" not in name for name in receipt["members"])
    assert archive_episode(session, archive_root, "episode-1", agent=agent, final=final) == receipt
    (record / "episode.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="archive package changed"):
        verify_episode(record)


def test_episode_refuses_changed_final_input_and_symlink(tmp_path):
    session = tmp_path / "session"
    (session / "frozen").mkdir(parents=True)
    candidate = session / "frozen/circuit.spi"
    candidate.write_text("R1 IN OUT 50\n")
    (session / "session.json").write_text("{}")
    (session / "frozen.json").write_text(json.dumps({"candidate_sha256": file_digest(candidate)}))
    final = tmp_path / "final"
    (final / "inputs").mkdir(parents=True)
    (final / "inputs/circuit.spi").write_text("DIFFERENT")
    with pytest.raises(ValueError, match="final scorer candidate"):
        archive_episode(session, tmp_path / "archives", "episode-1", final=final)
    (final / "inputs/circuit.spi").write_bytes(candidate.read_bytes())
    (session / "secret-link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symbolic link"):
        archive_episode(session, tmp_path / "archives", "episode-1", final=final)


def test_archive_ended_episode_without_candidate_preserves_failure_evidence(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "session.json").write_text('{"collection_policy":"episode_end"}')
    with pytest.raises((ValueError, FileNotFoundError)):
        archive_episode(session, tmp_path / "archives", "still-open")
    receipt = {
        "state": "missing_candidate",
        "collection_source": "episode_end",
        "agent_submitted": False,
        "candidate_sha256": None,
        "termination_reason": "output_token_limit",
    }
    (session / "episode-end.json").write_text(json.dumps(receipt))
    archived = archive_episode(session, tmp_path / "archives", "ended")
    assert archived["candidate_sha256"] is None
    assert "session/episode-end.json" in archived["members"]
    assert verify_episode(tmp_path / "archives/episodes/ended") == archived
    with pytest.raises(ValueError, match="without a candidate"):
        archive_episode(session, tmp_path / "archives", "invalid-final", final=tmp_path / "final")
    (session / "candidate.spi").write_text("unexpected candidate")
    with pytest.raises(ValueError, match="missing candidate"):
        archive_episode(session, tmp_path / "archives", "contradictory")
