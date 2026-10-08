"""Constructed saved-evidence fixtures, not live model/simulator validation."""

import hashlib
import json
import stat
import subprocess
from pathlib import Path

import pytest

from alphaapollo.workflows.chips_episode_report import main


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def action(root, ident, tool, result, start, arguments=None):
    directory = root / "tools" / ident
    write(directory / "request.json", {"id": ident, "tool": tool, "arguments": arguments or {}})
    write(directory / "response.json", {"ok": True, "result": result, "action_id": ident})
    write(directory / "timing.json", {"started_at": start, "finished_at": start + 1})


@pytest.mark.parametrize("feedback", ("matching", "other", "missing"))
def test_report_tracks_restore_and_whether_final_candidate_was_publicly_simulated(
    tmp_path, feedback
):
    from alphaapollo.workflows.chips_episode_report import build_report

    root = tmp_path / "evidence"
    first, second = "R1 IN OUT 50\n", "R1 IN OUT 100\n"
    h1, h2 = [hashlib.sha256(text.encode()).hexdigest() for text in (first, second)]
    write(root / "pi-outcome.json", {"termination_reason": "final"})
    write(root / "report.json", {"state": "submitted", "candidate_sha256": h1})
    action(root, "w1", "analog_write", {"candidate_sha256": h1}, 1, {"content": first})
    action(
        root,
        "sim",
        "analog_simulate",
        {"candidate_sha256": h2 if feedback == "other" else h1, "state": "simulated"},
        3,
    )
    if feedback == "missing":
        (root / "tools/sim/response.json").unlink()
    action(root, "w2", "analog_write", {"candidate_sha256": h2}, 5, {"content": second})
    action(root, "restore", "analog_restore", {"candidate_sha256": h1}, 7, {"candidate_sha256": h1})
    action(root, "submit", "analog_submit", {"candidate_sha256": h1, "state": "submitted"}, 9)
    report = build_report(root)
    assert [row["sha256"] for row in report["candidate_changes"]] == [h1, h2, h1]
    restored = report["candidate_changes"][-1]
    assert restored["previous_sha256"] == h2 and restored["operation"] == "restore"
    assert "-R1 IN OUT 100" in restored["diff"]
    assert (
        report["outcome"]["final_candidate_publicly_simulated"]
        is {"matching": True, "other": False, "missing": None}[feedback]
    )


def test_native_report_retains_wire_payloads_and_canonical_turns(tmp_path):
    root = tmp_path / "native"
    write(root / "native-outcome.json", {"state": "completed", "termination_reason": "max_turns"})
    write(root / "report.json", {"state": "unsubmitted"})
    turn = {
        "index": 0,
        "generation_response": {"content": "observed native answer"},
        "environment_transition": {"raw_observation": {"ok": False}},
    }
    write(root / "runtime-result.json", {"turns": [turn], "termination_reason": "max_turns"})
    payload = {"model": "fixture", "messages": [{"role": "user", "content": "actual wire input"}]}
    write(
        root / "http/001-request.json",
        {
            "attempt": 1,
            "state": "dispatch_started",
            "elapsed_s": 2,
            "bytes": 100,
            "payload": payload,
        },
    )
    body = {
        "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    write(
        root / "http/001-response.json",
        {"attempt": 1, "status_code": 200, "elapsed_s": 5, "body": json.dumps(body)},
    )
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["agent_kind"] == "native"
    assert report["outcome"]["category"] == "budget_exhausted_unsubmitted"
    assert report["runtime_turns"] == [turn]
    assert report["metrics"]["model_requests"] == 1
    request = report["model_requests"][0]
    assert request["elapsed_seconds"] == 3
    assert request["started_at"] is None  # monotonic offsets are not Unix timestamps
    assert request["request"]["payload"] == payload
    assert request["response"]["body"] == json.dumps(body)
    html = (output / "report.html").read_text()
    assert "observed native answer" in html and "actual wire input" in html


def test_pi_report_attaches_captured_provider_payload_by_request_id(tmp_path):
    root = tmp_path / "pi"
    write(root / "pi-outcome.json", {"events": [], "termination_reason": "final"})
    write(root / "model-budget.jsonl", {"event": "request", "call": 1, "at": 10})
    wire = {
        "attempt": 1,
        "at": 10.1,
        "bytes": 100,
        "payload": {"messages": [{"role": "system", "content": "actual CLI context"}]},
    }
    write(root / "pi-wire-requests.jsonl", wire)
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["model_requests"][0]["provider_request"] == wire
    assert "actual CLI context" in (output / "report.html").read_text()


def test_pi_report_rejects_unmatched_payload_identity(tmp_path):
    root = tmp_path / "pi"
    write(root / "pi-outcome.json", {"events": [], "termination_reason": "final"})
    write(root / "model-budget.jsonl", {"event": "request", "call": 1, "at": 10})
    write(root / "pi-wire-requests.jsonl", {"attempt": 2, "payload": {}})
    assert main([str(root), "--output", str(tmp_path / "report")]) == 2


def test_report_reads_saved_success_without_execution_or_source_changes(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    write(root / "agent-input.json", {"system": "Only public tools", "prompt": "Design <RC>"})
    write(root / "pi-outcome.json", {"events": [], "termination_reason": "final"})
    write(
        root / "report.json",
        {
            "state": "verified",
            "task_id": "constructed-rc",
            "result": {"execution": "ok", "verdict": "pass", "score_authority": "development_only"},
        },
    )
    action(root, "s1", "vabench_simulate", {"status": "succeeded"}, 10)
    action(root, "s2", "vabench_submit", {"status": "submitted"}, 12)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def no_process(*args, **kwargs):
        raise AssertionError("offline reporting must not launch a process")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["outcome"]["category"] == "success"
    assert report["outcome"]["authority"] == "development_only"
    assert report["metrics"]["tool_calls"] == 2
    assert report["metrics"]["tool_elapsed_seconds"] == 2
    assert report["usage"]["totals"] is None
    assert report["evidence_gaps"]
    html = (output / "report.html").read_text()
    assert "Only public tools" in html and "Design &lt;RC&gt;" in html
    assert "vabench_simulate" in html and "development_only" in html
    assert "p0002" not in html and "gold answer" not in html
    assert (output / "timeline.csv").is_file()
    assert (output / "report.md").is_file()
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in output.iterdir())
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_unsubmitted_length_stop_preserves_budget_and_missing_usage(tmp_path):
    root = tmp_path / "analog"
    write(
        root / "pi-outcome.json",
        {"events": [], "termination_reason": "final", "usage": {"output": 4}},
    )
    write(root / "report.json", {"state": "unsubmitted", "final_score": "not_run"})
    (root / "tools").mkdir()
    records = [
        {"event": "request", "at": 10, "call": 1, "bytes": 200, "maxTokens": 4},
        {
            "event": "response",
            "at": 12,
            "usage": {"input": 8, "output": 2, "reasoning": 1},
            "stopReason": "toolUse",
        },
        {"event": "request", "at": 14, "call": 2, "bytes": 400, "maxTokens": 4},
        {
            "event": "response",
            "at": 19,
            "usage": {"output": 4, "reasoning": 4},
            "stopReason": "length",
        },
    ]
    (root / "model-budget.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["outcome"]["category"] == "budget_exhausted_unsubmitted"
    assert report["outcome"]["benchmark_success"] is None
    assert report["outcome"]["score"] is None
    assert report["metrics"]["model_requests"] == 2
    assert report["metrics"]["model_elapsed_seconds"] == 7
    assert report["usage"]["totals"]["output"] == 6
    assert report["usage"]["totals"]["reasoning"] == 5
    assert report["usage"]["totals"]["totalTokens"] is None
    assert report["usage"]["coverage"]["input"] == 1
    assert report["metrics"]["tool_calls"] == 0
    assert report["model_requests"][1]["stop_reason"] == "length"
    pressure = report["model_requests"][1]["output_budget"]
    assert pressure["configured_tokens"] == 4
    assert pressure["reported_reasoning_output_fraction"] == 1.0
    assert pressure["reported_output_minus_reasoning_tokens"] == 0
    assert report["metrics"]["model_length_stops"] == 1
    assert report["metrics"]["model_request_seconds_max"] == 5
    assert "maxTokens" in (output / "report.html").read_text()


def test_repair_links_candidate_changes_and_agent_tool_ids(tmp_path):
    root = tmp_path / "evidence"
    first, second = "V(out) <= 1;\n", "V(out) <+ 1;\n"
    h1 = hashlib.sha256(first.encode()).hexdigest()
    h2 = hashlib.sha256(second.encode()).hexdigest()

    def candidate(digest):
        return {"design.va": {"sha256": digest, "bytes": len(first)}}

    write(
        root / "pi-outcome.json",
        {
            "termination_reason": "final",
            "events": [
                {
                    "kind": "tool_result",
                    "call_id": "model-call-7",
                    "content": json.dumps(
                        {
                            "action_id": "sim2",
                            "ok": True,
                            "result": {"status": "succeeded", "candidate": candidate(h2)},
                        }
                    ),
                },
            ],
        },
    )
    write(
        root / "report.json",
        {
            "state": "verified",
            "candidate": candidate(h2),
            "result": {"execution": "ok", "verdict": "pass"},
        },
    )
    action(
        root,
        "write1",
        "vabench_write",
        {"path": "design.va", "sha256": h1},
        1,
        {"path": "design.va", "content": first},
    )
    action(root, "sim1", "vabench_simulate", {"status": "failed", "candidate": candidate(h1)}, 3)
    action(
        root,
        "write2",
        "vabench_write",
        {"path": "design.va", "sha256": h2},
        5,
        {"path": "design.va", "content": second},
    )
    action(root, "sim2", "vabench_simulate", {"status": "succeeded", "candidate": candidate(h2)}, 7)
    action(root, "submit", "vabench_submit", {"status": "submitted", "candidate": candidate(h2)}, 9)
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["outcome"]["category"] == "success_after_repair"
    assert report["repairs"] == [
        {
            "failed_action": "sim1",
            "successful_action": "sim2",
            "candidate_changed": True,
            "evidence": "observed_sequence_not_causal_proof",
        }
    ]
    assert report["actions"][3]["agent_call_ids"] == ["model-call-7"]
    assert "-V(out) <= 1;" in report["candidate_changes"][1]["diff"]
    assert "+V(out) <+ 1;" in report["candidate_changes"][1]["diff"]
    assert report["candidate_integrity"]["status"] == "reported_match"
    rendered = (output / "report.html").read_text()
    assert 'href="#tool-sim2"' in rendered
    assert 'id="tool-sim2"' in rendered
    assert "Controller seconds" in rendered


def test_extracted_analog_final_binds_frozen_bytes_and_keeps_continuous_score(tmp_path):
    root = tmp_path / "episode"
    content = b"R1 IN OUT 50\n"
    digest = hashlib.sha256(content).hexdigest()
    write(root / "agent/pi-outcome.json", {"events": [], "termination_reason": "final"})
    write(root / "agent/report.json", {"state": "submitted", "final_score": "not_run"})
    action(
        root / "agent",
        "submit",
        "analog_submit",
        {"state": "submitted", "candidate_sha256": digest},
        1,
    )
    write(root / "session/frozen.json", {"candidate_sha256": digest})
    for name in ("session/frozen/circuit.spi", "final/inputs/circuit.spi"):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    write(
        root / "final/result.json",
        {
            "state": "graded",
            "score": 0.5,
            "container_exit": 0,
            "tests_total": 2,
            "tests_passed": 1,
            "task_id": "fixture-rlc",
        },
    )
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["outcome"]["category"] == "graded_failure"
    assert report["outcome"]["score"] == 0.5
    assert report["outcome"]["benchmark_success"] is None
    assert report["outcome"]["all_recorded_tests_passed"] is False
    assert report["candidate_integrity"]["status"] == "bytes_matched"
    assert report["task_id"] == "fixture-rlc"
    (root / "final/inputs/circuit.spi").write_bytes(b"ALTERED")
    assert main([str(root), "--output", str(tmp_path / "bad-report")]) == 2
    assert not (tmp_path / "bad-report").exists()


def test_report_reverifies_public_archive_and_rejects_corruption(tmp_path):
    from alphaapollo.common.execution.chips.vabench_session import archive_episode

    root = tmp_path / "evidence"
    session = tmp_path / "session"
    (session / "candidate").mkdir(parents=True)
    (session / "actions").mkdir()
    content = b"module fixture; endmodule\n"
    (session / "candidate/design.va").write_bytes(content)
    candidate = {
        "design.va": {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    }
    write(session / "session.json", {"task_id": "fixture"})
    write(session / "frozen.json", {"candidate": candidate})
    root.mkdir()
    receipt = archive_episode(session, root / "archives", "case")
    write(
        root / "report.json",
        {
            "state": "verified",
            "candidate": candidate,
            "public_trace_sha256": receipt["sha256"],
            "result": {"execution": "ok", "verdict": "pass"},
        },
    )
    write(root / "pi-outcome.json", {"events": []})
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["archives"][0]["verified_members"] == 3
    assert report["candidate_integrity"]["status"] == "public_archive_verified"
    (root / "archives/episodes/case/episode.tar.gz").write_bytes(b"CORRUPTED")
    assert main([str(root), "--output", str(tmp_path / "corrupt-report")]) == 2
    assert not (tmp_path / "corrupt-report").exists()


@pytest.mark.parametrize("state", ["simulation_error", "infrastructure_error", "cleanup_failed"])
def test_analog_public_failures_are_not_successful_tool_calls(tmp_path, state):
    root = tmp_path / "evidence"
    write(root / "report.json", {"state": "unsubmitted"})
    action(root, "sim", "analog_simulate", {"state": state}, 1)
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["actions"][0]["failed"] is True
    assert report["outcome"]["benchmark_success"] is None


def test_unified_experiment_uses_saved_settings_and_bound_final_result(tmp_path):
    root = tmp_path / "experiment"
    settings = {"model": "fixture-model", "reasoning": {"parameter": "thinking", "value": "low"}}
    write(
        root / "experiment_manifest.json",
        {
            "benchmark": "analog_design_bench",
            "task_id": "fixture-rlc",
            "experiment_settings": settings,
        },
    )
    write(root / "results.jsonl", {"status": "completed", "score": 0.75})
    write(root / "agent/report.json", {"state": "submitted"})
    write(
        root / "agent/agent-input.json",
        {
            "system": "TASK POLICY",
            "prompt": "TASK",
            "tools": [{"name": "analog_simulate", "description": "Recorded description"}],
        },
    )
    action(
        root / "agent",
        "submit",
        "analog_submit",
        {"state": "submitted", "candidate_sha256": "a" * 64},
        1,
    )
    write(
        root / "final-result.json",
        {
            "state": "graded",
            "score": 0.75,
            "container_exit": 0,
            "frozen_candidate_sha256": "a" * 64,
        },
    )
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["experiment_settings"] == settings
    assert report["outcome"]["score"] == 0.75
    assert report["outcome"]["status"] == "completed"
    assert report["outcome"]["benchmark_success"] is None
    assert report["outcome"]["category"] == "graded"
    assert report["candidate_integrity"]["status"] == "reported_match"
    assert "Recorded description" in (output / "report.html").read_text()


def test_unfinished_request_and_missing_tool_response_stay_unknown(tmp_path):
    root = tmp_path / "evidence"
    write(root / "report.json", {"state": "unsubmitted"})
    write(
        root / "tools/pending/request.json",
        {"id": "pending", "tool": "analog_simulate", "arguments": {}},
    )
    write(root / "model-budget.jsonl", {"event": "request", "at": 10, "call": 1})
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["actions"][0]["failed"] is None
    assert report["metrics"]["unresolved_tool_calls"] == 1
    assert report["metrics"]["model_elapsed_seconds"] is None
    assert report["usage"]["totals"] is None
    assert report["outcome"]["category"] == "unsubmitted"


def test_preflight_block_is_reported_without_fabricating_a_run(tmp_path):
    root = tmp_path / "evidence"
    write(
        root / "results.jsonl",
        {"status": "blocked", "task_id": "fixture", "error_type": "credential_missing"},
    )
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["outcome"]["category"] == "blocked"
    assert report["outcome"]["benchmark_success"] is None
    assert report["metrics"]["tool_calls"] is None
    assert report["metrics"]["model_requests"] is None


def test_bad_json_and_existing_or_nested_output_are_refused(tmp_path):
    root = tmp_path / "evidence"
    write(root / "report.json", {"state": "unsubmitted"})
    assert main([str(root), "--output", str(root / "derived")]) == 2
    assert not (root / "derived").exists()
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    assert main([str(root), "--output", str(output)]) == 2
    (root / "model-budget.jsonl").write_text('{"event":')
    assert main([str(root), "--output", str(tmp_path / "bad")]) == 2
    assert not (tmp_path / "bad").exists()


def test_invalid_final_execution_does_not_publish_a_score(tmp_path):
    root = tmp_path / "experiment"
    write(root / "agent/report.json", {"state": "submitted"})
    action(
        root / "agent",
        "submit",
        "analog_submit",
        {"state": "submitted", "candidate_sha256": "a" * 64},
        1,
    )
    write(
        root / "final-result.json",
        {"state": "graded", "score": 1.0, "container_exit": 1, "frozen_candidate_sha256": "a" * 64},
    )
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["outcome"]["score"] is None
    assert report["outcome"]["benchmark_success"] is None
    assert report["final_result"]["score"] == 1.0  # Raw contradictory evidence is retained.


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_collected_candidate_report_separates_harness_submission_and_design(tmp_path, score):
    root = tmp_path / "experiment"
    digest = "a" * 64
    receipt = {
        "state": "collected",
        "collection_source": "episode_end",
        "agent_submitted": False,
        "candidate_sha256": digest,
        "termination_reason": "model_request_limit",
    }
    write(root / "agent/report.json", receipt)
    write(root / "agent/collection.json", receipt)
    write(
        root / "agent/pi-outcome.json",
        {
            "termination_reason": "external_error",
            "harness_termination_reason": "model_request_limit",
        },
    )
    write(
        root / "agent/model-budget.jsonl",
        {"event": "budget_context", "call": 1, "content": "Harness budget: 1"},
    )
    write(
        root / "final-result.json",
        {
            "state": "graded",
            "score": score,
            "container_exit": 0,
            "tests_total": 1,
            "tests_passed": int(score),
            "frozen_candidate_sha256": digest,
        },
    )
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["outcome"]["harness_status"] == "finalized"
    assert report["outcome"]["agent_submitted"] is False
    assert report["outcome"]["score"] == score
    assert report["outcome"]["category"] == (
        "collected_design_pass" if score else "collected_design_fail"
    )
    assert report["outcome"]["termination_reason"] == "model_request_limit"
    assert report["outcome"]["raw_termination_reason"] == "external_error"
    assert report["candidate_integrity"]["submitted"] is None
    assert report["candidate_integrity"]["collected"] == {"circuit.spi": digest}
    assert not any("Unknown model-budget event" in gap for gap in report["evidence_gaps"])
    assert report["budget_context"][0]["content"] == "Harness budget: 1"
    write(
        root / "final-result.json",
        {"state": "graded", "score": score, "frozen_candidate_sha256": "b" * 64},
    )
    assert main([str(root), "--output", str(tmp_path / "tampered")]) == 2


def test_ended_missing_candidate_is_visible_without_invented_grade(tmp_path):
    root = tmp_path / "episode"
    receipt = {
        "state": "missing_candidate",
        "agent_submitted": False,
        "collection_source": "episode_end",
        "candidate_sha256": None,
        "termination_reason": "output_token_limit",
    }
    write(root / "session/episode-end.json", receipt)
    write(root / "agent/report.json", receipt)
    output = tmp_path / "report"
    assert main([str(root), "--output", str(output)]) == 0
    report = json.loads((output / "episode-report.json").read_text())
    assert report["outcome"]["harness_status"] == "closed_without_candidate"
    assert report["outcome"]["agent_submitted"] is False
    assert report["outcome"]["score"] is None
