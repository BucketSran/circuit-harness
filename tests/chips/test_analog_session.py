"""Constructed public session fixtures; real model behavior is not asserted here."""

import json

import pytest

from circuit_harness.execution import analog_design_bench as adb
from circuit_harness.execution.analog_session import create_session, session_action


@pytest.fixture
def session(tmp_path, monkeypatch):
    task_id = "rlc-rf-bandpass-100mhz"
    source = tmp_path / "source/tasks" / task_id
    benches = source / "environment/starter/testbench"
    benches.mkdir(parents=True)
    (source / "instruction.md").write_text("Design a bandpass filter.")
    (source / "environment/starter/circuit.spi").write_text("* Empty starter\n")
    (benches / "tb_ac.spi").write_text(
        '.include "/app/circuit.spi"\nmeas ac gain_100mhz_db find x at=100Meg\n'
    )
    (benches / "tb_stopband.spi").write_text(
        '.include "/app/circuit.spi"\nmeas ac low_stop_peak_db find x at=90Meg\n'
    )
    (source / "tests").mkdir()
    (source / "tests/hidden.txt").write_text("PRIVATE_SENTINEL")
    (source / "solution").mkdir()
    (source / "solution/circuit.spi").write_text("PRIVATE_SENTINEL")
    monkeypatch.setitem(
        adb.TASKS,
        task_id,
        adb.Task(
            "test-commit",
            adb.tree_digest(source),
            "test@sha256:" + "a" * 64,
            adb.TASKS[task_id].public_rlc,
        ),
    )
    fake_podman = tmp_path / "fake-podman"
    fake_podman.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, re, sys\n"
        "args = sys.argv[1:]\n"
        "assert all('solution' not in x and '/tests' not in x for x in args)\n"
        "mount = next(x for x in args if x.endswith(':/app/public:ro')).split(':')[0]\n"
        "for bench in pathlib.Path(mount).glob('*.spi'):\n"
        "    for name in re.findall(r'meas ac ([a-z_0-9]+)', bench.read_text()):\n"
        "        print(f'{name} = -2.0')\n"
    )
    fake_podman.chmod(0o700)
    root = tmp_path / "session"
    create_session(tmp_path / "source", root, podman=str(fake_podman), max_simulations=1)
    return root


def call(root, action_id, tool, **arguments):
    return session_action(root, {"id": action_id, "tool": "analog_" + tool, "arguments": arguments})


def test_two_tasks_share_session_tools_but_keep_candidates_and_grading_separate(
    session, tmp_path, monkeypatch
):
    from circuit_harness.execution import analog_session as module

    task_id = "rlc-broadband-50-to-200-match"
    task = adb.TASKS[task_id]
    source = tmp_path / "source/tasks" / task_id
    original = "Public broadband specification.\nRun python3 /app/testbench/analyze_broadband.py."
    for name, relative in task.public_rlc.public_files.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(original if name == "instruction.md" else "* public fixture\n")
    monkeypatch.setitem(
        adb.TASKS,
        task_id,
        adb.Task("fixture", adb.tree_digest(source), task.image, task.public_rlc),
    )
    other = tmp_path / "broadband-session"
    ready = create_session(tmp_path / "source", other, task_id=task_id)
    assert ready["tools"] == list(module.TOOLS)
    assert ready["task_id"] == module.session_info(other)["task_id"] == task_id
    assert set(call(other, "list", "read", path="")["result"]["files"]) == set(
        task.public_rlc.public_files
    )
    assert original in (other / "public/instruction.md").read_text()
    assert (other / "original-instruction.md").read_text() == original
    assert not call(other, "hidden", "read", path="tests/test.py")["ok"]
    bandpass = ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    broad = bandpass.replace("rlc_rf_bandpass", "rlc_broadband_match")
    old_hash = call(session, "write", "write", content=bandpass)["result"]["candidate_sha256"]
    assert not call(other, "wrong-interface", "write", content=bandpass)["ok"]
    digest = call(other, "write", "write", content=broad)["result"]["candidate_sha256"]
    assert not call(other, "foreign-restore", "restore", candidate_sha256=old_hash)["ok"]
    seen = {}

    def simulate(source, candidate, output, **options):
        seen.update(options)
        return {
            "state": "simulated",
            "diagnostic_excerpt": "",
            "measurements": {"worst_gamma": 0.05},
            "missing_measurements": [],
            "sweep": {"gamma": [0.05] * 11},
        }

    monkeypatch.setattr(module, "run_public_rlc", simulate)
    feedback = call(other, "simulate", "simulate")["result"]
    assert seen["task_id"] == task_id
    assert feedback["sweep"]["gamma"] == [0.05] * 11
    assert feedback["candidate_sha256"] == digest
    history = call(other, "history", "history")["result"]
    assert len(history["candidates"]) == 1
    assert history["candidates"][0]["public_simulations"][0]["candidate_sha256"] == digest
    assert call(other, "submit", "submit")["result"]["candidate_sha256"] == digest
    assert (other / "frozen/circuit.spi").read_text() == broad
    assert (session / "candidate.spi").read_text() == bandpass

    def grade(actual_task, source, candidate, output, **options):
        seen["graded_task"] = actual_task
        (output / "inputs").mkdir(parents=True)
        (output / "inputs/circuit.spi").write_bytes(candidate.read_bytes())
        return {"state": "graded", "score": 0.0}

    monkeypatch.setattr(module, "run_case", grade)
    result = module.finalize_session(other, tmp_path / "final")
    assert seen["graded_task"] == task_id
    assert result["frozen_candidate_sha256"] == digest


def test_session_task_identity_cannot_change_between_actions(session):
    config = json.loads((session / "session.json").read_text())
    config["task_id"] = "rlc-broadband-50-to-200-match"
    (session / "session.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="task contract changed"):
        call(session, "read", "read", path="")
    assert not (session / "candidate.spi").exists()
    assert not list((session / "actions").glob("*"))


def test_pre_parameterization_session_still_uses_the_bandpass_contract(session):
    config = json.loads((session / "session.json").read_text())
    config["schema_version"] = 2
    config.pop("task_contract_sha256", None)
    (session / "session.json").write_text(json.dumps(config))
    content = ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    assert call(session, "write", "write", content=content)["ok"]


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_candidate_history_links_public_feedback_and_restores_exact_bytes(session, line_ending):
    first = (
        "* 保存候选 α\n.subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    ).replace("\n", line_ending)
    second = first.replace("50", "100")
    digest = call(session, "w1", "write", content=first)["result"]["candidate_sha256"]
    assert (session / "candidate.spi").read_bytes() == first.encode("utf-8")
    simulation = call(session, "s1", "simulate")["result"]
    second_digest = call(session, "w2", "write", content=second)["result"]["candidate_sha256"]
    history = call(session, "history", "history")["result"]
    assert history["current_candidate_sha256"] == second_digest
    versions = {row["candidate_sha256"]: row for row in history["candidates"]}
    assert set(versions) == {digest, second_digest}
    assert versions[digest]["public_simulations"] == [{"action_id": "s1", **simulation}]
    assert versions[second_digest]["public_simulations"] == []
    assert "PRIVATE_SENTINEL" not in json.dumps(history)
    restored = call(session, "restore", "restore", candidate_sha256=digest)
    assert restored["ok"] and restored["result"]["candidate_sha256"] == digest
    assert restored["budget"]["simulations_remaining"] == 0
    assert (session / "candidate.spi").read_bytes() == first.encode("utf-8")
    assert call(session, "restore", "restore", candidate_sha256=digest) == restored
    assert call(session, "submit", "submit")["result"]["candidate_sha256"] == digest
    assert (session / "frozen/circuit.spi").read_bytes() == first.encode("utf-8")
    assert (
        call(session, "late", "restore", candidate_sha256=second_digest)["error"]
        == "submission_frozen"
    )


def test_only_public_material_is_readable_and_invalid_write_is_rejected(session):
    listed = call(session, "list", "read", path="")
    assert listed["ok"]
    assert listed["result"]["files"] == [
        "instruction.md",
        "starter/circuit.spi",
        "testbench/tb_ac.spi",
        "testbench/tb_stopband.spi",
    ]
    assert "PRIVATE_SENTINEL" not in json.dumps(listed)
    assert "PRIVATE_SENTINEL" not in json.dumps(
        call(session, "bad", "read", path="../tests/hidden.txt")
    )
    assert not call(session, "invalid", "write", content=".include /private/solution.spi")["ok"]
    assert not (session / "candidate.spi").exists()


def test_restore_rejects_unknown_unsafe_or_corrupt_versions_without_changing_candidate(session):
    content = ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    digest = call(session, "write", "write", content=content)["result"]["candidate_sha256"]
    for ident, target in (("unknown", "0" * 64), ("escape", "../candidate.spi")):
        assert not call(session, ident, "restore", candidate_sha256=target)["ok"]
        assert (session / "candidate.spi").read_text() == content
    (session / "candidates" / (digest + ".spi")).write_text("corrupt")
    assert not call(session, "corrupt", "restore", candidate_sha256=digest)["ok"]
    assert (session / "candidate.spi").read_text() == content
    assert not call(session, "history", "history")["ok"]


def test_simulation_trace_deduplicates_and_submission_freezes(session):
    content = ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    assert call(session, "write-1", "write", content=content)["ok"]
    reply = call(session, "simulate-1", "simulate")
    assert reply["ok"]
    assert reply["result"]["state"] == "simulated"
    assert reply["result"]["measurements"]["gain_100mhz_db"] == -2.0
    assert reply["result"]["task_correctness"] == "not_evaluated"
    assert "score" not in json.dumps(reply)
    assert call(session, "simulate-1", "simulate") == reply
    assert len(list((session / "simulations").iterdir())) == 1
    assert (
        call(session, "simulate-2", "simulate")["result"]["state"] == "simulation_budget_exhausted"
    )
    assert call(session, "submit", "submit")["result"]["state"] == "submitted"
    assert (session / "frozen/circuit.spi").read_text() == content
    assert call(session, "late", "write", content=content)["error"] == "submission_frozen"
    with pytest.raises(ValueError, match="another request"):
        call(session, "write-1", "write", content=content + "* modified")
    assert (session / "actions/write-1/request.json").exists()
    assert (session / "actions/simulate-1/response.json").exists()


def test_interrupted_action_blocks_replay_and_later_mutation(session):
    lost = session / "actions/lost"
    lost.mkdir(parents=True)
    (lost / "request.json").write_text(
        json.dumps({"id": "lost", "tool": "analog_write", "arguments": {"content": "x"}})
    )
    assert call(session, "lost", "write", content="x")["error"] == "unknown_execution"
    assert call(session, "next", "read", path="")["error"] == "unresolved_previous_action"


def test_public_tamper_and_malformed_tool_are_rejected(session):
    with pytest.raises(ValueError, match="invalid tool"):
        session_action(session, {"id": "bad", "tool": [], "arguments": {}})
    (session / "public/instruction.md").write_text("CHANGED")
    assert not call(session, "read", "read", path="instruction.md")["ok"]
    content = ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    assert call(session, "write", "write", content=content)["ok"]
    assert not call(session, "simulate", "simulate")["ok"]
    assert not (session / "simulations").exists()


def test_source_pin_change_rejected_before_session_creation(tmp_path, monkeypatch):
    (tmp_path / "source/tasks/rlc-rf-bandpass-100mhz").mkdir(parents=True)
    monkeypatch.setitem(
        adb.TASKS,
        "rlc-rf-bandpass-100mhz",
        adb.Task(
            "test",
            "0" * 64,
            "test@sha256:" + "a" * 64,
            adb.TASKS["rlc-rf-bandpass-100mhz"].public_rlc,
        ),
    )
    with pytest.raises(ValueError, match="source pin mismatch"):
        create_session(tmp_path / "source", tmp_path / "session")
    assert not (tmp_path / "session").exists()


def test_detached_action_survives_launcher_and_deduplicates(session):
    import time

    from circuit_harness.execution.analog_session import action_response, enqueue_action

    content = ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    request = {"id": "detached", "tool": "analog_write", "arguments": {"content": content}}
    assert enqueue_action(session, request) == {"state": "accepted", "id": "detached"}
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        reply = action_response(session, "detached")
        if reply.get("state") != "pending":
            break
        time.sleep(0.05)
    assert reply["ok"]
    assert enqueue_action(session, request) == {"state": "accepted", "id": "detached"}
    assert len(list((session / "actions").iterdir())) == 1
    with pytest.raises(ValueError, match="another request"):
        enqueue_action(session, {**request, "arguments": {"content": "changed"}})


def test_detached_action_does_not_acknowledge_uncertain_launch(session, monkeypatch):
    from circuit_harness.execution.analog_session import action_response, enqueue_action

    request = {"id": "uncertain", "tool": "analog_read", "arguments": {"path": ""}}

    def fail_launch(*args, **kwargs):
        raise OSError("constructed launch failure")

    monkeypatch.setattr("subprocess.Popen", fail_launch)
    with pytest.raises(OSError, match="constructed launch failure"):
        enqueue_action(session, request)
    assert enqueue_action(session, request) == {
        "state": "unknown_execution",
        "id": "uncertain",
        "retry_safe": False,
    }
    assert action_response(session, "uncertain")["state"] == "unknown_execution"


def test_detached_action_rejects_invalid_envelope_before_launch(session):
    from circuit_harness.execution.analog_session import enqueue_action

    with pytest.raises(ValueError, match="invalid tool"):
        enqueue_action(session, {"id": "bad", "tool": "analog_secret", "arguments": {}})
    assert not (session / "requests").exists()


def test_detached_budget_rejection_is_a_persisted_terminal_reply(session):
    import time

    from circuit_harness.execution.analog_session import action_response, enqueue_action
    from circuit_harness.execution.journal import atomic_json

    assert call(session, "first", "read", path="")["ok"]
    config = json.loads((session / "session.json").read_text())
    atomic_json(session / "session.json", {**config, "max_actions": 1})
    request = {"id": "over-budget", "tool": "analog_read", "arguments": {"path": ""}}
    assert enqueue_action(session, request)["state"] == "accepted"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        reply = action_response(session, request["id"])
        if reply.get("state") != "pending":
            break
        time.sleep(0.02)
    assert reply["ok"] is False and reply["error"] == "action_budget_exhausted"
    assert reply["budget"]["actions_remaining"] == 0
    assert json.loads((session / "requests/over-budget/response.json").read_text()) == reply
    assert [p.name for p in (session / "actions").iterdir()] == ["first"]


def test_episode_end_collects_last_candidate_once_and_prevents_late_writes(session):
    from circuit_harness.execution.analog_session import close_session

    content = ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    assert call(session, "write", "write", content=content)["ok"]
    receipt = close_session(session, "model_request_limit")
    assert receipt["state"] == "collected"
    assert receipt["collection_source"] == "episode_end"
    assert receipt["agent_submitted"] is False
    assert (session / "frozen/circuit.spi").read_text() == content
    assert close_session(session, "model_request_limit") == receipt
    assert call(session, "late", "write", content=content)["error"] == "episode_closed"


def test_episode_end_missing_candidate_and_explicit_submission_are_distinct(session):
    from circuit_harness.execution.analog_session import close_session

    receipt = close_session(session, "output_token_limit")
    assert receipt["state"] == "missing_candidate"
    assert receipt["candidate_sha256"] is None
    assert not (session / "frozen.json").exists()


def test_episode_end_preserves_explicit_submission(session):
    from circuit_harness.execution.analog_session import close_session

    content = ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    call(session, "write", "write", content=content)
    submitted = call(session, "submit", "submit")["result"]
    original = (session / "frozen.json").read_bytes()
    receipt = close_session(session, "final")
    assert receipt["agent_submitted"] is True
    assert receipt["collection_source"] == "agent_submit"
    assert receipt["candidate_sha256"] == submitted["candidate_sha256"]
    assert (session / "frozen.json").read_bytes() == original


def test_episode_end_refuses_pending_or_unknown_server_work(session):
    from circuit_harness.execution.analog_session import close_session

    pending = session / "requests/pending"
    pending.mkdir(parents=True)
    (pending / "request.json").write_text("{}")
    reply = close_session(session, "deadline")
    assert reply["state"] == "awaiting_action_recovery"
    assert not (session / "episode-end.json").exists()
    with pytest.raises(ValueError, match="termination"):
        close_session(session, "ssh_disconnected")


def test_collected_candidate_can_be_graded_but_receipt_tampering_is_rejected(session, monkeypatch):
    from circuit_harness.execution import analog_session

    content = ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    call(session, "write", "write", content=content)
    receipt = analog_session.close_session(session, "deadline")

    def score(_task, _source, candidate, output, **_kwargs):
        (output / "inputs").mkdir(parents=True)
        (output / "inputs/circuit.spi").write_bytes(candidate.read_bytes())
        return {"state": "graded", "score": 0.0}

    monkeypatch.setattr(analog_session, "run_case", score)
    result = analog_session.finalize_session(session, session / "score")
    assert result["score"] == 0.0
    assert result["frozen_candidate_sha256"] == receipt["candidate_sha256"]
    (session / "episode-end.json").write_text(json.dumps({**receipt, "agent_submitted": True}))
    with pytest.raises(ValueError, match="acknowledged collection"):
        analog_session.finalize_session(session, session / "score-2")


def test_collection_waits_for_active_worker_and_preserves_legacy_rules(session):
    import fcntl

    from circuit_harness.execution.analog_session import close_session

    with (session / ".session.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert close_session(session, "deadline")["state"] == "awaiting_action_recovery"
    config = json.loads((session / "session.json").read_text())
    config.pop("collection_policy")
    (session / "session.json").write_text(json.dumps(config))
    assert close_session(session, "final")["state"] == "unsubmitted"
    assert not (session / "ending.json").exists()


def test_budget_feedback_reserves_last_action_for_submission(session):
    config = json.loads((session / "session.json").read_text())
    config["max_actions"] = 2
    (session / "session.json").write_text(json.dumps(config))
    content = ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    written = call(session, "write", "write", content=content)
    assert written["budget"]["actions_remaining"] == 1
    assert written["budget"]["simulations_remaining"] == 1
    rejected = call(session, "read", "read", path="")
    assert rejected["error"] == "action_reserved_for_submission"
    submitted = call(session, "submit", "submit")
    assert submitted["result"]["state"] == "submitted"
    assert submitted["budget"]["actions_remaining"] == 0
