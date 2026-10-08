"""Public session boundaries; fixtures do not stand in for EVAS acceptance."""

import json

import pytest

from circuit_harness.cli import main


def test_session_cli_passes_explicit_limits_to_server(tmp_path, capsys, monkeypatch):
    from circuit_harness.execution import vabench_session

    pin = tmp_path / "pin.json"
    pin.write_text('{"task_id":"v4-fixture"}')
    captured = {}

    def create(pin_data, directory, **limits):
        captured.update(pin=pin_data, directory=directory, limits=limits)
        return {"state": "ready"}

    monkeypatch.setattr(vabench_session, "create_session", create)
    assert (
        main(
            [
                "vabench-session",
                "--pin",
                str(pin),
                "--output",
                str(tmp_path / "session"),
                "--max-actions",
                "12",
                "--max-simulations",
                "2",
                "--timeout-s",
                "90",
            ]
        )
        == 0
    )
    assert captured["limits"] == {"max_actions": 12, "max_simulations": 2, "timeout_s": 90}
    assert json.loads(capsys.readouterr().out)["state"] == "ready"


def test_preflight_reports_enforced_session_limits(tmp_path, monkeypatch):
    from circuit_harness.execution import vabench_session

    root = tmp_path / "session"
    root.mkdir()
    (root / "session.json").write_text(
        json.dumps(
            {
                "pin": {},
                "task_id": "v4-fixture",
                "max_actions": 12,
                "max_simulations": 2,
                "timeout_s": 90,
            }
        )
    )
    monkeypatch.setattr(vabench_session, "verify_pin", lambda pin: None)
    monkeypatch.setattr(
        vabench_session,
        "_run_public_worker",
        lambda directory, config, action, *, preflight: (action / "feedback.json").write_text(
            '{"state":"ready"}'
        ),
    )
    reply = vabench_session.preflight_session(root)
    assert reply["limits"] == {"max_actions": 12, "max_simulations": 2, "timeout_s": 90}
    assert reply["task_id"] == "v4-fixture"
    assert len(reply["session_sha256"]) == 64


def test_public_session_cli_lists_no_private_paths(tmp_path, capsys):
    root = tmp_path / "episode"
    task = root / "public/task"
    task.mkdir(parents=True)
    (root / "public/submission").mkdir()
    (task / "instruction.md").write_text("Write dut.va")
    (root / "private-secret").write_text("HIDDEN_SENTINEL")
    (root / "session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "v4-fixture",
                "artifacts": ["dut.va"],
                "max_actions": 10,
            }
        )
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"id": "read-1", "tool": "vabench_read", "arguments": {"path": ""}})
    )
    assert main(["vabench-action", "--session", str(root), "--request", str(request)]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply["ok"]
    assert reply["result"]["files"] == ["task/instruction.md"]
    assert "HIDDEN_SENTINEL" not in json.dumps(reply)
    assert str(root) not in json.dumps(reply)


@pytest.fixture
def session(tmp_path):
    root = tmp_path / "session"
    (root / "public/task").mkdir(parents=True)
    (root / "public/submission").mkdir()
    (root / "public/task/instruction.md").write_text("PUBLIC")
    (root / "session.json").write_text(
        json.dumps(
            {
                "artifacts": ["dut.va"],
                "max_actions": 10,
                "max_simulations": 2,
            }
        )
    )
    return root


def call(root, id, tool, **arguments):
    from circuit_harness.execution.vabench_session import session_action

    return session_action(root, {"id": id, "tool": "vabench_" + tool, "arguments": arguments})


def test_repeated_write_is_not_reexecuted_and_freeze_is_terminal(session):
    assert call(session, "one", "write", path="dut.va", content="FIRST")["ok"]
    assert call(session, "two", "write", path="dut.va", content="SECOND")["ok"]
    assert call(session, "one", "write", path="dut.va", content="FIRST")["ok"]
    assert (session / "public/submission/dut.va").read_text() == "SECOND"
    assert call(session, "three", "submit")["result"]["status"] == "submitted"
    assert (
        call(session, "four", "write", path="dut.va", content="LATE")["error"]
        == "submission_frozen"
    )
    assert (session / "candidate/dut.va").read_text() == "SECOND"
    assert "score" not in json.dumps(call(session, "four", "read", path=""))


@pytest.mark.parametrize(
    "path", [".", "../session.json", "/etc/passwd", "task/../../session.json", "session.json"]
)
def test_read_rejects_private_and_escape_paths(session, path):
    assert not call(session, "one", "read", path=path)["ok"]


def test_read_error_explains_public_path_prefix_without_exposing_server_path(session):
    assert call(session, "write", "write", path="dut.va", content="CANDIDATE")["ok"]
    rejected = call(session, "bare", "read", path="dut.va")
    assert rejected["error"] == "ValueError"
    assert rejected["hint"] == "Read task/... or submission/...; use an empty path to list files."
    assert str(session) not in json.dumps(rejected)
    assert (
        call(session, "qualified", "read", path="submission/dut.va")["result"]["content"]
        == "CANDIDATE"
    )


def test_symlink_and_undeclared_write_rejected(session):
    (session / "public/task/link").symlink_to(session / "session.json")
    assert not call(session, "one", "read", path="task/link")["ok"]
    assert not call(session, "two", "write", path="../secret", content="x")["ok"]
    assert call(session, "three", "submit")["result"]["status"] == "candidate_incomplete"


def test_unknown_previous_action_blocks_further_mutation(session):
    previous = session / "actions/lost"
    previous.mkdir(parents=True)
    request = {
        "id": "lost",
        "tool": "vabench_write",
        "arguments": {"path": "dut.va", "content": "x"},
    }
    (previous / "request.json").write_text(json.dumps(request))
    assert (
        call(session, "lost", "write", path="dut.va", content="x")["error"] == "unknown_execution"
    )
    assert (
        call(session, "next", "write", path="dut.va", content="x")["error"]
        == "unresolved_previous_action"
    )
    assert not (session / "public/submission/dut.va").exists()


def test_detached_request_completes_and_deduplicates(session):
    import time

    from circuit_harness.execution.vabench_session import action_response, enqueue_action

    request = {
        "id": "detached",
        "tool": "vabench_write",
        "arguments": {"path": "dut.va", "content": "DETACHED"},
    }
    assert enqueue_action(session, request)["state"] == "accepted"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        reply = action_response(session, "detached")
        if reply.get("state") != "pending":
            break
        time.sleep(0.05)
    assert reply["ok"]
    assert enqueue_action(session, request)["state"] == "accepted"
    assert len(list((session / "actions").iterdir())) == 1
    with pytest.raises(ValueError, match="another request"):
        enqueue_action(session, {**request, "arguments": {"path": "dut.va", "content": "CHANGED"}})


def test_frozen_public_trace_archive_is_repeatable_and_checked(session, tmp_path):
    from circuit_harness.execution.vabench_session import archive_episode

    call(session, "write", "write", path="dut.va", content="CANDIDATE")
    call(session, "submit", "submit")
    root = tmp_path / "archives"
    first = archive_episode(session, root, "episode")
    assert archive_episode(session, root, "episode") == first
    assert "actions/write/response.json" in first["members"]
    assert first["candidate"]["dut.va"]["bytes"] == 9
    (root / "episodes/episode/episode.tar.gz").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="archive changed"):
        archive_episode(session, root, "episode")


def test_downloaded_episode_archive_checks_members(session, tmp_path):
    from circuit_harness.execution.vabench_session import (
        archive_episode,
        verify_episode_archive,
    )

    call(session, "write", "write", path="dut.va", content="CANDIDATE")
    call(session, "submit", "submit")
    root = tmp_path / "archives"
    archive_episode(session, root, "episode")
    record = root / "episodes/episode"
    assert verify_episode_archive(record)["candidate"]["dut.va"]["bytes"] == 9
    receipt = json.loads((record / "receipt.json").read_text())
    receipt["members"]["candidate/dut.va"]["sha256"] = "0" * 64
    (record / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="member changed"):
        verify_episode_archive(record)
