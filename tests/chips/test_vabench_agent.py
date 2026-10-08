"""Control-plane contracts; model/network doubles are explicitly local fixtures."""

import json

import pytest

from alphaapollo.common.execution.chips.analog_remote import LocalAnalog, RemoteAnalog
from alphaapollo.common.execution.chips.vabench_remote import LocalVabench, RemoteVabench


@pytest.fixture
def config():
    return {
        "host": "lab-host",
        "python": "/usr/bin/python3",
        "bundle": "/private/chips.pyz",
        "session": "/private/episode",
        "pi_cli": "/private/pi",
        "base_url": "http://127.0.0.1:1/v1",
        "model": "fixture",
        "policy_kind": "scripted_http_fixture",
    }


def test_lost_ack_retries_same_envelope_and_persists_action_join(config, tmp_path, monkeypatch):
    remote = RemoteVabench(config, tmp_path)
    envelopes = []

    def cli(*args, payload=None):
        if args[0] == "vabench-request":
            envelopes.append(payload)
            if len(envelopes) == 1:
                raise ConnectionError("lost acknowledgement")
            return {"state": "accepted"}
        return {"ok": True, "result": {"status": "succeeded"}}

    monkeypatch.setattr(remote, "cli", cli)
    monkeypatch.setattr(
        "alphaapollo.common.execution.chips.session_transport.time.sleep", lambda _: None
    )
    reply = remote.call("vabench_simulate", {}, action_id="same-id")
    assert len(envelopes) == 2 and envelopes[0] == envelopes[1]
    assert reply["action_id"] == "same-id"
    assert json.loads((tmp_path / "same-id/response.json").read_text()) == reply


def test_unknown_remote_action_blocks_new_action_until_recovered(config, tmp_path, monkeypatch):
    remote = RemoteVabench(config, tmp_path)
    reply = remote.call("vabench_simulate", {}, action_id="pending", timeout_s=0)
    assert reply["server_execution"] == "unknown"
    assert (
        remote.call("vabench_write", {"path": "dut.va", "content": "x"})["error"]
        == "recover_previous_action"
    )
    monkeypatch.setattr(
        remote,
        "cli",
        lambda *args, **kwargs: (
            {"state": "accepted"} if args[0] == "vabench-request" else {"ok": True}
        ),
    )
    assert remote.call("vabench_simulate", {}, action_id="pending")["ok"]


@pytest.mark.parametrize("transport_type", [RemoteVabench, LocalVabench, RemoteAnalog, LocalAnalog])
def test_recreated_transport_recovers_pending_id_before_allowing_new_actions(
    config, tmp_path, monkeypatch, transport_type
):
    original = transport_type(config, tmp_path)
    tool = "analog_simulate" if "Analog" in transport_type.__name__ else "vabench_simulate"
    original.call(tool, {}, action_id="lost", timeout_s=0)
    recovered = transport_type(config, tmp_path)
    seen = []

    def cli(*args, payload=None):
        seen.append((args[0], payload))
        return {"state": "accepted"} if payload is not None else {"ok": True, "result": {}}

    monkeypatch.setattr(recovered, "cli", cli)
    reply = recovered.call(tool, {}, action_id="new", timeout_s=0)
    assert reply["error"] == "recover_previous_action"
    assert reply["action_id"] == "lost"
    assert not (tmp_path / "new").exists()
    assert seen == []
    assert recovered.call(tool, {}, action_id="lost")["ok"]
    assert seen[0][1] == {"id": "lost", "tool": tool, "arguments": {}}
    assert transport_type(config, tmp_path).pending_action_id is None


def test_unknown_server_state_is_not_a_completed_action(config, tmp_path, monkeypatch):
    remote = RemoteVabench(config, tmp_path)
    monkeypatch.setattr(
        remote,
        "cli",
        lambda *args, **kwargs: (
            {"state": "accepted"}
            if args[0] == "vabench-request"
            else {"state": "unknown_execution"}
        ),
    )
    reply = remote.call("vabench_simulate", {}, action_id="lost")
    assert reply["server_execution"] == "unknown"
    assert not (tmp_path / "lost/response.json").exists()
    assert RemoteVabench(config, tmp_path).pending_action_id == "lost"


@pytest.mark.parametrize("transport_type", [LocalVabench, LocalAnalog])
@pytest.mark.parametrize("operation", ["read", "submit"])
def test_recovery_reuses_real_detached_worker_after_lost_ack(
    tmp_path, monkeypatch, transport_type, operation
):
    import sys

    from alphaapollo.common.execution.chips.bundle import build_cli
    from alphaapollo.common.execution.chips.journal import atomic_json, file_digest

    kind = "analog" if transport_type is LocalAnalog else "vabench"
    root = tmp_path / "session"
    public = root / "public"
    public.mkdir(parents=True)
    instruction = public / ("instruction.md" if kind == "analog" else "task/instruction.md")
    instruction.parent.mkdir(exist_ok=True)
    instruction.write_text("PUBLIC_FIXTURE")
    atomic_json(
        root / "session.json",
        {
            "max_actions": 4,
            "artifacts": ["dut.va"],
            "public_files": {"instruction.md": file_digest(instruction)},
        },
    )
    if operation == "submit":
        candidate = root / ("candidate.spi" if kind == "analog" else "public/submission/dut.va")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(
            ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
        )
    bundle = tmp_path / "chips.pyz"
    build_cli(bundle)
    config = {"python": sys.executable, "bundle": str(bundle), "session": str(root)}
    evidence = tmp_path / "evidence"
    original = transport_type(config, evidence)
    cli = original.cli

    def lose_ack(*args, **kwargs):
        reply = cli(*args, **kwargs)
        assert reply["state"] == "accepted"
        raise ConnectionError("fixture: client loses accepted acknowledgement")

    monkeypatch.setattr(original, "cli", lose_ack)
    tool = kind + "_" + operation
    arguments = {"path": instruction.relative_to(public).as_posix()} if operation == "read" else {}
    reply = original.call(tool, arguments, action_id="once", timeout_s=0.01)
    assert reply["server_execution"] == "unknown"
    started = (root / "requests/once/started.json").read_bytes()
    recovered = transport_type(config, evidence)
    assert (
        recovered.call(tool, arguments, action_id="new", timeout_s=0)["error"]
        == "recover_previous_action"
    )
    response = recovered.call(tool, arguments, action_id="once", timeout_s=10)
    if operation == "read":
        assert response["result"]["content"] == "PUBLIC_FIXTURE"
    else:
        assert response["result"]["state" if kind == "analog" else "status"] == "submitted"
        frozen = (root / "frozen.json").read_bytes()
        assert recovered.call(tool, arguments, action_id="once") == response
        mutation = {"content": "new"}
        if kind == "vabench":
            mutation["path"] = "dut.va"
        rejected = recovered.call(kind + "_write", mutation, action_id="after", timeout_s=5)
        assert rejected["error"] == "submission_frozen"
        assert (root / "frozen.json").read_bytes() == frozen
    assert (root / "requests/once/started.json").read_bytes() == started
    assert {p.name for p in (root / "requests").iterdir()} == (
        {"once", "after"} if operation == "submit" else {"once"}
    )
    assert [p.name for p in (root / "actions").iterdir()] == ["once"]


def test_interrupt_wait_stops_only_client_process_and_keeps_recoverable_action(tmp_path):
    import sys
    import threading
    import time

    script = tmp_path / "query.py"
    script.write_text("import time\ntime.sleep(3)\n")
    config = {"python": sys.executable, "bundle": str(script), "session": str(tmp_path / "session")}
    remote = LocalVabench(config, tmp_path / "evidence")
    replies = []
    thread = threading.Thread(
        target=lambda: replies.append(
            remote.call("vabench_simulate", {}, action_id="running", timeout_s=4)
        ),
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 2
    while not (tmp_path / "evidence/running/request.json").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    started = time.monotonic()
    remote.interrupt_wait()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert time.monotonic() - started < 2
    assert replies[0]["server_execution"] == "unknown"
    assert replies[0]["error"] == "wait_interrupted"
    assert LocalVabench(config, tmp_path / "evidence").pending_action_id == "running"


def test_local_transport_uses_local_bundle_and_copies_bounded_artifact(
    config, tmp_path, monkeypatch
):
    from alphaapollo.workflows.chips_vabench_agent import transport

    local = transport({**config, "transport": "local"}, tmp_path / "tools")
    assert isinstance(local, LocalVabench)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return __import__("subprocess").CompletedProcess(argv, 0, '{"state":"ready"}', "")

    monkeypatch.setattr(local, "_run_command", run)
    assert local.cli("vabench-preflight", "--session", config["session"]) == {"state": "ready"}
    assert calls[0][:3] == [config["python"], "-B", config["bundle"]]
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"archive")
    target = tmp_path / "download" / source.name
    local.download(source, target)
    assert target.read_bytes() == b"archive"
    with pytest.raises(ConnectionError, match="exceeded limit"):
        local.download(source, target, max_bytes=1)
    assert target.read_bytes() == b"archive"


def test_pi_key_only_in_model_environment_and_final_judge_not_tool(config, tmp_path, monkeypatch):
    from alphaapollo.reasoning.runtime.external import agents
    from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalRunOutcome
    from alphaapollo.workflows.chips_vabench_agent import run_pi

    captured = {}

    class Session:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, task, *, workspace):
            return ExternalRunOutcome(final_text="fixture")

        def close(self):
            pass

    monkeypatch.setattr(agents.pi, "PiSession", Session)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "LOCAL_SECRET_SENTINEL")
    run_pi(config, tmp_path)
    assert captured["tool_mode"] == "no_builtin"
    assert captured["env"]["CHIPS_MODEL_KEY"] == "LOCAL_SECRET_SENTINEL"
    child = captured["mcp_servers"]["vabench"]
    assert child["command"] == "/usr/bin/env" and child["args"][0] == "-i"
    assert "PYTHONUTF8=1" in child["args"]
    assert child["args"][child["args"].index("-m") + 1] == (
        "alphaapollo.workflows.chips_vabench_agent"
    )
    assert "LOCAL_SECRET_SENTINEL" not in json.dumps(child)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "LOCAL_SECRET_SENTINEL" not in path.read_text()
    with pytest.raises(ValueError, match="fresh evidence"):
        run_pi(config, tmp_path)


@pytest.mark.parametrize("model_id", ("glm-5.3", "glm-5.3-flash"))
def test_bigmodel_coding_plan_uses_glm_stream_compat(config, tmp_path, monkeypatch, model_id):
    from alphaapollo.reasoning.runtime.external import agents
    from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalRunOutcome
    from alphaapollo.workflows.chips_vabench_agent import run_pi

    class Session:
        def __init__(self, **kwargs):
            pass

        def run(self, task, *, workspace):
            return ExternalRunOutcome(final_text="fixture")

        def close(self):
            pass

    monkeypatch.setattr(agents.pi, "PiSession", Session)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "LOCAL_SECRET_SENTINEL")
    run_pi(
        {
            **config,
            "policy_kind": "remote_model",
            "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
            "model": model_id,
            "thinking": "low",
        },
        tmp_path,
    )
    settings = json.loads((tmp_path / "pi-home/.pi/agent/models.json").read_text())
    provider = settings["providers"]["chips-glm"]
    model = provider["models"][0]
    assert provider["baseUrl"] == "https://open.bigmodel.cn/api/coding/paas/v4"
    assert provider["apiKey"] == "${CHIPS_MODEL_KEY}"
    assert model["id"] == model_id
    assert model["reasoning"]
    assert model["compat"]["zaiToolStream"]
    assert model["compat"]["supportsReasoningEffort"] is True
    assert model["thinkingLevelMap"] == {"low": "low", "high": "high", "xhigh": "max"}


def test_real_vabench_pi_requires_an_explicit_thinking_level(config):
    from alphaapollo.workflows.chips_vabench_agent import validate_pilot

    remote = {**config, "policy_kind": "remote_model", "base_url": "https://example.invalid"}
    with pytest.raises(ValueError, match="explicit thinking level"):
        validate_pilot(remote)
    with pytest.raises(ValueError, match="cannot disable thinking"):
        validate_pilot({**remote, "model": "glm-5.3-flash", "thinking": "off"})


def test_pi_codex_uses_private_oauth_directory_and_four_public_tools(config, tmp_path, monkeypatch):
    from alphaapollo.reasoning.runtime.external import agents
    from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalRunOutcome
    from alphaapollo.workflows.chips_vabench_agent import run_pi

    auth_dir = tmp_path / "private-auth"
    auth_dir.mkdir(mode=0o700)
    (auth_dir / "auth.json").write_text(
        '{"openai-codex":{"type":"oauth","access":"PRIVATE_SENTINEL"}}'
    )
    (auth_dir / "auth.json").chmod(0o600)
    captured = {}

    class Session:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, task, *, workspace):
            return ExternalRunOutcome(final_text="fixture")

        def close(self):
            pass

    monkeypatch.setattr(agents.pi, "PiSession", Session)
    monkeypatch.delenv("CHIPS_MODEL_KEY", raising=False)
    evidence = tmp_path / "evidence"
    run_pi(
        {
            **{key: value for key, value in config.items() if key != "base_url"},
            "provider_kind": "openai_codex",
            "pi_auth_dir": str(auth_dir),
            "model": "gpt-5.6-luna",
            "thinking": "medium",
            "policy_kind": "remote_model",
        },
        evidence,
    )
    assert captured["provider"] == "openai-codex"
    assert captured["thinking"] == "medium"
    assert captured["tool_mode"] == "no_builtin"
    assert captured["env"]["PI_CODING_AGENT_DIR"] == str(auth_dir)
    assert "CHIPS_MODEL_KEY" not in captured["env"]
    assert "PRIVATE_SENTINEL" not in json.dumps(
        {key: value for key, value in captured.items() if key != "event_sink"}
    )
    assert "PRIVATE_SENTINEL" not in "".join(
        p.read_text() for p in evidence.rglob("*") if p.is_file()
    )


def test_pi_codex_rejects_public_or_missing_oauth_credentials(config, tmp_path):
    from alphaapollo.workflows.chips_vabench_agent import validate_pilot

    auth_dir = tmp_path / "auth"
    auth_dir.mkdir(mode=0o700)
    auth_file = auth_dir / "auth.json"
    auth_file.write_text('{"other":{"type":"oauth"}}')
    auth_file.chmod(0o600)
    operator = {
        **{key: value for key, value in config.items() if key != "base_url"},
        "provider_kind": "openai_codex",
        "pi_auth_dir": str(auth_dir),
        "policy_kind": "remote_model",
    }
    with pytest.raises(ValueError, match="OAuth credentials"):
        validate_pilot(operator)
    auth_file.write_text('{"openai-codex":{"type":"oauth"}}')
    auth_file.chmod(0o644)
    with pytest.raises(ValueError, match="OAuth credentials"):
        validate_pilot(operator)
    auth_file.chmod(0o600)
    auth_file.write_text('{"openai-codex":{"type":"api_key","key":"PRIVATE_SENTINEL"}}')
    with pytest.raises(ValueError, match="OAuth credentials"):
        validate_pilot(operator)


def test_only_acknowledged_submit_allows_final_scorer(tmp_path):
    from alphaapollo.workflows.chips_vabench_agent import submission_confirmed

    action = tmp_path / "tools" / "submit-id"
    action.mkdir(parents=True)
    (action / "request.json").write_text(json.dumps({"tool": "vabench_submit"}))
    assert not submission_confirmed(tmp_path)
    (action / "response.json").write_text(
        json.dumps({"ok": False, "result": {"status": "submitted"}})
    )
    assert not submission_confirmed(tmp_path)
    (action / "response.json").write_text(
        json.dumps({"ok": True, "result": {"status": "submitted"}})
    )
    assert submission_confirmed(tmp_path)


@pytest.mark.parametrize(
    "extra",
    [
        {"api_key": "SENTINEL"},
        {"max_model_calls": 0},
        {"max_output_tokens": True},
        {"base_url": "https://user:pass@example.com/v1"},
    ],
)
def test_pilot_rejects_secret_config_and_invalid_limits_before_writing(config, tmp_path, extra):
    from alphaapollo.workflows.chips_vabench_agent import run_pi

    target = tmp_path / "run"
    with pytest.raises(ValueError):
        run_pi({**config, **extra}, target)
    assert not target.exists()


def test_detach_starts_private_independent_operator_without_persisting_key(
    config, tmp_path, monkeypatch
):
    from alphaapollo.workflows.chips_vabench_agent import main

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    evidence = tmp_path / "evidence"
    captured = {}

    def popen(argv, **kwargs):
        captured.update({"argv": argv, **kwargs})
        return type("Process", (), {"pid": 1234})()

    monkeypatch.setenv("CHIPS_MODEL_KEY", "LOCAL_SECRET_SENTINEL")
    monkeypatch.setattr("alphaapollo.workflows.chips_vabench_agent.subprocess.Popen", popen)
    assert main(["detach", "--config", str(config_path), "--evidence", str(evidence)]) == 0
    assert captured["start_new_session"] and captured["stdin"] is not None
    assert captured["argv"][3] == "pi"
    assert json.loads((evidence / "launcher.json").read_text())["pid"] == 1234
    assert evidence.stat().st_mode & 0o777 == 0o700
    for path in evidence.rglob("*"):
        if path.is_file():
            assert "LOCAL_SECRET_SENTINEL" not in path.read_text()


@pytest.mark.parametrize("verdict, success", [("pass", True), ("fail", False)])
def test_single_pi_cell_stays_pending_until_separate_collection_verifies_result(
    config, tmp_path, monkeypatch, verdict, success
):
    """Constructed server/Agent replies exercise the real operator CLI ledger."""
    from alphaapollo.common.execution.chips.journal import atomic_json
    from alphaapollo.workflows import chips_vabench_agent as agent

    operator = {
        **config,
        "task_id": "v4-001",
        "job_root": "/private/jobs",
        "job_id": "pi-fixture-001",
        "archive_root": "/private/archives",
    }
    config_path = tmp_path / "operator.json"
    config_path.write_text(json.dumps(operator), encoding="utf-8")
    evidence = tmp_path / "evidence"
    calls = []

    class FixtureServer:
        def cli(self, command, *args, **kwargs):
            calls.append(command)
            if command == "vabench-preflight":
                return {
                    "state": "ready",
                    "task_id": "v4-001",
                    "session_sha256": "c" * 64,
                    "limits": {"max_simulations": 4},
                }
            assert command == "vabench-finalize"
            return {"state": "accepted"}

    def fake_pi(_config, location):
        action = location / "tools" / "submit-id"
        action.mkdir(parents=True)
        atomic_json(action / "request.json", {"tool": "vabench_submit"})
        atomic_json(action / "response.json", {"ok": True, "result": {"status": "submitted"}})
        atomic_json(location / "pi-outcome.json", {"termination_reason": "final"})
        (location / "model-budget.jsonl").write_text('{"usage":{}}\n', encoding="utf-8")
        return {"termination_reason": "final"}

    def fake_collect(_remote, _config, location):
        if calls.count("vabench-finalize") == 1 and not (location / "report.json").exists():
            atomic_json(location / "report.json", {"state": "pending"})
            return False
        atomic_json(
            location / "report.json",
            {
                "state": "verified",
                "result": {"execution": "ok", "verdict": verdict},
                "public_trace_sha256": "a" * 64,
                "final_archive_sha256": "b" * 64,
            },
        )
        return True

    monkeypatch.setattr(agent, "transport", lambda *_: FixtureServer())
    monkeypatch.setattr(agent, "run_pi", fake_pi)
    monkeypatch.setattr(agent, "collect_result", fake_collect)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "LOCAL_SECRET_SENTINEL")
    args = ["--config", str(config_path), "--evidence", str(evidence)]
    assert agent.main(["pi", *args]) == 1
    pending = json.loads((evidence / "results.jsonl").read_text(encoding="utf-8"))
    assert pending["status"] == "pending"
    assert pending["benchmark_success"] is None
    assert pending["task_identity_status"] == "matched"
    assert pending["session_sha256"] == "c" * 64
    budget = json.loads((evidence / "episode-budget.json").read_text())
    assert budget["model_settings"] == {"model": "fixture", "thinking": None}
    assert budget["experiment_settings"] == {
        "schema_version": 1,
        "agent": "pi",
        "runtime": "direct",
        "model": "fixture",
        "reasoning": {"parameter": "thinking", "value": None},
        "budgets": {
            "max_model_calls": 12,
            "max_request_bytes": 64000,
            "max_output_tokens": 4096,
            "episode_timeout_s": 600,
        },
    }
    wrong_config = tmp_path / "wrong-operator.json"
    wrong_config.write_text(json.dumps({**operator, "job_id": "other-job"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the recorded operator"):
        agent.main(["collect", "--config", str(wrong_config), "--evidence", str(evidence)])
    assert json.loads((evidence / "results.jsonl").read_text())["status"] == "pending"
    assert agent.main(["collect", *args]) == 0
    completed = json.loads((evidence / "results.jsonl").read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["benchmark_success"] is success
    assert completed["verdict"] == verdict
    assert completed["public_trace_sha256"] == "a" * 64
    assert completed["trajectory_refs"]["agent_outcome"] == "pi-outcome.json"
    assert completed["trajectory_refs"]["model_budget"] == "model-budget.jsonl"
    assert completed["tool_calls"] == 1
    assert completed["wall_elapsed_seconds"] >= 0
    summary = json.loads((evidence / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_attempts"] == 1
    assert summary["success_rate"] == int(success)
    assert calls == ["vabench-preflight", "vabench-finalize"]
    assert "LOCAL_SECRET_SENTINEL" not in "".join(
        path.read_text(encoding="utf-8") for path in evidence.rglob("*") if path.is_file()
    )


def test_new_record_rejects_existing_legacy_evidence_before_touching_it(
    config, tmp_path, monkeypatch
):
    from alphaapollo.workflows import chips_vabench_agent as agent

    config_path = tmp_path / "operator.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    evidence.chmod(0o700)
    (evidence / "operator.json").write_text('{"legacy":true}', encoding="utf-8")
    monkeypatch.setattr(agent, "transport", lambda *_: pytest.fail("must not contact server"))
    with pytest.raises(ValueError, match="fresh evidence"):
        agent.main(["pi", "--config", str(config_path), "--evidence", str(evidence)])
    assert (evidence / "operator.json").read_text(encoding="utf-8") == '{"legacy":true}'
    assert not (evidence / "task_manifest.json").exists()


def test_server_task_is_recorded_without_inventing_an_omitted_operator_task(
    config, tmp_path, monkeypatch
):
    from alphaapollo.workflows import chips_vabench_agent as agent

    config_path = tmp_path / "operator.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    evidence = tmp_path / "evidence"

    class FixtureServer:
        def cli(self, command, *args, **kwargs):
            assert command == "vabench-preflight"
            return {"state": "ready", "task_id": "v4-001", "session_sha256": "d" * 64}

    monkeypatch.setattr(agent, "transport", lambda *_: FixtureServer())
    monkeypatch.setattr(agent, "run_pi", lambda *_: {"termination_reason": "final"})
    assert agent.main(["pi", "--config", str(config_path), "--evidence", str(evidence)]) == 1
    row = json.loads((evidence / "results.jsonl").read_text(encoding="utf-8"))
    assert row["task_id"] is None
    assert row["server_task_id"] == "v4-001"
    assert row["task_identity_status"] == "server_reported"


def test_single_run_refuses_a_public_evidence_directory_before_writing(
    config, tmp_path, monkeypatch
):
    from alphaapollo.workflows import chips_vabench_agent as agent

    config_path = tmp_path / "operator.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o755)
    evidence.chmod(0o755)
    monkeypatch.setattr(agent, "transport", lambda *_: pytest.fail("must not contact server"))
    with pytest.raises(ValueError, match="private evidence directory"):
        agent.main(["pi", "--config", str(config_path), "--evidence", str(evidence)])
    assert not list(evidence.iterdir())
