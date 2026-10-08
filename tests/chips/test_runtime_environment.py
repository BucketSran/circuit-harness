"""Runtime contract fixtures, not model or simulator acceptance."""

import json
import os
import sys

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from alphaapollo.common.execution.chips import analog_session, vabench_session
from alphaapollo.common.execution.chips.journal import atomic_json, file_digest
from alphaapollo.common.generation.base import GenerationResponse, ToolCall
from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
from alphaapollo.reasoning.runtime.alphaapollo_agent_runtime import AlphaApolloAgentRuntime
from alphaapollo.reasoning.runtime.external.bridge.mcp_server import _local_tool_bridge, _tool_specs
from alphaapollo.reasoning.runtime.external_agent_runtime import (
    ExternalAgentRuntime,
    ExternalEvent,
    ExternalRunOutcome,
)


@pytest.mark.parametrize("session", [vabench_session, analog_session])
def test_shared_catalog_preserves_task_tool_contract(session):
    expected = session.tool_schemas()
    names = tuple(tool["function"]["name"] for tool in expected)
    assert [spec.to_openai_tool() for spec in _tool_specs(names)] == expected


def test_session_tools_require_an_environment_before_local_resources_start():
    names = ("vabench_read", "bash")
    with pytest.raises(ValueError, match="environment-socket"):
        _local_tool_bridge(names, specs=_tool_specs(names), max_tool_calls=4)


@pytest.mark.parametrize("session", [vabench_session, analog_session])
def test_real_mcp_process_discovers_only_granted_task_tools(session, tmp_path):
    # Discovery requires no simulator or live endpoint; calls will require the
    # Runtime-owned Environment endpoint, never a new local task authority.
    expected = [tool["function"] for tool in session.tool_schemas()]
    names = [tool["name"] for tool in expected]

    async def discover():
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "alphaapollo.reasoning.runtime.external.bridge.mcp_server",
                "--tools",
                ",".join(names),
                "--environment-socket",
                str(tmp_path / "endpoint.json"),
            ],
            env={"PATH": os.environ["PATH"]},
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                result = await client.list_tools()
                return [
                    {"name": t.name, "description": t.description, "parameters": t.inputSchema}
                    for t in result.tools
                ]

    observed = anyio.run(discover)
    assert observed == expected


@pytest.fixture(params=["vabench", "analog"])
def task_session(request, tmp_path, monkeypatch):
    """Real task read/write/freeze; simulation feedback is a labelled fixture."""
    kind = request.param
    module = vabench_session if kind == "vabench" else analog_session
    content = ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"

    def create(name):
        root = tmp_path / name
        public = root / "public"
        public.mkdir(parents=True)
        instruction = public / ("task/instruction.md" if kind == "vabench" else "instruction.md")
        instruction.parent.mkdir(exist_ok=True)
        instruction.write_text("Read, repair and submit the public candidate.")
        (root / "hidden.txt").write_text("PRIVATE_CANARY")
        config = {"max_actions": 24, "max_simulations": 4, "timeout_s": 5}
        if kind == "vabench":
            (public / "submission").mkdir()
            config.update(pin={}, artifacts=["dut.va"], public_files={})
            monkeypatch.setattr(vabench_session, "verify_pin", lambda pin: None)

            def worker(directory, config, action):
                candidate = (directory / "public/submission/dut.va").read_text()
                atomic_json(
                    action / "feedback.json",
                    {"state": "error" if "* bad" in candidate else "simulated"},
                )

            monkeypatch.setattr(vabench_session, "_run_public_worker", worker)
        else:
            config.update(
                source_root=str(root),
                public_files={"instruction.md": file_digest(instruction)},
                podman="fixture",
                podman_root=None,
                podman_runroot=None,
                runtime_image=None,
                offline_image_archive=None,
                podman_single_id=False,
                podman_no_cpu_limit=False,
            )

            def simulate(source, candidate, output, **kwargs):
                output.mkdir(parents=True)
                return {
                    "state": "error" if "* bad" in candidate.read_text() else "simulated",
                    "measurements": {},
                    "missing_measurements": [],
                    "diagnostic_excerpt": "fixture",
                }

            monkeypatch.setattr(analog_session, "run_public_rlc", simulate)
        atomic_json(root / "session.json", config)
        return root

    def actions():
        def write(candidate):
            return (
                {"path": "dut.va", "content": candidate}
                if kind == "vabench"
                else {"content": candidate}
            )

        return [
            (
                f"{kind}_read",
                {"path": "task/instruction.md" if kind == "vabench" else "instruction.md"},
            ),
            (f"{kind}_read", {"path": "../hidden.txt"}),
            (f"{kind}_write", write(content + "* bad\n")),
            (f"{kind}_simulate", {}),
            (f"{kind}_write", write(content)),
            (f"{kind}_simulate", {}),
            (f"{kind}_submit", {}),
        ]

    class Transport:
        def __init__(self, root):
            self.root = root
            self.replies = []

        def call(self, tool, arguments, *, action_id, **kwargs):
            reply = module.session_action(
                self.root, {"id": action_id, "tool": tool, "arguments": arguments}
            )
            self.replies.append(reply)
            return reply

    return kind, module, create, actions(), Transport, content


@pytest.mark.parametrize("runtime_kind", ["native", "external"])
def test_both_runtimes_preserve_session_actions_and_freeze(task_session, runtime_kind, tmp_path):
    from alphaapollo.common.environment.chips import ChipsEnvironment

    kind, module, create, actions, Transport, candidate = task_session
    baseline = Transport(create("baseline"))
    for index, (name, arguments) in enumerate(actions):
        baseline.call(name, arguments, action_id=f"baseline-{index}")
    remote = Transport(create(runtime_kind))
    environment = ChipsEnvironment(remote, task_kind=kind)
    requests = []
    observed = []
    events = []

    class Backend:
        def generate_batch(self, batch):
            assert len(batch) == 1
            request = batch[0]
            assert list(request.tools) == module.tool_schemas()
            index = len(requests)
            requests.append(request)
            name, arguments = actions[index]
            return [
                GenerationResponse(
                    request_id=request.request_id,
                    group_id=request.group_id,
                    sample_id=request.sample_id,
                    content="",
                    tool_calls=(
                        ToolCall(id=f"call-{index}", name=name, arguments=json.dumps(arguments)),
                    ),
                )
            ]

    class Session:
        def run(self, task, *, workspace):
            assert task.task_payload == {}
            assert "PRIVATE_CANARY" not in repr(task)

            async def run():
                params = StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "alphaapollo.reasoning.runtime.external.bridge.mcp_server",
                        "--tools",
                        ",".join(module.TOOLS),
                        "--environment-socket",
                        str(workspace / ".alphaapollo-environment.json"),
                    ],
                    env={"PATH": os.environ["PATH"]},
                )
                async with stdio_client(params) as (reader, writer):
                    async with ClientSession(reader, writer) as client:
                        await client.initialize()
                        for index, (name, arguments) in enumerate(actions):
                            response = await client.call_tool(name, arguments)
                            observed.append(json.loads(response.content[0].text))
                            call = ToolCall(
                                id=f"cli-{index}", name=name, arguments=json.dumps(arguments)
                            )
                            events.extend(
                                (
                                    ExternalEvent(kind="tool_call", tool_call=call),
                                    ExternalEvent(
                                        kind="tool_result",
                                        call_id=call.id,
                                        tool_id=name,
                                        content=response.content[0].text,
                                        failed=response.isError,
                                    ),
                                )
                            )

            anyio.run(run)
            return ExternalRunOutcome(final_text="Submitted.", events=tuple(events))

        def close(self):
            pass

    if runtime_kind == "native":
        runtime = AlphaApolloAgentRuntime(
            Backend(),
            model="scripted_fixture",
            environment_factory=lambda task: environment,
            tools=module.tool_schemas(),
            max_turns=8,
        )
    else:
        runtime = ExternalAgentRuntime(
            Session,
            agent="scripted_fixture",
            model="scripted_fixture",
            environment_factory=lambda task: environment,
            workspace_root=tmp_path / "workspace",
        )
    task = AgentTask(
        task_id="fixture",
        system="Use public tools.",
        prompt="Solve public task.",
        tools=tuple(module.TOOLS),
        task_payload={"hidden": "PRIVATE_CANARY"},
    )
    try:
        result = runtime.run_batch([task])[0]
    finally:
        if runtime_kind == "external":
            runtime.close()

    assert remote.replies == baseline.replies
    assert result.termination_reason == "submitted"
    transition = result.turns[-1].environment_transition
    assert transition.reward == 0
    assert transition.success is None
    frozen = remote.root / ("candidate/dut.va" if kind == "vabench" else "frozen/circuit.spi")
    assert frozen.read_text() == candidate
    assert len(list((remote.root / "actions").iterdir())) == len(actions)
    if runtime_kind == "external":
        assert observed == remote.replies
    else:
        for index, request in enumerate(requests[1:]):
            assert request.messages[-1]["role"] == "tool"
            assert request.messages[-1]["tool_call_id"] == f"call-{index}"
            assert json.loads(request.messages[-1]["content"]) == remote.replies[index]
        assert "PRIVATE_CANARY" not in repr(requests)


def test_environment_rejects_ungranted_and_malformed_calls_before_dispatch(task_session):
    from alphaapollo.common.environment.base import EnvironmentContext
    from alphaapollo.common.environment.chips import ChipsEnvironment

    kind, _, create, _, Transport, _ = task_session
    remote = Transport(create("invalid"))
    environment = ChipsEnvironment(remote, task_kind=kind)
    environment.init(EnvironmentContext(session_id="fixture", actor="solver"))
    for name, arguments, code in (
        ("bash", {"command": "cat ../hidden.txt"}, "unknown_tool"),
        (kind + "_read", {"path": 1}, "invalid_arguments"),
        (kind + "_read", {"path": "", "extra": "bad"}, "invalid_arguments"),
    ):
        transition = environment.step(
            {"id": "call", "function": {"name": name, "arguments": arguments}}
        )
        assert transition.raw_observation["error"]["code"] == code
        assert not transition.done
    assert remote.replies == []
    assert not (remote.root / "actions").exists()


def test_multicall_rejection_preserves_all_ids_and_reasoning_without_executing(task_session):
    from alphaapollo.common.environment.base import EnvironmentContext
    from alphaapollo.common.environment.chips import ChipsEnvironment

    kind, _, create, _, Transport, _ = task_session
    remote = Transport(create("multi"))
    environment = ChipsEnvironment(remote, task_kind=kind)
    environment.init(EnvironmentContext(session_id="fixture", actor="solver"))
    response = GenerationResponse(
        request_id="r",
        content="Inspect files.",
        reasoning_content="public fixture reasoning",
        tool_calls=(
            ToolCall(id="one", name=kind + "_read", arguments='{"path":""}'),
            ToolCall(id="two", name=kind + "_submit", arguments="{}"),
        ),
    )
    transition = environment.step(environment.project_response(response))
    assert transition.raw_observation["error"]["code"] == "multiple_tool_calls"
    assert not transition.done
    messages = environment.continuation_messages(response, transition)
    assert messages[0]["reasoning_content"] == response.reasoning_content
    assert [m["tool_call_id"] for m in messages[1:]] == ["one", "two"]
    assert all(json.loads(m["content"])["ok"] is False for m in messages[1:])
    assert remote.replies == []


@pytest.mark.parametrize("ending", ["text", "submit_incomplete", "close"])
def test_environment_never_infers_submission(task_session, ending):
    from alphaapollo.common.environment.base import EnvironmentContext
    from alphaapollo.common.environment.chips import ChipsEnvironment

    kind, _, create, _, Transport, _ = task_session
    remote = Transport(create("unfinished"))
    environment = ChipsEnvironment(remote, task_kind=kind)
    environment.init(EnvironmentContext(session_id="fixture", actor="solver"))
    if ending == "text":
        transition = environment.step("Finished! All tests pass.")
        assert transition.done
        assert transition.termination_reason == "final_response_without_submission"
        assert transition.success is None
    elif ending == "submit_incomplete":
        transition = environment.step(
            {"id": "submit", "function": {"name": kind + "_submit", "arguments": {}}}
        )
        assert not transition.done
    environment.close()
    environment.close()
    with pytest.raises(RuntimeError, match="not active"):
        environment.step("Another action")
    assert not (remote.root / "frozen.json").exists()


def test_runtime_bridge_close_preserves_detached_job_and_pending_action(tmp_path):
    """Real processes/loopback; the delayed solver is a constructed fixture."""
    import threading
    import time

    from alphaapollo.common.environment.base import EnvironmentContext
    from alphaapollo.common.environment.chips import ChipsEnvironment
    from alphaapollo.common.execution.chips.vabench_remote import LocalVabench
    from alphaapollo.reasoning.runtime.external.bridge.environment_socket import (
        EnvironmentSocketToolBridge,
        RuntimeEnvironmentSocket,
    )

    bundle = tmp_path / "fixture.py"
    bundle.write_text(
        "import json, pathlib, subprocess, sys, time\n"
        "root=pathlib.Path(__file__).parent\n"
        "if sys.argv[1] == 'vabench-request':\n"
        "    if not (root/'accepted').exists():\n"
        "        child=subprocess.Popen([sys.executable, '-c', "
        "\"import pathlib,sys,time;time.sleep(1.5);pathlib.Path(sys.argv[1]).write_text('done')\", "
        "str(root/'job-done')], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL, start_new_session=True)\n"
        "        (root/'accepted').write_text(str(child.pid))\n"
        "    print(json.dumps({'state':'accepted'}))\n"
        "else:\n"
        "    (root/'query-started').touch()\n"
        "    if not (root/'job-done').exists(): time.sleep(5)\n"
        "    print(json.dumps({'ok':True,'result':{'status':'simulated'}}))\n"
    )
    config = {"python": sys.executable, "bundle": str(bundle), "session": str(tmp_path)}
    remote = LocalVabench(config, tmp_path / "tools")
    environment = ChipsEnvironment(remote, task_kind="vabench")
    environment.init(EnvironmentContext(session_id="fixture", actor="solver"))
    endpoint = tmp_path / "endpoint.json"
    owner = RuntimeEnvironmentSocket(environment, endpoint)
    replies = []
    client = EnvironmentSocketToolBridge(endpoint)
    thread = threading.Thread(
        target=lambda: replies.append(
            client.dispatch(
                {
                    "id": "during-simulation",
                    "function": {"name": "vabench_simulate", "arguments": {}},
                },
                None,
            ).observation_payload()
        ),
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 3
    while not (tmp_path / "query-started").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert (tmp_path / "query-started").exists()
        assert not (tmp_path / "job-done").exists()
        owner.close()
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert replies[0]["error"] == "wait_interrupted"
        assert replies[0]["server_execution"] == "unknown"
        assert owner.transitions[-1].termination_reason == "awaiting_action_recovery"
    finally:
        remote.interrupt_wait()
        owner.close()
        environment.close()
    deadline = time.monotonic() + 3
    while not (tmp_path / "job-done").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (tmp_path / "job-done").read_text() == "done"
    worker = (tmp_path / "accepted").read_text()
    recovered = LocalVabench(config, tmp_path / "tools")
    assert recovered.pending_action_id
    reply = recovered.call("vabench_simulate", {}, action_id=recovered.pending_action_id)
    assert reply["ok"]
    assert (tmp_path / "accepted").read_text() == worker


@pytest.mark.skipif(
    not os.environ.get("CHIPS_TEST_PI_CLI"),
    reason="set CHIPS_TEST_PI_CLI for real Pi/local HTTP fixture",
)
@pytest.mark.parametrize("stop", [None, "calls", "bytes", "schema", "length"])
def test_real_pi_cli_uses_shared_runtime_for_repair_and_submission(
    task_session, tmp_path, monkeypatch, stop
):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from alphaapollo.workflows import chips_analog_agent, chips_vabench_agent

    kind, _, create, actions, Transport, _ = task_session
    if stop == "schema":
        # Pi rejects this before execute/MCP; the next model request repairs it.
        actions = [(f"{kind}_read", {})] + actions
    module = chips_vabench_agent if kind == "vabench" else chips_analog_agent
    remote = Transport(create("real-pi-fixture"))
    monkeypatch.setattr(module, "transport", lambda *args: remote)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(requests)
            requests.append(body)
            if stop == "length":
                delta, finish = {"role": "assistant", "content": "Incomplete plan."}, "length"
            elif index < len(actions):
                name, arguments = actions[index]
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"fixture-{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                delta, finish = {"role": "assistant", "content": "Submitted."}, "stop"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for part, reason in ((delta, None), ({}, finish)):
                chunk = {
                    "id": f"chat-{index}",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture-model",
                    "choices": [{"index": 0, "delta": part, "finish_reason": reason}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = {
        "transport": "local",
        "python": sys.executable,
        "bundle": "/private/unused.pyz",
        "session": str(remote.root),
        "pi_cli": os.environ["CHIPS_TEST_PI_CLI"],
        "model": "glm-5.3-flash",
        "thinking": "low",
        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "policy_kind": "scripted_http_fixture",
        "runtime": "external",
        "http_retry_limit": 0,
        "episode_timeout_s": 60,
        "max_model_calls": 1 if stop == "calls" else 12,
        "max_request_bytes": 1 if stop == "bytes" else 64000,
    }
    monkeypatch.setenv("CHIPS_MODEL_KEY", "TEST_ONLY_NOT_A_REAL_KEY")
    evidence = tmp_path / "real-pi-evidence"
    try:
        outcome = module.run_pi(config, evidence)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    for index, payload in enumerate(requests):
        notices = [
            m
            for m in payload["messages"]
            if m["role"] == "user" and "Harness budget" in json.dumps(m)
        ]
        assert len(notices) == 1
        notice = json.dumps(notices[0])
        assert f"request {index + 1} of {config['max_model_calls']}" in notice
        assert f"including this request: {config['max_model_calls'] - index}" in notice
    assert (evidence / "agent-input.json").is_file()
    if stop == "length":
        assert outcome["termination_reason"] == "truncated"
        assert outcome["harness_termination_reason"] == "output_token_limit"
        assert len(requests) == 1
        assert remote.replies == []
        return
    if stop in ("calls", "bytes"):
        assert outcome["termination_reason"] != "final"
        assert outcome["harness_termination_reason"] == (
            "model_request_limit" if stop == "calls" else "request_bytes_limit"
        )
        assert len(requests) == (1 if stop == "calls" else 0)
        assert not (remote.root / "frozen.json").exists()
        budget = [
            json.loads(line) for line in (evidence / "model-budget.jsonl").read_text().splitlines()
        ]
        assert budget[-1]["event"] == "budget_stop"
        return
    assert outcome["termination_reason"] == "final"
    assert requests[0]["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert requests[0]["reasoning_effort"] == "low"
    assert len(requests) == len(actions) + 1
    assert [t["function"]["name"] for t in requests[0]["tools"]] == list(
        module.tool_schemas()[i]["function"]["name"] for i in range(4)
    )
    assert "PRIVATE_CANARY" not in json.dumps(requests)
    tool_contents = [m["content"] for m in requests[-1]["messages"] if m["role"] == "tool"]
    if stop == "schema":
        assert tool_contents[0].startswith(f'Validation failed for tool "{kind}_read":')
        tool_contents = tool_contents[1:]
    tool_replies = [json.loads(content) for content in tool_contents]
    assert tool_replies == remote.replies
    assert tool_replies[1]["ok"] is False
    assert tool_replies[3]["result"]["state"] == "error"
    assert tool_replies[5]["result"]["state"] == "simulated"
    normalized = json.loads((evidence / "runtime-result.json").read_text())
    assert normalized["termination_reason"] == "submitted"
    assert len(normalized["turns"]) == len(actions)
    if stop == "schema":
        rejected = normalized["turns"][0]["environment_transition"]
        assert rejected["metadata"]["external_environment"]["environment_stepped"] is False
        assert rejected["observation"]["error_stage"] == "external_tool_validation"
        assert not rejected["env_action_valid"]
    assert (evidence / "pi-events.jsonl").stat().st_size > 0
    assert (remote.root / "frozen.json").is_file()


def test_both_task_sessions_preserve_a_submission_slot(task_session):
    kind, module, create, actions, _, _ = task_session
    root = create("budget-contract")
    config = json.loads((root / "session.json").read_text())
    atomic_json(root / "session.json", {**config, "budget_feedback": True, "max_actions": 2})
    tool, args = next((tool, args) for tool, args in actions if tool.endswith("_write"))
    reply = module.session_action(root, {"id": "write", "tool": tool, "arguments": args})
    assert reply["budget"]["actions_remaining"] == 1
    blocked = module.session_action(
        root, {"id": "read", "tool": kind + "_read", "arguments": {"path": ""}}
    )
    assert blocked["error"] == "action_reserved_for_submission"
    submitted = module.session_action(
        root, {"id": "submit", "tool": kind + "_submit", "arguments": {}}
    )
    assert submitted["budget"]["actions_remaining"] == 0
    assert (root / "frozen.json").is_file()
