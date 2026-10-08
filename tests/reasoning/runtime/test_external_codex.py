# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Codex session coverage, driven by a recorded real event stream.

``fixtures/codex_shell_stream.jsonl`` was captured from
``codex exec - --json`` (codex-cli 0.147.0). Thirty of its thirty-two startup
warnings were dropped, the remaining paths redacted, and the session identity
replaced with a fixed fixture value. Event structure is preserved; see
``fixtures/README.md`` for the publication boundary.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from alphaapollo.common.generation.base import SamplingOptions
from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
from alphaapollo.reasoning.runtime.external.agents.codex import CodexSession, parse_exec_json
from alphaapollo.reasoning.runtime.external.cli import CliRun
from alphaapollo.reasoning.runtime.external_agent_runtime import (
    ExternalAgentRuntime,
    project_outcome,
)

FIXTURE = Path(__file__).parent / "fixtures" / "codex_shell_stream.jsonl"
SAMPLING = SamplingOptions(temperature=1.0, max_tokens=1024)


@pytest.fixture(scope="module")
def recorded_stream() -> str:
    return FIXTURE.read_text()


def _task(system: str = "be terse") -> AgentTask:
    return AgentTask(task_id="task-1", system=system, prompt="compute 9/3*60 + 24")


class _Recorder:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.argv: list[str] = []
        self.kwargs: dict[str, Any] = {}

    def __call__(self, argv: list[str], **kwargs: Any) -> CliRun:
        self.argv = list(argv)
        self.kwargs = dict(kwargs)
        return CliRun(self.stdout, self.stderr, self.returncode, timed_out=False)


class _SequenceRecorder:
    def __init__(self, runs: list[CliRun]) -> None:
        self.runs = list(runs)
        self.calls = 0
        self.argvs: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> CliRun:
        self.argvs.append(list(argv))
        self.kwargs.append(dict(kwargs))
        run = self.runs[self.calls]
        self.calls += 1
        return run


class TestArgv:
    def test_isolation_flags_are_always_present(self) -> None:
        argv = CodexSession().argv()
        for flag in ("exec", "--json", "--ephemeral", "--ignore-user-config", "--ignore-rules"):
            assert flag in argv
        assert argv[2] == "-"  # prompt arrives on stdin

    def test_sandbox_defaults_to_read_only(self) -> None:
        argv = CodexSession().argv()
        assert argv[argv.index("-s") + 1] == "read-only"

    def test_model_and_config_overrides_are_forwarded(self) -> None:
        session = CodexSession(
            model="gpt-5.3-codex",
            sandbox="workspace-write",
            config_overrides=[
                "model_provider=vllm",
                'model_providers.vllm.base_url="http://127.0.0.1:8000/v1"',
            ],
        )
        argv = session.argv()
        assert argv[argv.index("-m") + 1] == "gpt-5.3-codex"
        assert argv[argv.index("-s") + 1] == "workspace-write"
        assert argv.count("-c") == 2
        assert 'model_providers.vllm.base_url="http://127.0.0.1:8000/v1"' in argv

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sandbox": "yolo"}, "sandbox must be one of"),
            ({"config_overrides": ["not-an-override"]}, "look like 'key=value'"),
            ({"mcp_auto_approve_tools": ["bash"]}, "must be a mapping"),
            ({"timeout_s": -1}, "timeout_s must be"),
            ({"model": ""}, "model must be"),
            ({"continuation": "yes"}, "continuation must be"),
            ({"eventless_error_retries": 4}, "between 0 and 3"),
            (
                {"raw_event_path": Path("/tmp/events.jsonl"), "eventless_error_retries": 1},
                "requires eventless_error_retries=0",
            ),
        ],
    )
    def test_rejects_invalid_settings(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            CodexSession(**kwargs)


class TestPromptHandling:
    def test_raw_events_are_private_and_visible_before_cli_exits(self, tmp_path: Path) -> None:
        marker = tmp_path / "child-ready"
        raw = tmp_path / "codex-events.jsonl"
        fake = tmp_path / "fake-codex"
        fake.write_text(
            f"#!{sys.executable}\n"
            "import json, time\n"
            "from pathlib import Path\n"
            "print(json.dumps({'type': 'thread.started', 'thread_id': 'fixture'}), flush=True)\n"
            f"Path({str(marker)!r}).write_text('ready')\n"
            "time.sleep(1)\n"
            "print(json.dumps({'type': 'turn.completed', 'usage': {}}), flush=True)\n",
            encoding="utf-8",
        )
        fake.chmod(0o700)
        session = CodexSession(cli=str(fake), raw_event_path=raw)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(session.run, _task(), workspace=tmp_path)
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists()
            assert raw.exists()
            assert '"type": "thread.started"' in raw.read_text()
            assert '"type": "turn.completed"' not in raw.read_text()
            assert future.result(timeout=5).termination_reason == "final"
        assert '"type": "turn.completed"' in raw.read_text()
        assert os.stat(raw).st_mode & 0o077 == 0

    def test_timeout_keeps_partial_raw_file_while_parsing_complete_records(
        self, tmp_path: Path
    ) -> None:
        raw = tmp_path / "codex-events.jsonl"
        fake = tmp_path / "slow-codex"
        fake.write_text(
            f"#!{sys.executable}\n"
            "import sys, time\n"
            'sys.stdout.write(\'{"type":"thread.started","thread_id":"fixture"}\\n{"type":\')\n'
            "sys.stdout.flush()\n"
            "time.sleep(5)\n",
            encoding="utf-8",
        )
        fake.chmod(0o700)
        outcome = CodexSession(cli=str(fake), raw_event_path=raw, timeout_s=1.5).run(
            _task(), workspace=tmp_path
        )
        assert outcome.termination_reason == "timeout"
        assert outcome.provider_metadata["dropped_partial_record"] is True
        assert raw.read_text().endswith('{"type":')

    def test_raw_stream_reaches_sink_before_parsing(self) -> None:
        raw = "not-json\n"
        seen = []
        recorder = _Recorder(stdout=raw)
        with pytest.raises(ValueError, match="not valid JSON"):
            CodexSession(runner=recorder, event_sink=seen.append).run(
                _task(), workspace=Path("/tmp")
            )
        assert seen == [raw]

    def test_timeout_sink_keeps_unfinished_last_record(self) -> None:
        raw = '{"type":"turn.started"}\n{"type":'
        seen = []

        def recorder(argv, **kwargs):
            return CliRun(raw, "", -1, timed_out=True)

        outcome = CodexSession(runner=recorder, event_sink=seen.append).run(
            _task(), workspace=Path("/tmp")
        )
        assert seen == [raw]
        assert outcome.termination_reason == "timeout"
        assert outcome.provider_metadata["dropped_partial_record"] is True

    def test_system_prompt_is_inlined_because_codex_has_no_system_flag(self) -> None:
        recorder = _Recorder(stdout='{"type":"turn.completed","usage":{}}')
        outcome = CodexSession(runner=recorder).run(_task(), workspace=Path("/tmp"))
        assert recorder.kwargs["prompt"] == "be terse\n\ncompute 9/3*60 + 24"
        assert "--system-prompt" not in recorder.argv
        assert outcome.provider_metadata["system_prompt_inlined"] is True

    def test_absent_system_prompt_is_reported_as_not_inlined(self) -> None:
        recorder = _Recorder(stdout='{"type":"turn.completed","usage":{}}')
        outcome = CodexSession(runner=recorder).run(_task(system=""), workspace=Path("/tmp"))
        assert recorder.kwargs["prompt"] == "compute 9/3*60 + 24"
        assert outcome.provider_metadata["system_prompt_inlined"] is False

    def test_workspace_is_the_working_directory(self, tmp_path: Path) -> None:
        recorder = _Recorder(stdout='{"type":"turn.completed","usage":{}}')
        CodexSession(runner=recorder).run(_task(), workspace=tmp_path)
        assert recorder.kwargs["workspace"] == tmp_path

    def test_continuation_resumes_the_exact_thread_without_repeating_system(
        self, tmp_path: Path
    ) -> None:
        thread_id = "019fffa2-a62c-72f2-9a1a-20d51bd18ece"
        stream = (
            json.dumps({"type": "thread.started", "thread_id": thread_id})
            + "\n"
            + json.dumps({"type": "turn.completed", "usage": {}})
        )
        recorder = _SequenceRecorder(
            [
                CliRun(stream, "", 0, timed_out=False),
                CliRun(stream, "", 0, timed_out=False),
            ]
        )
        session = CodexSession(
            continuation=True,
            model="gpt-5.6-sol",
            sandbox="workspace-write",
            runner=recorder,
        )

        first = session.run(_task(), workspace=tmp_path)
        second = session.run(
            AgentTask(
                task_id="task-2",
                system="be terse",
                prompt="revise using the feedback",
            ),
            workspace=tmp_path,
        )

        assert "--ephemeral" not in recorder.argvs[0]
        assert recorder.argvs[0][:3] == ["codex", "exec", "-"]
        assert recorder.argvs[1][recorder.argvs[1].index("resume") + 1] == thread_id
        assert "-s" not in recorder.argvs[1]
        assert recorder.kwargs[0]["prompt"].startswith("be terse\n\n")
        assert recorder.kwargs[1]["prompt"] == "revise using the feedback"
        assert first.provider_metadata["conversation_continued"] is False
        assert second.provider_metadata["conversation_continued"] is True


class TestParseRecordedStream:
    def test_command_execution_becomes_a_call_and_a_result(self, recorded_stream: str) -> None:
        outcome = parse_exec_json(recorded_stream)
        kinds = [event.kind for event in outcome.events]
        # The recorded run opens with an empty `agent_message`, which carries no
        # content and must not become a turn of its own.
        assert kinds == ["tool_call", "tool_result", "message"]

        call = outcome.events[0].tool_call
        assert call is not None
        assert call.name == "shell"
        assert "python3" in json.loads(call.arguments)["command"]
        result = outcome.events[1]
        assert result.call_id == call.id
        assert result.content.strip() == "204"
        assert result.failed is False

    def test_terminal_usage_and_thread_id_are_recorded(self, recorded_stream: str) -> None:
        outcome = parse_exec_json(recorded_stream)
        assert outcome.termination_reason == "final"
        assert outcome.final_text == "204"
        assert outcome.usage["output_tokens"] > 0
        assert outcome.provider_metadata["thread_id"]

    def test_startup_warnings_are_metadata_not_turns(self, recorded_stream: str) -> None:
        outcome = parse_exec_json(recorded_stream)
        assert outcome.provider_metadata["startup_warnings"] == 2
        assert len(outcome.provider_metadata["startup_warning_sample"]) == 2
        # Config noise must not become an observation, or it would invent turns.
        assert all(event.kind != "error" for event in outcome.events)

    def test_recorded_stream_projects_onto_a_two_turn_trajectory(
        self, recorded_stream: str
    ) -> None:
        result = project_outcome(
            _task(),
            parse_exec_json(recorded_stream),
            agent="codex",
            model="gpt-5.3-codex",
            sampling=SAMPLING,
        )
        assert [turn.index for turn in result.turns] == [0, 1]
        assert [call.name for call in result.turns[0].generation_response.tool_calls] == ["shell"]
        assert result.turns[0].environment_transition.observation.content.strip() == "204"
        assert result.turns[1].environment_transition.done is True
        assert result.metadata["trainable"] is False


class TestParseEdgeCases:
    def test_failed_command_marks_the_result(self) -> None:
        stream = json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "false",
                    "aggregated_output": "",
                    "exit_code": 1,
                    "status": "completed",
                },
            }
        )
        result = parse_exec_json(stream).events[1]
        assert result.failed is True

    def test_mcp_tool_call_keeps_server_and_tool_identity(self) -> None:
        stream = json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_2",
                    "type": "mcp_tool_call",
                    "server": "alphaapollo",
                    "tool": "bash",
                    "arguments": {"command": "echo 1"},
                    "result": {"content": [{"type": "text", "text": "1"}]},
                    "status": "completed",
                },
            }
        )
        call, result = parse_exec_json(stream).events
        assert call.tool_call is not None
        assert call.tool_call.name == "alphaapollo.bash"
        assert result.content == "1"

    def test_mcp_error_is_reported_as_a_failed_result(self) -> None:
        stream = json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_3",
                    "type": "mcp_tool_call",
                    "server": "alphaapollo",
                    "tool": "bash",
                    "error": {"message": "sandbox denied"},
                    "status": "failed",
                },
            }
        )
        result = parse_exec_json(stream).events[1]
        assert (result.failed, result.content) == (True, "sandbox denied")

    def test_turn_failed_is_an_external_error(self) -> None:
        stream = '{"type":"turn.failed","error":{"message":"model overloaded"}}'
        outcome = parse_exec_json(stream)
        assert outcome.termination_reason == "external_error"
        assert outcome.provider_metadata["error"] == "model overloaded"

    def test_missing_turn_completion_is_an_external_error(self) -> None:
        stream = '{"type":"thread.started","thread_id":"t-1"}'
        outcome = parse_exec_json(stream)
        assert outcome.termination_reason == "external_error"
        assert outcome.provider_metadata["missing_turn_completion"] is True

    def test_unknown_item_types_are_surfaced_rather_than_dropped(self) -> None:
        stream = "\n".join(
            [
                '{"type":"item.completed","item":{"id":"i","type":"todo_list","items":[]}}',
                '{"type":"telemetry","payload":{}}',
                '{"type":"turn.completed","usage":{}}',
            ]
        )
        unhandled = parse_exec_json(stream).provider_metadata["unhandled_event_types"]
        assert unhandled == ["item:todo_list", "telemetry"]

    def test_malformed_json_fails_loudly_with_the_line_number(self) -> None:
        with pytest.raises(ValueError, match="line 2 is not valid JSON"):
            parse_exec_json('{"type":"turn.completed","usage":{}}\nnope\n')


class TestProcessFailures:
    def test_explicit_retry_replays_only_an_eventless_external_error(self) -> None:
        recorder = _SequenceRecorder(
            [
                CliRun(
                    '{"type":"turn.failed","error":{"message":"service unavailable"}}',
                    "",
                    1,
                    timed_out=False,
                ),
                CliRun(
                    '{"type":"turn.completed","usage":{}}',
                    "",
                    0,
                    timed_out=False,
                ),
            ]
        )

        outcome = CodexSession(eventless_error_retries=2, runner=recorder).run(
            _task(), workspace=Path("/tmp")
        )

        assert outcome.termination_reason == "final"
        assert outcome.provider_metadata["eventless_error_retries"] == 1
        assert recorder.calls == 2

    def test_explicit_retry_never_replays_after_an_action_event(self) -> None:
        stream = "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item-1",
                            "type": "command_execution",
                            "command": "touch candidate",
                            "aggregated_output": "",
                            "exit_code": 0,
                            "status": "completed",
                        },
                    }
                ),
                '{"type":"turn.failed","error":{"message":"connection dropped"}}',
            ]
        )
        recorder = _SequenceRecorder([CliRun(stream, "", 1, timed_out=False)])

        outcome = CodexSession(eventless_error_retries=2, runner=recorder).run(
            _task(), workspace=Path("/tmp")
        )

        assert outcome.termination_reason == "external_error"
        assert recorder.calls == 1

    def test_non_zero_exit_overrides_the_stream(self) -> None:
        recorder = _Recorder(
            stdout='{"type":"turn.completed","usage":{}}',
            stderr="stream error: unauthorized",
            returncode=1,
        )
        outcome = CodexSession(runner=recorder).run(_task(), workspace=Path("/tmp"))
        assert outcome.termination_reason == "external_error"
        assert "unauthorized" in outcome.provider_metadata["stderr"]

    def test_timeout_keeps_the_partial_stream(self, recorded_stream: str) -> None:
        # A SIGKILL lands wherever the CLI's stdout buffer was, so the stream ends
        # mid-record. The complete records must survive that fragment.
        lines = recorded_stream.splitlines()
        partial = "\n".join(lines[:5]) + "\n" + lines[5][:40]

        def timing_out(argv: list[str], **kwargs: Any) -> CliRun:
            del argv, kwargs
            return CliRun(partial, "", -1, timed_out=True)

        outcome = CodexSession(timeout_s=1.0, runner=timing_out).run(
            _task(), workspace=Path("/tmp")
        )
        assert outcome.termination_reason == "timeout"
        assert outcome.provider_metadata["partial_stream"] is True
        assert outcome.provider_metadata["dropped_partial_record"] is True
        assert outcome.provider_metadata["thread_id"] == "00000000-0000-4000-8000-000000000001"
        assert outcome.provider_metadata["startup_warnings"] == 2


def test_runtime_drives_a_recorded_session_end_to_end(recorded_stream: str, tmp_path: Path) -> None:
    runtime = ExternalAgentRuntime(
        lambda: CodexSession(runner=_Recorder(stdout=recorded_stream)),
        agent="codex",
        model="gpt-5.3-codex",
        workspace_root=tmp_path,
    )
    results = runtime.run_batch([_task()])
    assert results[0].final_text == "204"
    assert results[0].metadata["policy_source"] == "external_agent"


@pytest.mark.skipif(
    os.environ.get("ALPHAAPOLLO_LIVE_CODEX") != "1" or shutil.which("codex") is None,
    reason="live Codex run requires ALPHAAPOLLO_LIVE_CODEX=1 and the codex CLI",
)
def test_live_codex_solves_a_tool_using_task(tmp_path: Path) -> None:
    runtime = ExternalAgentRuntime(
        lambda: CodexSession(sandbox="workspace-write", timeout_s=300.0),
        agent="codex",
        model="default",
        workspace_root=tmp_path,
    )
    task = AgentTask(
        task_id="live-codex-1",
        system="Use the shell to compute. End with exactly: Final answer: <number>",
        prompt="Compute 9/3*60 + 24 exactly using python3 and report the integer result.",
    )
    result = runtime.run(task)

    assert result.termination_reason == "final"
    assert "204" in result.final_text
    assert any(turn.generation_response.tool_calls for turn in result.turns)
    assert all(turn.generation_response.is_trainable is False for turn in result.turns)
