# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Claude Code session coverage, driven by a recorded real event stream.

The fixture in ``fixtures/claude_code_bash_stream.jsonl`` was captured from
``claude -p --output-format stream-json`` (CLI 2.1.231) and then trimmed of
machine-specific fields. Identities and reasoning text now use fixed fixture
values; event structure is preserved. See ``fixtures/README.md``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from alphaapollo.common.generation.base import SamplingOptions
from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
from alphaapollo.reasoning.runtime.external.agents.claude_code import (
    ClaudeCodeSession,
    parse_stream_json,
)
from alphaapollo.reasoning.runtime.external.cli import CliRun
from alphaapollo.reasoning.runtime.external_agent_runtime import (
    ExternalAgentRuntime,
    project_outcome,
)

FIXTURE = Path(__file__).parent / "fixtures" / "claude_code_bash_stream.jsonl"
SAMPLING = SamplingOptions(temperature=1.0, max_tokens=1024)


@pytest.fixture(scope="module")
def recorded_stream() -> str:
    return FIXTURE.read_text()


def _task() -> AgentTask:
    return AgentTask(task_id="task-1", system="be terse", prompt="print the boot id")


class _Recorder:
    """Stand in for ``subprocess.run`` and remember exactly how it was called."""

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


class TestArgv:
    def test_isolation_flags_are_always_present(self) -> None:
        argv = ClaudeCodeSession().argv(_task())
        for flag in ("-p", "--safe-mode", "--no-session-persistence", "--verbose"):
            assert flag in argv
        assert argv[argv.index("--output-format") + 1] == "stream-json"

    def test_system_prompt_replaces_rather_than_appends(self) -> None:
        argv = ClaudeCodeSession().argv(_task())
        assert argv[argv.index("--system-prompt") + 1] == "be terse"
        assert "--append-system-prompt" not in argv

    def test_optional_settings_are_forwarded(self) -> None:
        session = ClaudeCodeSession(
            model="sonnet",
            tools=["Bash", "Read"],
            permission_mode="bypassPermissions",
            max_budget_usd=0.5,
        )
        argv = session.argv(_task())
        assert argv[argv.index("--model") + 1] == "sonnet"
        assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"
        assert argv[argv.index("--max-budget-usd") + 1] == "0.5"
        assert argv[argv.index("--tools") + 1] == "Bash,Read"

    def test_variadic_tools_flag_is_last_so_it_cannot_swallow_another_flag(self) -> None:
        session = ClaudeCodeSession(tools=[], extra_args=["--add-dir", "/workspace"])
        argv = session.argv(_task())
        assert argv[-2:] == ["--tools", ""]

    def test_safe_mode_can_be_disabled_explicitly(self) -> None:
        assert "--safe-mode" not in ClaudeCodeSession(safe_mode=False).argv(_task())

    def test_prompt_is_not_placed_on_the_argument_vector(self) -> None:
        recorder = _Recorder(stdout='{"type":"result","subtype":"success","result":"ok"}')
        ClaudeCodeSession(runner=recorder).run(_task(), workspace=Path("/tmp"))
        assert "print the boot id" not in recorder.argv
        assert recorder.kwargs["prompt"] == "print the boot id"

    def test_workspace_is_the_working_directory(self, tmp_path: Path) -> None:
        recorder = _Recorder(stdout='{"type":"result","subtype":"success","result":"ok"}')
        ClaudeCodeSession(runner=recorder).run(_task(), workspace=tmp_path)
        assert recorder.kwargs["workspace"] == tmp_path


class TestConstruction:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"cli": " "}, "cli must be"),
            ({"model": ""}, "model must be"),
            ({"timeout_s": 0}, "timeout_s must be"),
            ({"tools": "Bash"}, "tools must be a sequence"),
            ({"permission_mode": ""}, "permission_mode must be"),
            ({"max_budget_usd": -1}, "max_budget_usd must be"),
            ({"safe_mode": "yes"}, "safe_mode must be a bool"),
        ],
    )
    def test_rejects_invalid_settings(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            ClaudeCodeSession(**kwargs)

    def test_closed_session_refuses_to_run(self) -> None:
        session = ClaudeCodeSession()
        session.close()
        with pytest.raises(RuntimeError, match="is closed"):
            session.run(_task(), workspace=Path("/tmp"))


class TestParseRecordedStream:
    def test_events_match_the_recorded_agent_behaviour(self, recorded_stream: str) -> None:
        outcome = parse_stream_json(recorded_stream)
        assert [event.kind for event in outcome.events] == [
            "reasoning",
            "tool_call",
            "tool_result",
            "message",
        ]
        call = outcome.events[1].tool_call
        assert call is not None
        assert call.name == "Bash"
        assert json.loads(call.arguments)["command"] == "cat /proc/sys/kernel/random/boot_id"
        assert outcome.events[2].call_id == call.id
        assert outcome.events[2].failed is False

    def test_result_event_supplies_the_final_text_and_usage(self, recorded_stream: str) -> None:
        outcome = parse_stream_json(recorded_stream)
        assert outcome.termination_reason == "final"
        assert outcome.final_text == "00000000-0000-4000-8000-000000000000"
        assert outcome.usage["output_tokens"] > 0

    def test_init_event_supplies_reproducibility_metadata(self, recorded_stream: str) -> None:
        metadata = parse_stream_json(recorded_stream).provider_metadata
        assert metadata["cli_version"] == "2.1.231"
        assert metadata["permission_mode"] == "bypassPermissions"
        assert metadata["available_tools"] == ["Bash"]
        assert metadata["session_id"]

    def test_recorded_stream_projects_onto_a_two_turn_trajectory(
        self, recorded_stream: str
    ) -> None:
        result = project_outcome(
            _task(),
            parse_stream_json(recorded_stream),
            agent="claude_code",
            model="claude-sonnet-5",
            sampling=SAMPLING,
        )
        assert [turn.index for turn in result.turns] == [0, 1]
        assert [call.name for call in result.turns[0].generation_response.tool_calls] == ["Bash"]
        assert result.turns[0].environment_transition.observation.content.startswith("00000000")
        assert result.turns[1].environment_transition.done is True
        assert result.metadata["trainable"] is False


class TestParseEdgeCases:
    def test_missing_result_event_is_reported_as_an_external_error(self) -> None:
        outcome = parse_stream_json('{"type":"assistant","message":{"content":[]}}')
        assert outcome.termination_reason == "external_error"
        assert outcome.provider_metadata["missing_result_event"] is True

    def test_max_turns_stop_is_truncation_not_failure(self) -> None:
        stream = '{"type":"result","subtype":"error_max_turns","is_error":true,"result":"partial"}'
        assert parse_stream_json(stream).termination_reason == "truncated"

    def test_execution_error_is_an_external_error(self) -> None:
        stream = '{"type":"result","subtype":"error_during_execution","is_error":true}'
        outcome = parse_stream_json(stream)
        assert outcome.termination_reason == "external_error"
        assert outcome.final_text == ""

    def test_unknown_event_types_are_surfaced_rather_than_dropped(self) -> None:
        stream = "\n".join(
            [
                '{"type":"telemetry_v2","payload":{}}',
                '{"type":"assistant","message":{"content":[{"type":"chart","data":1}]}}',
                '{"type":"result","subtype":"success","result":"ok"}',
            ]
        )
        unhandled = parse_stream_json(stream).provider_metadata["unhandled_event_types"]
        assert unhandled == ["assistant_block:chart", "telemetry_v2"]

    def test_string_content_is_accepted_as_a_text_block(self) -> None:
        stream = '{"type":"assistant","message":{"content":"hello"}}'
        events = parse_stream_json(stream).events
        assert [(event.kind, event.content) for event in events] == [("message", "hello")]

    def test_structured_tool_result_blocks_are_flattened(self) -> None:
        stream = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "line"}],
                            "is_error": True,
                        }
                    ]
                },
            }
        )
        event = parse_stream_json(stream).events[0]
        assert (event.content, event.failed, event.call_id) == ("line", True, "toolu_1")

    def test_malformed_json_fails_loudly_with_the_line_number(self) -> None:
        with pytest.raises(ValueError, match="line 2 is not valid JSON"):
            parse_stream_json('{"type":"result","subtype":"success"}\nnot-json\n')

    def test_blank_lines_are_ignored(self) -> None:
        outcome = parse_stream_json('\n\n{"type":"result","subtype":"success","result":"ok"}\n\n')
        assert outcome.final_text == "ok"


class TestProcessFailures:
    def test_non_zero_exit_overrides_an_apparently_successful_stream(self) -> None:
        recorder = _Recorder(
            stdout='{"type":"result","subtype":"success","result":"ok"}',
            stderr="fatal: credentials missing",
            returncode=1,
        )
        outcome = ClaudeCodeSession(runner=recorder).run(_task(), workspace=Path("/tmp"))
        assert outcome.termination_reason == "external_error"
        assert outcome.provider_metadata["returncode"] == 1
        assert "credentials missing" in outcome.provider_metadata["stderr"]

    def test_timeout_keeps_the_partial_stream(self, recorded_stream: str) -> None:
        # A SIGKILL lands wherever the CLI's stdout buffer was, so the stream ends
        # mid-record. The complete records must survive that fragment.
        lines = recorded_stream.splitlines()
        partial = "\n".join(lines[:5]) + "\n" + lines[5][:40]

        def timing_out(argv: list[str], **kwargs: Any) -> CliRun:
            del argv, kwargs
            return CliRun(partial, "", -1, timed_out=True)

        outcome = ClaudeCodeSession(timeout_s=1.0, runner=timing_out).run(
            _task(), workspace=Path("/tmp")
        )
        assert outcome.termination_reason == "timeout"
        assert outcome.provider_metadata["partial_stream"] is True
        assert outcome.provider_metadata["dropped_partial_record"] is True
        assert any(event.kind == "tool_call" for event in outcome.events)

    def test_timeout_decodes_byte_output(self) -> None:
        def timing_out(argv: list[str], **kwargs: Any) -> CliRun:
            del argv, kwargs
            return CliRun("", "", -1, timed_out=True)

        outcome = ClaudeCodeSession(timeout_s=1.0, runner=timing_out).run(
            _task(), workspace=Path("/tmp")
        )
        assert outcome.termination_reason == "timeout"


def test_runtime_drives_a_recorded_session_end_to_end(recorded_stream: str, tmp_path: Path) -> None:
    runtime = ExternalAgentRuntime(
        lambda: ClaudeCodeSession(runner=_Recorder(stdout=recorded_stream)),
        agent="claude_code",
        model="claude-sonnet-5",
        workspace_root=tmp_path,
    )
    results = runtime.run_batch([_task()])
    assert len(results) == 1
    assert results[0].final_text == "00000000-0000-4000-8000-000000000000"
    assert results[0].metadata["policy_source"] == "external_agent"


@pytest.mark.skipif(
    os.environ.get("ALPHAAPOLLO_LIVE_CLAUDE") != "1" or shutil.which("claude") is None,
    reason="live Claude Code run requires ALPHAAPOLLO_LIVE_CLAUDE=1 and the claude CLI",
)
def test_live_claude_code_solves_a_tool_using_task(tmp_path: Path) -> None:
    runtime = ExternalAgentRuntime(
        lambda: ClaudeCodeSession(
            model="sonnet",
            tools=["Bash"],
            permission_mode="bypassPermissions",
            timeout_s=240.0,
        ),
        agent="claude_code",
        model="sonnet",
        workspace_root=tmp_path,
    )
    task = AgentTask(
        task_id="live-1",
        system="Use the Bash tool to compute. End with exactly: Final answer: <number>",
        prompt="Compute 9/3*60 + 24 exactly using python3 and report the integer result.",
    )
    result = runtime.run(task)

    assert result.termination_reason == "final"
    assert "204" in result.final_text
    assert any(turn.generation_response.tool_calls for turn in result.turns)
    assert result.turns[-1].environment_transition.done is True
    assert all(turn.generation_response.is_trainable is False for turn in result.turns)
