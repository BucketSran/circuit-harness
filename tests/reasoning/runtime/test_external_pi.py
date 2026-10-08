# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""pi session coverage, driven by recorded real event streams.

Both fixtures were captured from ``pi -p --mode json`` (pi 0.80.10) and then
sanitized: signatures, identities, timestamps, paths and reasoning text use
fixture values; the 42 ``message_update`` deltas the adapter ignores were
dropped. Event structure is preserved. See ``fixtures/README.md``.

* ``pi_bash_stream.jsonl`` is a tool-using run. It is the reason the adapter
  de-duplicates tool calls: pi announces the same call twice.
* ``pi_empty_response.jsonl`` is a provider rejection — pi exits zero with an
  empty assistant message.
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
from alphaapollo.reasoning.runtime.external.agents.pi import (
    CodexViaPiSession,
    PiSession,
    parse_mode_json,
)
from alphaapollo.reasoning.runtime.external.cli import CliRun
from alphaapollo.reasoning.runtime.external_agent_runtime import (
    ExternalAgentRuntime,
    project_outcome,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLING = SamplingOptions(temperature=1.0, max_tokens=1024)


@pytest.fixture(scope="module")
def recorded_empty_stream() -> str:
    return (FIXTURES / "pi_empty_response.jsonl").read_text()


@pytest.fixture(scope="module")
def recorded_tool_stream() -> str:
    return (FIXTURES / "pi_bash_stream.jsonl").read_text()


def _task() -> AgentTask:
    return AgentTask(task_id="task-1", system="be terse", prompt="compute 9/3*60 + 24")


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


class TestArgv:
    def test_isolation_flags_are_always_present(self) -> None:
        argv = PiSession().argv(_task())
        for flag in ("-p", "--no-session", "--no-approve"):
            assert flag in argv
        assert argv[argv.index("--mode") + 1] == "json"

    def test_system_prompt_uses_the_native_flag(self) -> None:
        argv = PiSession().argv(_task())
        assert argv[argv.index("--system-prompt") + 1] == "be terse"

    def test_prompt_is_supplied_on_stdin(self) -> None:
        recorder = _Recorder(stdout="")
        PiSession(runner=recorder).run(_task(), workspace=Path("/tmp"))
        assert "compute 9/3*60 + 24" not in recorder.argv
        assert recorder.kwargs["prompt"] == "compute 9/3*60 + 24"

    @pytest.mark.parametrize(
        ("tool_mode", "expected"),
        [("none", "--no-tools"), ("no_builtin", "--no-builtin-tools")],
    )
    def test_tool_modes_map_to_their_flags(self, tool_mode: str, expected: str) -> None:
        assert expected in PiSession(tool_mode=tool_mode).argv(_task())

    def test_allowlist_mode_passes_the_names(self) -> None:
        argv = PiSession(tool_mode="allowlist", tools=["bash", "read"]).argv(_task())
        assert argv[argv.index("--tools") + 1] == "bash,read"

    def test_model_provider_and_thinking_are_forwarded(self) -> None:
        argv = PiSession(model="claude-sonnet-5", provider="anthropic", thinking="high").argv(
            _task()
        )
        assert argv[argv.index("--model") + 1] == "claude-sonnet-5"
        assert argv[argv.index("--provider") + 1] == "anthropic"
        assert argv[argv.index("--thinking") + 1] == "high"

    def test_codex_via_pi_pins_the_provider_and_uses_pi_tool_controls(self) -> None:
        argv = CodexViaPiSession(model="gpt-5.4-mini", tool_mode="no_builtin").argv(_task())

        assert argv[argv.index("--provider") + 1] == "openai-codex"
        assert argv[argv.index("--model") + 1] == "gpt-5.4-mini"
        assert "--no-builtin-tools" in argv

    def test_codex_via_pi_rejects_a_different_provider(self) -> None:
        with pytest.raises(ValueError, match="fixed to 'openai-codex'"):
            CodexViaPiSession(provider="openai")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"tool_mode": "everything"}, "tool_mode must be one of"),
            ({"tool_mode": "allowlist"}, "requires at least one tool name"),
            ({"timeout_s": 0}, "timeout_s must be"),
            ({"provider": ""}, "provider must be"),
        ],
    )
    def test_rejects_invalid_settings(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            PiSession(**kwargs)


class TestRecordedProviderRejection:
    def test_zero_exit_with_no_content_is_not_a_successful_empty_answer(
        self, recorded_empty_stream: str
    ) -> None:
        outcome = parse_mode_json(recorded_empty_stream)
        assert outcome.termination_reason == "external_error"
        assert outcome.provider_metadata["empty_response"] is True
        assert outcome.final_text == ""
        assert outcome.events == ()

    def test_session_and_model_identity_survive_the_failure(
        self, recorded_empty_stream: str
    ) -> None:
        metadata = parse_mode_json(recorded_empty_stream).provider_metadata
        assert metadata["session_id"]
        assert metadata["model"] == "gpt-5.6-sol"
        assert metadata["provider"] == "openai-codex"


def test_settled_pi_length_stop_is_reported_as_truncated() -> None:
    stream = "\n".join(
        json.dumps(item)
        for item in (
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "unfinished"}],
                    "stopReason": "length",
                },
            },
            {"type": "agent_settled"},
        )
    )
    outcome = parse_mode_json(stream)
    assert outcome.termination_reason == "truncated"


class TestRecordedToolStream:
    def test_one_real_call_is_reported_once_not_twice(self, recorded_tool_stream: str) -> None:
        outcome = parse_mode_json(recorded_tool_stream)
        kinds = [event.kind for event in outcome.events]
        # pi announces the call as a `toolCall` block *and* as
        # `tool_execution_start`; without de-duplication the run would report
        # two calls where the agent made one.
        assert kinds == ["reasoning", "tool_call", "tool_result", "reasoning", "message"]

        call = outcome.events[1].tool_call
        assert call is not None
        assert call.name == "bash"
        assert "python3" in json.loads(call.arguments)["command"]
        assert outcome.events[2].call_id == call.id
        assert outcome.events[2].content == "204.0\n"
        assert outcome.events[2].failed is False

    def test_authoritative_message_end_supplies_text_and_usage(
        self, recorded_tool_stream: str
    ) -> None:
        outcome = parse_mode_json(recorded_tool_stream)
        assert outcome.termination_reason == "final"
        assert outcome.final_text == "Final answer: 204"
        assert outcome.usage["totalTokens"] == 836
        assert outcome.provider_metadata["provider"] == "openai-codex"
        assert outcome.provider_metadata["model"] == "gpt-5.4-mini"

    def test_projection_yields_a_tool_turn_and_a_final_turn(
        self, recorded_tool_stream: str
    ) -> None:
        result = project_outcome(
            _task(),
            parse_mode_json(recorded_tool_stream),
            agent="pi",
            model="gpt-5.4-mini",
            sampling=SAMPLING,
        )
        assert [turn.index for turn in result.turns] == [0, 1]
        assert [call.name for call in result.turns[0].generation_response.tool_calls] == ["bash"]
        assert result.turns[0].generation_response.reasoning_content is not None
        assert result.turns[0].environment_transition.observation.content == "204.0\n"
        assert result.turns[0].environment_transition.done is False
        assert result.turns[1].generation_response.content == "Final answer: 204"
        assert result.turns[1].environment_transition.done is True
        assert result.metadata["trainable"] is False


class TestParseEdgeCases:
    @pytest.mark.parametrize("registered,dispatched", [(True, False), (False, False), (True, True)])
    def test_schema_rejection_requires_bridge_dispatch_evidence(self, registered, dispatched):
        records = []
        if registered:
            records.append({"type": "alphaapollo_mcp_tool", "toolName": "read"})
        records.append(
            {"type": "tool_execution_start", "toolCallId": "bad", "toolName": "read", "args": {}}
        )
        if dispatched:
            records.append(
                {"type": "alphaapollo_mcp_dispatch", "toolCallId": "bad", "toolName": "read"}
            )
        result = {
            "type": "tool_execution_end",
            "toolCallId": "bad",
            "toolName": "read",
            "result": {
                "content": [
                    {"type": "text", "text": 'Validation failed for tool "read":\nmissing path'}
                ]
            },
            "isError": True,
        }
        records.append(result)
        outcome = parse_mode_json("\n".join(json.dumps(record) for record in records))
        event = outcome.events[-1]
        assert event.rejected_before_dispatch is (registered and not dispatched)
        assert event.raw == result

    def test_failed_tool_execution_is_marked(self) -> None:
        stream = "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_execution_end",
                        "toolCallId": "call_1",
                        "toolName": "bash",
                        "result": {"content": [{"type": "text", "text": "boom"}]},
                        "isError": True,
                    }
                ),
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "x"}],
                        },
                    }
                ),
                '{"type":"agent_settled"}',
            ]
        )
        result = parse_mode_json(stream).events[0]
        assert (result.failed, result.content) == (True, "boom")

    def test_missing_agent_settled_is_reported(self) -> None:
        stream = json.dumps(
            {
                "type": "message_end",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
            }
        )
        outcome = parse_mode_json(stream)
        assert outcome.termination_reason == "external_error"
        assert outcome.provider_metadata["missing_agent_settled"] is True

    def test_user_messages_are_not_projected_as_agent_output(self) -> None:
        stream = "\n".join(
            [
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                    }
                ),
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "yo"}],
                        },
                    }
                ),
                '{"type":"agent_settled"}',
            ]
        )
        outcome = parse_mode_json(stream)
        assert [event.content for event in outcome.events] == ["yo"]

    def test_inline_tool_call_blocks_are_projected(self) -> None:
        stream = "\n".join(
            [
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": "call_9",
                                    "name": "read",
                                    "arguments": {"path": "a.txt"},
                                }
                            ],
                        },
                    }
                ),
                '{"type":"agent_settled"}',
            ]
        )
        call = parse_mode_json(stream).events[0].tool_call
        assert call is not None
        assert (call.id, call.name) == ("call_9", "read")

    def test_unknown_event_types_are_surfaced_rather_than_dropped(self) -> None:
        stream = "\n".join(
            [
                '{"type":"widget_update","payload":{}}',
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "x"}],
                        },
                    }
                ),
                '{"type":"agent_settled"}',
            ]
        )
        metadata = parse_mode_json(stream).provider_metadata
        assert metadata["unhandled_event_types"] == ["widget_update"]

    def test_malformed_json_fails_loudly_with_the_line_number(self) -> None:
        with pytest.raises(ValueError, match="line 2 is not valid JSON"):
            parse_mode_json('{"type":"agent_settled"}\nnope\n')


class TestProcessFailures:
    def test_non_zero_exit_is_an_external_error(self) -> None:
        recorder = _Recorder(stdout="", stderr="pi: provider unavailable", returncode=2)
        outcome = PiSession(runner=recorder).run(_task(), workspace=Path("/tmp"))
        assert outcome.termination_reason == "external_error"
        assert "provider unavailable" in outcome.provider_metadata["stderr"]

    def test_timeout_keeps_the_partial_stream(self, recorded_tool_stream: str) -> None:
        # A SIGKILL lands wherever the CLI's stdout buffer was, so the stream ends
        # mid-record. The complete records must survive that fragment.
        lines = recorded_tool_stream.splitlines()
        partial = "\n".join(lines[:7]) + "\n" + lines[7][:40]

        def timing_out(argv: list[str], **kwargs: Any) -> CliRun:
            del argv, kwargs
            return CliRun(partial, "", -1, timed_out=True)

        outcome = PiSession(timeout_s=1.0, runner=timing_out).run(_task(), workspace=Path("/tmp"))
        assert outcome.termination_reason == "timeout"
        assert outcome.provider_metadata["partial_stream"] is True
        assert outcome.provider_metadata["dropped_partial_record"] is True
        assert any(event.kind == "reasoning" for event in outcome.events)


def test_runtime_drives_a_recorded_session_end_to_end(
    recorded_tool_stream: str, tmp_path: Path
) -> None:
    runtime = ExternalAgentRuntime(
        lambda: PiSession(runner=_Recorder(stdout=recorded_tool_stream)),
        agent="pi",
        model="gpt-5.4-mini",
        workspace_root=tmp_path,
    )
    results = runtime.run_batch([_task()])
    assert results[0].final_text == "Final answer: 204"
    assert results[0].metadata["policy_source"] == "external_agent"


def test_codex_via_pi_records_both_the_logical_agent_and_runner(
    recorded_tool_stream: str,
) -> None:
    outcome = CodexViaPiSession(runner=_Recorder(stdout=recorded_tool_stream)).run(
        _task(), workspace=Path("/tmp")
    )

    assert outcome.provider_metadata["agent"] == "codex_via_pi"
    assert outcome.provider_metadata["runner"] == "pi"
    assert outcome.provider_metadata["provider"] == "openai-codex"


@pytest.mark.skipif(
    os.environ.get("ALPHAAPOLLO_LIVE_PI") != "1" or shutil.which("pi") is None,
    reason="live pi run requires ALPHAAPOLLO_LIVE_PI=1, the pi CLI, and pi credentials",
)
def test_live_pi_solves_a_tool_using_task(tmp_path: Path) -> None:
    runtime = ExternalAgentRuntime(
        lambda: PiSession(tool_mode="allowlist", tools=["bash"], timeout_s=300.0),
        agent="pi",
        model="default",
        workspace_root=tmp_path,
    )
    task = AgentTask(
        task_id="live-pi-1",
        system="Use bash to compute. End with exactly: Final answer: <number>",
        prompt="Compute 9/3*60 + 24 exactly using python3 and report the integer result.",
    )
    result = runtime.run(task)

    assert result.termination_reason == "final"
    assert "204" in result.final_text
    assert all(turn.generation_response.is_trainable is False for turn in result.turns)


@pytest.mark.parametrize("reason,expected", [("length", "truncated"), ("error", "external_error")])
def test_final_stop_reason_survives_empty_response_or_previous_text(reason, expected):
    records = []
    if reason == "error":
        records.append(
            {"type": "message_end", "message": {"role": "assistant", "content": "Earlier response"}}
        )
    records.extend(
        [
            {
                "type": "message_end",
                "message": {"role": "assistant", "content": [], "stopReason": reason},
            },
            {"type": "agent_settled"},
        ]
    )
    outcome = parse_mode_json("\n".join(json.dumps(r) for r in records))
    assert outcome.termination_reason == expected
