"""Constructed Pi v3 records exercise the native-to-ATIF boundary."""

import json

import pytest

pytest.importorskip("harbor")
from harbor.models.trajectories import Trajectory

from alphaapollo.workflows.harbor_chips.pi_trajectory import read_pi_trajectory


def session(tmp_path):
    records = [
        {"type": "session", "version": 3, "id": "session-fixture"},
        {"type": "model_change", "provider": "fixture", "modelId": "fixture-model"},
        {"type": "thinking_level_change", "thinkingLevel": "off"},
        {
            "type": "message",
            "message": {
                "role": "system",
                "content": "",
                "sections": {"preamble": "Solve the task.", "rules": "Use the public tools."},
                "toolsAdded": [
                    {
                        "name": "bash",
                        "description": "Run a command.",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    }
                ],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Repair the circuit."}],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "model": "fixture-model",
                "stopReason": "toolUse",
                "usage": {"input": 10, "cacheRead": 4, "output": 3, "cost": {"total": 0}},
                "content": [
                    {"type": "thinking", "thinking": "Inspect the public feedback."},
                    {
                        "type": "toolCall",
                        "id": "c1",
                        "name": "bash",
                        "arguments": {"command": "harness-public info"},
                    },
                ],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "c1",
                "toolName": "bash",
                "isError": False,
                "content": [{"type": "text", "text": "tail only [truncated]"}],
                "details": {
                    "truncation": {"truncated": True, "totalBytes": 100, "outputBytes": 20}
                },
            },
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "model": "fixture-model",
                "stopReason": "stop",
                "content": [{"type": "text", "text": "Done."}],
            },
        },
    ]
    for i, entry in enumerate(records[1:], 1):
        entry.update(
            id=f"e{i}", parentId=f"e{i - 1}" if i > 1 else None, timestamp="2026-01-01T00:00:00Z"
        )
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in records) + "\n")
    return path, records


def test_native_context_calls_and_truncation_survive_atif(tmp_path):
    path, _ = session(tmp_path)
    original = path.read_bytes()
    trajectory = read_pi_trajectory(path, agent_version="fixture-v1", requested_thinking="low")
    loaded = Trajectory.model_validate_json(trajectory.model_dump_json())
    assert loaded.steps[0].message == "Solve the task.\n\nUse the public tools."
    assert loaded.steps[1].message == "Repair the circuit."
    action = loaded.steps[2]
    assert action.tool_calls[0].function_name == "bash"
    assert action.tool_calls[0].arguments == {"command": "harness-public info"}
    assert action.observation.results[0].source_call_id == "c1"
    assert action.observation.results[0].content == "tail only [truncated]"
    assert action.observation.results[0].extra["truncation"]["truncated"] is True
    assert action.metrics.prompt_tokens == 14
    assert action.metrics.cost_usd is None
    assert loaded.agent.extra["requested_thinking"] == "low"
    assert loaded.agent.extra["recorded_thinking"] == "off"
    assert loaded.agent.tool_definitions[0]["function"]["name"] == "bash"
    assert loaded.extra["context_fidelity"] == "native_reconstruction"
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "damage",
    [
        "branch",
        "duplicate_entry",
        "missing_system",
        "unknown_tool",
        "changed_model",
        "duplicate_json",
        "changed_tools",
        "late_task",
    ],
)
def test_unsupported_or_ambiguous_history_is_rejected(tmp_path, damage):
    path, records = session(tmp_path)
    if damage == "branch":
        records[-1]["parentId"] = "e1"
    elif damage == "duplicate_entry":
        records[-1]["id"] = "e1"
    elif damage == "missing_system":
        records[3]["message"] = {"role": "user", "content": "No system context."}
    elif damage == "unknown_tool":
        records[5]["message"]["content"][1]["name"] = "private_grader"
        records[6]["message"]["toolName"] = "private_grader"
    elif damage == "changed_model":
        records[-1]["message"]["model"] = "another-model"
    elif damage == "changed_tools":
        records[4]["message"]["toolsAdded"] = records[3]["message"]["toolsAdded"]
    elif damage == "late_task":
        records.insert(6, records.pop(4))
        for i, record in enumerate(records[1:], 1):
            record["parentId"] = records[i - 1]["id"] if i > 1 else None
    raw = "\n".join(json.dumps(x) for x in records)
    if damage == "duplicate_json":
        raw = raw.replace('"model": "fixture-model"', '"model": "wrong", "model": "fixture-model"')
    path.write_text(raw)
    with pytest.raises(ValueError):
        read_pi_trajectory(path, agent_version="fixture-v1")


def test_literal_redaction_does_not_modify_source(tmp_path):
    path, records = session(tmp_path)
    records[5]["message"]["content"][1]["arguments"]["command"] = "echo secret-fixture-value"
    path.write_text("\n".join(json.dumps(x) for x in records))
    raw = path.read_bytes()
    trajectory = read_pi_trajectory(
        path, agent_version="fixture-v1", redact=("secret-fixture-value",)
    )
    assert "secret-fixture-value" not in trajectory.model_dump_json()
    assert trajectory.steps[2].tool_calls[0].arguments["command"] == "echo [REDACTED]"
    assert path.read_bytes() == raw


def test_redaction_rejects_unredacted_object_keys(tmp_path):
    path, records = session(tmp_path)
    records[5]["message"]["content"][1]["arguments"]["secret-fixture-key"] = "value"
    path.write_text("\n".join(json.dumps(x) for x in records))
    with pytest.raises(ValueError, match="object key"):
        read_pi_trajectory(path, agent_version="fixture-v1", redact=("secret-fixture-key",))


def test_prompt_usage_includes_cache_creation_tokens(tmp_path):
    path, records = session(tmp_path)
    records[5]["message"]["usage"]["cacheWrite"] = 2
    path.write_text("\n".join(json.dumps(x) for x in records))
    trajectory = read_pi_trajectory(path, agent_version="fixture-v1")
    assert trajectory.steps[2].metrics.prompt_tokens == 16
