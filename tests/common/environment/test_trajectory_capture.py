from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphaapollo.common.environment import (
    DefaultEnvironment,
    EnvironmentCapture,
    EnvironmentCaptureKind,
    EnvironmentContext,
    EnvironmentTrajectoryReader,
    RecordingEnvironmentEventSink,
    SensitiveEnvironmentInputError,
    TextOnlyToolBridge,
    TrajectoryEnvironmentEventSink,
    ensure_safe_environment_input,
    sanitize_capture_value,
)
from alphaapollo.common.execution import (
    ArtifactStore,
    ExecutionContext,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.trajectory import TrajectoryQuery, TrajectoryStore
from tests.common.environment.fakes import CanonicalTestToolBridge


class _CaptureExecutor:
    def __init__(self, stdout: str = "204\n") -> None:
        self.stdout = stdout

    def execute(
        self,
        request: ToolRequest,
        context: ExecutionContext,
    ) -> ToolResponse:
        return ToolResponse(
            call_id=request.call_id,
            tool_id=request.tool_id,
            stdout=self.stdout,
        )


def _store(tmp_path: Path) -> TrajectoryStore:
    return TrajectoryStore(
        jsonl_path=tmp_path / "events.jsonl",
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )


def test_full_interaction_capture_is_resolvable_and_replayable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_CaptureExecutor()),
        event_sink=TrajectoryEnvironmentEventSink(store),
    )
    initial = "Solve the walking-speed problem."
    code = "<python_code>print(204)</python_code>"
    final = "The answer is 204."

    env.init(
        session_id="full-capture",
        actor="solver",
        branch_id="solver-branch",
        initial_observation=initial,
        episode_context={
            "workflow": "vanilla",
            "model": "local-solver",
            "config_version": 1,
        },
    )
    tool_step = env.step(code)
    env.step(final)
    env.close()

    view = EnvironmentTrajectoryReader(store).read_session("full-capture")
    assert [record.kind for record in view.records] == [
        EnvironmentCaptureKind.MODEL_INPUT,
        EnvironmentCaptureKind.MODEL_OUTPUT,
        EnvironmentCaptureKind.TOOL_REQUEST,
        EnvironmentCaptureKind.TOOL_RESPONSE,
        EnvironmentCaptureKind.ENVIRONMENT_OBSERVATION,
        EnvironmentCaptureKind.MODEL_OUTPUT,
        EnvironmentCaptureKind.ENVIRONMENT_OBSERVATION,
        EnvironmentCaptureKind.FINAL_OUTPUT,
    ]
    assert view.records_of(EnvironmentCaptureKind.MODEL_INPUT)[0].content == {
        "role": "environment",
        "content": initial,
        "episode_context": {
            "workflow": "vanilla",
            "model": "local-solver",
            "config_version": 1,
        },
    }
    assert view.records_of(EnvironmentCaptureKind.MODEL_OUTPUT)[0].content == {
        "role": "assistant",
        "content": code,
    }
    request = view.records_of(EnvironmentCaptureKind.TOOL_REQUEST)[0].content
    assert request["arguments"] == {"code": "print(204)"}
    response = view.records_of(EnvironmentCaptureKind.TOOL_RESPONSE)[0].content
    assert response["stdout"] == "204\n"
    assert response["record"]["tool_id"] == "python"
    assert response["record"]["args"] == {"code": "print(204)"}
    observations = view.records_of(EnvironmentCaptureKind.ENVIRONMENT_OBSERVATION)
    assert observations[0].content["content"] == tool_step["observations"]
    assert observations[1].content["content"] == final
    assert view.records_of(EnvironmentCaptureKind.FINAL_OUTPUT)[0].content == {
        "role": "assistant",
        "content": final,
    }

    steps = view.replay_steps()
    assert [(step.round_index, step.step_index) for step in steps] == [
        (0, 1),
        (0, 2),
    ]
    assert steps[0].model_outputs[0].content["content"] == code
    assert steps[0].tool_requests[0].content["tool_id"] == "python"
    assert steps[0].tool_results[0].content["stdout"] == "204\n"
    assert steps[1].final_outputs[0].content["content"] == final

    events = store.read_events(TrajectoryQuery(session_id="full-capture"))
    captured = [event for event in events if "capture_ref" in event.payload]
    assert len(captured) == len(view.records)
    assert all(len(event.artifact_refs) == 1 for event in captured)
    assert all(event.payload["capture_status"] == "captured" for event in captured)
    exported = view.export_json()
    assert exported[0]["content"]["episode_context"]["workflow"] == "vanilla"
    exported[0]["content"]["episode_context"]["workflow"] = "mutated"
    assert view.records[0].content["episode_context"]["workflow"] == "vanilla"
    store.close()


def test_sensitive_initial_input_fails_before_model_loop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_CaptureExecutor()),
        event_sink=TrajectoryEnvironmentEventSink(store),
    )

    with pytest.raises(SensitiveEnvironmentInputError):
        env.init(
            session_id="gt-input",
            actor="solver",
            initial_observation="safe question",
            episode_context={"config_version": 1, "ground_truth": 204},
        )

    assert store.read_events() == []

    with pytest.raises(SensitiveEnvironmentInputError):
        env.init(
            session_id="secret-workspace",
            actor="solver",
            initial_observation="safe question",
            workspace_snapshot_ref="api_key=host-secret",
        )

    assert store.read_events() == []
    store.close()


@pytest.mark.parametrize("key", ["answer", "gold_answer", "solution", "task_payload"])
def test_evaluator_shaped_environment_input_keys_fail_closed(key: str) -> None:
    with pytest.raises(SensitiveEnvironmentInputError, match=key):
        ensure_safe_environment_input(
            {
                "model_input": {
                    "content": "safe question",
                    "episode_context": {key: "PRIVATE-EVALUATOR-VALUE"},
                }
            }
        )


def test_evaluator_input_key_guard_does_not_redact_model_output_fields() -> None:
    generated = {
        "answer": "model answer",
        "gold_answer": "model discussion",
        "solution": "model solution",
    }

    sanitized, flags, redactions = sanitize_capture_value(generated)

    assert sanitized == generated
    assert flags == ()
    assert redactions == ()


def test_default_environment_accepts_private_payload_without_capturing_it() -> None:
    canaries = {
        "answer": "PRIVATE-ANSWER-CANARY",
        "solution": "PRIVATE-SOLUTION-CANARY",
        "token": "PRIVATE-TOKEN-CANARY",
    }
    events = RecordingEnvironmentEventSink()
    env = DefaultEnvironment(
        tool_bridge=TextOnlyToolBridge(),
        event_sink=events,
    )

    env.init(
        EnvironmentContext(
            session_id="private-payload",
            actor="solver",
            task_id="private-payload",
            system_prompt="public system",
            user_prompt="public question",
            task_payload=canaries,
            metadata={"source": "public"},
        )
    )
    env.close()

    initial_capture = next(event.capture for event in events.events if event.capture is not None)
    assert initial_capture is not None
    episode_context = initial_capture["content"]["episode_context"]
    assert episode_context["source"] == "public"
    assert "task_payload" not in episode_context
    durable = json.dumps([event.capture for event in events.events])
    assert all(canary not in durable for canary in canaries.values())


def test_heuristic_terms_in_problem_text_are_redacted_not_rejected(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_CaptureExecutor()),
        event_sink=TrajectoryEnvironmentEventSink(store),
    )
    problem = "A glossary defines gold: a metal and answer key: a map legend."

    initial, _ = env.init(
        session_id="heuristic-text",
        actor="solver",
        initial_observation=problem,
        episode_context={"config_version": 1},
    )
    env.close()

    assert initial == problem
    view = EnvironmentTrajectoryReader(store).read_session("heuristic-text")
    record = view.records_of(EnvironmentCaptureKind.MODEL_INPUT)[0]
    assert record.content["content"] == "[REDACTED]"
    assert set(record.contamination_flags) == {"ground_truth"}
    store.close()


def test_capture_sanitizer_bounds_cycles_and_excessive_depth() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    sanitized_cycle, cycle_flags, cycle_redactions = sanitize_capture_value(cyclic)

    assert sanitized_cycle == {"self": "[TRUNCATED: cyclic reference]"}
    assert cycle_flags == ("capture_cycle",)
    assert cycle_redactions == ("$.content.self",)

    capture = EnvironmentCapture(
        kind=EnvironmentCaptureKind.MODEL_OUTPUT,
        content=cyclic,
    )
    assert capture.content == {"self": "[TRUNCATED: cyclic reference]"}
    assert capture.contamination_flags == ("capture_cycle",)

    deeply_nested: object = "leaf"
    for _ in range(100):
        deeply_nested = [deeply_nested]

    sanitized_depth, depth_flags, depth_redactions = sanitize_capture_value(deeply_nested)

    assert "capture_depth_limit" in depth_flags
    assert depth_redactions
    assert "maximum depth exceeded" in json.dumps(sanitized_depth)


def test_nested_ground_truth_and_credentials_are_redacted_from_artifacts(
    tmp_path: Path,
) -> None:
    canary = "expected_answer=204"
    credential = "api_key=sk-test-value"
    store = _store(tmp_path)
    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_CaptureExecutor(stdout=f"{canary}\n{credential}\n")),
        event_sink=TrajectoryEnvironmentEventSink(store),
    )
    env.init(
        session_id="redaction",
        actor="solver",
        initial_observation="Solve safely.",
    )
    tool_step = env.step(
        {
            "message": {
                "content": f"<python_code>print(1) # {canary}</python_code>",
                "metadata": {
                    "ground_truth": "204",
                    "password": "host-secret",
                },
            }
        }
    )
    env.close()

    assert "<tool_response>" in tool_step["observations"]
    assert canary not in tool_step["observations"]
    assert credential not in tool_step["observations"]
    assert "[REDACTED]" in tool_step["observations"]
    assert set(tool_step["metadata"]["observation"]["contamination_flags"]) == {
        "credential",
        "ground_truth",
    }

    events = store.read_events(TrajectoryQuery(session_id="redaction"))
    artifact_bytes = b"\n".join(
        store.get_blob(ref) for event in events for ref in event.artifact_refs
    )
    durable = (tmp_path / "events.jsonl").read_bytes() + artifact_bytes
    assert canary.encode() not in durable
    assert credential.encode() not in durable
    assert b"host-secret" not in durable
    assert b'"[REDACTED]"' in artifact_bytes

    redacted = [event for event in events if event.payload.get("capture_status") == "redacted"]
    assert redacted
    assert all(event.payload["contamination_flags"] for event in redacted)
    clean_events = store.read_events(
        TrajectoryQuery(session_id="redaction", exclude_contaminated=True)
    )
    assert len(clean_events) < len(events)

    view = EnvironmentTrajectoryReader(store).read_session("redaction")
    assert any(record.redactions for record in view.records)
    assert all(canary not in json.dumps(record.content) for record in view.records)
    observation = view.records_of(EnvironmentCaptureKind.ENVIRONMENT_OBSERVATION)[0]
    assert set(observation.contamination_flags) == {
        "credential",
        "ground_truth",
    }
    clean_view = EnvironmentTrajectoryReader(store).read_session(
        "redaction",
        exclude_contaminated=True,
    )
    assert all(not record.contamination_flags for record in clean_view.records)
    store.close()


def test_timeout_error_and_observation_are_captured(tmp_path: Path) -> None:
    class _TimeoutExecutor:
        def execute(
            self,
            request: ToolRequest,
            context: ExecutionContext,
        ) -> ToolResponse:
            raise TimeoutError("private host detail")

    store = _store(tmp_path)
    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_TimeoutExecutor()),
        event_sink=TrajectoryEnvironmentEventSink(store),
    )
    env.init(
        session_id="timeout-capture",
        actor="verifier",
        initial_observation="Verify the candidate.",
    )
    env.step("<python_code>while True: pass</python_code>")
    env.close()

    view = EnvironmentTrajectoryReader(store).read_session(
        "timeout-capture",
        actor="verifier",
    )
    response = view.records_of(EnvironmentCaptureKind.TOOL_RESPONSE)[0]
    assert response.content["error_stage"] == "timeout"
    assert response.content["error_code"] == "execution_timeout"
    assert response.content["attempted"] is True
    assert response.content["record"]["exit_code"] == 124
    assert "private host detail" not in json.dumps(response.content)
    observation = view.records_of(EnvironmentCaptureKind.ENVIRONMENT_OBSERVATION)[0]
    assert observation.content["error_code"] == "execution_timeout"
    store.close()


def test_solver_and_verifier_capture_views_remain_isolated(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sink = TrajectoryEnvironmentEventSink(store)
    solver = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_CaptureExecutor()),
        event_sink=sink,
    )
    verifier = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_CaptureExecutor()),
        event_sink=sink,
    )
    solver.init(
        session_id="solver-session",
        actor="solver",
        initial_observation="Solver input",
        workspace_snapshot_ref="solver-workspace",
    )
    verifier.init(
        session_id="verifier-session",
        actor="verifier",
        initial_observation="Verifier input",
        workspace_snapshot_ref="verifier-workspace",
    )
    solver.step("Solver final")
    verifier.step("Verifier final")
    solver.close()
    verifier.close()

    reader = EnvironmentTrajectoryReader(store)
    solver_view = reader.read_session("solver-session", actor="solver")
    verifier_view = reader.read_session("verifier-session", actor="verifier")
    assert all(record.actor == "solver" for record in solver_view.records)
    assert all(record.actor == "verifier" for record in verifier_view.records)
    assert "Verifier input" not in json.dumps([record.content for record in solver_view.records])
    assert "Solver input" not in json.dumps([record.content for record in verifier_view.records])
    store.close()


def test_raw_provider_response_and_usage_are_preserved(tmp_path: Path) -> None:
    store = _store(tmp_path)
    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_CaptureExecutor()),
        event_sink=TrajectoryEnvironmentEventSink(store),
    )
    env.init(
        session_id="raw-provider",
        actor="solver",
        initial_observation="Provider fixture",
    )
    raw_response = {
        "id": "response-1",
        "choices": [
            {
                "message": {
                    "content": "FINAL-204",
                    "reasoning_content": (
                        "The tool result verifies the arithmetic, so I can finalize."
                    ),
                    "tool_calls": [],
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
        },
    }
    env.step(raw_response)
    env.close()

    view = EnvironmentTrajectoryReader(store).read_session("raw-provider")
    model_output = view.records_of(EnvironmentCaptureKind.MODEL_OUTPUT)[0].content
    assert model_output == {
        "role": "assistant",
        "provider_response": raw_response,
    }
    assert (
        model_output["provider_response"]["choices"][0]["message"]["reasoning_content"]
        == "The tool result verifies the arithmetic, so I can finalize."
    )
    assert view.records_of(EnvironmentCaptureKind.FINAL_OUTPUT)[0].content["content"] == "FINAL-204"
    store.close()


def test_tool_artifact_refs_are_attached_to_execution_event(tmp_path: Path) -> None:
    store = _store(tmp_path)
    output_ref = store.put_blob(
        b"tool artifact",
        type_="text/plain",
        created_by="test.tool",
    )

    class _ArtifactExecutor:
        def execute(
            self,
            request: ToolRequest,
            context: ExecutionContext,
        ) -> ToolResponse:
            return ToolResponse(
                call_id=request.call_id,
                tool_id=request.tool_id,
                stdout="created artifact",
                artifacts=(output_ref,),
            )

    env = DefaultEnvironment(
        tool_bridge=CanonicalTestToolBridge(_ArtifactExecutor()),
        event_sink=TrajectoryEnvironmentEventSink(store),
    )
    env.init(
        session_id="tool-artifact",
        actor="solver",
        initial_observation="Create an artifact.",
    )
    env.step("<python_code>print('artifact')</python_code>")
    env.close()

    execution_event = next(
        event
        for event in store.read_events()
        if event.payload["environment_event"] == "tool_execution_completed"
    )
    assert len(execution_event.artifact_refs) == 2
    assert any(ref.id == output_ref.id for ref in execution_event.artifact_refs)
    response = (
        EnvironmentTrajectoryReader(store)
        .read_session("tool-artifact")
        .records_of(EnvironmentCaptureKind.TOOL_RESPONSE)[0]
    )
    assert response.content["artifacts"] == [output_ref.model_dump(mode="json")]
    assert store.get_blob(output_ref) == b"tool artifact"
    store.close()
