# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contract and projection coverage for the external agent Runtime."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from alphaapollo.common.environment.base import (
    EnvironmentInitResult,
    EnvironmentObservation,
    EnvironmentTransition,
)
from alphaapollo.common.generation.base import SamplingOptions, ToolCall
from alphaapollo.common.trajectory.recorder import SensitiveEnvironmentInputError
from alphaapollo.reasoning.runtime.agent_runtime import AgentResult, AgentTask
from alphaapollo.reasoning.runtime.external import FakeExternalSession, session_factory
from alphaapollo.reasoning.runtime.external.bridge.environment_socket import (
    ENVIRONMENT_SOCKET_NAME,
    EnvironmentSocketToolBridge,
)
from alphaapollo.reasoning.runtime.external.cli import CliSettings, run_cli
from alphaapollo.reasoning.runtime.external_agent_runtime import (
    POLICY_SOURCE_EXTERNAL_AGENT,
    ExternalAgentRuntime,
    ExternalAuxiliaryError,
    ExternalEvent,
    ExternalRunOutcome,
    encode_arguments,
    project_outcome,
    workflow_workspace_path,
)

SAMPLING = SamplingOptions(temperature=1.0, max_tokens=1024)


@pytest.mark.parametrize("rejected", [True, False])
def test_parallel_frontend_rejection_preserves_the_real_environment_step(rejected):
    calls = [
        ToolCall(id="bad", name="read", arguments="{}"),
        ToolCall(id="good", name="read", arguments='{"path":"a"}'),
    ]
    outcome = ExternalRunOutcome(
        final_text="",
        termination_reason="truncated",
        events=(
            *(ExternalEvent(kind="tool_call", tool_call=call) for call in calls),
            ExternalEvent(
                kind="tool_result",
                call_id="bad",
                tool_id="read",
                content="missing path",
                failed=True,
                rejected_before_dispatch=rejected,
            ),
            ExternalEvent(kind="tool_result", call_id="good", tool_id="read", content="file"),
        ),
    )
    transition = EnvironmentTransition(
        observation="file", reward=0.25, done=False, metadata={"tool_request": {"tool_id": "read"}}
    )
    kwargs = dict(agent="pi", model="m", sampling=SAMPLING, environment_transitions=[transition])
    if not rejected:
        with pytest.raises(RuntimeError, match="did not step"):
            project_outcome(AgentTask(task_id="task", system="", prompt="read"), outcome, **kwargs)
        return
    result = project_outcome(AgentTask(task_id="task", system="", prompt="read"), outcome, **kwargs)
    assert len(result.turns) == 1
    observed = result.turns[0].environment_transition
    assert observed.observation == "file"
    assert observed.reward == 0.25
    assert not observed.env_action_valid
    assert observed.metadata["external_environment"]["environment_stepped"] is True
    assert (
        observed.metadata["external_environment"]["pre_dispatch_rejections"][0]["call_id"] == "bad"
    )
    assert result.termination_reason == "truncated"


class _RecordingEnvironment:
    def __init__(self) -> None:
        self.actions: list[object] = []
        self.transitions: list[EnvironmentTransition] = []
        self.initialized = False
        self.closed = False

    def init(self, _context: object) -> EnvironmentInitResult:
        self.initialized = True
        return EnvironmentInitResult(observation=None)

    def step(self, action: object) -> EnvironmentTransition:
        self.actions.append(action)
        if isinstance(action, dict):
            transition = EnvironmentTransition(
                observation='<tool_response>\n{"ok": true, "content": "1"}\n</tool_response>',
                reward=0.25,
                done=False,
                metadata={"tool_request": {"tool_id": "bash"}},
            )
        else:
            transition = EnvironmentTransition(
                observation=str(action),
                reward=1.0,
                done=True,
                success=True,
                termination_reason="model_output",
            )
        self.transitions.append(transition)
        return transition

    def project_response(self, response: object) -> object:
        return response.content  # type: ignore[attr-defined,no-any-return]

    def close(self) -> None:
        self.closed = True


class _EnvironmentCallingSession:
    def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
        bridge = EnvironmentSocketToolBridge(workspace / ENVIRONMENT_SOCKET_NAME)
        call = _tool_call()
        result = bridge.dispatch(
            {
                "id": call.id,
                "function": {"name": call.name, "arguments": {"command": "echo 1"}},
            },
            None,
        )
        return ExternalRunOutcome(
            final_text="answer",
            events=(
                ExternalEvent(kind="tool_call", tool_call=call),
                ExternalEvent(
                    kind="tool_result",
                    content=json.dumps(result.observation_payload(), sort_keys=True),
                    call_id=call.id,
                    tool_id=call.name,
                ),
                ExternalEvent(kind="message", content="answer"),
            ),
        )

    def close(self) -> None:
        return None


def _task(task_id: str = "task-1", **overrides: object) -> AgentTask:
    fields: dict[str, object] = {
        "task_id": task_id,
        "system": "system prompt",
        "prompt": "public problem",
    }
    fields.update(overrides)
    return AgentTask(**fields)  # type: ignore[arg-type]


def _tool_call(index: int = 0, name: str = "bash") -> ToolCall:
    return ToolCall(id=f"call-{index}", name=name, arguments='{"command": "echo 1"}')


def _project(events: tuple[ExternalEvent, ...], **outcome_fields: object) -> AgentResult:
    outcome = ExternalRunOutcome(
        final_text=str(outcome_fields.pop("final_text", "answer")),
        events=events,
        **outcome_fields,  # type: ignore[arg-type]
    )
    return project_outcome(_task(), outcome, agent="fake", model="model-x", sampling=SAMPLING)


def _real_transition(
    *, tool_id: str | None = None, done: bool = False, reason: str | None = None
) -> EnvironmentTransition:
    return EnvironmentTransition(
        observation=f"observation:{tool_id or 'final'}",
        reward=0.5,
        done=done,
        success=True if done else None,
        termination_reason=reason,
        metadata={} if tool_id is None else {"tool_request": {"tool_id": tool_id}},
    )


class TestExternalEvent:
    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of"):
            ExternalEvent(kind="thought")

    def test_tool_call_event_requires_a_tool_call(self) -> None:
        with pytest.raises(TypeError, match="tool_call events require"):
            ExternalEvent(kind="tool_call")

    def test_tool_call_is_rejected_on_other_kinds(self) -> None:
        with pytest.raises(ValueError, match="only valid on a tool_call event"):
            ExternalEvent(kind="message", tool_call=_tool_call())

    def test_raw_is_copied_so_later_provider_mutation_cannot_rewrite_history(self) -> None:
        raw = {"nested": {"value": 1}}
        event = ExternalEvent(kind="message", content="hi", raw=raw)
        raw["nested"]["value"] = 2
        assert event.raw == {"nested": {"value": 1}}


class TestOutcomeValidation:
    def test_rejects_unknown_termination_reason(self) -> None:
        with pytest.raises(ValueError, match="termination_reason must be one of"):
            ExternalRunOutcome(final_text="x", termination_reason="gave_up")

    @pytest.mark.parametrize(
        "reason", ["final", "truncated", "external_error", "cancelled", "timeout"]
    )
    def test_accepts_every_declared_reason(self, reason: str) -> None:
        outcome = ExternalRunOutcome(final_text="x", termination_reason=reason)
        assert outcome.termination_reason == reason


class TestTurnProjection:
    def test_one_message_becomes_one_terminal_turn(self) -> None:
        result = _project((ExternalEvent(kind="message", content="answer"),))
        assert len(result.turns) == 1
        transition = result.turns[0].environment_transition
        assert isinstance(transition, EnvironmentTransition)
        assert transition.done is True
        assert transition.termination_reason == "final"
        assert transition.reward == 0.0

    def test_tool_round_trip_splits_generation_and_observation(self) -> None:
        events = (
            ExternalEvent(kind="reasoning", content="thinking"),
            ExternalEvent(kind="tool_call", tool_call=_tool_call()),
            ExternalEvent(kind="tool_result", content="1", call_id="call-0", tool_id="bash"),
            ExternalEvent(kind="message", content="answer"),
        )
        result = _project(events)

        assert [turn.index for turn in result.turns] == [0, 1]
        first, second = result.turns
        assert first.generation_response.content == ""
        assert first.generation_response.reasoning_content == "thinking"
        assert [call.name for call in first.generation_response.tool_calls] == ["bash"]
        assert first.environment_transition.done is False
        assert first.environment_transition.termination_reason is None
        observation = first.environment_transition.observation
        assert isinstance(observation, EnvironmentObservation)
        assert (observation.kind, observation.content, observation.tool_id) == (
            "external_tool_result",
            "1",
            "bash",
        )
        assert second.generation_response.content == "answer"
        assert second.environment_transition.done is True

    def test_parallel_tool_results_collapse_into_one_turn(self) -> None:
        events = (
            ExternalEvent(kind="tool_call", tool_call=_tool_call(0)),
            ExternalEvent(kind="tool_call", tool_call=_tool_call(1, name="read")),
            ExternalEvent(kind="tool_result", content="a", call_id="call-0", tool_id="bash"),
            ExternalEvent(kind="tool_result", content="b", call_id="call-1", tool_id="read"),
            ExternalEvent(kind="message", content="answer"),
        )
        result = _project(events)

        assert len(result.turns) == 2
        first = result.turns[0]
        assert len(first.generation_response.tool_calls) == 2
        observation = first.environment_transition.observation
        assert observation.content == "a\nb"
        # Two results cannot be attributed to one call id without inventing one.
        assert observation.call_id is None
        assert observation.tool_id is None

    def test_failed_tool_result_is_typed_on_the_observation(self) -> None:
        events = (
            ExternalEvent(kind="tool_call", tool_call=_tool_call()),
            ExternalEvent(kind="tool_result", content="boom", call_id="call-0", failed=True),
            ExternalEvent(kind="message", content="recovered"),
        )
        observation = _project(events).turns[0].environment_transition.observation
        assert (observation.error_stage, observation.error_code) == (
            "external_tool",
            "external_tool_error",
        )

    def test_trailing_tool_result_still_terminates_the_last_turn(self) -> None:
        events = (
            ExternalEvent(kind="tool_call", tool_call=_tool_call()),
            ExternalEvent(kind="tool_result", content="1", call_id="call-0"),
        )
        result = _project(events, termination_reason="truncated")
        assert len(result.turns) == 1
        assert result.turns[0].environment_transition.done is True
        assert result.turns[0].environment_transition.termination_reason == "truncated"

    def test_no_events_produces_a_result_without_turns(self) -> None:
        result = _project((), termination_reason="external_error", final_text="")
        assert result.turns == ()
        assert result.termination_reason == "external_error"

    def test_reconstructed_transcript_grows_across_turns(self) -> None:
        events = (
            ExternalEvent(kind="tool_call", tool_call=_tool_call()),
            ExternalEvent(kind="tool_result", content="42", call_id="call-0", tool_id="bash"),
            ExternalEvent(kind="message", content="answer"),
        )
        result = _project(events)
        first, second = (turn.generation_request.messages for turn in result.turns)

        assert [message["role"] for message in first] == ["system", "user"]
        assert [message["role"] for message in second] == ["system", "user", "assistant", "tool"]
        assert second[3]["content"] == "42"
        assert second[3]["tool_call_id"] == "call-0"
        # The transcript is derived from observed events, never claimed as the
        # request the external agent actually sent.
        assert result.turns[0].generation_request.provider_options["messages_reconstructed"] is True


class TestHonestLabelling:
    def test_generation_responses_are_never_trainable(self) -> None:
        events = (
            ExternalEvent(kind="tool_call", tool_call=_tool_call()),
            ExternalEvent(kind="tool_result", content="1", call_id="call-0"),
            ExternalEvent(kind="message", content="answer"),
        )
        for turn in _project(events).turns:
            response = turn.generation_response
            assert response.provenance is None
            assert response.prompt_token_ids is None
            assert response.response_token_ids is None
            assert response.response_logprobs is None
            assert response.is_trainable is False

    def test_result_metadata_marks_the_policy_source(self) -> None:
        result = _project((ExternalEvent(kind="message", content="answer"),))
        assert result.metadata["policy_source"] == POLICY_SOURCE_EXTERNAL_AGENT
        assert result.metadata["trainable"] is False
        assert result.metadata["external"]["agent"] == "fake"
        assert result.metadata["external"]["model"] == "model-x"

    def test_parallel_environment_steps_are_retained_in_one_generation_turn(self) -> None:
        events = (
            ExternalEvent(kind="tool_call", tool_call=_tool_call(0)),
            ExternalEvent(kind="tool_call", tool_call=_tool_call(1, name="read")),
            ExternalEvent(kind="tool_result", content="a", call_id="call-0", tool_id="bash"),
            ExternalEvent(kind="tool_result", content="b", call_id="call-1", tool_id="read"),
            ExternalEvent(kind="message", content="answer"),
        )
        outcome = ExternalRunOutcome(final_text="answer", events=events)

        result = project_outcome(
            _task(),
            outcome,
            agent="fake",
            model="m",
            sampling=SAMPLING,
            environment_transitions=(
                _real_transition(tool_id="read"),
                _real_transition(tool_id="bash"),
                _real_transition(done=True, reason="model_output"),
            ),
        )

        aggregate = result.turns[0].environment_transition
        assert aggregate.reward == 1.0
        assert aggregate.metadata["external_environment"]["aggregated_parallel_steps"] == 2
        assert aggregate.metadata["external_environment"]["environment_stepped"] is True
        assert aggregate.metadata["external_environment"]["synthesized"] is True
        assert result.turns[1].environment_transition.termination_reason == "model_output"

    def test_environment_termination_discards_later_external_turns(self) -> None:
        events = (
            ExternalEvent(kind="tool_call", tool_call=_tool_call()),
            ExternalEvent(kind="tool_result", content="limit", call_id="call-0", tool_id="bash"),
            ExternalEvent(kind="message", content="answer after the limit"),
        )
        outcome = ExternalRunOutcome(final_text="answer after the limit", events=events)

        result = project_outcome(
            _task(),
            outcome,
            agent="fake",
            model="m",
            sampling=SAMPLING,
            environment_transitions=(
                _real_transition(tool_id="bash", done=True, reason="max_steps"),
            ),
        )

        assert len(result.turns) == 1
        assert result.termination_reason == "max_steps"
        assert result.final_text == "answer after the limit"
        assert result.metadata["environment"]["discarded_post_termination_turns"] == 1
        assert result.metadata["environment"]["bounded_after_termination"] is True

    def test_terminal_parallel_step_retains_answer_and_accounts_for_rejected_call(self) -> None:
        events = (
            ExternalEvent(kind="tool_call", tool_call=_tool_call(0)),
            ExternalEvent(kind="tool_call", tool_call=_tool_call(1, name="read")),
            ExternalEvent(kind="tool_result", content="limit", call_id="call-0", tool_id="bash"),
            ExternalEvent(kind="tool_result", content="rejected", call_id="call-1", tool_id="read"),
            ExternalEvent(kind="message", content="bounded answer"),
        )
        outcome = ExternalRunOutcome(final_text="bounded answer", events=events)

        result = project_outcome(
            _task(),
            outcome,
            agent="fake",
            model="m",
            sampling=SAMPLING,
            environment_transitions=(
                # Parallel dispatch order is not guaranteed to match event
                # order: the second reported call may reach the socket first.
                _real_transition(tool_id="read", done=True, reason="max_tool_calls"),
            ),
            rejected_post_termination_calls=1,
        )

        assert result.final_text == "bounded answer"
        assert result.termination_reason == "max_tool_calls"
        assert result.metadata["environment"]["rejected_post_termination_tool_calls"] == 1
        assert result.metadata["environment"]["bounded_after_termination"] is True

    def test_task_metadata_is_preserved_alongside_the_labels(self) -> None:
        outcome = ExternalRunOutcome(
            final_text="answer", events=(ExternalEvent(kind="message", content="answer"),)
        )
        task = _task(metadata={"role": "proposer"})
        result = project_outcome(task, outcome, agent="fake", model="m", sampling=SAMPLING)
        assert result.metadata["role"] == "proposer"
        assert result.metadata["trainable"] is False

    def test_usage_is_attributed_once_to_the_terminal_turn(self) -> None:
        events = (
            ExternalEvent(kind="tool_call", tool_call=_tool_call()),
            ExternalEvent(kind="tool_result", content="1", call_id="call-0"),
            ExternalEvent(kind="message", content="answer"),
        )
        result = _project(events, usage={"output_tokens": 7})
        assert result.turns[0].generation_response.usage == {}
        assert result.turns[-1].generation_response.usage == {"output_tokens": 7}


class TestRunBatchContract:
    def test_environment_backed_run_retains_real_transitions(self) -> None:
        environments: list[_RecordingEnvironment] = []

        def environment_factory(_task: AgentTask) -> _RecordingEnvironment:
            environment = _RecordingEnvironment()
            environments.append(environment)
            return environment

        runtime = ExternalAgentRuntime(
            _EnvironmentCallingSession,
            agent="fake",
            model="m",
            environment_factory=environment_factory,
        )

        result = runtime.run(_task(tools=("bash",)))

        environment = environments[0]
        assert environment.initialized is True
        assert environment.closed is True
        assert len(result.turns) == 2
        assert result.turns[0].environment_transition.observation == (
            environment.transitions[0].observation
        )
        assert result.turns[1].environment_transition.observation == (
            environment.transitions[1].observation
        )
        assert result.metadata["environment"] == {
            "stepped": True,
            "transitions": 2,
            "discarded_post_termination_turns": 0,
            "rejected_post_termination_tool_calls": 0,
            "bounded_after_termination": False,
        }
        marker = result.turns[0].environment_transition.metadata["external_environment"]
        assert marker == {"environment_stepped": True, "synthesized": False}
        assert result.termination_reason == "model_output"

    @pytest.mark.parametrize("concurrency", [1, 4])
    def test_fresh_environments_remain_distinct_across_a_large_batch(
        self, concurrency: int
    ) -> None:
        environments: list[_RecordingEnvironment] = []

        def environment_factory(_task: AgentTask) -> _RecordingEnvironment:
            environment = _RecordingEnvironment()
            environments.append(environment)
            return environment

        runtime = ExternalAgentRuntime(
            session_factory("fake"),
            agent="fake",
            model="m",
            environment_factory=environment_factory,
            concurrency=concurrency,
        )

        results = runtime.run_batch([_task(f"task-{index}") for index in range(30)])

        assert len(results) == 30
        assert len({id(environment) for environment in environments}) == 30
        assert all(environment.closed for environment in environments)

    def test_external_and_native_context_fields_and_initial_observation_are_shared(self) -> None:
        captured: list[object] = []
        seen_prompts: list[str] = []

        class InitialEnvironment(_RecordingEnvironment):
            def init(self, context: object) -> EnvironmentInitResult:
                captured.append(context)
                self.initialized = True
                return EnvironmentInitResult(
                    observation="Environment-projected problem",
                    metadata={"source": "environment"},
                )

        class PromptSession(FakeExternalSession):
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                seen_prompts.append(task.prompt)
                return super().run(task, workspace=workspace)

        runtime = ExternalAgentRuntime(
            PromptSession,
            agent="fake",
            model="m",
            environment_factory=lambda _task: InitialEnvironment(),
        )
        result = runtime.run(_task("context", sample_id=7, routing_key="sticky", sampling_seed=13))

        context = captured[0]
        assert context.sample_id == 7  # type: ignore[attr-defined]
        assert context.routing_key == "sticky"  # type: ignore[attr-defined]
        assert context.seed == 13  # type: ignore[attr-defined]
        assert context.metadata == {}  # type: ignore[attr-defined]
        assert seen_prompts == ["Environment-projected problem"]
        assert result.turns[0].generation_request.messages[1]["content"] == (
            "Environment-projected problem"
        )
        assert result.metadata["environment_init"] == {"source": "environment"}

    def test_memory_context_reaches_external_agent_but_not_environment(self) -> None:
        captured: list[object] = []
        seen_prompts: list[str] = []

        class InitialEnvironment(_RecordingEnvironment):
            def init(self, context: object) -> EnvironmentInitResult:
                captured.append(context)
                self.initialized = True
                return EnvironmentInitResult(observation="Environment-projected problem")

        class PromptSession(FakeExternalSession):
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                seen_prompts.append(task.prompt)
                return super().run(task, workspace=workspace)

        memory_context = "# Workflow memory\nselected parent and SEARCH/REPLACE contract"
        runtime = ExternalAgentRuntime(
            PromptSession,
            agent="fake",
            model="m",
            environment_factory=lambda _task: InitialEnvironment(),
        )
        runtime.run(
            _task(
                "memory-context",
                prompt=f"public problem\n\n{memory_context}",
                metadata={
                    "environment_user_prompt": "public problem",
                    "agent_memory_context": memory_context,
                },
            )
        )

        context = captured[0]
        assert context.user_prompt == "public problem"  # type: ignore[attr-defined]
        assert "agent_memory_context" not in context.metadata  # type: ignore[attr-defined]
        assert seen_prompts == [f"Environment-projected problem\n\n{memory_context}"]

    def test_private_payload_reaches_only_the_environment_boundary(self) -> None:
        factory_tasks: list[AgentTask] = []
        contexts: list[object] = []
        session_tasks: list[AgentTask] = []

        class PayloadEnvironment(_RecordingEnvironment):
            def init(self, context: object) -> EnvironmentInitResult:
                contexts.append(context)
                return super().init(context)

        class CapturingSession(FakeExternalSession):
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                session_tasks.append(task)
                return super().run(task, workspace=workspace)

        def environment_factory(task: AgentTask) -> PayloadEnvironment:
            factory_tasks.append(task)
            return PayloadEnvironment()

        runtime = ExternalAgentRuntime(
            CapturingSession,
            agent="fake",
            model="m",
            environment_factory=environment_factory,
        )
        task = _task(
            "private",
            metadata={"source": "public"},
            task_payload={
                "answer": "PRIVATE-ANSWER-CANARY",
                "token": "PRIVATE-TOKEN-CANARY",
            },
        )

        result = runtime.run(task)

        assert factory_tasks == [task]
        assert factory_tasks[0].task_payload == {
            "answer": "PRIVATE-ANSWER-CANARY",
            "token": "PRIVATE-TOKEN-CANARY",
        }
        context = contexts[0]
        assert context.task_payload == {  # type: ignore[attr-defined]
            "answer": "PRIVATE-ANSWER-CANARY",
            "token": "PRIVATE-TOKEN-CANARY",
        }
        assert session_tasks[0].task_payload == {}
        assert session_tasks[0].metadata == {"source": "public"}
        assert result.metadata["source"] == "public"
        assert "PRIVATE-ANSWER-CANARY" not in repr(result)
        assert "PRIVATE-TOKEN-CANARY" not in repr(result)

    def test_unbridged_tool_activity_cannot_masquerade_as_environment_steps(self) -> None:
        runtime = ExternalAgentRuntime(
            session_factory(
                "fake",
                {
                    "tool_id": "bash",
                    "tool_arguments": {"command": "echo 1"},
                    "tool_result": "1",
                },
            ),
            agent="fake",
            model="m",
            environment_factory=lambda _task: _RecordingEnvironment(),
        )

        with pytest.raises(RuntimeError, match="did not match"):
            runtime.run(_task(tools=("bash",)))

    def test_workspace_hooks_wrap_a_successful_external_session(self) -> None:
        observed: list[str] = []

        def prepare(task: AgentTask, workspace: Path) -> None:
            assert task.task_id == "hooked"
            (workspace / "scratchpad.md").write_text("before", encoding="utf-8")

        def observe(task: AgentTask, workspace: Path) -> None:
            assert task.task_id == "hooked"
            observed.append((workspace / "scratchpad.md").read_text(encoding="utf-8"))

        runtime = ExternalAgentRuntime(
            session_factory("fake"),
            agent="fake",
            model="m",
            workspace_prepare=prepare,
            workspace_observe=observe,
        )

        result = runtime.run(_task("hooked"))

        assert result.termination_reason == "final"
        assert result.metadata["runtime_seconds"] >= 0.0
        assert observed == ["before"]

    def test_environment_budget_stops_later_calls_without_losing_the_answer(self) -> None:
        environments: list[_RecordingEnvironment] = []

        class TerminalEnvironment(_RecordingEnvironment):
            def step(self, action: object) -> EnvironmentTransition:
                self.actions.append(action)
                transition = EnvironmentTransition(
                    observation="budget reached",
                    reward=0.0,
                    done=True,
                    success=False,
                    termination_reason="max_tool_calls",
                    metadata={"tool_request": {"tool_id": "bash"}},
                )
                self.transitions.append(transition)
                return transition

        class IgnoringSession:
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                bridge = EnvironmentSocketToolBridge(workspace / ENVIRONMENT_SOCKET_NAME)
                calls = (_tool_call(0), _tool_call(1))
                payloads = [
                    bridge.dispatch(
                        {
                            "id": call.id,
                            "function": {"name": call.name, "arguments": {}},
                        },
                        None,
                    ).observation_payload()
                    for call in calls
                ]
                return ExternalRunOutcome(
                    final_text="The final answer is 204.",
                    events=(
                        *(ExternalEvent(kind="tool_call", tool_call=call) for call in calls),
                        *(
                            ExternalEvent(
                                kind="tool_result",
                                content=json.dumps(payload),
                                call_id=call.id,
                                tool_id=call.name,
                                failed=not payload["ok"],
                            )
                            for call, payload in zip(calls, payloads, strict=True)
                        ),
                        ExternalEvent(kind="message", content="The final answer is 204."),
                    ),
                )

            def close(self) -> None:
                return None

        def environment_factory(_task: AgentTask) -> TerminalEnvironment:
            environment = TerminalEnvironment()
            environments.append(environment)
            return environment

        result = ExternalAgentRuntime(
            IgnoringSession,
            agent="fake",
            model="m",
            environment_factory=environment_factory,
        ).run(_task(tools=("bash",)))

        assert result.final_text == "The final answer is 204."
        assert result.termination_reason == "max_tool_calls"
        assert len(environments[0].actions) == 1
        assert result.metadata["environment"]["rejected_post_termination_tool_calls"] == 1
        assert result.metadata["environment"]["bounded_after_termination"] is True

    def test_preserves_order_and_identity(self) -> None:
        runtime = ExternalAgentRuntime(
            session_factory("fake", {"echo_prompt": True}), agent="fake", model="m"
        )
        tasks = [_task(f"task-{index}", prompt=f"problem-{index}") for index in range(4)]
        results = runtime.run_batch(tasks)

        assert [result.task_id for result in results] == [task.task_id for task in tasks]
        assert [result.final_text.splitlines()[-1] for result in results] == [
            f"problem-{index}" for index in range(4)
        ]

    def test_concurrent_execution_still_preserves_order(self) -> None:
        runtime = ExternalAgentRuntime(
            session_factory("fake", {"echo_prompt": True}),
            agent="fake",
            model="m",
            concurrency=4,
        )
        tasks = [_task(f"task-{index}", prompt=f"problem-{index}") for index in range(8)]
        results = runtime.run_batch(tasks)
        assert [result.task_id for result in results] == [task.task_id for task in tasks]

    def test_reuses_one_verified_session_for_the_same_workflow_branch(self, tmp_path: Path) -> None:
        class ContinuingSession:
            supports_continuation = True

            def __init__(self) -> None:
                self.tasks: list[str] = []
                self.workspaces: list[Path] = []
                self.closed = False
                created.append(self)

            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                self.tasks.append(task.task_id)
                self.workspaces.append(workspace)
                return ExternalRunOutcome(
                    final_text=task.prompt,
                    events=(ExternalEvent(kind="message", content=task.prompt),),
                )

            def close(self) -> None:
                self.closed = True

        created: list[ContinuingSession] = []
        metadata = {
            "workflow_name": "iterative",
            "workflow_input_id": "problem",
            "branch_index": 0,
        }
        runtime = ExternalAgentRuntime(
            ContinuingSession,
            agent="codex",
            model="m",
            workspace_root=tmp_path,
            keep_workspaces=True,
            reuse_workspace=True,
            reuse_session=True,
        )

        runtime.run_batch([_task("first", metadata=metadata)])
        runtime.run_batch([_task("second", metadata=metadata)])

        assert len(created) == 1
        assert created[0].tasks == ["first", "second"]
        assert created[0].workspaces[0] == created[0].workspaces[1]
        assert created[0].closed is False
        runtime.close()
        assert created[0].closed is True

    def test_reuse_session_rejects_an_unverified_provider(self, tmp_path: Path) -> None:
        runtime = ExternalAgentRuntime(
            session_factory("fake"),
            agent="fake",
            model="m",
            workspace_root=tmp_path,
            reuse_workspace=True,
            reuse_session=True,
        )
        task = _task(
            metadata={
                "workflow_name": "iterative",
                "workflow_input_id": "problem",
                "branch_index": 0,
            }
        )

        with pytest.raises(TypeError, match="verified conversation continuation"):
            runtime.run_batch([task])

    def test_empty_batch_returns_empty_list(self) -> None:
        runtime = ExternalAgentRuntime(session_factory("fake"), agent="fake", model="m")
        assert runtime.run_batch(()) == []

    def test_duplicate_task_ids_are_rejected(self) -> None:
        runtime = ExternalAgentRuntime(session_factory("fake"), agent="fake", model="m")
        with pytest.raises(ValueError, match="unique task_id"):
            runtime.run_batch([_task("same"), _task("same")])

    def test_run_rejects_a_session_returning_the_wrong_record(self) -> None:
        class WrongSession:
            def run(self, task: AgentTask, *, workspace: Path) -> str:
                del task, workspace
                return "not an outcome"

            def close(self) -> None:
                return None

        runtime = ExternalAgentRuntime(WrongSession, agent="fake", model="m")
        with pytest.raises(TypeError, match="expected ExternalRunOutcome"):
            runtime.run_batch([_task()])

    def test_session_failure_propagates_and_still_closes_sessions(self) -> None:
        created: list[object] = []

        class FailingSession:
            def __init__(self) -> None:
                self.closed = False
                created.append(self)

            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                del workspace
                raise RuntimeError(f"cli exploded for {task.task_id}")

            def close(self) -> None:
                self.closed = True

        runtime = ExternalAgentRuntime(FailingSession, agent="fake", model="m")
        with pytest.raises(RuntimeError, match="cli exploded"):
            runtime.run_batch([_task()])
        assert created and all(session.closed for session in created)  # type: ignore[attr-defined]

    def test_close_failure_does_not_mask_the_original_error(self) -> None:
        class BadCloseSession:
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                del workspace
                raise RuntimeError("primary failure")

            def close(self) -> None:
                raise RuntimeError("close failure")

        runtime = ExternalAgentRuntime(BadCloseSession, agent="fake", model="m")
        with pytest.raises(RuntimeError, match="primary failure"):
            runtime.run_batch([_task()])

    def test_close_failure_surfaces_when_the_batch_succeeded(self) -> None:
        class BadCloseSession(FakeExternalSession):
            def close(self) -> None:
                raise RuntimeError("close failure")

        runtime = ExternalAgentRuntime(BadCloseSession, agent="fake", model="m")
        with pytest.raises(RuntimeError, match="close failure"):
            runtime.run_batch([_task()])

    def test_closed_runtime_refuses_further_work(self) -> None:
        runtime = ExternalAgentRuntime(session_factory("fake"), agent="fake", model="m")
        runtime.terminate("workflow_shutdown")
        with pytest.raises(RuntimeError, match="is closed"):
            runtime.run_batch([_task()])


class TestWorkspaces:
    def test_each_task_gets_a_distinct_existing_workspace(self, tmp_path: Path) -> None:
        sessions: list[FakeExternalSession] = []

        def factory() -> FakeExternalSession:
            session = FakeExternalSession()
            sessions.append(session)
            return session

        runtime = ExternalAgentRuntime(
            factory,
            agent="fake",
            model="m",
            workspace_root=tmp_path / "workspaces",
            keep_workspaces=True,
        )
        runtime.run_batch([_task("a"), _task("b")])

        used = [workspace for session in sessions for workspace in session.workspaces]
        assert len(set(used)) == 2
        assert all(workspace.is_dir() for workspace in used)

    def test_supplied_root_still_honours_keep_workspaces_false(self, tmp_path: Path) -> None:
        sessions: list[FakeExternalSession] = []

        def factory() -> FakeExternalSession:
            session = FakeExternalSession()
            sessions.append(session)
            return session

        root = tmp_path / "workspaces"
        runtime = ExternalAgentRuntime(factory, agent="fake", model="m", workspace_root=root)
        runtime.run_batch([_task("a")])

        assert not sessions[0].workspaces[0].exists()
        assert root.is_dir()

    def test_temporary_workspaces_are_removed_after_the_batch(self) -> None:
        sessions: list[FakeExternalSession] = []

        def factory() -> FakeExternalSession:
            session = FakeExternalSession()
            sessions.append(session)
            return session

        runtime = ExternalAgentRuntime(factory, agent="fake", model="m")
        runtime.run_batch([_task("a")])
        assert not sessions[0].workspaces[0].exists()

    def test_task_ids_with_separators_stay_inside_the_workspace_root(self, tmp_path: Path) -> None:
        sessions: list[FakeExternalSession] = []

        def factory() -> FakeExternalSession:
            session = FakeExternalSession()
            sessions.append(session)
            return session

        root = tmp_path / "workspaces"
        runtime = ExternalAgentRuntime(
            factory, agent="fake", model="m", workspace_root=root, keep_workspaces=True
        )
        runtime.run_batch([_task("input/../../escape:branch-0")])
        assert sessions[0].workspaces[0].parent == root

    def test_default_mode_never_reuses_a_kept_workspace_across_calls(self, tmp_path: Path) -> None:
        sessions: list[FakeExternalSession] = []

        def factory() -> FakeExternalSession:
            session = FakeExternalSession()
            sessions.append(session)
            return session

        runtime = ExternalAgentRuntime(
            factory,
            agent="fake",
            model="m",
            workspace_root=tmp_path,
            keep_workspaces=True,
        )

        runtime.run_batch([_task("same-task")])
        first = sessions[-1].workspaces[0]
        (first / "marker").write_text("first", encoding="utf-8")
        runtime.run_batch([_task("same-task")])
        second = sessions[-1].workspaces[0]

        assert first != second
        assert not (second / "marker").exists()

    def test_reuse_workspace_uses_the_same_directory_across_rounds(self, tmp_path: Path) -> None:
        sessions: list[FakeExternalSession] = []

        def factory() -> FakeExternalSession:
            session = FakeExternalSession()
            sessions.append(session)
            return session

        root = tmp_path / "workspaces"
        runtime = ExternalAgentRuntime(
            factory,
            agent="fake",
            model="m",
            workspace_root=root,
            reuse_workspace=True,
        )
        metadata = {
            "workflow_name": "iterative-search",
            "workflow_input_id": "problem-1",
            "branch_index": 0,
        }

        runtime.run_batch([_task("round-1", metadata=metadata)])
        first = sessions[-1].workspaces[0]
        marker = first / "algorithm.py"
        marker.write_text("round one", encoding="utf-8")
        runtime.run_batch([_task("round-2", metadata=metadata)])
        second = sessions[-1].workspaces[0]

        assert first == second
        assert second == workflow_workspace_path(
            root,
            workflow_name="iterative-search",
            input_id="problem-1",
            branch_index=0,
        )
        assert marker.read_text(encoding="utf-8") == "round one"
        assert second.is_dir()

        runtime.close()
        assert not second.exists()
        assert root.is_dir()

    def test_reuse_workspace_isolates_inputs_and_branches(self, tmp_path: Path) -> None:
        sessions: list[FakeExternalSession] = []

        def factory() -> FakeExternalSession:
            session = FakeExternalSession()
            sessions.append(session)
            return session

        runtime = ExternalAgentRuntime(
            factory,
            agent="fake",
            model="m",
            workspace_root=tmp_path,
            reuse_workspace=True,
            keep_workspaces=True,
        )
        tasks = [
            _task(
                f"task-{index}",
                metadata={
                    "workflow_name": "workflow",
                    "workflow_input_id": input_id,
                    "branch_index": branch,
                },
            )
            for index, (input_id, branch) in enumerate((("a", 0), ("a", 1), ("b", 0)))
        ]

        runtime.run_batch(tasks)
        used = [session.workspaces[0] for session in sessions]
        runtime.close()

        assert len(set(used)) == 3
        assert all(workspace.is_dir() for workspace in used)

    def test_reuse_workspace_requires_workflow_identity(self, tmp_path: Path) -> None:
        runtime = ExternalAgentRuntime(
            lambda: FakeExternalSession(),
            agent="fake",
            model="m",
            workspace_root=tmp_path,
            reuse_workspace=True,
        )

        with pytest.raises(ValueError, match="metadata.workflow_name"):
            runtime.run_batch([_task("round-1")])

    def test_reuse_workspace_refuses_duplicate_branch_identity(self, tmp_path: Path) -> None:
        runtime = ExternalAgentRuntime(
            lambda: FakeExternalSession(),
            agent="fake",
            model="m",
            workspace_root=tmp_path,
            reuse_workspace=True,
        )
        metadata = {
            "workflow_name": "workflow",
            "workflow_input_id": "same",
            "branch_index": 0,
        }

        with pytest.raises(ValueError, match="unique workflow_input_id"):
            runtime.run_batch(
                [_task("first", metadata=metadata), _task("second", metadata=metadata)]
            )

    def test_reuse_workspace_refuses_concurrent_use_of_the_same_branch(
        self, tmp_path: Path
    ) -> None:
        entered = threading.Event()
        release = threading.Event()
        thread_errors: list[BaseException] = []

        class BlockingSession:
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                del task, workspace
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release blocking session")
                return ExternalRunOutcome(final_text="done")

            def close(self) -> None:
                pass

        runtime = ExternalAgentRuntime(
            BlockingSession,
            agent="fake",
            model="m",
            workspace_root=tmp_path,
            reuse_workspace=True,
        )
        metadata = {
            "workflow_name": "workflow",
            "workflow_input_id": "same",
            "branch_index": 0,
        }

        def run_first() -> None:
            try:
                runtime.run_batch([_task("first", metadata=metadata)])
            except BaseException as exc:  # noqa: BLE001 - asserted below
                thread_errors.append(exc)

        thread = threading.Thread(target=run_first)
        thread.start()
        assert entered.wait(timeout=5)
        try:
            with pytest.raises(RuntimeError, match="cannot run.*concurrently"):
                runtime.run_batch([_task("second", metadata=metadata)])
            with pytest.raises(RuntimeError, match="while a batch is active"):
                runtime.close()
        finally:
            release.set()
            thread.join(timeout=5)

        assert not thread.is_alive()
        assert not thread_errors
        runtime.close()

    def test_reused_workspace_is_cleaned_after_a_failed_batch_and_close(
        self, tmp_path: Path
    ) -> None:
        used: list[Path] = []

        class FailingSession:
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                del task
                used.append(workspace)
                (workspace / "partial.py").write_text("partial", encoding="utf-8")
                raise RuntimeError("agent failed")

            def close(self) -> None:
                pass

        runtime = ExternalAgentRuntime(
            FailingSession,
            agent="fake",
            model="m",
            workspace_root=tmp_path / "workspaces",
            reuse_workspace=True,
        )
        metadata = {
            "workflow_name": "workflow",
            "workflow_input_id": "input",
            "branch_index": 0,
        }

        with pytest.raises(RuntimeError, match="agent failed"):
            runtime.run_batch([_task("failed", metadata=metadata)])
        assert used[0].is_dir()

        runtime.close()
        assert not used[0].exists()

    def test_reused_workspace_path_cannot_escape_its_root(self, tmp_path: Path) -> None:
        path = workflow_workspace_path(
            tmp_path,
            workflow_name="../../workflow",
            input_id="../../input",
            branch_index=0,
        )

        assert path.parent == tmp_path


class TestConstruction:
    def test_rejects_a_non_callable_factory(self) -> None:
        with pytest.raises(TypeError, match="session_factory must be callable"):
            ExternalAgentRuntime("claude", agent="fake", model="m")  # type: ignore[arg-type]

    def test_rejects_non_positive_concurrency(self) -> None:
        with pytest.raises(ValueError, match="concurrency must be at least one"):
            ExternalAgentRuntime(session_factory("fake"), agent="fake", model="m", concurrency=0)

    def test_rejects_non_boolean_reuse_workspace(self) -> None:
        with pytest.raises(TypeError, match="reuse_workspace must be a bool"):
            ExternalAgentRuntime(
                session_factory("fake"),
                agent="fake",
                model="m",
                reuse_workspace="yes",  # type: ignore[arg-type]
            )

    def test_registry_rejects_an_unknown_agent(self) -> None:
        with pytest.raises(ValueError, match="unknown external agent"):
            session_factory("gpt-cli")

    def test_registry_rejects_unknown_session_options(self) -> None:
        create = session_factory("fake", {"nonexistent": 1})
        with pytest.raises(TypeError):
            create()


class TestEncodeArguments:
    def test_passes_strings_through_unchanged(self) -> None:
        assert encode_arguments('{"a": 1}') == '{"a": 1}'

    def test_encodes_mappings_deterministically(self) -> None:
        assert encode_arguments({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'


def test_sessions_are_created_per_task_not_shared(tmp_path: Path) -> None:
    seen: list[int] = []
    lock = threading.Lock()

    class CountingSession(FakeExternalSession):
        def __init__(self) -> None:
            super().__init__()
            with lock:
                seen.append(id(self))

    runtime = ExternalAgentRuntime(
        CountingSession, agent="fake", model="m", workspace_root=tmp_path, concurrency=2
    )
    runtime.run_batch([_task("a"), _task("b"), _task("c")])
    assert len(set(seen)) == 3


class TestSafeInput:
    def test_private_payload_without_an_environment_fails_before_session_creation(self) -> None:
        sessions_created = 0

        def create_session() -> FakeExternalSession:
            nonlocal sessions_created
            sessions_created += 1
            return FakeExternalSession(answer="should not run")

        runtime = ExternalAgentRuntime(create_session, agent="fake", model="scripted")

        with pytest.raises(ValueError, match="task_payload requires environment_factory"):
            runtime.run(_task(task_payload={"answer": "PRIVATE-ANSWER-CANARY"}))

        assert sessions_created == 0

    def test_evaluator_only_material_never_reaches_an_external_process(self) -> None:
        # The same gate DefaultEnvironment applies, so a grading key smuggled
        # through metadata fails before a subprocess is ever spawned.
        runtime = ExternalAgentRuntime(
            lambda: FakeExternalSession(answer="ok"), agent="fake", model="scripted"
        )
        task = AgentTask(
            task_id="leaky",
            system="be terse",
            prompt="solve",
            metadata={"answer_key": "204"},
        )

        with pytest.raises(SensitiveEnvironmentInputError):
            runtime.run(task)

    def test_an_ordinary_problem_statement_still_runs(self) -> None:
        runtime = ExternalAgentRuntime(
            lambda: FakeExternalSession(answer="Final answer: 204"),
            agent="fake",
            model="scripted",
        )
        task = AgentTask(task_id="clean", system="be terse", prompt="what is 9/3*60 + 24?")

        assert "204" in runtime.run(task).final_text


class TestProcessGroup:
    def test_agent_stream_decodes_utf8_independent_of_host_locale(self, tmp_path: Path) -> None:
        run = run_cli(
            [sys.executable, "-c", "import sys;sys.stdout.buffer.write('芯片'.encode('utf-8'))"],
            prompt="",
            workspace=tmp_path,
            settings=CliSettings(cli=sys.executable, timeout_s=5),
        )
        assert run.stdout == "芯片"

    def test_a_timeout_kills_what_the_agent_spawned_too(self, tmp_path: Path) -> None:
        # Killing only the agent orphans the shell commands and MCP servers it
        # was running, which is how a batch evaluation goes flaky rather than
        # slow. The child prints its grandchild's pid before hanging.
        run = run_cli(
            ["bash", "-c", "sleep 60 & echo $!; wait"],
            prompt="",
            workspace=tmp_path,
            settings=CliSettings(cli="bash", timeout_s=0.5),
        )

        assert run.timed_out is True
        grandchild = int(run.stdout.strip())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        raise AssertionError(f"pid {grandchild} outlived the process group")

    def test_a_kill_mid_record_keeps_the_records_that_completed(self, tmp_path: Path) -> None:
        # The kill lands wherever the CLI's stdout buffer was, so a real timeout
        # usually ends mid-record. Every vendor parser rejects a malformed line
        # and the Runtime is fail-fast, so keeping the fragment would abort the
        # whole batch on a defect the timeout itself created.
        run = run_cli(
            ["bash", "-c", r"""printf '{"type":"a"}\n{"type":"par'; sleep 60"""],
            prompt="",
            workspace=tmp_path,
            settings=CliSettings(cli="bash", timeout_s=0.5),
        )

        assert run.timed_out is True
        assert run.stdout == '{"type":"a"}\n'
        assert run.dropped_partial_record is True


class TestFinalResponseMode:
    def test_native_activity_precedes_one_authoritative_evaluation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from alphaapollo.reasoning.runtime.external.bridge import environment_socket

        def forbidden_socket(*args: object, **kwargs: object) -> None:
            pytest.fail("final_response must not create an Environment socket")

        monkeypatch.setattr(environment_socket, "RuntimeEnvironmentSocket", forbidden_socket)
        environment = _RecordingEnvironment()
        session = FakeExternalSession(
            answer="answer", tool_id="bash", tool_arguments={"command": "echo 1"}, tool_result="1"
        )
        runtime = ExternalAgentRuntime(
            lambda: session,
            agent="fake",
            model="m",
            environment_factory=lambda task: environment,
            environment_mode="final_response",
            workspace_root=tmp_path,
        )

        result = runtime.run(_task())

        assert environment.actions == ["answer"]
        assert environment.closed and session.closed
        assert not session.workspaces[0].exists()
        native, final = result.turns
        assert native.generation_response.tool_calls[0].name == "bash"
        assert native.environment_transition.observation.content == "1"
        assert native.environment_transition.reward == 0.0
        assert native.environment_transition.done is False
        assert native.environment_transition.metadata["external"]["synthesized"] is True
        assert final.generation_response.content == "answer"
        assert final.environment_transition.reward == 1.0
        assert final.environment_transition.metadata["external_environment"] == {
            "environment_stepped": True,
            "synthesized": False,
        }
        assert result.termination_reason == "model_output"
        assert result.metadata["environment"]["mode"] == "final_response"
        assert result.metadata["environment"]["transitions"] == 1
        assert result.metadata["external"]["instruction_delivery"] == "user_context"
        assert result.metadata["trainable"] is False
        assert all(not turn.generation_response.is_trainable for turn in result.turns)
        assert all(turn.generation_response.response_token_ids is None for turn in result.turns)

    @pytest.mark.parametrize(
        ("events", "final_text", "expected_turns"),
        [
            ((), "", 1),
            ((), "answer", 1),
            ((ExternalEvent(kind="reasoning", content="thinking"),), "answer", 2),
            ((ExternalEvent(kind="tool_call", tool_call=_tool_call()),), "answer", 2),
            ((ExternalEvent(kind="message", content="interim"),), "answer", 2),
            ((ExternalEvent(kind="message", content="answer"),), "answer", 1),
            ((ExternalEvent(kind="error", content="failed", failed=True),), "", 2),
        ],
    )
    def test_projection_evaluates_final_text_without_losing_native_events(
        self, events: tuple[ExternalEvent, ...], final_text: str, expected_turns: int
    ) -> None:
        transition = _real_transition(done=True, reason="evaluated")
        result = project_outcome(
            _task(),
            ExternalRunOutcome(final_text=final_text, events=events),
            agent="fake",
            model="m",
            sampling=SAMPLING,
            environment_mode="final_response",
            environment_transitions=[transition],
        )
        assert len(result.turns) == expected_turns
        assert result.turns[-1].generation_response.content == final_text
        assert result.turns[-1].generation_response.tool_calls == ()
        assert result.turns[-1].environment_transition.reward == 0.5
        assert all(turn.environment_transition.reward == 0 for turn in result.turns[:-1])
        assert len(result.metadata["external"]["events"]) == len(events)

    @pytest.mark.parametrize("reason", ["timeout", "external_error", "cancelled", "truncated"])
    def test_evaluation_does_not_erase_external_failure_reason(self, reason: str) -> None:
        environment = _RecordingEnvironment()
        outcome = ExternalRunOutcome(
            final_text="partial",
            termination_reason=reason,
            events=(ExternalEvent(kind="error", content="native failure", raw={"code": 17}),),
        )

        class Session(FakeExternalSession):
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                return outcome

        runtime = ExternalAgentRuntime(
            Session,
            agent="fake",
            model="m",
            environment_mode="final_response",
            environment_factory=lambda task: environment,
        )
        result = runtime.run(_task())
        assert environment.actions == ["partial"]
        assert result.termination_reason == reason
        assert result.turns[-1].generation_response.finish_reason == reason
        assert result.turns[-1].environment_transition.termination_reason == "model_output"
        assert result.metadata["external"]["events"][0]["raw"] == {"code": 17}

    @pytest.mark.parametrize("observation", [None, "", "public Environment problem"])
    def test_native_system_and_memory_context_delivery_preserves_payload_isolation(
        self, observation: str | None
    ) -> None:
        contexts: list[object] = []
        tasks: list[AgentTask] = []
        memory = "# Workflow memory\nparent and diff instructions"

        class Environment(_RecordingEnvironment):
            def init(self, context: object) -> EnvironmentInitResult:
                contexts.append(context)
                return EnvironmentInitResult(observation=observation)

        class Session(FakeExternalSession):
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                tasks.append(task)
                return super().run(task, workspace=workspace)

        runtime = ExternalAgentRuntime(
            Session,
            agent="fake",
            model="m",
            environment_mode="final_response",
            environment_factory=lambda task: Environment(),
        )
        task = _task(
            prompt=f"public problem\n\n{memory}",
            task_payload={"answer": "PRIVATE-CANARY"},
            metadata={"environment_user_prompt": "public problem", "agent_memory_context": memory},
        )
        result = runtime.run(task)
        session_task = tasks[0]
        assert session_task.system == ""
        assert session_task.prompt.count(task.system) == 1
        assert session_task.prompt.count(memory) == 1
        assert (observation or "public problem") in session_task.prompt
        assert session_task.task_payload == {}
        assert contexts[0].task_payload == {"answer": "PRIVATE-CANARY"}
        assert contexts[0].system_prompt == task.system
        assert contexts[0].user_prompt == "public problem"
        assert "agent_memory_context" not in contexts[0].metadata
        assert "PRIVATE-CANARY" not in repr(result)
        assert result.turns[0].generation_request.messages[0]["content"] == ""
        assert result.turns[0].generation_request.messages[1]["content"] == session_task.prompt

    @pytest.mark.parametrize(
        ("task", "message"),
        [(_task(tools=("bash",)), "tools"), (_task(model="other"), "model")],
    )
    def test_rejects_conflicting_tasks_before_any_execution(
        self, tmp_path: Path, task: AgentTask, message: str
    ) -> None:
        def forbidden(*args: object) -> object:
            pytest.fail("validation must precede factory execution")

        runtime = ExternalAgentRuntime(
            forbidden,
            agent="fake",
            model="m",
            environment_mode="final_response",
            environment_factory=forbidden,
            workspace_root=tmp_path / "unused",
        )
        with pytest.raises(ValueError, match=message):
            runtime.run_batch([_task("valid"), task])
        assert not (tmp_path / "unused").exists()

    @pytest.mark.parametrize("mode", ["unknown", "", None])
    def test_rejects_unknown_modes(self, mode: object) -> None:
        with pytest.raises(ValueError, match="environment_mode"):
            ExternalAgentRuntime(
                FakeExternalSession, agent="fake", model="m", environment_mode=mode
            )

    def test_requires_environment_factory(self) -> None:
        with pytest.raises(ValueError, match="environment_factory"):
            ExternalAgentRuntime(
                FakeExternalSession, agent="fake", model="m", environment_mode="final_response"
            )

    @pytest.mark.parametrize("transitions", [None, [], [_real_transition(), _real_transition()]])
    def test_projection_requires_exactly_one_evaluation(self, transitions: object) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            project_outcome(
                _task(),
                ExternalRunOutcome(final_text="answer"),
                agent="fake",
                model="m",
                sampling=SAMPLING,
                environment_mode="final_response",
                environment_transitions=transitions,
            )

    @pytest.mark.parametrize("failure", ["init", "project_response", "step", "session"])
    def test_failures_propagate_and_close_every_resource(
        self, tmp_path: Path, failure: str
    ) -> None:
        class Environment(_RecordingEnvironment):
            def init(self, context: object) -> EnvironmentInitResult:
                if failure == "init":
                    raise RuntimeError("primary init")
                return super().init(context)

            def project_response(self, response: object) -> object:
                if failure == "project_response":
                    raise RuntimeError("primary project_response")
                return super().project_response(response)

            def step(self, action: object) -> EnvironmentTransition:
                if failure == "step":
                    raise RuntimeError("primary step")
                return super().step(action)

            def close(self) -> None:
                super().close()
                raise RuntimeError("secondary Environment cleanup")

        class Session(FakeExternalSession):
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                if failure == "session":
                    raise RuntimeError("primary session")
                return super().run(task, workspace=workspace)

            def close(self) -> None:
                super().close()
                raise RuntimeError("secondary session cleanup")

        environment = Environment()
        session = Session()
        runtime = ExternalAgentRuntime(
            lambda: session,
            agent="fake",
            model="m",
            environment_mode="final_response",
            environment_factory=lambda task: environment,
            workspace_root=tmp_path,
        )
        with pytest.raises(RuntimeError, match=f"primary {failure}"):
            runtime.run(_task())
        assert environment.closed and session.closed
        assert list(tmp_path.iterdir()) == []

    def test_memory_metadata_is_delivered_when_environment_has_no_observation(self) -> None:
        tasks: list[AgentTask] = []

        class Session(FakeExternalSession):
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                tasks.append(task)
                return super().run(task, workspace=workspace)

        runtime = ExternalAgentRuntime(
            Session,
            agent="fake",
            model="m",
            environment_mode="final_response",
            environment_factory=lambda task: _RecordingEnvironment(),
        )
        runtime.run(_task(model="m", metadata={"agent_memory_context": "parent instructions"}))
        assert tasks[0].prompt.count("parent instructions") == 1

    @pytest.mark.parametrize("observation", [123, "answer_key: PRIVATE-CANARY"])
    def test_invalid_initial_observation_never_reaches_session(
        self, tmp_path: Path, observation: object
    ) -> None:
        class Environment(_RecordingEnvironment):
            def init(self, context: object) -> EnvironmentInitResult:
                return EnvironmentInitResult(observation=observation)

        class Session(FakeExternalSession):
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                pytest.fail("invalid observation must not reach session execution")

        environment = Environment()
        session = Session()
        runtime = ExternalAgentRuntime(
            lambda: session,
            agent="fake",
            model="m",
            environment_mode="final_response",
            environment_factory=lambda task: environment,
            workspace_root=tmp_path,
        )
        with pytest.raises((TypeError, SensitiveEnvironmentInputError)):
            runtime.run(_task())
        assert environment.closed and session.closed
        assert list(tmp_path.iterdir()) == []


def test_final_response_preserves_materialized_environment_evidence() -> None:
    transition = EnvironmentTransition(
        observation="evaluated",
        reward=0.5,
        done=True,
        success=True,
        termination_reason="model_output",
        executed_action="materialized program",
        previous_observation="parent program",
        raw_observation={"geometry": [1, 2]},
    )
    result = project_outcome(
        _task(),
        ExternalRunOutcome(final_text="raw diff"),
        agent="fake",
        model="model-x",
        sampling=SAMPLING,
        environment_mode="final_response",
        environment_transitions=[transition],
    )
    actual = result.turns[-1].environment_transition
    assert actual.executed_action == transition.executed_action
    assert actual.previous_observation == transition.previous_observation
    assert actual.raw_observation == transition.raw_observation


_AUX_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ideas",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"ideas": {"type": "array", "items": {"type": "string"}}},
            "required": ["ideas"],
            "additionalProperties": False,
        },
    },
}


class TestExternalAuxiliary:
    def test_same_session_model_native_context_and_journal_evidence(self, tmp_path: Path) -> None:
        calls: list[tuple[AgentTask, Path]] = []
        closed: list[bool] = []

        class Session:
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                calls.append((task, workspace))
                assert workspace.exists()
                assert not (workspace / ENVIRONMENT_SOCKET_NAME).exists()
                return ExternalRunOutcome(
                    final_text='{"ideas": ["geometry"]}',
                    events=(ExternalEvent(kind="reasoning", content="plan", raw={"native": 1}),),
                    usage={"input_tokens": 10, "output_tokens": 5},
                    provider_metadata={"session_id": "native-id", "model": "reported-model"},
                )

            def close(self) -> None:
                closed.append(True)

        def forbidden(*args: object) -> None:
            raise AssertionError("auxiliary must not initialize candidate Environment or hooks")

        runtime = ExternalAgentRuntime(
            session_factory=Session,
            agent="codex",
            model="configured-model",
            environment_factory=forbidden,
            environment_mode="final_response",
            workspace_root=tmp_path,
            reuse_workspace=True,
            workspace_prepare=forbidden,
            workspace_observe=forbidden,
        )
        for _ in range(2):
            response = runtime.generate_auxiliary(
                request_id="same-request",
                system="planning instructions",
                prompt="find ideas",
                response_format=_AUX_SCHEMA,
            )
            assert response.request_id == "same-request"
            assert response.content == '{"ideas": ["geometry"]}'
            assert response.usage == {"input_tokens": 10, "output_tokens": 5}
            assert not response.is_trainable
            assert response.finish_reason == "final"
            metadata = response.backend_metadata
            assert metadata["policy_source"] == "external_agent"
            assert metadata["trainable"] is False
            assert metadata["runtime_seconds"] >= 0
            external = metadata["external"]
            assert external["agent"] == "codex"
            assert external["model"] == "reported-model"
            assert external["configured_model"] == "configured-model"
            assert external["session_id"] == "native-id"
            assert external["events"][0]["raw"] == {"native": 1}
            assert external["instruction_delivery"] == "user_context"
            assert external["response_format"] == {
                "requested": _AUX_SCHEMA,
                "delivery": "prompt",
                "enforcement": "local_json_schema",
                "provider_native": False,
            }
        assert len(closed) == 2
        assert calls[0][1] != calls[1][1]
        for task, workspace in calls:
            assert not workspace.exists()
            assert task.system == ""
            assert task.model == "configured-model"
            assert task.prompt.count("planning instructions") == 1
            assert "find ideas" in task.prompt
            assert '"additionalProperties": false' in task.prompt
            assert task.task_payload == {}
            assert not task.tools
        runtime.close()

    @pytest.mark.parametrize(
        "response_format",
        [
            [],
            {"type": "xml"},
            {"type": "json_object", "ignored": True},
            {"type": "json_schema"},
            {
                "type": "json_schema",
                "json_schema": {"name": "x", "strict": True, "schema": {"type": "bogus"}},
            },
            {"type": "json_schema", "json_schema": {"name": "x", "strict": "yes", "schema": {}}},
            {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {"$ref": "https://example.org/schema"}},
            },
        ],
    )
    def test_invalid_format_rejected_before_session_or_workspace(
        self, tmp_path: Path, response_format: object
    ) -> None:
        def forbidden() -> None:
            raise AssertionError("invalid response format reached session")

        runtime = ExternalAgentRuntime(
            session_factory=forbidden, agent="fake", model="x", workspace_root=tmp_path / "absent"
        )
        with pytest.raises((ValueError, TypeError), match="response_format"):
            runtime.generate_auxiliary(
                request_id="aux", system="plan", prompt="ideas", response_format=response_format
            )
        assert not (tmp_path / "absent").exists()

    @pytest.mark.parametrize(
        ("text", "reason", "response_format", "error"),
        [
            ("answer", "timeout", None, "timeout"),
            ("answer", "truncated", None, "truncated"),
            ("answer", "external_error", None, "external_error"),
            ("answer", "cancelled", None, "cancelled"),
            ("  ", "final", None, "empty"),
            ("```json\n{}\n```", "final", {"type": "json_object"}, "JSON"),
            ('{"x": NaN}', "final", {"type": "json_object"}, "JSON"),
            ('{"x": 1, "x": 2}', "final", {"type": "json_object"}, "JSON"),
            ("[]", "final", {"type": "json_object"}, "object"),
            ('{"ideas": [1]}', "final", _AUX_SCHEMA, "schema"),
            ('{"ideas": [], "extra": 1}', "final", _AUX_SCHEMA, "schema"),
            ("{}", "final", _AUX_SCHEMA, "schema"),
        ],
    )
    def test_invalid_outcome_fails_and_cleans_up(
        self, tmp_path: Path, text: str, reason: str, response_format: object, error: str
    ) -> None:
        closed = []

        class Session:
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                return ExternalRunOutcome(final_text=text, termination_reason=reason)

            def close(self) -> None:
                closed.append(True)

        runtime = ExternalAgentRuntime(
            session_factory=Session, agent="fake", model="x", workspace_root=tmp_path
        )
        with pytest.raises(ExternalAuxiliaryError, match=error) as failure:
            runtime.generate_auxiliary(
                request_id="aux", system="plan", prompt="ideas", response_format=response_format
            )
        assert failure.value.evidence["response"] == text
        assert failure.value.evidence["usage"] == {}
        assert (
            failure.value.evidence["backend_metadata"]["external"]["termination_reason"] == reason
        )
        assert failure.value.__cause__ is not None
        assert closed == [True]
        assert not list(tmp_path.iterdir())

    def test_session_error_survives_close_error(self, tmp_path: Path) -> None:
        class Session:
            def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
                raise RuntimeError("session failed")

            def close(self) -> None:
                raise RuntimeError("close failed")

        runtime = ExternalAgentRuntime(
            session_factory=Session, agent="fake", model="x", workspace_root=tmp_path
        )
        with pytest.raises(RuntimeError, match="session failed"):
            runtime.generate_auxiliary(request_id="aux", system="plan", prompt="ideas")
        assert not list(tmp_path.iterdir())

    def test_closed_runtime_rejects_auxiliary(self) -> None:
        runtime = ExternalAgentRuntime(session_factory=FakeExternalSession, agent="fake", model="x")
        runtime.close()
        with pytest.raises(RuntimeError, match="closed"):
            runtime.generate_auxiliary(request_id="aux", system="plan", prompt="ideas")


@pytest.mark.parametrize("keep_workspaces", [False, True])
def test_auxiliary_workspace_retention_and_close_failure(
    tmp_path: Path,
    keep_workspaces: bool,
) -> None:
    workspaces = []

    class Session:
        def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
            workspaces.append(workspace)
            return ExternalRunOutcome(final_text="plan")

        def close(self) -> None:
            raise RuntimeError("close failed")

    runtime = ExternalAgentRuntime(
        session_factory=Session,
        agent="fake",
        model="x",
        workspace_root=tmp_path,
        keep_workspaces=keep_workspaces,
    )
    with pytest.raises(RuntimeError, match="close failed"):
        runtime.generate_auxiliary(request_id="aux", system="plan", prompt="ideas")
    assert workspaces[0].exists() is keep_workspaces
    runtime.close()
    assert workspaces[0].exists() is keep_workspaces


@pytest.mark.parametrize("field", ["request_id", "system", "prompt"])
@pytest.mark.parametrize("value", ["", " ", None, 2])
def test_auxiliary_rejects_invalid_input_before_session(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    def forbidden() -> None:
        raise AssertionError("invalid input reached session")

    runtime = ExternalAgentRuntime(
        session_factory=forbidden,
        agent="fake",
        model="x",
        workspace_root=tmp_path / "absent",
    )
    arguments = {"request_id": "aux", "system": "plan", "prompt": "ideas", field: value}
    with pytest.raises(ValueError, match=field):
        runtime.generate_auxiliary(**arguments)
    assert not (tmp_path / "absent").exists()


def test_auxiliary_close_interruption_is_not_a_retryable_failure(tmp_path: Path) -> None:
    class Session:
        def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
            return ExternalRunOutcome(final_text="plan")

        def close(self) -> None:
            raise KeyboardInterrupt("stop planning")

    runtime = ExternalAgentRuntime(
        session_factory=Session,
        agent="fake",
        model="x",
        workspace_root=tmp_path,
    )
    with pytest.raises(KeyboardInterrupt, match="stop planning"):
        runtime.generate_auxiliary(request_id="aux", system="plan", prompt="ideas")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("reason", ["final", "external_error", "timeout", "truncated"])
@pytest.mark.parametrize("earlier_text", ["", "I will inspect the candidate."])
def test_tool_ended_stream_does_not_invent_a_final_environment_action(reason, earlier_text):
    class Session(_EnvironmentCallingSession):
        def run(self, task, *, workspace):
            outcome = super().run(task, workspace=workspace)
            prefix = (ExternalEvent(kind="message", content=earlier_text),) if earlier_text else ()
            return ExternalRunOutcome(
                final_text=earlier_text,
                events=prefix + outcome.events[:-1],
                termination_reason=reason,
            )

    environment = _RecordingEnvironment()
    runtime = ExternalAgentRuntime(
        Session,
        agent="fake",
        model="fixture",
        environment_factory=lambda _: environment,
    )
    try:
        result = runtime.run(_task(tools=("bash",)))
    finally:
        runtime.close()
    assert len(environment.actions) == 1
    assert len(result.turns) == 1
    assert result.turns[0].environment_transition.success is None
    assert result.turns[0].environment_transition.done is False
    assert result.termination_reason == reason
    assert environment.closed
