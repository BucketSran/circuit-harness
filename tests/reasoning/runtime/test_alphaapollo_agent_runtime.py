# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Layered fake tests for the batched Reasoning Runtime."""

from __future__ import annotations

import asyncio
import importlib
import math
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

import alphaapollo.reasoning as reasoning
from alphaapollo.common.trajectory import EpisodeTurn
from alphaapollo.reasoning.runtime import (
    AgentResult,
    AgentTask,
    AgentTurn,
    AlphaApolloAgentRuntime,
)
from alphaapollo.reasoning.runtime.environment_lifecycle import (
    EnvironmentInitialization,
    initialize_environment,
    normalize_environment_init,
)
from alphaapollo.reasoning.verification import AgentVerifier

runtime_module = importlib.import_module("alphaapollo.reasoning.runtime.alphaapollo_agent_runtime")


def test_reasoning_public_surface_is_canonical_and_excludes_legacy_exports() -> None:
    assert reasoning.AgentVerifier is AgentVerifier
    for name in (
        "Generation" + "Solver",
        "Problem" + "Session",
        "SingleBranch" + "Loop",
        "Token" + "Usage",
        "Trajectory" + "Slot",
    ):
        assert not hasattr(reasoning, name)


@dataclass(frozen=True)
class _Call:
    id: str
    name: str = "python"
    arguments: str = '{"code":"print(42)"}'


@dataclass(frozen=True)
class _Response:
    request_id: str
    content: str
    group_id: str = ""
    sample_id: int = 0
    finish_reason: str | None = "stop"
    tool_calls: tuple[_Call, ...] = ()
    usage: Mapping[str, Any] = field(
        default_factory=lambda: {"prompt_tokens": 2, "completion_tokens": 3}
    )


@dataclass(frozen=True)
class _Init:
    observation: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_observation: str | None = None


@dataclass(frozen=True)
class _Transition:
    observation: str
    reward: float
    done: bool
    termination_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool | None = None
    response_format_valid: bool = True
    env_action_valid: bool = True
    previous_observation: Any | None = None
    raw_observation: Any | None = None
    executed_action: Any | None = None


def _echo(request: Any, *, content: str = "done", **changes: Any) -> _Response:
    values = {
        "request_id": request.request_id,
        "content": content,
        "group_id": request.group_id,
        "sample_id": request.sample_id,
    }
    values.update(changes)
    return _Response(**values)


class _Backend:
    def __init__(self, scripts: Mapping[str, Sequence[_Response | BaseException]]) -> None:
        self.scripts = {key: list(value) for key, value in scripts.items()}
        self.batches: list[list[Any]] = []
        self.response_batches: list[list[_Response]] = []

    def generate_batch(self, requests: Sequence[Any]) -> list[_Response]:
        self.batches.append(list(requests))
        responses: list[_Response] = []
        for request in requests:
            task_id, turn_value = request.request_id.rsplit(":", 1)
            turn = int(turn_value)
            item = self.scripts[task_id][turn]
            if isinstance(item, BaseException):
                raise item
            responses.append(
                _Response(
                    request_id=request.request_id,
                    content=item.content,
                    group_id=request.group_id,
                    sample_id=request.sample_id,
                    finish_reason=item.finish_reason,
                    tool_calls=item.tool_calls,
                    usage=item.usage,
                )
            )
        self.response_batches.append(responses)
        return responses


class _Environment:
    """Fake Environment that owns tool/action and feedback projection."""

    def __init__(
        self,
        task_id: str,
        transitions: Sequence[_Transition | BaseException],
        events: list[tuple[str, str, Any]],
        *,
        init_failure: BaseException | None = None,
        terminate_failure: BaseException | None = None,
    ) -> None:
        self.task_id = task_id
        self.transitions = list(transitions)
        self.events = events
        self.actions: list[Any] = []
        self.init_failure = init_failure
        self.terminate_failure = terminate_failure

    def init(self, context: Any) -> _Init:
        assert context.task_id == self.task_id
        self.events.append((self.task_id, "init", None))
        if self.init_failure is not None:
            raise self.init_failure
        return _Init(
            observation=context.user_prompt,
            metadata={"task_id": self.task_id},
            raw_observation=context.user_prompt,
        )

    def project_response(self, response: _Response) -> Any:
        if not response.tool_calls:
            return response.content
        return {
            "role": "assistant",
            "content": response.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in response.tool_calls
            ],
        }

    def continuation_messages(
        self, response: _Response, transition: _Transition
    ) -> Sequence[Mapping[str, Any]]:
        if transition.done:
            return ()
        tool_request = transition.metadata["tool_request"]
        call_id = tool_request["call_id"]
        calls = [call for call in response.tool_calls if call.id == call_id]
        assert len(calls) == 1
        call = calls[0]
        return (
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": transition.observation,
            },
        )

    def step(self, action: Any) -> _Transition:
        self.actions.append(action)
        self.events.append((self.task_id, "step", action))
        result = self.transitions[len(self.actions) - 1]
        if isinstance(result, BaseException):
            raise result
        return result

    def terminate(self, reason: str) -> None:
        self.events.append((self.task_id, "terminate", reason))
        if self.terminate_failure is not None:
            raise self.terminate_failure

    def close(self) -> None:
        self.events.append((self.task_id, "close", None))


def _task(task_id: str, **metadata: Any) -> AgentTask:
    model = metadata.pop("model", None)
    system = metadata.pop("system", "solve")
    prompt = metadata.pop("prompt", f"problem {task_id}")
    routing_key = metadata.pop("routing_key", "")
    tools = tuple(metadata.pop("tools", ()))
    branch_id = metadata.pop("branch_id", "main")
    round_index = metadata.pop("round_index", 0)
    sample_id = metadata.pop("sample_id", 0)
    sampling_seed = metadata.pop("sampling_seed", None)
    environment_slot = metadata.pop("environment_slot", 0)
    environment_seed = metadata.pop("environment_seed", None)
    task_payload = metadata.pop("task_payload", {})
    return AgentTask(
        task_id=task_id,
        system=system,
        prompt=prompt,
        model=model,
        routing_key=routing_key,
        tools=tools,
        metadata=metadata,
        branch_id=branch_id,
        round_index=round_index,
        sample_id=sample_id,
        sampling_seed=sampling_seed,
        environment_slot=environment_slot,
        environment_seed=environment_seed,
        task_payload=task_payload,
    )


def _final(text: str = "done") -> _Transition:
    return _Transition(text, 1.0, True, "final", success=True)


def _tool(call_id: str) -> _Transition:
    return _Transition(
        observation="42",
        reward=0.0,
        done=False,
        termination_reason=None,
        metadata={
            "tool_request": {"call_id": call_id, "tool_id": "python"},
            "tool_response": {"call_id": call_id, "stdout": "42"},
        },
    )


def _runtime(
    backend: Any,
    environments: Mapping[str, _Environment],
    *,
    max_turns: int = 3,
) -> AlphaApolloAgentRuntime:
    return AlphaApolloAgentRuntime(
        backend,
        model="fake",
        environment_factory=lambda task: environments[task.task_id],
        max_turns=max_turns,
    )


def test_agent_result_rejects_non_contiguous_turn_indices() -> None:
    transition = _final()
    turn = AgentTurn(
        index=1,
        generation_request=object(),
        generation_response=object(),
        environment_transition=transition,
    )

    with pytest.raises(ValueError, match="contiguous"):
        AgentResult(task_id="task", final_text="done", turns=(turn,))


def test_agent_turn_reuses_the_common_episode_contract() -> None:
    transition = _final()
    turn = AgentTurn(
        index=0,
        generation_request=object(),
        generation_response=object(),
        environment_transition=transition,
    )

    assert isinstance(turn, EpisodeTurn)
    assert turn.env_reward == transition.reward
    assert not hasattr(turn, "__dict__")


def test_empty_system_uses_the_model_default_without_an_empty_message() -> None:
    events: list[tuple[str, str, Any]] = []
    backend = _Backend({"default": [_Response("unused", "answer")]})
    environment = _Environment("default", [_final()], events)

    _runtime(backend, {"default": environment}).run(_task("default", system=""))

    assert backend.batches[0][0].messages == ({"role": "user", "content": "problem default"},)


@pytest.mark.parametrize(
    ("generation_request", "generation_response"),
    ((None, object()), (object(), None)),
)
def test_agent_turn_reuses_common_non_null_generation_validation(
    generation_request: object | None,
    generation_response: object | None,
) -> None:
    with pytest.raises(TypeError, match="cannot be None"):
        AgentTurn(0, generation_request, generation_response, _final())


def test_agent_result_rejects_turns_after_a_terminal_transition() -> None:
    turns = (
        AgentTurn(0, object(), object(), _final()),
        AgentTurn(1, object(), object(), _tool("late-call")),
    )

    with pytest.raises(ValueError, match="final turn"):
        AgentResult(task_id="task", final_text="done", turns=turns)


@pytest.mark.parametrize(
    ("finish_reason", "truncated"),
    [("length", True), ("stop", False), ("tool_calls", False), (None, False)],
)
def test_a_result_records_whether_its_answer_bearing_reply_was_cut_off(
    finish_reason: str | None, truncated: bool
) -> None:
    """Two different "why did it stop", and both reach the record.

    The Environment is handed the model's text as an action and never sees a
    ``GenerationResponse``, so a reply the sampler clipped mid-sentence that
    still looks answer-shaped ends the episode ``final`` exactly like a complete
    one. ``termination_reason`` keeps saying what the Environment decided;
    ``output_truncated`` says what the sampler did.
    """

    result = AgentResult(
        task_id="task",
        final_text="Let the distance be $d$ miles. Then the",
        turns=(
            AgentTurn(
                0,
                object(),
                _Response(
                    "r", "Let the distance be $d$ miles. Then the", finish_reason=finish_reason
                ),
                _final(),
            ),
        ),
    )

    assert result.output_truncated is truncated
    assert result.termination_reason == "final"


def test_truncation_is_read_from_the_answer_bearing_turn_not_from_any_turn() -> None:
    """The last turn is the one whose text became ``final_text``.

    A turn clipped earlier in the loop is real evidence that ``max_tokens`` is
    low, but the model was fed the truncated text back and went on to answer, so
    nothing downstream reads it: extraction, verdict parsing, and scoring see
    ``final_text`` and no other turn. That earlier turn stays visible as its own
    ``finish_reason`` in the trajectory projection.
    """

    call_id = "mid-loop-call"
    recovered = AgentResult(
        task_id="task",
        final_text="answer 42",
        turns=(
            AgentTurn(
                0,
                object(),
                _Response("r0", "", tool_calls=(_Call(call_id),), finish_reason="length"),
                _tool(call_id),
            ),
            AgentTurn(1, object(), _Response("r1", "answer 42"), _final()),
        ),
    )

    assert recovered.output_truncated is False
    assert recovered.turns[0].generation_response.finish_reason == "length"


def test_a_result_with_no_turns_is_not_reported_as_truncated() -> None:
    assert AgentResult(task_id="task", final_text="done").output_truncated is False


def test_a_producer_cannot_declare_a_truncation_its_own_turns_deny() -> None:
    """One authority for the fact: the turns the result already carries.

    The field is derived at construction rather than passed in, so no Runtime can
    record a truncation flag that contradicts its own trajectory, and none has to
    remember to set one.
    """

    with pytest.raises(TypeError, match="output_truncated"):
        AgentResult(task_id="task", final_text="done", output_truncated=True)  # type: ignore[call-arg]


def test_tool_feedback_final_and_mixed_slots_preserve_complete_generation_steps() -> None:
    events: list[tuple[str, str, Any]] = []
    call_id = "provider-call-7"
    backend = _Backend(
        {
            "slow": [
                _Response(
                    "unused",
                    "",
                    tool_calls=(_Call(call_id),),
                    finish_reason="tool_calls",
                ),
                _Response("unused", "answer 42"),
            ],
            "fast": [_Response("unused", "answer now")],
        }
    )
    environments = {
        "slow": _Environment("slow", [_tool(call_id), _final()], events),
        "fast": _Environment("fast", [_final()], events),
    }

    results = _runtime(backend, environments).run_batch([_task("slow"), _task("fast")])

    assert [result.task_id for result in results] == ["slow", "fast"]
    assert [len(batch) for batch in backend.batches] == [2, 1]
    assert [len(result.turns) for result in results] == [2, 1]
    first_turn = results[0].turns[0]
    assert first_turn.generation_request is backend.batches[0][0]
    assert first_turn.generation_response is backend.response_batches[0][0]
    assert first_turn.environment_transition is not environments["slow"].transitions[0]
    assert first_turn.environment_transition.observation == "42"
    assert first_turn.generation_request.request_id == "slow:0"
    assert first_turn.generation_response.tool_calls[0].id == call_id
    assert not hasattr(first_turn, "rollout_request")
    action_call = environments["slow"].actions[0]["tool_calls"][0]
    assert action_call["id"] == call_id
    following_messages = backend.batches[1][0].messages
    assert following_messages[-1] == {
        "role": "tool",
        "tool_call_id": call_id,
        "content": "42",
    }
    assert following_messages[-2]["tool_calls"][0]["id"] == call_id
    assert [turn.generation_response.usage for turn in results[0].turns] == [
        {"prompt_tokens": 2, "completion_tokens": 3},
        {"prompt_tokens": 2, "completion_tokens": 3},
    ]
    assert not hasattr(results[0], "usage")
    assert not hasattr(first_turn, "usage")
    assert events[-2:] == [("slow", "close", None), ("fast", "close", None)]


def test_memory_context_reaches_the_model_but_never_the_environment() -> None:
    """The executor's split contract, enforced at the runtime layer.

    Environment-backed runtimes replace `AgentTask.prompt` with the
    environment-authored observation, so workflow-memory context must travel
    through `agent_memory_context` metadata into the first model message while
    the environment keeps seeing only the unaugmented task instruction.
    """

    events: list[tuple[str, str, Any]] = []
    captured_contexts: list[Any] = []

    class CapturingEnvironment(_Environment):
        def init(self, context: Any) -> _Init:
            captured_contexts.append(context)
            return super().init(context)

    memory_block = (
        "# Workflow memory (untrusted prior evidence)\n"
        '[{"success":false,"termination_reason":"time_limit"}]'
    )
    environment = CapturingEnvironment("m1", [_final()], events)
    backend = _Backend({"m1": [_Response("unused", "done")]})
    task = _task(
        "m1",
        prompt=f"problem m1\n\n{memory_block}",
        environment_user_prompt="problem m1",
        agent_memory_context=memory_block,
    )

    results = _runtime(backend, {"m1": environment}).run_batch([task])

    assert results[0].termination_reason == "final"
    # The environment saw the unaugmented instruction and no memory metadata.
    context = captured_contexts[0]
    assert context.user_prompt == "problem m1"
    assert "agent_memory_context" not in context.metadata
    # The first model message carries the environment-authored observation
    # (this fake echoes user_prompt) with the memory context attached after it.
    first_messages = backend.batches[0][0].messages
    assert first_messages[-1]["role"] == "user"
    assert first_messages[-1]["content"] == f"problem m1\n\n{memory_block}"


def test_memory_context_appends_a_text_block_to_multimodal_observations() -> None:
    from alphaapollo.reasoning.runtime.trajectory_slot import _TrajectorySlot

    memory_block = "# Workflow memory (untrusted prior evidence)\n[]"
    observation = [
        {"type": "text", "text": "pick up the black bowl"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]
    slot = _TrajectorySlot(
        task=_task("m2", agent_memory_context=memory_block),
        position=0,
    )
    slot.initialize(_Init(observation=observation, metadata={}))

    user_content = slot.messages[-1]["content"]
    assert user_content[:2] == observation
    assert user_content[-1] == {"type": "text", "text": memory_block}


def test_max_turns_terminates_only_still_active_slots() -> None:
    events: list[tuple[str, str, Any]] = []
    call_id = "loop"
    backend = _Backend(
        {
            "a": [
                _Response(
                    "unused",
                    "",
                    tool_calls=(_Call(call_id),),
                    finish_reason="tool_calls",
                )
            ]
            * 2
        }
    )
    environments = {"a": _Environment("a", [_tool(call_id), _tool(call_id)], events)}

    result = _runtime(backend, environments, max_turns=2).run(_task("a"))

    assert result.termination_reason == "max_turns"
    assert len(result.turns) == 2
    assert ("a", "terminate", "max_turns") in events
    assert events[-1] == ("a", "close", None)


def test_max_turn_termination_is_one_shot_when_environment_terminate_fails() -> None:
    events: list[tuple[str, str, Any]] = []
    call_id = "loop"
    backend = _Backend(
        {
            "a": [
                _Response(
                    "unused",
                    "",
                    tool_calls=(_Call(call_id),),
                    finish_reason="tool_calls",
                )
            ]
        }
    )
    environment = _Environment(
        "a",
        [_tool(call_id)],
        events,
        terminate_failure=RuntimeError("terminate failed"),
    )

    with pytest.raises(RuntimeError, match="terminate failed"):
        _runtime(backend, {"a": environment}, max_turns=1).run(_task("a"))

    assert [(kind, reason) for _, kind, reason in events if kind == "terminate"] == [
        ("terminate", "max_turns")
    ]
    assert events[-1] == ("a", "close", None)


def test_terminal_transition_does_not_request_continuation_messages() -> None:
    events: list[tuple[str, str, Any]] = []

    class TerminalEnvironment(_Environment):
        def continuation_messages(
            self, response: _Response, transition: _Transition
        ) -> Sequence[Mapping[str, Any]]:
            raise AssertionError((response, transition))

    environment = TerminalEnvironment("a", [_final("answer")], events)

    result = _runtime(_Backend({"a": [_Response("unused", "answer")]}), {"a": environment}).run(
        _task("a")
    )

    assert result.final_text == "answer"
    assert result.termination_reason == "final"


def test_last_allowed_turn_does_not_request_unused_continuation_messages() -> None:
    events: list[tuple[str, str, Any]] = []
    call_id = "last-turn"

    class LastTurnEnvironment(_Environment):
        def continuation_messages(
            self, response: _Response, transition: _Transition
        ) -> Sequence[Mapping[str, Any]]:
            raise AssertionError((response, transition))

    environment = LastTurnEnvironment("a", [_tool(call_id)], events)
    backend = _Backend(
        {
            "a": [
                _Response(
                    "unused",
                    "",
                    tool_calls=(_Call(call_id),),
                    finish_reason="tool_calls",
                )
            ]
        }
    )

    result = _runtime(backend, {"a": environment}, max_turns=1).run(_task("a"))

    assert result.termination_reason == "max_turns"
    assert len(result.turns) == 1
    assert ("a", "terminate", "max_turns") in events


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"done": "false"}, "done must be a bool"),
        ({"reward": "not-a-number"}, "reward must be a finite number"),
    ],
)
def test_invalid_transition_record_values_fail_loudly(
    changes: Mapping[str, Any], message: str
) -> None:
    events: list[tuple[str, str, Any]] = []
    values: dict[str, Any] = {
        "observation": "answer",
        "reward": 1.0,
        "done": True,
        "termination_reason": "final",
        "metadata": {},
        "success": True,
        "response_format_valid": True,
        "env_action_valid": True,
    }
    values.update(changes)
    invalid = _Transition(**values)  # type: ignore[arg-type]
    environment = _Environment("a", [invalid], events)

    with pytest.raises(TypeError, match=message):
        _runtime(_Backend({"a": [_Response("unused", "answer")]}), {"a": environment}).run(
            _task("a")
        )

    assert [(kind, reason) for _, kind, reason in events if kind == "terminate"] == [
        ("terminate", "runtime_error")
    ]
    assert events[-1] == ("a", "close", None)


def test_normalized_transition_metadata_is_isolated_and_immutable() -> None:
    source = {"nested": {"value": 1}}

    transition = runtime_module._normalize_transition(
        _Transition(
            observation="answer",
            reward=1.0,
            done=True,
            termination_reason="final",
            metadata=source,
            success=True,
        )
    )
    source["nested"]["value"] = 2

    assert transition.metadata == {"nested": {"value": 1}}
    with pytest.raises(TypeError, match="do not support mutation"):
        transition.metadata["nested"]["value"] = 3


def test_common_model_output_termination_normalizes_to_final() -> None:
    events: list[tuple[str, str, Any]] = []
    backend = _Backend({"a": [_Response("unused", "answer")]})
    transition = _Transition("answer", 1.0, True, "model_output")

    result = _runtime(backend, {"a": _Environment("a", [transition], events)}).run(_task("a"))

    assert result.termination_reason == "final"


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("backend failed"), KeyboardInterrupt(), asyncio.CancelledError()],
)
def test_generation_failure_and_cancellation_terminate_and_close_every_slot(
    failure: BaseException,
) -> None:
    events: list[tuple[str, str, Any]] = []
    backend = _Backend({"a": [failure], "b": [_Response("unused", "never")]})
    environments = {task_id: _Environment(task_id, [_final()], events) for task_id in ("a", "b")}

    with pytest.raises(type(failure)):
        _runtime(backend, environments).run_batch([_task("a"), _task("b")])

    expected_reason = (
        "cancelled"
        if isinstance(failure, (KeyboardInterrupt, asyncio.CancelledError))
        else "runtime_error"
    )
    assert Counter(
        (task, kind, value) for task, kind, value in events if kind == "terminate"
    ) == Counter({("a", "terminate", expected_reason): 1, ("b", "terminate", expected_reason): 1})
    assert Counter((task, kind) for task, kind, _ in events if kind == "close") == Counter(
        {("a", "close"): 1, ("b", "close"): 1}
    )


def test_termination_failure_isolated_and_primary_failure_preserved() -> None:
    events: list[tuple[str, str, Any]] = []
    failure = RuntimeError("generation failed")
    backend = _Backend({"a": [failure], "b": [_Response("unused", "never")]})
    environments = {
        "a": _Environment(
            "a", [_final()], events, terminate_failure=RuntimeError("terminate failed")
        ),
        "b": _Environment("b", [_final()], events),
    }

    with pytest.raises(RuntimeError, match="generation failed"):
        _runtime(backend, environments).run_batch([_task("a"), _task("b")])

    assert {task for task, kind, _ in events if kind == "terminate"} == {"a", "b"}
    assert {task for task, kind, _ in events if kind == "close"} == {"a", "b"}


def test_environment_failure_cleans_up_all_slots_and_preserves_exception() -> None:
    events: list[tuple[str, str, Any]] = []
    backend = _Backend({"a": [_Response("unused", "x")], "b": [_Response("unused", "y")]})
    environments = {
        "a": _Environment("a", [ValueError("environment failed")], events),
        "b": _Environment("b", [_final()], events),
    }

    with pytest.raises(ValueError, match="environment failed"):
        _runtime(backend, environments).run_batch([_task("a"), _task("b")])

    assert {task for task, kind, _ in events if kind == "terminate"} == {"a", "b"}
    assert {task for task, kind, _ in events if kind == "close"} == {"a", "b"}


@pytest.mark.parametrize(
    ("backend_factory", "message"),
    [
        (
            lambda: type(
                "MissingIdentityBackend",
                (),
                {
                    "generate_batch": lambda self, requests: [
                        SimpleNamespace(request_id=request.request_id, content="x")
                        for request in requests
                    ]
                },
            )(),
            "missing Generation identity",
        ),
        (
            lambda: type(
                "WrongOrderBackend",
                (),
                {
                    "generate_batch": lambda self, requests: [
                        _echo(request) for request in reversed(requests)
                    ]
                },
            )(),
            "does not echo",
        ),
        (
            lambda: type(
                "DuplicateBackend",
                (),
                {
                    "generate_batch": lambda self, requests: [
                        _echo(requests[0]),
                        _echo(requests[0]),
                    ]
                },
            )(),
            "duplicate response request_id",
        ),
        (
            lambda: type(
                "WrongGroupBackend",
                (),
                {
                    "generate_batch": lambda self, requests: [
                        _echo(request, group_id="wrong") for request in requests
                    ]
                },
            )(),
            "does not echo",
        ),
        (
            lambda: type(
                "WrongSampleBackend",
                (),
                {
                    "generate_batch": lambda self, requests: [
                        _echo(request, sample_id=request.sample_id + 10) for request in requests
                    ]
                },
            )(),
            "does not echo",
        ),
    ],
)
def test_generation_response_identity_is_mandatory_unique_and_ordered(
    backend_factory: Any, message: str
) -> None:
    events: list[tuple[str, str, Any]] = []
    environments = {task_id: _Environment(task_id, [_final()], events) for task_id in ("a", "b")}

    with pytest.raises(RuntimeError, match=message):
        _runtime(backend_factory(), environments).run_batch([_task("a"), _task("b")])

    assert {task for task, kind, _ in events if kind == "terminate"} == {"a", "b"}
    assert {task for task, kind, _ in events if kind == "close"} == {"a", "b"}
    assert not any(kind == "step" for _, kind, _ in events)


def test_generation_response_cardinality_mismatch_is_rejected_before_step() -> None:
    events: list[tuple[str, str, Any]] = []

    class ShortBackend:
        def generate_batch(self, requests: Sequence[Any]) -> list[_Response]:
            return [_echo(requests[0])]

    environments = {task_id: _Environment(task_id, [_final()], events) for task_id in ("a", "b")}
    with pytest.raises(RuntimeError, match="1 responses for 2 requests"):
        _runtime(ShortBackend(), environments).run_batch([_task("a"), _task("b")])

    assert not any(kind == "step" for _, kind, _ in events)


def test_duplicate_generation_sample_identity_is_rejected_before_backend_call() -> None:
    events: list[tuple[str, str, Any]] = []
    backend = _Backend({task_id: [_Response("unused", "never")] for task_id in ("a", "b")})
    environments = {task_id: _Environment(task_id, [_final()], events) for task_id in ("a", "b")}

    with pytest.raises(RuntimeError, match=r"duplicate Generation \(group_id, sample_id\)"):
        _runtime(backend, environments).run_batch(
            [
                _task("a", prompt="same", routing_key="shared", sample_id=0),
                _task("b", prompt="same", routing_key="shared", sample_id=0),
            ]
        )

    assert backend.batches == []
    assert not any(kind == "step" for _, kind, _ in events)
    assert {task for task, kind, _ in events if kind == "terminate"} == {"a", "b"}
    assert {task for task, kind, _ in events if kind == "close"} == {"a", "b"}


def test_distinct_generation_sample_ids_share_one_group() -> None:
    events: list[tuple[str, str, Any]] = []
    backend = _Backend(
        {
            "a": [_Response("unused", "sample zero")],
            "b": [_Response("unused", "sample one")],
        }
    )
    environments = {task_id: _Environment(task_id, [_final()], events) for task_id in ("a", "b")}

    results = _runtime(backend, environments).run_batch(
        [
            _task("a", prompt="same", routing_key="shared", sample_id=0),
            _task("b", prompt="same", routing_key="shared", sample_id=1),
        ]
    )

    requests = backend.batches[0]
    assert requests[0].group_id == requests[1].group_id
    assert [request.sample_id for request in requests] == [0, 1]
    assert [result.final_text for result in results] == ["sample zero", "sample one"]


def test_only_generate_batch_is_the_supported_generation_entrypoint() -> None:
    events: list[tuple[str, str, Any]] = []

    class TargetBackend:
        def __init__(self) -> None:
            self.batch_calls = 0
            self.single_calls = 0

        def generate_batch(self, requests: Sequence[Any]) -> list[_Response]:
            self.batch_calls += 1
            return [_echo(request, content="batch") for request in requests]

        def generate(self, request: Any) -> _Response:
            self.single_calls += 1
            raise AssertionError(request)

    backend = TargetBackend()
    environments = {task_id: _Environment(task_id, [_final()], events) for task_id in ("a", "b")}

    results = _runtime(backend, environments).run_batch([_task("a"), _task("b")])

    assert backend.batch_calls == 1
    assert backend.single_calls == 0
    assert [result.final_text for result in results] == ["batch", "batch"]

    class SingleOnlyBackend:
        def generate(self, request: Any) -> _Response:
            raise AssertionError(request)

    with pytest.raises(TypeError, match=r"must expose generate_batch\(\)"):
        _runtime(SingleOnlyBackend(), environments)


def test_runtime_builds_the_canonical_common_generation_records() -> None:
    """A strict Common backend must never be handed a private duck type."""

    generation = importlib.import_module("alphaapollo.common.generation.base")

    events: list[tuple[str, str, Any]] = []
    captured_response: Any | None = None

    class StrictBackend(generation.GenerationBackend):
        @property
        def capabilities(self) -> Any:
            return generation.BackendCapabilities(
                token_native=True,
                logprobs=True,
                tool_calls=True,
                policy_source=generation.POLICY_SOURCE_REMOTE_ENGINE,
            )

        def _generate_batch(self, requests: Sequence[Any]) -> list[Any]:
            nonlocal captured_response
            assert len(requests) == 1
            request = requests[0]
            assert isinstance(request, generation.GenerationRequest)
            assert isinstance(request.sampling, generation.SamplingOptions)
            captured_response = generation.GenerationResponse(
                request_id=request.request_id,
                group_id=request.group_id,
                sample_id=request.sample_id,
                content="canonical concrete response",
                prompt_token_ids=(11, 12),
                response_token_ids=(21, 22),
                response_logprobs=(-0.1, -0.2),
                provenance=generation.Provenance(
                    policy_model=request.model,
                    tokenizer_id="stacked-test-tokenizer",
                ),
            )
            return [captured_response]

    environment = _Environment("a", [_final()], events)
    runtime = AlphaApolloAgentRuntime(
        StrictBackend(),
        model="stacked-model",
        environment_factory=lambda _task: environment,
        temperature=0.0,
        max_tokens=32,
    )

    result = runtime.run(_task("a"))

    turn = result.turns[0]
    assert isinstance(turn.generation_request, generation.GenerationRequest)
    assert turn.generation_response is captured_response
    assert turn.generation_response.is_trainable is True
    assert turn.generation_response.response_token_ids == (21, 22)


def test_partial_init_failure_terminates_started_slots_and_closes_every_environment() -> None:
    events: list[tuple[str, str, Any]] = []
    backend = _Backend({task_id: [_Response("unused", "x")] for task_id in ("a", "b", "c")})
    environments = {
        "a": _Environment("a", [_final()], events),
        "b": _Environment("b", [_final()], events, init_failure=RuntimeError("init failed")),
        "c": _Environment("c", [_final()], events),
    }

    with pytest.raises(RuntimeError, match="init failed"):
        _runtime(backend, environments).run_batch([_task("a"), _task("b"), _task("c")])

    assert Counter((task, kind) for task, kind, _ in events if kind == "terminate") == Counter(
        {("a", "terminate"): 1, ("c", "terminate"): 1}
    )
    assert Counter((task, kind) for task, kind, _ in events if kind == "close") == Counter(
        {("a", "close"): 1, ("b", "close"): 1, ("c", "close"): 1}
    )


def test_init_failure_waits_for_pending_init_before_teardown() -> None:
    events: list[tuple[str, str, Any]] = []
    slow_started = threading.Event()
    release_slow = threading.Event()
    slow_finished = threading.Event()
    teardown_before_finish = threading.Event()

    class CoordinatedEnvironment(_Environment):
        def __init__(self, *args: Any, fail: bool, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.fail = fail

        def init(self, context: Any) -> _Init:
            if self.fail:
                assert slow_started.wait(timeout=1.0)
                events.append((self.task_id, "init", None))
                raise RuntimeError("init failed")
            events.append((self.task_id, "init_started", None))
            slow_started.set()
            assert release_slow.wait(timeout=2.0)
            slow_finished.set()
            events.append((self.task_id, "init_finished", None))
            return _Init(context.user_prompt, {"task_id": self.task_id})

        def terminate(self, reason: str) -> None:
            if not self.fail and not slow_finished.is_set():
                teardown_before_finish.set()
            super().terminate(reason)

        def close(self) -> None:
            if not self.fail and not slow_finished.is_set():
                teardown_before_finish.set()
            super().close()

    environments = {
        "fail": CoordinatedEnvironment("fail", [_final()], events, fail=True),
        "slow": CoordinatedEnvironment("slow", [_final()], events, fail=False),
    }
    runtime = _runtime(
        _Backend({task_id: [_Response("unused", "x")] for task_id in environments}),
        environments,
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            runtime.run_batch([_task("fail"), _task("slow")])
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert slow_started.wait(timeout=1.0)
    assert not teardown_before_finish.wait(timeout=0.1)
    release_slow.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "init failed"
    assert not teardown_before_finish.is_set()


@pytest.mark.parametrize(
    ("init_result", "message"),
    [
        (
            SimpleNamespace(observation="x", metadata="not-a-mapping"),
            "typed init record",
        ),
        (_Init("x", "not-a-mapping"), "metadata must be a mapping"),
    ],
)
def test_invalid_environment_init_record_fails_before_generation_and_cleans_up(
    init_result: Any, message: str
) -> None:
    events: list[tuple[str, str, Any]] = []

    class InvalidInitEnvironment(_Environment):
        def init(self, context: Any) -> Any:
            events.append((context.task_id, "init", None))
            return init_result

    backend = _Backend({"a": [_Response("unused", "never")]})
    environment = InvalidInitEnvironment("a", [_final()], events)

    with pytest.raises(TypeError, match=message):
        _runtime(backend, {"a": environment}).run(_task("a"))

    assert backend.batches == []
    assert events == [
        ("a", "init", None),
        ("a", "terminate", "runtime_error"),
        ("a", "close", None),
    ]


def test_environment_init_shapes_normalize_to_private_copied_record() -> None:
    source = {"nested": {"value": 1}}

    typed = normalize_environment_init(_Init("typed", source))
    legacy = normalize_environment_init(("legacy", None))
    source["nested"]["value"] = 2

    assert isinstance(typed, EnvironmentInitialization)
    assert typed.observation == "typed"
    assert typed.metadata == {"nested": {"value": 1}}
    with pytest.raises(TypeError, match="do not support mutation"):
        typed.metadata["nested"]["value"] = 3
    assert isinstance(legacy, EnvironmentInitialization)
    assert legacy.observation == "legacy"
    assert legacy.metadata == {}


def test_environment_context_receives_mutable_thawed_task_payload() -> None:
    events: list[tuple[str, str, Any]] = []
    captured: list[Any] = []

    class MutatingEnvironment(_Environment):
        def init(self, context: Any) -> _Init:
            captured.append(context)
            context.task_payload["items"].append("environment")
            context.environment_config["limits"]["turns"] = 2
            context.metadata["labels"].append("environment")
            return super().init(context)

    backend = _Backend({"a": [_Response("unused", "done")]})
    environment = MutatingEnvironment("a", [_final()], events)
    task = _task(
        "a",
        task_payload={
            "items": ["caller"],
            "answer": "PRIVATE-ANSWER-CANARY",
        },
        environment_config={"limits": {"turns": 1}},
        labels=["caller"],
    )

    result = _runtime(backend, {"a": environment}).run(task)

    context = captured[0]
    assert context.task_payload == {
        "items": ["caller", "environment"],
        "answer": "PRIVATE-ANSWER-CANARY",
    }
    assert context.environment_config == {"limits": {"turns": 2}}
    assert context.metadata == {"labels": ["caller", "environment"]}
    assert task.task_payload == {
        "items": ("caller",),
        "answer": "PRIVATE-ANSWER-CANARY",
    }
    assert task.metadata == {
        "environment_config": {"limits": {"turns": 1}},
        "labels": ("caller",),
    }
    assert "task_payload" not in result.metadata
    assert "PRIVATE-ANSWER-CANARY" not in repr(result)


def test_private_payload_rejects_legacy_two_argument_environment_init() -> None:
    class LegacyEnvironment:
        called = False

        def init(self, system: str, prompt: str) -> tuple[str, dict[str, Any]]:
            self.called = True
            return f"{system}: {prompt}", {}

    environment = LegacyEnvironment()
    task = _task("legacy", task_payload={"answer": "PRIVATE-LEGACY-INIT"})

    with pytest.raises(TypeError, match=r"legacy init\(system, prompt\).+cannot receive"):
        initialize_environment(environment, task, seed=None)

    assert environment.called is False


def test_environment_init_uses_unaugmented_prompt_when_memory_context_is_present() -> None:
    captured: list[Any] = []

    class ContextEnvironment:
        def init(self, context: Any) -> _Init:
            captured.append(context)
            return _Init("ready")

    task = _task(
        "memory-prompt",
        prompt="public instruction\n\nWorkflow memory: untrusted context",
        environment_user_prompt="public instruction",
    )

    initialize_environment(ContextEnvironment(), task, seed=None)

    assert captured[0].user_prompt == "public instruction"
    assert captured[0].metadata == {}


def test_native_environment_batch_receives_complete_mixed_active_masks() -> None:
    call_id = "batched-call"
    backend = _Backend(
        {
            "a": [
                _Response(
                    "unused",
                    "",
                    tool_calls=(_Call(call_id),),
                    finish_reason="tool_calls",
                ),
                _Response("unused", "a-final"),
            ],
            "b": [_Response("unused", "b-final")],
        }
    )

    class Batch(_Environment):
        def __init__(self) -> None:
            super().__init__("batch", [], [])
            self.masks: list[list[bool]] = []
            self.payloads: list[dict[str, Any]] = []
            self.round = 0
            self.closed = False

        def init(self, tasks: Sequence[AgentTask]) -> list[_Init]:
            self.payloads = [dict(task.task_payload) for task in tasks]
            return [_Init(task.prompt) for task in tasks]

        def projector(self, slot_index: int) -> Batch:
            assert slot_index in (0, 1)
            return self

        def step(
            self, actions: Sequence[Any | None], active_mask: Sequence[bool]
        ) -> list[_Transition | None]:
            self.masks.append(list(active_mask))
            if self.round == 0:
                assert actions[0]["tool_calls"][0]["id"] == call_id
                assert actions[1] == "b-final"
                result: list[_Transition | None] = [_tool(call_id), _final()]
            else:
                assert actions[0] == "a-final" and actions[1] is None
                result = [_final(), None]
            self.round += 1
            return result

        def terminate(self, active_mask: Sequence[bool], reason: str) -> None:
            raise AssertionError((active_mask, reason))

        def close(self) -> None:
            self.closed = True

    batch = Batch()
    runtime = AlphaApolloAgentRuntime(
        backend,
        model="fake",
        environment_batch_factory=lambda _tasks: batch,
    )

    results = runtime.run_batch(
        [
            _task("a", task_payload={"answer": "PRIVATE-BATCH-A"}),
            _task("b", task_payload={"answer": "PRIVATE-BATCH-B"}),
        ]
    )

    assert batch.masks == [[True, True], [True, False]]
    assert batch.payloads == [
        {"answer": "PRIVATE-BATCH-A"},
        {"answer": "PRIVATE-BATCH-B"},
    ]
    assert batch.closed is True
    assert [result.final_text for result in results] == ["a-final", "b-final"]
    assert "PRIVATE-BATCH" not in repr(results)


def test_missing_environment_projector_fails_loudly_and_cleans_up() -> None:
    events: list[tuple[str, str, Any]] = []

    class UnprojectedEnvironment:
        def init(self, context: Any) -> _Init:
            events.append((context.task_id, "init", None))
            return _Init(context.user_prompt)

        def step(self, action: Any) -> _Transition:
            raise AssertionError(action)

        def terminate(self, reason: str) -> None:
            events.append(("a", "terminate", reason))

        def close(self) -> None:
            events.append(("a", "close", None))

    runtime = AlphaApolloAgentRuntime(
        _Backend({"a": [_Response("unused", "x")]}),
        model="fake",
        environment_factory=lambda _task: UnprojectedEnvironment(),
    )

    with pytest.raises(TypeError, match="Environment projector must expose"):
        runtime.run(_task("a"))

    assert events[-2:] == [("a", "terminate", "runtime_error"), ("a", "close", None)]


def test_invalid_environment_batch_factory_result_is_closed_before_rejection() -> None:
    class InvalidBatch:
        def __init__(self) -> None:
            self.close_calls = 0

        def init(self, tasks: Sequence[AgentTask]) -> list[_Init]:
            return [_Init(task.prompt) for task in tasks]

        def step(
            self, actions: Sequence[Any | None], active_mask: Sequence[bool]
        ) -> list[_Transition | None]:
            raise AssertionError((actions, active_mask))

        def close(self) -> None:
            self.close_calls += 1

    batch = InvalidBatch()
    backend = _Backend({"a": [_Response("unused", "never")]})
    runtime = AlphaApolloAgentRuntime(
        backend,
        model="fake",
        environment_batch_factory=lambda _tasks: batch,
    )

    with pytest.raises(TypeError, match="init/step/terminate/close"):
        runtime.run(_task("a"))

    assert batch.close_calls == 1
    assert backend.batches == []


def test_environment_factory_must_return_distinct_instances_per_task() -> None:
    events: list[tuple[str, str, Any]] = []
    shared = _Environment("shared", [_final()], events)
    backend = _Backend({task_id: [_Response("unused", "never")] for task_id in ("a", "b")})
    runtime = AlphaApolloAgentRuntime(
        backend,
        model="fake",
        environment_factory=lambda _task: shared,
    )

    with pytest.raises(TypeError, match="distinct Environment"):
        runtime.run_batch([_task("a"), _task("b")])

    assert events == [("shared", "close", None)]
    assert backend.batches == []


def test_generation_request_surface_matches_pr_189_contract() -> None:
    events: list[tuple[str, str, Any]] = []

    class TargetGenerationBackend:
        def generate_batch(self, requests: Sequence[Any]) -> list[_Response]:
            assert len(requests) == 1
            request = requests[0]
            assert request.model == "role-model"
            assert request.messages[-1] == {"role": "user", "content": "problem a"}
            assert request.sampling.temperature == 0.25
            assert request.sampling.max_tokens == 128
            assert request.sampling.top_p == 0.9
            assert not hasattr(request.sampling, "n")
            assert isinstance(request.sample_id, int) and request.sample_id == 0
            assert request.tools == ({"type": "function", "function": {"name": "python"}},)
            assert request.tool_choice == "auto"
            assert request.provider_options == {"seed": 17}
            assert request.routing_key == "local"
            assert not hasattr(request, "context")
            return [_echo(request, content="compatible")]

    environment = _Environment("a", [_final()], events)
    runtime = AlphaApolloAgentRuntime(
        TargetGenerationBackend(),
        model="fallback-model",
        environment_factory=lambda _task: environment,
        temperature=0.25,
        max_tokens=128,
        top_p=0.9,
        seed=17,
        tools=({"type": "function", "function": {"name": "python"}},),
        tool_choice="auto",
    )

    result = runtime.run(_task("a", model="role-model", routing_key="local", tools=["python"]))

    assert result.final_text == "compatible"


def test_auxiliary_generation_uses_runtime_model_policy_and_structured_format() -> None:
    captured: list[Any] = []

    class Backend:
        def generate_batch(self, requests: Sequence[Any]) -> list[_Response]:
            captured.extend(requests)
            return [_echo(requests[0], content='{"paradigms": []}')]

    runtime = AlphaApolloAgentRuntime(
        Backend(),
        model="solver-model",
        environment_factory=lambda _task: object(),
        temperature=0.25,
        max_tokens=128,
        top_p=0.9,
        seed=17,
        provider_options={"extra_body": {"thinking": False}},
    )

    response = runtime.generate_auxiliary(
        request_id="packing:ada-paradigm:0",
        system="generate a paradigm",
        prompt="find a new construction family",
        response_format={"type": "json_object"},
    )

    assert response.content == '{"paradigms": []}'
    assert len(captured) == 1
    request = captured[0]
    assert request.model == "solver-model"
    assert request.messages == (
        {"role": "system", "content": "generate a paradigm"},
        {"role": "user", "content": "find a new construction family"},
    )
    assert request.sampling.temperature == 0.25
    assert request.sampling.max_tokens == 128
    assert request.sampling.top_p == 0.9
    assert request.tools == ()
    assert request.provider_options == {
        "extra_body": {"thinking": False},
        "response_format": {"type": "json_object"},
        "seed": 17,
    }


def test_generation_and_environment_identity_is_typed_stable_and_sample_specific() -> None:
    captured_batches: list[list[Any]] = []
    contexts: dict[str, Any] = {}

    class EchoBackend:
        def generate_batch(self, requests: Sequence[Any]) -> list[_Response]:
            captured_batches.append(list(requests))
            return [_echo(request) for request in requests]

    class IdentityEnvironment(_Environment):
        def init(self, context: Any) -> _Init:
            contexts[self.task_id] = context
            return super().init(context)

    def run(tasks: Sequence[AgentTask]) -> None:
        events: list[tuple[str, str, Any]] = []
        environments = {
            task.task_id: IdentityEnvironment(task.task_id, [_final()], events) for task in tasks
        }
        AlphaApolloAgentRuntime(
            EchoBackend(),
            model="fake",
            environment_factory=lambda task: environments[task.task_id],
            seed=100,
        ).run_batch(tasks)

    grouped_a = _task(
        "a",
        branch_id="branch-0",
        round_index=2,
        sample_id=0,
        routing_key="input:solve:iteration-3",
        prompt="shared problem",
    )
    grouped_b = _task(
        "b",
        branch_id="branch-4",
        round_index=2,
        sample_id=4,
        routing_key="input:solve:iteration-3",
        prompt="shared problem",
    )
    run((grouped_a, grouped_b))
    batch_request = captured_batches[-1][1]

    assert captured_batches[-1][0].group_id == captured_batches[-1][1].group_id
    assert captured_batches[-1][0].group_id.startswith("input:solve:iteration-3:0:")
    assert [request.sample_id for request in captured_batches[-1]] == [0, 4]
    assert [request.provider_options["seed"] for request in captured_batches[-1]] == [
        100,
        104,
    ]
    assert contexts["b"].branch_id == "branch-4"
    assert contexts["b"].round_index == 2
    assert contexts["b"].sample_id == 4
    assert contexts["b"].seed == 104
    assert contexts["b"].routing_key == "input:solve:iteration-3"

    run((grouped_b,))
    single_request = captured_batches[-1][0]
    assert (single_request.group_id, single_request.sample_id) == (
        batch_request.group_id,
        batch_request.sample_id,
    )
    assert single_request.provider_options["seed"] == batch_request.provider_options["seed"]

    different_prompt = AgentTask(
        task_id="c",
        system="solve",
        prompt="a different prompt",
        routing_key="input:solve:iteration-3",
        sample_id=5,
    )
    run((grouped_a, different_prompt))
    assert captured_batches[-1][0].group_id != captured_batches[-1][1].group_id


def test_generation_group_changes_between_turns_and_explicit_task_seed_wins() -> None:
    events: list[tuple[str, str, Any]] = []
    call_id = "tool"
    backend = _Backend(
        {
            "a": [
                _Response(
                    "unused",
                    "",
                    tool_calls=(_Call(call_id),),
                    finish_reason="tool_calls",
                ),
                _Response("unused", "answer"),
            ]
        }
    )
    environment = _Environment("a", [_tool(call_id), _final()], events)
    runtime = AlphaApolloAgentRuntime(
        backend,
        model="fake",
        environment_factory=lambda _task: environment,
        seed=100,
    )

    runtime.run(_task("a", routing_key="stable-group", sample_id=2, sampling_seed=999))

    requests = [batch[0] for batch in backend.batches]
    assert [request.sample_id for request in requests] == [2, 2]
    assert requests[0].group_id.startswith("stable-group:0:")
    assert requests[1].group_id.startswith("stable-group:1:")
    assert requests[0].group_id != requests[1].group_id
    assert [request.provider_options["seed"] for request in requests] == [999, 999]


def test_environment_input_is_explicit_and_not_echoed_into_result_metadata() -> None:
    events: list[tuple[str, str, Any]] = []
    captured_contexts = []

    class CapturingEnvironment(_Environment):
        def init(self, context: Any) -> _Init:
            captured_contexts.append(context)
            return super().init(context)

    environment = CapturingEnvironment("a", [_final()], events)
    backend = _Backend({"a": [_Response("unused", "answer")]})
    task = AgentTask(
        task_id="a",
        system="solve",
        prompt="question",
        environment_slot=3,
        environment_seed=17,
        task_payload={"ground_truth": "hidden"},
        metadata={"data_source": "search"},
    )

    result = AlphaApolloAgentRuntime(
        backend,
        model="fake",
        environment_factory=lambda _task: environment,
    ).run(task)

    assert captured_contexts[0].seed == 17
    assert captured_contexts[0].task_payload == {"ground_truth": "hidden"}
    assert result.metadata == {
        "data_source": "search",
        "environment_init": {"task_id": "a"},
    }


def test_task_tool_grants_filter_runtime_schemas_and_empty_grants_none() -> None:
    events: list[tuple[str, str, Any]] = []
    captured: list[list[Any]] = []

    class GrantBackend:
        def generate_batch(self, requests: Sequence[Any]) -> list[_Response]:
            captured.append(list(requests))
            return [_echo(request) for request in requests]

    environments = {
        task_id: _Environment(task_id, [_final()], events) for task_id in ("none", "python")
    }
    runtime = AlphaApolloAgentRuntime(
        GrantBackend(),
        model="fake",
        environment_factory=lambda task: environments[task.task_id],
        tools=(
            {"type": "function", "function": {"name": "python"}},
            {"type": "function", "function": {"name": "bash"}},
        ),
        tool_choice="auto",
    )

    runtime.run_batch([_task("none"), _task("python", tools=["python"])])

    assert captured[0][0].tools == ()
    assert captured[0][0].tool_choice is None
    assert captured[0][1].tools == ({"type": "function", "function": {"name": "python"}},)
    assert captured[0][1].tool_choice == "auto"


def test_unknown_task_tool_grant_fails_before_environment_allocation() -> None:
    backend = _Backend({"a": [_Response("unused", "never")]})
    created: list[str] = []

    def environment_factory(task: AgentTask) -> _Environment:
        created.append(task.task_id)
        return _Environment(task.task_id, [_final()], [])

    runtime = AlphaApolloAgentRuntime(
        backend,
        model="fake",
        environment_factory=environment_factory,
        tools=({"type": "function", "function": {"name": "python"}},),
    )

    with pytest.raises(ValueError, match="grants tools without Runtime schemas.*sudo"):
        runtime.run(_task("a", tools=["sudo"]))

    assert created == []
    assert backend.batches == []


def test_missing_or_none_task_model_uses_runtime_model() -> None:
    events: list[tuple[str, str, Any]] = []

    class ModelBackend:
        def generate_batch(self, requests: Sequence[Any]) -> list[_Response]:
            assert [request.model for request in requests] == ["fallback", "fallback"]
            return [_echo(request) for request in requests]

    environments = {
        task_id: _Environment(task_id, [_final()], events) for task_id in ("missing", "empty")
    }
    runtime = AlphaApolloAgentRuntime(
        ModelBackend(),
        model="fallback",
        environment_factory=lambda task: environments[task.task_id],
    )

    runtime.run_batch([_task("missing"), _task("empty", model=None)])


def test_task_execution_policy_is_explicit_and_validated() -> None:
    with pytest.raises(ValueError, match="model must be a non-empty"):
        AgentTask("task", "system", "prompt", model="")
    with pytest.raises(TypeError, match="routing_key must be a string"):
        AgentTask("task", "system", "prompt", routing_key=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        AgentTask("task", "system", "prompt", tools=("python", "python"))
    with pytest.raises(ValueError, match="branch_id"):
        AgentTask("task", "system", "prompt", branch_id="")
    with pytest.raises(ValueError, match="round_index"):
        AgentTask("task", "system", "prompt", round_index=-1)
    with pytest.raises(ValueError, match="sample_id"):
        AgentTask("task", "system", "prompt", sample_id=True)
    with pytest.raises(TypeError, match="sampling_seed"):
        AgentTask("task", "system", "prompt", sampling_seed="7")  # type: ignore[arg-type]


@pytest.mark.parametrize("temperature", [True, math.inf, -math.inf, math.nan, "0.5"])
def test_temperature_must_be_a_finite_number(temperature: Any) -> None:
    with pytest.raises(ValueError, match="temperature must be finite"):
        AlphaApolloAgentRuntime(
            _Backend({}),
            model="fake",
            environment_factory=lambda _task: object(),
            temperature=temperature,
        )


@pytest.mark.parametrize("top_p", [0, -0.1, 1.1, math.inf, math.nan])
def test_top_p_must_match_generation_contract(top_p: Any) -> None:
    with pytest.raises(ValueError, match="top_p"):
        AlphaApolloAgentRuntime(
            _Backend({}),
            model="fake",
            environment_factory=lambda _task: object(),
            top_p=top_p,
        )


def test_tool_choice_requires_declared_tool_schemas() -> None:
    with pytest.raises(ValueError, match="requires at least one tool schema"):
        AlphaApolloAgentRuntime(
            _Backend({}),
            model="fake",
            environment_factory=lambda _task: object(),
            tool_choice="auto",
        )


@pytest.mark.parametrize("tool_choice", ["", "   ", 7])
def test_tool_choice_must_be_a_nonempty_string(tool_choice: Any) -> None:
    with pytest.raises(ValueError, match="tool_choice must be non-empty"):
        AlphaApolloAgentRuntime(
            _Backend({}),
            model="fake",
            environment_factory=lambda _task: object(),
            tools=({"type": "function"},),
            tool_choice=tool_choice,
        )


@pytest.mark.parametrize("tools", ["python", ("python",)])
def test_tools_must_be_a_sequence_of_mappings(tools: Any) -> None:
    with pytest.raises(TypeError, match="tools must"):
        AlphaApolloAgentRuntime(
            _Backend({}),
            model="fake",
            environment_factory=lambda _task: object(),
            tools=tools,
        )


def test_window_image_history_keeps_only_the_newest_n_image_messages() -> None:
    window = runtime_module._window_image_history

    def image_message(turn: int) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": f"turn {turn}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{turn}"}},
            ],
        }

    messages = [
        {"role": "system", "content": "drive the robot"},
        image_message(0),
        {"role": "assistant", "content": "acting"},
        image_message(1),
        image_message(2),
    ]

    windowed = window(tuple(messages), 2)

    assert [m["role"] for m in windowed] == [m["role"] for m in messages]
    old_blocks = windowed[1]["content"]
    assert all(block.get("type") != "image_url" for block in old_blocks)
    assert old_blocks[0] == {"type": "text", "text": "turn 0"}
    assert old_blocks[-1] == {
        "type": "text",
        "text": runtime_module._OMITTED_IMAGES_TEXT,
    }
    assert any(block.get("type") == "image_url" for block in windowed[3]["content"])
    assert any(block.get("type") == "image_url" for block in windowed[4]["content"])
    # The slot's own history must stay complete; only the request copy changes.
    assert any(block.get("type") == "image_url" for block in messages[1]["content"])


def test_window_image_history_none_preserves_every_image() -> None:
    window = runtime_module._window_image_history
    messages = (
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}],
        },
    )

    assert window(messages, None) == messages


def test_window_image_history_counts_adjacent_image_messages_independently() -> None:
    window = runtime_module._window_image_history

    def image_message(label: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{label}"}},
            ],
        }

    # A single Environment continuation may return multiple adjacent messages;
    # the option deliberately counts messages, not implicit Environment turns.
    messages = (image_message("view-a"), image_message("view-b"))

    windowed = window(messages, 1)

    assert all(block["type"] != "image_url" for block in windowed[0]["content"])
    assert any(block["type"] == "image_url" for block in windowed[1]["content"])


def test_window_image_history_preserves_responses_content_dialect() -> None:
    window = runtime_module._window_image_history
    messages = (
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "old scene"},
                {"type": "input_image", "image_url": "data:image/png;base64,old"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "new scene"},
                {"type": "input_image", "image_url": "data:image/png;base64,new"},
            ],
        },
    )

    windowed = window(messages, 1)

    assert windowed[0]["content"] == [
        {"type": "input_text", "text": "old scene"},
        {"type": "input_text", "text": runtime_module._OMITTED_IMAGES_TEXT},
    ]
    assert windowed[1] == messages[1]
    # Request assembly must not mutate the stored trajectory snapshot.
    assert messages[0]["content"][1]["type"] == "input_image"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_image_history_messages_must_be_a_positive_integer(value: Any) -> None:
    with pytest.raises(ValueError, match="image_history_messages"):
        AlphaApolloAgentRuntime(
            _Backend({}),
            model="fake",
            environment_factory=lambda _task: object(),
            image_history_messages=value,
        )
