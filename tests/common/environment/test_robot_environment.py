# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the robot environment: lifecycle, budgets, and termination oracle."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from alphaapollo.common.artifacts.schemas import ArtifactRef
from alphaapollo.common.environment.base import (
    EnvironmentContext,
    EnvironmentLifecycleError,
    EnvironmentState,
)
from alphaapollo.common.environment.default.bridge import TextOnlyToolBridge, ToolBridgeResult
from alphaapollo.common.environment.robotics import (
    RobotEnvironment,
    RobotObservation,
    RobotTask,
    project_model_action,
)
from alphaapollo.common.execution import ToolError, ToolRequest, ToolResponse
from alphaapollo.common.execution.robotics import (
    RobotObservation as CanonicalRobotObservation,
)
from alphaapollo.common.execution.robotics import (
    RobotResetResult,
    RobotTransition,
)
from alphaapollo.common.execution.robotics import (
    RobotTask as CanonicalRobotTask,
)
from alphaapollo.common.execution.robotics.backends.fake import FakeBackend
from alphaapollo.common.execution.tools import ToolCatalog
from alphaapollo.common.execution.tools.robotics import ROBOTICS_TOOL_SPECS, VLA_ACT_SPEC
from alphaapollo.common.execution.tools.schemas import ToolCallRecord
from alphaapollo.common.generation import ToolCall
from alphaapollo.common.trajectory.recorder import EnvironmentEventKind


def _observation() -> RobotObservation:
    return RobotObservation(state=[0.0, 0.0, 0.0], artifact_refs=())


def _context(
    *,
    task_payload: dict[str, Any] | None = None,
    user_prompt: str = "pick the red cube",
) -> EnvironmentContext:
    payload = (
        {"benchmark": "fake", "environment_version": "test-v1"}
        if task_payload is None
        else task_payload
    )
    return EnvironmentContext(
        session_id="s1",
        actor="solver",
        task_id="t1",
        user_prompt=user_prompt,
        task_payload=payload,
    )


def _tool_call(
    name: str,
    arguments: str | None = None,
    call_id: str = "c1",
) -> dict[str, Any]:
    if arguments is None:
        arguments = json.dumps({"instruction": "pick the red cube"}) if name == "vla_act" else "{}"
    return {
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ]
    }


def _finish(status: str = "success") -> dict[str, Any]:
    return _tool_call(
        "finish",
        json.dumps({"status": status, "summary": "agent-requested stop"}),
    )


class FakeRobotBackend:
    """Deterministic backend with mutable authoritative episode state."""

    def __init__(self) -> None:
        self.reset_calls: list[RobotTask] = []
        self.reset_error: Exception | None = None
        self.reset_info: dict[str, Any] = {}
        self.close_count = 0
        self.closed = False
        self._terminated = False
        self._truncated = False
        self._success: bool | None = None
        self._termination_reason: str | None = None
        self.terminated_error: Exception | None = None
        self.truncated_error: Exception | None = None
        self.success_error: Exception | None = None
        self.termination_reason_error: Exception | None = None
        self.steps_used = 0
        self._observation = _observation()
        self.observe_error: Exception | None = None
        self.close_error: Exception | None = None

    def reset(self, task: RobotTask) -> RobotResetResult:
        self.reset_calls.append(task)
        if self.reset_error is not None:
            raise self.reset_error
        return RobotResetResult(observation=self._observation, info=self.reset_info)

    @property
    def terminated(self) -> bool:
        if self.terminated_error is not None:
            raise self.terminated_error
        return self._terminated

    @terminated.setter
    def terminated(self, value: bool) -> None:
        self._terminated = value

    @property
    def truncated(self) -> bool:
        if self.truncated_error is not None:
            raise self.truncated_error
        return self._truncated

    @truncated.setter
    def truncated(self, value: bool) -> None:
        self._truncated = value

    @property
    def success(self) -> bool | None:
        if self.success_error is not None:
            raise self.success_error
        return self._success

    @success.setter
    def success(self, value: bool | None) -> None:
        self._success = value

    @property
    def termination_reason(self) -> str | None:
        if self.termination_reason_error is not None:
            raise self.termination_reason_error
        return self._termination_reason

    @termination_reason.setter
    def termination_reason(self, value: str | None) -> None:
        self._termination_reason = value

    def observe(self) -> RobotObservation:
        if self.observe_error is not None:
            raise self.observe_error
        return self._observation

    def execute(self, action: Any) -> RobotTransition:
        del action
        return RobotTransition(
            observation=self._observation,
            terminated=self.terminated,
            truncated=self.truncated,
            success=self.success,
            termination_reason=self.termination_reason,
            steps_used=self.steps_used,
        )

    def close(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class SingleReadBackend(FakeRobotBackend):
    """Terminal backend whose verdict properties may be read only once.

    A second read raises, so an implementation that re-reads the properties
    after the environment's guarded snapshot lets the fault escape ``step``
    instead of ending only this episode. A real simulator handle behaves the
    same way once the episode's process or socket is gone.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reads: dict[str, int] = {}
        self._terminated = True
        self._success = True
        self._termination_reason = "task_success"

    def _read(self, name: str, value: Any) -> Any:
        self.reads[name] = self.reads.get(name, 0) + 1
        if self.reads[name] > 1:
            raise RuntimeError(f"backend {name} is readable once per step")
        return value

    @property
    def terminated(self) -> bool:
        return bool(self._read("terminated", self._terminated))

    @property
    def truncated(self) -> bool:
        return bool(self._read("truncated", self._truncated))

    @property
    def success(self) -> bool | None:
        return self._read("success", self._success)

    @property
    def termination_reason(self) -> str | None:
        return self._read("termination_reason", self._termination_reason)


class ScriptedBridge:
    """Return pre-built tool results and record each dispatched action."""

    def __init__(self, results: list[ToolBridgeResult]) -> None:
        self._results = list(results)
        self.dispatched: list[Any] = []
        self.closed = False
        self.dispatch_error: Exception | None = None
        self.close_error: Exception | None = None

    def dispatch(self, action: Any, context: Any) -> ToolBridgeResult:
        self.dispatched.append((action, context))
        if self.dispatch_error is not None:
            raise self.dispatch_error
        return self._results.pop(0)

    def close(self) -> None:
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class SelectiveFailingSink:
    def __init__(self, *failed_kinds: EnvironmentEventKind) -> None:
        self.failed_kinds = set(failed_kinds)
        self.payloads: dict[EnvironmentEventKind, dict[str, Any]] = {}

    def emit(
        self,
        kind: EnvironmentEventKind,
        session: Any,
        payload: dict[str, Any] | None = None,
        capture: Any = None,
    ) -> None:
        del session, capture
        if kind in self.failed_kinds:
            raise RuntimeError("trajectory store unavailable")
        self.payloads[kind] = dict(payload or {})


def _ok_result(call_id: str = "c1", tool_id: str = "vla_act") -> ToolBridgeResult:
    return ToolBridgeResult(
        request=ToolRequest(call_id=call_id, tool_id=tool_id),
        response=ToolResponse(call_id=call_id, tool_id=tool_id, stdout="{}"),
        record=ToolCallRecord(tool_id=tool_id, args={}, stdout="{}"),
    )


def _payload(transition: Any) -> dict[str, Any]:
    value = json.loads(transition.observation)
    assert isinstance(value, dict)
    return value


def _initial_payload(result: Any) -> dict[str, Any]:
    content = result.observation
    if isinstance(content, str):
        _, encoded = content.split("\n\n", 1)
    else:
        assert isinstance(content, list)
        encoded = content[1]["text"]
    value = json.loads(encoded)
    assert isinstance(value, dict)
    return value


def test_planner_text_does_not_terminate() -> None:
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=ScriptedBridge([]))
    env.init(_context())

    transition = env.step("task completed")

    assert transition.done is False
    assert transition.success is None
    payload = _payload(transition)
    assert payload["environment_terminated"] is False
    assert payload["environment_success"] is None
    assert payload["next_action_required"] is True
    assert payload["planner_text"] == "task completed"
    assert payload["turns_used"] == 0
    assert payload["episode_steps_used"] == 0


def test_init_fails_closed_when_image_artifacts_have_no_loader() -> None:
    backend = FakeRobotBackend()
    backend._observation = RobotObservation(
        state=[],
        artifact_refs=(ArtifactRef(id="4" * 64, type="image/png", created_by="test"),),
    )
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([]))

    with pytest.raises(RuntimeError, match="no artifact loader"):
        env.init(_context())

    assert env.state is EnvironmentState.FAILED


def test_finish_cannot_fabricate_success() -> None:
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=ScriptedBridge([]))
    env.init(_context())

    transition = env.step(_finish("success"))

    assert transition.done is False
    assert transition.success is None
    payload = _payload(transition)
    assert payload["environment_success"] is None
    assert payload["environment_terminated"] is False
    assert payload["next_action_required"] is True
    assert payload["tool"]["advisory"] is True
    assert payload["agent_claim"] == "success"


def test_custom_finish_tool_uses_the_canonical_finish_schema() -> None:
    env = RobotEnvironment(
        backend=FakeRobotBackend(),
        tool_bridge=ScriptedBridge([]),
        finish_tool_id="stop_episode",
        catalog=ToolCatalog((VLA_ACT_SPEC,)),
    )
    env.init(_context())

    transition = env.step(
        _tool_call(
            "stop_episode",
            json.dumps({"status": "stuck", "summary": "object is unreachable"}),
        )
    )

    assert transition.done is False
    payload = _payload(transition)
    assert payload["tool"]["advisory"] is True
    assert payload["agent_claim"] == "stuck"


@pytest.mark.parametrize(
    ("finish_tool_id", "catalog"),
    [("vla_act", None), ("vla_act", ToolCatalog((VLA_ACT_SPEC,)))],
)
def test_finish_tool_id_cannot_shadow_an_executable_tool(
    finish_tool_id: str,
    catalog: ToolCatalog | None,
) -> None:
    with pytest.raises(ValueError, match="collides with an executable robotics tool"):
        RobotEnvironment(
            backend=FakeRobotBackend(),
            tool_bridge=ScriptedBridge([]),
            finish_tool_id=finish_tool_id,
            catalog=catalog,
        )


def test_the_shipped_robotics_catalog_is_accepted_with_the_default_finish_id() -> None:
    """`ROBOTICS_TOOL_SPECS` bundles `FINISH_SPEC` as a declaration, not an executable.

    `build_robotics_tool_executors` registers no executor for `finish`, so the
    catalog the production factory passes carries it purely so the model sees
    the schema. Rejecting that combination would refuse the only wiring the
    workflow layer builds.
    """

    env = RobotEnvironment(
        backend=FakeRobotBackend(),
        tool_bridge=ScriptedBridge([]),
        catalog=ToolCatalog(ROBOTICS_TOOL_SPECS),
    )

    dispatchable = {spec.tool_id for spec in env._catalog.list_specs(include_internal=True)}
    assert "finish" not in dispatchable
    assert "vla_act" in dispatchable


def test_the_shipped_catalog_still_rejects_a_finish_id_naming_an_executable_tool() -> None:
    with pytest.raises(ValueError, match="collides with an executable robotics tool"):
        RobotEnvironment(
            backend=FakeRobotBackend(),
            tool_bridge=ScriptedBridge([]),
            catalog=ToolCatalog(ROBOTICS_TOOL_SPECS),
            finish_tool_id="vla_act",
        )


def test_custom_finish_tool_still_rejects_invalid_arguments() -> None:
    bridge = ScriptedBridge([])
    env = RobotEnvironment(
        backend=FakeRobotBackend(),
        tool_bridge=bridge,
        finish_tool_id="stop_episode",
        catalog=ToolCatalog((VLA_ACT_SPEC,)),
    )
    env.init(_context())

    transition = env.step(_tool_call("stop_episode", json.dumps({"status": "success"})))

    payload = _payload(transition)
    assert payload["tool"]["error"]["code"] == "invalid_arguments"
    assert "agent_claim" not in payload
    assert bridge.dispatched == []


def test_every_tool_result_exposes_episode_status() -> None:
    bridge = ScriptedBridge([_ok_result()])
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=bridge)
    env.init(_context())

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is False
    assert transition.response_format_valid is True
    assert transition.env_action_valid is True
    assert transition.action_valid is True
    payload = _payload(transition)
    for field in ("environment_terminated", "environment_success", "next_action_required"):
        assert field in payload
    assert payload["tool"]["ok"] is True


def test_backend_termination_closes_episode() -> None:
    backend = FakeRobotBackend()
    backend.terminated = True
    backend.success = True
    backend.termination_reason = "task_completed"
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([_ok_result()]))
    env.init(_context())

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is True
    assert transition.success is True
    assert transition.termination_reason == "task_completed"
    assert env.state is EnvironmentState.TERMINATED


@pytest.mark.parametrize(
    ("terminated", "truncated", "success"),
    [
        (True, False, None),
        (False, True, None),
        (False, True, True),
    ],
)
def test_backend_terminal_success_preserves_the_canonical_three_valued_state(
    terminated: bool,
    truncated: bool,
    success: bool | None,
) -> None:
    backend = FakeRobotBackend()
    backend.terminated = terminated
    backend.truncated = truncated
    backend.success = success
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([_ok_result()]))
    env.init(_context())

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is True
    assert transition.success is success
    assert transition.reward == (1.0 if success is True else 0.0)
    payload = _payload(transition)
    assert payload["environment_terminated"] is True
    assert payload["environment_success"] is success
    assert payload["next_action_required"] is False


def test_max_turns_budget_exhausted() -> None:
    bridge = ScriptedBridge([_ok_result(call_id="c1"), _ok_result(call_id="c2")])
    env = RobotEnvironment(
        backend=FakeRobotBackend(),
        tool_bridge=bridge,
        max_turns=1,
    )
    env.init(_context())

    first = env.step(_tool_call("vla_act", call_id="c1"))
    second = env.step(_tool_call("vla_act", call_id="c2"))

    assert first.done is False
    assert second.done is True
    assert second.success is False
    assert second.termination_reason == "max_turns_exhausted"
    assert len(bridge.dispatched) == 1  # the second call was never executed


def test_max_episode_steps_budget_exhausted() -> None:
    backend = FakeRobotBackend()
    bridge = ScriptedBridge([_ok_result()])
    env = RobotEnvironment(
        backend=backend,
        tool_bridge=bridge,
        max_episode_steps=5,
    )
    env.init(_context())
    backend.steps_used = 10  # the primitive consumed steps past the budget

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is True
    assert transition.success is False
    assert transition.termination_reason == "max_episode_steps_exhausted"
    payload = _payload(transition)
    assert payload["environment_terminated"] is True
    assert payload["next_action_required"] is False


def test_reset_consumed_step_budget_stops_before_any_planner_action() -> None:
    backend = FakeRobotBackend()
    backend.steps_used = 5
    bridge = ScriptedBridge([])
    env = RobotEnvironment(
        backend=backend,
        tool_bridge=bridge,
        max_episode_steps=5,
    )
    env.init(_context())

    transition = env.step("inspect before acting")

    assert transition.done is True
    assert transition.termination_reason == "max_episode_steps_exhausted"
    assert bridge.dispatched == []


def test_projection_error_is_nonterminal() -> None:
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=ScriptedBridge([]))
    env.init(_context())

    # A tool call without an id is a parse error, not a terminal answer.
    action = {
        "tool_calls": [{"type": "function", "function": {"name": "vla_act", "arguments": "{}"}}]
    }
    transition = env.step(action)

    assert transition.done is False
    assert transition.response_format_valid is False
    assert transition.env_action_valid is False
    assert transition.action_valid is False
    payload = _payload(transition)
    assert payload["tool"]["status"] == "rejected"


def test_bridge_error_is_nonterminal() -> None:
    error = ToolBridgeResult(
        request=ToolRequest(call_id="c1", tool_id="vla_act"),
        error=ToolError(
            stage="catalog",
            code="unknown_tool",
            message="unknown tool",
            call_id="c1",
            tool_id="vla_act",
        ),
    )
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=ScriptedBridge([error]))
    env.init(_context())

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is False
    assert transition.response_format_valid is True
    assert transition.env_action_valid is False
    assert transition.action_valid is False
    payload = _payload(transition)
    assert payload["tool"]["error"]["code"] == "unknown_tool"


def test_bridge_refusal_marks_the_action_unavailable() -> None:
    refused = ToolBridgeResult(
        refusal_code="tool_not_granted",
        refusal_message="tool is not granted",
    )
    env = RobotEnvironment(
        backend=FakeRobotBackend(),
        tool_bridge=ScriptedBridge([refused]),
    )
    env.init(_context())

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is False
    assert transition.response_format_valid is True
    assert transition.env_action_valid is False
    assert _payload(transition)["tool"]["error"]["code"] == "tool_not_granted"


def test_bridge_result_without_an_outcome_is_never_reported_as_advisory() -> None:
    env = RobotEnvironment(
        backend=FakeRobotBackend(),
        tool_bridge=ScriptedBridge([ToolBridgeResult()]),
    )
    env.init(_context())

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is True
    assert transition.success is False
    assert transition.termination_reason == "internal_error"
    payload = _payload(transition)
    assert "tool" not in payload
    assert "no outcome for 'vla_act'" in payload["error"]["message"]


def test_text_only_bridge_cannot_report_an_unexecuted_call_as_successful() -> None:
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=TextOnlyToolBridge())
    env.init(_context())

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is True
    assert transition.termination_reason == "internal_error"


def test_bridge_exception_is_an_episode_local_failure() -> None:
    bridge = ScriptedBridge([])
    bridge.dispatch_error = RuntimeError("executor unavailable")
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=bridge)
    env.init(_context())

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is True
    assert transition.termination_reason == "internal_error"
    assert transition.action_valid is True
    assert _payload(transition)["error"]["message"] == "executor unavailable"


def test_close_is_idempotent() -> None:
    backend = FakeRobotBackend()
    bridge = ScriptedBridge([])
    env = RobotEnvironment(backend=backend, tool_bridge=bridge)
    env.init(_context())

    env.close()
    env.close()

    assert backend.close_count == 1
    assert bridge.closed is True
    assert env.state is EnvironmentState.CLOSED


def test_close_records_all_cleanup_failures() -> None:
    backend = FakeRobotBackend()
    backend.close_error = RuntimeError("backend unavailable")
    bridge = ScriptedBridge([])
    bridge.close_error = RuntimeError("bridge unavailable")
    env = RobotEnvironment(backend=backend, tool_bridge=bridge)
    env.init(_context())

    env.close()

    assert env.session is not None
    assert env.session.cleanup_errors == [
        "RuntimeError: tool bridge cleanup failed",
        "RuntimeError: backend cleanup failed",
    ]
    assert env.state is EnvironmentState.CLOSED


def test_event_sink_failures_are_recorded_and_reported_on_close() -> None:
    sink = SelectiveFailingSink(EnvironmentEventKind.INITIALIZED)
    env = RobotEnvironment(
        backend=FakeRobotBackend(),
        tool_bridge=ScriptedBridge([]),
        event_sink=sink,
    )

    result = env.init(_context())
    env.close()

    expected = "environment_initialized: RuntimeError: event recording failed"
    assert result.metadata["session"]["event_errors"] == [expected]
    assert env.session is not None
    assert env.session.event_errors == [expected]
    assert sink.payloads[EnvironmentEventKind.CLOSED]["event_errors"] == [expected]


def test_close_before_init_logs_cleanup_failures(caplog: pytest.LogCaptureFixture) -> None:
    backend = FakeRobotBackend()
    backend.close_error = RuntimeError("backend unavailable")
    bridge = ScriptedBridge([])
    bridge.close_error = RuntimeError("bridge unavailable")
    env = RobotEnvironment(backend=backend, tool_bridge=bridge)

    with caplog.at_level("ERROR"):
        env.close()

    assert "tool bridge cleanup failed before environment close" in caplog.text
    assert "backend cleanup failed before environment close" in caplog.text
    assert env.state is EnvironmentState.CLOSED


def test_runtime_terminate_closes_the_active_episode() -> None:
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=ScriptedBridge([]))
    env.init(_context())

    env.terminate("runtime_cancelled")

    assert env.state is EnvironmentState.TERMINATED
    assert env.session is not None
    assert env.session.termination_reason == "runtime_cancelled"


def test_init_builds_robot_task() -> None:
    backend = FakeRobotBackend()
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([]))
    result = env.init(
        _context(
            task_payload={
                "benchmark": "libero",
                "instruction": "pick the red cube",
                "environment_version": "v1",
            }
        )
    )

    assert backend.reset_calls == [
        RobotTask(
            task_id="t1",
            benchmark="libero",
            instruction="pick the red cube",
            environment_version="v1",
        )
    ]
    assert isinstance(result.observation, str)
    assert result.observation.startswith("pick the red cube\n\n")
    assert _initial_payload(result)["observation"]["state"] == [0.0, 0.0, 0.0]


def test_init_projects_reset_observation_images_and_info_to_the_first_turn() -> None:
    fixed_id = "1" * 64
    wrist_id = "2" * 64
    backend = FakeRobotBackend()
    backend.reset_info = {"environment": {"suite": "libero_spatial"}}
    backend._observation = RobotObservation(
        state={"proprio": {"eef": [0.0, 0.1, 0.2]}},
        artifact_refs=(
            ArtifactRef(id=fixed_id, type="image/png", created_by="test"),
            ArtifactRef(id=wrist_id, type="image/png", created_by="test"),
        ),
        backend_metadata={
            "artifact_roles": {
                "fixed_camera": fixed_id,
                "wrist_camera": wrist_id,
            }
        },
    )
    loaded: list[str] = []

    def load_artifact(ref: ArtifactRef) -> bytes:
        loaded.append(ref.id)
        return f"frame-{ref.id}".encode()

    env = RobotEnvironment(
        backend=backend,
        tool_bridge=ScriptedBridge([]),
        load_artifact=load_artifact,
    )

    result = env.init(_context(user_prompt="open the top drawer"))

    assert loaded == [fixed_id, wrist_id]
    assert result.metadata["reset_info"] == {"environment": {"suite": "libero_spatial"}}
    assert isinstance(result.observation, list)
    assert result.observation[0] == {"type": "text", "text": "open the top drawer"}
    assert result.observation[2] == {"type": "text", "text": "FIXED_CAMERA"}
    assert result.observation[3]["type"] == "image_url"
    assert result.observation[4] == {"type": "text", "text": "WRIST_CAMERA"}
    assert result.observation[5]["type"] == "image_url"
    payload = _initial_payload(result)
    assert payload["observation"]["state"]["proprio"]["eef"] == [0.0, 0.1, 0.2]
    assert payload["turns_used"] == 0
    assert payload["episode_steps_used"] == 0


def test_init_adapts_libero_prepared_env_payload() -> None:
    backend = FakeRobotBackend()
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([]))

    env.init(
        _context(
            user_prompt="open the top drawer",
            task_payload={
                "benchmark": "libero",
                "environment_version": "libero-v1.0.1",
                "suite": "libero_spatial",
                "task_order_index": 0,
                "task_index": 3,
                "task_name": "open_drawer",
                "problem_folder": "libero_spatial",
                "bddl_file": "open_drawer.bddl",
                "init_states_file": "open_drawer.pruned_init",
                "demonstration_id": "open_drawer_demo",
                "demonstration_path": "libero_spatial/open_drawer_demo.hdf5",
                "initial_state_index": 0,
            },
        )
    )

    assert backend.reset_calls == [
        RobotTask(
            task_id="t1",
            benchmark="libero",
            instruction="open the top drawer",
            environment_version="libero-v1.0.1",
            backend_metadata={
                "suite": "libero_spatial",
                "task_order_index": 0,
                "task_index": 3,
                "task_name": "open_drawer",
                "problem_folder": "libero_spatial",
                "bddl_file": "open_drawer.bddl",
                "init_states_file": "open_drawer.pruned_init",
                "demonstration_id": "open_drawer_demo",
                "demonstration_path": "libero_spatial/open_drawer_demo.hdf5",
                "initial_state_index": 0,
            },
        )
    ]


def test_init_adapts_robocasa_prepared_env_payload() -> None:
    backend = FakeRobotBackend()
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([]))

    env.init(
        _context(
            user_prompt="set up the coffee station",
            task_payload={
                "benchmark": "robocasa",
                "environment_version": "robocasa-921c9a5+robosuite-5ce6643",
                "task_name": "CoffeeSetup",
                "layout_id": 2,
                "style_id": 5,
                "scene_seed": 42,
            },
        )
    )

    assert backend.reset_calls == [
        RobotTask(
            task_id="t1",
            benchmark="robocasa",
            instruction="set up the coffee station",
            environment_version="robocasa-921c9a5+robosuite-5ce6643",
            backend_metadata={
                "task_name": "CoffeeSetup",
                "layout_id": 2,
                "style_id": 5,
                "scene_seed": 42,
            },
        )
    ]


def test_init_keeps_nested_backend_metadata_compatibility() -> None:
    backend = FakeRobotBackend()
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([]))

    env.init(
        _context(
            task_payload={
                "benchmark": "libero",
                "environment_version": "libero-v1.0.1",
                "backend_metadata": {"suite": "libero_spatial", "task_index": 3},
            }
        )
    )

    assert backend.reset_calls[0].backend_metadata == {
        "suite": "libero_spatial",
        "task_index": 3,
    }


def test_init_rejects_duplicate_flat_and_nested_backend_metadata() -> None:
    backend = FakeRobotBackend()
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([]))

    with pytest.raises(ValueError, match="must not appear both flat"):
        env.init(
            _context(
                task_payload={
                    "benchmark": "libero",
                    "environment_version": "libero-v1.0.1",
                    "suite": "libero_spatial",
                    "backend_metadata": {"suite": "libero_object"},
                }
            )
        )

    assert backend.reset_calls == []


def test_environment_reexports_canonical_robot_contracts() -> None:
    assert RobotTask is CanonicalRobotTask
    assert RobotObservation is CanonicalRobotObservation


def test_canonical_fake_backend_initializes_without_an_adapter() -> None:
    backend = FakeBackend(initial_observation=_observation())
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([]))

    env.init(_context())

    assert backend.task == RobotTask(
        task_id="t1",
        benchmark="fake",
        instruction="pick the red cube",
        environment_version="test-v1",
    )
    env.close()


def test_init_rejects_missing_robot_task_identity() -> None:
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=ScriptedBridge([]))

    with pytest.raises(ValueError, match="benchmark"):
        env.init(_context(task_payload={"environment_version": "test-v1"}))

    other = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=ScriptedBridge([]))
    with pytest.raises(ValueError, match="environment_version"):
        other.init(_context(task_payload={"benchmark": "fake"}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("benchmark", 123),
        ("environment_version", 456),
        ("instruction", ["private instruction"]),
        ("backend_metadata", "not-a-mapping"),
    ],
)
def test_init_rejects_malformed_robot_task_payload_fields(
    field: str,
    value: Any,
) -> None:
    payload: dict[str, Any] = {
        "benchmark": "fake",
        "environment_version": "test-v1",
        "instruction": "pick the red cube",
        "backend_metadata": {},
    }
    payload[field] = value
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=ScriptedBridge([]))

    with pytest.raises(TypeError, match=field):
        env.init(_context(task_payload=payload))


def test_init_rejects_private_instruction_that_differs_from_public_prompt() -> None:
    backend = FakeRobotBackend()
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([]))

    with pytest.raises(ValueError, match="must match the public user_prompt"):
        env.init(
            _context(
                user_prompt="public instruction",
                task_payload={
                    "benchmark": "fake",
                    "environment_version": "test-v1",
                    "instruction": "private instruction",
                },
            )
        )

    assert backend.reset_calls == []


def test_init_requires_the_public_user_prompt_even_with_a_private_instruction() -> None:
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=ScriptedBridge([]))

    with pytest.raises(ValueError, match="user_prompt"):
        env.init(
            _context(
                user_prompt="",
                task_payload={
                    "benchmark": "fake",
                    "environment_version": "test-v1",
                    "instruction": "private instruction",
                },
            )
        )


def test_failed_reset_is_not_retried_and_cleanup_remains_observable() -> None:
    backend = FakeRobotBackend()
    backend.reset_error = RuntimeError("simulator reset failed")
    backend.close_error = RuntimeError("simulator cleanup failed")
    sink = SelectiveFailingSink()
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([]), event_sink=sink)

    with pytest.raises(RuntimeError, match="simulator reset failed"):
        env.init(_context())

    assert env.state is EnvironmentState.FAILED
    assert sink.payloads[EnvironmentEventKind.FAILED] == {
        "stage": "init",
        "error_type": "RuntimeError",
    }
    assert EnvironmentEventKind.INITIALIZED not in sink.payloads
    with pytest.raises(EnvironmentLifecycleError, match="init requires created state"):
        env.init(_context())
    assert len(backend.reset_calls) == 1

    env.close()

    assert env.session is not None
    assert env.session.cleanup_errors == ["RuntimeError: backend cleanup failed"]
    assert env.state is EnvironmentState.CLOSED


def test_catalog_rejects_invalid_arguments_without_dispatching() -> None:
    bridge = ScriptedBridge([])
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=bridge)
    env.init(_context())

    transition = env.step(_tool_call("vla_act", "{}"))

    assert transition.done is False
    assert transition.response_format_valid is True
    assert transition.env_action_valid is False
    payload = _payload(transition)
    assert payload["tool"]["error"]["code"] == "invalid_arguments"
    assert bridge.dispatched == []


@pytest.mark.parametrize("action", ["planner prose", _tool_call("vla_act")])
def test_observation_fault_is_an_episode_local_failure(action: Any) -> None:
    backend = FakeRobotBackend()
    bridge = ScriptedBridge([] if isinstance(action, str) else [_ok_result()])
    env = RobotEnvironment(backend=backend, tool_bridge=bridge)
    env.init(_context())
    backend.observe_error = RuntimeError("camera offline")

    transition = env.step(action)

    assert transition.done is True
    assert transition.success is False
    assert transition.termination_reason == "internal_error"
    payload = _payload(transition)
    assert payload["error"]["message"] == "camera offline"
    assert payload["environment_terminated"] is True
    assert payload["next_action_required"] is False


@pytest.mark.parametrize(
    ("action", "field"),
    [
        ("planner prose", "success_error"),
        (_tool_call("vla_act"), "termination_reason_error"),
    ],
)
def test_backend_status_fault_is_an_episode_local_failure(action: Any, field: str) -> None:
    backend = FakeRobotBackend()
    bridge = ScriptedBridge([] if isinstance(action, str) else [_ok_result()])
    env = RobotEnvironment(backend=backend, tool_bridge=bridge)
    env.init(_context())
    if field == "termination_reason_error":
        backend.terminated = True
    setattr(backend, field, RuntimeError("simulator handle is gone"))

    transition = env.step(action)

    assert transition.done is True
    assert transition.termination_reason == "internal_error"
    assert _payload(transition)["error"]["message"] == "simulator handle is gone"


def test_planner_text_uses_the_snapshotted_backend_terminal_state() -> None:
    backend = SingleReadBackend()
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([]))
    env.init(_context())

    transition = env.step("task appears complete")

    assert transition.done is True
    assert transition.success is True
    assert transition.termination_reason == "task_success"
    assert backend.reads == {
        "terminated": 1,
        "truncated": 1,
        "success": 1,
        "termination_reason": 1,
    }


def test_backend_verdict_read_after_the_snapshot_stays_episode_local() -> None:
    backend = SingleReadBackend()
    backend.reads["success"] = 1
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([]))
    env.init(_context())

    transition = env.step("task appears complete")

    assert transition.done is True
    assert transition.termination_reason == "internal_error"
    assert _payload(transition)["error"]["message"] == "backend success is readable once per step"


@pytest.mark.parametrize("success", [1, 0, "yes"])
def test_non_bool_backend_success_does_not_escape_the_step(success: Any) -> None:
    backend = FakeRobotBackend()
    backend.terminated = True
    backend.termination_reason = "task_completed"
    backend.success = success
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([_ok_result()]))
    env.init(_context())

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is True
    assert transition.success is bool(success)
    assert transition.reward == (1.0 if success else 0.0)
    assert _payload(transition)["environment_success"] is bool(success)


@pytest.mark.parametrize("reason", ["   ", 7, b"done"])
def test_unusable_backend_termination_reason_falls_back_to_the_environment(reason: Any) -> None:
    backend = FakeRobotBackend()
    backend.terminated = True
    backend.termination_reason = reason
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([_ok_result()]))
    env.init(_context())

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is True
    assert transition.termination_reason == "environment_terminated"


def test_mid_episode_backend_success_is_reported_as_unknown() -> None:
    backend = FakeRobotBackend()
    backend.success = False
    env = RobotEnvironment(backend=backend, tool_bridge=ScriptedBridge([_ok_result()]))
    env.init(_context())

    transition = env.step(_tool_call("vla_act"))

    assert transition.done is False
    payload = _payload(transition)
    assert payload["next_action_required"] is True
    assert payload["environment_success"] is None


def test_single_tool_call_continuation_keeps_the_matching_call() -> None:
    env = RobotEnvironment(
        backend=FakeRobotBackend(),
        tool_bridge=ScriptedBridge([_ok_result()]),
    )
    env.init(_context())
    transition = env.step(_tool_call("vla_act"))
    response = SimpleNamespace(
        content="",
        reasoning_content="inspect the scene before acting again",
        tool_calls=(ToolCall(id="c1", name="vla_act", arguments="{}"),),
    )

    messages = env.continuation_messages(response, transition)

    assert messages[0]["reasoning_content"] == "inspect the scene before acting again"
    assert [call["id"] for call in messages[0]["tool_calls"]] == ["c1"]
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "c1"


def test_terminal_transition_does_not_emit_continuation_messages() -> None:
    backend = FakeRobotBackend()
    backend.terminated = True
    backend.success = True
    backend.termination_reason = "task_success"
    env = RobotEnvironment(
        backend=backend,
        tool_bridge=ScriptedBridge([_ok_result()]),
    )
    env.init(_context())
    transition = env.step(_tool_call("vla_act"))

    messages = env.continuation_messages(
        SimpleNamespace(content="", reasoning_content=None, tool_calls=()),
        transition,
    )

    assert messages == ()


def test_artifact_load_fault_is_an_episode_local_failure() -> None:
    digest = "0" * 64
    backend = FakeRobotBackend()

    def missing_artifact(ref: Any) -> bytes:
        del ref
        raise ValueError("artifact content is missing")

    env = RobotEnvironment(
        backend=backend,
        tool_bridge=ScriptedBridge([]),
        load_artifact=missing_artifact,
    )
    env.init(_context())
    backend._observation = RobotObservation(
        state=[],
        artifact_refs=(
            ArtifactRef(
                id=digest,
                location=f"00/{digest}",
                hash=digest,
                type="image/png",
                created_by="test",
            ),
        ),
    )

    transition = env.step("inspect the scene")

    assert transition.done is True
    assert transition.success is False
    assert transition.termination_reason == "internal_error"
    assert _payload(transition)["error"]["message"] == "artifact content is missing"


@pytest.mark.parametrize(
    ("loaded", "message"),
    [
        (b"", "is empty"),
        ("not-bytes", "loader must return bytes"),
    ],
)
def test_invalid_image_artifact_is_an_episode_local_failure(
    loaded: object,
    message: str,
) -> None:
    digest = "3" * 64
    backend = FakeRobotBackend()
    env = RobotEnvironment(
        backend=backend,
        tool_bridge=ScriptedBridge([]),
        load_artifact=lambda ref: loaded,  # type: ignore[return-value]
    )
    env.init(_context())
    backend._observation = RobotObservation(
        state=[],
        artifact_refs=(ArtifactRef(id=digest, type="image/png", created_by="test"),),
    )

    transition = env.step("inspect the scene")

    assert transition.done is True
    assert transition.termination_reason == "internal_error"
    assert message in _payload(transition)["error"]["message"]


def test_continuation_uses_images_preloaded_during_the_environment_step() -> None:
    digest = "1" * 64
    backend = FakeRobotBackend()
    loaded: list[Any] = []

    def load_artifact(ref: Any) -> bytes:
        loaded.append(ref)
        return b"frame"

    env = RobotEnvironment(
        backend=backend,
        tool_bridge=ScriptedBridge([]),
        load_artifact=load_artifact,
    )
    env.init(_context())
    backend._observation = RobotObservation(
        state=[],
        artifact_refs=(
            ArtifactRef(
                id=digest,
                location=f"11/{digest}",
                hash=digest,
                type="image/png",
                created_by="test",
            ),
        ),
    )
    transition = env.step("inspect the scene")

    messages = env.continuation_messages(
        SimpleNamespace(content="", reasoning_content=None, tool_calls=()),
        transition,
    )

    assert len(loaded) == 1
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"][0]["type"] == "image_url"
    assert messages[-1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_parallel_tool_call_error_produces_a_valid_continuation() -> None:
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=ScriptedBridge([]))
    env.init(_context())
    action = {
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "vla_act", "arguments": "{}"},
            },
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "vla_act", "arguments": "{}"},
            },
        ]
    }
    transition = env.step(action)
    response = SimpleNamespace(
        content="",
        tool_calls=(
            ToolCall(id="c1", name="vla_act", arguments="{}"),
            ToolCall(id="c2", name="vla_act", arguments="{}"),
        ),
    )

    messages = env.continuation_messages(response, transition)

    assert "tool_calls" not in messages[0]
    assert messages[1]["role"] == "user"


def test_project_model_action_text_is_none() -> None:
    assert project_model_action("some planner prose") is None


def test_project_model_action_does_not_treat_python_tags_as_robot_tools() -> None:
    assert project_model_action("<python_code>print('inspect')</python_code>") is None
    assert project_model_action({"content": "<python_code>print('inspect')</python_code>"}) is None


def test_project_model_action_tool_call() -> None:
    projected = project_model_action(_tool_call("vla_act"))
    assert isinstance(projected, ToolRequest)
    assert projected.tool_id == "vla_act"


def test_project_response_preserves_tool_call_content_and_reasoning() -> None:
    bridge = ScriptedBridge([_ok_result()])
    env = RobotEnvironment(backend=FakeRobotBackend(), tool_bridge=bridge)
    env.init(_context())
    response = SimpleNamespace(
        content="move carefully",
        reasoning_content="the cube is near the gripper",
        tool_calls=(
            ToolCall(
                id="c1",
                name="vla_act",
                arguments=json.dumps({"instruction": "pick the red cube"}),
            ),
        ),
    )

    projected = env.project_response(response)
    transition = env.step(projected)

    assert projected["content"] == "move carefully"
    assert projected["reasoning_content"] == "the cube is near the gripper"
    assert transition.done is False
    dispatched = bridge.dispatched[0][0]
    assert "content" not in dispatched
    assert "reasoning_content" not in dispatched
    assert dispatched["tool_calls"][0]["function"]["name"] == "vla_act"
    assert env.project_response(SimpleNamespace(content="inspect", tool_calls=())) == "inspect"


def test_structured_robot_call_ignores_python_tags_in_assistant_content() -> None:
    action = {
        "content": "<python_code>print('reasoning only')</python_code>",
        **_tool_call("vla_act"),
    }

    projected = project_model_action(action)

    assert isinstance(projected, ToolRequest)
    assert projected.tool_id == "vla_act"


def test_observation_payload_serializes_nested_frozen_backend_metadata() -> None:
    from alphaapollo.common.environment.robotics.projection import observation_to_payload

    observation = RobotObservation(
        state={"proprio": {"robot0_eef_pos": [0.0, 0.1, 1.0]}},
        backend_metadata={
            "backend": "libero",
            "environment": {"suite": "libero_spatial", "camera": {"height": 256}},
            "artifact_roles": {"fixed_camera": "abc123"},
        },
    )

    payload = observation_to_payload(observation)

    encoded = json.loads(json.dumps(payload, sort_keys=True))
    assert encoded["backend_metadata"]["environment"]["camera"]["height"] == 256
    assert encoded["state"]["proprio"]["robot0_eef_pos"] == [0.0, 0.1, 1.0]
