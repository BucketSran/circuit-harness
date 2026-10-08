"""Semantic robotics tool and provider-adapter contract tests."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from alphaapollo.common.environment.default import ExecutorToolBridge
from alphaapollo.common.execution.robotics import (
    RobotAction,
    RobotGroundedPoint,
    RobotObservation,
    RobotResetResult,
    RobotTask,
    RobotTransition,
)
from alphaapollo.common.execution.tools import (
    ExecutionContext,
    ToolCatalog,
    ToolError,
    ToolExecutor,
    ToolRequest,
    ToolResponse,
    ToolSpec,
)
from alphaapollo.common.execution.tools.robotics import (
    FINISH_SPEC,
    ROBOTICS_EXECUTABLE_SPECS,
    ROBOTICS_TOOL_SPECS,
    MCPPerceptionProvider,
    MCPProviderError,
    MCPVLAProvider,
    PerceptionRequest,
    PerceptionResult,
    RoboticsToolExecutor,
    VLARequest,
    VLAResult,
    build_robotics_tool_executors,
)
from alphaapollo.common.execution.tools.robotics import mcp as mcp_adapters


def _observation(*, frame: str = "frame-1") -> RobotObservation:
    return RobotObservation(
        state={"camera": {"artifact_id": f"sha256:{frame}", "media_type": "image/png"}},
        artifact_refs=({"id": frame, "location": f"artifact://{frame}"},),
        timestamp="2026-08-16T00:00:00Z",
        backend_metadata={"backend": "fake"},
    )


@dataclass
class _VLAProvider:
    requests: list[VLARequest] = field(default_factory=list)
    closes: int = 0

    def predict(self, request: VLARequest) -> VLAResult:
        self.requests.append(request)
        return VLAResult(
            actions=(
                RobotAction(
                    kind="joint_delta",
                    arguments={"values": [0.1, 0.0]},
                    provenance={"provider": "fake-vla"},
                ),
                RobotAction(
                    kind="set_gripper",
                    arguments={"state": "close"},
                    provenance={"provider": "fake-vla"},
                ),
            ),
            provider="fake-vla",
            model="fake-v1",
            metadata={"checkpoint": "artifact://vla/fake-v1"},
        )

    def close(self) -> None:
        self.closes += 1


@dataclass
class _Backend:
    calls: list[RobotAction] = field(default_factory=list)
    closes: int = 0
    terminate_after: int | None = None
    terminal_success: bool = False

    @property
    def terminated(self) -> bool:
        return self.terminate_after is not None and len(self.calls) >= self.terminate_after

    @property
    def truncated(self) -> bool:
        return False

    @property
    def success(self) -> bool | None:
        return self.terminal_success if self.terminated else None

    @property
    def termination_reason(self) -> str | None:
        return "task_success" if self.terminated else None

    @property
    def steps_used(self) -> int:
        return len(self.calls) * 3

    def reset(self, task: RobotTask) -> RobotResetResult:
        return RobotResetResult(observation=self.observe(), info={"task_id": task.task_id})

    def observe(self) -> RobotObservation:
        return _observation()

    def execute(self, action: RobotAction) -> RobotTransition:
        self.calls.append(action)
        terminal = self.terminate_after is not None and len(self.calls) >= self.terminate_after
        return RobotTransition(
            observation=_observation(frame=f"frame-{len(self.calls) + 1}"),
            steps_used=3,
            terminated=terminal,
            truncated=False,
            success=self.terminal_success if terminal else None,
            termination_reason="task_success" if terminal else None,
            info={"backend": "fake"},
        )

    def close(self) -> None:
        self.closes += 1


@dataclass
class _PerceptionProvider:
    requests: list[PerceptionRequest] = field(default_factory=list)
    closes: int = 0

    def inspect(self, request: PerceptionRequest) -> PerceptionResult:
        self.requests.append(request)
        item = (
            {
                "label": request.query,
                "mask_ref": "artifact://mask/1",
                "score": 0.9,
                "centroid": [1.0, 2.0],
            }
            if request.operation == "segment"
            else {"label": request.query or "cup", "box": [1, 2, 3, 4], "score": 0.8}
        )
        return PerceptionResult(
            items=(item,),
            provider="fake-perception",
            model="fake-p1",
        )

    def close(self) -> None:
        self.closes += 1


def _handlers() -> tuple[
    Mapping[str, ToolExecutor],
    _VLAProvider,
    _PerceptionProvider,
    _Backend,
]:
    vla = _VLAProvider()
    perception = _PerceptionProvider()
    backend = _Backend()
    handlers = build_robotics_tool_executors(
        vla_provider=vla,
        perception_provider=perception,
        backend=backend,
    )
    return handlers, vla, perception, backend


def _invoke(
    handlers: Mapping[str, ToolExecutor],
    request: ToolRequest,
    context: ExecutionContext,
) -> ToolResponse:
    resolved = ToolCatalog(ROBOTICS_EXECUTABLE_SPECS).resolve(request)
    assert not isinstance(resolved, ToolError)
    return handlers[request.tool_id].execute(request, context)


def test_specs_are_semantic_provider_neutral_and_catalog_compatible() -> None:
    expected = [
        "finish",
        "vla_act",
        "segment",
        "detect_objects",
        "view_env_state",
        "view_camera_meta",
        "back_project",
        "set_gripper",
        "retreat",
        "move_to",
        "rotate_wrist",
    ]
    assert [spec.tool_id for spec in ROBOTICS_TOOL_SPECS] == expected
    assert [spec.tool_id for spec in ToolCatalog(ROBOTICS_TOOL_SPECS).list_specs()] == expected
    assert "advisory" in FINISH_SPEC.description
    assert all("pi0" not in spec.tool_id.lower() for spec in ROBOTICS_TOOL_SPECS)


def test_catalog_enforces_semantic_tool_schema_constraints() -> None:
    catalog = ToolCatalog(ROBOTICS_TOOL_SPECS)
    invalid_requests = (
        ToolRequest(
            call_id="finish-invalid",
            tool_id="finish",
            arguments={"status": "maybe", "summary": "unclear"},
        ),
        ToolRequest(
            call_id="segment-invalid",
            tool_id="segment",
            arguments={"query": "cup", "point": [0.1, 0.2, 0.3]},
        ),
        ToolRequest(
            call_id="detect-invalid",
            tool_id="detect_objects",
            arguments={"min_score": 1.1},
        ),
        ToolRequest(
            call_id="vla-extra",
            tool_id="vla_act",
            arguments={"instruction": "move", "provider": "pi0"},
        ),
        ToolRequest(
            call_id="retreat-negative",
            tool_id="retreat",
            arguments={"distance_m": -0.5},
        ),
        ToolRequest(
            call_id="move-negative-tolerance",
            tool_id="move_to",
            arguments={"x": 0.0, "y": 0.0, "z": 0.5, "tolerance_m": -5},
        ),
        ToolRequest(
            call_id="back-project-invalid",
            tool_id="back_project",
            arguments={"pixel": {"x": -1, "y": 2}},
        ),
        ToolRequest(
            call_id="vla-unbounded",
            tool_id="vla_act",
            arguments={"instruction": "move", "max_actions": 1_000_000},
        ),
    )

    errors = tuple(catalog.resolve(request) for request in invalid_requests)

    assert all(isinstance(error, ToolError) for error in errors)
    assert "must be one of" in errors[0].message
    assert "at most 2" in errors[1].message
    assert "less than or equal to 1.0" in errors[2].message
    assert "provider is not allowed" in errors[3].message
    # exclusiveMinimum is enforced, not silently skipped: a negative retreat
    # would drive the gripper into the table and report success.
    assert "must be greater than 0" in errors[4].message
    assert "must be greater than 0" in errors[5].message
    assert "greater than or equal to 0" in errors[6].message
    assert "less than or equal to 200" in errors[7].message


def test_catalog_rejects_schema_keywords_validation_cannot_enforce() -> None:
    spec = ToolSpec(
        tool_id="bad_tool",
        description="tool with an unenforceable constraint",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "pattern": "^[a-z]+$"}},
            "additionalProperties": False,
        },
        source="test",
    )

    with pytest.raises(ValueError, match="pattern"):
        ToolCatalog((spec,))


def test_vla_act_predicts_executes_and_returns_updated_environment_state() -> None:
    handlers, provider, _, backend = _handlers()

    response = _invoke(
        handlers,
        ToolRequest(
            call_id="vla-1",
            tool_id="vla_act",
            arguments={"instruction": "pick up the cup", "max_actions": 4},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=12),
    )
    payload = json.loads(response.stdout)

    assert response.exit_code == 0
    assert provider.requests[0].observation.state["camera"]["artifact_id"] == "sha256:frame-1"
    # The call-level timeout is one budget shared across re-predictions, so
    # each request sees the remaining allowance, never a fresh 12s.
    assert 0 < provider.requests[0].timeout_s <= 12
    assert provider.requests[1].timeout_s <= provider.requests[0].timeout_s
    # The budget of 4 spans two predict/execute chunks of two actions each,
    # each chunk conditioned on the fresh observation.
    assert len(provider.requests) == 2
    assert provider.requests[1].max_actions == 2
    assert [action.kind for action in backend.calls] == [
        "joint_delta",
        "set_gripper",
        "joint_delta",
        "set_gripper",
    ]
    assert payload["actions_executed"] == 4
    assert payload["steps_used"] == 12
    assert payload["provider"] == "fake-vla"
    assert payload["observation"]["state"]["camera"]["artifact_id"] == "sha256:frame-5"
    assert payload["environment_terminated"] is False
    assert payload["environment_success"] is None
    assert payload["next_action_required"] is True


def test_vla_act_timeout_budget_spans_re_predictions(monkeypatch) -> None:
    from alphaapollo.common.execution.tools.robotics import vla as vla_module

    ticks = iter((0.0, 1.0, 11.0))
    monkeypatch.setattr(vla_module, "_monotonic", lambda: next(ticks))
    handlers, provider, _, backend = _handlers()

    response = _invoke(
        handlers,
        ToolRequest(
            call_id="vla-budget",
            tool_id="vla_act",
            arguments={"instruction": "pick up the cup", "max_actions": 10},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=10),
    )
    payload = json.loads(response.stdout)

    # One chunk ran inside the budget; the second re-prediction found the
    # deadline spent and returned the partial result instead of paying for
    # another provider round trip.
    assert response.exit_code == 0
    assert len(provider.requests) == 1
    assert payload["actions_executed"] == len(backend.calls)
    assert payload["actions_executed"] < 10


def test_vla_act_fails_recoverably_when_budget_expires_before_any_action(
    monkeypatch,
) -> None:
    from alphaapollo.common.execution.tools.robotics import vla as vla_module

    ticks = iter((0.0, 11.0))
    monkeypatch.setattr(vla_module, "_monotonic", lambda: next(ticks))
    handlers, provider, _, _ = _handlers()

    response = _invoke(
        handlers,
        ToolRequest(
            call_id="vla-expired",
            tool_id="vla_act",
            arguments={"instruction": "pick up the cup"},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=10),
    )

    assert response.exit_code == 1
    assert "timeout budget exhausted" in response.stderr
    assert provider.requests == []


def test_vla_act_refuses_terminal_episode_without_provider_round_trip() -> None:
    handlers, provider, _, backend = _handlers()
    backend.terminate_after = 0

    response = _invoke(
        handlers,
        ToolRequest(
            call_id="vla-after-end",
            tool_id="vla_act",
            arguments={"instruction": "pick up the cup"},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=10),
    )

    assert response.exit_code == 1
    assert "already terminal" in response.stderr
    assert provider.requests == []


def test_vla_act_stops_chunk_on_backend_terminal_success() -> None:
    handlers, _, _, backend = _handlers()
    backend.terminate_after = 1
    backend.terminal_success = True

    response = _invoke(
        handlers,
        ToolRequest(
            call_id="vla-terminal",
            tool_id="vla_act",
            arguments={"instruction": "pick up the cup"},
        ),
        ExecutionContext(session_id="episode-1"),
    )
    payload = json.loads(response.stdout)

    assert len(backend.calls) == 1
    assert payload["actions_requested"] == 2
    assert payload["actions_executed"] == 1
    assert payload["environment_terminated"] is True
    assert payload["environment_success"] is True
    assert payload["next_action_required"] is False
    assert payload["termination_reason"] == "task_success"


def test_perception_tools_read_observation_without_executing_backend() -> None:
    handlers, _, provider, backend = _handlers()

    segment = _invoke(
        handlers,
        ToolRequest(
            call_id="segment-1",
            tool_id="segment",
            arguments={"query": "red cup", "point": [0.4, 0.6]},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=8),
    )
    detections = _invoke(
        handlers,
        ToolRequest(
            call_id="detect-1",
            tool_id="detect_objects",
            arguments={"min_score": 0.4},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=8),
    )

    segment_payload = json.loads(segment.stdout)
    detection_payload = json.loads(detections.stdout)
    assert [request.operation for request in provider.requests] == [
        "segment",
        "detect_objects",
    ]
    assert segment_payload["items"][0]["mask_ref"] == "artifact://mask/1"
    assert segment_payload["items"][0]["world_xyz"] is None
    assert segment_payload["items"][0]["grounding_valid"] is False
    assert segment_payload["observation"]["state"]["camera"]["artifact_id"] == ("sha256:frame-1")
    assert detection_payload["items"][0]["label"] == "cup"
    assert backend.calls == []


def test_handlers_close_providers_without_taking_backend_ownership() -> None:
    handlers, vla, perception, backend = _handlers()
    unique_handlers = {id(handler): handler for handler in handlers.values()}

    for handler in unique_handlers.values():
        handler.close()

    assert vla.closes == 1
    assert perception.closes == 1
    assert backend.closes == 0


def test_composite_executor_dispatches_and_closes_shared_handlers_once() -> None:
    handlers, vla, perception, backend = _handlers()
    executor = RoboticsToolExecutor(handlers)

    response = executor.execute(
        ToolRequest(
            call_id="vla-composite",
            tool_id="vla_act",
            arguments={"instruction": "pick up the cup", "max_actions": 2},
        ),
        ExecutionContext(session_id="episode-composite"),
    )
    executor.close()
    executor.close()

    assert response.exit_code == 0
    assert len(backend.calls) == 2
    assert vla.closes == 1
    assert perception.closes == 1
    assert backend.closes == 0


def test_robot_environment_applies_the_catalog_before_attempting_a_tool() -> None:
    # Argument validation stays a precondition of execution: the robotics
    # environment resolves requests through the catalog before dispatch, so
    # an invalid call never reaches the backend.
    from alphaapollo.common.environment.base import EnvironmentContext
    from alphaapollo.common.environment.robotics import RobotEnvironment

    handlers, _, _, backend = _handlers()
    environment = RobotEnvironment(
        backend=backend,
        tool_bridge=ExecutorToolBridge(RoboticsToolExecutor(handlers)),
        catalog=ToolCatalog(ROBOTICS_TOOL_SPECS),
    )
    environment.init(
        EnvironmentContext(
            session_id="episode-catalog",
            actor="solver",
            task_id="t1",
            user_prompt="move",
            task_payload={"benchmark": "fake", "environment_version": "test-v1"},
        )
    )

    transition = environment.step(
        {
            "tool_calls": [
                {
                    "id": "bad-vla",
                    "type": "function",
                    "function": {
                        "name": "vla_act",
                        "arguments": json.dumps({"instruction": "move", "provider": "not-allowed"}),
                    },
                }
            ]
        }
    )

    payload = json.loads(transition.observation)
    assert transition.done is False
    assert payload["tool"]["error"]["stage"] == "catalog"
    assert backend.calls == []


@dataclass
class _MCPClient:
    response: object
    calls: list[tuple[str, dict[str, Any], float | None]] = field(default_factory=list)
    closed: bool = False

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> object:
        self.calls.append((name, arguments, timeout_s))
        return self.response

    def close(self) -> None:
        self.closed = True


@dataclass
class _MCPResult:
    structured_content: dict[str, Any] | None = None
    content: tuple[object, ...] = ()


@dataclass
class _TextBlock:
    text: str


def test_mcp_vla_adapter_hides_transport_and_normalizes_structured_result() -> None:
    client = _MCPClient(
        _MCPResult(
            structured_content={
                "actions": [
                    {
                        "kind": "joint_delta",
                        "arguments": {"values": [0.1]},
                        "provenance": {"request_id": "remote-action-1"},
                    }
                ],
                "provider": "untrusted-remote-name",
                "model": "untrusted-remote-model",
                "metadata": {"request_id": "remote-1"},
            }
        )
    )
    provider = MCPVLAProvider(client=client, provider="remote-vla", model="vla-1")

    result = provider.predict(
        VLARequest(
            instruction="move left",
            observation=_observation(),
            max_actions=3,
            timeout_s=5,
        )
    )

    assert result.provider == "remote-vla"
    assert result.model == "vla-1"
    assert result.metadata["reported_provider"] == "untrusted-remote-name"
    assert result.metadata["reported_model"] == "untrusted-remote-model"
    assert result.actions[0].to_dict()["arguments"]["values"] == [0.1]
    assert result.actions[0].provenance["provider"] == "remote-vla"
    assert result.actions[0].provenance["transport"] == "mcp"
    assert client.calls[0][0] == "vla_act"
    assert client.calls[0][2] == 5


def test_mcp_perception_adapter_normalizes_json_text_content() -> None:
    client = _MCPClient(
        _MCPResult(
            content=(
                _TextBlock(
                    json.dumps(
                        {
                            "items": [
                                {
                                    "label": "cup",
                                    "mask_ref": "artifact://mask/remote",
                                }
                            ]
                        }
                    )
                ),
            )
        )
    )
    provider = MCPPerceptionProvider(
        client=client,
        provider="remote-perception",
        segment_tool_name="sam_segment",
    )

    result = provider.inspect(
        PerceptionRequest(
            operation="segment",
            observation=_observation(),
            query="cup",
            timeout_s=7,
        )
    )

    assert result.items[0]["mask_ref"] == "artifact://mask/remote"
    assert client.calls[0][0] == "sam_segment"
    assert client.calls[0][2] == 7


def test_mcp_vla_adapter_surfaces_server_error_text() -> None:
    client = _MCPClient(
        _MCPResult(content=(_TextBlock("Error executing tool vla_act: artifact not found"),))
    )
    provider = MCPVLAProvider(client=client, provider="remote-vla")

    with pytest.raises(MCPProviderError, match="server said: Error executing tool vla_act"):
        provider.predict(
            VLARequest(
                instruction="move left",
                observation=_observation(),
                max_actions=3,
                timeout_s=5,
            )
        )


def test_mcp_provider_close_releases_owned_client() -> None:
    vla_client = _MCPClient(_MCPResult(structured_content={"actions": []}))
    perception_client = _MCPClient(_MCPResult(structured_content={"items": []}))

    MCPVLAProvider(client=vla_client, provider="remote-vla").close()
    MCPPerceptionProvider(client=perception_client, provider="remote-perception").close()

    assert vla_client.closed is True
    assert perception_client.closed is True


def test_streamable_http_client_owns_one_async_session_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeTransport:
        async def __aenter__(self) -> tuple[object, object, None]:
            events.append("transport-enter")
            return object(), object(), None

        async def __aexit__(self, *_args: object) -> None:
            events.append("transport-exit")

    class FakeSession:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeSession:
            events.append("session-enter")
            return self

        async def initialize(self) -> None:
            events.append("initialize")

        async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
            **_kwargs: object,
        ) -> object:
            events.append(f"call:{name}:{arguments['value']}")
            return {"ok": True}

        async def __aexit__(self, *_args: object) -> None:
            events.append("session-exit")

    monkeypatch.setattr(
        mcp_adapters,
        "_load_mcp_sdk",
        lambda: (FakeSession, lambda _endpoint: FakeTransport()),
    )

    client = mcp_adapters.StreamableHTTPMCPClient("http://mcp.test", default_timeout_s=1)
    assert client.call_tool("probe", {"value": 3}) == {"ok": True}
    client.close()
    client.close()

    assert events == [
        "transport-enter",
        "session-enter",
        "initialize",
        "call:probe:3",
        "session-exit",
        "transport-exit",
    ]


def test_executable_specs_exclude_environment_owned_finish() -> None:
    assert [spec.tool_id for spec in ROBOTICS_EXECUTABLE_SPECS] == [
        "vla_act",
        "segment",
        "detect_objects",
        "view_env_state",
        "view_camera_meta",
        "back_project",
        "set_gripper",
        "retreat",
        "move_to",
        "rotate_wrist",
    ]


def test_package_layout_has_no_domain_specific_executor_or_duplicate_specs_module() -> None:
    package = Path(__file__).parents[4] / "alphaapollo/common/execution/tools/robotics"

    for forbidden in ("executor.py", "specs.py", "finish.py"):
        assert not (package / forbidden).exists()
    for module_name in ("vla.py", "perception.py"):
        source = (package / module_name).read_text()
        assert "tools.robotic." not in source
        assert "FastMCP" not in source


def test_vla_act_converts_provider_failure_into_recoverable_response() -> None:
    handlers, provider, _, backend = _handlers()

    def boom(request: Any) -> Any:
        raise RuntimeError("VLA service unreachable")

    provider.predict = boom  # type: ignore[method-assign]
    response = _invoke(
        handlers,
        ToolRequest(
            call_id="vla-fail-1",
            tool_id="vla_act",
            arguments={"instruction": "pick up the cup"},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=12),
    )

    assert response.exit_code == 1
    assert "VLA service unreachable" in response.stderr
    assert response.stdout == ""
    assert backend.calls == []


def test_perception_converts_provider_failure_into_recoverable_response() -> None:
    handlers, _, perception, backend = _handlers()

    def boom(request: Any) -> Any:
        raise TimeoutError("segmentation service timed out")

    perception.inspect = boom  # type: ignore[method-assign]
    response = _invoke(
        handlers,
        ToolRequest(
            call_id="seg-fail-1",
            tool_id="segment",
            arguments={"query": "black bowl"},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=12),
    )

    assert response.exit_code == 1
    assert "segmentation service timed out" in response.stderr
    assert backend.calls == []


@dataclass
class _ServoBackend:
    """Pure-python 7-dim OSC fake: positions/yaw integrate normalized commands."""

    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.8])
    yaw: float = 0.0
    gripper_closed: bool = False
    calls: list[list[float]] = field(default_factory=list)

    terminated = False
    truncated = False
    success = None
    termination_reason = None

    @property
    def steps_used(self) -> int:
        return len(self.calls)

    def reset(self, task: RobotTask) -> RobotResetResult:
        return RobotResetResult(observation=self.observe())

    def observe(self) -> RobotObservation:
        import math as _math

        half = self.yaw / 2.0
        return RobotObservation(
            state={
                "proprio": {
                    "robot0_eef_pos": list(self.pos),
                    "robot0_eef_quat": [0.0, 0.0, _math.sin(half), _math.cos(half)],
                    "robot0_gripper_qpos": [0.0, 0.0] if self.gripper_closed else [0.04, -0.04],
                }
            },
            backend_metadata={
                "environment": {
                    "cameras": {
                        "fixed_camera": {
                            "name": "agentview",
                            "height": 4,
                            "width": 4,
                            "intrinsics": [[1, 0, 2], [0, 1, 2], [0, 0, 1]],
                            "extrinsics": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        }
                    }
                }
            },
        )

    def execute(self, action: RobotAction) -> RobotTransition:
        values = [float(v) for v in action.arguments["values"]]
        assert len(values) == 7
        self.calls.append(values)
        for axis in range(3):
            self.pos[axis] += 0.05 * values[axis]
        self.yaw += 0.5 * values[5]
        self.gripper_closed = values[6] > 0
        return RobotTransition(
            observation=self.observe(),
            steps_used=1,
            terminated=False,
            truncated=False,
            success=None,
        )

    def back_project(self, *, camera: str, pixel_x: int, pixel_y: int) -> RobotGroundedPoint:
        return RobotGroundedPoint(
            camera=camera,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            image_width=4,
            image_height=4,
            depth_m=0.5,
            world_xyz=(pixel_x - 2.0, pixel_y - 2.0, 0.5),
            steps_used=self.steps_used,
            observation_timestamp="2026-08-16T00:00:00Z",
        )

    def close(self) -> None:
        return None


def _servo_handlers() -> tuple[Mapping[str, ToolExecutor], _ServoBackend]:
    backend = _ServoBackend()
    handlers = build_robotics_tool_executors(
        vla_provider=_VLAProvider(),
        perception_provider=_PerceptionProvider(),
        backend=backend,
    )
    return handlers, backend


def test_view_env_state_reports_proprio_and_episode_accounting() -> None:
    handlers, backend = _servo_handlers()
    response = _invoke(
        handlers,
        ToolRequest(call_id="state-1", tool_id="view_env_state", arguments={}),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )
    payload = json.loads(response.stdout)

    assert response.exit_code == 0
    assert payload["proprio"]["robot0_eef_pos"] == [0.0, 0.0, 0.8]
    assert payload["steps_used"] == 0
    assert payload["terminated"] is False


def test_view_camera_meta_selects_and_rejects_cameras() -> None:
    handlers, _ = _servo_handlers()
    ok = _invoke(
        handlers,
        ToolRequest(
            call_id="cam-1", tool_id="view_camera_meta", arguments={"camera": "fixed_camera"}
        ),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )
    payload = json.loads(ok.stdout)
    assert payload["cameras"]["fixed_camera"]["name"] == "agentview"

    unknown = _invoke(
        handlers,
        ToolRequest(call_id="cam-2", tool_id="view_camera_meta", arguments={"camera": "nope"}),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )
    assert unknown.exit_code == 1
    assert "available" in unknown.stderr

    no_meta_handlers, _, _, _ = _handlers()
    missing = _invoke(
        no_meta_handlers,
        ToolRequest(call_id="cam-3", tool_id="view_camera_meta", arguments={}),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )
    assert missing.exit_code == 1
    assert "camera metadata" in missing.stderr


def test_back_project_returns_auditable_world_coordinates() -> None:
    handlers, _ = _servo_handlers()
    response = _invoke(
        handlers,
        ToolRequest(
            call_id="project-1",
            tool_id="back_project",
            arguments={"camera": "fixed_camera", "pixel": {"x": 1, "y": 2}},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )
    payload = json.loads(response.stdout)

    assert response.exit_code == 0
    assert payload["pixel"] == {"x": 1, "y": 2}
    assert payload["world_xyz"] == [-1.0, 0.0, 0.5]
    assert payload["depth_m"] == 0.5
    assert payload["frame"] == "world"


def test_back_project_fails_recoverably_without_geometry_capability() -> None:
    handlers, _, _, _ = _handlers()
    response = _invoke(
        handlers,
        ToolRequest(
            call_id="project-missing",
            tool_id="back_project",
            arguments={"pixel": {"x": 1, "y": 2}},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )

    assert response.exit_code == 1
    assert "live metric depth" in response.stderr


def test_segment_returns_world_xyz_verifiable_by_back_project() -> None:
    handlers, _ = _servo_handlers()
    segment = _invoke(
        handlers,
        ToolRequest(
            call_id="segment-grounded",
            tool_id="segment",
            arguments={"query": "red cup"},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )
    projected = _invoke(
        handlers,
        ToolRequest(
            call_id="segment-verify",
            tool_id="back_project",
            arguments={"pixel": {"x": 1, "y": 2}},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )

    item = json.loads(segment.stdout)["items"][0]
    point = json.loads(projected.stdout)
    assert item["centroid_pixel"] == {"x": 1, "y": 2}
    assert item["grounding_valid"] is True
    assert item["world_xyz"] == point["world_xyz"]
    assert item["depth_m"] == point["depth_m"]


def test_set_gripper_holds_the_command_and_reports_state() -> None:
    handlers, backend = _servo_handlers()
    response = _invoke(
        handlers,
        ToolRequest(call_id="grip-1", tool_id="set_gripper", arguments={"open": False}),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )
    payload = json.loads(response.stdout)

    assert payload["gripper"] == "closed"
    assert payload["steps_used"] == 8
    assert backend.gripper_closed is True
    assert all(values[6] == 1.0 for values in backend.calls)


def test_move_to_servos_to_the_target_within_tolerance() -> None:
    handlers, backend = _servo_handlers()
    response = _invoke(
        handlers,
        ToolRequest(
            call_id="move-1",
            tool_id="move_to",
            arguments={"x": 0.12, "y": -0.08, "z": 0.9},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )
    payload = json.loads(response.stdout)

    assert payload["reached"] is True
    assert payload["error"] <= 0.01
    assert abs(backend.pos[0] - 0.12) < 0.01
    assert abs(backend.pos[2] - 0.9) < 0.01
    assert payload["steps_used"] == len(backend.calls)


def test_move_to_respects_the_step_budget() -> None:
    handlers, _ = _servo_handlers()
    response = _invoke(
        handlers,
        ToolRequest(
            call_id="move-2",
            tool_id="move_to",
            arguments={"x": 2.0, "y": 0.0, "z": 0.8, "max_steps": 3},
        ),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )
    payload = json.loads(response.stdout)

    assert payload["reached"] is False
    assert payload["steps_used"] == 3


def test_retreat_lifts_straight_up() -> None:
    handlers, backend = _servo_handlers()
    response = _invoke(
        handlers,
        ToolRequest(call_id="ret-1", tool_id="retreat", arguments={"distance_m": 0.15}),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )
    payload = json.loads(response.stdout)

    assert payload["reached"] is True
    assert abs(backend.pos[2] - 0.95) < 0.011
    assert abs(backend.pos[0]) < 1e-9 and abs(backend.pos[1]) < 1e-9


def test_rotate_wrist_converges_on_relative_yaw() -> None:
    handlers, backend = _servo_handlers()
    response = _invoke(
        handlers,
        ToolRequest(call_id="rot-1", tool_id="rotate_wrist", arguments={"delta_yaw_rad": 0.6}),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )
    payload = json.loads(response.stdout)

    assert payload["reached"] is True
    assert abs(backend.yaw - 0.6) <= 0.02


def test_primitives_fail_recoverably_without_proprioception() -> None:
    handlers, _, _, _ = _handlers()
    response = _invoke(
        handlers,
        ToolRequest(call_id="move-3", tool_id="move_to", arguments={"x": 0.1, "y": 0.0, "z": 0.8}),
        ExecutionContext(session_id="episode-1", timeout_s=5),
    )

    assert response.exit_code == 1
    assert "robot0_eef_pos" in response.stderr


def _live_client_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "alphaapollo-mcp-client"]


def _wait_until(predicate: Any, *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_streamable_http_client_retires_its_thread_when_the_transport_hangs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that accepts the connection and never speaks must leave nothing behind.

    One client is built per provider per episode, so a session that cannot be
    retired costs an event loop, a selector, and an open socket for every
    episode that reaches an unresponsive endpoint. The constructor's own
    cleanup must finish before it raises, not eventually.
    """

    monkeypatch.setattr(mcp_adapters, "_STARTUP_TIMEOUT_S", 0.5)
    monkeypatch.setattr(mcp_adapters, "_SHUTDOWN_TIMEOUT_S", 0.5)
    loops: list[asyncio.AbstractEventLoop] = []
    new_event_loop = asyncio.new_event_loop

    def record_loop() -> asyncio.AbstractEventLoop:
        loop = new_event_loop()
        loops.append(loop)
        return loop

    monkeypatch.setattr(asyncio, "new_event_loop", record_loop)

    class HangingTransport:
        async def __aenter__(self) -> tuple[object, object, None]:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def __aexit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        mcp_adapters,
        "_load_mcp_sdk",
        lambda: (object, lambda _endpoint: HangingTransport()),
    )

    with pytest.raises(mcp_adapters.MCPClientError, match="timed out while initializing"):
        mcp_adapters.StreamableHTTPMCPClient("http://mcp.test", default_timeout_s=1)

    assert not _live_client_threads(), "the constructor returned before its loop thread died"
    assert loops and all(loop.is_closed() for loop in loops), "the event loop was never closed"


def test_streamable_http_client_reports_why_a_started_session_ended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session that dies after the handshake must not read as "never started"."""

    monkeypatch.setattr(mcp_adapters, "_SHUTDOWN_TIMEOUT_S", 0.5)

    class FakeTransport:
        async def __aenter__(self) -> tuple[object, object, None]:
            return object(), object(), None

        async def __aexit__(self, *_args: object) -> None:
            return None

    class FakeSession:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeSession:
            return self

        async def initialize(self) -> None:
            return None

        async def call_tool(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("the session is gone")

        async def __aexit__(self, *_args: object) -> None:
            raise RuntimeError("transport dropped mid-episode")

    monkeypatch.setattr(
        mcp_adapters,
        "_load_mcp_sdk",
        lambda: (FakeSession, lambda _endpoint: FakeTransport()),
    )

    client = mcp_adapters.StreamableHTTPMCPClient("http://mcp.test", default_timeout_s=1)
    try:
        loop = client._loop
        stop_event = client._stop_event
        assert loop is not None and stop_event is not None
        loop.call_soon_threadsafe(stop_event.set)
        assert _wait_until(lambda: client._session is None)

        with pytest.raises(mcp_adapters.MCPClientError, match="transport dropped mid-episode"):
            client.call_tool("probe", {})
    finally:
        with pytest.raises(RuntimeError, match="transport dropped mid-episode"):
            client.close()

    assert _wait_until(lambda: not _live_client_threads())
