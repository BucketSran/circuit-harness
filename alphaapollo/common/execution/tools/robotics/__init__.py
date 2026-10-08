"""Semantic robotics tools built on AlphaApollo's shared tool contracts."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from alphaapollo.common.execution.robotics import RobotBackend
from alphaapollo.common.execution.tools.base import (
    ExecutionContext,
    ToolExecutor,
    ToolRequest,
    ToolResponse,
    ToolSpec,
)
from alphaapollo.common.execution.tools.robotics.inspection import (
    BACK_PROJECT_SPEC,
    INSPECTION_TOOL_SOURCE,
    VIEW_CAMERA_META_SPEC,
    VIEW_ENV_STATE_SPEC,
    InspectionTools,
)
from alphaapollo.common.execution.tools.robotics.mcp import (
    MCPClientError,
    MCPPerceptionProvider,
    MCPProviderError,
    MCPToolClient,
    MCPVLAProvider,
    StreamableHTTPMCPClient,
)
from alphaapollo.common.execution.tools.robotics.perception import (
    DETECT_OBJECTS_SPEC,
    SEGMENT_SPEC,
    PerceptionProvider,
    PerceptionRequest,
    PerceptionResult,
    PerceptionTools,
)
from alphaapollo.common.execution.tools.robotics.primitives import (
    MOVE_TO_SPEC,
    PRIMITIVE_TOOL_SOURCE,
    RETREAT_SPEC,
    ROTATE_WRIST_SPEC,
    SET_GRIPPER_SPEC,
    PrimitiveFault,
    PrimitiveTools,
)
from alphaapollo.common.execution.tools.robotics.vla import (
    VLA_ACT_SPEC,
    VLAActTool,
    VLAProvider,
    VLARequest,
    VLAResult,
)

FINISH_TOOL_SOURCE = "AlphaApollo robotics environment control tool"
FINISH_SPEC = ToolSpec(
    tool_id="finish",
    description=(
        "Request an intentional stop. The requested status is advisory and cannot "
        "override environment-derived success."
    ),
    parameters={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "failure", "stuck"],
                "description": "The agent's claimed terminal status.",
            },
            "summary": {
                "type": "string",
                "description": "A brief audit of the final state and attempted work.",
            },
        },
        "required": ["status", "summary"],
        "additionalProperties": False,
    },
    source=FINISH_TOOL_SOURCE,
    adaptations=("Handled by RobotEnvironment, not by a model provider.",),
)

ROBOTICS_EXECUTABLE_SPECS = (
    VLA_ACT_SPEC,
    SEGMENT_SPEC,
    DETECT_OBJECTS_SPEC,
    VIEW_ENV_STATE_SPEC,
    VIEW_CAMERA_META_SPEC,
    BACK_PROJECT_SPEC,
    SET_GRIPPER_SPEC,
    RETREAT_SPEC,
    MOVE_TO_SPEC,
    ROTATE_WRIST_SPEC,
)
ROBOTICS_TOOL_SPECS = (FINISH_SPEC, *ROBOTICS_EXECUTABLE_SPECS)


class RoboticsToolExecutor:
    """Dispatch semantic tool requests to one episode's provider-bound handlers."""

    def __init__(self, executors: Mapping[str, ToolExecutor]) -> None:
        if not isinstance(executors, Mapping):
            raise TypeError("executors must be a mapping")
        materialized = dict(executors)
        expected = {spec.tool_id for spec in ROBOTICS_EXECUTABLE_SPECS}
        if set(materialized) != expected:
            raise ValueError(
                "robotics executors must match executable specs: "
                f"missing={sorted(expected - set(materialized))}, "
                f"unexpected={sorted(set(materialized) - expected)}"
            )
        if any(not isinstance(executor, ToolExecutor) for executor in materialized.values()):
            raise TypeError("robotics executor values must implement ToolExecutor")
        self._executors = MappingProxyType(materialized)
        self._closed = False

    def execute(self, request: ToolRequest, context: ExecutionContext) -> ToolResponse:
        if self._closed:
            raise RuntimeError("robotics tool executor is closed")
        executor = self._executors.get(request.tool_id)
        if executor is None:
            raise ValueError(f"no robotics executor for {request.tool_id!r}")
        return executor.execute(request, context)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        closed: set[int] = set()
        first_error: Exception | None = None
        for executor in self._executors.values():
            identity = id(executor)
            if identity in closed:
                continue
            closed.add(identity)
            close = getattr(executor, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - close every provider-bound handler
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def build_robotics_tool_executors(
    *,
    vla_provider: VLAProvider,
    perception_provider: PerceptionProvider,
    backend: RobotBackend,
) -> Mapping[str, ToolExecutor]:
    """Build handlers for registration with the shared execution session."""

    vla_tool = VLAActTool(provider=vla_provider, backend=backend)
    perception_tools = PerceptionTools(provider=perception_provider, backend=backend)
    inspection_tools = InspectionTools(backend=backend)
    primitive_tools = PrimitiveTools(backend=backend)
    return MappingProxyType(
        {
            VLA_ACT_SPEC.tool_id: vla_tool,
            SEGMENT_SPEC.tool_id: perception_tools,
            DETECT_OBJECTS_SPEC.tool_id: perception_tools,
            VIEW_ENV_STATE_SPEC.tool_id: inspection_tools,
            VIEW_CAMERA_META_SPEC.tool_id: inspection_tools,
            BACK_PROJECT_SPEC.tool_id: inspection_tools,
            SET_GRIPPER_SPEC.tool_id: primitive_tools,
            RETREAT_SPEC.tool_id: primitive_tools,
            MOVE_TO_SPEC.tool_id: primitive_tools,
            ROTATE_WRIST_SPEC.tool_id: primitive_tools,
        }
    )


__all__ = [
    "BACK_PROJECT_SPEC",
    "DETECT_OBJECTS_SPEC",
    "FINISH_SPEC",
    "FINISH_TOOL_SOURCE",
    "INSPECTION_TOOL_SOURCE",
    "InspectionTools",
    "MCPClientError",
    "MCPPerceptionProvider",
    "MCPProviderError",
    "MCPToolClient",
    "MCPVLAProvider",
    "MOVE_TO_SPEC",
    "PRIMITIVE_TOOL_SOURCE",
    "PerceptionProvider",
    "PerceptionRequest",
    "PerceptionResult",
    "PerceptionTools",
    "PrimitiveFault",
    "PrimitiveTools",
    "RETREAT_SPEC",
    "ROBOTICS_EXECUTABLE_SPECS",
    "ROBOTICS_TOOL_SPECS",
    "ROTATE_WRIST_SPEC",
    "RoboticsToolExecutor",
    "SEGMENT_SPEC",
    "SET_GRIPPER_SPEC",
    "StreamableHTTPMCPClient",
    "VIEW_CAMERA_META_SPEC",
    "VIEW_ENV_STATE_SPEC",
    "VLAActTool",
    "VLAProvider",
    "VLARequest",
    "VLAResult",
    "VLA_ACT_SPEC",
    "build_robotics_tool_executors",
]
