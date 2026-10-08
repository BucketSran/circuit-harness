"""Public task sessions projected onto Apollo's Environment contract.

The injected transport owns durable action IDs and evidence. The existing task
session owns paths, simulator budgets and freezing; final grading remains an
operator action. Closing this adapter never cancels an accepted server job.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from alphaapollo.common.environment.base import (
    BaseEnvironment,
    EnvironmentContext,
    EnvironmentInitResult,
    EnvironmentTransition,
)
from alphaapollo.common.environment.default.projection import normalize
from alphaapollo.common.execution.tools.base import ToolError
from alphaapollo.common.execution.tools.chips import SESSION_TOOL_SPECS
from alphaapollo.common.execution.tools.registry import ToolCatalog
from alphaapollo.common.generation.base import GenerationResponse


class ChipsEnvironment(BaseEnvironment):
    def __init__(self, transport: Any, *, task_kind: str):
        if task_kind not in ("vabench", "analog"):
            raise ValueError("task_kind must be vabench or analog")
        if not callable(getattr(transport, "call", None)):
            raise TypeError("transport must expose call()")
        self._transport = transport
        self._submit_tool = f"{task_kind}_submit"
        self._status_field = "status" if task_kind == "vabench" else "state"
        self._catalog = ToolCatalog(
            spec for spec in SESSION_TOOL_SPECS if spec.tool_id.startswith(task_kind + "_")
        )
        self._initialized = False
        self._closed = False
        self._termination_reason: str | None = None

    def init(self, context: EnvironmentContext) -> EnvironmentInitResult:
        if self._initialized or self._closed:
            raise RuntimeError("use a fresh Environment for each episode")
        self._initialized = True
        return EnvironmentInitResult(
            observation=context.user_prompt,
            metadata={"evaluation": "not_performed"},
        )

    def step(self, action: Any) -> EnvironmentTransition:
        if not self._initialized or self._closed or self._termination_reason:
            raise RuntimeError("Environment is not active")
        request = None if isinstance(action, str) else normalize(action)
        if request is None:
            self._termination_reason = "final_response_without_submission"
            return EnvironmentTransition(
                observation=str(action),
                reward=0,
                done=True,
                termination_reason=self._termination_reason,
                metadata={"evaluation": "not_performed"},
            )
        if not isinstance(request, ToolError):
            resolved = self._catalog.resolve(request)
            if isinstance(resolved, ToolError):
                request = resolved
        metadata: dict[str, Any] = {"evaluation": "not_performed"}
        if isinstance(request, ToolError):
            if request.tool_id is not None:
                metadata["tool_request"] = {"tool_id": request.tool_id, "call_id": request.call_id}
            payload = {"ok": False, "error": asdict(request)}
        else:
            # Provider IDs may contain characters forbidden by the server's
            # filename contract. Keep their exact spelling in trace metadata.
            action_id = hashlib.sha256(request.call_id.encode()).hexdigest()
            metadata["tool_request"] = asdict(request)
            metadata["action_id"] = action_id
            payload = self._transport.call(request.tool_id, request.arguments, action_id=action_id)
            if (
                payload.get("server_execution") == "unknown"
                or payload.get("error") == "recover_previous_action"
            ):
                self._termination_reason = "awaiting_action_recovery"
            if (
                request.tool_id == self._submit_tool
                and payload.get("ok") is True
                and payload.get("result", {}).get(self._status_field) == "submitted"
            ):
                self._termination_reason = "submitted"
        # The shared MCP bridge decodes this established envelope back to the
        # original JSON reply, retaining each task's distinct diagnostic fields.
        return EnvironmentTransition(
            observation="<tool_response>\n" + json.dumps(payload) + "\n</tool_response>",
            raw_observation=payload,
            reward=0,
            done=self._termination_reason is not None,
            termination_reason=self._termination_reason,
            response_format_valid=not isinstance(request, ToolError) or request.stage != "parse",
            env_action_valid=payload.get("ok") is True,
            metadata=metadata,
        )

    def project_response(self, response: GenerationResponse) -> Any:
        if not response.tool_calls:
            return response.content
        message = {
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
        if response.reasoning_content is not None:
            message["reasoning_content"] = response.reasoning_content
        return message

    def continuation_messages(
        self, response: GenerationResponse, transition: EnvironmentTransition
    ):
        if transition.done:
            return ()
        # normalize() rejects a multi-call response atomically. Return that
        # rejection for every call ID so the next request remains protocol-valid.
        return (
            self.project_response(response),
            *(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(transition.raw_observation),
                }
                for call in response.tool_calls
            ),
        )

    def terminate(self, reason: str = "runtime_requested") -> None:
        super().terminate(reason)
        if self._termination_reason is None:
            self._termination_reason = reason

    def close(self) -> None:
        self.interrupt_wait()
        self._closed = True

    def interrupt_wait(self) -> None:
        """Release a client query while leaving server action ownership intact."""
        interrupt = getattr(self._transport, "interrupt_wait", None)
        if callable(interrupt):
            interrupt()
