# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate and build generic Environments and adapt generation responses.

The composition entry point supplies domain-specific task adaptation and tool
catalogs. These builders do not import that entry point or construct robotics
resources.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from alphaapollo.workflows.config import (
    ConfigError,
    ResourceConfig,
    _closed_options,
    _non_empty_string,
    _positive_int,
)
from alphaapollo.workflows.data import (
    _generation_provenance_record,
    _json_value,
    _wire_generation_tool_call,
)

if TYPE_CHECKING:
    from alphaapollo.common.execution.tools import ToolCatalog


@dataclass(frozen=True, slots=True)
class _TextInit:
    observation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _TextTransition:
    observation: str
    reward: float
    done: bool
    success: bool | None
    response_format_valid: bool
    env_action_valid: bool
    termination_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class _TextOnlyEnvironment:
    """Portable no-tool Environment used by the offline composition test path."""

    def __init__(self) -> None:
        self._closed = False

    def init(self, _context: object) -> _TextInit:
        if self._closed:
            raise RuntimeError("text-only Environment is closed")
        return _TextInit()

    def step(self, _action: object) -> _TextTransition:
        if self._closed:
            raise RuntimeError("text-only Environment is closed")
        return _TextTransition(
            observation="",
            reward=0.0,
            done=True,
            success=None,
            response_format_valid=True,
            env_action_valid=True,
            termination_reason="model_output",
        )

    def project_response(self, response: object) -> str:
        """Project a text-only Generation response without teaching Runtime its shape."""

        content = getattr(response, "content", None)
        if not isinstance(content, str):
            raise TypeError("Generation response content must be a string")
        return content

    def continuation_messages(
        self, _response: object, transition: object
    ) -> tuple[Mapping[str, Any], ...]:
        if not bool(getattr(transition, "done", False)):
            raise RuntimeError("text-only Environment returned a non-terminal transition")
        return ()

    def terminate(self, _reason: str = "workflow_shutdown") -> None:
        return None

    def close(self) -> None:
        self._closed = True


class _OwnedEnvironment:
    """Make Runtime-owned Environment cleanup observable and idempotent."""

    def __init__(self, environment: object) -> None:
        self._environment = environment
        self._closed = False

    def init(self, context: object) -> object:
        return self._environment.init(context)  # type: ignore[attr-defined, no-any-return]

    def step(self, action: object) -> object:
        return self._environment.step(action)  # type: ignore[attr-defined, no-any-return]

    def project_response(self, response: object) -> object:
        projector = getattr(self._environment, "project_response", None)
        if callable(projector):
            return projector(response)
        return _provider_response_action(response)

    def continuation_messages(
        self, response: object, transition: object
    ) -> tuple[Mapping[str, Any], ...]:
        continuation = getattr(self._environment, "continuation_messages", None)
        if callable(continuation):
            messages = continuation(response, transition)
            return _validate_continuation_messages(messages)
        return _default_continuation_messages(response, transition)

    def terminate(self, reason: str = "workflow_shutdown") -> None:
        if self._closed:
            return
        terminate = getattr(self._environment, "terminate", None)
        if callable(terminate):
            terminate(reason)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._environment, "close", None)
        if callable(close):
            close()


def _provider_response_action(response: object) -> dict[str, Any]:
    """Losslessly project #185's response into #175's provider-response ingress."""

    message = _generation_response_message(response)
    action: dict[str, Any] = {
        "id": getattr(response, "request_id", None),
        "choices": [
            {
                "message": message,
                "finish_reason": getattr(response, "finish_reason", None),
            }
        ],
        "usage": dict(getattr(response, "usage", {}) or {}),
        "backend_metadata": dict(getattr(response, "backend_metadata", {}) or {}),
    }
    for name in (
        "prompt_token_ids",
        "response_token_ids",
        "response_logprobs",
        "provenance",
    ):
        value = getattr(response, name, None)
        if value is not None:
            action[name] = (
                _generation_provenance_record(value) if name == "provenance" else _json_value(value)
            )
    return action


def _generation_response_message(response: object) -> dict[str, Any]:
    content = getattr(response, "content", None)
    if not isinstance(content, str):
        raise TypeError("Generation response content must be a string")
    message: dict[str, Any] = {"role": "assistant", "content": content}
    reasoning = getattr(response, "reasoning_content", None)
    if isinstance(reasoning, str):
        message["reasoning_content"] = reasoning
    raw_calls = getattr(response, "tool_calls", ()) or ()
    if isinstance(raw_calls, (str, bytes)) or not isinstance(raw_calls, Sequence):
        raise TypeError("Generation response tool_calls must be a sequence")
    if raw_calls:
        message["tool_calls"] = [_wire_generation_tool_call(call) for call in raw_calls]
    return message


def _default_continuation_messages(
    response: object, transition: object
) -> tuple[Mapping[str, Any], ...]:
    if bool(getattr(transition, "done", False)):
        return ()
    observation = str(getattr(transition, "observation", "") or "")
    assistant = _generation_response_message(response)
    calls = assistant.get("tool_calls", [])
    metadata = getattr(transition, "metadata", {}) or {}
    request = metadata.get("tool_request") if isinstance(metadata, Mapping) else None
    call_id = request.get("call_id") if isinstance(request, Mapping) else None
    matching = [call for call in calls if call.get("id") == call_id]
    if len(matching) == 1 and isinstance(call_id, str) and call_id:
        assistant["tool_calls"] = matching
        return (
            assistant,
            {"role": "tool", "tool_call_id": call_id, "content": observation},
        )
    assistant.pop("tool_calls", None)
    return (assistant, {"role": "user", "content": observation})


def _validate_continuation_messages(value: object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("Environment continuation_messages must return a sequence")
    messages: list[Mapping[str, Any]] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise TypeError("Environment continuation messages must be mappings")
        messages.append(dict(message))
    return tuple(messages)


def _validate_environment_resource(resource: ResourceConfig) -> None:
    if resource.type in {"text_only", "none"}:
        _closed_options(
            resource.options,
            allowed=set(),
            required=set(),
            where="environment.options",
        )
        return
    if resource.type != "default":
        raise ConfigError(
            f"environment.type must be 'default', 'text_only', or 'none', got {resource.type!r}"
        )
    options = _closed_options(
        resource.options,
        allowed={
            "max_steps",
            "max_tool_calls",
            "tool_timeout_s",
            "execution_mode",
            "feedback_mode",
            "enable_python_code",
        },
        required=set(),
        where="environment.options",
    )
    max_steps = options.get("max_steps")
    if max_steps is not None:
        _positive_int(max_steps, "environment.options.max_steps")
    max_tool_calls = options.get("max_tool_calls")
    if max_tool_calls is not None:
        _positive_int(max_tool_calls, "environment.options.max_tool_calls")
    timeout = options.get("tool_timeout_s")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ConfigError("environment.options.tool_timeout_s must be finite and positive")
    _non_empty_string(
        options.get("execution_mode", "default"), "environment.options.execution_mode"
    )
    feedback_mode = options.get("feedback_mode", "differentiated")
    _non_empty_string(feedback_mode, "environment.options.feedback_mode")
    if feedback_mode not in {"differentiated", "generic", "missing"}:
        raise ConfigError(
            "environment.options.feedback_mode must be differentiated, generic, or missing"
        )
    enable_python_code = options.get("enable_python_code", False)
    if not isinstance(enable_python_code, bool):
        raise ConfigError("environment.options.enable_python_code must be a bool")


def _build_environment(environment_type: str, options: Mapping[str, Any], task: object) -> object:
    if environment_type == "none":
        raise ConfigError(
            "environment.type 'none' was selected, but a runtime asked for an Environment"
        )
    if environment_type == "text_only":
        environment: object = _TextOnlyEnvironment()
    else:
        from alphaapollo.common.environment.default import (
            DefaultEnvironment,
            TextOnlyToolBridge,
        )
        from alphaapollo.common.execution.tools import INTERNAL_PYTHON_TOOL_ID

        allowed_tools = list(getattr(task, "tools", ()) or ())
        if options.get("enable_python_code", False):
            allowed_tools.append(INTERNAL_PYTHON_TOOL_ID)
        allowed_tools = list(dict.fromkeys(allowed_tools))
        if allowed_tools:
            from alphaapollo.common.environment.default import GatewayToolBridge
            from alphaapollo.common.execution import ExecutionRuntime, ToolGateway
            from alphaapollo.common.execution.tools import ToolCatalog, list_tool_specs
            from alphaapollo.common.execution.tools.python import (
                PYTHON_EXECUTE_SPEC,
                PYTHON_EXECUTE_TOOL_ID,
                PythonExecuteTool,
            )

            if PYTHON_EXECUTE_TOOL_ID in allowed_tools:
                runtime = ExecutionRuntime(
                    catalog=ToolCatalog(
                        (*list_tool_specs(include_internal=True), PYTHON_EXECUTE_SPEC)
                    ),
                    additional_tools=(PythonExecuteTool(),),
                )
            else:
                runtime = ExecutionRuntime()

            tool_bridge = GatewayToolBridge(
                ToolGateway(runtime=runtime),
                max_tool_calls=options.get("max_tool_calls"),
                allowed_tool_ids=allowed_tools,
            )
        else:
            tool_bridge = TextOnlyToolBridge()

        # ``DefaultEnvironment.grader_id`` is deliberately not wired here.
        # It makes the Environment produce reward from the gold answer, which
        # puts gold inside the process running the model -- a training
        # affordance, and the opposite of what this package guarantees. So
        # ``_validate_environment_resource`` does not list ``grader_id`` among
        # the accepted ``environment.options`` at all, with or without a
        # ``scoring`` block, and passing it through here would be reading a key
        # that composition has already refused. Granting it later is a
        # deliberate change that has to answer the leak question first.
        environment = DefaultEnvironment(
            tool_bridge=tool_bridge,
            max_steps=options.get("max_steps"),
            tool_timeout_s=options.get("tool_timeout_s"),
            execution_mode=options.get("execution_mode", "default"),
            feedback_mode=options.get("feedback_mode", "differentiated"),
        )
    return environment


def _tool_schemas(
    catalog: ToolCatalog, tool_ids: Sequence[str], *, runtime_name: str
) -> tuple[Mapping[str, Any], ...]:
    """Resolve ordered grants against the catalog selected by composition."""
    schemas: list[Mapping[str, Any]] = []
    for tool_id in tool_ids:
        try:
            spec = catalog.get(tool_id)
        except KeyError as exc:
            raise ConfigError(
                f"runtime {runtime_name!r} references unknown tool {tool_id!r}"
            ) from exc
        if not spec.model_visible:
            raise ConfigError(
                f"runtime {runtime_name!r} tool {tool_id!r} is not a native model tool"
            )
        schemas.append(spec.to_openai_tool())
    return tuple(schemas)
