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


"""Validate and construct configured Runtimes and their generation backends.

Native and external Runtimes share role tool-grant resolution. External session
configuration binds those grants to the existing bridge; generation backends
remain owned by the resource composition that creates them.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from alphaapollo.reasoning.runtime import AlphaApolloAgentRuntime
from alphaapollo.workflows.config import (
    ConfigError,
    ResourceConfig,
    RoleConfig,
    RunConfig,
    WorkflowConfig,
    _closed_options,
    _non_empty_string,
    _positive_int,
    _thaw_json,
)

from .lifecycle import _shutdown_resources, _terminate_resource

if TYPE_CHECKING:
    from .verifier import _AgentVerifierSpec


@dataclass(frozen=True, slots=True)
class _FakeResponse:
    request_id: str
    group_id: str
    sample_id: int
    content: str
    usage: Mapping[str, int]
    backend_metadata: Mapping[str, Any]
    finish_reason: str = "stop"
    reasoning_content: str | None = None
    tool_calls: tuple[object, ...] = ()


class _FakeBackend:
    """Deterministic GenerationBackend with verifier-aware structured output."""

    def __init__(
        self,
        *,
        verifier_formats: Mapping[str, str],
        text: str | None,
        verdict: str,
        feedback: str,
    ) -> None:
        self._verifier_formats = dict(verifier_formats)
        self._text = text
        self._verdict = verdict
        self._feedback = feedback
        self._closed = False

    def generate_batch(self, requests: Sequence[object]) -> list[_FakeResponse]:
        if self._closed:
            raise RuntimeError("fake backend is closed")
        materialized = _validate_generation_requests(requests)
        responses = [self._generate_one(request) for request in materialized]
        _validate_generation_response_identity(materialized, responses, backend="fake")
        return responses

    def _generate_one(self, request: object) -> _FakeResponse:
        request_id, group_id, sample_id = _generation_identity(request)
        messages = getattr(request, "messages", ())
        system = _message_content(messages, "system")
        prompt = _last_message_content(messages, "user")
        output_format = self._verifier_formats.get(system)
        if output_format == "json":
            content = json.dumps(
                {
                    "verdict": self._verdict,
                    "feedback": self._feedback,
                    "details": {"backend": "fake"},
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        elif output_format == "verdict-line":
            content = f"VERDICT: {self._verdict.upper()}\nFEEDBACK: {self._feedback}"
        else:
            content = self._text if self._text is not None else f"fake response: {prompt}"
        prompt_tokens = sum(len(str(message.get("content", ""))) for message in messages) // 4
        completion_tokens = max(1, len(content) // 4)
        return _FakeResponse(
            request_id=request_id,
            group_id=group_id,
            sample_id=sample_id,
            content=content,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            backend_metadata={"backend": "fake"},
        )

    def terminate(self, _reason: str = "workflow_shutdown") -> None:
        return None

    def close(self) -> None:
        self._closed = True


class _OwnedGenerationBackend:
    """Own a canonical GenerationBackend and its transport client lifecycle."""

    def __init__(self, backend: object) -> None:
        self._backend = backend
        self._closed = False

    @property
    def capabilities(self) -> object:
        return self._backend.capabilities  # type: ignore[attr-defined]

    def generate_batch(self, requests: Sequence[object]) -> object:
        if self._closed:
            raise RuntimeError("OpenAI-compatible backend is closed")
        return self._backend.generate_batch(requests)  # type: ignore[attr-defined]

    def terminate(self, reason: str = "workflow_shutdown") -> None:
        terminate = getattr(self._backend, "terminate", None)
        if callable(terminate):
            _terminate_resource(terminate, reason=reason)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()
            return
        client = getattr(self._backend, "_client", None)
        client_close = getattr(client, "close", None)
        if callable(client_close):
            client_close()


def _validate_generation_requests(requests: Sequence[object]) -> tuple[object, ...]:
    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
        raise TypeError("generation requests must be a sequence")
    materialized = tuple(requests)
    request_ids: set[str] = set()
    sample_ids: set[tuple[str, int]] = set()
    for request in materialized:
        request_id, group_id, sample_id = _generation_identity(request)
        if request_id in request_ids:
            raise RuntimeError(f"duplicate generation request_id {request_id!r}")
        sample_identity = (group_id, sample_id)
        if group_id and sample_identity in sample_ids:
            raise RuntimeError(
                f"duplicate generation sample identity ({group_id!r}, {sample_id!r})"
            )
        request_ids.add(request_id)
        if group_id:
            sample_ids.add(sample_identity)
    return materialized


def _generation_identity(value: object) -> tuple[str, str, int]:
    request_id = getattr(value, "request_id", None)
    group_id = getattr(value, "group_id", None)
    sample_id = getattr(value, "sample_id", None)
    if not isinstance(request_id, str) or not request_id.strip():
        raise RuntimeError("generation request_id must be a non-empty string")
    if not isinstance(group_id, str):
        raise RuntimeError("generation group_id must be a string")
    if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0:
        raise RuntimeError("generation sample_id must be a non-negative int")
    return request_id, group_id, sample_id


def _validate_generation_response_identity(
    requests: Sequence[object],
    responses: Sequence[object],
    *,
    backend: str,
) -> None:
    if len(responses) != len(requests):
        raise RuntimeError(
            f"{backend} backend returned {len(responses)} responses for {len(requests)} requests"
        )
    for index, (request, response) in enumerate(zip(requests, responses, strict=True)):
        expected = _generation_identity(request)
        actual = _generation_identity(response)
        if actual != expected:
            raise RuntimeError(
                f"{backend} backend changed generation identity at index {index}: "
                f"expected {expected!r}, got {actual!r}"
            )


def _validate_runtime_resource(name: str, resource: ResourceConfig) -> None:
    where = f"runtime {name!r}"
    if resource.type == "external":
        _validate_external_runtime_resource(resource, where=where)
        return
    if resource.type != "alphaapollo":
        raise ConfigError(
            f"{where}.type must be 'alphaapollo' or 'external', got {resource.type!r}"
        )
    options = _closed_options(
        resource.options,
        allowed={
            "backend",
            "model",
            "sampling",
            "max_turns",
            "tools",
            "image_history_messages",
        },
        required={"backend", "model"},
        where=f"{where}.options",
    )
    _non_empty_string(options["model"], f"{where}.options.model")
    max_turns = options.get("max_turns", 6)
    _positive_int(max_turns, f"{where}.options.max_turns")
    image_history_messages = options.get("image_history_messages")
    if image_history_messages is not None:
        _positive_int(image_history_messages, f"{where}.options.image_history_messages")
    _configured_runtime_tools(resource, where=where)
    sampling = _closed_options(
        options.get("sampling", {}),
        allowed={
            "temperature",
            "max_tokens",
            "top_p",
            "seed",
            "tool_choice",
            "provider_options",
        },
        required=set(),
        where=f"{where}.options.sampling",
    )
    temperature = sampling.get("temperature", 0.6)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ConfigError(f"{where}.options.sampling.temperature must be finite and non-negative")
    top_p = sampling.get("top_p", 1.0)
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(top_p)
        or not 0.0 < top_p <= 1.0
    ):
        raise ConfigError(f"{where}.options.sampling.top_p must be in (0, 1]")
    _positive_int(sampling.get("max_tokens", 4096), f"{where}.options.sampling.max_tokens")
    seed = sampling.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ConfigError(f"{where}.options.sampling.seed must be an int or null")
    tool_choice = sampling.get("tool_choice")
    if tool_choice is not None:
        _non_empty_string(tool_choice, f"{where}.options.sampling.tool_choice")
    provider_options = sampling.get("provider_options", {})
    if not isinstance(provider_options, Mapping):
        raise ConfigError(f"{where}.options.sampling.provider_options must be a mapping")
    _validate_backend_options(options["backend"], where=f"{where}.options.backend")


def _validate_external_runtime_resource(resource: ResourceConfig, *, where: str) -> None:
    """Validate one external agent runtime without importing its session module."""

    options = _closed_options(
        resource.options,
        allowed={
            "agent",
            "model",
            "concurrency",
            "workspace_root",
            "keep_workspaces",
            "reuse_workspace",
            "reuse_session",
            "session",
            "tools",
            "environment_mode",
        },
        required={"agent", "model"},
        where=f"{where}.options",
    )
    agent = _non_empty_string(options["agent"], f"{where}.options.agent")
    if agent not in _external_agents():
        raise ConfigError(
            f"{where}.options.agent must be one of {sorted(_external_agents())}, got {agent!r}"
        )
    _non_empty_string(options["model"], f"{where}.options.model")
    if options.get("environment_mode", "tool_loop") not in ("tool_loop", "final_response"):
        raise ConfigError(
            f"{where}.options.environment_mode must be 'tool_loop' or 'final_response'"
        )
    _positive_int(options.get("concurrency", 1), f"{where}.options.concurrency")
    workspace_root = options.get("workspace_root")
    if workspace_root is not None:
        _non_empty_string(workspace_root, f"{where}.options.workspace_root")
    if not isinstance(options.get("keep_workspaces", False), bool):
        raise ConfigError(f"{where}.options.keep_workspaces must be a bool")
    if not isinstance(options.get("reuse_workspace", False), bool):
        raise ConfigError(f"{where}.options.reuse_workspace must be a bool")
    if not isinstance(options.get("reuse_session", False), bool):
        raise ConfigError(f"{where}.options.reuse_session must be a bool")
    if options.get("reuse_session", False) and not options.get("reuse_workspace", False):
        raise ConfigError(f"{where}.options.reuse_session requires reuse_workspace")
    if options.get("reuse_session", False) and agent != "codex":
        raise ConfigError(
            f"{where}.options.reuse_session is currently verified only for agent 'codex'"
        )
    if not isinstance(options.get("session", {}), Mapping):
        raise ConfigError(f"{where}.options.session must be a mapping")
    _configured_runtime_tools(resource, where=where)


def _external_agents() -> tuple[str, ...]:
    from alphaapollo.reasoning.runtime.external import available_agents

    return available_agents()


def _probe_external_session(agent: str, options: Mapping[str, Any], *, where: str) -> None:
    from alphaapollo.reasoning.runtime.external import session_factory

    try:
        session = session_factory(agent, _thaw_json(options))()
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} cannot configure the {agent!r} session: {exc}") from exc
    _shutdown_resources((session,))


def _external_session_options(
    resource: ResourceConfig,
    tool_ids: Sequence[str],
    *,
    where: str,
    environment_backed: bool = False,
) -> dict[str, Any]:
    """Turn AlphaApollo tool grants into the MCP bridge that serves them.

    ``tools`` means the same thing for an external runtime as for a native one:
    these AlphaApollo tools, through the same catalog, policy, and sandbox. The
    difference is only how they reach the model, so the config stays symmetric
    and the bridge wiring is not something a recipe has to spell out.
    """

    from alphaapollo.reasoning.runtime.external.bridge.mcp_config import (
        BRIDGE_SERVER_NAME,
        alphaapollo_bridge,
    )

    options = _thaw_json(resource.options)
    session = dict(options.get("session", {}))
    if options.get("environment_mode") == "final_response":
        agent = options["agent"]
        if not environment_backed:
            raise ConfigError(f"{where} final_response mode requires an Environment")
        if agent not in {"codex", "claude_code", "pi", "codex_via_pi"}:
            raise ConfigError(f"{where} final_response mode requires a registered non-fake session")
        if tool_ids:
            raise ConfigError(
                f"{where} final_response mode keeps native tools; AlphaApollo tools are unsupported"
            )
        if session.get("extra_args"):
            raise ConfigError(
                f"{where} final_response mode does not support session.extra_args; "
                "use declared session options to preserve model, prompt, and tool controls"
            )
        if agent == "codex":
            overrides = session.get("config_overrides", ())
            if isinstance(overrides, (str, bytes)) or not isinstance(overrides, Sequence):
                raise ConfigError(f"{where} final_response config_overrides must be a sequence")
            for override in overrides:
                key = override.partition("=")[0].strip() if isinstance(override, str) else ""
                if key not in {
                    "model_provider",
                    "model_reasoning_effort",
                    "model_reasoning_summary",
                } and not key.startswith("model_providers."):
                    raise ConfigError(
                        f"{where} final_response config_overrides only support provider routing "
                        "and reasoning settings; model and native instructions/tools are fixed"
                    )
        if (agent == "claude_code" and session.get("tools") is not None) or (
            agent in {"pi", "codex_via_pi"}
            and (session.get("tool_mode", "default") != "default" or session.get("tools"))
        ):
            raise ConfigError(
                f"{where} final_response mode requires the native session tool defaults"
            )
        if session.get("model") not in (None, options["model"]):
            raise ConfigError(
                f"{where}.options.model and {where}.options.session.model must match "
                "in final_response mode"
            )
        session["model"] = options["model"]
        return session
    agent = options["agent"]
    runtime_model = options["model"]
    session_model = session.get("model")
    if session_model is not None and session_model != runtime_model:
        raise ConfigError(
            f"{where}.options.model and {where}.options.session.model must match for "
            f"agent {agent!r}"
        )
    if agent != "fake" and runtime_model not in {"default", "Codex CLI default"}:
        session["model"] = runtime_model
    if options.get("reuse_session", False):
        if agent != "codex":  # protected by semantic validation; defensive for direct calls
            raise ConfigError(
                f"{where}.options.reuse_session is currently verified only for agent 'codex'"
            )
        session["continuation"] = True
    if environment_backed:
        agent = options["agent"]
        if agent == "codex":
            raise ConfigError(
                f"{where} cannot use an Environment because Codex cannot disable all "
                "native execution tools; use environment.type 'none' or agent "
                "'codex_via_pi'"
            )
        if agent == "claude_code":
            native_tools = session.get("tools")
            if native_tools not in (None, (), []):
                raise ConfigError(
                    f"{where}.options.session.tools must be empty when an Environment "
                    "owns external-agent tools"
                )
            session["tools"] = ()
        elif agent in {"pi", "codex_via_pi"}:
            required_mode = "no_builtin" if tool_ids else "none"
            mode = session.get("tool_mode", required_mode)
            if mode != required_mode:
                raise ConfigError(
                    f"{where}.options.session.tool_mode must be {required_mode!r} when an "
                    "Environment owns external-agent tools"
                )
            session["tool_mode"] = mode
        elif agent == "fake":
            if tool_ids:
                raise ConfigError(
                    f"{where} cannot grant Environment-owned tools to the scripted fake agent"
                )
        else:
            raise ConfigError(
                f"{where} agent {agent!r} has no verified exclusive Environment tool mode"
            )
    if not tool_ids:
        return session
    declared = dict(session.get("mcp_servers") or {})
    if BRIDGE_SERVER_NAME in declared:
        raise ConfigError(
            f"{where}.options grants tools and also declares an "
            f"{BRIDGE_SERVER_NAME!r} MCP server; declare one or the other"
        )
    if environment_backed:
        from alphaapollo.reasoning.runtime.external.bridge.environment_socket import (
            ENVIRONMENT_SOCKET_NAME,
        )

        declared[BRIDGE_SERVER_NAME] = alphaapollo_bridge(
            tool_ids=tool_ids,
            environment_socket=ENVIRONMENT_SOCKET_NAME,
        )
    else:
        bridge = alphaapollo_bridge(tool_ids=tool_ids)
        if "emx_simulate" in tool_ids and os.environ.get("ALPHAAPOLLO_CHIPS_CONFIG"):
            bridge["env"]["ALPHAAPOLLO_CHIPS_CONFIG"] = os.environ["ALPHAAPOLLO_CHIPS_CONFIG"]
        declared[BRIDGE_SERVER_NAME] = bridge
    if options["agent"] == "codex":
        approvals = dict(session.get("mcp_auto_approve_tools") or {})
        approvals[BRIDGE_SERVER_NAME] = list(tool_ids)
        session["mcp_auto_approve_tools"] = approvals
    session["mcp_servers"] = declared
    return session


def _validate_backend_options(value: object, *, where: str) -> None:
    raw = _closed_options(
        value,
        allowed={"type", "options"},
        required={"type"},
        where=where,
    )
    backend_type = raw["type"]
    _non_empty_string(backend_type, f"{where}.type")
    if backend_type not in {"fake", "openai_compatible"}:
        raise ConfigError(
            f"{where}.type must be 'fake' or 'openai_compatible', got {backend_type!r}"
        )
    if backend_type == "fake":
        options = _closed_options(
            raw.get("options", {}),
            allowed={"text", "verdict", "feedback"},
            required=set(),
            where=f"{where}.options",
        )
        text = options.get("text")
        if text is not None and not isinstance(text, str):
            raise ConfigError(f"{where}.options.text must be a string or null")
        verdict = options.get("verdict", "pass")
        _non_empty_string(verdict, f"{where}.options.verdict")
        if verdict not in {"pass", "fail", "inconclusive"}:
            raise ConfigError(f"{where}.options.verdict must be pass, fail, or inconclusive")
        _non_empty_string(
            options.get("feedback", "fake verifier accepted"), f"{where}.options.feedback"
        )
        return
    options = _closed_options(
        raw.get("options", {}),
        allowed={
            "base_url",
            "api_key",
            "max_retries",
            "timeout",
            "return_token_ids",
            "max_concurrency",
        },
        required={"base_url"},
        where=f"{where}.options",
    )
    _non_empty_string(options["base_url"], f"{where}.options.base_url")
    _non_empty_string(options.get("api_key", "EMPTY"), f"{where}.options.api_key")
    max_retries = options.get("max_retries", 3)
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ConfigError(f"{where}.options.max_retries must be a non-negative int")
    timeout = options.get("timeout", 300.0)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ConfigError(f"{where}.options.timeout must be finite and positive")
    return_token_ids = options.get("return_token_ids", False)
    if not isinstance(return_token_ids, bool):
        raise ConfigError(f"{where}.options.return_token_ids must be a bool")
    max_concurrency = options.get("max_concurrency", 1)
    if (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency < 1
    ):
        raise ConfigError(f"{where}.options.max_concurrency must be a positive int")
    if return_token_ids and _canonical_openai_backend_type() is None:
        raise ConfigError(
            f"{where}.options.return_token_ids=true requires the canonical "
            "OpenAICompatibleGenerationBackend from #189; the legacy backend cannot "
            "return token-native generation data"
        )


def _needed_runtime_keys(
    config: RunConfig,
    verifier_specs: Mapping[str, _AgentVerifierSpec | None],
) -> set[str]:
    roles = {role.id: role for role in config.workflow.roles}
    needed = {roles[step.role].target for step in config.workflow.steps if step.kind == "agent"}
    needed.update(spec.runtime_key for spec in verifier_specs.values() if spec is not None)
    return needed


def _runtime_tool_ids(
    config: RunConfig,
    verifier_specs: Mapping[str, _AgentVerifierSpec | None],
    runtime_name: str,
) -> tuple[str, ...]:
    """Return the ordered union of native tools granted through one Runtime."""

    roles = {role.id: role for role in config.workflow.roles}
    selected = list(_configured_runtime_tools(config.runtimes[runtime_name]))
    for step in config.workflow.steps:
        role = roles[step.role]
        if step.kind == "agent":
            effective_runtime = role.target
        else:
            spec = verifier_specs[role.target]
            effective_runtime = None if spec is None else spec.runtime_key
        if effective_runtime != runtime_name:
            continue
        role_tools = _expand_tool_ids(
            role.tools,
            where=f"role {role.id!r}.tools",
        )
        for tool_id in role_tools:
            if tool_id not in selected:
                selected.append(tool_id)
    return tuple(selected)


def _configured_runtime_tools(
    resource: ResourceConfig,
    *,
    where: str = "runtime",
) -> tuple[str, ...]:
    tools = _thaw_json(resource.options).get("tools", ())
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        raise ConfigError(f"{where}.options.tools must be a sequence of tool ids")
    normalized: list[str] = []
    for index, tool_id in enumerate(tools):
        _non_empty_string(tool_id, f"{where}.options.tools[{index}]")
        normalized.append(tool_id)
    if len(set(normalized)) != len(normalized):
        raise ConfigError(f"{where}.options.tools contains duplicate entries")
    return _expand_tool_ids(
        normalized,
        where=f"{where}.options.tools",
    )


def _expand_tool_ids(tool_ids: Sequence[str], *, where: str) -> tuple[str, ...]:
    """Keep explicit tool IDs ordered and unique; catalogs validate availability."""
    return tuple(dict.fromkeys(tool_ids))


def _effective_workflow_config(
    config: RunConfig,
    verifier_specs: Mapping[str, _AgentVerifierSpec | None],
) -> WorkflowConfig:
    """Apply runtime tool grants to only the roles executed by that runtime."""

    role_kinds = {step.role: step.kind for step in config.workflow.steps}
    roles: list[RoleConfig] = []
    for role in config.workflow.roles:
        if role_kinds[role.id] == "agent":
            runtime_name = role.target
        else:
            spec = verifier_specs[role.target]
            runtime_name = None if spec is None else spec.runtime_key
        configured = (
            () if runtime_name is None else _configured_runtime_tools(config.runtimes[runtime_name])
        )
        declared = _expand_tool_ids(
            role.tools,
            where=f"role {role.id!r}.tools",
        )
        roles.append(replace(role, tools=tuple(dict.fromkeys((*declared, *configured)))))
    return replace(config.workflow, roles=tuple(roles))


def _build_runtime(
    name: str,
    resource: ResourceConfig,
    *,
    environment_factory: Any,
    verifier_formats: Mapping[str, str],
    tools: Sequence[Mapping[str, Any]],
    tool_ids: Sequence[str] = (),
) -> tuple[object, object | None]:
    options = _thaw_json(resource.options)
    if resource.type == "external":
        return (
            _build_external_runtime(
                resource,
                options,
                tool_ids,
                environment_factory=environment_factory,
                where=f"runtimes.{name}",
            ),
            None,
        )
    backend = _build_backend(options["backend"], verifier_formats=verifier_formats)
    sampling = dict(options.get("sampling", {}))
    try:
        runtime = AlphaApolloAgentRuntime(
            backend,
            model=options["model"],
            environment_factory=environment_factory,
            temperature=float(sampling.get("temperature", 0.6)),
            max_tokens=sampling.get("max_tokens", 4096),
            top_p=float(sampling.get("top_p", 1.0)),
            max_turns=options.get("max_turns", 6),
            seed=sampling.get("seed"),
            tools=tools,
            tool_choice=sampling.get("tool_choice"),
            provider_options=sampling.get("provider_options", {}),
            image_history_messages=options.get("image_history_messages"),
        )
    except BaseException:
        _shutdown_resources((backend,))
        raise
    return runtime, backend


def _build_external_runtime(
    resource: ResourceConfig,
    options: Mapping[str, Any],
    tool_ids: Sequence[str],
    *,
    environment_factory: Any = None,
    where: str,
) -> object:
    from alphaapollo.reasoning.runtime.external import session_factory
    from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalAgentRuntime

    agent = options["agent"]
    return ExternalAgentRuntime(
        session_factory(
            agent,
            _external_session_options(
                resource,
                tool_ids,
                where=where,
                environment_backed=environment_factory is not None,
            ),
        ),
        agent=agent,
        model=options["model"],
        workspace_root=options.get("workspace_root"),
        keep_workspaces=options.get("keep_workspaces", False),
        reuse_workspace=options.get("reuse_workspace", False),
        reuse_session=options.get("reuse_session", False),
        concurrency=options.get("concurrency", 1),
        environment_factory=environment_factory,
        environment_mode=options.get("environment_mode", "tool_loop"),
    )


def _build_backend(value: object, *, verifier_formats: Mapping[str, str]) -> object:
    raw = _thaw_json(value)  # semantic validation has already required a mapping
    backend_type = raw["type"]
    options = dict(raw.get("options", {}))
    if backend_type == "fake":
        return _FakeBackend(
            verifier_formats=verifier_formats,
            text=options.get("text"),
            verdict=options.get("verdict", "pass"),
            feedback=options.get("feedback", "fake verifier accepted"),
        )
    canonical_backend_type = _canonical_openai_backend_type()
    return _OwnedGenerationBackend(
        canonical_backend_type(
            base_url=options["base_url"],
            api_key=options.get("api_key", "EMPTY"),
            max_retries=options.get("max_retries", 3),
            timeout=float(options.get("timeout", 300.0)),
            return_token_ids=options.get("return_token_ids", False),
            max_concurrency=options.get("max_concurrency", 1),
        )
    )


def _canonical_openai_backend_type() -> Any:
    from alphaapollo.common.generation import OpenAICompatibleGenerationBackend

    return OpenAICompatibleGenerationBackend


def _message_content(messages: object, role: str) -> str:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        return ""
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == role:
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _last_message_content(messages: object, role: str) -> str:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        return ""
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == role:
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""
