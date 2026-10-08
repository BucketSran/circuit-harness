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

"""Validate Verifier declarations and bind them to already constructed Runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from alphaapollo.reasoning.verification import (
    AgentVerifier,
    AgentVerifierConfig,
    create_custom_verifier,
    get_custom_verifier_factory,
)
from alphaapollo.workflows.config import (
    ConfigError,
    ResourceConfig,
    RoleConfig,
    RunConfig,
    _closed_options,
    _non_empty_string,
    _thaw_json,
)


@dataclass(frozen=True, slots=True)
class _AgentVerifierSpec:
    role: RoleConfig
    runtime_key: str
    config: AgentVerifierConfig


def _validate_verifier_resource(name: str, resource: ResourceConfig, config: RunConfig) -> None:
    where = f"verifier {name!r}"
    if resource.type == "agent":
        options = _closed_options(
            resource.options,
            allowed={"runtime"},
            required={"runtime"},
            where=f"{where}.options",
        )
        runtime = options["runtime"]
        _non_empty_string(runtime, f"{where}.options.runtime")
        if runtime not in config.runtimes:
            raise ConfigError(f"{where} references unknown runtime {runtime!r}")
        return
    if resource.type == "custom":
        options = _closed_options(
            resource.options,
            allowed={"name", "config"},
            required={"name"},
            where=f"{where}.options",
        )
        registered_name = _non_empty_string(options["name"], f"{where}.options.name")
        custom_config = options.get("config", {})
        if not isinstance(custom_config, Mapping):
            raise ConfigError(f"{where}.options.config must be a mapping")
        try:
            get_custom_verifier_factory(registered_name)
        except ValueError as exc:
            raise ConfigError(f"{where}.options.name is invalid: {exc}") from exc
        return
    raise ConfigError(f"{where}.type must be 'agent' or 'custom', got {resource.type!r}")


def _resolve_verifier_specs(config: RunConfig) -> dict[str, _AgentVerifierSpec | None]:
    """Resolve validated Verifier declarations and their compatible role bindings."""

    roles = {role.id: role for role in config.workflow.roles}
    by_target: dict[str, list[RoleConfig]] = {}
    for step in config.workflow.steps:
        if step.kind == "verifier":
            role = roles[step.role]
            by_target.setdefault(role.target, []).append(role)

    resolved: dict[str, _AgentVerifierSpec | None] = {}
    for target, target_roles in by_target.items():
        resource = config.verifiers[target]
        unique_roles = list(dict.fromkeys(target_roles))
        first = unique_roles[0]
        expected_signature = _verifier_role_signature(first)
        for role in unique_roles[1:]:
            if _verifier_role_signature(role) != expected_signature:
                raise ConfigError(
                    f"verifier target {target!r} is shared by conflicting role configurations"
                )
        if resource.type == "custom":
            for role in unique_roles:
                _validate_non_agent_verifier_role(role, verifier_type=resource.type)
            resolved[target] = None
            continue
        system_prompt = _validate_agent_verifier_role(first)
        options = _thaw_json(resource.options)
        runtime_key = options["runtime"]
        runtime_model = dict(config.runtimes[runtime_key].options)["model"]
        try:
            verifier_config = AgentVerifierConfig(
                system_prompt=system_prompt,
                input_template=first.input_template or "",
                role=first.id,
                model=first.model or runtime_model,
                tools=first.tools,
                output_format=first.output_format or "",
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"verifier role {first.id!r} cannot configure AgentVerifier: {exc}"
            ) from exc
        resolved[target] = _AgentVerifierSpec(
            role=first,
            runtime_key=runtime_key,
            config=verifier_config,
        )
    return resolved


def _validate_agent_verifier_role(role: RoleConfig) -> str:
    where = f"agent verifier role {role.id!r}"
    system_prompt = _non_empty_string(role.system_prompt, f"{where}.system_prompt")
    if role.input_template is None or not role.input_template.strip():
        raise ConfigError(f"{where}.input_template must be a non-empty string")
    if role.output_format not in {"json", "verdict-line"}:
        raise ConfigError(
            f"{where}.output_format must be 'json' or 'verdict-line', got {role.output_format!r}"
        )
    return system_prompt


def _validate_non_agent_verifier_role(role: RoleConfig, *, verifier_type: str) -> None:
    """Reject AgentVerifier-only role fields for a non-agent Verifier."""

    configured: list[str] = []
    if role.system_prompt is not None:
        configured.append("system_prompt")
    if role.model is not None:
        configured.append("model")
    if role.input_template is not None:
        configured.append("input_template")
    if role.output_format is not None:
        configured.append("output_format")
    if role.tools:
        configured.append("tools")
    if configured:
        raise ConfigError(
            f"{verifier_type} verifier role {role.id!r} cannot configure "
            "AgentVerifier-only fields: "
            f"{configured}"
        )


def _verifier_role_signature(role: RoleConfig) -> tuple[object, ...]:
    return (
        role.id,
        role.system_prompt,
        role.model,
        role.tools,
        role.input_template,
        role.output_format,
    )


def _fake_verifier_formats(
    verifier_specs: Mapping[str, _AgentVerifierSpec | None],
) -> dict[str, dict[str, str]]:
    by_runtime: dict[str, dict[str, str]] = {}
    for spec in verifier_specs.values():
        if spec is None:
            continue
        formats = by_runtime.setdefault(spec.runtime_key, {})
        previous = formats.get(spec.config.system_prompt)
        if previous is not None and previous != spec.config.output_format:
            raise ConfigError(
                "fake backend cannot disambiguate identical verifier system prompts with "
                "different output formats"
            )
        formats[spec.config.system_prompt] = str(spec.config.output_format)
    return by_runtime


def _build_verifier(
    resource: ResourceConfig,
    spec: _AgentVerifierSpec | None,
    runtimes: Mapping[str, object],
) -> tuple[object, tuple[object, ...]]:
    if resource.type == "agent":
        assert spec is not None
        return AgentVerifier(runtimes[spec.runtime_key], spec.config), ()  # type: ignore[arg-type]
    options = _thaw_json(resource.options)
    if resource.type == "custom":
        return create_custom_verifier(options["name"], options.get("config", {})), ()
    raise ConfigError(f"unsupported verifier type {resource.type!r}")
