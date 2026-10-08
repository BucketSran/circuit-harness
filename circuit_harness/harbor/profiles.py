"""Compose independent agent and protocol connections into stock Harbor configs.

This module only prepares configuration. Harbor owns installation, trials and agent loops.
Credentials remain environment references until Harbor's AgentFactory resolves them.
"""

from copy import deepcopy
from typing import Any, Literal
from urllib.parse import urlsplit

from harbor.agents.factory import AgentFactory
from harbor.agents.model_connection import PROVIDERS
from harbor.models.trial.config import AgentConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import require_harbor_version

Protocol = Literal["openai-responses", "openai-chat-completions", "anthropic-messages"]


class Connection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol: Protocol
    model: str = Field(min_length=1, pattern=r"^\S+$")
    base_url: str = Field(min_length=1)
    key_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")

    @field_validator("base_url")
    @classmethod
    def explicit_http_endpoint(cls, value: str) -> str:
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or any(char.isspace() for char in value)
        ):
            raise ValueError("base_url requires an explicit HTTP(S) endpoint without credentials")
        return value


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    connections: list[Connection] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_protocols(self):
        protocols = [connection.protocol for connection in self.connections]
        if len(protocols) != len(set(protocols)):
            raise ValueError("each model connection protocol must be unique")
        return self


_SUPPORTED = {
    "codex": {"openai-responses"},
    "claude-code": {"anthropic-messages"},
    "pi": {"openai-responses", "openai-chat-completions", "anthropic-messages"},
    "mini-swe-agent": {"openai-chat-completions", "anthropic-messages"},
}
# These Harbor escape hatches bypass the explicitly selected API connection.
_CONNECTION_OVERRIDES = {
    "codex": {"CODEX_AUTH_JSON_PATH": False, "CODEX_FORCE_AUTH_JSON": True},
    "claude-code": {
        "CLAUDE_FORCE_OAUTH": True,
        "CLAUDE_CODE_USE_BEDROCK": True,
        "CLAUDE_CODE_USE_VERTEX": True,
        "CLAUDE_CODE_USE_FOUNDRY": True,
        "AWS_BEARER_TOKEN_BEDROCK": False,
    },
}


def validate_connection_environment(name, env=None):
    """Reject legacy routing controls at compilation and again on the run host."""
    import os

    explicit = env or {}
    for key, is_boolean in _CONNECTION_OVERRIDES.get(name, {}).items():
        value = explicit[key] if key in explicit else os.environ.get(key, "")
        active = (
            value.strip().lower() not in {"", "0", "false", "no"} if is_boolean else bool(value)
        )
        if active:
            raise ValueError(f"unset {key}: conflicts with the selected model connection")


_ROUTING_ENVS = (
    {
        name
        for access in PROVIDERS.values()
        for name in (*access.api_key_envs, *access.base_url_envs, *access.extra_envs)
    }
    | {
        "MSWEA_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_MODEL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CODEX_HOME",
        "PI_CODING_AGENT_DIR",
    }
    | {key for overrides in _CONNECTION_OVERRIDES.values() for key in overrides}
)
_ROUTING_OPTIONS = {
    "modelname",
    "modelapi",
    "modelclass",
    "modelprovider",
    "modelproviders",
    "apikey",
    "apikeyhelper",
    "apibase",
    "baseurl",
    "openaibaseurl",
    "endpoint",
    "provider",
    "providers",
    "authtoken",
    "headers",
    "extraheaders",
    "defaultheaders",
    "auth",
    "configfile",
    "customllmprovider",
    "wireapi",
    "apikeyenv",
}


def _reject_routing(value: Any, path: str = "agent") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("_", "").replace("-", "")
            if (
                key in _ROUTING_ENVS
                or normalized in _ROUTING_OPTIONS
                or key.endswith(("_API_KEY", "_AUTH_TOKEN", "_BASE_URL", "_API_BASE"))
                or (normalized == "model" and not isinstance(child, dict))
            ):
                raise ValueError(f"{path}.{key} conflicts with the selected model connection")
            _reject_routing(child, f"{path}.{key}")
    elif isinstance(value, list):
        for child in value:
            _reject_routing(child, path)


def _validate_agent_profile(value: dict[str, Any]) -> None:
    unknown = value.keys() - AgentConfig.model_fields.keys()
    if unknown:
        raise ValueError(f"unknown agent settings: {sorted(unknown)}")
    if value.get("name") not in _SUPPORTED:
        raise ValueError("agent name must be an explicitly selected supported Harbor built-in")
    if "model_name" in value or "import_path" in value:
        raise ValueError("agent profiles cannot declare model_name or import_path")
    _reject_routing(value)
    agent = AgentConfig.model_validate(value)
    kwargs = agent.kwargs
    native_config = kwargs.get("config")
    if native_config is not None and not isinstance(native_config, dict):
        raise ValueError(
            "profile native config must be inline, so routing conflicts can be checked"
        )
    if agent.name == "mini-swe-agent" and native_config is not None:
        model = native_config.get("model", {})
        if not isinstance(model, dict) or not isinstance(model.get("model_kwargs", {}), dict):
            raise ValueError("mini-swe native model and model_kwargs must be objects")
    agent_class = AgentFactory.get_agent_class_from_config(agent)
    agent_class.options_model.model_validate(agent.kwargs)
    if native_config is not None and not (
        agent_class.capabilities.native_config or agent.name == "mini-swe-agent"
    ):
        raise ValueError(f"{agent.name} does not support native config")


class ProfileCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, revalidate_instances="always")
    schema_version: Literal[1] = 1
    agents: dict[str, dict[str, Any]] = Field(min_length=1)
    models: dict[str, ModelProfile] = Field(min_length=1)

    @field_validator("agents")
    @classmethod
    def validate_agent_profiles(cls, value):
        for alias, agent in value.items():
            if not alias.strip():
                raise ValueError("agent alias must not be blank")
            _validate_agent_profile(agent)
        return value


def compile_agent(
    catalog: ProfileCatalog | dict[str, Any],
    agent_key: str,
    model_key: str,
    protocol: Protocol | None = None,
) -> AgentConfig:
    """Select a compatible connection; do not read credentials or run an agent."""
    require_harbor_version()
    catalog = ProfileCatalog.model_validate(catalog)
    if agent_key not in catalog.agents:
        raise ValueError(f"unknown agent alias: {agent_key}")
    if model_key not in catalog.models:
        raise ValueError(f"unknown model alias: {model_key}")
    agent = deepcopy(catalog.agents[agent_key])
    name = agent["name"]
    validate_connection_environment(name, agent.get("env"))
    connections = [
        connection
        for connection in catalog.models[model_key].connections
        if connection.protocol in _SUPPORTED[name]
        and (protocol is None or connection.protocol == protocol)
    ]
    if not connections:
        raise ValueError(
            f"no compatible connection for {agent_key}/{model_key} protocol={protocol}"
        )
    if len(connections) != 1:
        raise ValueError("ambiguous connection; select an explicit protocol")
    connection = connections[0]
    if name == "codex" and "/" in connection.model:
        raise ValueError(
            "Codex in Harbor 0.23 truncates slash-containing model IDs; choose another agent"
        )
    if (
        name == "mini-swe-agent"
        and connection.protocol == "openai-chat-completions"
        and agent.get("kwargs", {}).get("reasoning_effort") is not None
    ):
        raise ValueError(
            "mini-swe-agent reasoning_effort switches OpenAI to Responses; "
            "it is incompatible with the selected chat-completions protocol"
        )
    messages = connection.protocol == "anthropic-messages"
    provider = "anthropic" if messages else "openai"
    key_var = "ANTHROPIC_API_KEY" if messages else "OPENAI_API_KEY"
    url_var = "ANTHROPIC_BASE_URL" if messages else "OPENAI_BASE_URL"
    agent["model_name"] = (
        connection.model if name == "claude-code" else f"{provider}/{connection.model}"
    )
    agent["env"] = {
        **agent.get("env", {}),
        key_var: f"${{{connection.key_env}}}",
        url_var: connection.base_url,
    }
    if name == "pi":
        agent.setdefault("kwargs", {})["model_api"] = (
            "openai-completions"
            if connection.protocol == "openai-chat-completions"
            else connection.protocol
        )
    if name == "mini-swe-agent":
        agent["env"]["MSWEA_API_KEY"] = f"${{{connection.key_env}}}"
        kwargs = agent.setdefault("kwargs", {})
        native = kwargs["config"] = kwargs.get("config") or {}
        model = native.setdefault("model", {})
        model["model_class"] = "litellm"
        model.setdefault("model_kwargs", {})["api_base"] = connection.base_url
    result = AgentConfig.model_validate(agent)
    AgentFactory.get_agent_class_from_config(result).options_model.model_validate(result.kwargs)
    return result


def _reject_unknown_fields(raw: Any, parsed: Any, path: str) -> None:
    if isinstance(parsed, BaseModel) and isinstance(raw, dict):
        unknown = raw.keys() - type(parsed).model_fields.keys()
        if unknown:
            raise ValueError(f"unknown {path} settings: {sorted(unknown)}")
        for name, value in raw.items():
            _reject_unknown_fields(value, getattr(parsed, name), f"{path}.{name}")
    elif isinstance(parsed, list) and isinstance(raw, list):
        for index, (value, item) in enumerate(zip(raw, parsed, strict=True)):
            _reject_unknown_fields(value, item, f"{path}[{index}]")


def compile_job(
    catalog: ProfileCatalog | dict[str, Any],
    agent_key: str,
    model_key: str,
    job: dict[str, Any],
    protocol: Protocol | None = None,
):
    """Replace the template's agents with one selected stock AgentConfig."""
    from harbor.models.job.config import JobConfig

    if not isinstance(job, dict):
        raise ValueError("job template must be a JSON object")
    selected = compile_agent(catalog, agent_key, model_key, protocol)
    if job.get("environment", {}).get("import_path") in {
        "circuit_harness.harbor.docker_environment:CircuitDockerEnvironment",
        "circuit_harness.harbor.podman_environment:CircuitPodmanEnvironment",
    }:
        if (
            selected.resume_trajectory
            or selected.load_trajectory is not None
            or job.get("user_agent") is not None
        ):
            raise ValueError(
                "Circuit jobs support a single-step agent without resume or user_agent"
            )
        selected = selected.model_copy(
            update={
                "name": None,
                "import_path": "circuit_harness.harbor.installed_agent:CircuitAgent",
                "kwargs": {"agent_name": selected.name, "agent_kwargs": selected.kwargs},
            }
        )
    raw = {
        **deepcopy(job),
        "agents": [selected.model_dump(mode="json", context={"redact_sensitive_env": False})],
    }
    result = JobConfig.model_validate(raw)
    _reject_unknown_fields(raw, result, "job")
    from .task_bindings import validate_job_task_bindings

    validate_job_task_bindings(result)
    return result


def catalog_schema() -> dict[str, Any]:
    """Build the catalog schema from the pinned Harbor AgentConfig and options."""
    require_harbor_version()
    schema = ProfileCatalog.model_json_schema()
    stock_agent = AgentConfig.model_json_schema()
    definitions = schema.setdefault("$defs", {})
    definitions.update(stock_agent.pop("$defs", {}))
    stock_agent["additionalProperties"] = False
    stock_agent["required"] = ["name"]
    properties = stock_agent["properties"]
    properties.pop("model_name")
    properties.pop("import_path")
    properties["name"] = {"enum": list(_SUPPORTED), "type": "string"}
    stock_agent["allOf"] = []
    for name in _SUPPORTED:
        agent_class = AgentFactory.get_agent_class_from_config(AgentConfig(name=name))
        options = agent_class.options_model.model_json_schema()
        definitions.update(options.pop("$defs", {}))
        # Connection translation owns these options; profiles cannot preselect them.
        for field in ("model_api", "config_file"):
            options["properties"].pop(field, None)
        stock_agent["allOf"].append(
            {
                "if": {"properties": {"name": {"const": name}}},
                "then": {"properties": {"kwargs": options}},
            }
        )
    definitions["AgentProfile"] = stock_agent
    schema["properties"]["agents"]["additionalProperties"] = {"$ref": "#/$defs/AgentProfile"}
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def _read_json(path):
    import json

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique_keys)


def main(argv: list[str] | None = None) -> int:
    """Write a JobConfig for Harbor's existing runner; never launch a trial."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--protocol", choices=("openai-responses", "openai-chat-completions", "anthropic-messages")
    )
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_job(
            _read_json(args.catalog), args.agent, args.model, _read_json(args.job), args.protocol
        )
        serialized = json.dumps(result.model_dump(mode="json"), indent=2, allow_nan=False) + "\n"
        with args.output.open("x") as output:
            output.write(serialized)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
