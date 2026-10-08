"""Shared model settings for a Chips experiment, independent of its circuit task."""

from __future__ import annotations

FORCED_THINKING_MODELS = frozenset(("glm-5.3", "glm-5.3-flash"))

_PI_LEVELS = frozenset(("off", "minimal", "low", "medium", "high", "xhigh"))
_CODEX_LEVELS = frozenset(("low", "medium", "high", "xhigh", "max", "ultra"))
_GLM_LEVELS = frozenset(("low", "high", "xhigh"))
# Harness experiment bound, not a claim about a provider/model's supported limit.
_PI_MAX_OUTPUT_TOKENS = 32768


def validate_model_settings(config: dict, agent: str) -> None:
    """Reject ambiguous or unsupported reasoning controls before a real model run."""
    if not isinstance(config.get("model"), str) or not config["model"].strip():
        raise ValueError("an explicit model ID is required")
    if agent == "pi":
        field, allowed = "thinking", _PI_LEVELS
        tokens = config.get("max_output_tokens", 4096)
        if type(tokens) is not int or not 1 <= tokens <= _PI_MAX_OUTPUT_TOKENS:
            raise ValueError("invalid pilot limit: max_output_tokens")
    elif agent == "codex":
        field, allowed = "reasoning_effort", _CODEX_LEVELS
    else:
        raise ValueError("unsupported Chips agent")
    if config.get("policy_kind") == "remote_model" and field not in config:
        description = "thinking level" if agent == "pi" else field
        raise ValueError(f"an explicit {description} is required for a real model experiment")
    if field in config and config[field] not in allowed:
        raise ValueError(f"invalid {agent} {field}")
    if agent == "pi" and config["model"] in FORCED_THINKING_MODELS:
        if config.get("thinking") not in _GLM_LEVELS:
            raise ValueError("GLM-5.3 cannot disable thinking; use low, high or xhigh")


def experiment_settings_snapshot(config: dict, agent: str) -> dict:
    """Return the same reviewable experiment settings for every Chips task."""
    validate_model_settings(config, agent)
    field = "thinking" if agent == "pi" else "reasoning_effort"
    return {
        "schema_version": 1,
        "agent": agent,
        "runtime": config.get("runtime", "direct") if agent == "pi" else "direct",
        "model": config["model"],
        "reasoning": {"parameter": field, "value": config.get(field)},
        "budgets": {
            "max_model_calls": config.get("max_model_calls", 12) if agent == "pi" else None,
            "max_request_bytes": config.get("max_request_bytes", 64000) if agent == "pi" else None,
            "max_output_tokens": config.get("max_output_tokens", 4096) if agent == "pi" else None,
            "episode_timeout_s": config.get("episode_timeout_s", 600),
        },
    }


def pi_zai_model(config: dict) -> dict:
    """Describe a Z.ai model once for both Analog and VABench Pi sessions."""
    forced = config["model"] in FORCED_THINKING_MODELS
    model = {
        "id": config["model"],
        "name": config["model"],
        "reasoning": forced,
        "input": ["text"],
        "contextWindow": 1000000 if forced else 32768,
        "maxTokens": config.get("max_output_tokens", 4096),
        "compat": {},
    }
    if forced:
        model["thinkingLevelMap"] = {"low": "low", "high": "high", "xhigh": "max"}
        model["compat"] = {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": True,
            "maxTokensField": "max_tokens",
            "thinkingFormat": "zai",
            "zaiToolStream": True,
        }
    return model
