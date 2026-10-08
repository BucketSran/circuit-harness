"""One Pi launch recipe for Chips tasks, with an opt-in Apollo Runtime bridge.

Tasks supply their public AgentTask and transport. This module owns credentials,
CLI settings, model budgets and raw/normalized evidence, never simulator rules.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

from alphaapollo.common.execution.chips.journal import atomic_json
from alphaapollo.workflows.chips_experiment_settings import (
    experiment_settings_snapshot,
    pi_zai_model,
)


def validate_pi_runtime(config):
    if config.get("runtime", "direct") not in ("direct", "external"):
        raise ValueError("Pi runtime must be direct or external")
    if "http_retry_limit" in config:
        if type(config["http_retry_limit"]) is not int or config["http_retry_limit"] != 0:
            raise ValueError("http_retry_limit currently supports only explicit zero retries")
        if config.get("provider_kind", "glm_coding") != "glm_coding":
            raise ValueError("http_retry_limit requires an isolated GLM agent directory")


def _json_default(value):
    # AgentResult contains immutable mappings that dataclasses.asdict cannot
    # deepcopy. Preserve the canonical fields instead of converting them to repr.
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"unsupported evidence type: {type(value).__name__}")


def _write_record(path, value, key):
    encoded = json.dumps(value, default=_json_default)
    if key:
        encoded = encoded.replace(key, "[REDACTED]")
    result = json.loads(encoded)
    atomic_json(path, result)
    path.chmod(0o600)
    return result


def _termination_reason(outcome, evidence):
    raw = outcome.termination_reason
    if raw == "external_error" and outcome.provider_metadata.get("returncode") == 73:
        try:
            records = [
                json.loads(line)
                for line in (evidence / "model-budget.jsonl").read_text().splitlines()
                if line.strip()
            ]
        except (OSError, ValueError):
            records = []
        stop = records[-1] if records and isinstance(records[-1], dict) else {}
        if stop.get("event") == "budget_stop" and stop.get("reason") in {
            "model_request_limit",
            "request_bytes_limit",
        }:
            return stop["reason"]
    return {
        "timeout": "deadline",
        "truncated": "output_token_limit",
        "external_error": "runtime_error",
    }.get(raw, raw)


class _RecordedSession:
    def __init__(self, session, evidence, key):
        self.session, self.evidence, self.key = session, evidence, key
        self.payload = None

    def run(self, task, *, workspace):
        outcome = self.session.run(task, workspace=workspace)
        payload = {
            **_json_default(outcome),
            "harness_termination_reason": _termination_reason(outcome, self.evidence),
        }
        self.payload = _write_record(self.evidence / "pi-outcome.json", payload, self.key)
        return outcome

    def close(self):
        self.session.close()


def run_pi_task(config, evidence, task, *, task_kind, transport_factory):
    from alphaapollo.reasoning.runtime.external.agents.pi import EXTENSION_PATH, PiSession
    from alphaapollo.reasoning.runtime.external.bridge.mcp_config import alphaapollo_bridge

    validate_pi_runtime(config)
    provider = config.get("provider_kind", "glm_coding")
    key = os.environ.get("CHIPS_MODEL_KEY") if provider == "glm_coding" else None
    if provider == "glm_coding" and not key:
        raise ValueError("CHIPS_MODEL_KEY is required in the Pi process environment")
    evidence = Path(evidence).absolute()
    if evidence.exists() and (not evidence.is_dir() or (evidence / "operator.json").exists()):
        raise ValueError("use a fresh evidence directory for each model attempt")
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    config_path = evidence / "operator.json"
    atomic_json(config_path, config)
    settings = experiment_settings_snapshot(config, "pi")
    budget_text = (
        "\n\nHarness limits: " + json.dumps(settings["budgets"]) + ". "
        "Budget notices before each model request are runtime observations, not task data. "
        "Use the server budget fields in tool replies for remaining actions and simulations; "
        "an exhausted simulation allowance does not prevent submission. "
        "Follow the task's declared collection policy; "
        "automatic collection is not agent submission."
    )
    task = replace(task, system=(task.system or "") + budget_text)
    from alphaapollo.common.execution.tools.chips import SESSION_TOOL_SPECS

    input_path = evidence / "agent-input.json"
    initial = json.loads(input_path.read_text()) if input_path.exists() else {}
    _write_record(
        input_path,
        {
            **initial,
            "schema_version": 1,
            "task_id": task.task_id,
            "system": task.system,
            "prompt": task.prompt,
            "tools": [
                spec.to_openai_tool() for spec in SESSION_TOOL_SPECS if spec.tool_id in task.tools
            ],
            "native_tools": "disabled",
            "experiment_settings": settings,
            "dynamic_context": "model-budget.jsonl: budget_context events",
        },
        key,
    )
    home = evidence / "pi-home"
    agent_dir = home / ".pi/agent"
    agent_dir.mkdir(mode=0o700, parents=True)
    if config.get("http_retry_limit") == 0:
        atomic_json(
            agent_dir / "settings.json",
            {"retry": {"enabled": False, "maxRetries": 0, "provider": {"maxRetries": 0}}},
        )
    if provider == "glm_coding":
        atomic_json(
            agent_dir / "models.json",
            {
                "providers": {
                    "chips-glm": {
                        "baseUrl": config["base_url"],
                        "api": "openai-completions",
                        "apiKey": "${CHIPS_MODEL_KEY}",
                        "models": [pi_zai_model(config)],
                    }
                }
            },
        )
    workspace = evidence / "workspace"
    workspace.mkdir(mode=0o700)
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONUTF8": "1",
        "PI_CODING_AGENT_DIR": str(
            agent_dir if provider == "glm_coding" else Path(config["pi_auth_dir"])
        ),
        "CHIPS_MAX_MODEL_CALLS": str(config.get("max_model_calls", 12)),
        "CHIPS_MAX_REQUEST_BYTES": str(config.get("max_request_bytes", 64000)),
        "CHIPS_MAX_OUTPUT_TOKENS": str(config.get("max_output_tokens", 4096)),
        "CHIPS_MODEL_BUDGET_TRACE": str(evidence / "model-budget.jsonl"),
    }
    if key:
        env["CHIPS_MODEL_KEY"] = key

    def save_events(raw):
        descriptor = os.open(
            evidence / "pi-events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(raw.replace(key, "[REDACTED]") if key else raw)
            stream.flush()
            os.fsync(stream.fileno())

    external = config.get("runtime", "direct") == "external"
    if external:
        declaration = alphaapollo_bridge(
            tool_ids=task.tools,
            environment_socket=".alphaapollo-environment.json",
        )
        server_name = "alphaapollo"
        command, arguments = declaration["command"], declaration["args"]
    else:
        server_name = task_kind
        command = sys.executable
        arguments = [
            "-m",
            f"alphaapollo.workflows.chips_{task_kind}_agent",
            "serve",
            "--config",
            str(config_path),
            "--evidence",
            str(evidence / "tools"),
        ]
    # The stdio child gets only the transport environment; Pi retains the model
    # key. Relative endpoint lookup uses the Runtime-owned CLI workspace.
    package_parent = str(Path(__file__).resolve().parents[2])
    mcp_servers = {
        server_name: {
            "command": "/usr/bin/env",
            "args": [
                "-i",
                "HOME=" + str(Path.home()),
                "PATH=/usr/bin:/bin:/usr/local/bin",
                "PYTHONUTF8=1",
                "PYTHONPATH=" + package_parent,
                command,
                *arguments,
            ],
            "env": {"PYTHONPATH": package_parent},
        }
    }
    session = _RecordedSession(
        PiSession(
            cli=config["pi_cli"],
            provider="chips-glm" if provider == "glm_coding" else "openai-codex",
            model=config["model"],
            thinking=config.get("thinking"),
            timeout_s=config.get("episode_timeout_s", 600),
            tool_mode="no_builtin",
            env=env,
            event_sink=save_events,
            extra_args=(
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-themes",
                "-e",
                str(EXTENSION_PATH.with_name("pi_budget.ts")),
            ),
            mcp_servers=mcp_servers,
        ),
        evidence,
        key,
    )
    if not external:
        try:
            session.run(task, workspace=workspace)
            return session.payload
        finally:
            session.close()

    from alphaapollo.common.environment.chips import ChipsEnvironment
    from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalAgentRuntime

    runtime = ExternalAgentRuntime(
        lambda: session,
        agent="pi",
        model=config["model"],
        keep_workspaces=True,
        workspace_root=evidence / "runtime-workspaces",
        environment_factory=lambda _: ChipsEnvironment(
            transport_factory(config, evidence / "tools"),
            task_kind=task_kind,
        ),
        max_tokens=config.get("max_output_tokens", 4096),
    )
    try:
        result = runtime.run_batch([task])[0]
        _write_record(evidence / "runtime-result.json", result, key)
        return session.payload
    finally:
        runtime.close()
