"""Bounded native Apollo validation probe, not a new public experiment entry point.

Uses the same public vaBench task and Environment as Pi. HTTP hooks retain actual
request/response bodies (without headers) and enforce serialized request bytes.
The CLI supervisor bounds the entire child, including simulator client waits.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def run_native(config, evidence, *, http_transport=None, transport_factory=None):
    import httpx
    from openai import OpenAI

    from alphaapollo.common.environment.chips import ChipsEnvironment
    from alphaapollo.common.generation.backends.openai_compatible import (
        OpenAICompatibleGenerationBackend,
    )
    from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
    from alphaapollo.reasoning.runtime.alphaapollo_agent_runtime import AlphaApolloAgentRuntime
    from alphaapollo.workflows.chips_pi_runtime import _write_record
    from alphaapollo.workflows.chips_vabench_agent import agent_input, transport, validate_pilot

    validate_pilot(config)
    if config.get("provider_kind", "glm_coding") != "glm_coding":
        raise ValueError("this probe supports the GLM-compatible endpoint only")
    if config.get("thinking") not in ("low", "high", "xhigh"):
        raise ValueError("native GLM probe requires low, high or xhigh thinking")
    key = os.environ.get("CHIPS_MODEL_KEY")
    if not key:
        raise ValueError("CHIPS_MODEL_KEY is required")
    evidence = Path(evidence).absolute()
    evidence.mkdir(mode=0o700, parents=True, exist_ok=False)
    wire = evidence / "http"
    wire.mkdir(mode=0o700)
    _write_record(evidence / "operator.json", config, key)
    spec = agent_input(config, "pi")
    _write_record(evidence / "agent-input.json", spec, key)
    _write_record(
        evidence / "probe-settings.json",
        {
            "agent": "alphaapollo",
            "http_retries": 0,
            "max_model_calls": config.get("max_model_calls", 12),
            "max_request_bytes": config.get("max_request_bytes", 64000),
            "max_output_tokens": config.get("max_output_tokens", 4096),
            "thinking": config["thinking"],
            "native_tools": "disabled",
        },
        key,
    )
    count = 0
    started = time.monotonic()

    def on_request(request):
        nonlocal count
        body = request.read()
        count += 1
        blocked = count > config.get("max_model_calls", 12) or len(body) > config.get(
            "max_request_bytes", 64000
        )
        _write_record(
            wire / f"{count:03d}-request.json",
            {
                "attempt": count,
                "state": "rejected_before_send" if blocked else "dispatch_started",
                "elapsed_s": time.monotonic() - started,
                "bytes": len(body),
                "payload": json.loads(body),
            },
            key,
        )
        if blocked:
            raise ValueError("native model request budget exhausted before send")

    def on_response(response):
        response.read()
        _write_record(
            wire / f"{count:03d}-response.json",
            {
                "attempt": count,
                "status_code": response.status_code,
                "elapsed_s": time.monotonic() - started,
                "body": response.text,
            },
            key,
        )

    client = httpx.Client(
        transport=http_transport, event_hooks={"request": [on_request], "response": [on_response]}
    )
    sdk = None

    def client_factory(**kwargs):
        nonlocal sdk
        sdk = OpenAI(http_client=client, **kwargs)
        return sdk

    try:
        backend = OpenAICompatibleGenerationBackend(
            base_url=config["base_url"],
            api_key=key,
            max_retries=0,
            timeout=min(config.get("episode_timeout_s", 600), 300),
            client_factory=client_factory,
        )
        runtime = AlphaApolloAgentRuntime(
            backend,
            model=config["model"],
            tools=spec["tools"],
            max_turns=config.get("max_model_calls", 12),
            max_tokens=config.get("max_output_tokens", 4096),
            provider_options={
                "reasoning_effort": {"xhigh": "max"}.get(config["thinking"], config["thinking"]),
                "extra_body": {"thinking": {"type": "enabled", "clear_thinking": False}},
            },
            environment_factory=lambda _: ChipsEnvironment(
                (transport_factory or transport)(config, evidence / "tools"), task_kind="vabench"
            ),
        )
        result = runtime.run_batch(
            [
                AgentTask(
                    task_id=spec["task_id"],
                    system=spec["system"],
                    prompt=spec["prompt"],
                    tools=tuple(t["function"]["name"] for t in spec["tools"]),
                )
            ]
        )[0]
        payload = _write_record(evidence / "runtime-result.json", result, key)
        outcome = {"state": "completed", "termination_reason": payload["termination_reason"]}
    except Exception as exc:
        outcome = {"state": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        if sdk is not None:
            sdk.close()
        client.close()
    return _write_record(
        evidence / "native-outcome.json",
        {
            **outcome,
            "request_hooks": count,
            "elapsed_s": time.monotonic() - started,
            "evaluation": "not_performed",
        },
        key,
    )


def main():
    from alphaapollo.common.execution.chips.journal import atomic_json
    from alphaapollo.workflows.chips_analog_agent import load_private_key
    from alphaapollo.workflows.chips_vabench_agent import validate_pilot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_pilot(config)
    if args.worker:
        os.environ["CHIPS_MODEL_KEY"] = load_private_key(args.key_file)
        outcome = run_native(config, args.evidence)
        print(json.dumps(outcome))
        return 0 if outcome["state"] == "completed" else 1
    # A fresh evidence directory is also the no-overwrite guard for failed runs.
    if args.evidence.exists():
        raise ValueError("use a new evidence directory")
    child = subprocess.Popen(
        [sys.executable, __file__, *sys.argv[1:], "--worker"], start_new_session=True
    )
    try:
        return child.wait(timeout=config.get("episode_timeout_s", 600))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        args.evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_json(
            args.evidence / "native-outcome.json",
            {
                "state": "timeout",
                "server_execution": "unknown",
                "guidance": "Recover pending action IDs before finalization or further actions.",
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
