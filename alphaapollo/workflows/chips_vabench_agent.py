"""Operator entry point and MCP projection for public VABench tools."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from alphaapollo.common.execution.chips.journal import atomic_json
from alphaapollo.common.execution.chips.vabench_remote import LocalVabench, RemoteVabench
from alphaapollo.common.execution.chips.vabench_session import tool_schemas
from alphaapollo.workflows import chips_vabench_record
from alphaapollo.workflows.chips_experiment_settings import (
    experiment_settings_snapshot,
    validate_model_settings,
)

_PI_SYSTEM = (
    "Use only the four VABench public tools. Read task/instruction.md and "
    "task/public_contract.json. Implement the requested candidate, use public "
    "simulation feedback, revise if needed, then call vabench_submit. Public "
    "simulation success is not a final correctness score. No hidden evaluator "
    "is available."
)
_CODEX_SYSTEM = (
    "Use the four VABench public MCP tools for this task. Read task/instruction.md "
    "and task/public_contract.json. Implement the candidate, inspect public "
    "simulation feedback, revise if needed, and call vabench_submit. Public "
    "simulation success is not a final score. No hidden evaluator is available."
)
_PROMPT = "Complete the VABench task. Start by listing public files."


def agent_input(config, kind):
    return {
        "schema_version": 1,
        "task_id": config.get("task_id", "vabench"),
        "system": _PI_SYSTEM if kind == "pi" else _CODEX_SYSTEM,
        "prompt": _PROMPT,
        "tools": tool_schemas(),
        "native_tools": "disabled" if kind == "pi" else "available",
    }


def serve(config, evidence):
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    remote = transport(config, evidence)
    server = Server("chips-vabench-public")
    schemas = tool_schemas()
    call_lock = anyio.Lock()

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=s["function"]["name"],
                description=s["function"]["description"],
                inputSchema=s["function"]["parameters"],
            )
            for s in schemas
        ]

    @server.call_tool()
    async def call_tool(name, arguments):
        async with call_lock:
            result = await anyio.to_thread.run_sync(remote.call, name, arguments)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result))],
            isError=not result["ok"],
        )

    async def run():
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    anyio.run(run)


def validate_pilot(config):
    from urllib.parse import urlsplit

    allowed = {
        "host",
        "transport",
        "python",
        "bundle",
        "session",
        "task_id",
        "job_root",
        "job_id",
        "archive_root",
        "policy_kind",
        "pi_cli",
        "runtime",
        "http_retry_limit",
        "provider_kind",
        "pi_auth_dir",
        "base_url",
        "model",
        "thinking",
        "episode_timeout_s",
        "max_model_calls",
        "max_request_bytes",
        "max_output_tokens",
    }
    if set(config) - allowed:
        raise ValueError("unsupported pilot config field; secrets belong only in the environment")
    if config.get("transport", "ssh") not in ("ssh", "local"):
        raise ValueError("transport must be ssh or local")
    if config.get("policy_kind") not in ("remote_model", "scripted_http_fixture"):
        raise ValueError("declare policy_kind as remote_model or scripted_http_fixture")
    from alphaapollo.workflows.chips_pi_runtime import validate_pi_runtime

    validate_pi_runtime(config)
    provider_kind = config.get("provider_kind", "glm_coding")
    if provider_kind == "glm_coding":
        if "pi_auth_dir" in config:
            raise ValueError("pi_auth_dir is only for openai_codex")
        url = urlsplit(config["base_url"])
        local_fixture = config["policy_kind"] == "scripted_http_fixture" and url.hostname in (
            "127.0.0.1",
            "localhost",
        )
        if (
            url.username
            or url.password
            or url.query
            or url.fragment
            or not url.hostname
            or (url.scheme != "https" and not (local_fixture and url.scheme == "http"))
        ):
            raise ValueError("use an HTTPS provider base URL without credentials or query")
    elif provider_kind == "openai_codex":
        if config["policy_kind"] != "remote_model" or "base_url" in config:
            raise ValueError("openai_codex uses Pi OAuth, not a provider base URL")
        auth_dir = Path(config.get("pi_auth_dir", ""))
        auth_file = auth_dir / "auth.json"
        if (
            not auth_dir.is_absolute()
            or auth_dir.is_symlink()
            or not auth_dir.is_dir()
            or auth_dir.stat().st_mode & 0o077
            or auth_file.is_symlink()
            or not auth_file.is_file()
            or auth_file.stat().st_mode & 0o077
        ):
            raise ValueError("pi_auth_dir needs private Pi openai-codex OAuth credentials")
        credential = json.loads(auth_file.read_text(encoding="utf-8")).get("openai-codex")
        if not isinstance(credential, dict) or credential.get("type") != "oauth":
            raise ValueError("pi_auth_dir needs private Pi openai-codex OAuth credentials")
    else:
        raise ValueError("unsupported Pi provider_kind")
    for name, default, upper in (
        ("max_model_calls", 12, 24),
        ("max_request_bytes", 64000, 131072),
        ("episode_timeout_s", 600, 1800),
    ):
        value = config.get(name, default)
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"invalid pilot limit: {name}")
    validate_model_settings(config, "pi")


def validate_codex(config):
    allowed = {
        "host",
        "transport",
        "python",
        "bundle",
        "session",
        "task_id",
        "job_root",
        "job_id",
        "archive_root",
        "policy_kind",
        "codex_cli",
        "auth_kind",
        "model",
        "reasoning_effort",
        "episode_timeout_s",
    }
    if set(config) - allowed:
        raise ValueError("unsupported Codex config field; secrets belong only in the environment")
    if config.get("transport") != "ssh":
        raise ValueError("native Codex pilot requires ssh transport")
    if config.get("policy_kind") not in ("remote_model", "scripted_http_fixture"):
        raise ValueError("declare policy_kind as remote_model or scripted_http_fixture")
    if config.get("auth_kind") not in ("chatgpt", "api_key"):
        raise ValueError("declare auth_kind as chatgpt or api_key")
    if config["auth_kind"] == "api_key" and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for api_key auth")
    if not isinstance(config.get("codex_cli"), str) or not config["codex_cli"].strip():
        raise ValueError("an explicit codex_cli is required")
    validate_model_settings(config, "codex")
    timeout = config.get("episode_timeout_s", 600)
    if type(timeout) is not int or not 1 <= timeout <= 1800:
        raise ValueError("invalid Codex limit: episode_timeout_s")


def transport(config, evidence):
    if config.get("transport", "ssh") == "local":
        return LocalVabench(config, evidence)
    return RemoteVabench(config, evidence)


def run_pi(config, evidence):
    from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
    from alphaapollo.workflows.chips_pi_runtime import run_pi_task

    validate_pilot(config)
    spec = agent_input(config, "pi")
    task = AgentTask(
        task_id=spec["task_id"],
        system=spec["system"],
        prompt=spec["prompt"],
        tools=tuple(tool["function"]["name"] for tool in spec["tools"]),
    )
    return run_pi_task(config, evidence, task, task_kind="vabench", transport_factory=transport)


def run_codex(config, evidence):
    from dataclasses import asdict

    from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
    from alphaapollo.reasoning.runtime.external.agents.codex import CodexSession

    validate_codex(config)
    evidence = Path(evidence).absolute()
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    config_path = evidence / "operator.json"
    if config_path.exists():
        raise ValueError("use a fresh evidence directory for each model attempt")
    atomic_json(config_path, config)
    workspace = evidence / "workspace"
    workspace.mkdir(mode=0o700, exist_ok=True)
    # Codex keeps its login in the operator's home; provider keys for other
    # experiments are never passed to this process or its MCP child.
    env = {
        name: os.environ[name]
        for name in (
            "HOME",
            "PATH",
            "CODEX_HOME",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "SSL_CERT_FILE",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "NO_PROXY",
            "https_proxy",
            "http_proxy",
            "no_proxy",
        )
        if name in os.environ
    }
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("PATH", "/usr/bin:/bin")
    if config["auth_kind"] == "api_key":
        env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        auth_state = "api_key_env_present"
    else:
        login = subprocess.run(
            [config["codex_cli"], "login", "status"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if login.returncode != 0 or "chatgpt" not in (login.stdout + login.stderr).lower():
            raise ValueError("Codex ChatGPT login is not available on this machine")
        auth_state = "chatgpt_login_verified"
    atomic_json(
        evidence / "auth-preflight.json", {"kind": config["auth_kind"], "state": auth_state}
    )

    tool_names = tuple(schema["function"]["name"] for schema in tool_schemas())
    session = CodexSession(
        cli=config["codex_cli"],
        model=config["model"],
        timeout_s=config.get("episode_timeout_s", 600),
        sandbox="read-only",
        config_overrides=(
            (f'model_reasoning_effort="{config["reasoning_effort"]}"',)
            if "reasoning_effort" in config
            else ()
        ),
        env=env,
        raw_event_path=evidence / "codex-events.jsonl",
        mcp_servers={
            "vabench": {
                "command": "/usr/bin/env",
                "args": [
                    "-i",
                    "HOME=" + str(Path.home()),
                    "PATH=/usr/bin:/bin:/usr/local/bin",
                    "PYTHONUTF8=1",
                    "PYTHONPATH=" + str(Path(__file__).resolve().parents[2]),
                    sys.executable,
                    "-m",
                    "alphaapollo.workflows.chips_vabench_agent",
                    "serve",
                    "--config",
                    str(config_path),
                    "--evidence",
                    str(evidence / "tools"),
                ],
            }
        },
        mcp_auto_approve_tools={"vabench": tool_names},
    )
    task_spec = agent_input(config, "codex")
    task = AgentTask(
        task_id=task_spec["task_id"], system=task_spec["system"], prompt=task_spec["prompt"]
    )
    try:
        outcome = session.run(task, workspace=workspace)
    finally:
        session.close()
    result = asdict(outcome)
    (evidence / "codex-outcome.json").write_text(json.dumps(result, default=str), encoding="utf-8")
    return result


def submission_confirmed(evidence):
    """Only an acknowledged public submit permits the operator final scorer."""
    for request_path in (Path(evidence) / "tools").glob("*/request.json"):
        if json.loads(request_path.read_text(encoding="utf-8")).get("tool") != "vabench_submit":
            continue
        response_path = request_path.with_name("response.json")
        if not response_path.exists():
            continue
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("ok") and response.get("result", {}).get("status") == "submitted":
            return True
    return False


def collect_result(remote, config, evidence):
    from alphaapollo.common.execution.chips.archive import verify_archive
    from alphaapollo.common.execution.chips.vabench_session import verify_episode_archive

    evidence = Path(evidence)
    job = str(Path(config["job_root"]) / config["job_id"])
    deadline = time.monotonic() + 420
    while time.monotonic() < deadline:
        try:
            status = remote.cli("job-status", job)
        except (ConnectionError, TimeoutError):
            time.sleep(2)
            continue
        if status.get("state") == "finished":
            archive = status.get("archive", {}).get("state")
            if archive == "failed":
                atomic_json(
                    evidence / "report.json",
                    {
                        "state": "archive_failed",
                        "job": status,
                        "guidance": "Use job-archive to retry archiving; do not rerun simulation.",
                    },
                )
                return False
            if archive == "verified":
                break
        time.sleep(2)
    else:
        atomic_json(evidence / "report.json", {"state": "pending", "job_id": config["job_id"]})
        return False
    archive = Path(config["archive_root"])
    for folder, names in (
        (config["job_id"], ("receipt.json", "job.tar.gz")),
        ("episodes/" + config["job_id"], ("receipt.json", "episode.tar.gz")),
    ):
        for name in names:
            remote.download(archive / folder / name, evidence / "archives" / folder / name)
    final = verify_archive(evidence / "archives" / config["job_id"])
    public = verify_episode_archive(evidence / "archives/episodes" / config["job_id"])
    if final["request"]["identity"]["candidate"] != public["candidate"]:
        raise ValueError("final scorer used a different candidate from the public episode")
    agent = {"kind": "scripted_tools", "real_model": False}
    if (evidence / "operator.json").exists():
        operator = json.loads((evidence / "operator.json").read_text(encoding="utf-8"))
        kind = "codex" if "codex_cli" in operator else "pi"
        outcome_path = evidence / f"{kind}-outcome.json"
        outcome = (
            json.loads(outcome_path.read_text(encoding="utf-8")) if outcome_path.exists() else None
        )
        agent = {
            "kind": kind,
            "policy_kind": operator.get("policy_kind", "unclassified"),
            "real_model_requested": operator.get("policy_kind") == "remote_model",
            "model": operator.get("model"),
            "termination_reason": outcome.get(
                "harness_termination_reason", outcome["termination_reason"]
            )
            if outcome
            else "operator_failed_after_submission",
            "last_message_usage": outcome.get("usage", {}) if outcome else {},
        }
    report = {
        "state": "verified",
        "agent": agent,
        "task_id": config.get("task_id"),
        "candidate": public["candidate"],
        "result": final["completion"]["result"],
        "public_trace_sha256": public["sha256"],
        "final_archive_sha256": final["package"]["sha256"],
        "model_cost": {"status": "not_measured"},
    }
    atomic_json(evidence / "report.json", report)
    return report["result"]["execution"] == "ok"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("serve", "script", "pi", "codex", "detach", "finalize", "collect", "recover"),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--script", type=Path)
    parser.add_argument("--action-id")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.mode == "detach":
        validate_pilot(config)
        if config.get("provider_kind", "glm_coding") == "glm_coding" and not os.environ.get(
            "CHIPS_MODEL_KEY"
        ):
            raise ValueError("CHIPS_MODEL_KEY is required in the launcher environment")
        args.evidence.mkdir(mode=0o700, parents=True, exist_ok=False)
        child_env = {**os.environ, "PYTHONUTF8": "1"}
        with (args.evidence / "operator.log").open("wb") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "alphaapollo.workflows.chips_vabench_agent",
                    "pi",
                    "--config",
                    str(args.config),
                    "--evidence",
                    str(args.evidence),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_env,
                start_new_session=True,
                close_fds=True,
            )
        atomic_json(args.evidence / "launcher.json", {"pid": child.pid, "state": "started"})
        print(json.dumps({"pid": child.pid, "evidence": str(args.evidence)}))
        return 0
    if args.mode == "serve":
        serve(config, args.evidence)
        return 0
    if args.mode in ("pi", "codex"):
        (validate_pilot if args.mode == "pi" else validate_codex)(config)
        chips_vabench_record.begin(args.evidence, config, args.mode, agent_input(config, args.mode))
    recording = (args.evidence / "task_manifest.json").is_file()
    if recording and args.mode not in ("pi", "codex"):
        chips_vabench_record.require_operator(args.evidence, config)
    try:
        remote = transport(config, args.evidence / "tools")
        if args.mode == "recover":
            if args.action_id is None:
                parser.error("--action-id is required")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.action_id):
                parser.error("invalid --action-id")
            request = json.loads(
                (args.evidence / "tools" / args.action_id / "request.json").read_text(
                    encoding="utf-8"
                )
            )
            reply = remote.call(request["tool"], request["arguments"], action_id=request["id"])
            print(json.dumps(reply))
            return 0 if reply["ok"] else 1
        if args.mode == "collect":
            return 0 if collect_result(remote, config, args.evidence) else 1
        if args.mode == "script":
            if args.script is None:
                parser.error("--script is required")
            for request in json.loads(args.script.read_text(encoding="utf-8")):
                reply = remote.call(request["tool"], request["arguments"], action_id=request["id"])
                if not reply["ok"]:
                    print(json.dumps(reply))
                    return 1
        elif args.mode in ("pi", "codex"):
            preflight = remote.cli("vabench-preflight", "--session", config["session"], timeout=60)
            atomic_json(args.evidence / "preflight.json", preflight)
            if preflight.get("state") != "ready":
                chips_vabench_record.update(
                    args.evidence, status="blocked", error_type="preflight_not_ready"
                )
                raise ValueError("public environment preflight did not pass")
            if not chips_vabench_record.confirm_task(args.evidence, preflight):
                chips_vabench_record.update(
                    args.evidence, status="blocked", error_type="task_identity_mismatch"
                )
                raise ValueError("server task ID does not match the operator config")
            atomic_json(
                args.evidence / "episode-budget.json",
                {
                    "episode_timeout_s": config.get("episode_timeout_s", 600),
                    "model_settings": (
                        {"model": config["model"], "thinking": config.get("thinking")}
                        if args.mode == "pi"
                        else {
                            "model": config["model"],
                            "reasoning_effort": config.get("reasoning_effort"),
                        }
                    ),
                    "experiment_settings": experiment_settings_snapshot(config, args.mode),
                    "server_limits": preflight.get("limits"),
                    "model_requests": {
                        "limit": config.get("max_model_calls", 12) if args.mode == "pi" else None,
                        "enforced": args.mode == "pi",
                    },
                    "request_bytes": {
                        "limit": config.get("max_request_bytes", 64000)
                        if args.mode == "pi"
                        else None,
                        "enforced": args.mode == "pi",
                    },
                    "output_tokens_per_request": {
                        "limit": config.get("max_output_tokens", 4096)
                        if args.mode == "pi"
                        else None,
                        "enforced": args.mode == "pi",
                    },
                },
            )
            chips_vabench_record.update(args.evidence, status="running")
            outcome = (
                run_pi(config, args.evidence)
                if args.mode == "pi"
                else run_codex(config, args.evidence)
            )
            if not submission_confirmed(args.evidence):
                atomic_json(
                    args.evidence / "report.json",
                    {
                        "state": "unsubmitted",
                        "termination_reason": outcome.get(
                            "harness_termination_reason", outcome["termination_reason"]
                        ),
                        "guidance": "Inspect agent and public actions; no final scorer started.",
                    },
                )
                return 1
        # Finalization is a separate operator action after the model exits.
        reply = remote.cli(
            "vabench-finalize",
            "--session",
            config["session"],
            "--root",
            config["job_root"],
            "--job-id",
            config["job_id"],
            "--archive-root",
            config["archive_root"],
            timeout=60,
        )
        atomic_json(args.evidence / "final-submission.json", reply)
        print(json.dumps(reply))
        return 0 if collect_result(remote, config, args.evidence) else 1
    finally:
        if recording and args.mode != "recover":
            error = sys.exc_info()[0]
            chips_vabench_record.reconcile(
                args.evidence, exception_type=error.__name__ if error else None
            )


if __name__ == "__main__":
    raise SystemExit(main())
