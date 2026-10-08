"""Pi and MCP entry point for the bounded Analog RLC public session."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib import error, request
from urllib.parse import urlsplit

from alphaapollo.common.execution.chips.analog_design_bench import TASKS, rlc_contract
from alphaapollo.common.execution.chips.analog_public import TASK_ID
from alphaapollo.common.execution.chips.analog_remote import LocalAnalog, RemoteAnalog
from alphaapollo.common.execution.chips.analog_session import (
    session_info,
    session_task_id,
    tool_schemas,
)
from alphaapollo.common.execution.chips.journal import atomic_json
from alphaapollo.workflows.chips_experiment_settings import (
    experiment_settings_snapshot,
    validate_model_settings,
)

# Keep experiment behavior separate from the current session's tool and grading
# protocol. Circuit specifications remain in the public task files.
_EXPERIMENT_RULES = """You are the circuit-design agent for one controlled experiment.
Use the task instructions and tool contracts to identify the deliverable,
permitted devices, units, and performance constraints.

Create a runnable candidate early. Briefly state the purpose of a design change
or simulation, then inspect the returned result before making a dependent
change. Use circuit knowledge to propose revisions and observed evidence to
support claims about their effects.

Distinguish simulator execution, measurement completeness, and satisfaction of
the specification. Compare all constraints and trade-offs; one improved metric
does not establish overall improvement. Do not invent measurements or treat
missing evidence as zero. Report uncertainty when execution or results are
unknown.

Follow runtime budget notices and tool-reported budgets. End with a brief
summary of the deliverable state, artifact identity if known, observed
performance, and unmet or unverified requirements. Claim only results supported
by the feedback actually available to you."""

_ANALOG_PROTOCOL = """Use only the six advertised Analog task tools. Read the public
instruction, starter, and testbenches with analog_read. Write a bounded passive
RLC candidate with analog_write and inspect public feedback from analog_simulate
before deciding whether to revise it. The task files define the circuit and
numerical requirements; tool schemas define accepted arguments.

Track the current candidate by its returned hash. Use analog_history to compare
same-session public feedback and analog_restore to recover a saved candidate
when useful. History order is not a quality ranking. The harness collects the
current candidate, not an automatically selected historical best.

Leave enough model and action budget to select a candidate, restore it if needed,
and call analog_submit. Prefer a candidate with successful public simulation;
identify any untested final edits. If an action's execution is pending or unknown,
stop issuing new task actions and report it for controller recovery. Do not
retry it as a new action.

Confirm submission from the analog_submit response, then stop calling task
tools. Automatic collection is not Agent submission. Public diagnostics are
not a final correctness score; the hidden final evaluator is unavailable.
Do not claim a final score or certified task success."""


def validate_pilot(config: dict) -> None:
    if isinstance(config, dict) and {"archive_root", "final_output"} & set(config):
        raise ValueError(
            "unsupported pilot config field: archive_root and final_output belong "
            "in the experiment plan, not the Analog operator"
        )
    allowed = {
        "task_id",
        "host",
        "transport",
        "python",
        "bundle",
        "session",
        "pi_cli",
        "runtime",
        "http_retry_limit",
        "base_url",
        "model",
        "thinking",
        "policy_kind",
        "max_model_calls",
        "max_request_bytes",
        "max_output_tokens",
        "episode_timeout_s",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("unsupported pilot config field; secrets belong only in the environment")
    pilot_task_id(config)
    if config.get("transport") not in ("local", "ssh"):
        raise ValueError("transport must be local or ssh")
    if config["transport"] == "ssh" and not isinstance(config.get("host"), str):
        raise ValueError("SSH transport requires a host alias")
    if config.get("policy_kind") not in ("remote_model", "scripted_http_fixture"):
        raise ValueError("declare the model policy kind")
    for name in ("python", "bundle", "session", "pi_cli"):
        if not isinstance(config.get(name), str) or not Path(config[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    from alphaapollo.workflows.chips_pi_runtime import validate_pi_runtime

    validate_pi_runtime(config)
    validate_model_settings(config, "pi")
    url = urlsplit(config["base_url"])
    fixture = config["policy_kind"] == "scripted_http_fixture" and url.hostname in (
        "127.0.0.1",
        "localhost",
    )
    if (
        url.username
        or url.password
        or url.query
        or url.fragment
        or not url.hostname
        or (url.scheme != "https" and not (fixture and url.scheme == "http"))
    ):
        raise ValueError("use an HTTPS model URL without credentials or query")
    for name, default, upper in (
        ("max_model_calls", 12, 24),
        ("max_request_bytes", 64000, 131072),
        ("episode_timeout_s", 600, 1800),
    ):
        value = config.get(name, default)
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"invalid pilot limit: {name}")


def pilot_task_id(config: dict) -> str:
    task_id = config.get("task_id", TASK_ID)
    if not isinstance(task_id, str):
        raise ValueError("task_id must name a supported public RLC task")
    rlc_contract(task_id)
    return task_id


def _bound_session(config: dict, evidence: Path) -> dict:
    info = (
        session_info(Path(config["session"]))
        if config["transport"] == "local"
        else transport(config, evidence / "binding").cli(
            "analog-info", "--session", config["session"]
        )
    )
    if info.get("task_id") != pilot_task_id(config):
        raise ValueError("Analog operator task_id must match session task_id")
    return info


def transport(config: dict, evidence: Path):
    return (
        LocalAnalog(config, evidence)
        if config["transport"] == "local"
        else RemoteAnalog(config, evidence)
    )


def serve(config: dict, evidence: Path) -> None:
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    _bound_session(config, evidence)
    remote = transport(config, evidence)
    server = Server("chips-analog-public")
    lock = anyio.Lock()

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                inputSchema=tool["function"]["parameters"],
            )
            for tool in tool_schemas()
        ]

    @server.call_tool()
    async def call_tool(name, arguments):
        async with lock:
            result = await anyio.to_thread.run_sync(remote.call, name, arguments)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result))],
            isError=not result.get("ok", False),
        )

    async def run():
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    anyio.run(run)


def run_pi(config: dict, evidence: Path) -> dict:
    from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
    from alphaapollo.workflows.chips_pi_runtime import run_pi_task

    validate_pilot(config)
    binding = _bound_session(config, evidence)
    task = AgentTask(
        task_id=binding["task_id"],
        tools=tuple(tool["function"]["name"] for tool in tool_schemas()),
        system=(
            "Experiment rules:\n"
            + _EXPERIMENT_RULES
            + "\n\nAnalog public-session protocol:\n"
            + _ANALOG_PROTOCOL
        ),
        prompt="Complete the public Analog RLC task. Start by listing public files.",
    )
    return run_pi_task(config, evidence, task, task_kind="analog", transport_factory=transport)


def submission_confirmed(evidence: Path) -> bool:
    for request_path in (Path(evidence) / "tools").glob("*/request.json"):
        if json.loads(request_path.read_text(encoding="utf-8")).get("tool") != "analog_submit":
            continue
        response_path = request_path.with_name("response.json")
        if not response_path.exists():
            continue
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("ok") and response.get("result", {}).get("state") == "submitted":
            return True
    return False


def load_private_key(path: Path) -> str:
    """Read a per-user key without putting its value in a command argument."""
    path = Path(path).absolute()
    parent = path.parent
    directory = parent.stat()
    if (
        parent.is_symlink()
        or not stat.S_ISDIR(directory.st_mode)
        or directory.st_uid != os.getuid()
        or stat.S_IMODE(directory.st_mode) != 0o700
    ):
        raise ValueError("private key file requires an owned 0700 parent directory")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("private key file must be an owned regular 0600 file")
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError("private key file exceeds size limit")
    key = raw.decode("utf-8").strip()
    if not key or any(character.isspace() for character in key):
        raise ValueError("private key file must contain one nonempty token")
    return key


def preflight_pilot(config: dict, *, key_file: Path | None = None) -> dict:
    """Check a server-local Analog pilot without calling a model or simulator."""
    from alphaapollo.common.execution.chips.analog_design_bench import tree_digest
    from alphaapollo.common.execution.chips.journal import file_digest

    validate_pilot(config)
    if config["transport"] != "local":
        raise ValueError("Analog preflight currently requires server-local transport")
    checks: dict[str, str] = {}
    if key_file is not None and os.environ.get("CHIPS_MODEL_KEY"):
        raise ValueError("choose either a private key file or an environment key")
    if key_file is None:
        checks["credential"] = "ready" if os.environ.get("CHIPS_MODEL_KEY") else "missing"
    else:
        try:
            load_private_key(key_file)
            checks["credential"] = "ready"
        except (OSError, UnicodeError, ValueError):
            checks["credential"] = "invalid"

    python = Path(config["python"])
    checks["python"] = "ready" if python.is_file() and os.access(python, os.X_OK) else "missing"
    checks["pi_cli"] = (
        "ready"
        if Path(config["pi_cli"]).is_file() and os.access(config["pi_cli"], os.X_OK)
        else "missing"
    )
    checks["bundle"] = "ready" if zipfile.is_zipfile(config["bundle"]) else "invalid"

    process_env = {
        name: value
        for name, value in os.environ.items()
        if name in {"PATH", "HOME", "USER", "LOGNAME", "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME"}
    }
    try:
        dependencies = subprocess.run(
            [config["python"], "-c", "import mcp, anyio"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            env=process_env,
            check=False,
        )
        checks["python_deps"] = "ready" if dependencies.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        checks["python_deps"] = "unavailable"

    session = Path(config["session"])
    try:
        details = json.loads((session / "session.json").read_text(encoding="utf-8"))
        task_id = session_task_id(details)
        public_files = rlc_contract(task_id).public_files
        checks["session"] = (
            "ready"
            if task_id == pilot_task_id(config) and not (session / "frozen.json").exists()
            else "unavailable"
        )
        checks["public_files"] = (
            "ready"
            if set(details["public_files"]) == set(public_files)
            and all(
                not (session / "public" / name).is_symlink()
                and file_digest(session / "public" / name) == digest
                for name, digest in details["public_files"].items()
            )
            else "changed"
        )
        source = Path(details["source_root"]) / "tasks" / task_id
        checks["source_pin"] = (
            "ready" if tree_digest(source) == TASKS[task_id].source_sha256 else "changed"
        )
    except (OSError, ValueError, KeyError, TypeError):
        checks.setdefault("session", "unavailable")
        checks.setdefault("public_files", "unavailable")
        checks.setdefault("source_pin", "unavailable")
        details = None

    try:
        if details is None:
            raise ValueError("session unavailable")
        command = [details["podman"]]
        if details["podman_root"] is not None:
            command += ["--root", details["podman_root"], "--runroot", details["podman_runroot"]]
        command += ["image", "exists", details["runtime_image"] or TASKS[task_id].image]
        image = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            env=process_env,
            check=False,
        )
        checks["podman_image"] = "ready" if image.returncode == 0 else "unavailable"
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
        checks["podman_image"] = "unavailable"

    try:
        # Pi receives no proxy environment, so test the same direct route.
        opener = request.build_opener(request.ProxyHandler({}))
        with opener.open(request.Request(config["base_url"], method="HEAD"), timeout=5):
            pass
        checks["model_https"] = "ready"
    except error.HTTPError as response:
        checks["model_https"] = "ready" if 400 <= response.code < 500 else "unavailable"
    except (OSError, ValueError):
        checks["model_https"] = "unavailable"
    return {
        "state": "ready" if all(value == "ready" for value in checks.values()) else "blocked",
        "checks": checks,
        "model_settings": {
            "model": config["model"],
            "thinking": config.get("thinking"),
            "max_model_calls": config.get("max_model_calls", 12),
            "max_output_tokens": config.get("max_output_tokens", 4096),
        },
        "experiment_settings": experiment_settings_snapshot(config, "pi"),
        "model_auth": "not_tested",
        "simulator_run": "not_tested",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "pi", "detach", "preflight"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_pilot(config)
    if args.mode == "preflight":
        result = preflight_pilot(config, key_file=args.key_file)
        print(json.dumps(result))
        return 0 if result["state"] == "ready" else 1
    if args.evidence is None:
        parser.error("--evidence is required for serve, pi and detach")
    if args.key_file is not None:
        if args.mode == "serve" or os.environ.get("CHIPS_MODEL_KEY"):
            raise ValueError("private key file is only for Pi launch without an environment key")
        os.environ["CHIPS_MODEL_KEY"] = load_private_key(args.key_file)
    if args.mode == "detach":
        if not os.environ.get("CHIPS_MODEL_KEY"):
            raise ValueError("CHIPS_MODEL_KEY is required in the launcher environment")
        args.evidence.mkdir(mode=0o700, parents=True, exist_ok=False)
        with (args.evidence / "operator.log").open("wb") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "alphaapollo.workflows.chips_analog_agent",
                    "pi",
                    "--config",
                    str(args.config),
                    "--evidence",
                    str(args.evidence),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUTF8": "1"},
                start_new_session=True,
                close_fds=True,
            )
        atomic_json(args.evidence / "launcher.json", {"pid": child.pid, "state": "started"})
        print(json.dumps({"pid": child.pid, "evidence": str(args.evidence)}))
        return 0
    if args.mode == "serve":
        serve(config, args.evidence)
        return 0
    outcome = run_pi(config, args.evidence)
    report = finish_episode(config, args.evidence, outcome)
    state = report["state"]
    atomic_json(
        args.evidence / "report.json",
        {
            **report,
            "policy_kind": config["policy_kind"],
            "model": config["model"],
            "termination_reason": outcome.get(
                "harness_termination_reason", outcome["termination_reason"]
            ),
            "final_score": "not_run",
        },
    )
    print(json.dumps({"state": state, "final_score": "not_run"}))
    return 0 if state in {"submitted", "collected"} else 1


def finish_episode(config: dict, evidence: Path, outcome: dict) -> dict:
    """Collect through the operator channel only after a known Agent termination."""
    if submission_confirmed(evidence):
        return {"state": "submitted", "agent_submitted": True, "collection_source": "agent_submit"}
    reason = outcome.get("harness_termination_reason", outcome["termination_reason"])
    reason = {"timeout": "deadline", "truncated": "output_token_limit"}.get(reason, reason)
    from alphaapollo.common.execution.chips.analog_session import END_REASONS

    if reason not in END_REASONS:
        return {"state": "unsubmitted", "agent_submitted": submission_confirmed(evidence)}
    try:
        receipt = transport(config, evidence / "tools").cli(
            "analog-close", "--session", config["session"], "--reason", reason, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {"state": "awaiting_action_recovery", "termination_reason": reason}
    atomic_json(evidence / "collection.json", receipt)
    return receipt


if __name__ == "__main__":
    raise SystemExit(main())
