"""Native Codex phases and MCP tools for the bounded gain-authoring pilot."""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from alphaapollo.common.execution.chips.authoring_session import tool_schemas
from alphaapollo.common.execution.chips.journal import atomic_json, file_digest
from alphaapollo.common.execution.chips.session_transport import RemoteSessionTransport
from alphaapollo.common.execution.chips.spectre_testbench import REQUIREMENTS
from alphaapollo.common.execution.chips.task_authoring import inspect_draft, require_confirmation


class RemoteGain(RemoteSessionTransport):
    """Reuse the existing deduplicated SSH request/response and local trace transport."""

    request_command = "gain-request"
    response_command = "gain-response"


def phase_input(phase, config):
    materials = Path(config["materials"])
    if phase == "build":
        if not config.get("confirmation"):
            raise ValueError("operator confirmation is required before model construction")
        draft = json.loads(Path(config["draft"]).read_text())
        receipt = json.loads(Path(config["confirmation"]).read_text())
        values = require_confirmation(draft, materials, REQUIREMENTS, receipt)
        return {
            "system": "Construct a differential voltage-gain testbench using the two gain MCP "
            "tools. The operator has confirmed the conditions. Choose AC excitations and "
            "voltage-ratio nodes, run public validation, inspect the feedback, revise if needed, "
            "then explicitly submit. Do not change conditions or access other files/tools. "
            "The candidate is bounded JSON; the harness renders Spectre syntax. "
            "A correct measurement can identify a DUT that fails the specification. "
            "The public fixture is a linear single-pole gain model. No PDK is used.",
            "prompt": "Confirmed task conditions: "
            + json.dumps(values)
            + "\nMeasure gain at the confirmed frequency and compare it with the minimum gain. "
            "Only gain_simulate and gain_submit are part of this task. "
            "Three simulations maximum; independent evaluation follows the frozen submission.",
            "images": [],
            "tools": tool_schemas(),
            "confirmation": receipt,
        }
    if phase != "extract":
        raise ValueError("unknown authoring phase")
    sources = []
    for name in (config["notes"], config["image"]):
        path = materials / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or path.is_symlink()
            or not path.is_file()
            or materials.resolve() not in path.resolve().parents
        ):
            raise ValueError("material must be an existing relative regular file")
        sources.append({"path": name, "sha256": file_digest(path)})
    return {
        "system": "Read the supplied text and attached image as source material. Extract a task "
        "draft; do not solve or simulate. Return ONLY a JSON object, no Markdown fences. "
        "Never invent an unknown value. Status is explicit, inferred, missing or conflict. "
        "Every non-missing field needs evidence with source filename, page (1 here), and region "
        "describing its location. Inferred fields also need a note. Convert numbers into canonical "
        "units. This is a synthetic engineering fixture, not a request for expert approval.",
        "prompt": "Return {schema_version:1, task_id:"
        + json.dumps(config["task_id"])
        + ", sources:"
        + json.dumps(sources)
        + ", fields:{...}}. "
        "Each required field is {value,unit,status,evidence:[{source,page,region}]}; "
        "canonical field/unit mapping: "
        + json.dumps(REQUIREMENTS)
        + ". dut_ports is an ordered list of five node names. "
        "Use null value for missing/conflicting fields. The attached image is "
        + config["image"]
        + ". Text source follows:\n"
        + (materials / config["notes"]).read_text(),
        "images": [str((materials / config["image"]).absolute())],
        "tools": [],
        "sources": sources,
    }


def run_phase(phase, config, evidence):
    """Run one bounded phase; never approve a draft or retry a model episode implicitly."""
    from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
    from alphaapollo.reasoning.runtime.external.agents.codex import CodexSession

    spec = phase_input(phase, config)
    if config.get("reasoning_effort") not in {"low", "medium", "high", "xhigh"}:
        raise ValueError("reasoning_effort must be explicitly configured")
    if not config.get("model") or not config.get("codex_cli"):
        raise ValueError("model and codex_cli must be explicitly configured")
    evidence = Path(evidence).absolute()
    evidence.mkdir(mode=0o700, parents=True, exist_ok=False)
    workspace = evidence / "workspace"
    workspace.mkdir(mode=0o700)
    atomic_json(evidence / "operator.json", config)
    atomic_json(evidence / "agent-input.json", spec)
    servers, approvals = {}, {}
    if phase == "build":
        servers = {
            "gain": {
                "command": "/usr/bin/env",
                "args": [
                    "-i",
                    "HOME=" + str(Path.home()),
                    "PATH=/usr/bin:/bin:/usr/local/bin",
                    "PYTHONUTF8=1",
                    "PYTHONPATH=" + str(Path(__file__).resolve().parents[2]),
                    sys.executable,
                    "-m",
                    "alphaapollo.workflows.chips_task_authoring_agent",
                    "serve",
                    "--config",
                    str(evidence / "operator.json"),
                    "--evidence",
                    str(evidence / "tools"),
                ],
            }
        }
        approvals = {"gain": [s["function"]["name"] for s in tool_schemas()]}
    timeout = 360 if phase == "extract" else 840
    session = CodexSession(
        cli=config["codex_cli"],
        model=config["model"],
        timeout_s=timeout,
        sandbox="read-only",
        config_overrides=[f'model_reasoning_effort="{config["reasoning_effort"]}"'],
        extra_args=[arg for path in spec["images"] for arg in ("--image", path)],
        mcp_servers=servers,
        mcp_auto_approve_tools=approvals,
        raw_event_path=evidence / "codex-events.jsonl",
    )
    atomic_json(
        evidence / "runtime.json",
        {
            "argv": session.argv(),
            "timeout_s": timeout,
            "system_prompt_mode": "inlined_user_prompt",
            "native_tools": "available",
            "isolation": "cooperative_development",
            "model_retries": 0,
        },
    )
    started = time.monotonic()
    try:
        outcome = session.run(
            AgentTask(task_id=config["task_id"], system=spec["system"], prompt=spec["prompt"]),
            workspace=workspace,
        )
    finally:
        session.close()
        atomic_json(evidence / "timing.json", {"elapsed_s": time.monotonic() - started})
    atomic_json(evidence / "outcome.json", asdict(outcome))
    if phase == "extract":
        draft = json.loads(outcome.final_text)
        atomic_json(evidence / "draft.json", draft)
        report = inspect_draft(draft, config["materials"], REQUIREMENTS)
        atomic_json(evidence / "draft-review.json", report)
        return report
    submissions = []
    for request_path in (evidence / "tools").glob("*/request.json"):
        response_path = request_path.with_name("response.json")
        if (
            json.loads(request_path.read_text()).get("tool") == "gain_submit"
            and response_path.exists()
        ):
            response = json.loads(response_path.read_text())
            if response.get("ok") and response.get("result", {}).get("status") == "submitted":
                submissions.append(response)
    result = {
        "status": "submitted" if submissions else "not_submitted",
        "termination_reason": outcome.termination_reason,
        "submissions": submissions,
    }
    atomic_json(evidence / "phase-result.json", result)
    return result


def serve(config, evidence):
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    remote = RemoteGain(config["remote"], evidence)
    server = Server("chips-gain-authoring")
    lock = anyio.Lock()

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=s["function"]["name"],
                description=s["function"]["description"],
                inputSchema=s["function"]["parameters"],
            )
            for s in tool_schemas()
        ]

    @server.call_tool()
    async def call_tool(name, arguments):
        async with lock:
            result = await anyio.to_thread.run_sync(remote.call, name, arguments)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result))],
            isError=not result["ok"],
        )

    async def run():
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    anyio.run(run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("extract", "build", "serve"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    config = json.loads(args.config.read_text())
    if args.phase == "serve":
        serve(config, args.evidence)
        return 0
    result = run_phase(args.phase, config, args.evidence)
    print(json.dumps(result))
    return int(result["status"] not in {"ready_for_confirmation", "submitted"})


if __name__ == "__main__":
    raise SystemExit(main())
