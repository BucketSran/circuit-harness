#!/usr/bin/env python3
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

"""Run one task through several external coding agents and compare the results.

This is the live counterpart to the offline example: it drives the real CLIs
through :class:`ExternalAgentRuntime` and prints what each one actually did, so
the trajectory contract can be checked against agents that behave differently.

    python -m alphaapollo.reasoning.runtime.external.smoke --agents claude_code,codex
    python -m alphaapollo.reasoning.runtime.external.smoke --agents codex --out runs/smoke
    python -m alphaapollo.reasoning.runtime.external.smoke --agents claude_code --bridged

``--bridged`` swaps each agent's native shell for AlphaApollo's own tools, which
additionally needs rootless podman and the ``mcp`` extra. The ``via`` column
then shows which tools were really called. Claude Code can be left with nothing
but the bridge; native Codex keeps its own execution tools; and pi-hosted runs,
including ``codex_via_pi``, reach the bridge through a bundled extension and see
the tools under their plain AlphaApollo names.

Requires the selected CLIs on PATH and working credentials for each. Every agent
runs in its own temporary workspace; nothing is written to the repository.

The expected answer is compared *after* the run and is never shown to an agent —
it is grading input, not model input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from shutil import which
from typing import Any

from alphaapollo.reasoning.runtime.agent_runtime import AgentResult, AgentTask
from alphaapollo.reasoning.runtime.external import alphaapollo_bridge, session_factory
from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalAgentRuntime

# A small execution task tests the agent/bridge path without a domain benchmark.
PROBLEM = (
    'Create sample.json containing {"value": 42} in the current workspace. '
    "Read it with python3 and report the stored value."
)
SYSTEM_PROMPT = (
    "Use your shell tool to create and read the file. "
    "End your reply with exactly: Final answer: <value>"
)
EXPECTED_ANSWER = "42"

#: CLI binary and session options per agent. Tool access is granted narrowly:
#: each agent only needs a shell to create and read its own fixture.
AGENTS: dict[str, dict[str, Any]] = {
    "claude_code": {
        "binary": "claude",
        "session": {
            "model": "sonnet",
            "tools": ["Bash"],
            "permission_mode": "bypassPermissions",
            "safe_mode": True,
        },
    },
    "codex": {
        "binary": "codex",
        "session": {"sandbox": "workspace-write"},
    },
    "codex_via_pi": {
        "binary": "pi",
        "session": {
            "provider": "openai-codex",
            "tool_mode": "allowlist",
            "tools": ["bash"],
        },
    },
    "pi": {
        "binary": "pi",
        "session": {"tool_mode": "allowlist", "tools": ["bash"]},
    },
}

#: What each agent needs so the bridge is reachable at all. Claude Code loses
#: --safe-mode because it would disable MCP servers; Codex explicitly approves
#: only the AlphaApollo bridge's bash tool while retaining workspace isolation;
#: pi loses its default tool mode, which would leave its own tools beside ours.
BRIDGED_SESSIONS: dict[str, dict[str, Any]] = {
    "claude_code": {"tools": [], "safe_mode": False, "permission_mode": "bypassPermissions"},
    "codex": {
        "sandbox": "workspace-write",
        "mcp_auto_approve_tools": {"alphaapollo": ["bash"]},
    },
    "codex_via_pi": {"provider": "openai-codex", "tool_mode": "no_builtin", "tools": []},
    "pi": {"tool_mode": "no_builtin", "tools": []},
}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    selected = [name.strip() for name in arguments.agents.split(",") if name.strip()]
    unknown = [name for name in selected if name not in AGENTS]
    if unknown:
        print(f"unknown agents: {unknown}; available: {sorted(AGENTS)}", file=sys.stderr)
        return 2

    task = AgentTask(
        task_id="file-roundtrip-external-smoke",
        system=SYSTEM_PROMPT,
        prompt=PROBLEM,
    )

    rows: list[dict[str, Any]] = []
    for name in selected:
        rows.append(
            _run_agent(
                name,
                task,
                timeout_s=arguments.timeout,
                out=arguments.out,
                bridged=arguments.bridged,
            )
        )

    _print_table(rows)
    failures = [row for row in rows if row["status"] != "ok"]
    if failures:
        print(f"\n{len(failures)} of {len(rows)} agent(s) did not produce a usable result.")
        return 1
    return 0


def _run_agent(
    name: str,
    task: AgentTask,
    *,
    timeout_s: float,
    out: Path | None,
    bridged: bool = False,
) -> dict[str, Any]:
    spec = AGENTS[name]
    binary = spec["binary"]
    if which(binary) is None:
        print(f"[{name}] skipped: {binary!r} is not on PATH")
        return {"agent": name, "status": "missing_cli"}

    options = {**spec["session"], "timeout_s": timeout_s}
    if bridged:
        options.update(BRIDGED_SESSIONS[name])
        options["mcp_servers"] = {
            "alphaapollo": alphaapollo_bridge(tool_ids=("bash",), timeout_s=60.0)
        }
    runtime = ExternalAgentRuntime(
        session_factory(name, options), agent=name, model=str(options.get("model", "default"))
    )
    print(f"[{name}] running (timeout {timeout_s:.0f}s) ...")
    started = time.monotonic()
    try:
        result = runtime.run(task)
    except Exception as exc:  # noqa: BLE001 - one agent must not abort the comparison
        print(f"[{name}] failed: {type(exc).__name__}: {exc}")
        return {"agent": name, "status": "error", "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        runtime.terminate()
    elapsed = time.monotonic() - started

    if out is not None:
        destination = out / f"{name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(_summary(result), indent=2, default=str))
        print(f"[{name}] trajectory summary written to {destination}")

    calls = [call for turn in result.turns for call in turn.generation_response.tool_calls]
    return {
        "agent": name,
        "status": "ok" if result.termination_reason == "final" else result.termination_reason,
        "turns": len(result.turns),
        "tool_calls": len(calls),
        # Which tools were really used, which is the whole question in bridged
        # mode: an agent that kept its own shell says so here.
        "via": ",".join(dict.fromkeys(call.name for call in calls)) or "-",
        "correct": EXPECTED_ANSWER in result.final_text,
        "trainable": any(turn.generation_response.is_trainable for turn in result.turns),
        "seconds": round(elapsed, 1),
        "final_line": _final_line(result.final_text),
    }


def _summary(result: AgentResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "termination_reason": result.termination_reason,
        "metadata": dict(result.metadata),
        "final_text": result.final_text,
        "turns": [
            {
                "index": turn.index,
                "text": turn.generation_response.content,
                "reasoning": turn.generation_response.reasoning_content,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in turn.generation_response.tool_calls
                ],
                "usage": dict(turn.generation_response.usage),
                "observation": turn.environment_transition.observation.to_dict(),
                "done": turn.environment_transition.done,
                "termination_reason": turn.environment_transition.termination_reason,
            }
            for turn in result.turns
        ],
    }


def _final_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:60] if lines else ""


def _print_table(rows: Sequence[dict[str, Any]]) -> None:
    print()
    header = (
        f"{'agent':<14}{'status':<15}{'turns':>6}{'tools':>7}{'correct':>9}{'sec':>7}  "
        f"{'via':<26}final line"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["status"] in {"missing_cli", "error"}:
            print(
                f"{row['agent']:<14}{row['status']:<15}{'-':>6}{'-':>7}{'-':>9}{'-':>7}  "
                f"{'-':<26}{row.get('detail', '')[:60]}"
            )
            continue
        print(
            f"{row['agent']:<14}{row['status']:<15}{row['turns']:>6}{row['tool_calls']:>7}"
            f"{str(row['correct']):>9}{row['seconds']:>7}  {row['via'][:24]:<26}{row['final_line']}"
        )
    print()
    print(f"expected answer: {EXPECTED_ANSWER} (compared after the run, never shown to an agent)")
    trainable = [row["agent"] for row in rows if row.get("trainable")]
    if trainable:
        print(f"WARNING: {trainable} reported trainable turns; external text must never be.")
    else:
        print("all external turns are correctly marked non-trainable")


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m alphaapollo.reasoning.runtime.external.smoke",
        description="Run one task through several external coding agents and compare them.",
    )
    parser.add_argument(
        "--agents",
        default="claude_code,codex",
        help=f"comma-separated agents to run; available: {','.join(sorted(AGENTS))}",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="per-agent timeout in seconds")
    parser.add_argument(
        "--bridged",
        action="store_true",
        help="give each agent AlphaApollo's tools over MCP instead of its own",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="directory for per-agent trajectory summaries"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
