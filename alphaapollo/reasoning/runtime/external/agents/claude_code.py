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

"""Claude Code as an external agent session.

The CLI is driven headlessly (``claude -p --output-format stream-json``) with the
prompt on stdin, which keeps long prompts off the argument vector and leaves the
variadic flags unambiguous.  Only the parsing lives here; turning events into a
trajectory is :mod:`alphaapollo.reasoning.runtime.external_agent_runtime`'s job.

Reproducibility is enforced by construction, not by convention: ``--safe-mode``
drops the operator's CLAUDE.md, hooks, plugins, skills, and MCP servers, and
``--no-session-persistence`` keeps runs from leaking into the resume history.
A run that skips those flags silently inherits whatever the developer machine
happens to have configured.

``mcp_servers`` is the bridged path (see
``docs/reasoning/external-agent-tool-bridge.md``).  Its tools reach the model as
``mcp__<server>__<tool>``, which ``--tools`` does not govern -- that flag
selects built-ins -- so a run wanting *only* bridged tools passes ``tools=[]``
and grants the MCP ones through ``permission_mode``.  ``--safe-mode`` disables
MCP servers *including* the ones named on the command line, so the two cannot be
combined; a bridged run sets ``--setting-sources ""`` instead, which is as much
isolation as the CLI will still give back.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from alphaapollo.common.generation.base import ToolCall
from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
from alphaapollo.reasoning.runtime.external.bridge.mcp_config import (
    claude_mcp_config,
    normalize_mcp_servers,
)
from alphaapollo.reasoning.runtime.external.cli import CliSettings, finalize_outcome, run_cli
from alphaapollo.reasoning.runtime.external_agent_runtime import (
    ExternalEvent,
    ExternalRunOutcome,
    encode_arguments,
    flatten_content_blocks,
)

__all__ = ["ClaudeCodeSession", "parse_stream_json"]

#: Result subtypes that mean the agent stopped at a budget, not at an answer.
_TRUNCATED_SUBTYPES = ("error_max_turns", "error_max_budget")


class ClaudeCodeSession:
    """One headless ``claude -p`` invocation per task.

    ``runner`` is injectable so tests never need the real CLI, an API key, or a
    network round trip.
    """

    def __init__(
        self,
        *,
        cli: str = "claude",
        model: str | None = None,
        timeout_s: float = 600.0,
        tools: Sequence[str] | None = None,
        permission_mode: str | None = None,
        safe_mode: bool = True,
        max_budget_usd: float | None = None,
        mcp_servers: Mapping[str, Any] | None = None,
        extra_args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        runner: Callable[..., Any] | None = None,
    ) -> None:
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a non-empty string when provided")
        if tools is not None:
            if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
                raise TypeError("tools must be a sequence of tool names")
            tools = tuple(str(tool) for tool in tools)
        if permission_mode is not None and (
            not isinstance(permission_mode, str) or not permission_mode.strip()
        ):
            raise ValueError("permission_mode must be a non-empty string when provided")
        if not isinstance(safe_mode, bool):
            raise TypeError("safe_mode must be a bool")
        if max_budget_usd is not None and (
            isinstance(max_budget_usd, bool)
            or not isinstance(max_budget_usd, (int, float))
            or max_budget_usd <= 0
        ):
            raise ValueError("max_budget_usd must be a positive number when provided")
        self._settings = CliSettings(
            cli=cli, timeout_s=timeout_s, env=env, extra_args=extra_args, runner=runner
        )
        self._model = model
        self._tools = tools
        self._permission_mode = permission_mode
        self._safe_mode = safe_mode
        self._max_budget_usd = max_budget_usd
        self._mcp_servers = normalize_mcp_servers(mcp_servers)
        if self._mcp_servers and safe_mode:
            raise ValueError(
                "safe_mode disables MCP servers, including --mcp-config ones; "
                "a bridged run must set safe_mode=False"
            )
        self._closed = False

    def argv(self, task: AgentTask) -> list[str]:
        """Build the exact command line, exposed so tests can assert isolation."""

        argv = [
            self._settings.cli,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
        ]
        if self._safe_mode:
            argv.append("--safe-mode")
        if self._model is not None:
            argv.extend(("--model", self._model))
        if task.system:
            argv.extend(("--system-prompt", task.system))
        if self._permission_mode is not None:
            argv.extend(("--permission-mode", self._permission_mode))
        if self._max_budget_usd is not None:
            argv.extend(("--max-budget-usd", str(self._max_budget_usd)))
        if self._mcp_servers:
            # --strict-mcp-config pins the run to these servers even when the
            # operator's own MCP configuration would otherwise be merged in, and
            # empty --setting-sources recovers the settings, hooks, and
            # permission-rule isolation that --safe-mode cannot provide here.
            argv.extend(("--mcp-config", claude_mcp_config(self._mcp_servers)))
            argv.extend(("--strict-mcp-config", "--setting-sources", ""))
        argv.extend(self._settings.extra_args)
        if self._tools is not None:
            # A variadic option must come last; "" is the CLI's "no built-in tools".
            argv.extend(("--tools", ",".join(self._tools)))
        return argv

    def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
        if self._closed:
            raise RuntimeError("ClaudeCodeSession is closed")
        if not isinstance(task, AgentTask):
            raise TypeError("task must be an AgentTask")
        run = run_cli(
            self.argv(task), prompt=task.prompt, workspace=workspace, settings=self._settings
        )
        return finalize_outcome(
            parse_stream_json(run.stdout), run, timeout_s=self._settings.timeout_s
        )

    def close(self) -> None:
        self._closed = True


def parse_stream_json(stdout: str) -> ExternalRunOutcome:
    """Normalize a ``--output-format stream-json`` stream into one outcome.

    Unknown top-level event types and unknown content blocks are counted and
    named in the provider metadata rather than dropped, so a CLI upgrade that
    adds a record shows up as visible drift instead of a silently shorter
    trajectory.
    """

    if not isinstance(stdout, str):
        raise TypeError("stdout must be a string")

    events: list[ExternalEvent] = []
    metadata: dict[str, Any] = {"agent": "claude_code"}
    unhandled: list[str] = []
    final_text = ""
    usage: dict[str, Any] = {}
    termination_reason = "external_error"
    saw_result = False

    for number, line in enumerate(stdout.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"claude stream-json line {number} is not valid JSON: {exc}") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"claude stream-json line {number} is not a JSON object")

        record_type = record.get("type")
        if record_type == "system":
            if record.get("subtype") == "init":
                metadata.update(
                    session_id=record.get("session_id"),
                    model=record.get("model"),
                    permission_mode=record.get("permissionMode"),
                    cli_version=record.get("claude_code_version"),
                    available_tools=list(record.get("tools") or ()),
                )
        elif record_type == "assistant":
            events.extend(_assistant_events(record, unhandled))
        elif record_type == "user":
            events.extend(_user_events(record, unhandled))
        elif record_type == "result":
            saw_result = True
            final_text = record.get("result") or ""
            if not isinstance(final_text, str):
                final_text = str(final_text)
            raw_usage = record.get("usage")
            usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
            termination_reason = _termination_reason(record)
            metadata.update(
                subtype=record.get("subtype"),
                num_turns=record.get("num_turns"),
                stop_reason=record.get("stop_reason"),
                total_cost_usd=record.get("total_cost_usd"),
                permission_denials=len(record.get("permission_denials") or ()),
            )
        elif record_type in {"rate_limit_event", "stream_event"}:
            continue
        else:
            unhandled.append(str(record_type))

    if not saw_result:
        metadata["missing_result_event"] = True
    if unhandled:
        metadata["unhandled_event_types"] = sorted(set(unhandled))
    return ExternalRunOutcome(
        final_text=final_text,
        events=tuple(events),
        termination_reason=termination_reason,
        usage=usage,
        provider_metadata=metadata,
    )


def _termination_reason(record: Mapping[str, Any]) -> str:
    subtype = str(record.get("subtype") or "")
    if subtype.startswith(_TRUNCATED_SUBTYPES):
        return "truncated"
    if record.get("is_error") or subtype.startswith("error"):
        return "external_error"
    return "final"


def _content_blocks(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    message = record.get("message")
    if not isinstance(message, Mapping):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, Sequence):
        return []
    return [block for block in content if isinstance(block, Mapping)]


def _assistant_events(record: Mapping[str, Any], unhandled: list[str]) -> list[ExternalEvent]:
    events: list[ExternalEvent] = []
    for block in _content_blocks(record):
        block_type = block.get("type")
        if block_type == "text":
            events.append(
                ExternalEvent(kind="message", content=str(block.get("text") or ""), raw=block)
            )
        elif block_type == "thinking":
            events.append(
                ExternalEvent(kind="reasoning", content=str(block.get("thinking") or ""), raw=block)
            )
        elif block_type == "tool_use":
            call_id = str(block.get("id") or "")
            name = str(block.get("name") or "")
            if not call_id or not name:
                unhandled.append("tool_use_without_identity")
                continue
            events.append(
                ExternalEvent(
                    kind="tool_call",
                    tool_call=ToolCall(
                        id=call_id, name=name, arguments=encode_arguments(block.get("input", {}))
                    ),
                    raw=block,
                )
            )
        else:
            unhandled.append(f"assistant_block:{block_type}")
    return events


def _user_events(record: Mapping[str, Any], unhandled: list[str]) -> list[ExternalEvent]:
    events: list[ExternalEvent] = []
    for block in _content_blocks(record):
        if block.get("type") != "tool_result":
            unhandled.append(f"user_block:{block.get('type')}")
            continue
        call_id = block.get("tool_use_id")
        events.append(
            ExternalEvent(
                kind="tool_result",
                content=flatten_content_blocks(block.get("content")),
                call_id=str(call_id) if call_id else None,
                failed=bool(block.get("is_error")),
                raw=block,
            )
        )
    return events
