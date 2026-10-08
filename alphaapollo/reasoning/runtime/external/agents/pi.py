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

"""pi (earendil-works/pi) as an external agent session.

Driven through ``pi -p --mode json`` with the prompt on stdin: one shot in, one
JSONL event stream out.  pi also offers ``--mode rpc`` for bidirectional
control, which is what a steering or multi-turn driver would need; a single
``AgentTask`` does not, so this adapter stays on the simpler unidirectional
mode.

pi-hosted runs are the ones whose built-in tool ids (``read``, ``bash``, ``edit``,
``write``, ``grep``, ``find``, ``ls``) are exactly this repository's
``PUBLIC_TOOL_IDS`` -- they were vendored from it (see ``PI_SOURCE``).  It is
also the one with no MCP client, so ``mcp_servers`` is served through
``pi_extension.ts``, a bundled extension that speaks MCP to the same server the
other native runners spawn.  Because an extension registers plain names, a
bridged pi-hosted run is the one whose model-facing surface matches the native path:
``bash``, not ``mcp__alphaapollo__bash``.  Bridging costs the tool mode --
``--no-tools`` would disable extension tools too, and the default leaves pi's
own beside ours -- so a bridged run must choose ``no_builtin`` or ``allowlist``.

pi exits zero even when a provider rejects the request, emitting an assistant
message with empty content.  That silence is reported as ``external_error``
rather than an empty successful answer.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from alphaapollo.common.generation.base import ToolCall
from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
from alphaapollo.reasoning.runtime.external.bridge.mcp_config import normalize_mcp_servers
from alphaapollo.reasoning.runtime.external.cli import CliSettings, finalize_outcome, run_cli
from alphaapollo.reasoning.runtime.external_agent_runtime import (
    ExternalEvent,
    ExternalRunOutcome,
    encode_arguments,
    flatten_content_blocks,
)

__all__ = [
    "EXTENSION_PATH",
    "SERVERS_ENV",
    "CodexViaPiSession",
    "PiSession",
    "parse_mode_json",
]

_TOOL_MODES = ("default", "none", "no_builtin", "allowlist")

#: Tool modes that actually leave the bridged tools reachable and alone.
_BRIDGED_TOOL_MODES = ("no_builtin", "allowlist")

#: The bundled extension, loaded with ``-e``; it is packaged data, not source.
EXTENSION_PATH = Path(__file__).parent.parent / "bridge" / "pi_extension.ts"

#: How the extension learns which servers to speak to. pi has no way to pass
#: arguments to an extension, so the declaration travels in the environment.
SERVERS_ENV = "ALPHAAPOLLO_MCP_SERVERS"


class PiSession:
    """One headless ``pi -p --mode json`` invocation per task."""

    def __init__(
        self,
        *,
        cli: str = "pi",
        model: str | None = None,
        provider: str | None = None,
        thinking: str | None = None,
        timeout_s: float = 600.0,
        tool_mode: str = "default",
        tools: Sequence[str] = (),
        mcp_servers: Mapping[str, Any] | None = None,
        extra_args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        runner: Callable[..., Any] | None = None,
        event_sink: Callable[[str], None] | None = None,
    ) -> None:
        for name, value in (("model", model), ("provider", provider), ("thinking", thinking)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided")
        if tool_mode not in _TOOL_MODES:
            raise ValueError(f"tool_mode must be one of {list(_TOOL_MODES)}, got {tool_mode!r}")
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
            raise TypeError("tools must be a sequence of names")
        tool_names = tuple(str(tool) for tool in tools)
        if tool_mode == "allowlist" and not tool_names:
            raise ValueError("tool_mode 'allowlist' requires at least one tool name")
        servers = normalize_mcp_servers(mcp_servers)
        if servers:
            remote = sorted(name for name, spec in servers.items() if "url" in spec)
            if remote:
                raise ValueError(f"the pi extension speaks stdio only; {remote} declare a url")
            if tool_mode not in _BRIDGED_TOOL_MODES:
                raise ValueError(
                    f"tool_mode {tool_mode!r} would disable or shadow the bridged tools; "
                    f"a bridged run must use one of {list(_BRIDGED_TOOL_MODES)}"
                )
            # A partial environment would strip pi's own credentials, so the
            # declaration is added to one rather than replacing it.
            env = {
                **(dict(env) if env is not None else dict(os.environ)),
                SERVERS_ENV: json.dumps(servers, sort_keys=True),
            }
        self._settings = CliSettings(
            cli=cli, timeout_s=timeout_s, env=env, extra_args=extra_args, runner=runner
        )
        self._bridged = bool(servers)
        self._model = model
        self._provider = provider
        self._thinking = thinking
        self._tool_mode = tool_mode
        self._tools = tool_names
        self._event_sink = event_sink
        self._closed = False

    def argv(self, task: AgentTask) -> list[str]:
        """Build the exact command line, exposed so tests can assert isolation."""

        argv = [self._settings.cli, "-p", "--mode", "json", "--no-session", "--no-approve"]
        if task.system:
            argv.extend(("--system-prompt", task.system))
        if self._provider is not None:
            argv.extend(("--provider", self._provider))
        if self._model is not None:
            argv.extend(("--model", self._model))
        if self._thinking is not None:
            argv.extend(("--thinking", self._thinking))
        if self._tool_mode == "none":
            argv.append("--no-tools")
        elif self._tool_mode == "no_builtin":
            argv.append("--no-builtin-tools")
        elif self._tool_mode == "allowlist":
            argv.extend(("--tools", ",".join(self._tools)))
        if self._bridged:
            argv.extend(("-e", str(EXTENSION_PATH)))
        argv.extend(self._settings.extra_args)
        return argv

    def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
        if self._closed:
            raise RuntimeError("PiSession is closed")
        if not isinstance(task, AgentTask):
            raise TypeError("task must be an AgentTask")
        run = run_cli(
            self.argv(task), prompt=task.prompt, workspace=workspace, settings=self._settings
        )
        if self._event_sink is not None:
            self._event_sink(run.stdout)
        return finalize_outcome(
            parse_mode_json(run.stdout), run, timeout_s=self._settings.timeout_s
        )

    def close(self) -> None:
        self._closed = True


class CodexViaPiSession(PiSession):
    """Run an OpenAI Codex provider through pi's exclusive tool loop.

    Native ``codex exec`` still exposes Codex-owned execution tools.  pi can
    select the same ``openai-codex`` provider while ``--no-builtin-tools``
    leaves only the AlphaApollo extension visible, so Environment-backed runs
    use this explicit adapter instead of claiming the native CLI is bounded.
    """

    def __init__(self, *, provider: str = "openai-codex", **options: Any) -> None:
        if provider != "openai-codex":
            raise ValueError("CodexViaPiSession provider is fixed to 'openai-codex'")
        super().__init__(provider=provider, **options)

    def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
        outcome = super().run(task, workspace=workspace)
        return ExternalRunOutcome(
            final_text=outcome.final_text,
            events=outcome.events,
            termination_reason=outcome.termination_reason,
            usage=outcome.usage,
            provider_metadata={
                **outcome.provider_metadata,
                "agent": "codex_via_pi",
                "runner": "pi",
            },
        )


def parse_mode_json(stdout: str) -> ExternalRunOutcome:
    """Normalize a ``pi --mode json`` stream into one outcome.

    Assistant content comes from the authoritative ``message_end`` record rather
    than by re-assembling ``message_update`` deltas, and tool activity comes from
    the ``tool_execution_*`` events, which carry the call id on both halves.

    pi announces one call twice -- as a ``toolCall`` block inside the assistant
    message and again as ``tool_execution_start`` -- so calls are de-duplicated
    by id, first announcement wins. Without that, every tool-using run would
    report twice the tool calls it made.
    """

    if not isinstance(stdout, str):
        raise TypeError("stdout must be a string")

    events: list[ExternalEvent] = []
    metadata: dict[str, Any] = {"agent": "pi"}
    unhandled: list[str] = []
    usage: dict[str, Any] = {}
    seen_call_ids: set[str] = set()
    bridged_tools: set[str] = set()
    dispatched_ids: set[str] = set()
    final_text = ""
    settled = False
    last_stop_reason = None

    for number, line in enumerate(stdout.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"pi json line {number} is not valid JSON: {exc}") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"pi json line {number} is not a JSON object")

        record_type = record.get("type")
        if record_type == "session":
            metadata.update(session_id=record.get("id"), session_version=record.get("version"))
        elif record_type == "message_end":
            message = record.get("message")
            # `user` and `toolResult` roles repeat input we already have; only the
            # assistant record describes what the agent produced.
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                last_stop_reason = message.get("stopReason")
                text = _project_assistant(message, events, seen_call_ids)
                if text:
                    final_text = text
                usage = _usage(message) or usage
                metadata.setdefault("model", message.get("model"))
                metadata.setdefault("provider", message.get("provider"))
        elif record_type == "tool_execution_start":
            event = _tool_call_event(record)
            assert event.tool_call is not None  # guaranteed by ExternalEvent
            if event.tool_call.id not in seen_call_ids:
                seen_call_ids.add(event.tool_call.id)
                events.append(event)
        elif record_type == "alphaapollo_mcp_tool":
            bridged_tools.add(str(record.get("toolName")))
        elif record_type == "alphaapollo_mcp_dispatch":
            dispatched_ids.add(str(record.get("toolCallId")))
        elif record_type == "tool_execution_end":
            event = _tool_result_event(record)
            # Pi validates before calling the extension. Require our own bridge
            # registration/dispatch evidence, not just an arbitrary error string.
            rejected = (
                event.failed
                and event.tool_id in bridged_tools
                and event.call_id in seen_call_ids
                and event.call_id not in dispatched_ids
                and event.content.startswith(f'Validation failed for tool "{event.tool_id}":\n')
            )
            events.append(replace(event, rejected_before_dispatch=rejected))
        elif record_type in {
            "agent_start",
            "turn_start",
            "turn_end",
            "message_start",
            "message_update",
            "tool_execution_update",
            "queue_update",
            "agent_end",
            "compaction_start",
            "compaction_end",
            "auto_retry_start",
            "auto_retry_end",
        }:
            continue
        elif record_type == "agent_settled":
            settled = True
        elif record_type == "extension_error":
            metadata["extension_error"] = str(record.get("message") or record.get("error") or "")
        else:
            unhandled.append(str(record_type))

    has_content = any(event.kind in {"message", "tool_call"} for event in events)
    if not settled:
        metadata["missing_agent_settled"] = True
    if unhandled:
        metadata["unhandled_event_types"] = sorted(set(unhandled))
    if not has_content:
        metadata["empty_response"] = True
    termination_reason = "external_error"
    if settled and last_stop_reason == "length":
        termination_reason = "truncated"
    elif settled and has_content and last_stop_reason not in {"error", "aborted"}:
        termination_reason = "final"
    return ExternalRunOutcome(
        final_text=final_text,
        events=tuple(events),
        termination_reason=termination_reason,
        usage=usage,
        provider_metadata=metadata,
    )


def _usage(message: Mapping[str, Any]) -> dict[str, Any]:
    raw = message.get("usage")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _project_assistant(
    message: Mapping[str, Any], events: list[ExternalEvent], seen_call_ids: set[str]
) -> str:
    content = message.get("content")
    if isinstance(content, str):
        if content:
            events.append(ExternalEvent(kind="message", content=content, raw=message))
        return content
    if not isinstance(content, Sequence):
        return ""
    text = ""
    for block in content:
        if not isinstance(block, Mapping):
            continue
        block_type = str(block.get("type") or "")
        if block_type == "text":
            value = str(block.get("text") or "")
            if value:
                events.append(ExternalEvent(kind="message", content=value, raw=block))
                text = value
        elif block_type == "thinking":
            value = str(block.get("thinking") or block.get("text") or "")
            if value:
                events.append(ExternalEvent(kind="reasoning", content=value, raw=block))
        elif block_type in {"toolCall", "tool_call"}:
            call_id = str(block.get("id") or "call")
            if call_id in seen_call_ids:
                continue
            seen_call_ids.add(call_id)
            events.append(
                ExternalEvent(
                    kind="tool_call",
                    tool_call=ToolCall(
                        id=call_id,
                        name=str(block.get("name") or "tool"),
                        arguments=encode_arguments(block.get("arguments", block.get("input", {}))),
                    ),
                    raw=block,
                )
            )
    return text


def _tool_call_event(record: Mapping[str, Any]) -> ExternalEvent:
    return ExternalEvent(
        kind="tool_call",
        tool_call=ToolCall(
            id=str(record.get("toolCallId") or "call"),
            name=str(record.get("toolName") or "tool"),
            arguments=encode_arguments(record.get("args", {})),
        ),
        raw=record,
    )


def _tool_result_event(record: Mapping[str, Any]) -> ExternalEvent:
    result = record.get("result")
    content = result.get("content") if isinstance(result, Mapping) else result
    call_id = record.get("toolCallId")
    tool_id = record.get("toolName")
    return ExternalEvent(
        kind="tool_result",
        content=flatten_content_blocks(content),
        call_id=str(call_id) if call_id else None,
        tool_id=str(tool_id) if tool_id else None,
        failed=bool(record.get("isError")),
        raw=record,
    )
