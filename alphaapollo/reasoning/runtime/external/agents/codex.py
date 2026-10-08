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

"""OpenAI Codex as an external agent session.

Driven through ``codex exec - --json`` with the prompt on stdin.  The official
TypeScript and Python SDKs spawn the same binary and read the same JSONL event
stream, so this is the vendor's own integration seam rather than a workaround.

Two Codex-specific facts shape this adapter:

* There is no ``--system-prompt``.  The role's system text is prepended to the
  prompt and the outcome records ``system_prompt_inlined`` so a reader never
  assumes Codex received it as a separate system message.
* ``--ignore-user-config`` skips ``config.toml`` but still loads the operator's
  ``$CODEX_HOME/agents/*.toml``, which arrive as non-fatal ``error`` items.
  Those are counted in the outcome metadata instead of becoming trajectory
  events, because turning startup warnings into observations would invent turns.

Model routing stays config-driven: ``config_overrides`` are passed verbatim as
``-c key=value`` so a run can point Codex at a local server without touching the
operator's ``~/.codex/config.toml``.  ``mcp_servers`` uses the same mechanism,
and its calls arrive as ``mcp_tool_call`` items this parser already projects.

Headless MCP calls need an explicit approval policy.  The adapter grants
``approval_mode=\"approve\"`` only to the configured server/tool pairs and keeps
Codex under its selected sandbox.  This avoids both interactive cancellation
and the CLI's all-or-nothing no-sandbox escape hatch.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from alphaapollo.common.generation.base import ToolCall
from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
from alphaapollo.reasoning.runtime.external.bridge.mcp_config import (
    codex_approval_overrides,
    codex_config_overrides,
    normalize_mcp_servers,
)
from alphaapollo.reasoning.runtime.external.cli import CliSettings, finalize_outcome, run_cli
from alphaapollo.reasoning.runtime.external_agent_runtime import (
    ExternalEvent,
    ExternalRunOutcome,
    encode_arguments,
    flatten_content_blocks,
)

__all__ = ["CodexSession", "parse_exec_json"]

_SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")

#: Item types that describe an action plus its result in one completed record.
_ACTION_ITEMS = ("command_execution", "file_change", "mcp_tool_call", "web_search")


class CodexSession:
    """A headless Codex channel, optionally continued across tasks."""

    def __init__(
        self,
        *,
        cli: str = "codex",
        model: str | None = None,
        timeout_s: float = 600.0,
        sandbox: str = "read-only",
        config_overrides: Sequence[str] = (),
        mcp_servers: Mapping[str, Any] | None = None,
        mcp_auto_approve_tools: Mapping[str, Sequence[str]] | None = None,
        ignore_user_config: bool = True,
        continuation: bool = False,
        eventless_error_retries: int = 0,
        extra_args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        runner: Callable[..., Any] | None = None,
        event_sink: Callable[[str], None] | None = None,
        raw_event_path: Path | None = None,
    ) -> None:
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a non-empty string when provided")
        if sandbox not in _SANDBOX_MODES:
            raise ValueError(f"sandbox must be one of {list(_SANDBOX_MODES)}, got {sandbox!r}")
        if isinstance(config_overrides, (str, bytes)) or not isinstance(config_overrides, Sequence):
            raise TypeError("config_overrides must be a sequence of arguments")
        for override in config_overrides:
            if not isinstance(override, str) or "=" not in override:
                raise ValueError("config_overrides entries must look like 'key=value'")
        if not isinstance(ignore_user_config, bool):
            raise TypeError("ignore_user_config must be a bool")
        if not isinstance(continuation, bool):
            raise TypeError("continuation must be a bool")
        if (
            isinstance(eventless_error_retries, bool)
            or not isinstance(eventless_error_retries, int)
            or not 0 <= eventless_error_retries <= 3
        ):
            raise ValueError("eventless_error_retries must be an int between 0 and 3")
        if raw_event_path is not None and eventless_error_retries:
            raise ValueError("raw_event_path requires eventless_error_retries=0")
        self._settings = CliSettings(
            cli=cli,
            timeout_s=timeout_s,
            env=env,
            extra_args=extra_args,
            runner=runner,
            stdout_path=raw_event_path,
        )
        self._model = model
        self._sandbox = sandbox
        servers = normalize_mcp_servers(mcp_servers)
        if mcp_auto_approve_tools is not None and not isinstance(mcp_auto_approve_tools, Mapping):
            raise TypeError("mcp_auto_approve_tools must be a mapping")
        approvals = dict(mcp_auto_approve_tools or {})
        unknown_servers = set(approvals) - set(servers)
        if unknown_servers:
            raise ValueError(
                f"mcp_auto_approve_tools names undeclared servers: {sorted(unknown_servers)}"
            )
        if servers and set(approvals) != set(servers):
            missing = sorted(set(servers) - set(approvals))
            raise ValueError(
                "headless Codex MCP servers need explicit auto-approved tool ids; "
                f"missing approvals for {missing}"
            )
        self._config_overrides = (
            tuple(config_overrides)
            + tuple(codex_config_overrides(servers))
            + tuple(codex_approval_overrides(approvals))
        )
        self._ignore_user_config = ignore_user_config
        self._continuation = continuation
        self._eventless_error_retries = eventless_error_retries
        self._event_sink = event_sink
        self._closed = False
        self._thread_id: str | None = None
        self._system_prompt: str | None = None

    @property
    def supports_continuation(self) -> bool:
        """Whether this configured channel can safely be retained by a Runtime."""

        return self._continuation

    def argv(self) -> list[str]:
        """Build the exact command line, exposed so tests can assert isolation."""

        if self._thread_id is None:
            argv = [self._settings.cli, "exec", "-", "--json", "--skip-git-repo-check"]
            if not self._continuation:
                argv.append("--ephemeral")
        else:
            argv = [
                self._settings.cli,
                "exec",
                "resume",
                self._thread_id,
                "-",
                "--json",
                "--skip-git-repo-check",
            ]
        if self._ignore_user_config:
            argv.extend(("--ignore-user-config", "--ignore-rules"))
        # ``codex exec resume`` inherits the original sandbox and does not expose
        # ``-s``. The initial invocation records the requested sandbox in the
        # persisted conversation.
        if self._thread_id is None:
            argv.extend(("-s", self._sandbox))
        if self._model is not None:
            argv.extend(("-m", self._model))
        for override in self._config_overrides:
            argv.extend(("-c", override))
        argv.extend(self._settings.extra_args)
        return argv

    def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
        if self._closed:
            raise RuntimeError("CodexSession is closed")
        if not isinstance(task, AgentTask):
            raise TypeError("task must be an AgentTask")
        if self._thread_id is not None and task.system != self._system_prompt:
            raise ValueError("a continued Codex conversation cannot change its system prompt")
        outcome: ExternalRunOutcome | None = None
        retries = 0
        for attempt in range(self._eventless_error_retries + 1):
            resuming = self._thread_id is not None
            # Codex has no --system-prompt, so the role's system text is prepended
            # to the first turn. A resumed conversation already contains it.
            inline_system = not resuming and bool(task.system)
            prompt = f"{task.system}\n\n{task.prompt}" if inline_system else task.prompt
            run = run_cli(self.argv(), prompt=prompt, workspace=workspace, settings=self._settings)
            if self._event_sink is not None:
                self._event_sink(run.raw_stdout if run.raw_stdout is not None else run.stdout)
            outcome = finalize_outcome(
                parse_exec_json(run.stdout),
                run,
                timeout_s=self._settings.timeout_s,
                extra={
                    "system_prompt_inlined": inline_system,
                    "conversation_continued": resuming,
                },
            )
            reported_thread = outcome.provider_metadata.get("thread_id")
            if self._continuation:
                if not isinstance(reported_thread, str) or not reported_thread.strip():
                    raise RuntimeError(
                        "Codex continuation requires a non-empty thread_id in the JSON stream"
                    )
                if self._thread_id is not None and reported_thread != self._thread_id:
                    raise RuntimeError("resumed Codex conversation reported a different thread_id")
                self._thread_id = reported_thread
                if self._system_prompt is None:
                    self._system_prompt = task.system
            if outcome.termination_reason != "external_error" or outcome.events:
                break
            if attempt < self._eventless_error_retries:
                retries += 1
        assert outcome is not None
        if retries:
            outcome = ExternalRunOutcome(
                final_text=outcome.final_text,
                events=outcome.events,
                termination_reason=outcome.termination_reason,
                usage=outcome.usage,
                provider_metadata={
                    **dict(outcome.provider_metadata),
                    "eventless_error_retries": retries,
                },
            )
        return outcome

    def close(self) -> None:
        self._closed = True


def parse_exec_json(stdout: str) -> ExternalRunOutcome:
    """Normalize a ``codex exec --json`` stream into one outcome.

    Only ``item.completed`` is projected: it carries the whole record, and using
    ``item.started`` as well would double every action.  Unknown event and item
    types are named in the metadata rather than dropped.
    """

    if not isinstance(stdout, str):
        raise TypeError("stdout must be a string")

    events: list[ExternalEvent] = []
    metadata: dict[str, Any] = {"agent": "codex"}
    unhandled: list[str] = []
    warnings: list[str] = []
    usage: dict[str, Any] = {}
    final_text = ""
    termination_reason = "external_error"
    seen_completion = False

    for number, line in enumerate(stdout.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"codex exec line {number} is not valid JSON: {exc}") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"codex exec line {number} is not a JSON object")

        record_type = record.get("type")
        if record_type == "thread.started":
            metadata["thread_id"] = record.get("thread_id")
        elif record_type in {"turn.started", "item.started", "item.updated"}:
            continue
        elif record_type == "turn.completed":
            seen_completion = True
            termination_reason = "final"
            raw_usage = record.get("usage")
            usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
        elif record_type == "turn.failed":
            seen_completion = True
            termination_reason = "external_error"
            metadata["error"] = _error_message(record.get("error"))
        elif record_type == "error":
            termination_reason = "external_error"
            metadata["error"] = str(record.get("message") or "")
        elif record_type == "item.completed":
            item = record.get("item")
            if not isinstance(item, Mapping):
                unhandled.append("item.completed_without_item")
                continue
            text = _project_item(item, events, unhandled, warnings)
            if text:
                final_text = text
        else:
            unhandled.append(str(record_type))

    if not seen_completion:
        metadata["missing_turn_completion"] = True
    if warnings:
        # Startup warnings are configuration noise, not agent behaviour.
        metadata["startup_warnings"] = len(warnings)
        metadata["startup_warning_sample"] = warnings[:3]
    if unhandled:
        metadata["unhandled_event_types"] = sorted(set(unhandled))
    return ExternalRunOutcome(
        final_text=final_text,
        events=tuple(events),
        termination_reason=termination_reason,
        usage=usage,
        provider_metadata=metadata,
    )


def _error_message(error: Any) -> str:
    if isinstance(error, Mapping):
        return str(error.get("message") or "")
    return str(error or "")


def _project_item(
    item: Mapping[str, Any],
    events: list[ExternalEvent],
    unhandled: list[str],
    warnings: list[str],
) -> str:
    """Append the events for one completed item; return its final text, if any."""

    item_type = str(item.get("type") or "")
    if item_type == "agent_message":
        text = str(item.get("text") or "")
        if text:
            events.append(ExternalEvent(kind="message", content=text, raw=item))
        return text
    if item_type == "reasoning":
        text = str(item.get("text") or "")
        if text:
            events.append(ExternalEvent(kind="reasoning", content=text, raw=item))
        return ""
    if item_type == "error":
        # Non-fatal per the vendor schema; recorded as metadata, not a turn.
        warnings.append(str(item.get("message") or ""))
        return ""
    if item_type in _ACTION_ITEMS:
        events.extend(_action_events(item, item_type))
        return ""
    unhandled.append(f"item:{item_type}")
    return ""


def _action_events(item: Mapping[str, Any], item_type: str) -> list[ExternalEvent]:
    """Split one completed action item into its call and its result.

    Codex reports an action and its outcome as a single record, but a trajectory
    needs the request the agent made and the observation it received separately.
    """

    call_id = str(item.get("id") or "")
    tool_id, arguments, output, failed = _action_shape(item, item_type)
    if not call_id:
        call_id = f"{tool_id}-call"
    return [
        ExternalEvent(
            kind="tool_call",
            tool_call=ToolCall(id=call_id, name=tool_id, arguments=encode_arguments(arguments)),
            raw=item,
        ),
        ExternalEvent(
            kind="tool_result",
            content=output,
            call_id=call_id,
            tool_id=tool_id,
            failed=failed,
            raw=item,
        ),
    ]


def _action_shape(item: Mapping[str, Any], item_type: str) -> tuple[str, Any, str, bool]:
    status_failed = str(item.get("status") or "") == "failed"
    if item_type == "command_execution":
        exit_code = item.get("exit_code")
        failed = status_failed or (isinstance(exit_code, int) and exit_code != 0)
        return (
            "shell",
            {"command": item.get("command")},
            str(item.get("aggregated_output") or ""),
            failed,
        )
    if item_type == "file_change":
        changes = item.get("changes") or []
        return "apply_patch", {"changes": changes}, encode_arguments(changes), status_failed
    if item_type == "mcp_tool_call":
        error = item.get("error")
        result = item.get("result")
        content = flatten_content_blocks(
            result.get("content") if isinstance(result, Mapping) else result
        )
        if isinstance(error, Mapping) and error.get("message"):
            content = str(error["message"])
        tool_id = f"{item.get('server') or 'mcp'}.{item.get('tool') or 'tool'}"
        return tool_id, item.get("arguments") or {}, content, status_failed or error is not None
    query = str(item.get("query") or "")
    return "web_search", {"query": query}, query, status_failed
