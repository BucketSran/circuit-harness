"""Memory-owned tools over the current Common execution contract."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from alphaapollo.common.execution.tools import (
    ExecutionContext,
    ToolRequest,
    ToolResponse,
)

from .grep_stm import GrepError, TranscriptSource, grep
from .scratchpad import (
    bounded_scratchpad,
    read_scratchpad,
    update_scratchpad,
    write_scratchpad,
)

GREP_WORKSPACE_TOOL_ID = "grep_workspace"
SCRATCHPAD_TOOL_ID = "scratchpad"
TRANSCRIPT_GREP_PATTERN_CAP = 512
TRANSCRIPT_GREP_OUTPUT_CAP = 8_192


@runtime_checkable
class SandboxFiles(Protocol):
    def read_file(self, path: str) -> Any: ...

    def write_file(self, path: str, content: str) -> None: ...


def _ok(request: ToolRequest, stdout: str) -> ToolResponse:
    return ToolResponse(call_id=request.call_id, tool_id=request.tool_id, stdout=stdout)


def _error(request: ToolRequest, reason: str, message: str) -> ToolResponse:
    return ToolResponse(
        call_id=request.call_id,
        tool_id=request.tool_id,
        stderr=f"{reason}: {message}",
        exit_code=1,
    )


class ScratchpadToolAdapter:
    """Append, read, or replace a section in one per-branch scratchpad."""

    tool_id = SCRATCHPAD_TOOL_ID

    def __init__(self, session: SandboxFiles, *, lifecycle_mode: str = "per_branch") -> None:
        self._session = session
        self._lifecycle_mode = lifecycle_mode

    def execute(self, request: ToolRequest, context: ExecutionContext) -> ToolResponse:
        del context
        if self._lifecycle_mode != "per_branch":
            return _error(
                request,
                "invalid_lifecycle_mode",
                "scratchpad requires lifecycle_mode=per_branch",
            )
        args = request.arguments
        op = str(args.get("op", "")).strip().lower()
        try:
            current = read_scratchpad(self._session)
            if op == "read":
                rendered, _ = bounded_scratchpad(current)
                return _ok(request, rendered)
            updated = update_scratchpad(
                current,
                op=op,
                content=str(args.get("content", "")),
                section=str(args.get("section", "")),
            )
            write_scratchpad(self._session, updated)
        except ValueError as exc:
            return _error(request, "invalid_request", str(exc))
        except Exception as exc:  # noqa: BLE001 - protected sandbox boundary
            return _error(
                request,
                "runtime_error",
                f"scratchpad operation failed with {type(exc).__name__}",
            )
        return _ok(request, f"scratchpad updated ({len(updated)} chars)")


def loop_served_grep_source(request: ToolRequest) -> str | None:
    if request.tool_id != GREP_WORKSPACE_TOOL_ID:
        return None
    source = str(request.arguments.get("source", "")).strip().lower()
    return None if source in {"", "files"} else source


def serve_transcript_grep(
    request: ToolRequest,
    messages: list[dict[str, str]],
    *,
    output_cap: int = TRANSCRIPT_GREP_OUTPUT_CAP,
) -> ToolResponse | None:
    """Serve branch transcript grep; shared memory remains auto-retrieved only."""

    source = loop_served_grep_source(request)
    if source is None:
        return None
    if source != "transcript":
        return _error(
            request,
            "invalid_request",
            f"{source!r} is not a grep source; expected files|transcript",
        )
    pattern = str(request.arguments.get("pattern", ""))
    if not pattern.strip():
        return _error(request, "invalid_request", "pattern is required")
    if len(pattern) > TRANSCRIPT_GREP_PATTERN_CAP:
        return _error(
            request,
            "invalid_request",
            f"pattern too long (max {TRANSCRIPT_GREP_PATTERN_CAP} chars)",
        )
    try:
        result = grep(
            pattern,
            (TranscriptSource(messages),),
            fixed_string=bool(request.arguments.get("fixed_string", False)),
            ignore_case=bool(request.arguments.get("ignore_case", False)),
            output_mode=str(request.arguments.get("output_mode", "content")),
            head_limit=request.arguments.get("head_limit"),
        )
    except GrepError as exc:
        return _error(request, "invalid_request", str(exc))
    rendered = _render_grep_result(result)
    if len(rendered) > output_cap:
        rendered = rendered[:output_cap] + "\n[truncated]"
    return _ok(request, rendered)


def _render_grep_result(result: Any) -> str:
    if result.output_mode == "content":
        return "".join(f"{hit.locator}:{hit.line_no}:{hit.line}\n" for hit in result.hits)
    if result.output_mode == "locators_with_matches":
        return "".join(f"{locator}\n" for locator in result.locators)
    return "".join(f"{locator}:{count}\n" for locator, count in result.counts)


__all__ = [
    "GREP_WORKSPACE_TOOL_ID",
    "SCRATCHPAD_TOOL_ID",
    "ScratchpadToolAdapter",
    "loop_served_grep_source",
    "serve_transcript_grep",
]
