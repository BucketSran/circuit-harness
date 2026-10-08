# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"Typed tool contracts and schemas."

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from alphaapollo.common.artifacts.schemas import ArtifactRef, require_json_value

_TOOL_ERROR_STAGES = frozenset({"parse", "catalog", "policy", "acquire"})


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """System-owned context for one atomic tool invocation.

    The upper layer supplies these values. They must never be accepted from the
    model-controlled tool arguments. ``timeout_s`` is the system/policy hard
    upper bound. A tool may request a shorter timeout, but never a longer one.
    Policy must fail closed when an execution profile requires a timeout and
    the selected backend cannot enforce it.
    """

    session_id: str
    branch_id: str = "main"
    workspace_snapshot_ref: str | None = None
    actor: str = "solver"
    mode: str = "default"
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("session_id", "branch_id", "actor", "mode"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.workspace_snapshot_ref is not None and (
            not isinstance(self.workspace_snapshot_ref, str)
            or not self.workspace_snapshot_ref.strip()
        ):
            raise ValueError("workspace_snapshot_ref must be non-empty when provided")
        if self.timeout_s is not None:
            if (
                isinstance(self.timeout_s, bool)
                or not isinstance(self.timeout_s, (int, float))
                or not math.isfinite(self.timeout_s)
                or self.timeout_s <= 0
            ):
                raise ValueError("timeout_s must be a finite positive number when provided")

    def effective_timeout_s(self, requested_s: float | None = None) -> float | None:
        """Return the smaller applicable system and tool-requested timeout."""
        if requested_s is not None and (
            isinstance(requested_s, bool)
            or not isinstance(requested_s, (int, float))
            or not math.isfinite(requested_s)
            or requested_s <= 0
        ):
            raise ValueError("requested_s must be a finite positive number when provided")
        if self.timeout_s is None:
            return requested_s
        if requested_s is None:
            return self.timeout_s
        return min(self.timeout_s, requested_s)


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """Canonical runtime request for one atomic tool call.

    ``call_id`` and ``source`` are provider/ingress metadata. Only ``arguments``
    are model-controlled tool parameters.
    """

    call_id: str
    tool_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    source: str = "direct"

    def __post_init__(self) -> None:
        for field_name in ("call_id", "tool_id", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if not isinstance(self.arguments, dict):
            raise TypeError("arguments must be a dict")
        arguments = deepcopy(self.arguments)
        require_json_value(arguments, path="$.arguments")
        object.__setattr__(self, "arguments", arguments)


@dataclass(frozen=True, slots=True)
class ToolError:
    """Typed failure raised before a backend execution attempt begins."""

    stage: str
    code: str
    message: str
    call_id: str | None = None
    tool_id: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in _TOOL_ERROR_STAGES:
            allowed = ", ".join(sorted(_TOOL_ERROR_STAGES))
            raise ValueError(f"stage must be one of: {allowed}")
        for field_name in ("code", "message"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in ("call_id", "tool_id"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class ToolResponse:
    call_id: str
    tool_id: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.tool_id.strip():
            raise ValueError("call_id and tool_id must be non-empty")
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise TypeError("exit_code must be an int")


@runtime_checkable
class ToolExecutor(Protocol):
    def execute(self, request: ToolRequest, context: ExecutionContext) -> ToolResponse: ...


PI_SOURCE_COMMIT = "edbf941dad7f94edcf028f254db311b0efe44b6f"
PI_SOURCE = f"earendil-works/pi@{PI_SOURCE_COMMIT}"
PUBLIC_TOOL_IDS = ("read", "bash", "edit", "write", "grep", "find", "ls")
INTERNAL_PYTHON_TOOL_ID = "python"


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Immutable runtime-local tool definition and model schema."""

    tool_id: str
    description: str
    parameters: Mapping[str, Any]
    model_visible: bool = True
    source: str = PI_SOURCE
    adaptations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("tool_id", "description", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        if not isinstance(self.model_visible, bool):
            raise TypeError("model_visible must be a bool")
        if isinstance(self.adaptations, (str, bytes)):
            raise TypeError("adaptations must be a sequence of strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.adaptations):
            raise ValueError("adaptations must contain non-empty strings")
        frozen_parameters = _freeze_json(dict(self.parameters))
        if frozen_parameters.get("type") != "object":
            raise ValueError("tool parameters must be a JSON object schema")
        object.__setattr__(self, "parameters", frozen_parameters)
        object.__setattr__(self, "adaptations", tuple(self.adaptations))

    def to_openai_tool(self) -> dict[str, Any]:
        """Return a fresh OpenAI-compatible function tool definition."""
        if not self.model_visible:
            raise ValueError(f"internal tool {self.tool_id!r} is not model-visible")
        return {
            "type": "function",
            "function": {
                "name": self.tool_id,
                "description": self.description,
                "parameters": _thaw_json(self.parameters),
            },
        }


def _object_schema(
    properties: dict[str, dict[str, Any]],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = list(required)
    return schema


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _number(description: str) -> dict[str, Any]:
    return {"type": "number", "description": description}


def _integer(description: str, *, minimum: int) -> dict[str, Any]:
    return {"type": "integer", "minimum": minimum, "description": description}


def _boolean(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


_TOOL_SPECS = {
    "read": ToolSpec(
        tool_id="read",
        description=(
            "Read the contents of a text file. Output is truncated to 2000 lines or 50KB "
            "(whichever is hit first). Use offset/limit for large files and continue with "
            "offset when the full file is needed."
        ),
        parameters=_object_schema(
            {
                "path": _string("Path to the file to read (relative or absolute)"),
                "offset": _integer("Line number to start reading from (1-indexed)", minimum=1),
                "limit": _integer("Maximum number of lines to read", minimum=1),
            },
            required=("path",),
        ),
        adaptations=("Apollo v1 read is text-only; Pi image attachment behavior is deferred.",),
    ),
    "bash": ToolSpec(
        tool_id="bash",
        description=(
            "Execute a bash command in the current working directory. Returns stdout and "
            "stderr. Output is truncated to the last 2000 lines or 50KB (whichever is hit "
            "first). Optionally request a timeout in seconds; Apollo always enforces a "
            "wall-clock timeout, using the smaller of the requested timeout and the "
            "system-owned ExecutionContext timeout, or the 30-second Podman default when "
            "neither specifies a shorter timeout."
        ),
        parameters=_object_schema(
            {
                "command": _string("Bash command to execute"),
                "timeout": _number("Timeout in seconds (optional, no default timeout)"),
            },
            required=("command",),
        ),
    ),
    "edit": ToolSpec(
        tool_id="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must "
            "match a unique, non-overlapping region of the original file."
        ),
        parameters=_object_schema(
            {
                "path": _string("Path to the file to edit (relative or absolute)"),
                "edits": {
                    "type": "array",
                    "description": (
                        "One or more targeted replacements. Each edit is matched against the "
                        "original file, not incrementally. Overlapping or nested edits are invalid."
                    ),
                    "minItems": 1,
                    "items": _object_schema(
                        {
                            "oldText": _string(
                                "Exact text for one targeted replacement. It must be unique in "
                                "the original file and must not overlap another edit."
                            ),
                            "newText": _string("Replacement text for this targeted edit."),
                        },
                        required=("oldText", "newText"),
                    ),
                },
            },
            required=("path", "edits"),
        ),
        adaptations=(
            "Apollo makes Pi's documented one-or-more edits requirement explicit with minItems=1.",
        ),
    ),
    "write": ToolSpec(
        tool_id="write",
        description=(
            "Write content to a file. Creates the file if it does not exist, overwrites it "
            "if it does, and automatically creates parent directories."
        ),
        parameters=_object_schema(
            {
                "path": _string("Path to the file to write (relative or absolute)"),
                "content": _string("Content to write to the file"),
            },
            required=("path", "content"),
        ),
    ),
    "grep": ToolSpec(
        tool_id="grep",
        description=(
            "Search file contents for a pattern. Returns matching lines with file paths and "
            "line numbers, respects .gitignore, and truncates output to 100 matches or 50KB."
        ),
        parameters=_object_schema(
            {
                "pattern": _string("Search pattern (regex or literal string)"),
                "path": _string("Directory or file to search (default: current directory)"),
                "glob": _string("Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"),
                "ignoreCase": _boolean("Case-insensitive search (default: false)"),
                "literal": _boolean(
                    "Treat pattern as literal string instead of regex (default: false)"
                ),
                "context": _integer(
                    "Number of lines to show before and after each match (default: 0)",
                    minimum=0,
                ),
                "limit": _integer("Maximum number of matches to return (default: 100)", minimum=1),
            },
            required=("pattern",),
        ),
    ),
    "find": ToolSpec(
        tool_id="find",
        description=(
            "Search for files by glob pattern. Returns matching paths relative to the search "
            "directory, respects .gitignore, and truncates output to 1000 results or 50KB."
        ),
        parameters=_object_schema(
            {
                "pattern": _string(
                    "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"
                ),
                "path": _string("Directory to search in (default: current directory)"),
                "limit": _integer("Maximum number of results (default: 1000)", minimum=1),
            },
            required=("pattern",),
        ),
    ),
    "ls": ToolSpec(
        tool_id="ls",
        description=(
            "List directory contents sorted alphabetically, with '/' suffixes for directories "
            "and dotfiles included. Output is truncated to 500 entries or 50KB."
        ),
        parameters=_object_schema(
            {
                "path": _string("Directory to list (default: current directory)"),
                "limit": _integer("Maximum number of entries to return (default: 500)", minimum=1),
            }
        ),
    ),
    INTERNAL_PYTHON_TOOL_ID: ToolSpec(
        tool_id=INTERNAL_PYTHON_TOOL_ID,
        description="Internal Apollo v2 <python_code> compatibility entry.",
        parameters=_object_schema(
            {"code": _string("Python source extracted from <python_code>")},
            required=("code",),
        ),
        model_visible=False,
        source="AlphaApollo v2 compatibility contract",
        adaptations=(
            "Not exported to the model; public Python execution is performed through bash.",
        ),
    ),
}


def get_tool_spec(tool_id: str) -> ToolSpec:
    """Return the canonical public or internal tool definition."""
    try:
        return _TOOL_SPECS[tool_id]
    except KeyError as exc:
        raise KeyError(f"unknown tool id: {tool_id}") from exc


def list_tool_specs(*, include_internal: bool = False) -> tuple[ToolSpec, ...]:
    """List canonical specs in stable model-prompt order."""
    specs = tuple(_TOOL_SPECS[tool_id] for tool_id in PUBLIC_TOOL_IDS)
    if include_internal:
        specs += (_TOOL_SPECS[INTERNAL_PYTHON_TOOL_ID],)
    return specs


def export_openai_tools() -> list[dict[str, Any]]:
    """Export exactly the seven public tools as OpenAI function schemas."""
    return [spec.to_openai_tool() for spec in list_tool_specs()]
