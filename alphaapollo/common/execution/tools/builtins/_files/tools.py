# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Host-side workspace tool adapters and result normalization."""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from alphaapollo.common.execution.sandbox.base import CancellationToken, OutputSink, SandboxBackend
from alphaapollo.common.execution.tools.base import ToolRequest
from alphaapollo.common.execution.tools.builtins._files.command import (
    build_workspace_command,
    non_empty_string_problem,
    optional_integer_problem,
)
from alphaapollo.common.execution.tools.builtins._shell.bash import BashTool
from alphaapollo.common.execution.tools.schemas import CostProgressRecord, ToolCallRecord
from alphaapollo.common.execution.workspace import ArtifactStore

_CONTROL_PREFIX = "__ALPHAAPOLLO_TOOL_ERROR__"
_ARTIFACT_PREFIX = "__ALPHAAPOLLO_TOOL_ARTIFACT__"
_ARTIFACT_PLACEHOLDER = "__ALPHAAPOLLO_TOOL_ARTIFACT_ID__"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkspaceTool:
    """Base adapter that executes one public workspace tool in the sandbox."""

    tool_id: str
    _bash_tool: BashTool = field(default_factory=BashTool, repr=False)
    _workspace_root: str = field(default="/workspace", repr=False)

    def validate_request(self, request: ToolRequest) -> str | None:
        return None

    def requested_timeout_s(self, request: ToolRequest) -> float | None:
        return None

    def execute(
        self,
        backend: SandboxBackend,
        request: ToolRequest,
        *,
        artifact_store: ArtifactStore | None = None,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ToolCallRecord:
        if request.tool_id != self.tool_id:
            raise ValueError(f"{type(self).__name__} cannot execute {request.tool_id!r}")
        command_request = ToolRequest(
            call_id=request.call_id,
            tool_id=self._bash_tool.tool_id,
            arguments={
                "command": build_workspace_command(
                    self.tool_id,
                    request.arguments,
                    workspace_root=self._workspace_root,
                    capture_artifact=artifact_store is not None,
                )
            },
            source=request.source,
        )
        record = self._bash_tool.execute(
            backend,
            command_request,
            artifact_store=artifact_store,
            on_output=on_output,
            cancellation=cancellation,
        )
        return _normalize_workspace_record(
            record,
            self.tool_id,
            backend=backend,
            artifact_store=artifact_store,
        )


def _normalize_workspace_record(
    record: ToolCallRecord,
    tool_id: str,
    *,
    backend: SandboxBackend,
    artifact_store: ArtifactStore | None,
) -> ToolCallRecord:
    stderr_lines = record.stderr.splitlines()
    error_metadata: dict[str, str] | None = None
    artifact_path: str | None = None
    visible_lines: list[str] = []
    for line in stderr_lines:
        if record.exit_code == 2 and line.startswith(_CONTROL_PREFIX) and error_metadata is None:
            try:
                candidate = json.loads(line[len(_CONTROL_PREFIX) :])
            except json.JSONDecodeError:
                visible_lines.append(line)
                continue
            if (
                isinstance(candidate, dict)
                and isinstance(candidate.get("code"), str)
                and isinstance(candidate.get("message"), str)
            ):
                error_metadata = {
                    "code": candidate["code"],
                    "message": candidate["message"],
                }
                continue
        if record.exit_code == 0 and line.startswith(_ARTIFACT_PREFIX) and artifact_path is None:
            try:
                candidate = json.loads(line[len(_ARTIFACT_PREFIX) :])
            except json.JSONDecodeError:
                visible_lines.append(line)
                continue
            candidate_path = candidate.get("path") if isinstance(candidate, dict) else None
            if isinstance(candidate_path, str) and candidate_path.startswith(
                "/tmp/alphaapollo-tool-output-"
            ):
                artifact_path = candidate_path
                continue
        visible_lines.append(line)

    update: dict[str, Any] = {"tool_id": tool_id}
    if error_metadata is not None:
        sandbox = dict(record.sandbox or {})
        sandbox["tool_error"] = error_metadata
        update["sandbox"] = sandbox
        cleaned = "\n".join(visible_lines).strip()
        update["stderr"] = cleaned or (f"{error_metadata['code']}: {error_metadata['message']}")
    elif artifact_path is not None:
        update["stderr"] = "\n".join(visible_lines).strip()
        _attach_workspace_artifact(
            update,
            record,
            tool_id=tool_id,
            remote_path=artifact_path,
            backend=backend,
            artifact_store=artifact_store,
        )
    return record.model_copy(update=update)


def _attach_workspace_artifact(
    update: dict[str, Any],
    record: ToolCallRecord,
    *,
    tool_id: str,
    remote_path: str,
    backend: SandboxBackend,
    artifact_store: ArtifactStore | None,
) -> None:
    artifact_id: str | None = None
    if artifact_store is not None:
        try:
            with TemporaryDirectory(prefix="alphaapollo-tool-artifact-") as temporary:
                destination = Path(temporary) / "output.txt"
                backend.copy_out(remote_path, str(destination))
                artifact = artifact_store.put_file(
                    destination,
                    type_=f"{tool_id}_output",
                    created_by=tool_id,
                )
                reference = artifact_store.ref(artifact)
            artifact_id = reference.id
            update["artifacts"] = [*record.artifacts, reference]
            update["cost"] = record.cost + CostProgressRecord(artifacts_produced=1)
        except Exception as exc:  # noqa: BLE001 - bounded output still returns safely
            logger.warning(
                "%s output artifact persistence failed with %s",
                tool_id,
                type(exc).__name__,
            )
        finally:
            try:
                backend.exec(f"rm -f -- {shlex.quote(remote_path)}")
            except Exception as exc:  # noqa: BLE001 - container release remains authoritative
                logger.warning(
                    "%s remote output cleanup failed with %s",
                    tool_id,
                    type(exc).__name__,
                )
    replacement = artifact_id or "unavailable"
    update["stdout"] = record.stdout.replace(_ARTIFACT_PLACEHOLDER, replacement)


@dataclass(frozen=True, slots=True)
class ReadTool(WorkspaceTool):
    tool_id: str = "read"

    def validate_request(self, request: ToolRequest) -> str | None:
        return (
            non_empty_string_problem(request.arguments.get("path"), "path")
            or optional_integer_problem(request.arguments, "offset")
            or optional_integer_problem(request.arguments, "limit")
        )


@dataclass(frozen=True, slots=True)
class WriteTool(WorkspaceTool):
    tool_id: str = "write"

    def validate_request(self, request: ToolRequest) -> str | None:
        problem = non_empty_string_problem(request.arguments.get("path"), "path")
        if problem is not None:
            return problem
        if not isinstance(request.arguments.get("content"), str):
            return "content must be a string"
        return None


@dataclass(frozen=True, slots=True)
class EditTool(WorkspaceTool):
    tool_id: str = "edit"

    def validate_request(self, request: ToolRequest) -> str | None:
        problem = non_empty_string_problem(request.arguments.get("path"), "path")
        if problem is not None:
            return problem
        edits = request.arguments.get("edits")
        if not isinstance(edits, list) or not edits:
            return "edits must contain at least one replacement"
        for index, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return f"edits[{index}] must be an object"
            if not isinstance(edit.get("oldText"), str) or not edit["oldText"]:
                return f"edits[{index}].oldText must be a non-empty string"
            if not isinstance(edit.get("newText"), str):
                return f"edits[{index}].newText must be a string"
        return None


@dataclass(frozen=True, slots=True)
class GrepTool(WorkspaceTool):
    tool_id: str = "grep"

    def validate_request(self, request: ToolRequest) -> str | None:
        arguments = request.arguments
        problem = non_empty_string_problem(arguments.get("pattern"), "pattern")
        if problem is not None:
            return problem
        if "path" in arguments:
            problem = non_empty_string_problem(arguments["path"], "path")
            if problem is not None:
                return problem
        if "glob" in arguments:
            problem = non_empty_string_problem(arguments["glob"], "glob")
            if problem is not None:
                return problem
        return optional_integer_problem(
            arguments, "context", minimum=0
        ) or optional_integer_problem(arguments, "limit")


@dataclass(frozen=True, slots=True)
class FindTool(WorkspaceTool):
    tool_id: str = "find"

    def validate_request(self, request: ToolRequest) -> str | None:
        arguments = request.arguments
        problem = non_empty_string_problem(arguments.get("pattern"), "pattern")
        if problem is not None:
            return problem
        if "path" in arguments:
            problem = non_empty_string_problem(arguments["path"], "path")
            if problem is not None:
                return problem
        return optional_integer_problem(arguments, "limit")


@dataclass(frozen=True, slots=True)
class LsTool(WorkspaceTool):
    tool_id: str = "ls"

    def validate_request(self, request: ToolRequest) -> str | None:
        arguments = request.arguments
        if "path" in arguments:
            problem = non_empty_string_problem(arguments["path"], "path")
            if problem is not None:
                return problem
        return optional_integer_problem(arguments, "limit")
