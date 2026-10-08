# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bash tool adaptation and bounded ToolCallRecord rendering."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from alphaapollo.common.execution.output import format_truncated_output, truncate_tail
from alphaapollo.common.execution.sandbox.base import (
    CancellationToken,
    OutputSink,
    SandboxBackend,
    StreamingSandboxBackend,
)
from alphaapollo.common.execution.tools.base import ToolRequest
from alphaapollo.common.execution.tools.schemas import CostProgressRecord, ToolCallRecord
from alphaapollo.common.execution.workspace import ArtifactStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BashTool:
    """Execute a validated bash request through an already-isolated backend."""

    tool_id: str = "bash"

    def validate_request(self, request: ToolRequest) -> str | None:
        command = request.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return "bash command must be a non-empty string"
        return None

    def requested_timeout_s(self, request: ToolRequest) -> float | None:
        return request.arguments.get("timeout")

    def execute(
        self,
        backend: SandboxBackend,
        request: ToolRequest,
        *,
        artifact_store: ArtifactStore | None = None,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ToolCallRecord:
        """Run the model command without interpreting it on the host."""
        if request.tool_id != self.tool_id:
            raise ValueError(f"BashTool cannot execute {request.tool_id!r}")
        command = request.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("bash command must be a non-empty string")
        if isinstance(backend, StreamingSandboxBackend):
            return backend.exec_stream(
                command,
                on_output=on_output,
                cancellation=cancellation,
                artifact_store=artifact_store,
            )
        record = backend.exec(command)
        return _bounded_output(record, artifact_store)


def _bounded_output(
    record: ToolCallRecord,
    artifact_store: ArtifactStore | None,
) -> ToolCallRecord:
    stdout = truncate_tail(record.stdout)
    stderr = truncate_tail(record.stderr)
    if not stdout.truncated and not stderr.truncated:
        return record

    artifacts = list(record.artifacts)
    metadata: dict[str, object] = {}
    rendered: dict[str, str] = {}
    produced = 0
    for stream_name, original, result in (
        ("stdout", record.stdout, stdout),
        ("stderr", record.stderr, stderr),
    ):
        if not result.truncated:
            rendered[stream_name] = original
            continue
        artifact_id: str | None = None
        if artifact_store is not None:
            try:
                artifact = artifact_store.put(
                    original.encode("utf-8"),
                    type_=f"bash_{stream_name}",
                    created_by="bash",
                )
                ref = artifact_store.ref(artifact)
                artifacts.append(ref)
                artifact_id = ref.id
                produced += 1
            except Exception as exc:  # noqa: BLE001 - output still returns safely
                logger.warning(
                    "bash %s artifact persistence failed with %s",
                    stream_name,
                    type(exc).__name__,
                )
        stream_metadata = asdict(result)
        stream_metadata.pop("content")
        stream_metadata["artifact_id"] = artifact_id
        metadata[stream_name] = stream_metadata
        rendered[stream_name] = format_truncated_output(stream_name, result, artifact_id)

    sandbox = dict(record.sandbox or {})
    sandbox["output_truncation"] = metadata
    cost = record.cost + CostProgressRecord(artifacts_produced=produced)
    return record.model_copy(
        update={
            "stdout": rendered["stdout"],
            "stderr": rendered["stderr"],
            "sandbox": sandbox,
            "artifacts": artifacts,
            "cost": cost,
        }
    )
