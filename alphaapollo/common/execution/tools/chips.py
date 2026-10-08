"""Chips tool descriptions and the bounded SSH EMX executor.

Session tools execute through a Runtime-owned Environment. Their schemas stay
owned by the stdlib-only task sessions, which are also shipped in offline bundles.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import replace
from pathlib import Path

from alphaapollo.common.execution.chips import analog_session, vabench_session
from alphaapollo.common.execution.tools.base import (
    ExecutionContext,
    ToolRequest,
    ToolResponse,
    ToolSpec,
)

SESSION_TOOL_SPECS = tuple(
    ToolSpec(
        tool_id=tool["function"]["name"],
        description=tool["function"]["description"],
        parameters=tool["function"]["parameters"],
        source="Chips public task session",
    )
    for session in (vabench_session, analog_session)
    for tool in session.tool_schemas()
)

EMX_TOOL_ID = "emx_simulate"
EMX_TOOL_SPEC = ToolSpec(
    tool_id=EMX_TOOL_ID,
    description=(
        "Convert layout JSON to GDS using the configured converter, submit an idempotent "
        "SSH EMX job, and retrieve verified logs/results. execution=ok means the command "
        "completed, not design acceptance. If unknown, resume the returned run_id; never "
        "create a replacement job automatically. Commands, server, PDK and ports are "
        "operator-configured. No circuit-quality score is inferred."
    ),
    parameters={
        "type": "object",
        "properties": {
            "layout_json": {
                "type": "string",
                "description": "JSON object accepted by the operator's layout converter",
            },
            "resume_run_id": {
                "type": "string",
                "description": "Run ID from a previous incomplete call; requires identical layout",
            },
        },
        "required": ["layout_json"],
        "additionalProperties": False,
    },
    source="Chips SSH EMX harness",
)


class EmxTool:
    def __init__(self, *, working_directory: Path):
        self.root = working_directory / "runs" / "chips"
        self.cancel = threading.Event()
        self.lock = threading.Lock()

    def execute(self, request: ToolRequest, context: ExecutionContext) -> ToolResponse:
        if self.cancel.is_set():
            return ToolResponse(
                request.call_id, request.tool_id, stderr="EMX tool is closed", exit_code=1
            )
        try:
            if os.name != "posix":
                raise ValueError("the Chips SSH executor requires a POSIX host")
            from alphaapollo.common.execution.chips.emx import EmxConfig, run_emx

            if request.tool_id != EMX_TOOL_ID or set(request.arguments) - {
                "layout_json",
                "resume_run_id",
            }:
                raise ValueError("invalid EMX tool request")
            config_path = os.environ.get("ALPHAAPOLLO_CHIPS_CONFIG")
            if not config_path:
                raise ValueError(
                    "set ALPHAAPOLLO_CHIPS_CONFIG to the operator's absolute config path"
                )
            if not Path(config_path).is_absolute():
                raise ValueError("ALPHAAPOLLO_CHIPS_CONFIG must be absolute")
            config = EmxConfig.load(Path(config_path))
            limit = context.effective_timeout_s(config.timeout_s)
            config = replace(config, timeout_s=limit or config.timeout_s)
            run_id = request.arguments.get("resume_run_id", uuid.uuid4().hex)
            if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
                raise ValueError("invalid resume_run_id")
            directory = self.root / run_id
            if "resume_run_id" in request.arguments and not (directory / "intent.json").exists():
                raise ValueError("resume_run_id is not an existing run")
            with self.lock:
                result = run_emx(
                    json.loads(request.arguments["layout_json"]),
                    config,
                    directory,
                    resume="resume_run_id" in request.arguments,
                    cancel=self.cancel,
                    call_id=request.call_id,
                )
            logs = {}
            for name in (
                "artifacts/emx.stdout.log",
                "artifacts/emx.stderr.log",
                "convert.stdout.log",
                "convert.stderr.log",
            ):
                path = directory / name
                if path.is_file():
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 4000))
                        logs[name] = stream.read().decode(errors="replace")
            result.update(run_id=run_id, run_directory=str(directory), log_tails=logs)
            return ToolResponse(
                request.call_id,
                request.tool_id,
                stdout=json.dumps(result),
                exit_code=0 if result["execution"] == "ok" else 1,
            )
        except (ValueError, TypeError, KeyError, OSError) as error:
            return ToolResponse(
                request.call_id,
                request.tool_id,
                stderr=f"{type(error).__name__}: {error}",
                exit_code=2,
            )

    def close(self):
        self.cancel.set()
