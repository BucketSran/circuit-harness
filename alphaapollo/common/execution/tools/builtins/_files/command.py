# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Injection-safe workspace worker command construction and validation."""

from __future__ import annotations

import base64
import json
import shlex
from typing import Any

from alphaapollo.common.execution.tools.builtins._files.worker import _WORKSPACE_WORKER


def build_workspace_command(
    tool_id: str,
    arguments: dict[str, Any],
    *,
    workspace_root: str = "/workspace",
    capture_artifact: bool = False,
) -> str:
    """Encode one workspace request into an injection-safe in-container command."""
    payload = json.dumps(
        {
            "tool_id": tool_id,
            "arguments": arguments,
            "capture_artifact": capture_artifact,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded_payload = base64.b64encode(payload).decode("ascii")
    encoded_source = base64.b64encode(_WORKSPACE_WORKER.encode("utf-8")).decode("ascii")
    loader = (
        "import base64,sys;"
        f"source=base64.b64decode({encoded_source!r});"
        f"sys.argv=['workspace-tool',{encoded_payload!r}];"
        f"namespace={{'WORKSPACE_ROOT':{workspace_root!r},'__name__':'__main__'}};"
        "exec(compile(source,'<workspace_tool>','exec'),namespace)"
    )
    return f"python -c {shlex.quote(loader)}"


def positive_integer_problem(value: object, name: str, *, minimum: int = 1) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        return f"{name} must be a {qualifier} integer"
    return None


def optional_integer_problem(
    arguments: dict[str, Any], name: str, *, minimum: int = 1
) -> str | None:
    if name not in arguments:
        return None
    return positive_integer_problem(arguments[name], name, minimum=minimum)


def non_empty_string_problem(value: object, name: str) -> str | None:
    if not isinstance(value, str) or not value:
        return f"{name} must be a non-empty string"
    return None
