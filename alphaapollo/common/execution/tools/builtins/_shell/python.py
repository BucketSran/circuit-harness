# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compatibility Python tool and in-sandbox command construction."""

from __future__ import annotations

import ast
import base64
import shlex
from dataclasses import dataclass, field

from alphaapollo.common.execution.sandbox.base import CancellationToken, OutputSink, SandboxBackend
from alphaapollo.common.execution.tools.base import INTERNAL_PYTHON_TOOL_ID, ToolRequest
from alphaapollo.common.execution.tools.builtins._shell.bash import BashTool
from alphaapollo.common.execution.tools.schemas import ToolCallRecord
from alphaapollo.common.execution.workspace import ArtifactStore

# Keep v2's convenient standard-library namespace for Python programs.
# These are all Python standard-library imports and execute inside the sandbox.
# ``sys``/``io`` are deliberately imported by module, not star-imported: their
# wildcard names (sys.path/argv/exit, io.open) shadow builtins and cause
# surprising user-code failures.
V2_PRE_IMPORTS = """\
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from json import *
from builtins import *
from typing import *
from io import BytesIO, StringIO

import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(6*10**5)
"""


def wrap_v2_python_code(code: str) -> str:
    """Preserve v2's pre-import and implicit-final-print behavior.

    A final expression is wrapped in ``print(...)``. If the final line is not an
    expression, v2 prints the most recently discovered simple assignment. Syntax
    errors are intentionally left in the source so they become an attempted,
    sandbox-recorded execution failure rather than a host-side parse failure.
    """
    if not isinstance(code, str):
        raise TypeError("python code must be a string")
    if not code.strip():
        raise ValueError("python code must be non-empty")

    lines = code.rstrip().split("\n")
    last_line = lines[-1].strip()
    if not last_line.startswith("print("):
        try:
            ast.parse(last_line, mode="eval")
        except (SyntaxError, ValueError):
            variable = _last_simple_assignment(code)
            if variable is not None:
                lines.append(f"print({variable})")
        else:
            leading = lines[-1][: len(lines[-1]) - len(lines[-1].lstrip())]
            lines[-1] = f"{leading}print({last_line})"

    body = "\n".join(lines)
    return f"{V2_PRE_IMPORTS}\n{body}\n"


def build_python_command(code: str) -> str:
    """Build a shell-safe command that decodes and executes source in-container."""
    wrapped = wrap_v2_python_code(code)
    encoded = base64.b64encode(wrapped.encode("utf-8")).decode("ascii")
    loader = (
        "import base64;"
        f"source=base64.b64decode({encoded!r});"
        "exec(compile(source, '<python_code>', 'exec'))"
    )
    return f"python -c {shlex.quote(loader)}"


def _last_simple_assignment(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in reversed(tree.body):
        if isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                return target.id
    return None


@dataclass(frozen=True, slots=True)
class PythonTool:
    """Execute internal v2 Python requests through an isolated backend."""

    tool_id: str = INTERNAL_PYTHON_TOOL_ID
    _bash_tool: BashTool = field(default_factory=BashTool, repr=False)

    def validate_request(self, request: ToolRequest) -> str | None:
        code = request.arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return "python code must be a non-empty string"
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
            raise ValueError(f"PythonTool cannot execute {request.tool_id!r}")
        if request.source != "python_code":
            raise ValueError("internal Python execution requires <python_code> ingress")
        code = request.arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("python code must be a non-empty string")

        command_request = ToolRequest(
            call_id=request.call_id,
            tool_id=self._bash_tool.tool_id,
            arguments={"command": build_python_command(code)},
            source=request.source,
        )
        record = self._bash_tool.execute(
            backend,
            command_request,
            artifact_store=artifact_store,
            on_output=on_output,
            cancellation=cancellation,
        )
        return record.model_copy(update={"tool_id": self.tool_id})


PythonCompatibilityTool = PythonTool
