# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pre-execution authorization for atomic coding tools.

This module owns the policy gate that runs before sandbox acquisition. The
Environment-facing ``ToolGateway`` and its session lifecycle are implemented
in ``alphaapollo.common.execution.session`` and exported by ``common.execution``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from alphaapollo.common.execution.tools.base import (
    INTERNAL_PYTHON_TOOL_ID,
    ExecutionContext,
    ToolError,
    ToolRequest,
    ToolSpec,
)

_PATH_TOOL_IDS = frozenset({"read", "edit", "write", "grep", "find", "ls"})


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Fail-closed checks that do not require a sandbox or host filesystem access."""

    workspace_root: str = "/workspace"
    allowed_actors: frozenset[str] = field(
        default_factory=lambda: frozenset({"solver", "verifier"})
    )
    allowed_modes: frozenset[str] = field(
        default_factory=lambda: frozenset({"default", "isolated"})
    )
    require_timeout: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.workspace_root, str)
            or not self.workspace_root.strip()
            or "\x00" in self.workspace_root
        ):
            raise ValueError("workspace_root must be a non-empty POSIX path")
        root = PurePosixPath(self.workspace_root)
        if (
            not root.is_absolute()
            or ".." in root.parts
            or str(root) != self.workspace_root
            or self.workspace_root.startswith("//")
        ):
            raise ValueError("workspace_root must be an absolute normalized POSIX path")
        object.__setattr__(
            self,
            "allowed_actors",
            _freeze_allowlist(self.allowed_actors, "allowed_actors"),
        )
        object.__setattr__(
            self,
            "allowed_modes",
            _freeze_allowlist(self.allowed_modes, "allowed_modes"),
        )
        if not isinstance(self.require_timeout, bool):
            raise TypeError("require_timeout must be a bool")

    def authorize(
        self,
        request: ToolRequest,
        context: ExecutionContext,
        spec: ToolSpec,
    ) -> ToolError | None:
        """Authorize one catalogued request before sandbox acquisition."""
        if context.actor not in self.allowed_actors:
            return _policy_error(request, "actor_forbidden", "execution actor is not allowed")
        if context.mode not in self.allowed_modes:
            return _policy_error(request, "mode_forbidden", "execution mode is not allowed")
        if request.tool_id == INTERNAL_PYTHON_TOOL_ID and request.source != "python_code":
            return _policy_error(
                request,
                "internal_tool_forbidden",
                "internal Python compatibility tool requires <python_code> ingress",
            )
        if request.tool_id in _PATH_TOOL_IDS:
            path = request.arguments.get("path", ".")
            problem = self._validate_workspace_path(path)
            if problem is not None:
                return _policy_error(request, "path_forbidden", problem)

        properties = spec.parameters.get("properties", {})
        requested_timeout = (
            request.arguments.get("timeout")
            if isinstance(properties, Mapping) and "timeout" in properties
            else None
        )
        try:
            effective_timeout = context.effective_timeout_s(requested_timeout)
        except ValueError:
            return _policy_error(
                request,
                "invalid_timeout",
                "tool timeout must be a finite positive number",
            )
        if self.require_timeout and effective_timeout is None:
            return _policy_error(
                request,
                "timeout_required",
                "execution policy requires an enforceable timeout",
            )
        return None

    def _validate_workspace_path(self, value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return "tool path must be a non-empty string"
        if "\x00" in value:
            return "tool path must not contain NUL"
        path = PurePosixPath(value)
        if ".." in path.parts:
            return "tool path traversal is not allowed"
        root = PurePosixPath(self.workspace_root)
        candidate = path if path.is_absolute() else root / path
        try:
            candidate.relative_to(root)
        except ValueError:
            return f"tool path must stay within {root}"
        return None


def _policy_error(request: ToolRequest, code: str, message: str) -> ToolError:
    return ToolError(
        stage="policy",
        code=code,
        message=message,
        call_id=request.call_id,
        tool_id=request.tool_id,
    )


def _freeze_allowlist(value: object, field_name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of strings")
    try:
        result = frozenset(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a collection of strings") from exc
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return result


__all__ = ["ExecutionPolicy"]
