# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Execution runtime and episode-scoped runtime sessions."""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field

from alphaapollo.common.execution._session.backend import (
    _SandboxAcquirer,
    _SessionSandboxManager,
    _ToolAdapter,
)
from alphaapollo.common.execution.sandbox.base import (
    PODMAN_DEFAULT,
    CancellationToken,
    OutputSink,
    SandboxProfile,
)
from alphaapollo.common.execution.sandbox.manager import SandboxManager
from alphaapollo.common.execution.tools.base import ExecutionContext, ToolError, ToolRequest
from alphaapollo.common.execution.tools.builtins.files import (
    EditTool,
    FindTool,
    GrepTool,
    LsTool,
    ReadTool,
    WriteTool,
)
from alphaapollo.common.execution.tools.builtins.shell import BashTool, PythonTool
from alphaapollo.common.execution.tools.gateway import ExecutionPolicy
from alphaapollo.common.execution.tools.registry import ToolCatalog
from alphaapollo.common.execution.tools.schemas import ToolCallRecord
from alphaapollo.common.execution.workspace import ArtifactStore

ToolInvocationResult = ToolCallRecord | ToolError

_ISOLATED_SANDBOX_KIND = "podman"
_WORKSPACE_ROOT = "/workspace"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutionRuntime:
    """Run preflighted coding-agent tool requests in rootless Podman.

    The v2 ``<python_code>`` compatibility adapter and the public workspace
    tools share one execution seam. Per-call isolation is the default;
    ``open_session`` explicitly reuses one backend across an Environment episode.
    Catalog and policy checks are repeated here so direct callers cannot bypass
    the execution safety boundary.
    """

    sandbox_manager: _SandboxAcquirer = field(default_factory=SandboxManager)
    catalog: ToolCatalog = field(default_factory=ToolCatalog)
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    bash_tool: BashTool = field(default_factory=BashTool)
    python_tool: PythonTool = field(default_factory=PythonTool)
    read_tool: ReadTool = field(default_factory=ReadTool)
    write_tool: WriteTool = field(default_factory=WriteTool)
    edit_tool: EditTool = field(default_factory=EditTool)
    grep_tool: GrepTool = field(default_factory=GrepTool)
    find_tool: FindTool = field(default_factory=FindTool)
    ls_tool: LsTool = field(default_factory=LsTool)
    additional_tools: tuple[_ToolAdapter, ...] = field(default_factory=tuple, repr=False)
    artifact_store: ArtifactStore | None = None
    sandbox_kind: str = _ISOLATED_SANDBOX_KIND
    sandbox_profile: SandboxProfile = PODMAN_DEFAULT
    _tools_by_id: dict[str, _ToolAdapter] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sandbox_kind != _ISOLATED_SANDBOX_KIND:
            raise ValueError("execution runtime only permits the isolated podman backend")
        if not isinstance(self.sandbox_profile, SandboxProfile):
            raise TypeError("execution runtime sandbox_profile must be a SandboxProfile")
        self.sandbox_profile.require_kind(self.sandbox_kind)
        if self.sandbox_profile.network:
            raise ValueError("execution runtime sandbox profile must disable network access")
        if self.sandbox_profile.allow_host_mounts:
            raise ValueError("execution runtime sandbox profile must disable host mounts")
        if self.sandbox_profile.timeout_seconds is None:
            raise ValueError("execution runtime sandbox profile must enforce a finite timeout")
        if not isinstance(self.additional_tools, tuple):
            raise TypeError("execution runtime additional_tools must be a tuple")
        adapters: tuple[_ToolAdapter, ...] = (
            self.read_tool,
            self.bash_tool,
            self.edit_tool,
            self.write_tool,
            self.grep_tool,
            self.find_tool,
            self.ls_tool,
            self.python_tool,
            *self.additional_tools,
        )
        by_id = {tool.tool_id: tool for tool in adapters}
        if len(by_id) != len(adapters):
            raise ValueError("execution runtime tool adapters must have unique ids")
        self._tools_by_id = by_id

    def open_session(self, context: ExecutionContext) -> ExecutionRuntimeSession:
        """Create a lazy, workspace-persistent execution session for one actor."""
        return ExecutionRuntimeSession(self, context)

    def invoke(
        self,
        request: ToolRequest,
        context: ExecutionContext,
        *,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ToolInvocationResult:
        """Execute one request or return a typed error before an attempt starts."""
        if not isinstance(request, ToolRequest):
            raise TypeError("ExecutionRuntime.invoke expects ToolRequest")
        if not isinstance(context, ExecutionContext):
            raise TypeError("ExecutionRuntime.invoke expects ExecutionContext")

        spec = self.catalog.resolve(request)
        if isinstance(spec, ToolError):
            return spec
        denial = self.policy.authorize(request, context, spec)
        if denial is not None:
            return denial
        tool = self._tools_by_id.get(request.tool_id)
        if tool is None:
            return _runtime_error(
                request,
                stage="catalog",
                code="tool_not_implemented",
                message=f"tool {request.tool_id!r} has no execution adapter",
            )
        problem = tool.validate_request(request)
        if problem is not None:
            return _runtime_error(
                request,
                stage="catalog",
                code="invalid_arguments",
                message=problem,
            )

        requested_timeout = tool.requested_timeout_s(request)
        try:
            effective_timeout = context.effective_timeout_s(requested_timeout)
        except ValueError:
            return _runtime_error(
                request,
                stage="policy",
                code="invalid_timeout",
                message="tool timeout must be a finite positive number",
            )

        applied_timeout = (
            effective_timeout
            if effective_timeout is not None
            else self.sandbox_profile.timeout_seconds
        )
        profile = self.sandbox_profile.with_overrides(timeout_seconds=applied_timeout)
        try:
            backend = self.sandbox_manager.acquire(
                self.sandbox_kind,
                profile=profile,
                network=False,
                tool_id=request.tool_id,
            )
        except Exception as exc:  # noqa: BLE001 - return a typed pre-execution error
            logger.warning("sandbox acquisition failed with %s", type(exc).__name__)
            return _runtime_error(
                request,
                stage="acquire",
                code="sandbox_unavailable",
                message="unable to acquire isolated sandbox",
            )

        started_at = time.perf_counter()
        cleanup_error: Exception | None = None
        try:
            record = tool.execute(
                backend,
                request,
                artifact_store=self.artifact_store,
                on_output=on_output,
                cancellation=cancellation,
            )
        except Exception as exc:  # noqa: BLE001 - an attempted call remains a record
            logger.warning("sandbox execution failed with %s", type(exc).__name__)
            record = ToolCallRecord(
                tool_id=request.tool_id,
                exit_code=-1,
                stderr="sandbox execution failed",
            )
        finally:
            try:
                backend.release()
            except Exception as exc:  # noqa: BLE001 - preserve an attempted-call record
                logger.warning("sandbox release failed with %s", type(exc).__name__)
                cleanup_error = exc

        elapsed = time.perf_counter() - started_at
        result = _canonical_record(
            record,
            request,
            context,
            profile,
            applied_timeout,
            elapsed,
        )
        if cleanup_error is not None:
            result = _with_cleanup_failure(result)
        return result


# Runtime-bound sessions -----------------------------------------------------
class ExecutionRuntimeSession:
    """Reuse one isolated backend for calls from one immutable execution context."""

    def __init__(self, runtime: ExecutionRuntime, context: ExecutionContext) -> None:
        if not isinstance(runtime, ExecutionRuntime):
            raise TypeError("ExecutionRuntimeSession expects ExecutionRuntime")
        if not isinstance(context, ExecutionContext):
            raise TypeError("ExecutionRuntimeSession expects ExecutionContext")
        self.context = context
        self._sandbox_manager = _SessionSandboxManager(
            runtime.sandbox_manager,
            timeout_seconds=(
                context.timeout_s
                if context.timeout_s is not None
                else runtime.sandbox_profile.timeout_seconds
            ),
        )
        self._runtime = ExecutionRuntime(
            sandbox_manager=self._sandbox_manager,
            catalog=runtime.catalog,
            policy=runtime.policy,
            bash_tool=runtime.bash_tool,
            python_tool=runtime.python_tool,
            read_tool=runtime.read_tool,
            write_tool=runtime.write_tool,
            edit_tool=runtime.edit_tool,
            grep_tool=runtime.grep_tool,
            find_tool=runtime.find_tool,
            ls_tool=runtime.ls_tool,
            additional_tools=runtime.additional_tools,
            artifact_store=runtime.artifact_store,
            sandbox_kind=runtime.sandbox_kind,
            sandbox_profile=runtime.sandbox_profile,
        )

    @property
    def closed(self) -> bool:
        return self._sandbox_manager.closed

    def invoke(
        self,
        request: ToolRequest,
        context: ExecutionContext | None = None,
        *,
        on_output: OutputSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ToolInvocationResult:
        if self.closed:
            raise RuntimeError("execution session is closed")
        if not isinstance(request, ToolRequest):
            raise TypeError("ExecutionRuntimeSession.invoke expects ToolRequest")
        active_context = self.context if context is None else context
        if not isinstance(active_context, ExecutionContext):
            raise TypeError("ExecutionRuntimeSession.invoke expects ExecutionContext")
        if active_context != self.context:
            return _runtime_error(
                request,
                stage="policy",
                code="session_context_mismatch",
                message="tool call context does not match the execution session",
            )
        return self._runtime.invoke(
            request,
            active_context,
            on_output=on_output,
            cancellation=cancellation,
        )

    def close(self) -> None:
        self._sandbox_manager.close()

    def __enter__(self) -> ExecutionRuntimeSession:
        if self.closed:
            raise RuntimeError("execution session is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _canonical_record(
    record: ToolCallRecord,
    request: ToolRequest,
    context: ExecutionContext,
    profile: SandboxProfile,
    effective_timeout: float | None,
    elapsed: float,
) -> ToolCallRecord:
    sandbox = dict(record.sandbox or {})
    sandbox.update(
        {
            "kind": _ISOLATED_SANDBOX_KIND,
            "workspace_root": _WORKSPACE_ROOT,
            "network": False,
            "profile": profile.name,
            "image": profile.image,
            "resource_limits": {
                "cpu_seconds": profile.cpu_seconds,
                "memory_bytes": profile.memory_bytes,
                "max_processes": profile.max_processes,
                "max_open_files": profile.max_open_files,
                "cpus": profile.cpus,
            },
            "effective_timeout_s": effective_timeout,
            "session_id": context.session_id,
            "branch_id": context.branch_id,
            "actor": context.actor,
            "mode": context.mode,
        }
    )
    if context.workspace_snapshot_ref is not None:
        sandbox["workspace_snapshot_ref"] = context.workspace_snapshot_ref
    cost = record.cost.model_copy(
        update={
            "tool_calls": max(1, record.cost.tool_calls),
            "sandbox_time_s": record.cost.sandbox_time_s + elapsed,
            "wall_clock_s": record.cost.wall_clock_s + elapsed,
        }
    )
    return ToolCallRecord(
        tool_id=request.tool_id,
        args=deepcopy(request.arguments),
        sandbox=sandbox,
        stdout=record.stdout,
        stderr=record.stderr,
        exit_code=record.exit_code,
        fs_diff=deepcopy(record.fs_diff),
        artifacts=list(record.artifacts),
        cost=cost,
    )


def _with_cleanup_failure(record: ToolCallRecord) -> ToolCallRecord:
    message = "sandbox cleanup failed"
    stderr = f"{record.stderr}\n{message}" if record.stderr else message
    return record.model_copy(
        update={
            "stderr": stderr,
            "exit_code": record.exit_code if record.exit_code != 0 else -1,
        }
    )


def _runtime_error(
    request: ToolRequest,
    *,
    stage: str,
    code: str,
    message: str,
) -> ToolError:
    return ToolError(
        stage=stage,
        code=code,
        message=message,
        call_id=request.call_id,
        tool_id=request.tool_id,
    )
