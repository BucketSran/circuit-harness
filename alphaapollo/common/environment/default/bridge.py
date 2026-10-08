# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Environment-facing tool bridge contracts and adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from alphaapollo.common.environment.default.projection import (
    FEEDBACK_MODES,
    FeedbackMode,
    format_tool_response_payload,
    normalize,
    tool_response_payload,
)
from alphaapollo.common.execution import (
    ExecutionContext,
    ToolError,
    ToolExecutor,
    ToolGateway,
    ToolGatewayResult,
    ToolGatewaySession,
    ToolRequest,
    ToolResponse,
)
from alphaapollo.common.execution.tools.registry import ToolCatalog
from alphaapollo.common.execution.tools.schemas import ToolCallRecord


@dataclass(frozen=True, slots=True)
class ToolBridgeResult:
    """Normalized result for zero or one tool call.

    An attempted call is proven by a retained ``ToolCallRecord``. Environment
    never infers an attempt from exit-code success or accepts a caller-supplied
    boolean that can drift from the execution audit record.
    """

    model_output: str | None = None
    request: ToolRequest | None = None
    response: ToolResponse | None = None
    error: ToolError | None = None
    record: ToolCallRecord | None = None
    refusal_code: str | None = None
    refusal_message: str | None = None

    def __post_init__(self) -> None:
        if self.model_output is not None and not isinstance(self.model_output, str):
            raise TypeError("model_output must be a string")
        if self.request is not None and not isinstance(self.request, ToolRequest):
            raise TypeError("request must be ToolRequest")
        if self.response is not None and not isinstance(self.response, ToolResponse):
            raise TypeError("response must be ToolResponse")
        if self.error is not None and not isinstance(self.error, ToolError):
            raise TypeError("error must be ToolError")
        if self.record is not None and not isinstance(self.record, ToolCallRecord):
            raise TypeError("record must be ToolCallRecord")
        if (self.refusal_code is None) is not (self.refusal_message is None):
            raise ValueError("tool refusal code and message must be provided together")
        if self.refusal_code is not None and not self.refusal_code.strip():
            raise ValueError("tool refusal code must be non-empty")
        if self.refusal_message is not None and not self.refusal_message.strip():
            raise ValueError("tool refusal message must be non-empty")
        if self.model_output is not None and any(
            value is not None
            for value in (
                self.request,
                self.response,
                self.error,
                self.record,
                self.refusal_code,
                self.refusal_message,
            )
        ):
            raise ValueError("ordinary model output cannot contain a tool result")
        if self.refusal_code is not None and any(
            value is not None for value in (self.request, self.response, self.error, self.record)
        ):
            raise ValueError("a tool refusal cannot contain a tool execution result")
        if self.response is not None and self.request is None:
            raise ValueError("a tool response requires its normalized request")
        if self.response is not None and self.error is not None:
            raise ValueError("a bridge result cannot contain both response and error")
        if self.request is not None and self.response is None and self.error is None:
            raise ValueError("a normalized request requires a response or error")
        if self.response is not None and self.record is None:
            raise ValueError("an attempted tool response requires its audit record")
        if self.record is not None and self.response is None:
            raise ValueError("an audit record requires its canonical tool response")
        if self.error is not None and self.record is not None:
            raise ValueError("a pre-execution tool error cannot contain an audit record")
        if (
            self.response is not None
            and self.record is not None
            and self.response.tool_id != self.record.tool_id
        ):
            raise ValueError("tool response and audit record identities must agree")

    @property
    def attempted(self) -> bool:
        return self.record is not None

    @property
    def is_tool_call(self) -> bool:
        return self.request is not None or self.error is not None or self.refusal_code is not None

    @property
    def refused(self) -> bool:
        return self.refusal_code is not None

    def observation_payload(
        self, *, feedback_mode: FeedbackMode = "differentiated"
    ) -> dict[str, Any]:
        """Return the canonical Tool-owned model observation payload."""
        if feedback_mode not in FEEDBACK_MODES:
            raise ValueError(
                f"unknown feedback_mode {feedback_mode!r}; expected one of {sorted(FEEDBACK_MODES)}"
            )
        if not self.is_tool_call:
            raise ValueError("ordinary model output has no tool observation")
        if self.refusal_code is not None:
            return {
                "ok": False,
                "call_id": None,
                "tool_id": None,
                "error": {
                    "stage": "policy",
                    "code": self.refusal_code,
                    "message": self.refusal_message,
                    "attempted": False,
                },
            }
        outcome = self.response if self.response is not None else self.error
        assert outcome is not None
        return tool_response_payload(outcome, request=self.request, feedback_mode=feedback_mode)

    @staticmethod
    def format_observation(payload: dict[str, Any]) -> str:
        """Use the Tool layer's injection-safe canonical response formatter."""
        return format_tool_response_payload(payload)

    def observation_text(self, *, feedback_mode: FeedbackMode = "differentiated") -> str:
        return self.format_observation(self.observation_payload(feedback_mode=feedback_mode))


@runtime_checkable
class ToolBridge(Protocol):
    """Environment-facing seam for exactly one model action."""

    def dispatch(self, action: Any, context: ExecutionContext) -> ToolBridgeResult: ...


class RoutedToolBridge:
    """Route one granted model tool call to one of several owned bridges.

    The router owns only shared catalog validation, the global call budget, and
    child lifecycle. Each child retains its tool-specific execution semantics.
    """

    def __init__(
        self,
        bridges_by_tool_id: Mapping[str, ToolBridge],
        *,
        catalog: ToolCatalog,
        max_tool_calls: int | None = None,
    ) -> None:
        if not isinstance(bridges_by_tool_id, Mapping) or not bridges_by_tool_id:
            raise ValueError("RoutedToolBridge requires at least one tool route")
        if not isinstance(catalog, ToolCatalog):
            raise TypeError("RoutedToolBridge requires a ToolCatalog")
        if max_tool_calls is not None and (
            isinstance(max_tool_calls, bool)
            or not isinstance(max_tool_calls, int)
            or max_tool_calls < 1
        ):
            raise ValueError("max_tool_calls must be positive when provided")
        routes: dict[str, ToolBridge] = {}
        for tool_id, bridge in bridges_by_tool_id.items():
            if not isinstance(tool_id, str) or not tool_id.strip():
                raise ValueError("tool routes require non-empty tool ids")
            if not isinstance(bridge, ToolBridge):
                raise TypeError("tool routes must contain ToolBridge implementations")
            spec = catalog.get(tool_id)
            if not spec.model_visible:
                raise ValueError(f"internal tool {tool_id!r} cannot be routed to a model")
            routes[tool_id] = bridge
        self._bridges_by_tool_id = routes
        self._catalog = catalog
        self._max_tool_calls = max_tool_calls
        self._tool_calls = 0
        self._closed_child_ids: set[int] = set()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def dispatch(self, action: Any, context: ExecutionContext) -> ToolBridgeResult:
        if self._closed:
            raise RuntimeError("RoutedToolBridge is closed")
        normalized = normalize(action)
        if normalized is None:
            return ToolBridgeResult(model_output=_extract_text_output(action))
        if isinstance(normalized, ToolError):
            return ToolBridgeResult(error=normalized)
        self._tool_calls += 1
        if self._max_tool_calls is not None and self._tool_calls > self._max_tool_calls:
            return ToolBridgeResult(
                refusal_code="tool_budget_exhausted",
                refusal_message="tool budget exhausted; provide your final answer now",
            )
        resolved = self._catalog.resolve(normalized)
        if isinstance(resolved, ToolError):
            return ToolBridgeResult(request=normalized, error=resolved)
        bridge = self._bridges_by_tool_id.get(normalized.tool_id)
        if bridge is None:
            return ToolBridgeResult(
                request=normalized,
                error=ToolError(
                    stage="policy",
                    code="tool_not_granted",
                    message=f"tool {normalized.tool_id!r} is not granted to this session",
                    call_id=normalized.call_id,
                    tool_id=normalized.tool_id,
                ),
            )
        return bridge.dispatch(action, context)

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        visited: set[int] = set()
        for bridge in self._bridges_by_tool_id.values():
            identity = id(bridge)
            if identity in visited or identity in self._closed_child_ids:
                continue
            visited.add(identity)
            close = getattr(bridge, "close", None)
            try:
                if callable(close):
                    close()
            except BaseException as exc:  # noqa: BLE001 - close every owned child
                errors.append(exc)
            else:
                self._closed_child_ids.add(identity)
        distinct_children = {id(bridge) for bridge in self._bridges_by_tool_id.values()}
        self._closed = self._closed_child_ids == distinct_children
        if errors:
            raise errors[0]


# Bridge implementations -----------------------------------------------------
class GatewayToolBridge:
    """Production adapter for one already-composed downstream ToolGateway.

    Gateway construction, catalogs, policies, sandbox profiles, and tool
    enablement remain downstream/upstream composition responsibilities.
    """

    def __init__(
        self,
        gateway: ToolGateway,
        *,
        max_tool_calls: int | None = None,
        allowed_tool_ids: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(gateway, ToolGateway):
            raise TypeError("GatewayToolBridge requires a resolved ToolGateway")
        if max_tool_calls is not None and (
            isinstance(max_tool_calls, bool)
            or not isinstance(max_tool_calls, int)
            or max_tool_calls < 1
        ):
            raise ValueError("max_tool_calls must be a positive integer when provided")
        self._gateway = gateway
        self._max_tool_calls = max_tool_calls
        if isinstance(allowed_tool_ids, (str, bytes)):
            raise TypeError("allowed_tool_ids must be an iterable of tool ids")
        if allowed_tool_ids is None:
            self._allowed_tool_ids: frozenset[str] | None = None
        else:
            tool_ids = frozenset(allowed_tool_ids)
            if any(not isinstance(tool_id, str) or not tool_id.strip() for tool_id in tool_ids):
                raise ValueError("allowed_tool_ids must contain only non-empty strings")
            self._allowed_tool_ids = tool_ids
        self._tool_calls = 0
        self._session: ToolGatewaySession | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def dispatch(self, action: Any, context: ExecutionContext) -> ToolBridgeResult:
        if self._closed:
            raise RuntimeError("GatewayToolBridge is closed")
        normalized = normalize(action)
        if normalized is None:
            return ToolBridgeResult(model_output=_extract_text_output(action))
        self._tool_calls += 1
        if self._max_tool_calls is not None and self._tool_calls > self._max_tool_calls:
            return ToolBridgeResult(
                refusal_code="tool_budget_exhausted",
                refusal_message="tool budget exhausted; provide your final answer now",
            )
        if isinstance(normalized, ToolError):
            return ToolBridgeResult(error=normalized)
        if self._allowed_tool_ids is not None and normalized.tool_id not in self._allowed_tool_ids:
            return ToolBridgeResult(
                request=normalized,
                error=ToolError(
                    stage="policy",
                    code="tool_not_granted",
                    message=f"tool {normalized.tool_id!r} is not granted to this role",
                    call_id=normalized.call_id,
                    tool_id=normalized.tool_id,
                ),
            )

        if self._session is None:
            self._session = self._gateway.open_session(context)
        gateway_result = self._session.invoke(normalized, context)
        if not isinstance(gateway_result, ToolGatewayResult):
            raise TypeError("ToolGateway.invoke must return ToolGatewayResult")
        outcome = gateway_result.outcome
        if isinstance(outcome, ToolError):
            if gateway_result.record is not None:
                raise ValueError("pre-execution ToolError cannot contain ToolCallRecord")
            if outcome.call_id not in {None, normalized.call_id} or outcome.tool_id not in {
                None,
                normalized.tool_id,
            }:
                raise ValueError("ToolGateway error identity does not match request")
            return ToolBridgeResult(request=normalized, error=outcome)
        if not isinstance(outcome, ToolResponse):
            raise TypeError("ToolGateway outcome must be ToolResponse or ToolError")
        if outcome.call_id != normalized.call_id or outcome.tool_id != normalized.tool_id:
            raise ValueError("ToolGateway response identity does not match request")
        if gateway_result.record is None:
            raise ValueError("ToolGateway attempted response is missing ToolCallRecord")
        if not isinstance(gateway_result.record, ToolCallRecord):
            raise TypeError("ToolGateway record must be ToolCallRecord")
        return ToolBridgeResult(
            request=normalized,
            response=outcome,
            record=gateway_result.record,
        )

    def close(self) -> None:
        """Release the lazily opened execution session; failed cleanup is retryable."""
        if self._closed:
            return
        if self._session is not None:
            self._session.close()
        self._closed = True


class FakeToolBridge:
    """Deterministic bridge used by tests and upstream runtime development."""

    def __init__(self, results: Iterable[ToolBridgeResult]) -> None:
        self._results = iter(results)
        self.actions: list[Any] = []
        self.contexts: list[ExecutionContext] = []

    def dispatch(self, action: Any, context: ExecutionContext) -> ToolBridgeResult:
        self.actions.append(action)
        self.contexts.append(context)
        try:
            return next(self._results)
        except StopIteration as exc:
            raise RuntimeError("FakeToolBridge has no result for this action") from exc


class TextOnlyToolBridge:
    """Treat every model action as final output without exposing any tools."""

    def dispatch(self, action: Any, context: ExecutionContext) -> ToolBridgeResult:
        del context
        return ToolBridgeResult(model_output=_extract_text_output(action))


class ExecutorToolBridge:
    """Adapt an injected atomic ToolExecutor to the canonical Environment bridge.

    This keeps hermetic/offline AlphaApollo-managed runs on DefaultEnvironment.
    Production sandbox runs should use GatewayToolBridge.
    """

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        max_tool_calls: int | None = None,
        allowed_tool_ids: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(executor, ToolExecutor):
            raise TypeError("ExecutorToolBridge requires ToolExecutor")
        if max_tool_calls is not None and (
            isinstance(max_tool_calls, bool)
            or not isinstance(max_tool_calls, int)
            or max_tool_calls < 1
        ):
            raise ValueError("max_tool_calls must be positive when provided")
        if isinstance(allowed_tool_ids, (str, bytes)):
            raise TypeError("allowed_tool_ids must be an iterable of tool ids")
        self._executor = executor
        self._max_tool_calls = max_tool_calls
        self._allowed_tool_ids = None if allowed_tool_ids is None else frozenset(allowed_tool_ids)
        if self._allowed_tool_ids is not None and any(
            not isinstance(tool_id, str) or not tool_id.strip()
            for tool_id in self._allowed_tool_ids
        ):
            raise ValueError("allowed_tool_ids must contain only non-empty strings")
        self._tool_calls = 0
        self._closed = False

    def dispatch(self, action: Any, context: ExecutionContext) -> ToolBridgeResult:
        if self._closed:
            raise RuntimeError("ExecutorToolBridge is closed")
        parsed = normalize(action)
        if parsed is None:
            return ToolBridgeResult(model_output=_extract_text_output(action))
        self._tool_calls += 1
        if self._max_tool_calls is not None and self._tool_calls > self._max_tool_calls:
            return ToolBridgeResult(
                refusal_code="budget_exhausted",
                refusal_message="tool budget exhausted; provide your final answer now",
            )
        if isinstance(parsed, ToolError):
            return ToolBridgeResult(error=parsed)
        if self._allowed_tool_ids is not None and parsed.tool_id not in self._allowed_tool_ids:
            return ToolBridgeResult(
                error=ToolError(
                    stage="policy",
                    code="tool_not_allowed",
                    message=f"tool {parsed.tool_id!r} is not granted to this role",
                    call_id=parsed.call_id,
                    tool_id=parsed.tool_id,
                )
            )
        response = self._executor.execute(parsed, context)
        if not isinstance(response, ToolResponse):
            raise TypeError("ToolExecutor must return ToolResponse")
        if response.call_id != parsed.call_id or response.tool_id != parsed.tool_id:
            raise ValueError("ToolExecutor response identity must match its request")
        record = ToolCallRecord(
            tool_id=response.tool_id,
            args=parsed.arguments,
            stdout=response.stdout,
            stderr=response.stderr,
            exit_code=response.exit_code,
            artifacts=list(response.artifacts),
        )
        return ToolBridgeResult(request=parsed, response=response, record=record)

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self._executor, "close", None)
        if callable(close):
            close()
        self._closed = True


def _extract_text_output(action: Any) -> str:
    if isinstance(action, str):
        return action
    raw: Any = action
    model_dump = getattr(action, "model_dump", None)
    if not isinstance(raw, Mapping) and callable(model_dump):
        try:
            raw = model_dump()
        except (TypeError, ValueError):
            return ""
    if not isinstance(raw, Mapping):
        return ""
    choices = raw.get("choices")
    if (
        isinstance(choices, Sequence)
        and not isinstance(choices, (str, bytes))
        and len(choices) == 1
        and isinstance(choices[0], Mapping)
    ):
        raw = choices[0].get("message", {})
    elif isinstance(raw.get("message"), Mapping):
        raw = raw["message"]
    if not isinstance(raw, Mapping):
        return ""
    content = raw.get("content")
    return content if isinstance(content, str) else ""
