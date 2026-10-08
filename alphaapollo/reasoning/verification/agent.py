"""Agent-backed verification as a thin adapter over ``AgentRuntime``.

This module owns prompt rendering and judgment normalization only.  It never
calls a Generation backend or an Environment: all agent execution is delegated in
one batch to the injected Runtime.
"""

from __future__ import annotations

import copy
import json
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from alphaapollo.reasoning._immutable import _thaw_json
from alphaapollo.reasoning.runtime.agent_runtime import AgentResult, AgentRuntime, AgentTask
from alphaapollo.reasoning.verification.base import (
    VerificationContractError,
    VerificationRequest,
    VerificationResult,
    Verifier,
    _validate_result_batch,
)

_TEMPLATE_FIELDS = frozenset(
    ("request_id", "problem", "candidate", "metadata", "role", "model", "tools")
)
_JSON_FIELDS = frozenset(("verdict", "feedback", "details"))
_VERDICTS = frozenset(("pass", "fail", "inconclusive"))
_VERDICT_LINE_RE = re.compile(
    r"\AVERDICT[ \t]*:[ \t]*(PASS|FAIL|INCONCLUSIVE)[ \t]*\Z",
    re.IGNORECASE,
)
_FEEDBACK_LINE_RE = re.compile(r"\AFEEDBACK[ \t]*:[ \t]*(.*)\Z", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AgentVerifierConfig:
    """Fully resolved, caller-owned configuration for an agent verifier role.

    There are intentionally no built-in verifier instructions.  A versioned
    Workflow configuration (or another explicit caller) supplies every field.
    ``model`` and granted tool names are copied to explicit :class:`AgentTask`
    policy fields; the injected Runtime remains the execution owner.
    """

    system_prompt: str
    input_template: str
    role: str
    model: str
    tools: tuple[str, ...]
    output_format: str

    def __post_init__(self) -> None:
        for name in ("system_prompt", "input_template", "role", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.tools, (str, bytes)) or not isinstance(self.tools, Sequence):
            raise TypeError("tools must be a sequence of tool identifiers")
        tools = tuple(self.tools)
        if any(not isinstance(tool, str) or not tool.strip() for tool in tools):
            raise ValueError("tools must contain only non-empty strings")
        if len(set(tools)) != len(tools):
            raise ValueError("tools must not contain duplicates")
        if not isinstance(self.output_format, str):
            raise TypeError("output_format must be a string")
        output_format = self.output_format.strip().lower().replace("_", "-")
        if output_format not in {"json", "verdict-line"}:
            raise ValueError("output_format must be json or verdict-line")
        _validate_input_template(self.input_template)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "output_format", output_format)


class AgentVerifier(Verifier):
    """Render verification tasks, delegate them once, and parse their results."""

    def __init__(self, runtime: AgentRuntime, config: AgentVerifierConfig) -> None:
        if not hasattr(runtime, "run_batch") or not callable(runtime.run_batch):
            raise TypeError("runtime must provide run_batch(tasks)")
        if not isinstance(config, AgentVerifierConfig):
            raise TypeError("config must be an AgentVerifierConfig")
        self._runtime = runtime
        self.config = config

    def verify_batch(
        self,
        requests: Sequence[VerificationRequest],
    ) -> list[VerificationResult]:
        requests = _coerce_requests(requests)
        if not requests:
            return []

        tasks = [self._task_for(request) for request in requests]
        agent_results = self._runtime.run_batch(tasks)
        _validate_agent_results(requests, agent_results)
        results = [
            self._normalize(request, agent_result)
            for request, agent_result in zip(requests, agent_results, strict=True)
        ]
        return _validate_result_batch(requests, results, verifier=type(self).__name__)

    def _task_for(self, request: VerificationRequest) -> AgentTask:
        metadata = _thaw_json(request.metadata)
        metadata.update(
            {
                "actor": "verifier",
                "verification_request_id": request.request_id,
                "role": self.config.role,
            }
        )
        return AgentTask(
            task_id=request.request_id,
            system=self.config.system_prompt,
            prompt=_render_input(self.config, request),
            model=self.config.model,
            routing_key=request.routing_key,
            tools=self.config.tools,
            metadata=metadata,
            branch_id=request.branch_id,
            round_index=request.round_index,
            sample_id=request.sample_id,
            sampling_seed=request.sampling_seed,
        )

    def _normalize(
        self,
        request: VerificationRequest,
        agent_result: AgentResult,
    ) -> VerificationResult:
        if agent_result.termination_reason != "final":
            reason = agent_result.termination_reason
            return VerificationResult(
                request_id=request.request_id,
                verdict="inconclusive",
                candidate=request.candidate,
                candidate_ref=request.candidate_ref,
                feedback=(
                    f"verifier agent terminated with reason {reason!r}; "
                    "its output was not evaluated"
                ),
                details={
                    "output_format": self.config.output_format,
                    "role": self.config.role,
                    "model": self.config.model,
                    "termination_reason": reason,
                    "parse_skipped": True,
                },
                agent_result=agent_result,
            )
        try:
            if self.config.output_format == "json":
                verdict, feedback, parsed_details = _parse_json(agent_result.final_text)
            else:
                verdict, feedback, parsed_details = _parse_verdict_line(agent_result.final_text)
        except ValueError as exc:
            return VerificationResult(
                request_id=request.request_id,
                verdict="inconclusive",
                candidate=request.candidate,
                candidate_ref=request.candidate_ref,
                feedback=f"{_unreadable_reason(agent_result)}: {exc}",
                details={
                    "output_format": self.config.output_format,
                    "role": self.config.role,
                    "model": self.config.model,
                    "termination_reason": agent_result.termination_reason,
                    "parse_error": str(exc),
                },
                agent_result=agent_result,
            )

        return VerificationResult(
            request_id=request.request_id,
            verdict=verdict,
            candidate=request.candidate,
            candidate_ref=request.candidate_ref,
            feedback=feedback,
            details={
                **parsed_details,
                "output_format": self.config.output_format,
                "role": self.config.role,
                "model": self.config.model,
                "termination_reason": agent_result.termination_reason,
            },
            agent_result=agent_result,
        )


def _unreadable_reason(agent_result: AgentResult) -> str:
    """Name why a judgment could not be read, distinguishing truncation.

    A reply the sampler cut off at ``max_tokens`` still arrives with
    ``termination_reason == "final"``: the Environment that ends the episode
    never sees the backend's ``finish_reason``. Measured on AIME 2026, one
    verifier spent all 12,000 completion tokens deliberating inside a JSON
    string and was cut mid-token, so its judgment was incomplete rather than
    malformed. Say which it was, because raising the budget fixes one and never
    the other.

    ``AgentResult.output_truncated`` is read rather than re-derived from the
    last turn's ``finish_reason`` here. This function used to do its own lookup,
    which was correct while it was the only consumer; the scorer and the report
    now ask the same question, and two derivations of one fact from one source
    is exactly the drift the field exists to prevent.
    """

    if agent_result.output_truncated:
        return (
            "invalid verifier output (the reply was cut off at the sampling "
            "max_tokens limit, so no complete judgment was ever emitted)"
        )
    return "invalid verifier output"


def _coerce_requests(
    requests: Sequence[VerificationRequest],
) -> tuple[VerificationRequest, ...]:
    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
        raise TypeError("requests must be a sequence of VerificationRequest objects")
    materialized = tuple(requests)
    for index, request in enumerate(materialized):
        if not isinstance(request, VerificationRequest):
            raise TypeError(f"requests[{index}] must be a VerificationRequest")
    return materialized


def _validate_agent_results(
    requests: Sequence[VerificationRequest],
    results: object,
) -> None:
    if not isinstance(results, list):
        raise VerificationContractError("AgentRuntime.run_batch() must return a list")
    if len(results) != len(requests):
        raise VerificationContractError(
            f"AgentRuntime.run_batch() returned {len(results)} results for "
            f"{len(requests)} verification requests"
        )
    for index, (request, result) in enumerate(zip(requests, results, strict=True)):
        if not isinstance(result, AgentResult):
            raise VerificationContractError(
                f"AgentRuntime.run_batch() result {index} is not an AgentResult"
            )
        if result.task_id != request.request_id:
            raise VerificationContractError(
                f"AgentRuntime.run_batch() changed task identity at index {index}: "
                f"expected {request.request_id!r}, got {result.task_id!r}"
            )


def _validate_input_template(template: str) -> None:
    try:
        parsed = tuple(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"input_template is invalid: {exc}") from exc
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in _TEMPLATE_FIELDS:
            raise ValueError(
                f"input_template uses unsupported field {field_name!r}; "
                f"allowed fields are {', '.join(sorted(_TEMPLATE_FIELDS))}"
            )
        if format_spec:
            raise ValueError("input_template fields may not use format specifications")
        if conversion:
            raise ValueError("input_template fields may not use conversions")


def _render_input(config: AgentVerifierConfig, request: VerificationRequest) -> str:
    values = {
        "request_id": request.request_id,
        "problem": request.problem,
        "candidate": request.candidate,
        "metadata": json.dumps(_thaw_json(request.metadata), sort_keys=True, ensure_ascii=False),
        "role": config.role,
        "model": config.model,
        "tools": json.dumps(config.tools, ensure_ascii=False),
    }
    try:
        return config.input_template.format_map(values)
    except (KeyError, ValueError) as exc:  # defensive; config was validated at construction
        raise VerificationContractError(f"failed to render verifier input_template: {exc}") from exc


def _parse_json(text: str) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(text, str):
        raise ValueError("agent final_text is not a string")
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        # Carry the decoder's own complaint: it is the only thing that says
        # *why* a judgment was discarded, and ``parse_error`` is all a reader of
        # the record gets. "Invalid \\escape" and "Unterminated string" have
        # different causes and different fixes.
        raise ValueError(f"expected one JSON object with no surrounding text: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON verifier output must be an object")
    unknown = set(payload) - _JSON_FIELDS
    if unknown:
        raise ValueError(f"JSON verifier output has unknown fields: {sorted(unknown)!r}")
    if "verdict" not in payload:
        raise ValueError("JSON verifier output is missing verdict")
    verdict = payload["verdict"]
    if not isinstance(verdict, str) or verdict.strip().lower() not in _VERDICTS:
        raise ValueError("JSON verdict must be pass, fail, or inconclusive")
    feedback = payload.get("feedback", "")
    if not isinstance(feedback, str):
        raise ValueError("JSON feedback must be a string")
    details = payload.get("details", {})
    if not isinstance(details, Mapping):
        raise ValueError("JSON details must be an object")
    return verdict.strip().lower(), feedback, copy.deepcopy(dict(details))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON verifier output contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON verifier output contains non-standard constant {value!r}")


def _parse_verdict_line(text: str) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(text, str):
        raise ValueError("agent final_text is not a string")
    verdicts: list[str] = []
    feedback_values: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        verdict_match = _VERDICT_LINE_RE.fullmatch(stripped)
        if verdict_match is not None:
            verdicts.append(verdict_match.group(1).lower())
            continue
        feedback_match = _FEEDBACK_LINE_RE.fullmatch(stripped)
        if feedback_match is not None:
            feedback_values.append(feedback_match.group(1).strip())
    if len(verdicts) != 1:
        raise ValueError("verdict-line output must contain exactly one VERDICT line")
    if len(feedback_values) > 1:
        raise ValueError("verdict-line output may contain at most one FEEDBACK line")
    feedback = feedback_values[0] if feedback_values else ""
    return verdicts[0], feedback, {}


__all__ = ["AgentVerifier", "AgentVerifierConfig"]
