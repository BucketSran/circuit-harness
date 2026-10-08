"""Stable, batch-first contracts for Reasoning verification.

Verification is a semantic Workflow step.  It is deliberately smaller than an
agent Runtime: a verifier receives public problem/candidate data and returns one
ordered judgment for every request.  Agent-backed verification and deterministic
verification share these records so Workflow orchestration does not need to know
how a judgment was produced.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from alphaapollo.reasoning._immutable import _freeze_json_mapping
from alphaapollo.reasoning.runtime.agent_runtime import AgentResult

VerificationVerdict = Literal["pass", "fail", "inconclusive"]

_VERDICTS: frozenset[str] = frozenset(("pass", "fail", "inconclusive"))


class VerificationContractError(RuntimeError):
    """A verifier violated ordered, one-result-per-request batch semantics."""


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    """One candidate judgment requested by a Workflow step.

    ``request_id`` is the authoritative identity.  Adapters must copy it to the
    corresponding downstream task/check and return it unchanged.  ``problem`` and
    ``candidate`` are intentionally plain text: dataset conversion and Workflow
    state projection happen above Reasoning.  ``candidate_ref`` identifies the
    producing in-process result when one exists; it is intentionally a string until
    Common freezes a durable reference contract. Branch/round/sample/routing fields
    carry execution identity through an AgentVerifier without asking it to interpret
    arbitrary metadata. ``metadata`` is recursively immutable JSON attribution
    only and must not override any authoritative field.
    """

    request_id: str
    problem: str
    candidate: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_ref: str | None = None
    branch_id: str = "main"
    round_index: int = 0
    sample_id: int = 0
    sampling_seed: int | None = None
    routing_key: str = ""
    candidate_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.problem, str):
            raise TypeError("problem must be a string")
        if not isinstance(self.candidate, str):
            raise TypeError("candidate must be a string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if self.candidate_ref is not None and (
            not isinstance(self.candidate_ref, str) or not self.candidate_ref.strip()
        ):
            raise ValueError("candidate_ref must be a non-empty string when provided")
        if not isinstance(self.branch_id, str) or not self.branch_id.strip():
            raise ValueError("branch_id must be a non-empty string")
        _require_non_negative_int(self.round_index, "round_index")
        _require_non_negative_int(self.sample_id, "sample_id")
        if self.sampling_seed is not None and (
            isinstance(self.sampling_seed, bool) or not isinstance(self.sampling_seed, int)
        ):
            raise TypeError("sampling_seed must be an integer when provided")
        if not isinstance(self.routing_key, str):
            raise TypeError("routing_key must be a string")
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_mapping(self.metadata, where="metadata"),
        )
        object.__setattr__(self, "candidate_sha256", _candidate_fingerprint(self.candidate))


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Normalized verifier output with optional evidence and agent provenance.

    ``candidate`` and ``candidate_ref`` bind the judgment to the exact input and
    its producing result; ``candidate_sha256`` is derived locally and cannot be
    supplied by a verifier.  Batch validation rejects any binding that differs
    from the corresponding :class:`VerificationRequest`.

    ``agent_result`` is retained by identity rather than projected into another
    wrapper.  This keeps the complete agent trajectory reachable from the Workflow
    result while deterministic verifiers simply leave it as ``None``.  Certification
    facts have one typed owner here rather than being duplicated in ``details``;
    ``details`` is normalized to recursively immutable JSON audit data.

    ``witness`` is deliberately opaque until Common owns a durable artifact-reference
    contract under #186.  A deterministic verifier may expose the handle returned by
    its injected store; Workflow orchestration must not interpret that handle.
    """

    request_id: str
    verdict: VerificationVerdict | str
    candidate: str
    candidate_ref: str | None = None
    candidate_sha256: str = field(init=False)
    feedback: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    agent_result: AgentResult | None = None
    trust_level: int = 0
    false_positive_risk: float = 1.0
    witness: object | None = None
    certified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.verdict, str):
            raise TypeError("verdict must be a string")
        verdict = self.verdict.strip().lower()
        if verdict not in _VERDICTS:
            raise ValueError("verdict must be pass, fail, or inconclusive")
        if not isinstance(self.candidate, str):
            raise TypeError("candidate must be a string")
        if self.candidate_ref is not None and (
            not isinstance(self.candidate_ref, str) or not self.candidate_ref.strip()
        ):
            raise ValueError("candidate_ref must be a non-empty string when provided")
        if not isinstance(self.feedback, str):
            raise TypeError("feedback must be a string")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        if self.agent_result is not None and not isinstance(self.agent_result, AgentResult):
            raise TypeError("agent_result must be an AgentResult when provided")
        if isinstance(self.trust_level, bool) or not isinstance(self.trust_level, int):
            raise TypeError("trust_level must be an int")
        if not 0 <= self.trust_level <= 4:
            raise ValueError("trust_level must be in [0, 4]")
        if isinstance(self.false_positive_risk, bool) or not isinstance(
            self.false_positive_risk, (int, float)
        ):
            raise TypeError("false_positive_risk must be numeric")
        if not 0.0 <= self.false_positive_risk <= 1.0:
            raise ValueError("false_positive_risk must be in [0, 1]")
        if not isinstance(self.certified, bool):
            raise TypeError("certified must be a bool")
        if self.certified and (
            verdict != "pass"
            or self.trust_level < 2
            or self.false_positive_risk != 0.0
            or self.witness is None
        ):
            raise ValueError(
                "certified results require pass, trust_level >= 2, "
                "false_positive_risk == 0, and a witness"
            )
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "candidate_sha256", _candidate_fingerprint(self.candidate))
        object.__setattr__(self, "false_positive_risk", float(self.false_positive_risk))
        object.__setattr__(
            self,
            "details",
            _freeze_json_mapping(self.details, where="details"),
        )


class Verifier(ABC):
    """Batch-first semantic interface consumed by ``WorkflowExecutor``."""

    def select_candidate_ref(self, request: VerificationRequest) -> str | None:
        """Select a previously produced candidate before a Workflow judgment.

        The default preserves the candidate already chosen by the Workflow.
        Stateful Verifiers may return another prior ``AgentResult.task_id``;
        ``WorkflowExecutor`` resolves that reference from the current branch
        history and remains the sole owner of the corresponding candidate text.
        Direct calls to :meth:`verify` do not perform history selection.
        """

        if not isinstance(request, VerificationRequest):
            raise TypeError("request must be a VerificationRequest")
        return request.candidate_ref

    @abstractmethod
    def verify_batch(
        self,
        requests: Sequence[VerificationRequest],
    ) -> list[VerificationResult]:
        """Return one ordered result with matching identity per request."""

    def verify(self, request: VerificationRequest) -> VerificationResult:
        """Batch-size-one convenience method with contract validation."""

        if not isinstance(request, VerificationRequest):
            raise TypeError("request must be a VerificationRequest")
        results = self.verify_batch((request,))
        _validate_result_batch((request,), results, verifier=type(self).__name__)
        return results[0]


def _validate_result_batch(
    requests: Sequence[VerificationRequest],
    results: object,
    *,
    verifier: str,
) -> list[VerificationResult]:
    """Validate the cardinality, type, identity, and order of a verifier batch."""

    if not isinstance(results, list):
        raise VerificationContractError(f"{verifier}.verify_batch() must return a list")
    if len(results) != len(requests):
        raise VerificationContractError(
            f"{verifier}.verify_batch() returned {len(results)} results for "
            f"{len(requests)} requests"
        )
    for index, (request, result) in enumerate(zip(requests, results, strict=True)):
        if not isinstance(result, VerificationResult):
            raise VerificationContractError(
                f"{verifier}.verify_batch() result {index} is not a VerificationResult"
            )
        if result.request_id != request.request_id:
            raise VerificationContractError(
                f"{verifier}.verify_batch() changed request identity at index {index}: "
                f"expected {request.request_id!r}, got {result.request_id!r}"
            )
        if result.candidate != request.candidate:
            raise VerificationContractError(
                f"{verifier}.verify_batch() changed candidate at index {index}"
            )
        if result.candidate_ref != request.candidate_ref:
            raise VerificationContractError(
                f"{verifier}.verify_batch() changed candidate_ref at index {index}: "
                f"expected {request.candidate_ref!r}, got {result.candidate_ref!r}"
            )
        if result.candidate_sha256 != request.candidate_sha256:
            raise VerificationContractError(
                f"{verifier}.verify_batch() changed candidate digest at index {index}"
            )
        if result.certified:
            # Import lazily to keep the base records independent of the witness
            # implementation during module initialization.
            from alphaapollo.reasoning.verification.witness import request_fingerprint

            witness_request_sha256 = getattr(result.witness, "request_sha256", None)
            try:
                expected_request_sha256 = request_fingerprint(request)
            except (TypeError, ValueError) as exc:
                raise VerificationContractError(
                    f"{verifier}.verify_batch() certified an unfingerprintable request "
                    f"at index {index}"
                ) from exc
            if witness_request_sha256 != expected_request_sha256:
                raise VerificationContractError(
                    f"{verifier}.verify_batch() returned a certificate not bound to "
                    f"the request at index {index}"
                )
    return results


def _candidate_fingerprint(candidate: str) -> str:
    return hashlib.sha256(candidate.encode("utf-8")).hexdigest()


def _require_non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


__all__ = [
    "VerificationContractError",
    "VerificationRequest",
    "VerificationResult",
    "VerificationVerdict",
    "Verifier",
]
