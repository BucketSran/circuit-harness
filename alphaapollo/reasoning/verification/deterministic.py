"""Deterministic recomputation and the sole certification policy."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from alphaapollo.reasoning._immutable import _freeze_json_mapping
from alphaapollo.reasoning.verification.base import (
    VerificationRequest,
    VerificationResult,
    Verifier,
    _validate_result_batch,
)
from alphaapollo.reasoning.verification.witness import (
    RecomputeResult,
    ValidatedWitness,
    WitnessRecorder,
    request_fingerprint,
)

_SENSITIVE_FEEDBACK_RE = re.compile(
    r"(?:official|expected|ground[- ]truth|gold)\s*"
    r"(?:answer|value|result)?\s*[:=]?\s*[^.;\n]*",
    re.IGNORECASE,
)
_MAX_FEEDBACK = 4096


class DeterministicVerificationError(RuntimeError):
    """Expected checker failure carrying safe, structured audit details."""

    def __init__(
        self,
        feedback: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(feedback)
        self.feedback = _sanitize_feedback(feedback)
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class DeterministicCheck:
    """A checker result with immutable details after binding validation."""

    passed: bool
    feedback: str = ""
    witness: ValidatedWitness | None = None
    trust_level: int = 0
    false_positive_risk: float = 1.0
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")
        if not isinstance(self.feedback, str):
            raise TypeError("feedback must be a string")
        if self.witness is not None and not isinstance(self.witness, ValidatedWitness):
            raise TypeError("witness must be a ValidatedWitness")
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
        object.__setattr__(self, "false_positive_risk", float(self.false_positive_risk))
        object.__setattr__(
            self,
            "details",
            _freeze_json_mapping(self.details, where="details"),
        )


class PersistedDeterministicChecker:
    """Run independent recomputation and durably bind its witness to a request."""

    def __init__(
        self,
        recompute_fn: object,
        *,
        store: object,
        checker_id: str,
        checker_version: str = "1",
    ) -> None:
        if not callable(recompute_fn):
            raise TypeError("recompute_fn must be callable")
        self._recompute_fn = recompute_fn
        self.recorder = WitnessRecorder(
            store,
            checker_id=checker_id,
            checker_version=checker_version,
        )

    def __call__(self, request: VerificationRequest) -> DeterministicCheck:
        result = _coerce_recompute_result(self._recompute_fn(request))
        witness = self.recorder.record(request, result)
        return DeterministicCheck(
            passed=result.passed,
            feedback=result.feedback,
            witness=witness,
            trust_level=result.trust_level,
            false_positive_risk=result.false_positive_risk,
            details={
                **dict(result.details),
                "checker_id": self.recorder.checker_id,
                "checker_version": self.recorder.checker_version,
                "witness_kind": "deterministic_recompute",
            },
        )


class DeterministicVerifier(Verifier):
    """Batch verifier with a fail-closed, candidate-bound FP=0 gate."""

    verifier_id = "reasoning.deterministic"
    verifier_version = "1"

    def __init__(self, check_fn: object, *, verifier_id: str | None = None) -> None:
        if not callable(check_fn):
            raise TypeError("check_fn must be callable")
        self._check_fn = check_fn
        if verifier_id is not None:
            if not isinstance(verifier_id, str) or not verifier_id.strip():
                raise ValueError("verifier_id must be a non-empty string")
            self.verifier_id = verifier_id

    def verify_batch(
        self,
        requests: Sequence[VerificationRequest],
    ) -> list[VerificationResult]:
        materialized = _coerce_requests(requests)
        results = [self._verify_one(request) for request in materialized]
        return _validate_result_batch(materialized, results, verifier=type(self).__name__)

    def _verify_one(self, request: VerificationRequest) -> VerificationResult:
        try:
            check = _coerce_check(self._check_fn(request))
        except DeterministicVerificationError as exc:
            return _inconclusive(
                request,
                feedback=exc.feedback,
                details={**exc.details, "error_type": type(exc).__name__},
            )
        except Exception as exc:  # noqa: BLE001 - one failed check must not reorder a batch
            return _inconclusive(
                request,
                feedback="deterministic check failed",
                details={
                    "error_type": type(exc).__name__,
                    "failure_class": "timeout" if isinstance(exc, TimeoutError) else "execution",
                },
            )

        witness_bound = self._witness_is_valid(check, request)
        certified = (
            check.passed
            and check.trust_level >= 2
            and check.false_positive_risk == 0.0
            and witness_bound
        )
        return VerificationResult(
            request_id=request.request_id,
            verdict="pass" if check.passed else "fail",
            candidate=request.candidate,
            candidate_ref=request.candidate_ref,
            feedback=_sanitize_feedback(check.feedback),
            details={**dict(check.details), "deterministic": True},
            trust_level=check.trust_level,
            false_positive_risk=check.false_positive_risk,
            witness=check.witness,
            certified=certified,
        )

    def _witness_is_valid(
        self,
        check: DeterministicCheck,
        request: VerificationRequest,
    ) -> bool:
        """Re-read persisted evidence before allowing certification.

        ``ValidatedWitness`` is a transport record, not an unforgeable token.
        Only the checker that owns the injected store can establish that the
        referenced bytes exist, match their digest, and bind to this request.
        """

        if check.witness is None or not isinstance(self._check_fn, PersistedDeterministicChecker):
            return False
        if check.witness.request_sha256 != request_fingerprint(request):
            return False
        try:
            document = self._check_fn.recorder.read(check.witness, request=request)
            return isinstance(document, Mapping) and document.get("result") == {
                "passed": check.passed,
                "trust_level": check.trust_level,
                "false_positive_risk": check.false_positive_risk,
                "feedback": check.feedback,
            }
        except Exception:  # noqa: BLE001 - evidence/store failures must fail closed
            return False


def _coerce_check(value: object) -> DeterministicCheck:
    if isinstance(value, DeterministicCheck):
        return value
    if isinstance(value, RecomputeResult):
        return DeterministicCheck(
            passed=value.passed,
            feedback=value.feedback,
            trust_level=value.trust_level,
            false_positive_risk=value.false_positive_risk,
            details=value.details,
        )
    if isinstance(value, bool):
        return DeterministicCheck(passed=value)
    if isinstance(value, Mapping):
        return DeterministicCheck(
            passed=_mapping_passed(value),
            feedback=str(value.get("feedback", "")),
            witness=value.get("witness"),
            trust_level=int(value.get("trust_level", 0)),
            false_positive_risk=float(value.get("false_positive_risk", 1.0)),
            details={
                key: item
                for key, item in value.items()
                if key
                not in {
                    "passed",
                    "pass",
                    "feedback",
                    "witness",
                    "trust_level",
                    "false_positive_risk",
                }
            },
        )
    raise TypeError("check_fn must return DeterministicCheck, RecomputeResult, bool, or a mapping")


def _coerce_recompute_result(value: object) -> RecomputeResult:
    if isinstance(value, RecomputeResult):
        return value
    if isinstance(value, bool):
        return RecomputeResult(passed=value)
    if isinstance(value, Mapping):
        log = value.get("recompute_log", ())
        if isinstance(log, str):
            log = (log,)
        return RecomputeResult(
            passed=_mapping_passed(value),
            recompute_log=tuple(str(line) for line in log or ()),
            feedback=str(value.get("feedback", "")),
            trust_level=int(value.get("trust_level", 0)),
            false_positive_risk=float(value.get("false_positive_risk", 1.0)),
            details={
                key: item
                for key, item in value.items()
                if key
                not in {
                    "passed",
                    "pass",
                    "recompute_log",
                    "feedback",
                    "trust_level",
                    "false_positive_risk",
                }
            },
        )
    raise TypeError("recompute_fn must return RecomputeResult, bool, or a mapping")


def _mapping_passed(value: Mapping[str, object]) -> bool:
    raw = value.get("passed", value.get("pass", False))
    return raw if isinstance(raw, bool) else False


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


def _inconclusive(
    request: VerificationRequest,
    *,
    feedback: str,
    details: Mapping[str, Any],
) -> VerificationResult:
    return VerificationResult(
        request_id=request.request_id,
        verdict="inconclusive",
        candidate=request.candidate,
        candidate_ref=request.candidate_ref,
        feedback=_sanitize_feedback(feedback),
        details={**dict(details), "deterministic": True},
    )


def _sanitize_feedback(text: object) -> str:
    value = str(text or "").replace("\x00", " ").strip()
    return _SENSITIVE_FEEDBACK_RE.sub("[redacted]", value)[:_MAX_FEEDBACK]


__all__ = [
    "DeterministicCheck",
    "DeterministicVerificationError",
    "DeterministicVerifier",
    "PersistedDeterministicChecker",
]
