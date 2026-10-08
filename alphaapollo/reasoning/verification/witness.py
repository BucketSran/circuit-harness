"""Durable, candidate-bound evidence for deterministic verification.

The store is injected and intentionally duck-typed.  Verification owns the
witness document and validation rules, but it does not freeze the durable store
or artifact-reference interface that belongs to Common under #186.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from alphaapollo.reasoning._immutable import _freeze_json_mapping, _thaw_json
from alphaapollo.reasoning.verification.base import VerificationRequest

WITNESS_MEDIA_TYPE = "application/vnd.alphaapollo.verification-witness+json"
WITNESS_KIND = "deterministic_recompute"
WITNESS_SCHEMA_VERSION = "1"
_CREATED_BY_PREFIX = "reasoning.verification.deterministic:"


@dataclass(frozen=True, slots=True)
class RecomputeResult:
    """An independently recomputed result with immutable audit details."""

    passed: bool
    recompute_log: tuple[str, ...] = ()
    feedback: str = ""
    trust_level: int = 0
    false_positive_risk: float = 1.0
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")
        if not isinstance(self.feedback, str):
            raise TypeError("feedback must be a string")
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
        log = tuple(self.recompute_log)
        if any(not isinstance(line, str) or not line.strip() for line in log):
            raise ValueError("recompute_log entries must be non-empty strings")
        object.__setattr__(self, "recompute_log", log)
        object.__setattr__(self, "false_positive_risk", float(self.false_positive_risk))
        object.__setattr__(
            self,
            "details",
            _freeze_json_mapping(self.details, where="details"),
        )


@dataclass(frozen=True, slots=True)
class ValidatedWitness:
    """Opaque store handle whose bytes and request binding were validated."""

    reference: object
    request_sha256: str
    checker_id: str
    checker_version: str

    def __post_init__(self) -> None:
        for name in ("request_sha256", "checker_id", "checker_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        _reference_value(self.reference, "id")
        _reference_value(self.reference, "hash")
        _reference_value(self.reference, "type")
        _reference_value(self.reference, "created_by")


class WitnessRecorder:
    """Persist and revalidate canonical recomputation evidence."""

    def __init__(
        self,
        store: object,
        *,
        checker_id: str,
        checker_version: str = "1",
    ) -> None:
        if not isinstance(checker_id, str) or not checker_id.strip():
            raise ValueError("checker_id must be a non-empty string")
        if not isinstance(checker_version, str) or not checker_version.strip():
            raise ValueError("checker_version must be a non-empty string")
        for method in ("put_blob", "get_blob"):
            operation = getattr(store, method, None)
            if operation is None or not callable(operation):
                raise TypeError(f"store must provide {method}()")
        self._store = store
        self.checker_id = checker_id
        self.checker_version = checker_version

    @property
    def created_by(self) -> str:
        return f"{_CREATED_BY_PREFIX}{self.checker_id}"

    def record(
        self,
        request: VerificationRequest,
        result: RecomputeResult,
    ) -> ValidatedWitness:
        """Write a witness and return it only after a complete read-back check."""

        if not isinstance(request, VerificationRequest):
            raise TypeError("request must be a VerificationRequest")
        if not isinstance(result, RecomputeResult):
            raise TypeError("result must be a RecomputeResult")
        if (
            result.passed
            and result.trust_level >= 2
            and result.false_positive_risk == 0.0
            and not result.recompute_log
        ):
            raise ValueError("a certifying deterministic result requires a recompute log")

        request_sha256 = request_fingerprint(request)
        document = {
            "schema_version": WITNESS_SCHEMA_VERSION,
            "kind": WITNESS_KIND,
            "producer_kind": "deterministic_checker",
            "checker": {"id": self.checker_id, "version": self.checker_version},
            "candidate": {
                "request_id": request.request_id,
                "candidate_ref": request.candidate_ref,
                "candidate_sha256": request.candidate_sha256,
                "request_sha256": request_sha256,
            },
            "result": {
                "passed": result.passed,
                "trust_level": result.trust_level,
                "false_positive_risk": result.false_positive_risk,
                "feedback": result.feedback,
            },
            "recompute_log": list(result.recompute_log),
            "details": _thaw_json(result.details),
        }
        reference = self._store.put_blob(
            _canonical_json(document),
            type_=WITNESS_MEDIA_TYPE,
            created_by=self.created_by,
        )
        witness = ValidatedWitness(
            reference=reference,
            request_sha256=request_sha256,
            checker_id=self.checker_id,
            checker_version=self.checker_version,
        )
        self.read(witness, request=request)
        return witness

    def read(
        self,
        witness: ValidatedWitness | object,
        *,
        request: VerificationRequest | None = None,
    ) -> dict[str, object]:
        """Resolve and validate a witness produced by this recorder."""

        reference = witness.reference if isinstance(witness, ValidatedWitness) else witness
        if _reference_value(reference, "type") != WITNESS_MEDIA_TYPE:
            raise ValueError("artifact is not a deterministic verifier witness")
        if _reference_value(reference, "created_by") != self.created_by:
            raise ValueError("witness producer does not match this deterministic checker")
        content = self._store.get_blob(reference)
        if not isinstance(content, bytes):
            raise TypeError("store.get_blob() must return bytes")
        digest = hashlib.sha256(content).hexdigest()
        if (
            _reference_value(reference, "id") != digest
            or _reference_value(reference, "hash") != digest
        ):
            raise ValueError("witness content does not match its sha256 reference")
        try:
            document = json.loads(content)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("witness is not valid UTF-8 JSON") from exc
        if not isinstance(document, dict):
            raise ValueError("witness document must be a JSON object")
        if content != _canonical_json(document):
            raise ValueError("witness document is not canonical JSON")
        if (
            document.get("schema_version") != WITNESS_SCHEMA_VERSION
            or document.get("kind") != WITNESS_KIND
            or document.get("producer_kind") != "deterministic_checker"
            or document.get("checker") != {"id": self.checker_id, "version": self.checker_version}
        ):
            raise ValueError("witness document metadata does not match the recorder")
        candidate = document.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("witness candidate binding is malformed")
        if isinstance(witness, ValidatedWitness):
            if (
                witness.checker_id != self.checker_id
                or witness.checker_version != self.checker_version
                or candidate.get("request_sha256") != witness.request_sha256
            ):
                raise ValueError("validated witness metadata does not match its document")
        if request is not None:
            expected = {
                "request_id": request.request_id,
                "candidate_ref": request.candidate_ref,
                "candidate_sha256": request.candidate_sha256,
                "request_sha256": request_fingerprint(request),
            }
            if candidate != expected:
                raise ValueError("witness is not bound to this verification request")
        return document


def request_fingerprint(request: VerificationRequest) -> str:
    """Hash the complete public verification request using canonical JSON."""

    if not isinstance(request, VerificationRequest):
        raise TypeError("request must be a VerificationRequest")
    payload = {
        "request_id": request.request_id,
        "problem": request.problem,
        "candidate": request.candidate,
        "candidate_ref": request.candidate_ref,
        "branch_id": request.branch_id,
        "round_index": request.round_index,
        "sample_id": request.sample_id,
        "sampling_seed": request.sampling_seed,
        "routing_key": request.routing_key,
        "metadata": _thaw_json(request.metadata),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _reference_value(reference: object, name: str) -> str:
    value = (
        reference.get(name) if isinstance(reference, Mapping) else getattr(reference, name, None)
    )
    if not isinstance(value, str) or not value:
        raise TypeError(f"witness reference must provide non-empty {name}")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("witness content must be canonical JSON") from exc
    return encoded.encode("utf-8")


__all__ = [
    "RecomputeResult",
    "ValidatedWitness",
    "WITNESS_KIND",
    "WITNESS_MEDIA_TYPE",
    "WITNESS_SCHEMA_VERSION",
    "WitnessRecorder",
    "request_fingerprint",
]
