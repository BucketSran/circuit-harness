from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from alphaapollo.reasoning.verification import (
    DeterministicCheck,
    DeterministicVerifier,
    PersistedDeterministicChecker,
    RecomputeResult,
    ValidatedWitness,
    VerificationRequest,
    VerificationResult,
    request_fingerprint,
)


@dataclass(frozen=True)
class Ref:
    id: str
    hash: str
    type: str
    created_by: str


class MemoryWitnessStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def put_blob(self, content: bytes, *, type_: str, created_by: str) -> Ref:
        digest = hashlib.sha256(content).hexdigest()
        self.blobs[digest] = content
        return Ref(id=digest, hash=digest, type=type_, created_by=created_by)

    def get_blob(self, ref: Ref) -> bytes:
        return self.blobs[ref.id]


def request(candidate: str = "The candidate derives the answer as 42.") -> VerificationRequest:
    return VerificationRequest("candidate-1", "Compute the answer.", candidate)


def test_persisted_deterministic_evidence_is_the_only_certifying_path() -> None:
    seen: list[VerificationRequest] = []

    def recompute(item: VerificationRequest) -> RecomputeResult:
        seen.append(item)
        return RecomputeResult(
            passed=True,
            trust_level=2,
            false_positive_risk=0.0,
            feedback="independent arithmetic matched",
            recompute_log=("evaluate the public expression exactly",),
        )

    checker = PersistedDeterministicChecker(
        recompute,
        store=MemoryWitnessStore(),
        checker_id="test.exact",
    )
    result = DeterministicVerifier(checker).verify(request())

    assert result.verdict == "pass"
    assert result.trust_level == 2
    assert result.false_positive_risk == 0.0
    assert result.certified is True
    assert result.witness is not None
    assert result.candidate == request().candidate
    assert result.candidate_sha256 == request().candidate_sha256
    assert seen == [request()]


def test_deterministic_defaults_and_unpersisted_checks_cannot_certify() -> None:
    bare = DeterministicVerifier(lambda _request: True).verify(request())
    unpersisted = DeterministicVerifier(
        lambda _request: DeterministicCheck(
            passed=True,
            trust_level=2,
            false_positive_risk=0.0,
        )
    ).verify(request())

    assert bare.trust_level == 0
    assert bare.false_positive_risk == 1.0
    assert bare.certified is False
    assert unpersisted.verdict == "pass"
    assert unpersisted.certified is False


def test_arbitrary_checker_cannot_certify_a_forged_validated_witness() -> None:
    item = request()
    forged = ValidatedWitness(
        reference=Ref(
            id="forged",
            hash="forged",
            type="application/vnd.alphaapollo.verification-witness+json",
            created_by="reasoning.verification.deterministic:forged",
        ),
        request_sha256=request_fingerprint(item),
        checker_id="forged",
        checker_version="1",
    )

    result = DeterministicVerifier(
        lambda _request: DeterministicCheck(
            passed=True,
            witness=forged,
            trust_level=2,
            false_positive_risk=0.0,
        )
    ).verify(item)

    assert result.verdict == "pass"
    assert result.certified is False


def test_deterministic_mapping_string_false_fails_closed() -> None:
    result = DeterministicVerifier(
        lambda _request: {
            "passed": "false",
            "trust_level": 2,
            "false_positive_risk": 0.0,
        }
    ).verify(request())

    assert result.verdict == "fail"
    assert result.certified is False


def test_verification_result_rejects_impossible_certification_state() -> None:
    with pytest.raises(ValueError, match="certified results require"):
        VerificationResult(
            request_id="candidate-1",
            verdict="pass",
            candidate="The candidate derives the answer as 42.",
            trust_level=2,
            false_positive_risk=0.0,
            certified=True,
        )
