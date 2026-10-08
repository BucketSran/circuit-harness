"""Batch-first agent and deterministic verification interfaces."""

from alphaapollo.reasoning.verification.agent import (
    AgentVerifier,
    AgentVerifierConfig,
)
from alphaapollo.reasoning.verification.base import (
    VerificationContractError,
    VerificationRequest,
    VerificationResult,
    VerificationVerdict,
    Verifier,
)
from alphaapollo.reasoning.verification.deterministic import (
    DeterministicCheck,
    DeterministicVerificationError,
    DeterministicVerifier,
    PersistedDeterministicChecker,
)
from alphaapollo.reasoning.verification.registry import (
    CustomVerifierFactory,
    available_custom_verifiers,
    create_custom_verifier,
    get_custom_verifier_factory,
    register_custom_verifier,
)
from alphaapollo.reasoning.verification.witness import (
    WITNESS_KIND,
    WITNESS_MEDIA_TYPE,
    WITNESS_SCHEMA_VERSION,
    RecomputeResult,
    ValidatedWitness,
    WitnessRecorder,
    request_fingerprint,
)

__all__ = [
    "AgentVerifier",
    "AgentVerifierConfig",
    "CustomVerifierFactory",
    "DeterministicCheck",
    "DeterministicVerificationError",
    "DeterministicVerifier",
    "PersistedDeterministicChecker",
    "RecomputeResult",
    "ValidatedWitness",
    "VerificationContractError",
    "VerificationRequest",
    "VerificationResult",
    "VerificationVerdict",
    "Verifier",
    "WITNESS_KIND",
    "WITNESS_MEDIA_TYPE",
    "WITNESS_SCHEMA_VERSION",
    "WitnessRecorder",
    "available_custom_verifiers",
    "create_custom_verifier",
    "get_custom_verifier_factory",
    "register_custom_verifier",
    "request_fingerprint",
]
