"""Canonical, import-light Reasoning Runtime and Verification surface.

Names resolve lazily so importing this package does not pull in the Common
integrations either subpackage needs. Scoring, resume, reporting, and workflow
topology belong to ``alphaapollo.workflows``; superseded Runtime, Workflow,
solver/session, and certification engines have no compatibility exports or
import paths.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_RUNTIME_EXPORTS = {
    name: ("alphaapollo.reasoning.runtime", name)
    for name in (
        "AgentResult",
        "AgentRuntime",
        "AgentTask",
        "AgentTurn",
        "AlphaApolloAgentRuntime",
        "EnvironmentProjector",
    )
}

_VERIFICATION_EXPORTS = {
    name: ("alphaapollo.reasoning.verification", name)
    for name in (
        "AgentVerifier",
        "AgentVerifierConfig",
        "VerificationContractError",
        "VerificationRequest",
        "VerificationResult",
        "Verifier",
    )
}

_EXPORTS = {**_RUNTIME_EXPORTS, **_VERIFICATION_EXPORTS}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
