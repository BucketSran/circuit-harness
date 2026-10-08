# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contract tests for explicitly registered custom Verifiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from alphaapollo.reasoning.verification import (
    VerificationRequest,
    VerificationResult,
    Verifier,
    available_custom_verifiers,
    create_custom_verifier,
    get_custom_verifier_factory,
    register_custom_verifier,
)


class _ConfiguredVerifier(Verifier):
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
        return [
            VerificationResult(
                request_id=request.request_id,
                verdict="pass",
                candidate=request.candidate,
                candidate_ref=request.candidate_ref,
            )
            for request in requests
        ]


def test_registered_factory_receives_config_and_returns_fresh_verifiers() -> None:
    name = "tests.registry.configured"
    register_custom_verifier(name, _ConfiguredVerifier, replace=True)

    first = create_custom_verifier(name, {"limit": 3})
    second = create_custom_verifier(name, {"limit": 4})

    assert isinstance(first, _ConfiguredVerifier)
    assert first.config == {"limit": 3}
    assert isinstance(second, _ConfiguredVerifier)
    assert second.config == {"limit": 4}
    assert first is not second
    assert get_custom_verifier_factory(name) is _ConfiguredVerifier
    assert name in available_custom_verifiers()


def test_duplicate_registration_is_refused() -> None:
    name = "tests.registry.duplicate"
    register_custom_verifier(name, _ConfiguredVerifier, replace=True)

    with pytest.raises(ValueError, match="already registered"):
        register_custom_verifier(name, _ConfiguredVerifier)


def test_unknown_name_reports_available_registrations() -> None:
    with pytest.raises(ValueError, match="unknown custom Verifier.*available"):
        get_custom_verifier_factory("tests.registry.missing")


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_name_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        register_custom_verifier(name, _ConfiguredVerifier)


def test_non_verifier_factory_result_is_refused() -> None:
    name = "tests.registry.wrong-result"
    register_custom_verifier(name, lambda _config: object(), replace=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="expected Verifier"):
        create_custom_verifier(name)
