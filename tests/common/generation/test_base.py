"""The Generation contract: flat one-request-one-response, honest token labelling.

Pins the #185 core: no nested candidates, request identity echoed one-to-one,
token fields present-or-None (never fabricated), the trainable gate keyed on
token-native capability, and the tool-call passthrough carrying only id/name/raw
args.
"""

from __future__ import annotations

import math

import pytest

from alphaapollo.common.generation import (
    BATCH_EXECUTION_CLIENT_CONCURRENT,
    BackendCapabilities,
    CapabilityError,
    GenerationBackend,
    GenerationError,
    GenerationRequest,
    GenerationResponse,
    Provenance,
    SamplingOptions,
    ToolCall,
    fingerprint,
    require_trainable,
    verify_batch_identity,
)


def _req(request_id: str = "r0", *, group_id: str = "g", sample_id: int = 0) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        model="qwen",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingOptions(temperature=0.7, max_tokens=8),
        group_id=group_id,
        sample_id=sample_id,
    )


# --- request/sampling ----------------------------------------------------------


def test_sampling_options_have_no_n() -> None:
    opts = SamplingOptions(temperature=0.7, max_tokens=8)
    assert not hasattr(opts, "n")
    with pytest.raises(ValueError):
        SamplingOptions(temperature=math.inf, max_tokens=8)
    with pytest.raises(ValueError):
        SamplingOptions(temperature=0.7, max_tokens=0)


def test_request_requires_identity() -> None:
    with pytest.raises(ValueError, match="request_id"):
        GenerationRequest(
            request_id=" ",
            model="m",
            messages=[{"role": "user", "content": "x"}],
            sampling=SamplingOptions(temperature=0.0, max_tokens=4),
        )


# --- response token honesty ----------------------------------------------------


def test_text_only_response_is_not_trainable() -> None:
    resp = GenerationResponse(request_id="r0", content="hello")
    assert not resp.is_trainable
    assert resp.prompt_token_ids is None and resp.response_token_ids is None


def test_token_native_response_is_trainable() -> None:
    resp = GenerationResponse(
        request_id="r0",
        content="hi",
        prompt_token_ids=[1, 2, 3],
        response_token_ids=[10, 11],
        response_logprobs=[-0.1, -0.2],
        provenance=Provenance(policy_model="qwen", tokenizer_id="qwen-tok"),
    )
    assert resp.is_trainable


def test_response_rejects_misaligned_token_span() -> None:
    with pytest.raises(ValueError, match="same length"):
        GenerationResponse(
            request_id="r0",
            content="hi",
            response_token_ids=[10, 11, 12],
            response_logprobs=[-0.1],
        )


def test_response_rejects_bool_token_ids() -> None:
    with pytest.raises(TypeError):
        GenerationResponse(
            request_id="r0",
            content="hi",
            response_token_ids=[True, 2],
            response_logprobs=[-0.1, -0.2],
        )


# --- tool call passthrough -----------------------------------------------------


def test_tool_call_is_passthrough_only() -> None:
    call = ToolCall(id="call_1", name="python", arguments='{"code": "print(1)"}')
    resp = GenerationResponse(request_id="r0", content="", tool_calls=(call,))
    assert resp.has_tool_calls
    assert resp.tool_calls[0].id == "call_1"  # provider id preserved for multi-turn echo
    with pytest.raises(ValueError):
        ToolCall(id="", name="python", arguments="{}")


# --- capabilities / gate -------------------------------------------------------


def test_capabilities_reject_token_native_without_logprobs() -> None:
    with pytest.raises(ValueError, match="must also report logprobs"):
        BackendCapabilities(
            token_native=True, logprobs=False, tool_calls=False, policy_source="remote_engine"
        )


def test_capabilities_validate_batch_execution_contract() -> None:
    with pytest.raises(ValueError, match="batch_execution"):
        BackendCapabilities(
            token_native=False,
            logprobs=False,
            tool_calls=False,
            policy_source="remote_engine",
            batch_execution="parallel-ish",
        )
    with pytest.raises(ValueError, match="sequential.*max_concurrency"):
        BackendCapabilities(
            token_native=False,
            logprobs=False,
            tool_calls=False,
            policy_source="remote_engine",
            max_concurrency=2,
        )
    caps = BackendCapabilities(
        token_native=False,
        logprobs=False,
        tool_calls=False,
        policy_source="remote_engine",
        batch_execution=BATCH_EXECUTION_CLIENT_CONCURRENT,
        max_concurrency=2,
    )
    assert caps.max_concurrency == 2


class _FakeBackend(GenerationBackend):
    def __init__(self, caps: BackendCapabilities) -> None:
        self._caps = caps

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._caps

    def _generate_batch(self, requests):
        return [
            GenerationResponse(
                request_id=r.request_id, content="ok", group_id=r.group_id, sample_id=r.sample_id
            )
            for r in requests
        ]


def test_gate_rejects_non_trainable_backend() -> None:
    backend = _FakeBackend(
        BackendCapabilities(
            token_native=False, logprobs=False, tool_calls=True, policy_source="remote_engine"
        )
    )
    with pytest.raises(CapabilityError, match="not trainable"):
        require_trainable(backend)


def test_gate_accepts_token_native_remote_backend() -> None:
    # remote HTTP, yet token-native -> trainable (the return_token_ids / SGLang case)
    backend = _FakeBackend(
        BackendCapabilities(
            token_native=True, logprobs=True, tool_calls=False, policy_source="remote_engine"
        )
    )
    caps = require_trainable(backend)
    assert caps.trainable


# --- batch semantics -----------------------------------------------------------


def test_generate_convenience_delegates_to_batch() -> None:
    backend = _FakeBackend(
        BackendCapabilities(
            token_native=False, logprobs=False, tool_calls=False, policy_source="remote_engine"
        )
    )
    resp = backend.generate(_req("r7", sample_id=3))
    assert resp.request_id == "r7" and resp.sample_id == 3


def test_verify_batch_identity_catches_mismatch() -> None:
    reqs = [_req("r0", sample_id=0), _req("r1", sample_id=1)]
    good = [
        GenerationResponse(
            request_id=r.request_id, content="x", group_id=r.group_id, sample_id=r.sample_id
        )
        for r in reqs
    ]
    verify_batch_identity(reqs, good)  # no raise

    swapped = [good[1], good[0]]
    with pytest.raises(GenerationError, match="request_id"):
        verify_batch_identity(reqs, swapped)

    with pytest.raises(GenerationError, match="responses for"):
        verify_batch_identity(reqs, good[:1])


def test_fingerprint_is_stable_and_short() -> None:
    a = fingerprint("qwen", "rev1", "template-x")
    b = fingerprint("qwen", "rev1", "template-x")
    c = fingerprint("qwen", "rev2", "template-x")
    assert a == b and a != c and len(a) == 16
