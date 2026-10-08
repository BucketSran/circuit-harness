"""VerlLLMServerGenerationBackend: reuse verl token-in-token-out via injected client.

Fake async client shaped like verl LLMServerClient.generate + fake TokenOutput.
No verl, no Ray, no GPU. Pins token-native trainability, provenance stamping,
one-request-one-response fan-out, and fail-loud on a span without logprobs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from alphaapollo.common.generation import (
    BATCH_EXECUTION_CLIENT_CONCURRENT,
    GenerationError,
    GenerationRequest,
    Provenance,
    SamplingOptions,
    VerlLLMServerGenerationBackend,
    require_trainable,
)


@dataclass
class _FakeTokenOutput:
    token_ids: list[int]
    log_probs: list[float] | None = None
    stop_reason: str | None = "completed"


class _FakeClient:
    def __init__(self, *, log_probs: list[float] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._log_probs = log_probs if log_probs is not None else [-0.1, -0.2, -0.3]

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls.append({"prompt_ids": list(prompt_ids), "sampling": dict(sampling_params)})
        return _FakeTokenOutput(token_ids=[10, 11, 12], log_probs=list(self._log_probs))


class _FakeTokenizer:
    def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize):
        return [1, 2, 3, 4]

    def decode(self, token_ids) -> str:
        return "text:" + ",".join(str(t) for t in token_ids)


def _backend(
    client: _FakeClient,
    *,
    tokenizer=None,
    **options: Any,
) -> VerlLLMServerGenerationBackend:
    return VerlLLMServerGenerationBackend(
        client,
        tokenizer or _FakeTokenizer(),
        model_name="qwen3-0.6b",
        tokenizer_id="qwen-tok",
        weights_version="step-7",
        actor="solver",
        **options,
    )


def _req(request_id: str = "r0", *, sample_id: int = 0, tools=None) -> GenerationRequest:
    kwargs: dict[str, Any] = {
        "request_id": request_id,
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "hi"}],
        "sampling": SamplingOptions(temperature=0.6, max_tokens=16),
        "group_id": "g",
        "sample_id": sample_id,
    }
    if tools is not None:
        kwargs["tools"] = tools
    return GenerationRequest(**kwargs)


def test_backend_is_trainable() -> None:
    caps = require_trainable(_backend(_FakeClient()))
    assert caps.trainable and caps.token_native


def test_verl_batch_enforces_concurrency_bound_and_records_timing() -> None:
    class _DelayedClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.max_active = 0

        async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.03)
                return await super().generate(
                    request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    **kwargs,
                )
            finally:
                self.active -= 1

    client = _DelayedClient()
    backend = _backend(client, max_concurrency=2)
    results = backend.generate_batch([_req(f"r{i}", sample_id=i) for i in range(5)])

    assert client.max_active == 2
    assert backend.capabilities.batch_execution == BATCH_EXECUTION_CLIENT_CONCURRENT
    assert backend.capabilities.max_concurrency == 2
    assert [result.request_id for result in results] == [f"r{i}" for i in range(5)]
    assert all(
        result.backend_metadata["batch_execution"] == "client_concurrent" for result in results
    )
    assert results[-1].backend_metadata["request_timing"]["queue_s"] > 0


def test_verl_batch_failure_waits_for_every_submitted_request() -> None:
    class _FailingClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.completed = 0

        async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
            self.active += 1
            try:
                await asyncio.sleep(0.01)
                if request_id == "r1":
                    raise TimeoutError("controlled timeout")
                result = await super().generate(
                    request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    **kwargs,
                )
                self.completed += 1
                return result
            finally:
                self.active -= 1

    client = _FailingClient()
    backend = _backend(client, max_concurrency=2)
    requests = [_req(f"r{i}", sample_id=i) for i in range(4)]
    requests = [
        GenerationRequest(
            request_id=request.request_id,
            model=request.model,
            messages=request.messages,
            sampling=request.sampling,
            group_id=request.group_id,
            sample_id=request.sample_id,
            routing_key=request.request_id,
        )
        for request in requests
    ]

    with pytest.raises(TimeoutError, match="controlled timeout"):
        backend.generate_batch(requests)

    assert client.active == 0
    assert client.completed == 3


def test_backend_forces_logprobs_and_stamps_provenance() -> None:
    client = _FakeClient()
    resp = _backend(client).generate(_req("r1"))

    assert len(client.calls) == 1
    assert client.calls[0]["prompt_ids"] == [1, 2, 3, 4]
    assert client.calls[0]["sampling"]["logprobs"] is True

    assert resp.is_trainable
    assert resp.request_id == "r1"
    assert resp.prompt_token_ids == (1, 2, 3, 4)
    assert resp.response_token_ids == (10, 11, 12)
    assert resp.response_logprobs == (-0.1, -0.2, -0.3)
    assert resp.content == "text:10,11,12"
    assert resp.provenance == Provenance(
        policy_model="qwen3-0.6b", tokenizer_id="qwen-tok", weights_version="step-7", actor="solver"
    )


def test_weights_version_changes_only_when_learning_publishes_a_sync() -> None:
    backend = _backend(_FakeClient())

    assert backend.current_provenance.weights_version == "step-7"
    backend.update_weights_version("step-8")

    response = backend.generate(_req("after-sync"))
    assert response.provenance == backend.current_provenance
    assert response.provenance.weights_version == "step-8"

    with pytest.raises(ValueError, match="non-empty"):
        backend.update_weights_version("")


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("left", [2, 3, 4, 5]),
        ("right", [1, 2, 3, 4]),
        ("middle", [1, 2, 4, 5]),
    ],
)
def test_prompt_limit_is_applied_before_generation(policy, expected) -> None:
    class _LongTokenizer(_FakeTokenizer):
        def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize):
            return [1, 2, 3, 4, 5]

    client = _FakeClient()
    response = _backend(
        client,
        tokenizer=_LongTokenizer(),
        max_prompt_tokens=4,
        prompt_truncation=policy,
    ).generate(_req("long"))

    assert client.calls[0]["prompt_ids"] == expected
    assert response.prompt_token_ids == tuple(expected)


def test_prompt_error_policy_fails_before_generation() -> None:
    class _LongTokenizer(_FakeTokenizer):
        def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize):
            return [1, 2, 3, 4, 5]

    client = _FakeClient()
    backend = _backend(
        client,
        tokenizer=_LongTokenizer(),
        max_prompt_tokens=4,
        prompt_truncation="error",
    )

    with pytest.raises(GenerationError, match="exceeding the configured limit"):
        backend.generate(_req("long"))
    assert client.calls == []


def test_prompt_limit_covers_accumulated_multi_turn_history() -> None:
    class _HistoryTokenizer(_FakeTokenizer):
        def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize):
            assert len(conversation) == 3
            return [1, 2, 3, 4, 5, 6]

    request = GenerationRequest(
        request_id="multi-turn",
        model="qwen3-0.6b",
        messages=[
            {"role": "user", "content": "first observation"},
            {"role": "assistant", "content": "first action"},
            {"role": "user", "content": "second observation"},
        ],
        sampling=SamplingOptions(temperature=0.6, max_tokens=16),
        group_id="g",
        sample_id=0,
    )
    client = _FakeClient()

    response = _backend(
        client,
        tokenizer=_HistoryTokenizer(),
        max_prompt_tokens=4,
        prompt_truncation="left",
    ).generate(request)

    assert client.calls[0]["prompt_ids"] == [3, 4, 5, 6]
    assert response.prompt_token_ids == (3, 4, 5, 6)


def test_batch_fans_out_one_call_per_request_in_order() -> None:
    client = _FakeClient()
    reqs = [_req("a", sample_id=0), _req("b", sample_id=1), _req("c", sample_id=2)]
    out = _backend(client).generate_batch(reqs)
    assert len(client.calls) == 3
    assert [r.request_id for r in out] == ["a", "b", "c"]
    assert all(r.is_trainable for r in out)


def test_batch_fans_out_concurrently() -> None:
    class _BarrierClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = 0
            self.release = asyncio.Event()

        async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
            self.started += 1
            if self.started == 2:
                self.release.set()
            await asyncio.wait_for(self.release.wait(), timeout=1.0)
            return await super().generate(
                request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )

    client = _BarrierClient()
    output = _backend(client).generate_batch([_req("a", sample_id=0), _req("b", sample_id=1)])
    assert [item.request_id for item in output] == ["a", "b"]


def test_provider_options_reach_verl_without_overriding_standard_keys() -> None:
    client = _FakeClient()
    request = _req("provider")
    request = GenerationRequest(
        request_id=request.request_id,
        model=request.model,
        messages=request.messages,
        sampling=request.sampling,
        group_id=request.group_id,
        provider_options={"top_k": 20},
    )
    _backend(client).generate(request)
    assert client.calls[0]["sampling"]["top_k"] == 20

    conflicting = GenerationRequest(
        request_id="conflict",
        model=request.model,
        messages=request.messages,
        sampling=request.sampling,
        provider_options={"temperature": 2.0},
    )
    with pytest.raises(GenerationError, match="cannot override standard keys"):
        _backend(_FakeClient()).generate(conflicting)


def test_extra_sampling_cannot_override_declared_sampling() -> None:
    """A key in both ``sampling`` and ``extra_sampling`` is refused, naming the key.

    Rule: ``extra_sampling`` adds provider knobs beside the keys the request
    declares; it never decides one of them. Refused at construction, before any
    generation is paid for, and refused on key presence rather than on a differing
    value -- the escape hatch claiming the key is the conflict.
    """
    with pytest.raises(GenerationError, match="extra_sampling cannot override standard keys"):
        VerlLLMServerGenerationBackend(
            _FakeClient(),
            _FakeTokenizer(),
            model_name="qwen3-0.6b",
            tokenizer_id="qwen-tok",
            extra_sampling={"temperature": 1.5},
        )


def test_extra_sampling_conflict_names_every_conflicting_key() -> None:
    with pytest.raises(GenerationError, match=r"standard keys: max_tokens, temperature$"):
        VerlLLMServerGenerationBackend(
            _FakeClient(),
            _FakeTokenizer(),
            model_name="qwen3-0.6b",
            tokenizer_id="qwen-tok",
            extra_sampling={"temperature": 1.5, "max_tokens": 99, "repetition_penalty": 1.05},
        )


def test_non_conflicting_extra_sampling_still_reaches_the_engine() -> None:
    """The escape hatch keeps working: a key the request does not declare passes through."""
    client = _FakeClient()
    backend = VerlLLMServerGenerationBackend(
        client,
        _FakeTokenizer(),
        model_name="qwen3-0.6b",
        tokenizer_id="qwen-tok",
        extra_sampling={"repetition_penalty": 1.05, "top_k": 20},
    )
    backend.generate(_req("r1"))
    sampling = client.calls[0]["sampling"]
    assert sampling["repetition_penalty"] == 1.05
    assert sampling["top_k"] == 20
    # ...and the declared sampling is what the engine was told to use.
    assert sampling["temperature"] == 0.6
    assert sampling["max_tokens"] == 16


def test_backend_refuses_tools() -> None:
    tools = [{"type": "function", "function": {"name": "c", "parameters": {}}}]
    with pytest.raises(GenerationError, match="does not parse tool"):
        _backend(_FakeClient()).generate(_req(tools=tools))


def test_span_without_logprobs_fails_loudly() -> None:
    class _NoLogprob:
        async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
            return _FakeTokenOutput(token_ids=[10, 11], log_probs=None)

    with pytest.raises(GenerationError, match="no log_probs"):
        _backend(_NoLogprob()).generate(_req())


def test_rejects_non_client() -> None:
    with pytest.raises(TypeError, match="VerlTokenClient"):
        VerlLLMServerGenerationBackend(object(), _FakeTokenizer(), model_name="m", tokenizer_id="t")
