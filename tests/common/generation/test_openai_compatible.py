"""OpenAI-compatible generation backend: text mode + trainable return_token_ids mode.

Uses a fake OpenAI client shaped like a vLLM chat response
(choice.token_ids, response.prompt_token_ids, logprobs.content[].logprob). Pins
one-request-one-response, tool-call passthrough, and that the token/provenance
fields are populated only in return_token_ids mode and never fabricated.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from alphaapollo.common.generation import (
    BATCH_EXECUTION_CLIENT_CONCURRENT,
    BATCH_EXECUTION_SEQUENTIAL,
    GenerationError,
    GenerationRequest,
    OpenAICompatibleGenerationBackend,
    Provenance,
    SamplingOptions,
    require_trainable,
)


class _Completions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


class _Client:
    def __init__(self, response: Any) -> None:
        self.chat = SimpleNamespace(completions=_Completions(response))


def _logprobs(values: list[float]) -> Any:
    return SimpleNamespace(content=[SimpleNamespace(logprob=v) for v in values])


def _tool_call(call_id: str, name: str, arguments: str) -> Any:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _response(
    *,
    content: str | None = "hello",
    token_ids: list[int] | None = None,
    logprobs: list[float] | None = None,
    prompt_token_ids: list[int] | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
) -> Any:
    choice = SimpleNamespace(
        index=0,
        message=SimpleNamespace(content=content, reasoning_content=None, tool_calls=tool_calls),
        finish_reason=finish_reason,
        token_ids=token_ids,
        logprobs=_logprobs(logprobs) if logprobs is not None else None,
    )
    return SimpleNamespace(
        id="chatcmpl-1",
        model="qwen3-0.6b",
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=3, total_tokens=6),
        prompt_token_ids=prompt_token_ids,
    )


def _backend(response: Any, **kwargs: Any) -> tuple[OpenAICompatibleGenerationBackend, _Client]:
    client = _Client(response)
    backend = OpenAICompatibleGenerationBackend(
        base_url="http://vllm:8000/v1", client_factory=lambda **_: client, **kwargs
    )
    return backend, client


def _req(request_id: str = "r0", *, group_id: str = "g", sample_id: int = 0, tools=None):
    kwargs: dict[str, Any] = {
        "request_id": request_id,
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "hi"}],
        "sampling": SamplingOptions(temperature=0.7, max_tokens=8),
        "group_id": group_id,
        "sample_id": sample_id,
    }
    if tools is not None:
        kwargs["tools"] = tools
    return GenerationRequest(**kwargs)


# --- text mode -----------------------------------------------------------------


def test_text_mode_is_not_trainable_and_leaves_tokens_none() -> None:
    backend, _ = _backend(_response(content="hi"))
    assert backend.capabilities.trainable is False
    resp = backend.generate(_req("r0", sample_id=2))
    assert resp.request_id == "r0" and resp.sample_id == 2
    assert resp.content == "hi"
    assert not resp.is_trainable
    assert resp.response_token_ids is None and resp.provenance is None


def test_backend_run_identity_binds_endpoint_and_sdk_retry_policy() -> None:
    response = _response(content="ok")
    first_client = _Client(response)
    second_client = _Client(response)
    first = OpenAICompatibleGenerationBackend(
        base_url="http://first.example/v1",
        max_retries=2,
        client_factory=lambda **_: first_client,
    )
    second = OpenAICompatibleGenerationBackend(
        base_url="http://second.example/v1",
        max_retries=4,
        client_factory=lambda **_: second_client,
    )

    assert first.run_identity != second.run_identity
    assert first.run_identity["max_retries"] == 2
    assert second.run_identity["max_retries"] == 4
    assert "first.example" not in repr(first.run_identity)


def test_tool_calls_passed_through_with_id() -> None:
    resp_obj = _response(content=None, tool_calls=[_tool_call("call_1", "python", '{"code":"x"}')])
    backend, _ = _backend(resp_obj)
    resp = backend.generate(_req())
    assert resp.has_tool_calls
    assert resp.tool_calls[0].id == "call_1"  # provider id preserved for multi-turn echo
    assert resp.tool_calls[0].name == "python"
    assert resp.tool_calls[0].arguments == '{"code":"x"}'


# --- return_token_ids (trainable) mode -----------------------------------------


def test_return_token_ids_mode_is_trainable() -> None:
    backend, _ = _backend(_response(token_ids=[10, 11]), return_token_ids=True)
    caps = require_trainable(backend)
    assert caps.trainable and caps.token_native


def test_return_token_ids_requests_extension_and_surfaces_tokens() -> None:
    resp_obj = _response(
        token_ids=[10, 11, 12], logprobs=[-0.1, -0.2, -0.3], prompt_token_ids=[1, 2, 3]
    )
    backend, client = _backend(
        resp_obj,
        return_token_ids=True,
        tokenizer_id="qwen-tok",
        weights_version="step-42",
        actor="solver",
    )
    resp = backend.generate(_req("r5"))

    sent = client.chat.completions.kwargs
    assert sent["logprobs"] is True
    assert sent["extra_body"]["return_token_ids"] is True

    assert resp.is_trainable
    assert resp.prompt_token_ids == (1, 2, 3)
    assert resp.response_token_ids == (10, 11, 12)
    assert resp.response_logprobs == (-0.1, -0.2, -0.3)
    assert resp.provenance == Provenance(
        policy_model="qwen3-0.6b",
        tokenizer_id="qwen-tok",
        weights_version="step-42",
        actor="solver",
    )


def test_missing_ids_fail_loudly() -> None:
    backend, _ = _backend(
        _response(token_ids=None, logprobs=[-0.1], prompt_token_ids=[1]), return_token_ids=True
    )
    with pytest.raises(GenerationError, match="no.*integer token ids"):
        backend.generate(_req())


def test_missing_prompt_ids_fail_loudly() -> None:
    backend, _ = _backend(
        _response(token_ids=[10], logprobs=[-0.1], prompt_token_ids=None), return_token_ids=True
    )
    with pytest.raises(GenerationError, match="no prompt_token_ids"):
        backend.generate(_req())


# --- batch semantics -----------------------------------------------------------


def test_batch_preserves_identity_and_order() -> None:
    backend, _ = _backend(_response(content="ok"))
    reqs = [_req("a", sample_id=0), _req("b", sample_id=1), _req("c", sample_id=2)]
    out = backend.generate_batch(reqs)
    assert [r.request_id for r in out] == ["a", "b", "c"]
    assert [r.sample_id for r in out] == [0, 1, 2]


def test_bounded_batch_is_really_concurrent_and_observable() -> None:
    response = _response(content="ok")

    class _DelayedCompletions:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def create(self, **kwargs: Any) -> Any:
            del kwargs
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.06)
                return response
            finally:
                with self._lock:
                    self.active -= 1

    completions = _DelayedCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAICompatibleGenerationBackend(
        base_url="http://vllm:8000/v1",
        client_factory=lambda **_: client,
        max_concurrency=2,
    )
    requests = [_req(f"r{i}", sample_id=i) for i in range(4)]

    started = time.monotonic()
    results = backend.generate_batch(requests)
    elapsed = time.monotonic() - started

    assert completions.max_active == 2
    assert elapsed < 0.21  # four 60ms calls cannot finish this fast sequentially
    assert [result.request_id for result in results] == [request.request_id for request in requests]
    assert backend.capabilities.batch_execution == BATCH_EXECUTION_CLIENT_CONCURRENT
    assert backend.capabilities.max_concurrency == 2
    assert all(
        result.backend_metadata["batch_execution"] == "client_concurrent" for result in results
    )
    assert all(result.backend_metadata["request_timing"]["service_s"] >= 0.05 for result in results)
    assert all(result.backend_metadata["physical_attempts"] == "UNKNOWN" for result in results)


def test_openai_default_transport_remains_sequential() -> None:
    backend, _ = _backend(_response())
    assert backend.capabilities.batch_execution == BATCH_EXECUTION_SEQUENTIAL
    assert backend.capabilities.max_concurrency == 1


def test_concurrent_failure_waits_for_submitted_work_and_returns_no_partial_batch() -> None:
    response = _response(content="ok")

    class FailingCompletions:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.calls = 0
            self.active = 0

        def create(self, **kwargs: Any) -> Any:
            content = kwargs["messages"][0]["content"]
            with self.lock:
                self.calls += 1
                self.active += 1
            try:
                time.sleep(0.02)
                if content == "fail":
                    raise TimeoutError("controlled timeout")
                return response
            finally:
                with self.lock:
                    self.active -= 1

    completions = FailingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAICompatibleGenerationBackend(
        base_url="http://vllm:8000/v1",
        client_factory=lambda **_: client,
        max_concurrency=2,
    )
    requests = [_req(f"r{i}", sample_id=i) for i in range(4)]
    requests[1] = replace(
        requests[1],
        messages=[{"role": "user", "content": "fail"}],
    )

    with pytest.raises(TimeoutError, match="controlled timeout"):
        backend.generate_batch(requests)

    assert completions.calls == 4
    assert completions.active == 0


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_openai_rejects_invalid_max_concurrency(value: Any) -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        OpenAICompatibleGenerationBackend(
            base_url="http://vllm:8000/v1",
            client_factory=lambda **_: _Client(_response()),
            max_concurrency=value,
        )


def test_routing_key_is_not_sent_and_never_as_the_request_id_header() -> None:
    """Rule: this backend does not carry ``routing_key`` -- deliberately, not by oversight.

    A routing key is a sticky hint for a replica pool and is deliberately shared
    (the workflow executor gives one key to all k branches of a step). An
    OpenAI-compatible endpoint offers no field with that meaning. ``X-Request-Id``
    is the nearest candidate and vLLM does adopt it, but it adopts it as the
    *unique request id*: sending a shared key there makes k independent generations
    report one ``response_id`` -- a value that is persisted into trajectory and
    learning capture records -- while routing nothing, since a single endpoint has
    no replicas. So the key stays behind rather than being relabelled into an
    identifier. This fails if a future edit sends it anyway.
    """
    backend, client = _backend(_response(content="hi"))
    req = GenerationRequest(
        request_id="r0",
        model="qwen3-0.6b",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingOptions(temperature=0.7, max_tokens=8),
        routing_key="input-3:solve:iteration-2",
    )
    resp = backend.generate(req)
    sent = client.chat.completions.kwargs
    assert "extra_headers" not in sent
    assert not any("input-3" in str(value) for value in sent.values())
    # The server's own id is what identifies the generation, not the routing key.
    assert resp.backend_metadata["response_id"] == "chatcmpl-1"


def test_routing_key_does_not_disturb_caller_headers() -> None:
    backend, client = _backend(_response(content="hi"))
    req = GenerationRequest(
        request_id="r0",
        model="qwen3-0.6b",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingOptions(temperature=0.7, max_tokens=8),
        provider_options={"extra_headers": {"X-Tenant": "team-a"}},
        routing_key="sticky",
    )
    backend.generate(req)
    assert client.chat.completions.kwargs["extra_headers"] == {"X-Tenant": "team-a"}


def test_provider_options_cannot_override_standard_keys() -> None:
    backend, _ = _backend(_response(content="hi"))
    req = GenerationRequest(
        request_id="r0",
        model="qwen3-0.6b",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingOptions(temperature=0.7, max_tokens=8),
        provider_options={"temperature": 1.5},
    )
    with pytest.raises(GenerationError, match="cannot override standard keys"):
        backend.generate(req)
