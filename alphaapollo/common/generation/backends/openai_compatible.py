# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI-compatible generation backend (Issue #185).

Speaks ``chat.completions`` over HTTP. Two modes:

  * default (text): returns semantic content, reasoning, and passed-through tool
    calls; leaves the token/provenance fields ``None``. Not trainable -- a stock
    chat endpoint returns token strings, not integer ids.
  * ``return_token_ids=True``: against a vLLM server, additionally requests the
    ``return_token_ids`` extension plus ``logprobs``, and surfaces the integer
    prompt/response token ids and per-token logprobs. This path is trainable, and
    was verified BIT-EXACT against a native verl rollout (same engine, same
    prompt ids, greedy: identical token ids, logprob max-abs-diff 0.0). "Remote
    HTTP" is therefore not the axis that decides trainability.

The OpenAI package is imported lazily. When ``return_token_ids`` is requested but
the endpoint does not return ids/logprobs, generation fails loudly rather than
silently degrading a training rollout to text.

``routing_key`` is NOT carried by this backend, and the omission is deliberate --
see :meth:`OpenAICompatibleGenerationBackend._one`.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from alphaapollo.common.generation.base import (
    BATCH_EXECUTION_CLIENT_CONCURRENT,
    BATCH_EXECUTION_SEQUENTIAL,
    POLICY_SOURCE_REMOTE_ENGINE,
    BackendCapabilities,
    GenerationBackend,
    GenerationError,
    GenerationRequest,
    GenerationResponse,
    Provenance,
    ToolCall,
    reject_standard_key_overrides,
    with_backend_metadata,
)

_STANDARD_REQUEST_KEYS = frozenset(
    {"model", "messages", "temperature", "max_tokens", "top_p", "n", "tools", "tool_choice"}
)


class OpenAICompatibleGenerationBackend(GenerationBackend):
    """One generation per request over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        max_retries: int = 3,
        timeout: float = 300.0,
        max_concurrency: int = 1,
        client_factory: Callable[..., Any] | None = None,
        return_token_ids: bool = False,
        tokenizer_id: str | None = None,
        weights_version: str = "",
        tokenizer_fingerprint: str = "",
        chat_template_fingerprint: str = "",
        actor: str = "",
    ) -> None:
        # Validate the capability switch BEFORE importing/constructing any client,
        # so a truthy string/int can't flip an eval endpoint into a purported
        # token-native one and an invalid value can't allocate client resources or
        # trigger factory side effects first. Same boolean contract as the factory.
        if not isinstance(return_token_ids, bool):
            raise TypeError(
                f"return_token_ids must be a bool, got {type(return_token_ids).__name__}"
            )
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - install concern
                raise ImportError(
                    "openai package required for OpenAICompatibleGenerationBackend: "
                    "pip install openai"
                ) from exc
            client_factory = OpenAI
        self._client = client_factory(
            api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout
        )
        self._return_token_ids = return_token_ids
        self._tokenizer_id = tokenizer_id
        self._weights_version = weights_version
        self._tokenizer_fingerprint = tokenizer_fingerprint
        self._chat_template_fingerprint = chat_template_fingerprint
        self._actor = actor
        self._max_concurrency = max_concurrency
        self._run_identity = {
            "backend": f"{type(self).__module__}.{type(self).__qualname__}",
            "client": f"{type(self._client).__module__}.{type(self._client).__qualname__}",
            "endpoint_sha256": hashlib.sha256(str(base_url).encode("utf-8")).hexdigest(),
            "max_retries": max_retries,
            "timeout": timeout,
            "return_token_ids": return_token_ids,
            "tokenizer_id": tokenizer_id,
            "weights_version": weights_version,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "chat_template_fingerprint": chat_template_fingerprint,
            "actor": actor,
        }

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            token_native=self._return_token_ids,
            logprobs=self._return_token_ids,
            tool_calls=True,
            policy_source=POLICY_SOURCE_REMOTE_ENGINE,
            batch_execution=(
                BATCH_EXECUTION_CLIENT_CONCURRENT
                if self._max_concurrency > 1
                else BATCH_EXECUTION_SEQUENTIAL
            ),
            max_concurrency=self._max_concurrency,
        )

    @property
    def run_identity(self) -> dict[str, Any]:
        """Return non-secret endpoint and retry identity for resumable runs."""

        return dict(self._run_identity)

    def _generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResponse]:
        requests = list(requests)
        if not requests:
            return []
        batch_started = time.monotonic()
        if self._max_concurrency == 1 or len(requests) == 1:
            responses = [self._timed_one(request, batch_started) for request in requests]
        else:
            worker_count = min(self._max_concurrency, len(requests))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="alphaapollo-generation",
            ) as executor:
                futures = [
                    executor.submit(self._timed_one, request, time.monotonic())
                    for request in requests
                ]
                # Resolve in request order. The context waits for every submitted
                # request before propagating an error, so callers never receive a
                # partial batch while provider work is still running.
                responses = [future.result() for future in futures]
        batch_completed = time.monotonic()
        batch_metadata = {
            "batch_timing": {
                "started_monotonic_s": batch_started,
                "completed_monotonic_s": batch_completed,
                "elapsed_s": batch_completed - batch_started,
            },
            "batch_execution": self.capabilities.batch_execution,
            "configured_max_concurrency": self._max_concurrency,
        }
        return [with_backend_metadata(response, batch_metadata) for response in responses]

    def _timed_one(
        self,
        request: GenerationRequest,
        queued_at: float,
    ) -> GenerationResponse:
        dispatch_started = time.monotonic()
        response = self._one(request)
        completed = time.monotonic()
        return with_backend_metadata(
            response,
            {
                "request_timing": {
                    "queued_monotonic_s": queued_at,
                    "dispatch_started_monotonic_s": dispatch_started,
                    "completed_monotonic_s": completed,
                    "queue_s": dispatch_started - queued_at,
                    "service_s": completed - dispatch_started,
                },
                "logical_dispatches": 1,
                # The OpenAI SDK may retry internally. It does not expose a stable
                # public attempt count, so claiming one physical HTTP attempt would
                # fabricate provider-visible accounting.
                "physical_attempts": "UNKNOWN",
            },
        )

    def _one(self, request: GenerationRequest) -> GenerationResponse:
        reject_standard_key_overrides(
            option_name="provider_options",
            options=request.provider_options,
            standard_keys=_STANDARD_REQUEST_KEYS,
        )

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.sampling.temperature,
            "max_tokens": request.sampling.max_tokens,
            "top_p": request.sampling.top_p,
        }
        if request.tools:
            kwargs["tools"] = list(request.tools)
            if request.tool_choice is not None:
                kwargs["tool_choice"] = request.tool_choice
        kwargs.update(request.provider_options)
        # ``request.routing_key`` is deliberately NOT sent, and specifically not as
        # the ``X-Request-Id`` header. Measured against vLLM 0.19.1 (qwen3.5-27b,
        # 2026-08-16): the server does adopt that header -- the completion returns as
        # ``chatcmpl-<header>`` -- but adopting it as the *request id* is the problem,
        # not the feature. A routing key is deliberately shared (the workflow executor
        # gives one key to all k branches of a step), while a request id must identify
        # one generation; sending the first as the second made a k=4 batch record one
        # ``response_id`` for four independent generations, and that field is persisted
        # into trajectory records and learning capture records. Nor would it buy what
        # the key is for: a single endpoint has no replicas to be sticky about, and
        # ``X-Request-Id`` is not consulted for replica affinity, so the header is a
        # relabelling rather than routing. The same probe showed the alternatives are
        # not carriers either -- an extra body field and the standard ``user`` field
        # were both accepted and ignored. This transport therefore has no faithful
        # carrier for a sticky-session key; see ``GenerationRequest.routing_key``.
        if self._return_token_ids:
            # Force these AFTER provider_options so a caller cannot silently
            # disable the trainable-mode requirements the backend advertises.
            kwargs["logprobs"] = True
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["return_token_ids"] = True
            kwargs["extra_body"] = extra_body

        raw = self._client.chat.completions.create(**kwargs)
        choices = getattr(raw, "choices", None)
        if not choices:
            raise GenerationError(f"request {request.request_id!r}: response contained no choices")
        if len(choices) != 1:
            # One request is one generation; >1 choice means the provider ignored
            # that. Fail loudly rather than silently drop paid samples.
            raise GenerationError(
                f"request {request.request_id!r}: expected 1 choice, got {len(choices)}"
            )
        choice = choices[0]

        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        reasoning = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )
        reasoning_content = reasoning if isinstance(reasoning, str) else None
        tool_calls = _tool_calls_of(message, request.request_id)
        if content is None and (reasoning_content is not None or tool_calls):
            content = ""
        if not isinstance(content, str):
            raise GenerationError(f"request {request.request_id!r}: choice carried no text content")
        finish_reason = getattr(choice, "finish_reason", None)

        prompt_ids: tuple[int, ...] | None = None
        response_ids: tuple[int, ...] | None = None
        logprobs: tuple[float, ...] | None = None
        provenance: Provenance | None = None
        if self._return_token_ids:
            prompt_ids = _prompt_token_ids_of(raw, request.request_id)
            response_ids = _choice_token_ids(choice, request.request_id)
            logprobs = _choice_logprobs(choice, request.request_id)
            served_model = getattr(raw, "model", None) or request.model
            provenance = Provenance(
                policy_model=served_model,
                tokenizer_id=self._tokenizer_id or served_model,
                weights_version=self._weights_version,
                tokenizer_fingerprint=self._tokenizer_fingerprint,
                chat_template_fingerprint=self._chat_template_fingerprint,
                actor=self._actor,
            )

        return GenerationResponse(
            request_id=request.request_id,
            group_id=request.group_id,
            sample_id=request.sample_id,
            content=content,
            reasoning_content=reasoning_content,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            tool_calls=tool_calls,
            usage=_object_mapping(getattr(raw, "usage", None)),
            backend_metadata={
                "response_id": getattr(raw, "id", None),
                "model": getattr(raw, "model", None),
            },
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
            response_logprobs=logprobs,
            provenance=provenance,
        )


def _tool_calls_of(message: Any, request_id: str) -> tuple[ToolCall, ...]:
    """Pass provider-structured tool calls through verbatim (id/name/raw args)."""
    raw_calls = getattr(message, "tool_calls", None)
    if raw_calls is None:
        return ()
    if isinstance(raw_calls, (str, bytes)) or not isinstance(raw_calls, Sequence):
        raise GenerationError(f"request {request_id!r}: malformed tool_calls field")
    calls: list[ToolCall] = []
    for call in raw_calls:
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        call_id = getattr(call, "id", None)
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise GenerationError(f"request {request_id!r}: tool call missing id or name")
        calls.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=_tool_args(arguments, request_id),
            )
        )
    return tuple(calls)


def _tool_args(arguments: Any, request_id: str) -> str:
    # ToolCall.arguments is the raw provider JSON string, passed through verbatim.
    # None is not a JSON string and "" is not a faithful absence marker, so a
    # standards-conforming call (arguments is always a string, "{}" for no args)
    # is the only accepted shape; anything else fails loudly rather than get rewritten.
    if not isinstance(arguments, str):
        raise GenerationError(
            f"request {request_id!r}: tool call arguments are {type(arguments).__name__}, not the "
            "raw provider JSON string; refusing to rewrite malformed transport data"
        )
    return arguments


def _int_tuple(ids: Any, request_id: str) -> tuple[int, ...]:
    out: list[int] = []
    for i in ids:
        if isinstance(i, bool) or not isinstance(i, int):
            raise GenerationError(
                f"request {request_id!r}: token id {i!r} is not an integer; refusing to coerce "
                "(a re-tokenized/float id must not pass as an engine fact)"
            )
        out.append(i)
    return tuple(out)


def _prompt_token_ids_of(raw: Any, request_id: str) -> tuple[int, ...]:
    ids = getattr(raw, "prompt_token_ids", None)
    if ids is None and hasattr(raw, "model_extra"):
        ids = (raw.model_extra or {}).get("prompt_token_ids")
    if ids is None or isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise GenerationError(
            f"request {request_id!r}: return_token_ids requested but no prompt_token_ids returned"
        )
    return _int_tuple(ids, request_id)


def _choice_token_ids(choice: Any, request_id: str) -> tuple[int, ...]:
    ids = getattr(choice, "token_ids", None)
    if ids is None:
        message = getattr(choice, "message", None)
        ids = getattr(message, "token_ids", None)
    if ids is None:
        provider_fields = getattr(choice, "provider_specific_fields", None)
        if isinstance(provider_fields, Mapping):
            ids = provider_fields.get("token_ids")
    if ids is None or isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise GenerationError(
            f"request {request_id!r}: return_token_ids requested but the endpoint returned no "
            "integer token ids (not a vLLM server with the extension)"
        )
    return _int_tuple(ids, request_id)


def _choice_logprobs(choice: Any, request_id: str) -> tuple[float, ...]:
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None)
    if content is None and isinstance(logprobs, Mapping):
        content = logprobs.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        raise GenerationError(
            f"request {request_id!r}: return_token_ids requested but no per-token logprobs returned"
        )
    values: list[float] = []
    for token in content:
        value = getattr(token, "logprob", None)
        if value is None and isinstance(token, Mapping):
            value = token.get("logprob")
        if value is None:
            raise GenerationError(f"request {request_id!r}: a logprob entry was empty")
        # A logprob is an engine fact: reject bool/non-numeric/non-finite rather
        # than coerce (float(True)==1.0 would fabricate a trainable signal).
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GenerationError(
                f"request {request_id!r}: logprob {value!r} is not a real number; won't coerce"
            )
        as_float = float(value)
        if not math.isfinite(as_float):
            raise GenerationError(f"request {request_id!r}: logprob {value!r} is not finite")
        values.append(as_float)
    return tuple(values)


def _object_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, Mapping) else {}
    raw = getattr(value, "__dict__", None)
    return dict(raw) if isinstance(raw, dict) else {}


__all__ = ["OpenAICompatibleGenerationBackend"]
