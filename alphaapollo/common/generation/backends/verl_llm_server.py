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

"""Pinned-verl LLM-server generation backend (Issue #185).

Directly reuses verl's own rollout server through a thin adapter -- it does not
copy or reimplement it. We hand verl's ``LLMServerClient`` prompt token ids and
get back a verl ``TokenOutput`` (token-in-token-out), then adapt it to the
neutral :class:`GenerationResponse`. This is the training-path source: the token
ids and logprobs are exactly what verl's trainer would see (verified on A100-1,
and the learning-side DataProto rebuilt from these matches native verl
byte-for-byte).

Reuse is scoped to the one-generation call only. The agent loop, multi-turn
masking, and ``DataProto`` assembly are NOT here (reasoning and learning own
those). The verl server/Ray/weight-sync lifecycle is owned outside Common: an
already-built client is injected. verl is imported lazily by whatever provides
that client, so importing this module never pulls in verl, Ray, or Torch.

``LLMServerClient.generate`` is async and returns one sequence per call. A batch
therefore fans out those independent calls concurrently and restores request
order with ``asyncio.gather``. This preserves Reasoning's active-set batching on
the Learning training path instead of serializing every trajectory at transport.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

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
    _verify_request_identities,
    reject_standard_key_overrides,
    verify_batch_identity,
    with_backend_metadata,
)

# The sampling keys this backend derives from ``request.sampling`` on every call.
# ``extra_sampling`` may add provider-specific knobs beside them but may not claim
# one of them, because then two inputs would define one behaviour. Mirrors
# ``_STANDARD_REQUEST_KEYS`` in the OpenAI-compatible backend: the forced
# ``logprobs`` flag is the backend's own trainability invariant, not a caller
# declaration, so it is not in this set (same treatment as over there).
_STANDARD_SAMPLING_KEYS = frozenset({"temperature", "top_p", "max_tokens"})

_PROMPT_TRUNCATION_POLICIES = frozenset({"left", "right", "middle", "error"})


@runtime_checkable
class TokenOutputLike(Protocol):
    """The subset of verl ``TokenOutput`` this backend reads (structural)."""

    token_ids: Sequence[int]
    log_probs: Sequence[float] | None
    stop_reason: str | None


@runtime_checkable
class VerlTokenClient(Protocol):
    """verl ``LLMServerClient.generate`` -- async, prompt_ids in, TokenOutput out."""

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: Sequence[int],
        sampling_params: Mapping[str, Any],
        **kwargs: Any,
    ) -> TokenOutputLike: ...


@runtime_checkable
class PromptTokenizer(Protocol):
    """Encode a prompt to ids and decode a response to text (a plain tokenizer)."""

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> Sequence[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


def _int_tuple(ids: Any, request_id: str) -> tuple[int, ...]:
    out: list[int] = []
    for i in ids:
        if isinstance(i, bool) or not isinstance(i, int):
            raise GenerationError(
                f"request {request_id!r}: token id {i!r} is not an integer; refusing to coerce"
            )
        out.append(i)
    return tuple(out)


def _float_tuple(values: Any, request_id: str) -> tuple[float, ...]:
    # Logprobs are engine facts too: reject bool/non-numeric/non-finite rather than
    # relabel them (float(True)==1.0 would silently fabricate a trainable signal).
    out: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise GenerationError(
                f"request {request_id!r}: logprob {v!r} is not a real number; refusing to coerce"
            )
        f = float(v)
        if not math.isfinite(f):
            raise GenerationError(f"request {request_id!r}: logprob {v!r} is not finite")
        out.append(f)
    return tuple(out)


def _limit_prompt(
    prompt_ids: tuple[int, ...],
    *,
    request_id: str,
    max_tokens: int | None,
    truncation: str,
) -> tuple[int, ...]:
    """Apply the configured limit before the prompt reaches the verl server."""

    if max_tokens is None or len(prompt_ids) <= max_tokens:
        return prompt_ids
    if truncation == "left":
        return prompt_ids[-max_tokens:]
    if truncation == "right":
        return prompt_ids[:max_tokens]
    if truncation == "middle":
        left = max_tokens // 2
        right = max_tokens - left
        return prompt_ids[:left] + prompt_ids[-right:]
    raise GenerationError(
        f"request {request_id!r}: prompt has {len(prompt_ids)} tokens, "
        f"exceeding the configured limit of {max_tokens}"
    )


class VerlLLMServerGenerationBackend(GenerationBackend):
    """Trainable generation backend over an injected verl ``LLMServerClient``."""

    def __init__(
        self,
        client: VerlTokenClient,
        tokenizer: PromptTokenizer,
        *,
        model_name: str,
        tokenizer_id: str,
        weights_version: str = "",
        tokenizer_fingerprint: str = "",
        chat_template_fingerprint: str = "",
        actor: str = "",
        extra_sampling: Mapping[str, Any] | None = None,
        max_prompt_tokens: int | None = None,
        prompt_truncation: str = "error",
        max_concurrency: int = 32,
    ) -> None:
        if not isinstance(client, VerlTokenClient):
            raise TypeError("client must implement the VerlTokenClient protocol")
        if not isinstance(tokenizer, PromptTokenizer):
            raise TypeError("tokenizer must implement the PromptTokenizer protocol")
        if not model_name.strip() or not tokenizer_id.strip():
            raise ValueError("model_name and tokenizer_id must be non-empty")
        if max_prompt_tokens is not None and (
            isinstance(max_prompt_tokens, bool)
            or not isinstance(max_prompt_tokens, int)
            or max_prompt_tokens < 1
        ):
            raise ValueError("max_prompt_tokens must be a positive integer when provided")
        if prompt_truncation not in _PROMPT_TRUNCATION_POLICIES:
            raise ValueError(
                f"prompt_truncation must be one of {sorted(_PROMPT_TRUNCATION_POLICIES)}"
            )
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        # Refuse here, not at first generate: ``extra_sampling`` is fixed at
        # construction and the keys it may not claim are static, so the conflict is
        # knowable before any inference is paid for.
        reject_standard_key_overrides(
            option_name="extra_sampling",
            options=extra_sampling or {},
            standard_keys=_STANDARD_SAMPLING_KEYS,
        )
        self._client = client
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._tokenizer_id = tokenizer_id
        self._weights_version = weights_version
        self._tokenizer_fingerprint = tokenizer_fingerprint
        self._chat_template_fingerprint = chat_template_fingerprint
        self._actor = actor
        self._extra_sampling = dict(extra_sampling or {})
        self._max_prompt_tokens = max_prompt_tokens
        self._prompt_truncation = prompt_truncation
        self._max_concurrency = max_concurrency

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            token_native=True,
            logprobs=True,
            tool_calls=False,
            policy_source=POLICY_SOURCE_REMOTE_ENGINE,
            batch_execution=(
                BATCH_EXECUTION_CLIENT_CONCURRENT
                if self._max_concurrency > 1
                else BATCH_EXECUTION_SEQUENTIAL
            ),
            max_concurrency=self._max_concurrency,
        )

    @property
    def current_provenance(self) -> Provenance:
        """Return the policy identity that new responses will carry."""

        return Provenance(
            policy_model=self._model_name,
            tokenizer_id=self._tokenizer_id,
            weights_version=self._weights_version,
            tokenizer_fingerprint=self._tokenizer_fingerprint,
            chat_template_fingerprint=self._chat_template_fingerprint,
            actor=self._actor,
        )

    @property
    def run_identity(self) -> dict[str, Any]:
        """Return model, transport, and tokenizer identity for resumable runs."""

        extra_sampling = json.dumps(
            self._extra_sampling,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "backend": f"{type(self).__module__}.{type(self).__qualname__}",
            "client": f"{type(self._client).__module__}.{type(self._client).__qualname__}",
            "tokenizer": (
                f"{type(self._tokenizer).__module__}.{type(self._tokenizer).__qualname__}"
            ),
            "model_name": self._model_name,
            "tokenizer_id": self._tokenizer_id,
            "weights_version": self._weights_version,
            "tokenizer_fingerprint": self._tokenizer_fingerprint,
            "chat_template_fingerprint": self._chat_template_fingerprint,
            "actor": self._actor,
            "extra_sampling_sha256": hashlib.sha256(extra_sampling.encode("utf-8")).hexdigest(),
            "max_prompt_tokens": self._max_prompt_tokens,
            "prompt_truncation": self._prompt_truncation,
        }

    def update_weights_version(self, weights_version: str) -> None:
        """Stamp generations only after Learning has synchronized rollout weights."""

        if not isinstance(weights_version, str) or not weights_version.strip():
            raise ValueError("weights_version must be a non-empty string")
        self._weights_version = weights_version

    def _generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResponse]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.agenerate_batch(requests))
        raise GenerationError(
            "generate_batch() cannot run inside an active event loop; await agenerate_batch()"
        )

    async def agenerate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[GenerationResponse]:
        """Async: one concurrent verl generate per request, preserving order."""
        requests = list(requests)
        # Pre-flight: reject duplicate identities BEFORE spending any generation,
        # matching the sync path so an awaited batch cannot burn inference on an
        # invalid batch (and never reaches LLMServerClient.generate for a dup).
        _verify_request_identities(requests)
        if not requests:
            return []
        batch_started = time.monotonic()
        semaphore = asyncio.Semaphore(self._max_concurrency)
        outcomes = list(
            await asyncio.gather(
                *(self._timed_one(request, batch_started, semaphore) for request in requests),
                return_exceptions=True,
            )
        )
        errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if errors:
            # Wait for every submitted request before exposing the first error.
            # The caller receives no partial batch and no sibling request remains
            # in flight after the failure boundary.
            raise errors[0]
        results = [outcome for outcome in outcomes if isinstance(outcome, GenerationResponse)]
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
        results = [with_backend_metadata(response, batch_metadata) for response in results]
        # Same identity guarantee as the sync path: a caller awaiting this method
        # directly cannot obtain an unverified batch.
        verify_batch_identity(requests, results)
        return results

    async def _timed_one(
        self,
        request: GenerationRequest,
        queued_at: float,
        semaphore: asyncio.Semaphore,
    ) -> GenerationResponse:
        async with semaphore:
            dispatch_started = time.monotonic()
            response = await self._one(request)
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
                "physical_attempts": "UNKNOWN",
            },
        )

    async def _one(self, request: GenerationRequest) -> GenerationResponse:
        if request.tools:
            raise GenerationError(
                f"request {request.request_id!r}: VerlLLMServerGenerationBackend does not parse "
                "tool calls; expose tools through the Environment parser layer"
            )
        # This backend serves the model it was constructed for; routing selects the
        # backend before a request reaches it. We refuse a mismatch rather than
        # silently serve a different model than the request names (and misattribute
        # its provenance). The served identity is what lands in provenance below.
        # (GenerationRequest requires a non-empty model, so there is nothing to skip.)
        if request.model != self._model_name:
            raise GenerationError(
                f"request {request.request_id!r}: this backend serves model "
                f"{self._model_name!r}, not {request.model!r}; routing selects the "
                "backend, so set request.model to the served model name"
            )
        prompt_ids = _limit_prompt(
            self._encode_prompt(request),
            request_id=request.request_id,
            max_tokens=self._max_prompt_tokens,
            truncation=self._prompt_truncation,
        )
        sampling_params: dict[str, Any] = {
            "temperature": request.sampling.temperature,
            "top_p": request.sampling.top_p,
            "max_tokens": request.sampling.max_tokens,
        }
        # Disjoint by construction: __init__ refused any extra_sampling key that
        # this request's declared sampling already owns, so this adds knobs rather
        # than overriding what the caller declared.
        sampling_params.update(self._extra_sampling)
        reserved = {"temperature", "top_p", "max_tokens", "logprobs", "n"}
        collisions = sorted(reserved.intersection(request.provider_options))
        if collisions:
            raise GenerationError(
                f"request {request.request_id!r}: provider_options cannot override "
                f"standard keys: {collisions}"
            )
        sampling_params.update(request.provider_options)
        sampling_params["logprobs"] = True

        # The outer request_id is verl's sticky-session / load-balancer key: requests
        # sharing it land on one replica and reuse its prefix cache. Honour an explicit
        # routing_key, else fall back to the request's own id (never a fresh per-call
        # UUID). Note the key's real scope is a step batch, not one trajectory: the
        # workflow executor gives every branch of a step the same key, so the k samples
        # of a prompt share a replica -- which is the point, since they share a prefix.
        # It is therefore NOT a unique per-generation id and must not be used as one.
        transport_id = request.routing_key or request.request_id
        output = await self._client.generate(
            request_id=transport_id,
            prompt_ids=list(prompt_ids),
            sampling_params=sampling_params,
        )
        log_probs = getattr(output, "log_probs", None)
        if log_probs is None:
            raise GenerationError(
                f"request {request.request_id!r}: verl TokenOutput carried no log_probs; "
                "request sampling_params['logprobs']=True so the span is trainable"
            )
        response_ids = _int_tuple(output.token_ids, request.request_id)
        content = self._tokenizer.decode(list(response_ids))
        if not isinstance(content, str):
            raise GenerationError(
                f"request {request.request_id!r}: tokenizer.decode did not return text"
            )

        return GenerationResponse(
            request_id=request.request_id,
            group_id=request.group_id,
            sample_id=request.sample_id,
            content=content,
            finish_reason=getattr(output, "stop_reason", None),
            backend_metadata={"engine": "verl_llm_server", "model": self._model_name},
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
            response_logprobs=_float_tuple(log_probs, request.request_id),
            provenance=self.current_provenance,
        )

    def _encode_prompt(self, request: GenerationRequest) -> tuple[int, ...]:
        ids = self._tokenizer.apply_chat_template(
            list(request.messages), add_generation_prompt=True, tokenize=True
        )
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence) or not ids:
            raise GenerationError(
                f"request {request.request_id!r}: tokenizer returned an empty prompt encoding"
            )
        return _int_tuple(ids, request.request_id)


# Common defines the Generation contract and this injected-client adapter only. It
# must NOT construct verl's LLMServerManager, Ray resources, rollout replicas, or
# Trainer objects. In training, Learning owns actor_rollout_wg and the HYBRID
# LLMServerManager, obtains an LLMServerClient from it, and injects it here; for
# standalone evaluation a Workflow composition root wires the client. Either way the
# server / Ray / weight-sync lifecycle lives with its owner, not in Common. See the
# ownership flow in #189 (ShawFeng99) -- the previous Common-owned launcher was
# removed because a manager created without ``worker_group`` cannot participate in
# the actor's weight-synchronization lifecycle.

# The *Like / *Client / *Tokenizer Protocols above are internal typing helpers for
# the injected-client seam -- they describe the shape this backend reads, not a new
# public verl abstraction. Only the backend is the public surface.
__all__ = [
    "VerlLLMServerGenerationBackend",
]
