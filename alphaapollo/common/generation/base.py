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

"""The single-generation generation contract (Issue #185).

A Generation is exactly one model generation for one request. There is no nested
candidate list: a caller that wants ``n`` samples for one prompt issues ``n``
requests sharing a ``group_id`` with distinct ``sample_id`` values. This keeps
per-sample identity, provenance, and token facts unambiguous, and it matches how
verl and slime actually work (one server call returns one sequence).

Responsibility boundary (deliberately narrow):

  * We guarantee the *data* is correct and honestly labelled: integer token ids
    that the engine actually used (never re-tokenized), the engine's own sampling
    logprobs (never recomputed), and provenance so a downstream consumer can
    verify trainability and detect tokenizer/policy drift.
  * We do NOT parse tool calls, build masks, pad tensors, or construct a
    ``DataProto``. Tool semantics belong to Environment; batch/training collation
    belongs to Learning. A structured tool call is only *passed through* when a
    backend (e.g. a vLLM server with a tool-call parser) already produced one.
  * Optional token/provenance fields are populated when a backend can truthfully
    provide them and left ``None`` otherwise. They are never zero-filled or
    reconstructed to look like on-policy facts.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

# Where a generation physically happened. This is provenance, not the axis that
# decides trainability -- a remote engine (vLLM/SGLang) can be token-native.
POLICY_SOURCE_LOCAL_ENGINE = "local_engine"
POLICY_SOURCE_REMOTE_ENGINE = "remote_engine"
_POLICY_SOURCES = frozenset({POLICY_SOURCE_LOCAL_ENGINE, POLICY_SOURCE_REMOTE_ENGINE})

BATCH_EXECUTION_SEQUENTIAL = "sequential"
BATCH_EXECUTION_CLIENT_CONCURRENT = "client_concurrent"
BATCH_EXECUTION_SERVER_BATCHED = "server_batched"
_BATCH_EXECUTION_MODES = frozenset(
    {
        BATCH_EXECUTION_SEQUENTIAL,
        BATCH_EXECUTION_CLIENT_CONCURRENT,
        BATCH_EXECUTION_SERVER_BATCHED,
    }
)


class GenerationError(RuntimeError):
    """Raised when a backend cannot return a valid, correctly-labelled response."""


class CapabilityError(GenerationError):
    """Raised when a backend cannot deliver a capability a caller requires."""


@dataclass(frozen=True)
class BackendCapabilities:
    """A backend's honest, transport-level data contract.

    ``token_native`` is load-bearing: True iff the backend returns integer token
    ids plus per-token sampling logprobs, i.e. the response can feed RL training.
    ``trainable`` derives from it. Keeping this a first-class, checkable value is
    what makes "never mark API text as PPO/GRPO eligible" an enforced rule rather
    than a convention. ``batch_execution`` describes where concurrency occurs;
    ``max_concurrency`` is the transport's enforced upper bound, not a claim that
    every short batch will reach that overlap.
    """

    token_native: bool
    logprobs: bool
    tool_calls: bool
    policy_source: str
    batch_execution: str = BATCH_EXECUTION_SEQUENTIAL
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        for name in ("token_native", "logprobs", "tool_calls"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"BackendCapabilities.{name} must be a bool")
        if self.policy_source not in _POLICY_SOURCES:
            allowed = sorted(_POLICY_SOURCES)
            raise ValueError(f"policy_source must be one of {allowed}, got {self.policy_source!r}")
        if self.token_native and not self.logprobs:
            raise ValueError("token_native backends must also report logprobs")
        if self.batch_execution not in _BATCH_EXECUTION_MODES:
            allowed = sorted(_BATCH_EXECUTION_MODES)
            raise ValueError(
                f"batch_execution must be one of {allowed}, got {self.batch_execution!r}"
            )
        if (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or self.max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        if self.batch_execution == BATCH_EXECUTION_SEQUENTIAL and self.max_concurrency != 1:
            raise ValueError("sequential batch execution requires max_concurrency=1")

    @property
    def trainable(self) -> bool:
        return self.token_native


@dataclass(frozen=True)
class Provenance:
    """Who produced a token span and with which tokenizer/template.

    Carries the metadata a trainer needs for correctness, not decoration.
    ``weights_version`` identifies the behaviour policy so off-policy correction
    (importance sampling / TIS) knows which policy the logprobs came from; blank
    is a legitimate "current / on-policy" convention. ``tokenizer_fingerprint``
    and ``chat_template_fingerprint`` let a trainer assert its own tokenizer/
    template match the one that produced the ids, so silent retokenization drift
    is detectable. Fingerprints are opaque stable hashes; :func:`fingerprint` is
    the recommended way to compute one.

    The contract only guarantees these fields are *present and well-typed*; it does not
    decide which are mandatory. Whether a blank field is acceptable is the
    consumer's policy -- the learning/export layer enforces the completeness it
    needs at DataProto-assembly / TIS time, keeping the contract neutral about
    downstream training decisions. This is why :meth:`GenerationResponse.is_trainable`
    checks that a provenance record exists, not that every field is filled.
    """

    policy_model: str
    tokenizer_id: str
    weights_version: str = ""
    tokenizer_fingerprint: str = ""
    chat_template_fingerprint: str = ""
    actor: str = ""

    def __post_init__(self) -> None:
        for name in ("policy_model", "tokenizer_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Provenance.{name} must be a non-empty string")
        for name in (
            "weights_version",
            "tokenizer_fingerprint",
            "chat_template_fingerprint",
            "actor",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"Provenance.{name} must be a string")


def fingerprint(*parts: str) -> str:
    """Stable short hash of the given identity parts (tokenizer/template stamp)."""
    digest = hashlib.sha256(" ".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass(frozen=True)
class ToolCall:
    """A model-emitted tool call, passed through verbatim -- transport only.

    NOTE: the generation contract must not define ``ToolCallRequest`` (#185). This
    ``ToolCall`` is intentionally NOT that: it carries
    no tool semantics, does not decode ``arguments``, and does not know how to
    execute anything. It exists only so a provider that already produced a
    *structured* call (a vLLM server with a tool-call parser) can hand the
    Environment the provider's own ``id`` -- which multi-turn tool-response
    messages must echo -- instead of forcing a re-parse from text that would lose
    the id. Environment owns all tool meaning and execution.
    """

    id: str
    name: str
    arguments: str

    def __post_init__(self) -> None:
        for name in ("id", "name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ToolCall.{name} must be a non-empty string")
        if not isinstance(self.arguments, str):
            raise TypeError("ToolCall.arguments must be the raw provider JSON string")


@dataclass(frozen=True)
class SamplingOptions:
    """Provider-neutral sampling surface for ONE generation.

    There is no ``n``: one request is one generation. Multiple samples for a
    prompt are multiple requests sharing a ``group_id``.
    """

    temperature: float
    max_tokens: int
    top_p: float = 1.0

    def __post_init__(self) -> None:
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive int")


@dataclass(frozen=True)
class GenerationRequest:
    """One model-generation request.

    ``request_id`` uniquely identifies this generation. ``group_id`` ties the
    ``n`` requests of one prompt together; ``sample_id`` distinguishes them.
    ``tools`` carries OpenAI-compatible schemas only when the calling policy
    granted tool use; the generation layer neither selects tools nor infers permission.
    ``provider_options`` is an escape hatch that cannot override standard keys.
    ``routing_key`` is a sticky-session hint for a *pool* of replicas: a shared key
    asks that the requests carrying it land together, so they reuse one prefix cache.
    It is deliberately NOT unique -- the workflow executor gives one key to every
    branch of a step, precisely so the k samples of a prompt share a replica.

    Only a backend that talks to a replica pool can act on it, and today that is
    ``VerlLLMServerGenerationBackend`` alone, which passes it to its client as the
    transport request id. ``OpenAICompatibleGenerationBackend`` does not carry it:
    that transport offers no field with sticky-routing semantics, and the nearest
    candidate (``X-Request-Id``) is a *unique request identifier*, so borrowing it
    for a shared key would corrupt the recorded ``response_id`` while still not
    routing anything. So this field is honoured by one backend and inert in the
    other -- stated here rather than discovered, until the contract grows a
    capability that says which backends can act on it.
    """

    request_id: str
    model: str
    messages: Sequence[Mapping[str, Any]]
    sampling: SamplingOptions
    group_id: str = ""
    sample_id: int = 0
    tools: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    tool_choice: str | None = None
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    routing_key: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not self.messages:
            raise ValueError("messages must be non-empty")
        if isinstance(self.messages, (str, bytes)) or not isinstance(self.messages, Sequence):
            raise TypeError("messages must be a sequence of mappings")
        if any(not isinstance(m, Mapping) for m in self.messages):
            raise TypeError("each message must be a mapping")
        if not isinstance(self.sampling, SamplingOptions):
            raise TypeError("sampling must be a SamplingOptions")
        if (
            isinstance(self.sample_id, bool)
            or not isinstance(self.sample_id, int)
            or self.sample_id < 0
        ):
            raise ValueError("sample_id must be a non-negative int")
        if isinstance(self.tools, (str, bytes)) or not isinstance(self.tools, Sequence):
            raise TypeError("tools must be a sequence of tool schemas")
        # A backend forwards this verbatim as a transport id (an HTTP header for the
        # OpenAI-compatible path), so a non-string would fail deep in a client
        # library instead of naming the offending input here.
        if not isinstance(self.routing_key, str):
            raise TypeError("routing_key must be a string")


@dataclass(frozen=True)
class GenerationResponse:
    """One generation's result: correct data, honestly labelled.

    Echoes request identity. ``content`` is the reasoning-facing text (may be
    empty for a pure tool call). ``tool_calls`` are passed through only if the
    backend already structured them. The token fields are the trainable
    projection: present with ``provenance`` for a token-native backend, all
    ``None`` for a text-only one -- never fabricated.
    """

    request_id: str
    content: str
    group_id: str = ""
    sample_id: int = 0
    reasoning_content: str | None = None
    finish_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)
    prompt_token_ids: tuple[int, ...] | None = None
    response_token_ids: tuple[int, ...] | None = None
    response_logprobs: tuple[float, ...] | None = None
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if any(not isinstance(call, ToolCall) for call in self.tool_calls):
            raise TypeError("tool_calls must contain ToolCall records")
        object.__setattr__(
            self, "prompt_token_ids", _opt_int_tuple(self.prompt_token_ids, "prompt_token_ids")
        )
        object.__setattr__(
            self,
            "response_token_ids",
            _opt_int_tuple(self.response_token_ids, "response_token_ids"),
        )
        object.__setattr__(
            self, "response_logprobs", _opt_float_tuple(self.response_logprobs, "response_logprobs")
        )
        # A trainable span needs ids AND aligned logprobs AND provenance; enforce
        # the alignment so a half-populated response cannot masquerade as trainable.
        if self.response_token_ids is not None and self.response_logprobs is not None:
            if len(self.response_token_ids) != len(self.response_logprobs):
                raise ValueError("response_token_ids and response_logprobs must be the same length")
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance or None")

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_trainable(self) -> bool:
        """True iff a non-empty token span with logprobs and a provenance record.

        This gates on the *presence* of the trainable projection, not on the
        completeness of provenance metadata: a blank ``weights_version`` is a
        legitimate on-policy rollout, and fingerprint completeness is the
        learning layer's concern to validate at export/TIS time, not this
        contract's to hard-block (it stays neutral about downstream training policy).
        """
        return (
            self.prompt_token_ids is not None
            and self.response_token_ids is not None
            and len(self.response_token_ids) > 0
            and self.response_logprobs is not None
            and self.provenance is not None
        )


def _opt_int_tuple(value: Any, label: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of ints or None")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{label} must contain only ints")
        out.append(item)
    return tuple(out)


def _opt_float_tuple(value: Any, label: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of floats or None")
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{label} must contain only floats")
        as_float = float(item)
        if not math.isfinite(as_float):
            raise ValueError(f"{label} must be finite")
        out.append(as_float)
    return tuple(out)


class GenerationBackend(ABC):
    """A pluggable single-generation backend. One request -> one response.

    Implementations own only the transport to their engine. They must preserve
    input/output cardinality and order, echo each request's identity, and fail
    loudly on malformed output -- :func:`verify_batch_identity` is provided for
    that. They must not import AgentRuntime, Workflow, Environment, DataProto,
    reward managers, or trainer code.
    """

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """What this backend can truthfully provide (drives the trainable gate)."""

    @abstractmethod
    def _generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResponse]:
        """Backend hook: generate one response per request, in the same order."""

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResponse]:
        """Generate one response per request, then enforce one-to-one identity.

        Concrete here (not abstract) so cardinality/order/identity verification
        runs for every backend, including future subclasses -- a caller cannot
        get an unverified batch. Backends implement :meth:`_generate_batch`.
        """
        requests = list(requests)
        # Pre-flight: reject duplicate identities before spending provider work.
        _verify_request_identities(requests)
        responses = list(self._generate_batch(requests))
        verify_batch_identity(requests, responses)
        return responses

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Batch-size-one convenience over :meth:`generate_batch`, same semantics."""
        responses = self.generate_batch([request])
        if len(responses) != 1:
            raise GenerationError(f"generate() expected exactly one response, got {len(responses)}")
        return responses[0]


def _verify_request_identities(requests: Sequence[GenerationRequest]) -> None:
    """Reject duplicate identities on the request side (needs no responses).

    ``request_id`` must be unique across the batch. ``(group_id, sample_id)`` must
    be unique only among requests that actually carry a grouping id -- an ordinary
    batch of unrelated prompts leaves ``group_id`` blank and shares the default
    ``("", 0)`` slot, which is not a collision.
    """
    seen_request: set[str] = set()
    seen_sample: set[tuple[str, int]] = set()
    for req in requests:
        if req.request_id in seen_request:
            raise GenerationError(f"duplicate request_id {req.request_id!r} in batch")
        seen_request.add(req.request_id)
        if req.group_id:
            sample_key = (req.group_id, req.sample_id)
            if sample_key in seen_sample:
                raise GenerationError(f"duplicate (group_id, sample_id) {sample_key!r} in batch")
            seen_sample.add(sample_key)


def verify_batch_identity(
    requests: Sequence[GenerationRequest],
    responses: Sequence[GenerationResponse],
) -> None:
    """Fail loudly unless responses match requests one-to-one, in order, by id."""
    if len(requests) != len(responses):
        raise GenerationError(
            f"backend returned {len(responses)} responses for {len(requests)} requests"
        )
    _verify_request_identities(requests)
    for index, (req, resp) in enumerate(zip(requests, responses, strict=True)):
        if resp.request_id != req.request_id:
            raise GenerationError(
                f"response {index} has request_id {resp.request_id!r}, expected {req.request_id!r}"
            )
        if resp.group_id != req.group_id or resp.sample_id != req.sample_id:
            raise GenerationError(
                f"response {index} identity does not echo request {req.request_id!r}"
            )


def reject_standard_key_overrides(
    *,
    option_name: str,
    options: Mapping[str, Any],
    standard_keys: frozenset[str],
) -> None:
    """Refuse an escape hatch that would decide a key the caller already declared.

    Backends take provider-specific extras (``provider_options``,
    ``extra_sampling``) beside the keys they derive from the request itself. When
    both name one key there is no honest winner: resolving it by merge order
    publishes a run that reads as if the declared value was honoured. So the
    combination is refused, naming the conflicting keys.

    The rule is deliberately on key *presence*, not on differing values -- a caller
    who repeats a value has still asked the escape hatch to own that key, and a
    rule that depended on equality would pass or fail with the sampling values.

    One authority so every backend states the same rule; two backends that each
    read correctly on their own but teach a caller two different rules is the
    defect this exists to prevent.
    """
    conflicts = standard_keys.intersection(options)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise GenerationError(f"{option_name} cannot override standard keys: {names}")


def with_backend_metadata(
    response: GenerationResponse,
    metadata: Mapping[str, Any],
) -> GenerationResponse:
    """Return a response with additional transport metadata, without mutation."""

    return replace(response, backend_metadata=dict(response.backend_metadata) | dict(metadata))


def require_trainable(backend: GenerationBackend) -> BackendCapabilities:
    """Fail loudly unless ``backend`` can produce trainable token-native output."""
    caps = backend.capabilities
    if not isinstance(caps, BackendCapabilities) or not caps.trainable:
        raise CapabilityError(
            f"{type(backend).__name__} is not trainable; use a token-native backend "
            "(a local engine or an OpenAI server with return_token_ids)."
        )
    return caps


__all__ = [
    "BATCH_EXECUTION_CLIENT_CONCURRENT",
    "BATCH_EXECUTION_SEQUENTIAL",
    "BATCH_EXECUTION_SERVER_BATCHED",
    "POLICY_SOURCE_LOCAL_ENGINE",
    "POLICY_SOURCE_REMOTE_ENGINE",
    "BackendCapabilities",
    "CapabilityError",
    "Provenance",
    "GenerationBackend",
    "GenerationError",
    "GenerationRequest",
    "GenerationResponse",
    "SamplingOptions",
    "ToolCall",
    "fingerprint",
    "reject_standard_key_overrides",
    "require_trainable",
    "verify_batch_identity",
    "with_backend_metadata",
]
