"""Coverage for #185 acceptance gaps + hardening from the codex review.

Adds the enumerated OpenAI cases (reasoning content, provider exception, usage),
verl finish_reason + fingerprint provenance, and pins the review fixes: batch
identity enforcement + duplicate rejection, >1-choice rejection, non-int token
rejection, non-string tool-arg rejection, empty-span non-trainability, and that
provider_options cannot disable the forced trainable-mode flags.

Also pins the two backends to ONE rule where they used to disagree: an escape
hatch may not claim a key the request already declares. And pins the deliberate
asymmetry that remains -- only a backend talking to a replica pool acts on a
routing_key -- so it stays a stated limit rather than an accident.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from alphaapollo.common.generation import (
    BackendCapabilities,
    GenerationBackend,
    GenerationError,
    GenerationRequest,
    GenerationResponse,
    Provenance,
    SamplingOptions,
    VerlLLMServerGenerationBackend,
    fingerprint,
    verify_batch_identity,
)
from alphaapollo.common.generation.backends.openai_compatible import (
    OpenAICompatibleGenerationBackend,
)


def _req(rid="r0", *, gid="g", sid=0):
    return GenerationRequest(
        request_id=rid,
        model="qwen",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingOptions(temperature=0.7, max_tokens=8),
        group_id=gid,
        sample_id=sid,
    )


# --- base contract hardening ---------------------------------------------------


def test_verify_batch_rejects_duplicate_identity():
    reqs = [_req("a", sid=0), _req("a", sid=0)]  # duplicate request_id + sample
    resps = [GenerationResponse(request_id="a", content="x", group_id="g", sample_id=0)] * 2
    with pytest.raises(GenerationError, match="duplicate"):
        verify_batch_identity(reqs, resps)


def test_default_group_batch_is_not_a_duplicate():
    # Two unrelated prompts with distinct request_ids and no grouping id share the
    # default ("", 0) slot; that must NOT be treated as a duplicate (regression:
    # the uniqueness fix must not reject the ordinary batch-unrelated-prompts case).
    client = _client(_resp(content="ans"))
    b = _backend(client)
    reqs = [
        GenerationRequest(
            request_id="a",
            model="qwen",
            messages=[{"role": "user", "content": "hi"}],
            sampling=SamplingOptions(temperature=0.7, max_tokens=8),
        ),
        GenerationRequest(
            request_id="b",
            model="qwen",
            messages=[{"role": "user", "content": "yo"}],
            sampling=SamplingOptions(temperature=0.7, max_tokens=8),
        ),
    ]
    out = b.generate_batch(reqs)
    assert [r.request_id for r in out] == ["a", "b"]


def test_empty_span_is_not_trainable():
    resp = GenerationResponse(
        request_id="r0",
        content="",
        prompt_token_ids=[1, 2],
        response_token_ids=[],
        response_logprobs=[],
        provenance=Provenance(policy_model="m", tokenizer_id="t"),
    )
    assert resp.is_trainable is False  # empty response span must not be trainable


def test_sampling_rejects_bad_ranges():
    with pytest.raises(ValueError, match="temperature"):
        SamplingOptions(temperature=-0.1, max_tokens=4)
    with pytest.raises(ValueError, match="top_p"):
        SamplingOptions(temperature=0.7, max_tokens=4, top_p=1.5)


def test_request_rejects_non_mapping_messages():
    with pytest.raises(TypeError, match="mapping"):
        GenerationRequest(
            request_id="r0",
            model="m",
            messages="not messages",
            sampling=SamplingOptions(temperature=0.0, max_tokens=4),
        )


# --- OpenAI backend ------------------------------------------------------------


class _Completions:
    def __init__(self, response, exc=None):
        self.response, self.exc, self.kwargs = response, exc, {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.response


def _client(response=None, exc=None):
    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions(response, exc)))


def _resp(
    *,
    content="hi",
    reasoning=None,
    token_ids=None,
    logprobs=None,
    prompt_ids=None,
    n_choices=1,
    tool_calls=None,
):
    ch = SimpleNamespace(
        message=SimpleNamespace(
            content=content, reasoning_content=reasoning, tool_calls=tool_calls
        ),
        finish_reason="stop",
        token_ids=token_ids,
        logprobs=SimpleNamespace(content=[SimpleNamespace(logprob=v) for v in logprobs])
        if logprobs is not None
        else None,
    )
    return SimpleNamespace(
        id="c1",
        model="qwen",
        choices=[ch] * n_choices,
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        prompt_token_ids=prompt_ids,
    )


def _backend(client, **kw):
    return OpenAICompatibleGenerationBackend(
        base_url="http://x", client_factory=lambda **_: client, **kw
    )


def test_openai_reasoning_and_usage_surfaced():
    b = _backend(_client(_resp(content="ans", reasoning="because")))
    r = b.generate(_req())
    assert r.reasoning_content == "because"
    assert r.usage["total_tokens"] == 5


def test_openai_provider_exception_propagates():
    b = _backend(_client(exc=RuntimeError("provider down")))
    with pytest.raises(RuntimeError, match="provider down"):
        b.generate(_req())


def test_openai_rejects_multiple_choices():
    b = _backend(_client(_resp(n_choices=2)))
    with pytest.raises(GenerationError, match="expected 1 choice"):
        b.generate(_req())


def test_openai_rejects_non_int_token_ids():
    b = _backend(
        _client(_resp(token_ids=[2, 3.9], logprobs=[-0.1, -0.2], prompt_ids=[1])),
        return_token_ids=True,
    )
    with pytest.raises(GenerationError, match="not an integer"):
        b.generate(_req())


def test_openai_rejects_non_string_tool_args():
    tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="f", arguments={"not": "str"}))
    b = _backend(_client(_resp(content=None, tool_calls=[tc])))
    with pytest.raises(GenerationError, match="raw provider JSON string"):
        b.generate(_req())


def test_provider_options_cannot_disable_return_token_ids():
    client = _client(_resp(token_ids=[10], logprobs=[-0.1], prompt_ids=[1]))
    b = _backend(client, return_token_ids=True)
    req = GenerationRequest(
        request_id="r0",
        model="qwen",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingOptions(temperature=0.7, max_tokens=8),
        provider_options={"logprobs": False, "extra_body": {}},
    )
    b.generate(req)
    assert client.chat.completions.kwargs["logprobs"] is True  # forced back on
    assert client.chat.completions.kwargs["extra_body"]["return_token_ids"] is True


# --- verl backend --------------------------------------------------------------


@dataclass
class _TO:
    token_ids: list
    log_probs: list | None
    stop_reason: str | None = "completed"


class _VClient:
    async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
        return _TO(token_ids=[10, 11], log_probs=[-0.1, -0.2])


class _Tok:
    def apply_chat_template(self, c, *, add_generation_prompt, tokenize):
        return [1, 2]

    def decode(self, ids):
        return "t"


def test_verl_finish_reason_and_fingerprint_provenance():
    b = VerlLLMServerGenerationBackend(
        _VClient(),
        _Tok(),
        model_name="qwen",
        tokenizer_id="tok",
        weights_version="step-9",
        tokenizer_fingerprint=fingerprint("tok", "rev"),
        chat_template_fingerprint=fingerprint("tmpl"),
    )
    r = b.generate(_req())
    assert r.finish_reason == "completed"
    assert r.provenance.weights_version == "step-9"
    assert r.provenance.tokenizer_fingerprint == fingerprint("tok", "rev")
    assert r.provenance.chat_template_fingerprint == fingerprint("tmpl")


def test_verl_backend_construction_imports_no_verl_or_ray():
    # #189 (ShawFeng99 [P1]): Common defines the contract + injected-client adapter
    # only. Importing and constructing the backend with a fake client must not import
    # verl, initialize Ray, or start any model resource. Checked in a fresh
    # interpreter so an unrelated earlier import in this session can't mask it.
    import subprocess
    import sys

    script = (
        "import sys\n"
        "from alphaapollo.common.generation.backends.verl_llm_server import "
        "VerlLLMServerGenerationBackend\n"
        "class _C:\n"
        "    async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):\n"
        "        return None\n"
        "class _T:\n"
        "    def apply_chat_template(self, c, *, add_generation_prompt, tokenize):\n"
        "        return [1]\n"
        "    def decode(self, ids):\n"
        "        return 't'\n"
        "VerlLLMServerGenerationBackend(_C(), _T(), model_name='qwen', tokenizer_id='t')\n"
        "assert 'verl' not in sys.modules, 'verl imported'\n"
        "assert 'ray' not in sys.modules, 'ray imported'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_verl_rejects_request_model_mismatch():
    # request.model is advisory (routing already picked the backend); a non-empty
    # mismatch is refused rather than silently serving a different model.
    b = VerlLLMServerGenerationBackend(_VClient(), _Tok(), model_name="qwen", tokenizer_id="t")
    req = GenerationRequest(
        request_id="r",
        model="a-different-model",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingOptions(temperature=0.0, max_tokens=4),
    )
    with pytest.raises(GenerationError, match="serves model"):
        b.generate(req)


def test_verl_rejects_non_int_tokens():
    class _BadClient:
        async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
            return _TO(token_ids=[10, 3.9], log_probs=[-0.1, -0.2])

    b = VerlLLMServerGenerationBackend(_BadClient(), _Tok(), model_name="qwen", tokenizer_id="t")
    with pytest.raises(GenerationError, match="not an integer"):
        b.generate(_req())


def test_verl_rejects_bool_logprob():
    class _BoolLogprob:
        async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
            return _TO(token_ids=[10, 11], log_probs=[True, -0.2])

    b = VerlLLMServerGenerationBackend(_BoolLogprob(), _Tok(), model_name="qwen", tokenizer_id="t")
    with pytest.raises(GenerationError, match="not a real number"):
        b.generate(_req())


def test_verl_client_exception_propagates():
    class _Down:
        async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
            raise RuntimeError("verl server down")

    b = VerlLLMServerGenerationBackend(_Down(), _Tok(), model_name="qwen", tokenizer_id="t")
    with pytest.raises(RuntimeError, match="verl server down"):
        b.generate(_req())


def test_verl_uses_routing_key_as_transport_id():
    class _Recording:
        def __init__(self):
            self.seen: list[str] = []

        async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
            self.seen.append(request_id)
            return _TO(token_ids=[10, 11], log_probs=[-0.1, -0.2])

    client = _Recording()
    b = VerlLLMServerGenerationBackend(client, _Tok(), model_name="qwen", tokenizer_id="t")
    req = GenerationRequest(
        request_id="rid",
        model="qwen",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingOptions(temperature=0.7, max_tokens=8),
        routing_key="sticky-key",
    )
    b.generate(req)
    assert client.seen == ["sticky-key"]  # sticky session key, not a fresh UUID


def test_verl_falls_back_to_request_id_when_no_routing_key():
    class _Recording:
        def __init__(self):
            self.seen: list[str] = []

        async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
            self.seen.append(request_id)
            return _TO(token_ids=[10, 11], log_probs=[-0.1, -0.2])

    client = _Recording()
    b = VerlLLMServerGenerationBackend(client, _Tok(), model_name="qwen", tokenizer_id="t")
    b.generate(_req("rid7"))
    assert client.seen == ["rid7"]


def test_async_verl_batch_rejects_duplicate_identity():
    class _Ok:
        async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
            return _TO(token_ids=[10, 11], log_probs=[-0.1, -0.2])

    b = VerlLLMServerGenerationBackend(_Ok(), _Tok(), model_name="qwen", tokenizer_id="t")
    reqs = [_req("dup", gid="g", sid=0), _req("dup", gid="g", sid=0)]
    with pytest.raises(GenerationError, match="duplicate"):
        asyncio.run(b.agenerate_batch(reqs))


def test_async_verl_preflight_rejects_dup_before_calling_generate():
    # #189 [P2]: the async path must reject duplicate identities BEFORE spending any
    # generation -- generate() must not be reached for an invalid batch.
    class _Counting:
        def __init__(self):
            self.calls = 0

        async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
            self.calls += 1
            return _TO(token_ids=[10, 11], log_probs=[-0.1, -0.2])

    client = _Counting()
    b = VerlLLMServerGenerationBackend(client, _Tok(), model_name="qwen", tokenizer_id="t")
    reqs = [_req("dup", gid="g", sid=0), _req("dup", gid="g", sid=0)]
    with pytest.raises(GenerationError, match="duplicate"):
        asyncio.run(b.agenerate_batch(reqs))
    assert client.calls == 0  # rejected at preflight, no inference spent


# --- openai logprob strictness + constructor contract --------------------------


def test_openai_rejects_bool_logprob():
    b = _backend(
        _client(_resp(token_ids=[10], logprobs=[True], prompt_ids=[1])),
        return_token_ids=True,
    )
    with pytest.raises(GenerationError, match="not a real number"):
        b.generate(_req())


def test_openai_constructor_rejects_non_bool_return_token_ids():
    with pytest.raises(TypeError, match="return_token_ids must be a bool"):
        OpenAICompatibleGenerationBackend(
            base_url="http://x", client_factory=lambda **_: _client(), return_token_ids="false"
        )


def test_openai_constructor_validates_bool_before_building_client():
    # A raising/side-effecting factory must not run before the capability switch
    # is validated: the TypeError must win over the factory's own error.
    def _boom(**_):
        raise RuntimeError("factory called")

    with pytest.raises(TypeError, match="return_token_ids must be a bool"):
        OpenAICompatibleGenerationBackend(
            base_url="http://x", client_factory=_boom, return_token_ids="false"
        )


def test_openai_rejects_none_tool_args():
    tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="f", arguments=None))
    b = _backend(_client(_resp(content=None, tool_calls=[tc])))
    with pytest.raises(GenerationError, match="raw provider JSON string"):
        b.generate(_req())


# --- #185 acceptance criteria (explicit) --------------------------------------


class _ScriptedBackend(GenerationBackend):
    """A backend that replays scripted (content, tool_calls) per request, echoing identity."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.calls = 0

    @property
    def capabilities(self):
        return BackendCapabilities(
            token_native=False, logprobs=False, tool_calls=True, policy_source="remote_engine"
        )

    def _generate_batch(self, requests):
        out = []
        for r in requests:
            content, tool_calls = self._script[self._i]
            self._i += 1
            self.calls += 1
            out.append(
                GenerationResponse(
                    request_id=r.request_id,
                    group_id=r.group_id,
                    sample_id=r.sample_id,
                    content=content,
                    tool_calls=tuple(tool_calls),
                )
            )
        return out


def test_ac4_n_samples_are_separate_requests_not_nested_candidates():
    # #185 AC4: n samples for one prompt = n requests sharing group_id with distinct
    # sample_id, NEVER a nested candidate list, and SamplingOptions carries no `n`.
    assert "n" not in {f.name for f in dataclasses.fields(SamplingOptions)}
    assert not hasattr(GenerationResponse, "candidates")
    reqs = [
        GenerationRequest(
            request_id=f"g0-{i}",
            model="qwen",
            messages=[{"role": "user", "content": "hi"}],
            sampling=SamplingOptions(temperature=0.7, max_tokens=8),
            group_id="grp",
            sample_id=i,
        )
        for i in range(3)
    ]
    backend = _ScriptedBackend([("s0", []), ("s1", []), ("s2", [])])
    out = backend.generate_batch(reqs)
    assert [r.sample_id for r in out] == [0, 1, 2]  # each sample is its own response
    assert all(r.group_id == "grp" for r in out)
    assert all(not hasattr(r, "candidates") for r in out)


def test_ac5_cancellation_propagates_and_is_not_swallowed():
    # #185 AC5: a cancelled transport call surfaces loudly; the backend never turns a
    # cancellation into a fabricated response. (Concurrent scheduling/backpressure is a
    # #185 non-goal owned by the runtime; the backend only guarantees propagation.)
    class _Cancelled(Exception):
        pass

    b = _backend(_client(exc=_Cancelled("request cancelled")))
    with pytest.raises(_Cancelled, match="cancelled"):
        b.generate(_req())


def test_ac11_two_generations_form_one_tool_to_final_trajectory():
    # #185 AC11: a fake Runtime drives the backend twice (turn 1 -> tool call, turn 2 ->
    # final) to form ONE tool->final trajectory, with no second generation loop and the
    # backend never touching an Environment.
    from alphaapollo.common.generation import ToolCall

    turn1_tool = ToolCall(id="c1", name="get_weather", arguments='{"city":"SF"}')
    backend = _ScriptedBackend([("", [turn1_tool]), ("The weather is sunny.", [])])

    class _FakeRuntime:
        """Calls generate(); on a tool call, appends a tool result and calls generate() again."""

        def __init__(self, backend):
            self.backend = backend
            self.trajectory = []

        def run(self, messages):
            messages = list(messages)
            turn = 0
            while True:
                turn += 1
                resp = self.backend.generate(
                    GenerationRequest(
                        request_id=f"turn-{turn}",
                        model="qwen",
                        messages=messages,
                        sampling=SamplingOptions(temperature=0.0, max_tokens=8),
                    )
                )
                self.trajectory.append(resp)
                if not resp.has_tool_calls:
                    return self.trajectory
                # Environment (not the backend) would execute the tool; here we fake it.
                messages = messages + [
                    {"role": "assistant", "content": resp.content},
                    {"role": "tool", "content": "sunny"},
                ]

    runtime = _FakeRuntime(backend)
    traj = runtime.run([{"role": "user", "content": "weather in SF?"}])

    assert len(traj) == 2  # two Generations compose one Trajectory
    assert traj[0].has_tool_calls and traj[0].tool_calls[0].name == "get_weather"
    assert not traj[1].has_tool_calls and traj[1].content == "The weather is sunny."
    assert backend.calls == 2  # exactly two generations, no nested/second loop
    # the backend exposes no Environment lifecycle -- the runtime owns the loop
    assert not any(hasattr(backend, m) for m in ("init", "step", "close"))


# --- one rule across both backends ---------------------------------------------


def test_both_backends_state_one_override_conflict_rule():
    """Divergence detector: both escape hatches refuse a claimed key the same way.

    ``provider_options`` (OpenAI-compatible) and ``extra_sampling`` (verl) are the
    same idea -- provider extras beside the keys the request declares -- so they
    must not teach a caller two different rules. Both route through
    ``reject_standard_key_overrides``; this fails if a future edit gives either its
    own wording, its own winner-picking, or its own silence.
    """
    b = _backend(_client(_resp(content="hi")))
    req = GenerationRequest(
        request_id="r0",
        model="qwen",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingOptions(temperature=0.7, max_tokens=8),
        provider_options={"temperature": 1.5},
    )
    with pytest.raises(GenerationError) as openai_raised:
        b.generate(req)
    with pytest.raises(GenerationError) as verl_raised:
        VerlLLMServerGenerationBackend(
            _VClient(),
            _Tok(),
            model_name="qwen",
            tokenizer_id="tok",
            extra_sampling={"temperature": 1.5},
        )

    openai_message = str(openai_raised.value)
    verl_message = str(verl_raised.value)
    assert openai_message == "provider_options cannot override standard keys: temperature"
    assert verl_message == "extra_sampling cannot override standard keys: temperature"
    # Same sentence once the hatch's own name is factored out: one rule, two doors.
    assert openai_message.replace("provider_options", "<hatch>") == verl_message.replace(
        "extra_sampling", "<hatch>"
    )


def test_only_the_replica_pool_backend_acts_on_routing_key():
    """A routing_key reaches verl's client and is not relabelled into an HTTP id.

    Rule: ``routing_key`` is a sticky hint for a pool of replicas, deliberately
    shared across the k branches of a step. verl has a pool, so it passes the key
    to its client as the transport request id. An OpenAI-compatible endpoint has no
    such field: ``X-Request-Id`` means *unique request*, so sending a shared key
    there would give k independent generations one recorded ``response_id`` without
    routing anything. This fails if either half of that flips -- if verl stops
    honouring the key, or if the OpenAI path starts smuggling it into a header.
    """
    seen: list[str] = []

    class _Recording:
        async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
            seen.append(request_id)
            return _TO(token_ids=[10, 11], log_probs=[-0.1, -0.2])

    shared_key = "input-7:solve:step-a:iteration-1"
    branches = [
        GenerationRequest(
            request_id=f"rid-{k}",
            model="qwen",
            messages=[{"role": "user", "content": "hi"}],
            sampling=SamplingOptions(temperature=0.7, max_tokens=8),
            group_id="g",
            sample_id=k,
            routing_key=shared_key,
        )
        for k in range(2)
    ]

    VerlLLMServerGenerationBackend(
        _Recording(), _Tok(), model_name="qwen", tokenizer_id="tok"
    ).generate_batch(branches)
    assert seen == [shared_key, shared_key]  # both branches, one replica: the point

    client = _client(_resp(content="hi"))
    responses = _backend(client).generate_batch(branches)
    assert "extra_headers" not in client.chat.completions.kwargs
    # Each branch keeps whatever id the server issued; the shared routing key never
    # becomes the recorded generation id.
    assert all(r.backend_metadata["response_id"] == "c1" for r in responses)
    assert all(shared_key not in str(r.backend_metadata["response_id"]) for r in responses)
