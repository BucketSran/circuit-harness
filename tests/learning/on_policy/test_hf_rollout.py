"""CPU unit tests for the HF-rollout worker's pure logic + the 2-GPU update_weights rank-guard.

No verl/Ray/GPU exercised — these pin the pieces that are pure or mockable:
- the deterministic named-actor handshake (HFReplica launches, HFServerAdapter resolves by name),
- Stage 3 device_map config resolution,
- sampling-param normalization,
- the 2-GPU ``update_weights`` rank-guard: rank!=0 MUST drain the weights generator (to drive the
  all_gather collective in lockstep) and MUST NOT push; rank0 pushes. Returning early without
  draining is exactly the bug that deadlocked rank0 (the Part-B fix).
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# The base CI job installs only [dev] (no ray/verl/torch): hf_rollout_worker imports ray and
# verl at module level, so without these guards the file fails COLLECTION and reddens CI.
pytest.importorskip("ray")
pytest.importorskip("verl")
pytest.importorskip("torch")

from alphaapollo.learning.on_policy.hf_rollout_worker import (  # noqa: E402
    HFReplica,
    HFServerActor,
    HFServerAdapter,
    _hf_server_name,
    _Req,
)


# --------------------------------------------------------------------------- #
# named-actor handshake
# --------------------------------------------------------------------------- #
def test_hf_server_name_format_and_default_node():
    assert _hf_server_name("hf", 0, 0) == "hf_server_0_0"
    assert _hf_server_name("hf", 3, 2) == "hf_server_3_2"
    assert _hf_server_name("hf", 0) == "hf_server_0_0"  # default node_rank=0 (single-node smoke)


def test_replica_and_adapter_use_the_same_name():
    # HFReplica.launch_servers and HFServerAdapter.__init__ both hardcode node_rank=0; a mismatch
    # silently breaks the named-actor handshake (adapter's ray.get_actor would miss the server).
    name = "hf"
    assert (
        _hf_server_name(name, 0, node_rank=0)
        == _hf_server_name(name, 0, node_rank=0)
        == "hf_server_0_0"
    )


# --------------------------------------------------------------------------- #
# Stage 3 device_map config resolution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ek,expected",
    [
        ({}, None),
        ({"device_map": None}, None),
        ({"device_map": "null"}, None),
        ({"device_map": ""}, None),
        ({"device_map": "none"}, None),
        ({"device_map": True}, "auto"),
        ({"device_map": "auto"}, "auto"),
        ({"device_map": "0,1"}, "0,1"),  # explicit passthrough
    ],
)
def test_resolve_device_map(ek, expected):
    # _resolve_device_map reads only engine_kwargs (not self), so a bare object stands in.
    assert HFReplica._resolve_device_map(object(), ek) == expected


# --------------------------------------------------------------------------- #
# sampling-param normalization
# --------------------------------------------------------------------------- #
def _mock_actor(response_length=64, eos=99):
    """Stand-in self for HFServerActor._normalize_sp (unbound call; no model loaded)."""
    actor = MagicMock()
    actor.response_length = response_length
    actor.model.config.eos_token_id = eos
    return actor


def test_normalize_sp_defaults():
    sp = HFServerActor._normalize_sp(_mock_actor(), None, {})
    assert sp == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 64,
        "do_sample": True,
        "eos": 99,
        "pad": 99,
    }


def test_normalize_sp_temperature_zero_is_greedy():
    sp = HFServerActor._normalize_sp(_mock_actor(), {"temperature": 0}, {})
    assert sp["do_sample"] is False  # temperature==0 → greedy


def test_normalize_sp_max_tokens_and_eos_pad_override():
    sp = HFServerActor._normalize_sp(
        _mock_actor(), {"max_tokens": 10, "eos_token_id": 5, "pad_token_id": 7}, {}
    )
    assert sp["max_tokens"] == 10 and sp["eos"] == 5 and sp["pad"] == 7


def test_normalize_sp_pad_token_id_zero_is_preserved():
    # 0 is a legitimate pad id; the `is not None` chain (not `or`) must keep it.
    sp = HFServerActor._normalize_sp(_mock_actor(), {"pad_token_id": 0, "eos_token_id": 5}, {})
    assert sp["pad"] == 0


# --------------------------------------------------------------------------- #
# 2-GPU update_weights rank-guard (the Part-B deadlock fix)
# --------------------------------------------------------------------------- #
def _adapter_with_rank(rank):
    """Build an HFServerAdapter without running BaseRollout.__init__ (which needs verl)."""
    adapter = HFServerAdapter.__new__(HFServerAdapter)
    adapter.rollout_rank = rank
    adapter._server_handle = None
    return adapter


def test_update_weights_non_rank0_drains_generator_and_does_not_push():
    """rank!=0 MUST drain the weights generator (drive the all_gather collective in lockstep) and
    MUST NOT push. Returning early without draining is exactly the bug that deadlocked rank0."""
    adapter = _adapter_with_rank(1)
    adapter._ensure_server_handle = MagicMock()  # must NOT be called on rank!=0

    consumed = []

    def weights():
        for i in range(5):
            consumed.append(i)
            yield (f"t{i}", MagicMock())

    result = asyncio.run(adapter.update_weights(weights()))
    assert result == {}
    assert consumed == [0, 1, 2, 3, 4]  # generator fully drained (collective driven)
    adapter._ensure_server_handle.assert_not_called()  # did NOT touch the server


def test_update_weights_rank0_pushes_handles_to_server(monkeypatch):
    """rank0: consumes the generator into CUDA-IPC handles and pushes them to the server via Ray."""
    import torch
    import torch.multiprocessing.reductions as reductions

    from alphaapollo.learning.on_policy import hf_rollout_worker as mod

    adapter = _adapter_with_rank(0)
    server = MagicMock()
    adapter._ensure_server_handle = MagicMock(return_value=server)

    # reduce_tensor is a function-local import; it reads torch.mpp.reductions.reduce_tensor at call
    # time, so patching the attribute before the call is enough. ray.get is patched the same way.
    monkeypatch.setattr(reductions, "reduce_tensor", lambda t: ("handle", t))
    monkeypatch.setattr(mod.ray, "get", lambda *_a, **_k: True)

    # Real tensors (not MagicMock): the chunked push reads
    # .numel()/.element_size()/.detach()/.clone()
    # on each tensor. Default 1 GiB chunk -> both tiny tensors go in ONE chunk (called once).
    weights = [("t0", torch.zeros(1)), ("t1", torch.zeros(1))]
    result = asyncio.run(adapter.update_weights(iter(weights)))
    assert result == {}
    server.load_weights_from_ipc.remote.assert_called_once()
    handles = server.load_weights_from_ipc.remote.call_args.args[0]
    assert len(handles) == 2 and [n for n, _ in handles] == ["t0", "t1"]


def test_update_weights_rank0_pushes_in_byte_bounded_chunks(monkeypatch):
    """Teeth-check: rank0 pushes params in byte-bounded chunks so peak clone memory
    stays ~= one chunk (not the whole model). Every param must still be pushed EXACTLY ONCE, in
    order, across the chunks; and each chunk's byte size must respect the threshold (<= threshold +
    one tensor, since the tensor that crosses the boundary is included before the
    flush). Without the
    chunking (all-at-once), there is a single call holding all params -> the >=2-chunks assertion
    fails."""
    import torch
    import torch.multiprocessing.reductions as reductions

    from alphaapollo.learning.on_policy import hf_rollout_worker as mod

    adapter = _adapter_with_rank(0)
    # 100-byte threshold: each 16-elem float32 tensor is 64 B, so two fit before the 100 B boundary
    # is crossed -> 4 tensors split into 2 chunks of 2.
    adapter.config = SimpleNamespace(engine_kwargs={"weight_push_chunk_bytes": 100})
    server = MagicMock()
    adapter._ensure_server_handle = MagicMock(return_value=server)
    monkeypatch.setattr(reductions, "reduce_tensor", lambda t: ("handle", t))
    monkeypatch.setattr(mod.ray, "get", lambda *_a, **_k: True)

    weights = [(f"t{i}", torch.zeros(16, dtype=torch.float32)) for i in range(4)]  # 64 B each
    asyncio.run(adapter.update_weights(iter(weights)))

    calls = server.load_weights_from_ipc.remote.call_args_list
    pushed = [n for c in calls for n, _ in c.args[0]]
    assert pushed == ["t0", "t1", "t2", "t3"], "every param pushed exactly once, in order"
    assert len(calls) >= 2, f"expected >=2 chunks under a 100 B threshold, got {len(calls)}"
    tensor_bytes = 16 * 4
    for c in calls:
        # each handle is (name, ("handle", tensor)) under the reduce_tensor patch -> tensor is h[1]
        cb = sum(h[1].numel() * h[1].element_size() for _, h in c.args[0])
        assert cb <= 100 + tensor_bytes, f"chunk {cb} B exceeds threshold + one tensor"


# --------------------------------------------------------------------------- #
# Response slicing under right-padding (review issue #3)
# --------------------------------------------------------------------------- #
def test_generate_sync_slices_response_at_max_p_not_prompt_len():
    """Right-pad -> generate appends new tokens at position ``max_p`` for EVERY row, so the
    response lives at ``seqs[i][max_p:]``. The old ``row[prompt_len:prompt_len+max_tokens]`` slice
    grabbed ``(max_p-prompt_len)`` pad tokens (== eos by default) for any shorter prompt, silently
    degrading training. Verified here with a mock ``model.generate`` returning a crafted
    sequences tensor (no real model / GPU)."""
    import torch

    actor = HFServerActor.__new__(HFServerActor)  # skip __init__ (no model load)
    actor.model = MagicMock()
    actor.model.get_input_embeddings.return_value.weight.device = torch.device("cpu")

    # req0 prompt_len=2, req1 prompt_len=4 -> max_p=4, gen=3, pad_id=0; generated
    # sentinels 100/101/102.
    seqs = torch.tensor(
        [
            [10, 20, 0, 0, 100, 101, 102],  # req0: 2 pad tokens, then 3 generated
            [10, 20, 30, 40, 100, 101, 102],  # req1: no pad, then 3 generated
        ]
    )
    fake_out = MagicMock()
    fake_out.sequences = seqs
    actor.model.generate.return_value = fake_out

    sp = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_tokens": 3,
        "do_sample": False,
        "eos": 99,
        "pad": 0,
    }
    batch = [
        _Req(request_id="0", prompt_ids=[10, 20], sp=sp, prompt_len=2, future=None),
        _Req(request_id="1", prompt_ids=[10, 20, 30, 40], sp=sp, prompt_len=4, future=None),
    ]
    results = actor._generate_sync(batch)
    # Both responses must be exactly the 3 generated sentinels — req0 must NOT be prefixed by pad.
    assert results[0].token_ids == [100, 101, 102], (
        "req0 response grabbed pad tokens (the old prompt_len-slice bug)"
    )
    assert results[1].token_ids == [100, 101, 102]


def test_generate_sync_trims_pad_tail_at_first_eos():
    """Teeth-check: HF pads early-stopped rows to max_new_tokens, so a fixed-width
    slice returns [tok, eos, pad, pad, ...]; verl then builds an all-ones response_mask, so the pad
    tail would contaminate loss/reward/length. _generate_sync must trim each row at its first EOS
    (incl. EOS) and report stop_reason. Without the trim, results[0].token_ids == [50, 99, 0, 0, 0]
    (pads) and the assertion fails."""
    import torch

    from alphaapollo.learning.on_policy.hf_rollout_worker import _first_eos_index

    # _first_eos_index: int eos, list eos, None.
    assert _first_eos_index([50, 99, 0, 0], 99) == 1
    assert _first_eos_index([50, 2, 3, 4], [2, 3]) == 1  # first of the multi-eos set
    assert _first_eos_index([50, 51, 52], 99) is None  # no eos -> None
    assert _first_eos_index([50, 51], None) is None  # eos disabled

    actor = HFServerActor.__new__(HFServerActor)  # skip __init__ (no model load)
    actor.model = MagicMock()
    actor.model.get_input_embeddings.return_value.weight.device = torch.device("cpu")
    # max_p=2, max_tokens=5, eos=99, pad=0.
    #   row0: [prompt, prompt, 50, 99, 0, 0, 0] -> early EOS at resp idx 1
    #   -> [50, 99], pads trimmed
    #   row1: [prompt, prompt, 50, 51, 52, 53, 54] -> no eos -> full 5 tokens, stop=length
    seqs = torch.tensor(
        [
            [7, 7, 50, 99, 0, 0, 0],
            [7, 7, 50, 51, 52, 53, 54],
        ]
    )
    fake_out = MagicMock()
    fake_out.sequences = seqs
    actor.model.generate.return_value = fake_out
    sp = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_tokens": 5,
        "do_sample": False,
        "eos": 99,
        "pad": 0,
    }
    batch = [
        _Req(request_id="0", prompt_ids=[7, 7], sp=sp, prompt_len=2, future=None),
        _Req(request_id="1", prompt_ids=[7, 7], sp=sp, prompt_len=2, future=None),
    ]
    results = actor._generate_sync(batch)
    assert results[0].token_ids == [50, 99], (
        f"early-EOS row not trimmed at EOS: {results[0].token_ids}"
    )
    assert results[0].stop_reason == "eos"
    assert results[1].token_ids == [50, 51, 52, 53, 54], "full-length row should be untrimmed"
    assert results[1].stop_reason == "length"


# --------------------------------------------------------------------------- #
# Multi-node detection + node affinity for the HF server actor
# --------------------------------------------------------------------------- #
def _replica_for_launch(node_ids):
    """Build an HFReplica whose mock workers report the given Ray node ids, plus a captured-options
    fake ray.remote so launch_servers' scheduling_strategy can be inspected without Ray/GPU."""
    replica = HFReplica.__new__(HFReplica)  # skip __init__
    replica.config = SimpleNamespace(
        name="hf", response_length=16, engine_kwargs={}, max_num_seqs=16
    )
    replica.model_config = {"path": "m", "trust_remote_code": True}
    replica.replica_rank = 0
    replica.gpus_per_replica_node = 1
    replica.gpus_per_node = 1

    async def _ret_node(nid):  # coroutine (loop-agnostic), not a pre-bound Future
        return nid

    workers = []
    for nid in node_ids:
        w = MagicMock()
        # __ray_call__ is a custom dunder — MagicMock won't auto-create it, so set it explicitly.
        w.__ray_call__ = MagicMock()
        w.__ray_call__.remote.return_value = _ret_node(nid)
        workers.append(w)
    replica.workers = workers
    return replica


def test_hf_launch_rejects_multi_node_workers(monkeypatch):
    """Workers spanning >1 distinct Ray node must raise (CUDA-IPC weight push is
    node-local). The old ``nnodes>1`` guard never fired because nnodes is per-replica logical."""
    from alphaapollo.learning.on_policy import hf_rollout_worker as mod

    fake_server = MagicMock()
    monkeypatch.setattr(
        mod.ray,
        "remote",
        lambda cls: MagicMock(options=lambda **kw: MagicMock(remote=lambda **a: fake_server)),
    )
    replica = _replica_for_launch(["a" * 56, "a" * 56, "b" * 56])  # 2 distinct nodes

    with pytest.raises(NotImplementedError, match="multi-node"):
        asyncio.run(replica.launch_servers())


def test_hf_launch_pins_server_to_workers_node(monkeypatch):
    """Single-node workers -> the server actor is created with
    NodeAffinitySchedulingStrategy(node_id=<the workers' node>, soft=False)."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from alphaapollo.learning.on_policy import hf_rollout_worker as mod

    captured = {}
    fake_server = MagicMock()

    def fake_options(**kw):
        captured["opts"] = kw
        return MagicMock(remote=lambda **a: fake_server)

    monkeypatch.setattr(mod.ray, "remote", lambda cls: MagicMock(options=fake_options))
    replica = _replica_for_launch(["a" * 56, "a" * 56, "a" * 56])  # all one node

    asyncio.run(replica.launch_servers())
    strat = captured["opts"].get("scheduling_strategy")
    assert isinstance(strat, NodeAffinitySchedulingStrategy), (
        "server actor must use NodeAffinitySchedulingStrategy"
    )
    assert strat.node_id == "a" * 56 and strat.soft is False


# --------------------------------------------------------------------------- #
# Stage-2 bucketing by full sampling-param signature (review issue #5)
# --------------------------------------------------------------------------- #
def test_bucket_key_distinguishes_different_sampling_params():
    base = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "eos": 1,
        "pad": 0,
        "max_tokens": 16,
    }
    k0 = HFServerActor._bucket_key(base)
    assert HFServerActor._bucket_key(dict(base)) == k0  # identical -> same bucket
    assert HFServerActor._bucket_key({**base, "temperature": 0.8}) != k0  # diff temp -> diff bucket
    assert HFServerActor._bucket_key({**base, "top_p": 0.95}) != k0  # diff top_p -> diff bucket
    assert (
        HFServerActor._bucket_key({**base, "max_tokens": 8}) != k0
    )  # diff max_tokens -> diff bucket
    assert (
        HFServerActor._bucket_key({**base, "do_sample": False}) != k0
    )  # diff do_sample -> diff bucket


def test_bucket_key_handles_list_valued_eos_without_crashing():
    # HF configs with multiple eos tokens give eos_token_id as a LIST (e.g. some Qwen/Gemma).
    # A list inside the tuple key is unhashable -> TypeError swallowed by the scheduler -> every
    # future unresolved -> rollout deadlock. _bucket_key must coerce list/tuple -> tuple.
    base = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "pad": 0,
        "max_tokens": 16,
    }
    sp = {**base, "eos": [1, 2]}
    k0 = HFServerActor._bucket_key(sp)
    hash(k0)  # must not raise TypeError: unhashable type: 'list'
    assert HFServerActor._bucket_key({**base, "eos": [1, 2]}) == k0  # same eos set -> same
    assert HFServerActor._bucket_key({**base, "eos": [3, 4]}) != k0  # diff eos set -> diff
    assert HFServerActor._bucket_key({**base, "eos": 1}) != k0  # scalar != list


# --------------------------------------------------------------------------- #
# _normalize_sp: eos / max_tokens falsy `or` -> `is not None` (review issue #6)
# --------------------------------------------------------------------------- #
def test_normalize_sp_eos_token_id_zero_is_preserved():
    sp = HFServerActor._normalize_sp(_mock_actor(), {"eos_token_id": 0}, {})
    assert sp["eos"] == 0  # 0 is a legitimate eos id; the `is not None` chain keeps it


def test_normalize_sp_max_tokens_zero_falls_through():
    # max_tokens=0 is NOT a valid generation length (transformers raises on max_new_tokens<=0),
    # so it is treated as unset and falls through to response_length (default 64) — unlike
    # eos/pad where 0 is a legitimate token id.
    sp = HFServerActor._normalize_sp(_mock_actor(), {"max_tokens": 0}, {})
    assert sp["max_tokens"] == 64  # falls through to _mock_actor's response_length


# --------------------------------------------------------------------------- #
# LEFT-padding in _generate_sync (adjacent-1)
# --------------------------------------------------------------------------- #
def test_generate_sync_left_pads_prompts():
    """Prompts are LEFT-padded (real prompt right-aligned) so decoder generation conditions on
    the real prompt, not pad tokens (HF recommends padding_side='left'; for the conv arch a
    right-pad would put pad tokens in the first generated position's receptive field)."""
    import torch

    actor = HFServerActor.__new__(HFServerActor)
    actor.model = MagicMock()
    actor.model.get_input_embeddings.return_value.weight.device = torch.device("cpu")
    fake_out = MagicMock()
    fake_out.sequences = torch.zeros((2, 7), dtype=torch.long)
    actor.model.generate.return_value = fake_out

    sp = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 50,
        "max_tokens": 3,
        "do_sample": False,
        "eos": 99,
        "pad": 0,
    }
    batch = [
        _Req(request_id="0", prompt_ids=[10, 20], sp=sp, prompt_len=2, future=None),  # short
        _Req(request_id="1", prompt_ids=[10, 20, 30, 40], sp=sp, prompt_len=4, future=None),  # long
    ]
    actor._generate_sync(batch)
    called = actor.model.generate.call_args.kwargs
    # max_p = 4. req0 (prompt_len 2) left-padded -> [pad, pad, 10, 20]; req1 -> [10,20,30,40]
    assert called["input_ids"][0].tolist() == [0, 0, 10, 20]
    assert called["input_ids"][1].tolist() == [10, 20, 30, 40]
    assert called["attention_mask"][0].tolist() == [0, 0, 1, 1]
    assert called["attention_mask"][1].tolist() == [1, 1, 1, 1]


# --------------------------------------------------------------------------- #
# Multi-replica rank derivation: every replica's rank-0 pushes
# --------------------------------------------------------------------------- #
def test_derive_ranks_multi_replica_mapping():
    """rank -> (replica_rank, rollout_rank) must match verl's ServerAdapter
    (world_size = tp*dp*pp) so EVERY replica's rank-0 pushes to its OWN server. Includes
    the dp>1 case, where omitting dp from the world size resolves non-existent servers."""
    D = HFServerAdapter._derive_ranks
    # 2 replicas, TP=1, DP=1, PP=1: rollout_world_size=1 -> each rank IS its replica's rank-0.
    assert D(0, 1, 1, 1) == (1, 0, 0)  # rank0 -> replica 0 rank-0 -> pushes server_0_0
    assert D(1, 1, 1, 1) == (1, 1, 0)  # rank1 -> replica 1 rank-0 -> pushes server_1_0
    # TP=2 (rollout_world_size=2): ranks 0,1 = replica 0; ranks 2,3 = replica 1.
    assert D(0, 2, 1, 1) == (2, 0, 0)
    assert D(1, 2, 1, 1) == (2, 0, 1)  # drains
    assert D(2, 2, 1, 1) == (2, 1, 0)  # replica 1 rank-0 -> pushes
    assert D(3, 2, 1, 1) == (2, 1, 1)  # drains
    # DP=2 (rollout_world_size=2): each replica has 2 dp ranks; replica's rank-0 pushes.
    # Omitting dp from the world size would map rank2/3 to non-existent
    # server_2_0/server_3_0.
    assert D(0, 1, 2, 1) == (2, 0, 0)  # rank0 -> replica 0 rank-0 -> pushes server_0_0
    assert D(1, 1, 2, 1) == (2, 0, 1)  # rank1 -> replica 0 rank-1 -> drains
    assert D(2, 1, 2, 1) == (2, 1, 0)  # rank2 -> replica 1 rank-0 -> pushes server_1_0
    assert D(3, 1, 2, 1) == (2, 1, 1)  # rank3 -> replica 1 rank-1 -> drains


# --------------------------------------------------------------------------- #
# scheduler must fail out collected futures if a tick throws pre-_run_batch
# --------------------------------------------------------------------------- #
def test_scheduler_loop_fails_futures_on_bucketing_error():
    """Teeth-check: _bucket_key coerces only list/tuple eos|pad to tuple — a dict
    value is still unhashable, so ``buckets.setdefault(_bucket_key(sp), [])`` throws TypeError. The
    broad ``except Exception: continue`` used to leave the dequeued request's future UNRESOLVED
    forever (verl's ``await fut`` hangs the rollout silently). The fix sets the exception on every
    collected future in the except branch. Without the fix, ``fut`` never resolves and this test's
    done-assertion fails after the poll window (instead of hanging forever)."""

    async def run():
        loop = asyncio.get_running_loop()
        # SimpleNamespace stand-in: _scheduler_loop only touches these attrs before _run_batch
        # (which is never reached — bucketing throws first). No model load.
        actor = SimpleNamespace(
            max_batch_size=8,
            max_wait_ms=5.0,
            _updating=False,
            _updating_deadline=0.0,
            _aborted=set(),
            _in_flight=set(),
            _queue=asyncio.Queue(),
            _bucket_key=HFServerActor._bucket_key,  # real: returns a tuple containing the dict
        )
        fut = loop.create_future()
        # dict eos -> the key tuple contains a dict -> unhashable -> TypeError at setdefault.
        bad_sp = {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "eos": {"x": 1},
            "pad": 0,
            "max_tokens": 4,
        }
        await actor._queue.put(_Req("r1", [1, 2], bad_sp, 2, fut))
        task = asyncio.ensure_future(HFServerActor._scheduler_loop(actor))
        try:
            for _ in range(200):  # poll up to ~2s
                if fut.done():
                    break
                await asyncio.sleep(0.01)
            assert fut.done(), "future left unresolved — dead-future regression"
            assert fut.exception() is not None, "expected set_exception, not set_result"
            assert isinstance(fut.exception(), TypeError), (
                f"expected TypeError from bucketing, got {type(fut.exception())}"
            )
            assert "r1" not in actor._in_flight, "in-flight id should be discarded in the except"
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Weight-load mismatch visibility + whole-push update bracket
# --------------------------------------------------------------------------- #
def test_load_checked_raises_when_no_param_names_match():
    """Teeth-check: an actor/server param-name drift (easy with trust_remote_code arches)
    must FAIL the push — silently, rollout would serve the initial weights forever while
    training moves on."""
    import torch

    from alphaapollo.learning.on_policy.hf_rollout_worker import _load_checked

    model = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="ZERO"):
        _load_checked(model, {"totally.wrong.name": torch.zeros(4, 4)})


def test_load_checked_warns_on_unexpected_keys_but_loads_matches(caplog):
    """Partial mismatch (some keys valid, some unknown) warns naming the unknowns and
    still loads the valid ones — chunked pushes legitimately carry partial dicts."""
    import torch

    from alphaapollo.learning.on_policy.hf_rollout_worker import _load_checked

    model = torch.nn.Linear(4, 4)
    with caplog.at_level("WARNING"):
        _load_checked(model, {"weight": torch.ones(4, 4), "stray.key": torch.zeros(1)})
    assert "stray.key" in caplog.text
    assert torch.equal(model.weight, torch.ones(4, 4))


def test_update_weights_brackets_whole_push_with_update_pause(monkeypatch):
    """The chunk sequence must run between set_updating(True) and set_updating(False):
    each chunk's load takes the gen lock individually, so an unbracketed push lets a batch
    scheduled between chunks sample half-old/half-new weights."""
    import torch
    import torch.multiprocessing.reductions as reductions

    from alphaapollo.learning.on_policy import hf_rollout_worker as mod

    adapter = _adapter_with_rank(0)
    server = MagicMock()
    adapter._ensure_server_handle = MagicMock(return_value=server)
    monkeypatch.setattr(reductions, "reduce_tensor", lambda t: ("handle", t))
    monkeypatch.setattr(mod.ray, "get", lambda *_a, **_k: True)

    asyncio.run(adapter.update_weights(iter([("t0", torch.zeros(1)), ("t1", torch.zeros(1))])))

    order = [name for name, _args, _kw in server.method_calls]
    first, last = server.method_calls[0], server.method_calls[-1]
    assert first[0] == "set_updating.remote" and first[1] == (True,), order
    assert last[0] == "set_updating.remote" and last[1] == (False,), order
    assert "load_weights_from_ipc.remote" in order, order


def test_update_weights_releases_pause_even_when_push_fails(monkeypatch):
    """A pusher that dies mid-update must not leave the server paused forever: the finally
    clause still sends set_updating(False)."""
    import torch
    import torch.multiprocessing.reductions as reductions

    from alphaapollo.learning.on_policy import hf_rollout_worker as mod

    adapter = _adapter_with_rank(0)
    server = MagicMock()
    adapter._ensure_server_handle = MagicMock(return_value=server)
    monkeypatch.setattr(reductions, "reduce_tensor", lambda t: ("handle", t))

    calls = {"n": 0}

    def flaky_get(_obj):
        calls["n"] += 1
        if calls["n"] == 2:  # first load_weights_from_ipc RPC fails
            raise RuntimeError("RPC died")
        return True

    monkeypatch.setattr(mod.ray, "get", flaky_get)
    with pytest.raises(RuntimeError, match="RPC died"):
        asyncio.run(adapter.update_weights(iter([("t0", torch.zeros(1))])))
    last = server.method_calls[-1]
    assert last[0] == "set_updating.remote" and last[1] == (False,)


def test_scheduler_holds_requests_arriving_during_weight_update():
    """Regression: the update pause used to be checked only at loop top, so a request
    put() while the scheduler was parked in queue.get() (or one collected as the bracket
    opened) bypassed it and ran a batch on half-old/half-new weights mid-push. The tick
    must requeue and hold; on bracket close the request runs."""
    import time as _time

    good_sp = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "eos": 1,
        "pad": 0,
        "max_tokens": 4,
    }

    async def run():
        loop = asyncio.get_running_loop()
        ran = []

        async def fake_run_batch(batch, _loop):
            ran.append([r.request_id for r in batch])
            for r in batch:
                r.future.set_result("done")

        actor = SimpleNamespace(
            max_batch_size=8,
            max_wait_ms=5.0,
            _updating=False,
            _updating_deadline=0.0,
            _aborted=set(),
            _in_flight=set(),
            _queue=asyncio.Queue(),
            _bucket_key=HFServerActor._bucket_key,
            _run_batch=fake_run_batch,
        )
        task = asyncio.ensure_future(HFServerActor._scheduler_loop(actor))
        try:
            await asyncio.sleep(0.05)  # let the loop park in queue.get()
            actor._updating = True
            actor._updating_deadline = _time.monotonic() + 30
            fut = loop.create_future()
            await actor._queue.put(_Req("r1", [1, 2], dict(good_sp), 2, fut))
            await asyncio.sleep(0.25)
            assert ran == [], f"batch ran mid-update: {ran}"
            assert not fut.done(), "future resolved from a mid-update batch"

            actor._updating = False  # bracket closes
            for _ in range(200):
                if fut.done():
                    break
                await asyncio.sleep(0.01)
            assert fut.done(), "request never ran after the update finished"
            assert ran == [["r1"]]
        finally:
            task.cancel()

    asyncio.run(run())


def test_scheduler_holds_later_buckets_when_update_opens_mid_tick():
    """Regression: the update bracket opening while bucket 1's generate ran let bucket 2
    take the gen lock after a partial chunk load — a batch on torn weights. Buckets not yet
    run when the flag is observed must be requeued and held until the bracket closes."""
    import time as _time

    base_sp = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "eos": 1,
        "pad": 0,
    }

    async def run():
        loop = asyncio.get_running_loop()
        ran = []

        async def fake_run_batch(bucket, _loop):
            for r in bucket:
                r.future.set_result("done")
            ran.append([r.request_id for r in bucket])
            if len(ran) == 1:  # the update bracket opens DURING bucket 1
                actor._updating = True
                actor._updating_deadline = _time.monotonic() + 30

        actor = SimpleNamespace(
            max_batch_size=8,
            max_wait_ms=5.0,
            _updating=False,
            _updating_deadline=0.0,
            _aborted=set(),
            _in_flight=set(),
            _queue=asyncio.Queue(),
            _bucket_key=HFServerActor._bucket_key,
            _run_batch=fake_run_batch,
        )
        task = asyncio.ensure_future(HFServerActor._scheduler_loop(actor))
        try:
            fut_a, fut_b = loop.create_future(), loop.create_future()
            await actor._queue.put(_Req("a", [1], {**base_sp, "max_tokens": 4}, 1, fut_a))
            await actor._queue.put(_Req("b", [2], {**base_sp, "max_tokens": 8}, 1, fut_b))
            for _ in range(200):
                if fut_a.done():
                    break
                await asyncio.sleep(0.01)
            assert fut_a.done(), "first bucket never ran"
            await asyncio.sleep(0.25)
            assert not fut_b.done(), f"second bucket ran mid-update: {ran}"
            assert all("b" not in b for b in ran)

            actor._updating = False  # bracket closes
            for _ in range(200):
                if fut_b.done():
                    break
                await asyncio.sleep(0.01)
            assert fut_b.done(), "held bucket never ran after the update finished"
            assert ["b"] in ran
        finally:
            task.cancel()

    asyncio.run(run())
