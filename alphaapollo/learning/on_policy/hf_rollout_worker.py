"""verl 0.9 'hf' rollout — standalone HF-generate server + Ray-RPC weight transfer.

For custom architectures vLLM can't load (e.g. conv-based causal LMs). verl 0.9's rollout is
server-based: RayPPOTrainer → LLMServerClient → load_balancer → ``server.generate.remote()`` on a
Ray actor returning ``TokenOutput``. Generation NEVER goes through ``worker.self.rollout``; it goes
through a Ray actor. The actor worker is a *fused worker* whose methods are name-prefixed, so it
cannot serve ``generate`` directly. We therefore mirror the vLLM shape: a **standalone HF server
actor** (own model copy) + **Ray-RPC weight transfer** (no ZMQ — HF scale is fine over Ray).

Pieces (all model-agnostic; new arches only need an HF modeling file):

- ``HFServerActor`` — plain ``ray.remote`` class (so ``generate`` is Ray-visible). Loads the HF
  model itself; ``generate → TokenOutput``; ``load_weights[_from_ipc]``.
  * Stage 2: continuous batching (asyncio.Queue + scheduler + run_in_executor) — collapses the N
    concurrent ``generate.remote()`` verl fires into ONE padded ``model.generate(batch)`` per tick.
    For conv LMs (no KV cache) this cuts forward count N×T → ~T.
  * Stage 3: optional ``device_map`` model parallel (HF native) so models too big for one GPU load
    across GPUs; weight-push remaps each IPC handle to its param's actual device.
- ``HFReplica(RolloutReplica)`` — ``launch_servers`` creates the HF server actor under a
  deterministic Ray **named actor** name (vLLM does the same for its control plane), with
  ``max_concurrency`` so verl's concurrent RPCs land as coroutines (Stage 2 prerequisite).
- ``HFServerAdapter(BaseRollout)`` — bound to ``worker.self.rollout``; ``update_weights`` resolves
  the server by name via ``ray.get_actor`` and pushes the actor's per-tensor params over Ray RPC.

Address handoff = deterministic named actor (``ray.get_actor("{name}_server_{replica}_{node}")``),
identical to how vLLM/SGLang couple their ServerAdapter to their server — no new verl machinery.

Stage 2/3 config knobs live in the mutable ``actor_rollout_ref.rollout.engine_kwargs`` dict:
    engine_kwargs.max_batch_size=16     # Stage 2 batch cap per scheduler tick
    engine_kwargs.max_wait_ms=5.0       # Stage 2 wait window to fill a batch
    engine_kwargs.device_map=null       # Stage 3: null=single .cuda(); "auto"/"0,1"=model parallel
    engine_kwargs.weight_push_chunk_bytes=1073741824  # update_weights chunk size (peak-mem bound)

Launch flags:
    actor_rollout_ref.rollout.name=hf
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive  # in-process → update_weights
    actor_rollout_ref.model.use_remove_padding=false            # engine feeds padded [bs,seq]
    actor_rollout_ref.model.enable_gradient_checkpointing=false
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal as _signal
import sys as _sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import ray
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import RolloutReplica, RolloutReplicaRegistry, TokenOutput

logger = logging.getLogger(__name__)

# Debug aid: `kill -USR1 <worker_pid>` dumps all-thread Python tracebacks to stderr (Ray captures
# them into the run log) — lets you localize a hung worker WITHOUT sudo (py-spy needs ptrace). Zero
# overhead unless signaled. This is how the update_weights all_gather deadlock (see HFServerAdapter.
# update_weights) was pinned to rank0-in-collective / rank1-idle.
try:
    import faulthandler as _faulthandler

    _faulthandler.register(_signal.SIGUSR1, file=_sys.stderr, all_threads=True)
except Exception as _e:  # pragma: no cover
    logger.debug("faulthandler SIGUSR1 hook not installed: %s", _e)


@dataclass
class _Req:
    """One in-flight generation request in the Stage 2 batching queue."""

    request_id: str
    prompt_ids: list
    sp: dict  # normalized sampling params (temperature/top_p/top_k/max_tokens/do_sample/eos/pad)
    prompt_len: int
    future: Any  # asyncio.Future resolving to TokenOutput


def _first_eos_index(tokens, eos_ids):
    """Index of the first EOS token in ``tokens`` (inclusive stop), or ``None``.

    ``eos_ids`` may be an int, a list/tuple of ints (HF configs with multiple eos tokens,
    e.g. some Qwen/Gemma), or ``None`` (no eos configured -> never stops on eos). Used by
    ``_generate_sync`` to trim each generated row at its first EOS so HF's per-row pad tail
    does not leak into the response."""
    if eos_ids is None:
        return None
    if isinstance(eos_ids, (list, tuple)):
        eos_set = set(eos_ids)
        for i, t in enumerate(tokens):
            if t in eos_set:
                return i
    else:
        for i, t in enumerate(tokens):
            if t == eos_ids:
                return i
    return None


# A weight update pauses generation for at most this long before the server resumes on its
# own: a pusher that died between set_updating(True) and set_updating(False) must not freeze
# rollout forever. Late resume is recoverable; a stuck server is not.
_UPDATE_PAUSE_TIMEOUT_S = 300.0


def _load_checked(model, state_dict) -> None:
    """``load_state_dict(strict=False)`` with name mismatches made visible.

    strict=False is required — chunked pushes carry partial dicts, and the server model holds
    buffers no push includes. But a FULLY unmatched dict means the actor's param names and the
    HF server's have drifted apart (easy with trust_remote_code custom arches): silently, the
    server would keep serving the initial weights while training moves on — RL that "runs"
    while the rollout policy never tracks the actor. Zero matches therefore raise; partially
    unexpected keys warn, naming a sample.
    """
    model_keys = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    incoming = set(state_dict)
    unexpected = sorted(incoming - model_keys)
    if unexpected:
        logger.warning(
            "weight push carries %d keys the model does not define (first: %s); "
            "check the actor/server param naming.",
            len(unexpected),
            unexpected[:5],
        )
    if not (incoming & model_keys):
        raise RuntimeError(
            f"weight push matched ZERO of the model's {len(model_keys)} params "
            f"({len(incoming)} incoming, first: {sorted(incoming)[:5]}): the actor and the "
            "HF server model disagree on parameter names, so rollout would never see updates."
        )
    model.load_state_dict(state_dict, strict=False)


# --------------------------------------------------------------------------- #
# Standalone HF server actor (vLLM-HttpServer-shaped). Plain class → ray.remote().
# --------------------------------------------------------------------------- #
class HFServerActor:
    """Loads the HF model in its own Ray actor process; serves generate + weight load.

    Stage 2: ``generate`` enqueues the request and awaits a future; a background scheduler drains
    the queue, pads up to ``max_batch_size`` requests (or within ``max_wait_ms``) into ONE
    ``model.generate(batch)`` run in a dedicated worker thread (so the event loop isn't blocked),
    and resolves each future. For conv LMs (no KV cache) this collapses N×T forwards → ~T.
    Stage 3: if ``device_map`` is set, the model is sharded across GPUs via HF native device_map
    (pipeline model parallel); weight-push remaps each IPC handle to its param's actual device.
    """

    def __init__(
        self,
        model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        attn_implementation="eager",
        response_length=256,
        max_batch_size=16,
        max_wait_ms=5.0,
        device_map=None,
        device=None,
    ):
        import torch
        from transformers import AutoModelForCausalLM

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            dtype, torch.bfloat16
        )
        if device_map in (None, False):
            # Fast path: single GPU, no pipeline bubble (default for small models). Pin to a
            # SPECIFIC GPU when `device` is given: without the pin every replica's
            # server lands on cuda:0 — RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES lets the actor
            # see all GPUs and .cuda() defaults to cuda:0.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                dtype=torch_dtype,
                attn_implementation=attn_implementation,
            )
            tgt = f"cuda:{int(device)}" if device is not None else "cuda"
            self.model = self.model.to(tgt).eval()
        else:
            # Stage 3: HF native device_map (pipeline model parallel) across visible GPUs.
            dm = "auto" if device_map is True else device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                dtype=torch_dtype,
                attn_implementation=attn_implementation,
                device_map=dm,
            ).eval()
        # name -> cuda device index, fixed at load. Stage 3 per-param weight-push remap (a[6]).
        self._param_device = {n: p.device.index for n, p in self.model.named_parameters()}

        self.response_length = int(response_length)
        self.global_steps = 0
        # Stage 2 batching state
        self.max_batch_size = int(max_batch_size)
        self.max_wait_ms = float(max_wait_ms)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._gen_lock = asyncio.Lock()  # serialize model.generate vs load_state_dict
        self._aborted: set = set()
        self._in_flight: set = set()  # request IDs currently in a running _run_batch
        self._updating = False  # weight update in flight: scheduler pauses dequeuing
        self._updating_deadline = 0.0
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-gen")
        self._scheduler_task = None
        logger.info(
            "HFServerActor loaded %s (resp_len=%d, max_batch=%d, max_wait_ms=%.1f, device_map=%s)",
            model_path,
            self.response_length,
            self.max_batch_size,
            self.max_wait_ms,
            device_map,
        )

    # ---- sampling-param normalization (shared by generate + scheduler) ----
    def _normalize_sp(self, sampling_params, kwargs):
        sp = sampling_params or {}
        temperature = float(sp.get("temperature", 1.0))
        top_p = float(sp.get("top_p", 1.0))
        top_k = int(sp.get("top_k", -1))
        # max_tokens=0 is NOT a valid generation length (transformers raises on
        # max_new_tokens<=0), so treat 0/falsy as unset and fall through to response_length —
        # this differs from eos/pad below, where 0 IS a legitimate token id (those use
        # `is not None`). max_tokens=0 must fall through, not be preserved, or
        # GenerationConfig(max_new_tokens=0) crashes generate.
        max_tokens = (
            sp.get("max_tokens")
            or sp.get("max_new_tokens")
            or sp.get("response_length")
            or self.response_length
        )
        do_sample = temperature > 0
        # `is not None` (not `or`): an explicit eos_token_id of 0 must be preserved.
        eos = sp.get("eos_token_id")
        if eos is None:
            eos = kwargs.get("eos_token_id")
        if eos is None:
            eos = getattr(self.model.config, "eos_token_id", None)
        # `is not None` (not `or`): an explicit pad_token_id of 0 must be preserved — `0 or eos`
        # would wrongly fall through to eos. (pinned by
        # test_normalize_sp_pad_token_id_zero_is_preserved)
        pad = sp.get("pad_token_id")
        if pad is None:
            pad = kwargs.get("pad_token_id")
        if pad is None:
            pad = eos
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": int(max_tokens),
            "do_sample": do_sample,
            "eos": eos,
            "pad": pad,
        }

    def _ensure_scheduler(self):
        # Lazy start: the Ray actor's event loop is running by the first generate() call but may
        # not be the current loop at __init__ time.
        if self._scheduler_task is None:
            loop = asyncio.get_running_loop()
            self._scheduler_task = loop.create_task(self._scheduler_loop())

    async def generate(
        self,
        request_id,
        *,
        prompt_ids,
        sampling_params,
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        priority=0,
        **kwargs,
    ):
        self._ensure_scheduler()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        sp = self._normalize_sp(sampling_params, kwargs)
        req = _Req(
            request_id=str(request_id),
            prompt_ids=list(prompt_ids),
            sp=sp,
            prompt_len=len(prompt_ids),
            future=fut,
        )
        await self._queue.put(req)
        try:
            return await fut
        except asyncio.CancelledError:
            self._aborted.add(str(request_id))
            return TokenOutput(token_ids=[], stop_reason="aborted")

    async def _scheduler_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            # A multi-chunk weight update is in flight: hold off dequeuing (the pre-_run_batch
            # recheck below is the definitive gate; this one avoids dequeue/requeue churn).
            # Requests stay queued (they are NOT aborted). A deadline guards against a pusher
            # that died between set_updating(True/False).
            if self._updating:
                if time.monotonic() > self._updating_deadline:
                    logger.error(
                        "weight-update pause exceeded %.0fs — resuming generation; "
                        "a pusher likely died mid-update",
                        _UPDATE_PAUSE_TIMEOUT_S,
                    )
                    self._updating = False
                else:
                    await asyncio.sleep(0.002)
                    continue
            # Requests collected this tick. None until the first dequeue succeeds; tracked so the
            # except branch can fail out any requests it holds — otherwise a throw between dequeue
            # and _run_batch leaves their futures UNRESOLVED forever.
            batch = None
            try:
                first = await self._queue.get()
                self._in_flight.add(first.request_id)
                batch = [first]
                # loop.time() is SECONDS, so convert ms->s: a bare `+ max_wait_ms`
                # made the default 5.0 wait 5 SECONDS to fill a batch, not 5 ms.
                deadline = loop.time() + self.max_wait_ms / 1000.0
                # Greedily collect more up to max_batch_size OR max_wait_ms.
                while len(batch) < self.max_batch_size:
                    timeout = max(0.0, deadline - loop.time())
                    if timeout <= 0:
                        break
                    try:
                        r = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                        batch.append(r)
                        self._in_flight.add(r.request_id)
                    except asyncio.TimeoutError:
                        break
                # Drop anything aborted while queued/collecting; resolve + untrack them. (Tracking
                # from DEQUEUE closes the window where abort during batch
                # collection missed the dequeued-but-not-yet-running requests.)
                kept = []
                for r in batch:
                    if r.request_id in self._aborted:
                        if not r.future.done():
                            r.future.set_result(TokenOutput(token_ids=[], stop_reason="aborted"))
                        self._in_flight.discard(r.request_id)
                    else:
                        kept.append(r)
                batch = kept
                if not batch:
                    continue

                def requeue(reqs):
                    """Return requests to the queue (not aborted, not held locally) so abort's
                    get_nowait drain can still reach them and the pause can re-examine them."""
                    for r in reqs:
                        self._queue.put_nowait(r)
                        self._in_flight.discard(r.request_id)

                # A weight-update bracket may have opened while this tick was parked in
                # queue.get() or collecting — the loop-top check cannot see those. Running now
                # would batch on half-updated weights, so requeue the whole tick and wait.
                if self._updating:
                    requeue(batch)
                    await asyncio.sleep(0.002)
                    continue
                # Bucket by the FULL sampling-param signature (not just do_sample) so every request
                # in a bucket shares ONE GenerationConfig. Otherwise batch[0]'s
                # temp/top_p/top_k/eos/pad/max_tokens silently override the others.
                buckets = {}
                for r in batch:
                    buckets.setdefault(self._bucket_key(r.sp), []).append(r)
                pending = list(buckets.values())
                for i, bucket in enumerate(pending):
                    # Same window BETWEEN buckets: bucket i-1's generate released the gen lock,
                    # a chunk may have loaded, and bucket i would run on torn weights.
                    if self._updating:
                        for b in pending[i:]:
                            requeue(b)
                        await asyncio.sleep(0.002)
                        break
                    await self._run_batch(bucket, loop)
            except Exception as e:
                logger.exception("HFServerActor scheduler tick failed; failing collected requests")
                # If the tick failed AFTER collecting requests but BEFORE _run_batch resolved them
                # — e.g. _bucket_key raised on an unhashable eos/pad (only list/tuple are coerced,
                # so a dict/set value still throws at buckets.setdefault) — their futures would stay
                # UNRESOLVED forever and verl's ``await fut`` would hang the rollout silently.
                # (_run_batch handles its OWN failures with set_exception; this closes the pre-batch
                # window.) Fail them out so the client sees the error instead of an infinite hang.
                if batch is not None:
                    for r in batch:
                        if not r.future.done():
                            r.future.set_exception(e)
                        self._in_flight.discard(r.request_id)
                continue

    async def _run_batch(self, batch, loop):
        # _in_flight is populated at DEQUEUE in _scheduler_loop (so abort catches the batch during
        # the collection window too); here we just run, resolve, and untrack.
        try:
            try:
                async with self._gen_lock:
                    outs = await loop.run_in_executor(self._exec, self._generate_sync, batch)
            except Exception as e:
                logger.exception("HFServerActor batch generate failed")
                for r in batch:
                    if not r.future.done():
                        r.future.set_exception(e)
                return
            for req, out in zip(batch, outs, strict=True):
                if req.request_id in self._aborted:
                    if not req.future.done():
                        req.future.set_result(TokenOutput(token_ids=[], stop_reason="aborted"))
                elif not req.future.done():
                    req.future.set_result(out)
        finally:
            for r in batch:
                self._in_flight.discard(r.request_id)

    @staticmethod
    def _bucket_key(sp):
        """Sampling-param signature for Stage-2 batching: requests sharing a key share ONE
        GenerationConfig. Keying on the full signature (not just do_sample) prevents batch[0]'s
        temp/top_p/top_k/eos/pad/max_tokens from silently overriding the others.

        eos/pad may be a Python list (HF configs with multiple eos tokens, e.g. some Qwen/Gemma) —
        coerce list/tuple values to tuple so the key stays HASHABLE (a list in the tuple raised
        TypeError in ``setdefault``, which the scheduler's broad ``except`` swallowed, leaving
        every future unresolved -> rollout deadlock)."""

        def _h(v):
            return tuple(v) if isinstance(v, (list, tuple)) else v

        return (
            sp["do_sample"],
            sp["temperature"],
            sp["top_p"],
            sp["top_k"],
            _h(sp["eos"]),
            _h(sp["pad"]),
            sp["max_tokens"],
        )

    def _generate_sync(self, batch):
        """Runs in self._exec (single worker thread). One padded batch generate."""
        import torch
        from transformers import GenerationConfig

        device = self.model.get_input_embeddings().weight.device  # device_map-aware input placement
        max_p = max(r.prompt_len for r in batch)
        sp = batch[
            0
        ].sp  # bucketed by full sampling-param signature -> representative of the bucket
        pad_id = sp["pad"] if sp["pad"] is not None else 0
        # LEFT-pad (real prompt right-aligned). HF warns that right-padding corrupts decoder
        # generation (it recommends padding_side='left'); for the conv arch the first generated
        # token's receptive field would otherwise land on pad tokens. generate appends at max_p
        # regardless of pad side, so the response slice row[max_p:] is unchanged.
        input_ids = torch.full((len(batch), max_p), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(batch), max_p), dtype=torch.long, device=device)
        for i, r in enumerate(batch):
            off = max_p - r.prompt_len
            input_ids[i, off:] = torch.tensor(r.prompt_ids, dtype=torch.long, device=device)
            attn[i, off:] = 1
        max_tokens = max(r.sp["max_tokens"] for r in batch)
        gen_cfg = GenerationConfig(
            do_sample=sp["do_sample"],
            num_beams=1,
            temperature=max(sp["temperature"], 1e-6) if sp["do_sample"] else 1.0,
            top_p=sp["top_p"],
            top_k=max(0, sp["top_k"]),
            max_new_tokens=int(max_tokens),
            eos_token_id=sp["eos"],
            pad_token_id=sp["pad"],
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                generation_config=gen_cfg,
                return_dict_in_generate=True,
                use_cache=True,
            )
        seqs = out.sequences  # [B, max_p + gen]
        results = []
        for i, _r in enumerate(batch):
            row = seqs[i].tolist()
            # New tokens are appended at position max_p for EVERY row regardless of pad side, so the
            # response lives at row[max_p:]. The OLD slice
            # row[prompt_len:prompt_len+max_tokens] grabbed (max_p-prompt_len) pad tokens (== eos
            # by default) for any shorter prompt, making verl see the response as pre-terminated
            # (review issue #3). max_p+max_tokens never overruns: HF pads out.sequences to
            # max_p + max_new_tokens (= max_tokens here) with pad_token_id for early-stopped rows.
            resp = row[max_p : max_p + max_tokens]
            # Trim at the first EOS (incl.): HF pads early-stopped rows out to max_new_tokens, so a
            # fixed-width slice would return [tok, eos, pad, pad, ...]. verl's agent loop builds an
            # all-ones response_mask over the returned tokens (single_turn_agent_loop.py:71), so pad
            # tail tokens would contaminate log-probs/loss/reward/response_length. Trim so only the
            # real response remains; report eos vs length.
            eos_idx = _first_eos_index(resp, sp["eos"])
            if eos_idx is not None:
                resp = resp[: eos_idx + 1]
                stop_reason = "eos"
            else:
                stop_reason = "length"
            results.append(TokenOutput(token_ids=resp, stop_reason=stop_reason))
        return results

    async def set_updating(self, flag: bool):
        """Bracket a multi-chunk weight update: pause the scheduler for the WHOLE push.

        Each chunk's load takes the gen lock individually, so without this bracket a batch
        scheduled between chunks samples half-old/half-new weights. A deadline bounds a lost
        resume (pusher died mid-update): the scheduler resumes and logs loudly rather than
        freezing rollout forever.
        """
        self._updating = flag
        self._updating_deadline = time.monotonic() + _UPDATE_PAUSE_TIMEOUT_S if flag else 0.0
        return True

    async def load_weights(self, state_dict, global_steps=None):
        async with self._gen_lock:
            _load_checked(self.model, state_dict)
        if global_steps is not None:
            self.global_steps = global_steps
        return True

    async def load_weights_from_ipc(self, handles, global_steps=None):
        """Receive CUDA-IPC handles and load_state_dict. Each handle is the ``(func, args)`` tuple
        from ``torch.multiprocessing.reductions.reduce_tensor``. Stage 3: remap ``args[6]`` to each
        param's ACTUAL device (device_map-aware) so the IPC tensor opens on the right GPU;
        falls back
        to the first param's device for buffers/unknown names (load_state_dict fixes placement)."""
        async with self._gen_lock:
            default_dev = next(self.model.parameters()).device.index
            state_dict = {}
            for name, handle in handles:
                func, args = handle
                a = list(args)
                a[6] = self._param_device.get(name, default_dev)
                state_dict[name] = func(*a)
            _load_checked(self.model, state_dict)
        if global_steps is not None:
            self.global_steps = global_steps
        return True

    # ---- control RPCs RolloutReplica fans out (replica.py:265-299) ----
    async def wake_up(self, *args, **kwargs):
        return {}

    async def sleep(self, *args, **kwargs):
        return {}

    async def clear_kv_cache(self, *args, **kwargs):
        return {}

    async def set_global_steps(self, global_steps):
        self.global_steps = global_steps
        return {}

    async def abort_all_requests(self, *args, **kwargs):
        # Drain queued requests as aborted; in-flight batches resolve aborted after they finish
        # (HF generate has no mid-step cancellation hook).
        while True:
            try:
                r = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._aborted.add(r.request_id)
            if not r.future.done():
                r.future.set_result(TokenOutput(token_ids=[], stop_reason="aborted"))
        # Also mark the currently-running batch's requests aborted: they already left
        # the queue so the drain loop above can't reach them. Adding them to _aborted makes the
        # in-flight _run_batch return aborted results once its generate finishes (HF generate has no
        # mid-step cancellation hook).
        self._aborted |= self._in_flight
        return {}

    async def resume_generation(self, *args, **kwargs):
        # verl pairs abort_all_requests -> resume_generation each step (replica.py:277-279). Trim
        # _aborted to only the still-in-flight ids: a plain .clear() would UN-ABORT an
        # in-flight batch if its generate (running in the executor thread) hasn't finished by resume
        # time — the event loop processes resume before the executor completes, so _run_batch's
        # post-generate check would find _aborted empty and return a normal completion instead of
        # aborted. Intersection keeps the in-flight aborts, clears the stale ones (leak fix).
        self._aborted &= self._in_flight
        return {}

    def get_server_address(self):
        return "hf-server"

    def get_device_distribution(self):
        """Debug aid: {cuda_device_index: num_params} — verifies Stage 3 device_map spread."""
        from collections import Counter

        return dict(Counter(self._param_device.values()))

    def get_master_address(self):
        return "hf-server"


# --------------------------------------------------------------------------- #
# HF replica: builds the standalone HF server actor under a deterministic name.
# --------------------------------------------------------------------------- #
def _hf_server_name(rollout_name, replica_rank, node_rank=0):
    return f"{rollout_name}_server_{replica_rank}_{node_rank}"


class HFReplica(RolloutReplica):
    """Creates a standalone HFServerActor (Ray named actor) per replica node."""

    async def launch_servers(self):
        rollout_name = self.config.name
        # single-node smoke: node_rank 0; multi-node would iterate worker node_ids like vLLM.
        name = _hf_server_name(rollout_name, self.replica_rank, node_rank=0)
        model_path = (
            self.model_config.get("local_path") or self.model_config.get("path")
            if hasattr(self.model_config, "get")
            else None
        )
        trust = (
            bool(self.model_config.get("trust_remote_code", False))
            if hasattr(self.model_config, "get")
            else True
        )
        resp_len = self.config.response_length
        # Stage 2/3 knobs live in the mutable engine_kwargs dict (no verl dataclass edit needed).
        ek = dict(getattr(self.config, "engine_kwargs", None) or {})
        device_map = self._resolve_device_map(ek)
        max_batch = int(ek.get("max_batch_size", 16))
        max_wait = float(ek.get("max_wait_ms", 5.0))
        # Multi-node detection via the replica's workers' REAL Ray node ids.
        # RolloutReplica.nnodes (world_size // gpus_per_replica_node) is a PER-REPLICA logical
        # count — always 1 when TP=DP=PP=1 — NOT the number of physical Ray nodes, so it can
        # never detect a multi-node trainer. If the server actor (no node affinity) landed on
        # a different node than the worker that pushes weights, the CUDA-IPC push would hang
        # or fail. Query the workers' node ids (as sglang/vLLM do): if they span
        # >1 node, the HF path can't safely do node-local CUDA-IPC weight push -> reject; otherwise
        # pin the server actor to that single node (NodeAffinitySchedulingStrategy below) so Ray
        # co-locates it with the workers.
        workers = list(getattr(self, "workers", None) or [])
        if workers:
            node_ids = await asyncio.gather(
                *[
                    w.__ray_call__.remote(lambda _self: ray.get_runtime_context().get_node_id())
                    for w in workers
                ]
            )
            distinct_nodes = set(node_ids)
            if len(distinct_nodes) > 1:
                raise NotImplementedError(
                    "HF-rollout multi-node replicas are not supported: CUDA-IPC weight push is "
                    f"node-local, but this replica's workers span {len(distinct_nodes)} Ray nodes. "
                    "Use single-node replicas, or implement cross-node weight transfer."
                )
            server_node_id = node_ids[0]
        else:
            # No worker handles (not hybrid mode): bind to this process's node as the best available
            # co-location; we couldn't run the multi-node check, so log it.
            server_node_id = ray.get_runtime_context().get_node_id()
            logger.warning(
                "HFReplica: self.workers empty; binding server to this node (%s) without "
                "a multi-node check.",
                server_node_id,
            )
        # Stage-3 device_map + multi-replica: accelerate's "auto" does NOT know replica boundaries,
        # so multiple replicas would each span ALL visible GPUs → OOM pile-on. Reject clearly until
        # CUDA_VISIBLE_DEVICES is restricted per replica.
        if device_map not in (None, False) and int(getattr(self, "gpus_per_node", 1)) > int(
            getattr(self, "gpus_per_replica_node", 1)
        ):
            raise NotImplementedError(
                "HF-rollout device_map + multi-replica: accelerate device_map='auto' "
                "ignores replica boundaries → replicas pile on the same GPUs (OOM). "
                "Use single-replica, or restrict each actor's CUDA_VISIBLE_DEVICES "
                "to its replica's GPUs."
            )
        device = None
        if device_map in (None, False):
            gprn = int(getattr(self, "gpus_per_replica_node", 1))
            device = self.replica_rank * gprn
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        server = (
            ray.remote(HFServerActor)
            .options(
                name=name,
                # Pin the server actor to the replica's node: without
                # affinity, Ray could schedule it on a different node than the worker
                # that pushes weights, and the CUDA-IPC weight push
                # (load_weights_from_ipc) would fail/hang across nodes. soft=False so Ray
                # honors the placement instead of falling back to any node.
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=server_node_id, soft=False
                ),
                # Inherit the parent's CUDA_VISIBLE_DEVICES (like vLLMHttpServer) so the
                # server shares the actor worker's GPUs instead of being GPU-isolated by Ray.
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
                # Stage 2 prerequisite: let verl's N concurrent generate.remote() land as coroutines
                # (Ray default max_concurrency=1 would serialize them at the dispatch
                # layer). RolloutReplica defines max_concurrency = max(1000,
                # max_num_seqs+16) (replica.py:256-260).
                max_concurrency=self.max_concurrency,
            )
            .remote(
                model_path=model_path,
                trust_remote_code=trust,
                response_length=resp_len,
                max_batch_size=max_batch,
                max_wait_ms=max_wait,
                device_map=device_map,
                device=device,
            )
        )
        self.servers = [server]
        self._server_handle = server
        self._server_address = name
        logger.info(
            "HFReplica launched HFServerActor: %s (max_batch=%d, max_wait_ms=%.1f, device_map=%s)",
            name,
            max_batch,
            max_wait,
            device_map,
        )

    def _resolve_device_map(self, engine_kwargs):
        """Stage 3: None = single-GPU fast path; otherwise HF device_map ('auto' / explicit)."""
        dm = engine_kwargs.get("device_map", None)
        if dm in (None, "null", "", "none"):
            return None
        if dm is True or dm == "auto":
            return "auto"
        return dm  # e.g. "0,1" / list / dict — passthrough to HF

    def _get_server_name_prefix(self):
        return f"{self.config.name}_"


# --------------------------------------------------------------------------- #
# HF ServerAdapter: bound to worker.self.rollout. Pushes actor weights to the HF server over Ray.
# --------------------------------------------------------------------------- #
class HFServerAdapter(BaseRollout):
    """Weight-sync adapter. update_weights pushes the actor's per-tensor params to the HF server
    (resolved by deterministic Ray named-actor name). resume/release are no-ops (shared nothing
    between actor and the HF server process except the pushed weights)."""

    def __init__(self, config=None, model_config=None, device_mesh=None, replica_rank=0, **kwargs):
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        rollout_name = getattr(config, "name", "hf") if config is not None else "hf"
        # Derive replica/rank the way verl's ServerAdapter does: the rollout engine's
        # per-replica world size is tp*pp; replica_rank = RANK // rollout_world_size
        # (which replica),
        # rollout_rank = RANK % rollout_world_size (rank within it). The server name uses
        # replica_rank so it matches HFReplica.launch_servers' _server_{replica}_0, and each
        # replica's rank-0 pushes to its OWN server — otherwise with DP>1 only replica-0's server is
        # ever updated and the others serve STALE weights. (Prior code used global RANK as
        # rollout_rank, so rank-1 in a 2-replica TP=1 setup drained+returned and hf_server_1_0 was
        # never updated.)
        rank = int(os.environ.get("RANK", 0))
        tp = int(getattr(config, "tensor_model_parallel_size", 1)) if config is not None else 1
        dp = int(getattr(config, "data_parallel_size", 1)) if config is not None else 1
        pp = int(getattr(config, "pipeline_model_parallel_size", 1)) if config is not None else 1
        _rws, self.replica_rank, self.rollout_rank = self._derive_ranks(rank, tp, dp, pp)
        self._server_name = _hf_server_name(rollout_name, self.replica_rank, node_rank=0)
        self._server_handle = None

    @staticmethod
    def _derive_ranks(rank, tp, dp, pp):
        """verl ServerAdapter rank derivation: the rollout engine's per-replica
        world size is tp*dp*pp (mcore/verl RolloutReplica.world_size = tp*dp*pp, layout
        workers[r*world_size:(r+1)*world_size] — replica.py:107-111,139). replica_rank = rank //
        rollout_world_size (which replica), rollout_rank = rank % rollout_world_size (rank within
        it). Each replica's rank-0 pushes to its OWN server. The world size MUST include
        dp: omitting it resolves non-existent server names for dp>1.

        Returns ``(rollout_world_size, replica_rank, rollout_rank)``."""
        rollout_world_size = max(1, tp * dp * pp)
        return rollout_world_size, rank // rollout_world_size, rank % rollout_world_size

    def _ensure_server_handle(self):
        if self._server_handle is None:
            self._server_handle = ray.get_actor(self._server_name)
        return self._server_handle

    async def update_weights(self, weights, global_steps=None, **kwargs):
        # `weights` is a LAZY generator; iterating it runs `get_per_tensor_param -> full_tensor()`,
        # an NCCL all_gather COLLECTIVE (transformer_impl.py:838-846). EVERY rank must iterate it in
        # lockstep — if a non-rank-0 returns without iterating, rank0's all_gather
        # DEADLOCKS (found via faulthandler: rank0 pinned in transformer_impl.py:841
        # <genexpr>, rank1 idle). So non-rank-0 must still drain the generator (driving
        # the collective) before returning. Single-GPU is fine
        # because the only rank IS rank0 and the all_gather is a self-gather.
        if self.rollout_rank != 0:
            for (
                _
            ) in weights:  # rank!=0: drive the all_gather collective in lockstep, discard results
                pass
            return {}
        # rank0: ship CUDA-IPC handles over Ray RPC (TRT-LLM style, trtllm_rollout.py:506-520). Ray
        # ships KB handles, not GB; HFServerActor rebuilds each tensor on the same GPU (zero copy).
        # .detach().clone() so each handle owns stable storage; the clone is freed once the server's
        # load_state_dict has copied it (load_state_dict does param.copy_ — an in-place copy, so the
        # IPC-mapped source is safe to release after ray.get returns).
        #
        # Chunked push: pushing every param at once materializes EVERY param as a clone
        # simultaneously and holds them until the RPC returns -> a FULL model copy on
        # rank0's GPU on top of the resident model (~2x peak — the OOM point as models
        # grow). Push in byte-bounded chunks instead: each chunk's clones drop (handles
        # go out of scope) before the next chunk is built, so peak clone memory ~= one
        # chunk, not the whole model.
        #
        # This does NOT perturb the all_gather collective: `weights` drives one all_gather
        # PER ITEM (inside full_tensor, at next() time), and each item's all_gather has
        # ALREADY completed by the time we build its handle. Chunking only inserts ray.get
        # waits between groups of items; rank!=0 still drains the full generator (it
        # blocks at each all_gather until rank0 arrives).
        # Chunk size via engine_kwargs.weight_push_chunk_bytes (default 1 GiB; set very large to
        # recover the all-at-once behavior for small models that prefer one round-trip).
        from torch.multiprocessing.reductions import reduce_tensor

        cfg = getattr(self, "config", None)
        ek = getattr(cfg, "engine_kwargs", None) or {}
        chunk_bytes = int(ek.get("weight_push_chunk_bytes") or (1024**3))
        server = self._ensure_server_handle()
        # Pause generation for the WHOLE chunk sequence (each chunk's load takes the gen lock
        # individually — without the bracket, a batch between chunks sees torn weights).
        ray.get(server.set_updating.remote(True))
        try:
            chunk = []
            used = 0
            for name, t in weights:
                chunk.append((name, t))
                used += t.numel() * t.element_size()
                if used >= chunk_bytes:
                    handles = [(n, reduce_tensor(x.detach().clone())) for n, x in chunk]
                    ray.get(server.load_weights_from_ipc.remote(handles, global_steps=global_steps))
                    chunk = []
                    used = 0
            if chunk:
                handles = [(n, reduce_tensor(x.detach().clone())) for n, x in chunk]
                ray.get(server.load_weights_from_ipc.remote(handles, global_steps=global_steps))
        finally:
            ray.get(server.set_updating.remote(False))
        return {}

    async def resume(self, tags=None, *args, **kwargs):
        return None

    async def release(self, *args, **kwargs):
        return None

    async def generate_sequences(self, *args, **kwargs):
        raise NotImplementedError(
            "HFServerAdapter.generate_sequences unused — generation goes via the HFServerActor "
            "Ray-actor generate() (HFReplica._server_handle)."
        )


# --------------------------------------------------------------------------- #
# Worker: builds the actor via super(); self.rollout = HFServerAdapter (weight push only).
# --------------------------------------------------------------------------- #
class AAHFRolloutWorker(ActorRolloutRefWorker):
    """ActorRolloutRefWorker whose self.rollout is HFServerAdapter (hf path). Generation and
    weight transfer are both handled by the standalone HFServerActor + this adapter."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import verl.workers.engine_workers as _ew

        is_hf = self.config.rollout.get("name", None) == "hf"
        orig_get_rollout_class = _ew.get_rollout_class
        if is_hf:
            _ew.get_rollout_class = lambda name, mode: HFServerAdapter
        try:
            super().init_model()
        finally:
            _ew.get_rollout_class = orig_get_rollout_class


# --------------------------------------------------------------------------- #
# Register 'hf' on import.
# --------------------------------------------------------------------------- #
def _load_hf():
    return HFReplica


try:
    RolloutReplicaRegistry.register("hf", _load_hf)
except Exception as e:  # already registered
    logger.debug("RolloutReplicaRegistry 'hf' register skipped: %s", e)
