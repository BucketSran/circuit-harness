"""Await local replay cleanup even when Harbor cancels its verifier phase."""

import asyncio
import threading
from pathlib import Path

from circuit_harness.execution.benchmark_replay import replay_candidate
from circuit_harness.execution.journal import atomic_json


async def evaluate_replay(candidate, config, directory):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    cancel = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            replay_candidate,
            Path(candidate["candidate_directory"]),
            config.task_package,
            config.opensource.model_dump(),
            directory / "replay",
            cancel=cancel,
        )
    )
    try:
        receipt = await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel.set()
        # run_process wakes on this event, kills its owned process group and
        # waits for it. Docker removal is separately bounded by the runner.
        # Repeated cancellation must not detach a still-running worker.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancel.set()
            except Exception:
                break
        error = worker.exception()
        atomic_json(
            directory / "local-execution.json",
            {
                "state": "cancelled",
                "worker_finished": True,
                "error_type": type(error).__name__ if error else None,
                "receipt": "replay/receipt.json" if not error else None,
            },
        )
        raise
    atomic_json(
        directory / "local-execution.json",
        {
            "state": "completed",
            "worker_finished": True,
            "receipt": "replay/receipt.json",
        },
    )
    return receipt["result"]
