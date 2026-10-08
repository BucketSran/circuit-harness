"""Shared session preparation and freezing for native and installed Harbor agents."""

import asyncio
import os
import sys
from pathlib import Path


async def prepare_session(config, directory: Path, trial_dir: Path):
    import json

    payload = {
        "task": config.task,
        "materials": str(config.materials),
        "checkout": str(config.checkout) if config.checkout is not None else None,
        "kernel": str(config.kernel) if config.kernel is not None else None,
        "directory": str(directory),
        "image": config.image,
        "backend": config.public_backend,
        "cpu_limit": config.public_cpu_limit,
        "public_remote": config.public_remote,
        "public_task_package": str(config.public_task_package)
        if config.public_task_package is not None
        else None,
        "codex": str(config.public_codex) if config.public_codex is not None else None,
        "python": str(config.public_python) if config.public_python is not None else sys.executable,
        "max_actions": config.max_actions,
        "max_simulations": config.max_simulations,
        "timeout_s": config.simulation_timeout_s,
        "max_output_bytes": config.max_output_bytes,
    }
    # Preparation also belongs to Harbor's finite environment-build phase.
    # Use a cancellable process, not a background thread that copies after timeout.
    code = (
        "import json,sys; from pathlib import Path; "
        "from circuit_harness.execution.current_evas_session import create_session; "
        "p=json.load(sys.stdin); "
        "p.update({k:Path(p[k]) if p[k] is not None else None "
        "for k in ('materials','checkout','kernel','directory')}); "
        "create_session(**p)"
    )
    output = trial_dir / "session-setup.log"
    root = Path(__file__).resolve().parents[2]
    process = None
    with output.open("wb") as stream:
        launch = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                stdin=asyncio.subprocess.PIPE,
                stdout=stream,
                stderr=stream,
                start_new_session=True,
                env={"PATH": os.defpath, "PYTHONPATH": str(root)},
            )
        )
        try:
            try:
                process = await asyncio.shield(launch)
            except asyncio.CancelledError:
                process = await launch
                raise
            await process.communicate(json.dumps(payload).encode())
            if process.returncode:
                raise RuntimeError("public session preparation failed; inspect session-setup.log")
        finally:
            if process is not None:
                import signal

                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()


async def freeze_session(directory: Path, reason: str, timeout_s: float) -> dict:
    from circuit_harness.execution.current_evas_session import close_session

    deadline = asyncio.get_running_loop().time() + timeout_s + 5
    while True:
        receipt = close_session(directory, reason)
        if receipt.get("state") != "awaiting_action_recovery":
            return receipt
        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError("public action remains unresolved; retain session for recovery")
        await asyncio.sleep(0.05)
