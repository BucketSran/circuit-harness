"""Task-declared public Spectre adapter, with durable uncertain-job recovery."""

from __future__ import annotations

import fcntl
import json
import subprocess
import time
import uuid

from circuit_harness.execution.evaluation.benchmark_spectre import package_identity, public_feedback
from circuit_harness.execution.runtime.journal import atomic_json
from circuit_harness.execution.transport.benchmark_remote import RemoteBenchmarkSpectre


class RemotePublicSpectre(RemoteBenchmarkSpectre):
    purpose = "public"

    def __init__(self, config, evidence, *, deadline):
        super().__init__(config, evidence)
        self.deadline = deadline

    def _remaining(self, timeout):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("public wait budget exhausted; query original job")
        return min(timeout, remaining)

    def cli(self, *arguments, payload=None, timeout=30):
        return super().cli(*arguments, payload=payload, timeout=self._remaining(timeout))

    def download(self, remote_path, destination, *, max_bytes=256 * 1024 * 1024, timeout=90):
        return super().download(
            remote_path, destination, max_bytes=max_bytes, timeout=self._remaining(timeout)
        )


def run_remote_public(*, directory, config, candidate, output, action_id):
    output.mkdir(mode=0o700, exist_ok=True)
    started = time.monotonic()
    with (output / ".wait.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"execution": "unknown_execution", "wait_elapsed_s": 0.0}
        package = directory / "runtime/public-task"
        if package_identity(package, purpose="public") != config["public_package"]:
            raise ValueError("pinned public package changed")
        deadline = started + config["timeout_s"]
        transport = RemotePublicSpectre(
            config["public_remote"], output / "transport", deadline=deadline
        )
        pending = output / "remote-job.json"
        first = not pending.exists()
        if first:
            atomic_json(pending, {"job_id": "public-" + uuid.uuid4().hex})
        job_id = json.loads(pending.read_text())["job_id"]
        execution = "unknown_execution"
        try:
            # Persist before any network action. Uncertain submit is never replayed.
            if first:
                transport.submit(candidate, package, job_id)
            while time.monotonic() < deadline:
                state = transport.query(job_id)
                if state.get("state") == "finished":
                    result = public_feedback(transport.retrieve(job_id))
                    return {
                        "execution": result["execution"],
                        "backend": "remote_spectre",
                        **result["feedback"],
                        "wait_elapsed_s": time.monotonic() - started,
                    }
                if state.get("state") not in {"running", "queued", "archived"}:
                    break
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        except ValueError:
            execution = "invalid_result"
        except (ConnectionError, OSError, TimeoutError, subprocess.SubprocessError):
            # Transport corruption is not a candidate score or retry permission.
            pass
        finally:
            elapsed = time.monotonic() - started
            previous = (
                json.loads((output / "wait.json").read_text())
                if (output / "wait.json").exists()
                else {"elapsed_s": 0}
            )
            atomic_json(output / "wait.json", {"elapsed_s": previous["elapsed_s"] + elapsed})
        return {"execution": execution, "backend": "remote_spectre", "wait_elapsed_s": elapsed}
