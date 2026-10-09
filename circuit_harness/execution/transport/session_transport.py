"""Durable local/SSH action transport shared by Chips public sessions."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from circuit_harness.execution.runtime.journal import atomic_json


class RemoteSessionTransport:
    """Transport an idempotent public action to a task-owned server session."""

    request_command: str
    response_command: str

    def __init__(self, config, evidence):
        self.config = dict(config)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["host"]):
            raise ValueError("host must be an SSH config alias")
        for key in ("python", "bundle", "session"):
            if not Path(config[key]).is_absolute():
                raise ValueError(f"{key} must be absolute")
        self._open_evidence(evidence)

    def _open_evidence(self, evidence):
        self._wait_interrupted = threading.Event()
        self._process_lock = threading.Lock()
        self._process = None
        self.evidence = Path(evidence)
        self.evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
        pending = []
        for path in sorted(self.evidence.glob("*/request.json")):
            request = json.loads(path.read_text(encoding="utf-8"))
            if request.get("id") != path.parent.name:
                raise ValueError("persisted action ID does not match its evidence directory")
            response_path = path.with_name("response.json")
            response = (
                json.loads(response_path.read_text(encoding="utf-8"))
                if response_path.exists()
                else {}
            )
            if type(response.get("ok")) is not bool:
                pending.append(request["id"])
        if len(pending) > 1:
            raise ValueError("multiple unresolved actions; reconcile evidence before continuing")
        self.pending_action_id = pending[0] if pending else None

    def interrupt_wait(self):
        """Stop this client's query/SSH process, never the detached server job."""
        self._wait_interrupted.set()
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()

    def _run_command(self, argv, *, payload, timeout):
        with self._process_lock:
            if self._wait_interrupted.is_set():
                raise ConnectionError("client wait interrupted; server execution is unknown")
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self._process = process
        try:
            try:
                stdout, stderr = process.communicate(
                    None if payload is None else json.dumps(payload),
                    timeout=timeout,
                )
            except BaseException:
                process.kill()
                process.communicate()
                raise
            return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        finally:
            with self._process_lock:
                self._process = None

    def cli(self, *arguments, payload=None, timeout=30):
        command = shlex.join([self.config["python"], "-B", self.config["bundle"], *arguments])
        result = self._run_command(
            [
                "ssh",
                "-T",
                "-o",
                "ControlMaster=no",
                "-o",
                "ControlPath=none",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=10",
                self.config["host"],
                command,
            ],
            payload=payload,
            timeout=timeout,
        )
        if result.returncode not in (0, 1):
            raise ConnectionError("SSH command failed; server action may still be running")
        try:
            return json.loads(result.stdout)
        except ValueError:
            raise ConnectionError(
                "no structured remote reply; do not infer action completion"
            ) from None

    def call(self, tool, arguments, *, action_id=None, timeout_s=210):
        action_id = action_id or uuid.uuid4().hex
        if self.pending_action_id not in (None, action_id):
            return {
                "ok": False,
                "error": "recover_previous_action",
                "action_id": self.pending_action_id,
            }
        self.pending_action_id = action_id
        request = {"id": action_id, "tool": tool, "arguments": arguments}
        record = self.evidence / action_id
        record.mkdir(mode=0o700, exist_ok=True)
        if (record / "request.json").exists():
            if json.loads((record / "request.json").read_text(encoding="utf-8")) != request:
                raise ValueError("local action ID belongs to another request")
        else:
            atomic_json(record / "request.json", request)
        deadline = time.monotonic() + timeout_s
        started = time.time()
        acknowledged = False
        unknown_error = "transport_timeout"
        while time.monotonic() < deadline and not self._wait_interrupted.is_set():
            try:
                if not acknowledged:
                    reply = self.cli(
                        self.request_command,
                        "--session",
                        self.config["session"],
                        "--request",
                        "-",
                        payload=request,
                    )
                    if reply.get("state") == "unknown_execution":
                        unknown_error = "unknown_execution"
                        break
                    acknowledged = reply.get("state") == "accepted"
                    if not acknowledged:
                        raise ValueError("remote request was not accepted")
                reply = self.cli(
                    self.response_command,
                    "--session",
                    self.config["session"],
                    "--id",
                    action_id,
                )
                if reply.get("state") != "pending":
                    if type(reply.get("ok")) is not bool:
                        unknown_error = "unknown_execution"
                        break
                    reply = {**reply, "action_id": action_id}
                    atomic_json(record / "response.json", reply)
                    atomic_json(
                        record / "timing.json",
                        {"started_at": started, "finished_at": time.time()},
                    )
                    self.pending_action_id = None
                    return reply
            except (ConnectionError, subprocess.TimeoutExpired):
                # Resubmit only the identical envelope; the server deduplicates ID.
                pass
            self._wait_interrupted.wait(min(1, max(0, deadline - time.monotonic())))
        reply = {
            "ok": False,
            "error": "wait_interrupted" if self._wait_interrupted.is_set() else unknown_error,
            "action_id": action_id,
            "server_execution": "unknown",
            "guidance": "Recover this same action ID before starting another action.",
        }
        atomic_json(record / "transport.json", reply)
        atomic_json(record / "timing.json", {"started_at": started, "finished_at": time.time()})
        return reply

    def download(self, remote_path, destination, *, max_bytes=256 * 1024 * 1024, timeout=90):
        """Operator-only artifact transfer; never a model-facing file tool."""
        destination = Path(destination)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        command = shlex.join(["cat", "--", str(remote_path)])
        with partial.open("wb") as stream:
            result = subprocess.run(
                [
                    "ssh",
                    "-T",
                    "-o",
                    "ControlMaster=no",
                    "-o",
                    "ControlPath=none",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    "ConnectTimeout=10",
                    self.config["host"],
                    command,
                ],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        if result.returncode or partial.stat().st_size > max_bytes:
            raise ConnectionError("artifact download failed or exceeded limit")
        partial.replace(destination)


class LocalSessionTransport(RemoteSessionTransport):
    """Use the same durable session protocol without an SSH hop."""

    def __init__(self, config, evidence):
        self.config = dict(config)
        for key in ("python", "bundle", "session"):
            if not Path(config[key]).is_absolute():
                raise ValueError(f"{key} must be absolute")
        self._open_evidence(evidence)

    def cli(self, *arguments, payload=None, timeout=30):
        result = self._run_command(
            [self.config["python"], "-B", self.config["bundle"], *arguments],
            payload=payload,
            timeout=timeout,
        )
        if result.returncode not in (0, 1):
            raise ConnectionError("local command failed; server action may still be running")
        try:
            return json.loads(result.stdout)
        except ValueError:
            raise ConnectionError(
                "no structured local reply; do not infer action completion"
            ) from None

    def download(self, remote_path, destination, *, max_bytes=256 * 1024 * 1024):
        source = Path(remote_path)
        if not source.is_file() or source.stat().st_size > max_bytes:
            raise ConnectionError("artifact copy failed or exceeded limit")
        destination = Path(destination)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        try:
            with source.open("rb") as input_stream, partial.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
            if partial.stat().st_size > max_bytes:
                raise ConnectionError("artifact copy exceeded limit")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
