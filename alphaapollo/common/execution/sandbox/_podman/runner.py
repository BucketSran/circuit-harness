# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bounded and streaming subprocess runners for Podman commands."""

from __future__ import annotations

import contextlib
import logging
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from alphaapollo.common.execution.output import OutputAccumulator, TruncationResult
from alphaapollo.common.execution.process_group import signal_process_group
from alphaapollo.common.execution.sandbox.base import CancellationToken, OutputChunk, OutputSink

logger = logging.getLogger(__name__)

PODMAN = "podman"
WORKSPACE = "/workspace"
MAX_OUTPUT_BYTES = 1024 * 1024
_TRUNCATION_NOTICE = "\n[output truncated after 1048576 bytes]"
_OUTPUT_DRAIN_TIMEOUT = 10.0
#: Wall-clock budget for one bounded ``podman exec`` read of a container-side
#: output file. Short because the drain loop issues many of them in sequence.
_REMOTE_READ_TIMEOUT = 2.0


class PodmanCliError(Exception):
    """Raised on a malformed podman_cli invocation (fail-loud)."""


@dataclass(frozen=True)
class PodmanResult:
    """The captured result of one ``podman`` invocation."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    cancelled: bool = False
    stdout_truncation: TruncationResult | None = None
    stderr_truncation: TruncationResult | None = None
    stdout_capture_path: Path | None = None
    stderr_capture_path: Path | None = None


#: A runner executes a podman argv and returns a PodmanResult. Production uses
#: ``default_runner`` (real subprocess); tests inject a fake to assert argv/behaviour.
Runner = Callable[[Sequence[str]], PodmanResult]


def _read_capped(handle: object) -> str:
    """Read a subprocess output file with a bounded memory footprint."""
    # ``TemporaryFile`` exposes seek/read, but keeping this helper duck-typed
    # makes it straightforward to use with file-like test doubles.
    handle.seek(0)  # type: ignore[attr-defined]
    raw = handle.read(MAX_OUTPUT_BYTES + 1)  # type: ignore[attr-defined]
    if len(raw) > MAX_OUTPUT_BYTES:
        return raw[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace") + _TRUNCATION_NOTICE
    return raw.decode("utf-8", errors="replace")


def default_runner(argv: Sequence[str], *, timeout_seconds: float | None = None) -> PodmanResult:
    """Run ``argv`` via subprocess (never shell) and capture stdout/stderr/exit_code.

    Never raises on a non-zero exit — the caller inspects ``exit_code``. Output is
    spooled to temporary files and capped when read back, so a noisy tool cannot
    exhaust the host process memory. Only a genuinely missing ``podman`` binary
    raises (a configuration error, fail-loud).
    """
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise PodmanCliError("timeout_seconds must be positive when provided")
    try:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            # ``subprocess.run(timeout=)`` signals the direct child only, so a
            # timed-out ``podman run``/``create`` would leave any host-side helper
            # it forked running and still appending to the spool files read below.
            # Lead a new session and stop the group as a unit instead.
            with subprocess.Popen(  # noqa: S603 — fixed argv, no shell, injection-safe
                list(argv),
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            ) as process:
                try:
                    exit_code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    signal_process_group(process, signal.SIGKILL)
                    process.wait()
                    return PodmanResult(
                        stdout=_read_capped(stdout_file),
                        stderr=_read_capped(stderr_file)
                        + f"\ncommand timed out after {timeout_seconds}s",
                        exit_code=124,
                        timed_out=True,
                    )
                finally:
                    # Any other exit with the client alive — a KeyboardInterrupt in
                    # ``wait``, say — must not orphan the session we created.
                    if process.poll() is None:
                        signal_process_group(process, signal.SIGKILL)
            return PodmanResult(
                stdout=_read_capped(stdout_file),
                stderr=_read_capped(stderr_file),
                exit_code=exit_code,
            )
    except FileNotFoundError as exc:
        raise PodmanCliError(
            f"podman binary not found while running {list(argv)!r}; is Podman installed?"
        ) from exc
    except OSError as exc:
        return PodmanResult(
            stdout="",
            stderr=f"podman invocation failed: {exc}",
            exit_code=-1,
        )


def _safe_emit(
    sink: OutputSink | None,
    stream: str,
    text: str,
) -> None:
    if sink is None or not text:
        return
    try:
        sink(OutputChunk(stream=stream, text=text))  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - an observer cannot break execution
        logger.warning("output sink failed with %s", type(exc).__name__)


def streaming_runner(
    argv: Sequence[str],
    *,
    timeout_seconds: float | None = None,
    on_output: OutputSink | None = None,
    cancellation: CancellationToken | None = None,
) -> PodmanResult:
    """Run an ordinary argv with live output and cooperative cancellation.

    This generic helper is useful for tests and non-Podman subprocesses. Production
    ``podman exec`` uses :func:`_streaming_podman_exec`, whose container-side spool
    avoids a rootless Podman relay bug for large output. Both paths expose the same
    Pi-like chunks, bounded tail, and complete capture contract.
    """
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise PodmanCliError("timeout_seconds must be positive when provided")
    stdout = OutputAccumulator(temp_file_prefix="apollo-podman-stdout-")
    stderr = OutputAccumulator(temp_file_prefix="apollo-podman-stderr-")
    process: subprocess.Popen[bytes] | None = None
    output_reader: threading.Thread | None = None
    output_done = threading.Event()
    started_at = time.monotonic()
    timed_out = False
    cancelled = False
    reader_errors: list[BaseException] = []
    stdout_source = tempfile.NamedTemporaryFile(
        prefix="apollo-podman-source-stdout-",
        delete=False,
    )
    stderr_source = tempfile.NamedTemporaryFile(
        prefix="apollo-podman-source-stderr-",
        delete=False,
    )
    stdout_source_path = Path(stdout_source.name)
    stderr_source_path = Path(stderr_source.name)

    def follow_output_files() -> None:
        try:
            with (
                stdout_source_path.open("rb", buffering=0) as stdout_reader,
                stderr_source_path.open("rb", buffering=0) as stderr_reader,
            ):
                streams = (
                    ("stdout", stdout_reader, stdout),
                    ("stderr", stderr_reader, stderr),
                )
                drain_deadline: float | None = None
                while True:
                    made_progress = False
                    for stream_name, reader, accumulator in streams:
                        chunk = reader.read(32 * 1024)
                        if not chunk:
                            continue
                        made_progress = True
                        text = accumulator.append(chunk)
                        _safe_emit(on_output, stream_name, text)
                    # Once the process has stopped, keep looping until both files
                    # are drained. A no-progress pass then proves their EOF stable.
                    # A timeout or cancellation kills the whole process group, so an
                    # ordinary background child stops writing and that EOF arrives.
                    # Two writers still outlive the command: one left behind by a
                    # command that exited *normally* (the success path deliberately
                    # kills nothing), and one that called ``setsid()`` itself and so
                    # left the group before it was signalled. Cap the post-exit drain
                    # by a wall-clock budget for those two, or ``output_reader.join()``
                    # hangs the surrounding execution.
                    if output_done.is_set():
                        if not made_progress:
                            break
                        if drain_deadline is None:
                            drain_deadline = time.monotonic() + _OUTPUT_DRAIN_TIMEOUT
                        elif time.monotonic() >= drain_deadline:
                            break
                    if not made_progress:
                        output_done.wait(0.02)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the owner thread
            reader_errors.append(exc)

    try:
        process = subprocess.Popen(  # noqa: S603 - fixed argv, never shell
            list(argv),
            stdout=stdout_source,
            stderr=stderr_source,
            # Lead a new session so a timeout or cancellation below signals the
            # whole subtree. Signalling the direct child alone leaves whatever it
            # spawned running and still appending to the files being drained.
            start_new_session=True,
        )
        # Popen duplicated these descriptors for the child. Closing the parent's
        # writers lets the follower read independently and ensures clean teardown.
        stdout_source.close()
        stderr_source.close()
        output_reader = threading.Thread(
            target=follow_output_files,
            name="apollo-podman-output",
            daemon=True,
        )
        output_reader.start()

        stop_requested_at: float | None = None
        while process.poll() is None:
            if cancellation is not None and cancellation.cancelled:
                if not cancelled and not timed_out:
                    cancelled = True
                    stop_requested_at = time.monotonic()
                    signal_process_group(process, signal.SIGTERM)
            elif (
                timeout_seconds is not None
                and time.monotonic() - started_at >= timeout_seconds
                and not cancelled
                and not timed_out
            ):
                timed_out = True
                stop_requested_at = time.monotonic()
                signal_process_group(process, signal.SIGTERM)
            if (
                stop_requested_at is not None
                and time.monotonic() - stop_requested_at >= 0.5
                and process.returncode is None
            ):
                signal_process_group(process, signal.SIGKILL)
            if cancellation is not None and not cancelled:
                cancellation.wait(0.02)
            else:
                time.sleep(0.02)

        try:
            exit_code = process.wait(timeout=0.5 if (timed_out or cancelled) else None)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGKILL)
            exit_code = process.wait()
        # The group is signalled above, before the drain below. The reverse order
        # would wait on writers nothing has stopped yet.
        output_done.set()
        output_reader.join()
        if reader_errors:
            raise RuntimeError(
                f"stream reader failed with {type(reader_errors[0]).__name__}"
            ) from reader_errors[0]
        stdout_snapshot = stdout.finish()
        stderr_snapshot = stderr.finish()
        if timed_out:
            exit_code = 124
        elif cancelled:
            exit_code = 130
        return PodmanResult(
            stdout=stdout_snapshot.content,
            stderr=stderr_snapshot.content,
            exit_code=exit_code,
            timed_out=timed_out,
            cancelled=cancelled,
            stdout_truncation=(
                stdout_snapshot.truncation if stdout_snapshot.truncation.truncated else None
            ),
            stderr_truncation=(
                stderr_snapshot.truncation if stderr_snapshot.truncation.truncated else None
            ),
            stdout_capture_path=stdout_snapshot.capture_path,
            stderr_capture_path=stderr_snapshot.capture_path,
        )
    except FileNotFoundError as exc:
        stdout.discard()
        stderr.discard()
        raise PodmanCliError(
            f"podman binary not found while running {list(argv)!r}; is Podman installed?"
        ) from exc
    except OSError as exc:
        stdout.discard()
        stderr.discard()
        return PodmanResult(
            stdout="",
            stderr=f"podman invocation failed: {exc}",
            exit_code=-1,
        )
    except Exception:
        stdout.discard()
        stderr.discard()
        raise
    finally:
        if process is not None and process.poll() is None:
            signal_process_group(process, signal.SIGKILL)
            process.wait()
        stdout_source.close()
        stderr_source.close()
        output_done.set()
        if output_reader is not None and output_reader.is_alive():
            output_reader.join()
        stdout_source_path.unlink(missing_ok=True)
        stderr_source_path.unlink(missing_ok=True)


def _streaming_podman_exec(
    name: str,
    command: str,
    *,
    timeout_seconds: float | None,
    on_output: OutputSink | None,
    cancellation: CancellationToken | None,
) -> PodmanResult:
    """Stream a sandbox command through bounded reads of container-side files.

    Rootless Podman 5.x can prematurely stop relaying a large ``podman exec``
    stream when its client is managed with ``Popen``.  The command therefore writes
    to private files *inside the sandbox*, while short, bounded ``podman exec``
    reads follow those files.  User input is still a distinct argv element and is
    never interpolated into the host or wrapper shell.
    """
    invocation_id = uuid.uuid4().hex
    remote_stdout = f"/tmp/alphaapollo-{invocation_id}.stdout"
    remote_stderr = f"/tmp/alphaapollo-{invocation_id}.stderr"
    wrapper = 'bash -c "$1" >"$2" 2>"$3"; alphaapollo_code=$?; exit "$alphaapollo_code"'
    argv = [
        PODMAN,
        "exec",
        "--workdir",
        WORKSPACE,
        name,
        "bash",
        "-c",
        wrapper,
        "alphaapollo-bash",
        command,
        remote_stdout,
        remote_stderr,
    ]
    stdout = OutputAccumulator(temp_file_prefix="apollo-podman-stdout-")
    stderr = OutputAccumulator(temp_file_prefix="apollo-podman-stderr-")
    process: subprocess.Popen[bytes] | None = None
    offsets = {"stdout": 0, "stderr": 0}
    started_at = time.monotonic()
    timed_out = False
    cancelled = False

    def read_chunk(stream_name: str, remote_path: str) -> bool:
        start_byte = offsets[stream_name] + 1
        read_argv = [
            PODMAN,
            "exec",
            name,
            "bash",
            "-c",
            'test ! -f "$1" || tail -c "+$2" "$1" | head -c 32768',
            "alphaapollo-read",
            remote_path,
            str(start_byte),
        ]
        # Same process-group contract as every other spawn here: a read that has
        # to be stopped must stop the whole client, not just its leader.
        with subprocess.Popen(  # noqa: S603 - fixed argv, never shell
            read_argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        ) as reader:
            try:
                data, _stderr = reader.communicate(timeout=_REMOTE_READ_TIMEOUT)
            except subprocess.TimeoutExpired:
                signal_process_group(reader, signal.SIGKILL)
                # Kill before draining: a surviving helper holds these pipes.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    reader.communicate(timeout=_REMOTE_READ_TIMEOUT)
                # KNOWN GAP: ``poll_once`` cannot tell this from "no data yet", so a
                # persistently timing-out read ends the drain loop as if the stream
                # were complete and the captured output is silently short. Logged so
                # a truncated capture is at least traceable; giving the caller a
                # distinguishable signal would change the drain and truncation
                # contract, so it is deliberately left for its own change.
                logger.warning(
                    "sandbox output read timed out after %ss for %s; "
                    "captured output may be incomplete",
                    _REMOTE_READ_TIMEOUT,
                    stream_name,
                )
                return False
            finally:
                if reader.poll() is None:
                    signal_process_group(reader, signal.SIGKILL)
        if not data:
            return False
        offsets[stream_name] += len(data)
        accumulator = stdout if stream_name == "stdout" else stderr
        text = accumulator.append(data)
        _safe_emit(on_output, stream_name, text)
        return True

    def poll_once() -> bool:
        stdout_progress = read_chunk("stdout", remote_stdout)
        stderr_progress = read_chunk("stderr", remote_stderr)
        return stdout_progress or stderr_progress

    try:
        process = subprocess.Popen(  # noqa: S603 - fixed argv, never shell
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Lead a new session so stopping this exec also stops the host-side
            # helpers the podman client forked, rather than leaving them behind.
            # The container's own supervisor is not reachable here and must not
            # be: it was started by a separate invocation and already detached
            # into its own session, and the sandbox process runs inside the
            # container. Only ``kill_container`` stops those, which is what the
            # backend does on timeout or cancellation.
            start_new_session=True,
        )
        stop_requested_at: float | None = None
        while process.poll() is None:
            poll_once()
            if cancellation is not None and cancellation.cancelled:
                cancelled = True
                stop_requested_at = time.monotonic()
                signal_process_group(process, signal.SIGTERM)
                break
            if timeout_seconds is not None and time.monotonic() - started_at >= timeout_seconds:
                timed_out = True
                stop_requested_at = time.monotonic()
                signal_process_group(process, signal.SIGTERM)
                break
            if cancellation is not None:
                cancellation.wait(0.02)
            else:
                time.sleep(0.02)

        try:
            exit_code = process.wait(timeout=0.5 if stop_requested_at is not None else None)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGKILL)
            exit_code = process.wait()

        # A normally completed command has closed both files, so drain every
        # remaining bounded chunk. On cancellation/timeout the backend immediately
        # kills the container; do not wait on a still-running sandbox process here.
        # The drain is still bounded, and the process-group kill above does not
        # shorten it: a background writer here runs *inside* the container, where
        # no host-side signal reaches it — only ``kill_container`` does — so it
        # keeps the remote files growing. Cap the drain by a wall-clock budget and
        # stop early on cancellation, or the episode hangs here.
        if not timed_out and not cancelled:
            drain_deadline = time.monotonic() + _OUTPUT_DRAIN_TIMEOUT
            while (
                poll_once()
                and (cancellation is None or not cancellation.cancelled)
                and time.monotonic() < drain_deadline
            ):
                pass
        stdout_snapshot = stdout.finish()
        stderr_snapshot = stderr.finish()
        if timed_out:
            exit_code = 124
        elif cancelled:
            exit_code = 130
        return PodmanResult(
            stdout=stdout_snapshot.content,
            stderr=stderr_snapshot.content,
            exit_code=exit_code,
            timed_out=timed_out,
            cancelled=cancelled,
            stdout_truncation=(
                stdout_snapshot.truncation if stdout_snapshot.truncation.truncated else None
            ),
            stderr_truncation=(
                stderr_snapshot.truncation if stderr_snapshot.truncation.truncated else None
            ),
            stdout_capture_path=stdout_snapshot.capture_path,
            stderr_capture_path=stderr_snapshot.capture_path,
        )
    except FileNotFoundError as exc:
        stdout.discard()
        stderr.discard()
        raise PodmanCliError(
            f"podman binary not found while running {argv!r}; is Podman installed?"
        ) from exc
    except Exception:
        stdout.discard()
        stderr.discard()
        raise
    finally:
        if process is not None and process.poll() is None:
            signal_process_group(process, signal.SIGKILL)
            process.wait()
        # Cleanup is best-effort: cancellation/timeout may already have killed the
        # entire container, which is the stronger isolation guarantee.
        try:
            default_runner(
                [PODMAN, "exec", name, "rm", "-f", remote_stdout, remote_stderr],
                timeout_seconds=2,
            )
        except Exception as exc:  # noqa: BLE001 - teardown cannot mask execution
            logger.debug(
                "failed to clean sandbox output files for %s with %s",
                name,
                type(exc).__name__,
            )
