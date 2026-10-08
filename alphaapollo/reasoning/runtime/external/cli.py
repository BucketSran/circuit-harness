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

"""The part of driving an agent CLI that is identical for every agent.

All three shipped sessions spawn one process per task, write the prompt to
stdin, read a JSONL stream from stdout, and have to turn a timeout or a
non-zero exit into an honest outcome.  Only two things actually differ: the
argument vector, and how the stream is parsed.  Everything else lives here so a
fourth agent adds those two things and nothing else.
"""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alphaapollo.reasoning.runtime.external_agent_runtime import ExternalRunOutcome

__all__ = ["CliRun", "CliSettings", "finalize_outcome", "run_cli"]

#: Enough stderr to diagnose a failure without pasting a whole log into a record.
_STDERR_LIMIT = 2000

#: How long to wait for the pipes after killing the group, before giving up.
_DRAIN_TIMEOUT_S = 10.0


@dataclass(frozen=True, slots=True)
class CliSettings:
    """Validated settings shared by every CLI-backed session."""

    cli: str = "cli"
    timeout_s: float = 600.0
    # Never repr'd: a bridged pi session copies the whole process environment in
    # here, and a stray debug print would put the operator's keys in a log.
    env: Mapping[str, str] | None = field(default=None, repr=False)
    extra_args: Sequence[str] = ()
    #: When set, the CLI writes stdout directly to a private file during execution.
    stdout_path: Path | None = None
    #: Injected instead of spawning: same signature as :func:`_spawn`, so a test
    #: describes the outcome it wants rather than faking a CompletedProcess.
    runner: Callable[..., CliRun] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cli, str) or not self.cli.strip():
            raise ValueError("cli must be a non-empty string")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a positive number")
        if self.env is not None and not isinstance(self.env, Mapping):
            raise TypeError("env must be a mapping")
        if isinstance(self.extra_args, (str, bytes)) or not isinstance(self.extra_args, Sequence):
            raise TypeError("extra_args must be a sequence of arguments")
        if self.stdout_path is not None and not isinstance(self.stdout_path, Path):
            raise TypeError("stdout_path must be a Path")
        object.__setattr__(self, "timeout_s", float(self.timeout_s))
        object.__setattr__(self, "env", dict(self.env) if self.env is not None else None)
        object.__setattr__(self, "extra_args", tuple(str(argument) for argument in self.extra_args))


@dataclass(frozen=True, slots=True)
class CliRun:
    """What one CLI invocation produced, including a timed-out partial stream."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    #: A timed-out stream ended mid-record and that fragment was discarded.
    dropped_partial_record: bool = False
    #: For private evidence only; parsers continue to consume complete ``stdout`` records.
    raw_stdout: str | None = None


def run_cli(
    argv: Sequence[str],
    *,
    prompt: str,
    workspace: Path,
    settings: CliSettings,
) -> CliRun:
    """Run one agent CLI with the prompt on stdin and never raise on timeout.

    A timeout is an outcome, not an exception: the partial stream is the only
    record of what the agent did before it was cut off, so it is preserved.
    """

    kwargs = dict(
        prompt=prompt, workspace=workspace, timeout_s=settings.timeout_s, env=settings.env
    )
    if settings.runner is None:
        run = _spawn(list(argv), **kwargs, stdout_path=settings.stdout_path)
    else:
        run = settings.runner(list(argv), **kwargs)
        if settings.stdout_path is not None:
            descriptor = os.open(settings.stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(run.raw_stdout if run.raw_stdout is not None else run.stdout)
    if not run.timed_out:
        return run
    # SIGKILL lands wherever the CLI's stdout buffer happened to be, so the last
    # line of a killed stream is routinely half a JSON record. Every parser here
    # rejects a malformed line, and the Runtime is fail-fast, so keeping that
    # fragment turns one slow task into a whole batch aborting on a defect the
    # kill created. Only the complete records are an account of what the agent
    # did; the fragment is reported, not parsed. A malformed line in a stream
    # that ran to completion is still an error.
    stdout, dropped = _complete_records(run.stdout)
    return CliRun(
        stdout,
        run.stderr,
        run.returncode,
        timed_out=True,
        dropped_partial_record=dropped,
        raw_stdout=run.stdout,
    )


def _complete_records(stdout: str) -> tuple[str, bool]:
    """Split a killed stream into its whole lines and whether a fragment followed."""

    end = stdout.rfind("\n")
    if end == -1:
        return "", bool(stdout.strip())
    return stdout[: end + 1], bool(stdout[end + 1 :].strip())


def _spawn(
    argv: Sequence[str],
    *,
    prompt: str,
    workspace: Path,
    timeout_s: float,
    env: Mapping[str, str] | None,
    stdout_path: Path | None = None,
) -> CliRun:
    # start_new_session puts the agent and everything it spawns into one process
    # group, so a timeout takes the whole tree down. Killing only the agent would
    # orphan the shell commands and MCP servers it was in the middle of running,
    # which is how a batch evaluation ends up flaky rather than slow.
    output_stream = None
    if stdout_path is not None:
        descriptor = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        output_stream = os.fdopen(descriptor, "wb", buffering=0)
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=output_stream if output_stream is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(workspace),
            env=None if env is None else dict(env),
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(prompt, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(process)
            try:
                # Bounded, because anything still holding the pipe after a group
                # kill would otherwise hang the batch rather than merely slow it.
                stdout, stderr = process.communicate(timeout=_DRAIN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", "partial stream unreadable: the process tree outlived its kill"
    finally:
        if output_stream is not None:
            output_stream.close()
    if stdout_path is not None:
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    return CliRun(stdout or "", stderr, -1 if timed_out else process.returncode, timed_out)


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # The agent is already gone, or the group is not ours to signal; killing
        # the process we do own is the most that can still be done.
        process.kill()


def finalize_outcome(
    outcome: ExternalRunOutcome,
    run: CliRun,
    *,
    timeout_s: float,
    extra: Mapping[str, Any] | None = None,
) -> ExternalRunOutcome:
    """Apply the process-level verdict on top of what the stream claimed.

    A timeout or a non-zero exit is authoritative even when the stream looked
    complete, because a CLI can print a well-formed final record and still fail.
    """

    metadata = dict(outcome.provider_metadata)
    metadata.update(extra or {})
    termination_reason = outcome.termination_reason
    if run.timed_out:
        termination_reason = "timeout"
        metadata.update(timeout_s=timeout_s, partial_stream=True)
        if run.dropped_partial_record:
            metadata["dropped_partial_record"] = True
        if run.stderr.strip():
            # Carries the drain diagnostic when the tree outlived its own kill,
            # which nothing else in the outcome would show.
            metadata["stderr"] = run.stderr[-_STDERR_LIMIT:]
    elif run.returncode != 0:
        termination_reason = "external_error"
        metadata.update(returncode=run.returncode, stderr=run.stderr[-_STDERR_LIMIT:])
    return ExternalRunOutcome(
        final_text=outcome.final_text,
        events=outcome.events,
        termination_reason=termination_reason,
        usage=outcome.usage,
        provider_metadata=metadata,
    )
