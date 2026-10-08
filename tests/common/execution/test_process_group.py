# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contract tests for group-wide signalling of sandboxed children.

Every process here is real and every signal is really delivered: the behaviour
under test is a kernel behaviour, and a mocked ``os.killpg`` would only confirm
that the call was written, not that it reached the subtree.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from alphaapollo.common.execution.process_group import signal_process_group

# A child that spawns one grandchild in its own group, prints the grandchild pid,
# then idles. Both processes outlive the test unless they are signalled. The
# grandchild is reaped on a background thread so that a killed one disappears
# from the process table instead of lingering as a zombie the probes below
# would still count as alive.
_SPAWNER = (
    "import subprocess, sys, threading, time;"
    "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
    "threading.Thread(target=kid.wait, daemon=True).start();"
    "print(kid.pid, flush=True);"
    "time.sleep(60)"
)


def _alive(pid: int) -> bool:
    """True while ``pid`` exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.02)
    return not _alive(pid)


def test_signal_reaches_the_whole_group_not_just_the_leader() -> None:
    """A group signal must stop the grandchild too, or a timeout orphans it."""
    process = subprocess.Popen(
        [sys.executable, "-c", _SPAWNER],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        grandchild = int(process.stdout.readline().strip())
        assert _alive(grandchild)

        signal_process_group(process, signal.SIGKILL)

        assert _wait_gone(grandchild), "grandchild survived a group kill"
        assert process.wait(timeout=5) != 0
    finally:
        if process.poll() is None:
            signal_process_group(process, signal.SIGKILL)
            process.wait()
        process.stdout.close()


def test_a_non_leader_never_takes_its_parents_group_down() -> None:
    """Signalling a non-leader must hit that process alone, never its group.

    Without ``start_new_session=True`` a child's group is the *caller's* group, so
    a naive ``killpg(getpgid(pid))`` would signal this interpreter. Here the group
    leader is a live process we own, and it must survive while the member dies.
    """
    leader = subprocess.Popen(
        [sys.executable, "-c", _SPAWNER],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        member = int(leader.stdout.readline().strip())
        # The member shares the leader's group but does not lead it — the same
        # shape as a child spawned without ``start_new_session``.
        assert os.getpgid(member) == leader.pid

        class _PidHandle:
            """A real pid we own, presented without the Popen that spawned it."""

            pid = member

            def send_signal(self, sig: int) -> None:
                os.kill(self.pid, sig)

        signal_process_group(_PidHandle(), signal.SIGKILL)

        assert _wait_gone(member), "the addressed process was not signalled"
        assert leader.poll() is None, "the group leader was signalled through a non-leader"
    finally:
        if leader.poll() is None:
            signal_process_group(leader, signal.SIGKILL)
            leader.wait()
        leader.stdout.close()


def test_signalling_an_exited_group_is_not_an_error() -> None:
    """Teardown must never raise: it runs beside the result the caller reports."""
    process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        start_new_session=True,
    )
    process.wait()
    signal_process_group(process, signal.SIGKILL)  # reaped pid: a no-op, not a raise
