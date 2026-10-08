# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Group-wide signalling for sandboxed child processes.

A sandboxed command may spawn children of its own. ``Popen.terminate()``,
``Popen.kill()`` and the ``timeout=`` of :func:`subprocess.run` all signal only
the direct child, so a timeout leaves whatever that child started still running
— still burning CPU, still holding the inherited pipes or output files the
caller is about to drain.

Backends therefore start sandboxed commands with ``start_new_session=True``,
which makes the child a session and process-group leader, and stop them through
:func:`signal_process_group` so the signal reaches the whole subtree as a unit.
The two halves are one contract: without the flag at the spawn site this helper
deliberately declines to signal (see below), and the orphan comes back.
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

__all__ = ["signal_process_group"]


def signal_process_group(process: subprocess.Popen, sig: int) -> None:
    """Send ``sig`` to the process group led by ``process``.

    ``process`` must have been spawned with ``start_new_session=True``. The group
    is resolved at signal time rather than cached at spawn time, so a reaped pid
    can never be mistaken for a live group after the number is recycled.

    Two cases fall back to signalling the child alone, both deliberately:

    * ``process`` does not lead its own group — the caller dropped
      ``start_new_session=True``. Its group is then the *parent's* group, and
      signalling it would kill this interpreter along with the sandbox.
    * The group is gone (already exited, or not ours to signal).

    Never raises: stopping a runaway command is teardown, and teardown must not
    replace the result the caller is about to report.
    """
    try:
        pgid = os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError):
        return  # already reaped, or not ours — nothing left to stop
    if pgid != process.pid:
        logger.warning(
            "pid %s does not lead its own process group (pgid=%s); "
            "signalling the child alone. Spawn it with start_new_session=True "
            "so a timeout cannot orphan what it started.",
            process.pid,
            pgid,
        )
        _signal_child(process, sig)
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass  # every member exited between getpgid and killpg
    except OSError as exc:
        logger.warning("failed to signal process group %s with %s: %s", pgid, sig, exc)
        _signal_child(process, sig)


def _signal_child(process: subprocess.Popen, sig: int) -> None:
    try:
        process.send_signal(sig)
    except (ProcessLookupError, OSError) as exc:
        logger.debug("failed to signal child %s with %s: %s", process.pid, sig, exc)
