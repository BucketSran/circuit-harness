# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""What the local backend's resource caps do and do not bound.

Every claim here is measured against the running kernel rather than asserted
against the source: an rlimit that "is set" and an rlimit that *applies* are
different facts, and the difference is exactly what these tests exist to pin.
"""

from __future__ import annotations

import os
import pathlib
import resource
import subprocess
import sys

import pytest

from alphaapollo.common.execution.sandbox import local as local_backend
from alphaapollo.common.execution.sandbox.local import (
    DEFAULT_ADDRESS_SPACE_BYTES,
    LocalSubprocessBackend,
    SandboxError,
)
from alphaapollo.common.execution.sandbox.manager import SandboxManager


def _run_isolated(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return its combined output.

    Used where the check must alter process-wide state (lowering a hard rlimit is
    irreversible), so it cannot run in the pytest process. ``PYTHONPATH`` is
    pinned to the tree this test imported, so the probe cannot silently exercise
    a differently-installed copy of the package.
    """
    repo_root = pathlib.Path(local_backend.__file__).resolve().parents[4]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    return (completed.stdout + completed.stderr).strip()


# --------------------------------------------------------------------------
# Process count: not bounded, and the option is refused rather than ignored
# --------------------------------------------------------------------------
def test_sandboxed_code_can_fork_under_the_shipped_defaults() -> None:
    """The default sandbox must be able to run ordinary code that spawns a child.

    This replaces the signal the old ``RLIMIT_NPROC`` cap carried. That cap did
    not bound the sandbox — being per-uid, it counted the host account's own
    processes — and at the shipped default of 64 it blocked every ``fork`` here
    with ``BlockingIOError: [Errno 35]`` instead. Removing it is only correct if
    the default sandbox now runs such code, which is what this measures.
    """
    be = LocalSubprocessBackend(timeout=30)
    try:
        rec = be.exec(
            "import subprocess, sys\n"
            "out = subprocess.run([sys.executable, '-c', \"print('child ok')\"],\n"
            "                     capture_output=True, text=True)\n"
            "print(out.stdout.strip() or out.stderr.strip())\n"
        )
        assert rec.exit_code == 0, rec.stderr
        assert rec.stdout.strip() == "child ok"
    finally:
        be.release()


def test_max_processes_is_refused_and_names_where_the_cap_lives() -> None:
    """An option the backend cannot honour is refused, never quietly ignored.

    ``RLIMIT_NPROC`` is per real uid, so no value of ``max_processes`` means what
    the name says here. Accepting it and applying something else is the silent-
    configuration defect; the refusal points at the backend that does implement a
    per-sandbox process cap.
    """
    with pytest.raises(SandboxError) as excinfo:
        LocalSubprocessBackend(max_processes=64)
    message = str(excinfo.value)
    assert "max_processes=64" in message
    assert "per real uid" in message
    assert "podman" in message
    assert "--pids-limit" in message


def test_manager_refuses_an_explicitly_requested_process_cap() -> None:
    """The refusal has to survive the acquisition path callers actually use.

    ``SandboxManager.acquire("python", ...)`` forwards caller kwargs verbatim and
    never applies a profile, which is *why* the backend can treat any
    ``max_processes`` it sees as explicitly requested rather than inherited.
    """
    with pytest.raises(SandboxError):
        SandboxManager().acquire("python", max_processes=64)

    # The same acquisition without the option is unaffected.
    backend = SandboxManager().acquire("python")
    try:
        assert backend.exec("print('ok')").stdout.strip() == "ok"
    finally:
        backend.release()


def test_manager_refuses_an_explicitly_requested_absence_of_network() -> None:
    """``network=False`` is refused here, and the polarity is the point.

    The local backend runs the command as an ordinary host process, so it has the
    host's network at every setting of this option. That makes the *restrictive*
    request the impossible one — the reverse of how a refusal usually reads — and
    the one that must not be silently accepted, because a caller who asked for no
    network and was given the host's would never learn it from the result. The
    permissive value describes what this backend does, so it is honoured, and an
    omitted option requests nothing and is unaffected.
    """
    with pytest.raises(SandboxError) as excinfo:
        SandboxManager().acquire("python", network=False)
    message = str(excinfo.value)
    assert "network=False" in message
    assert "no network namespace" in message
    assert "restrictive value that is refused" in message  # the polarity, stated
    assert "podman" in message  # where the capability lives

    # network=True and an omitted network are the same acquisition, and both run.
    for kwargs in ({"network": True}, {}):
        backend = SandboxManager().acquire("python", **kwargs)
        try:
            assert backend.exec("print('ok')").stdout.strip() == "ok"
        finally:
            backend.release()


#: Assign a source address for a route to TEST-NET-1 without sending a packet: a
#: UDP ``connect`` only selects the route. The address it yields is the host's, so
#: running it inside and outside the sandbox compares the two network views.
_SOURCE_ADDRESS_PROBE = (
    "import socket\n"
    "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
    "s.connect(('192.0.2.1', 53))\n"  # TEST-NET-1: routable, answers nothing
    "print(s.getsockname()[0])\n"
    "s.close()\n"
)


def test_the_local_backend_really_does_have_the_host_network() -> None:
    """The refusal above rests on a measurement, not on an assumption.

    If this backend did bound the network, refusing ``network=False`` would be
    wrong. It does not: the sandbox resolves the same host source address this test
    process does, so the two share one network namespace and ``network=False``
    could only ever have been a claim. The host is the control, and the check skips
    where the host itself has no route to compare against.
    """
    try:
        expected = subprocess.run(  # noqa: S603 — fixed argv, our own probe
            [sys.executable, "-c", _SOURCE_ADDRESS_PROBE],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:  # pragma: no cover — host networking stalled
        pytest.skip("host source-address probe did not complete")
    if expected.returncode != 0:
        pytest.skip(f"host has no route to compare against: {expected.stderr.strip()}")

    backend = SandboxManager().acquire("python")
    try:
        record = backend.exec(_SOURCE_ADDRESS_PROBE)
    finally:
        backend.release()
    assert record.exit_code == 0, record.stderr
    assert record.stdout.strip() == expected.stdout.strip()


# --------------------------------------------------------------------------
# The caps that do apply, and the ones the host refuses
# --------------------------------------------------------------------------
def test_enforceable_caps_reach_the_sandboxed_child() -> None:
    """A cap the backend reports as applied must be observable inside the child."""
    be = LocalSubprocessBackend(timeout=30, cpu_seconds=17, max_open_files=57)
    try:
        rec = be.exec(
            "import resource\n"
            "print('CPU', resource.getrlimit(resource.RLIMIT_CPU))\n"
            "print('NOFILE', resource.getrlimit(resource.RLIMIT_NOFILE))\n"
            "print('NPROC', resource.getrlimit(resource.RLIMIT_NPROC))\n"
        )
        assert rec.exit_code == 0, rec.stderr
        assert "CPU (17, 17)" in rec.stdout
        assert "NOFILE (57, 57)" in rec.stdout
        # NPROC is deliberately untouched: the child inherits the host's limit.
        host_nproc = resource.getrlimit(resource.RLIMIT_NPROC)
        assert f"NPROC {host_nproc}" in rec.stdout
    finally:
        be.release()


def test_a_cap_above_the_host_hard_limit_is_refused_not_silently_dropped() -> None:
    """Asking for more than the host allows must fail loudly at construction.

    ``setrlimit`` rejects a soft limit above the hard limit, so such a cap simply
    would not apply — leaving the sandbox running uncapped while the caller
    believes otherwise. The check runs in a throwaway interpreter because
    lowering a hard limit cannot be undone.
    """
    probe = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))\n"
        "from alphaapollo.common.execution.sandbox.local import (\n"
        "    LocalSubprocessBackend, SandboxError,\n"
        ")\n"
        "try:\n"
        "    LocalSubprocessBackend(max_open_files=256)\n"
        "    print('ACCEPTED-UNCAPPED')\n"
        "except SandboxError as exc:\n"
        "    print('REFUSED', exc)\n"
        "be = LocalSubprocessBackend(max_open_files=64)\n"
        "be.release()\n"
        "print('WITHIN-LIMIT-OK')\n"
    )
    output = _run_isolated(probe)
    assert "REFUSED" in output, output
    assert "hard limit of 128" in output, output
    assert "WITHIN-LIMIT-OK" in output, output


def test_address_space_cap_is_applied_or_reported_never_silently_absent() -> None:
    """``RLIMIT_AS`` either reaches the child or is named as unavailable.

    Darwin aliases ``RLIMIT_AS`` to ``RLIMIT_RSS`` and refuses every useful value,
    so on macOS the sandbox genuinely has no memory bound and must say so rather
    than appear to have one. Elsewhere the cap must actually arrive in the child.
    """
    unenforceable = resource.RLIMIT_AS in local_backend._UNENFORCEABLE_LIMITS
    assert unenforceable == (sys.platform == "darwin")

    be = LocalSubprocessBackend(timeout=30)
    try:
        applied = dict(be._rlimits)
        rec = be.exec("import resource; print('AS', resource.getrlimit(resource.RLIMIT_AS))")
        assert rec.exit_code == 0, rec.stderr
        if unenforceable:
            assert resource.RLIMIT_AS not in applied
            assert f"AS {resource.getrlimit(resource.RLIMIT_AS)}" in rec.stdout
        else:
            want = DEFAULT_ADDRESS_SPACE_BYTES
            assert applied[resource.RLIMIT_AS] == want
            assert f"AS ({want}, {want})" in rec.stdout
    finally:
        be.release()


def test_unenforceable_cap_is_announced_once_per_process(caplog) -> None:
    """A dropped cap must leave a trace; silence is the failure mode."""
    if not local_backend._UNENFORCEABLE_LIMITS:
        pytest.skip("every cap is enforceable on this platform")
    local_backend._warned_unenforceable.clear()
    try:
        with caplog.at_level("WARNING", logger=local_backend.logger.name):
            LocalSubprocessBackend().release()
            LocalSubprocessBackend().release()
        warnings = [r for r in caplog.records if "not enforceable" in r.getMessage()]
        assert len(warnings) == 1, "expected exactly one warning per process"
        assert "address_space_bytes" in warnings[0].getMessage()
    finally:
        local_backend._warned_unenforceable.clear()


#: Address-space values the probe below tries: below, around and above the
#: memory an ordinary sandbox would want. All three are values a caller could
#: plausibly pass as ``address_space_bytes``.
_ADDRESS_SPACE_PROBE_VALUES = (64 * 1024**2, 2 * 1024**3, 16 * 1024**3)


def _probe_address_space(value: int) -> str:
    """Return a pristine interpreter's verdict on ``setrlimit(RLIMIT_AS, value)``.

    Each value needs its *own* process. ``setrlimit`` sets the hard limit as well
    as the soft one and lowering a hard limit is irreversible, so a loop over the
    three values in a single interpreter stops measuring the kernel after the
    first acceptance: on Linux, 64 MiB is accepted and thereby pins the hard limit
    at 64 MiB, and the 2 GiB and 16 GiB attempts are then refused with
    ``ValueError: not allowed to raise maximum limit`` — the residue of the first
    probe, not the platform's answer for those values. Only a fresh process starts
    from the host's real hard limit.
    """
    return _run_isolated(
        "import resource\n"
        f"value = {value}\n"
        "try:\n"
        "    resource.setrlimit(resource.RLIMIT_AS, (value, value))\n"
        "    print(value, 'OK')\n"
        "except (ValueError, OSError) as exc:\n"
        "    print(value, 'REFUSED', type(exc).__name__)\n"
    )


def test_darwin_refuses_every_useful_address_space_value() -> None:
    """The measurement behind ``_UNENFORCEABLE_LIMITS``, taken from the kernel.

    If a future macOS starts accepting ordinary ``RLIMIT_AS`` values, this fails
    and the platform entry should be removed rather than carried forward on
    reputation.

    The expectation is derived from the host's own hard limit rather than assumed:
    a value at or below it must be accepted, one above it must be refused. That is
    what makes the Darwin result a finding — macOS reports ``hard=RLIM_INFINITY``
    and refuses these values anyway, so the refusal cannot be explained by a hard
    limit and is a property of the platform, exactly as ``local.py`` claims.
    """
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    results = {value: _probe_address_space(value) for value in _ADDRESS_SPACE_PROBE_VALUES}
    report = "\n".join(results.values())

    if sys.platform == "darwin":
        # Unchanged in strictness: every value refused. The hard limit is asserted
        # too, so the refusals can never be excused by the host having a low one.
        assert hard == resource.RLIM_INFINITY, report
        assert report.count("REFUSED") == 3, report
    else:
        for value, output in results.items():
            within_hard_limit = hard == resource.RLIM_INFINITY or value <= hard
            expected = "OK" if within_hard_limit else "REFUSED"
            assert f"{value} {expected}" in output, (
                f"RLIMIT_AS hard limit is {hard}; expected {value} to be {expected}\n{report}"
            )
