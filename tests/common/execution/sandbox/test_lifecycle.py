"""GPU/container-free tests for SandboxLifecycleManager (pure lifecycle state machine).

A FAKE backend factory records every build and release, so these tests assert the
*decision logic* (when a box is built / reused / forked / torn down) with no Podman
or Docker present.

Covers:
  - PER_PROBLEM reuses one box across branches/calls;
  - PER_BRANCH builds one box per branch and isolates them;
  - PER_CALL builds a fresh box every acquire;
  - fork derives a distinct child from a live parent (PER_BRANCH only);
  - release / release_all tear down correctly and are idempotent;
  - fail-loud on invalid operations.
"""

from __future__ import annotations

import pytest

from alphaapollo.common.execution.sandbox.base import (
    PYTHON_DEFAULT,
    VERIFIER_DEFAULT,
    SandboxLifecycle,
    SandboxLifecycleError,
    SandboxLifecycleManager,
)


class _FakeBackend:
    """A backend double that records whether it was released."""

    def __init__(self, index: int, profile) -> None:
        self.index = index
        self.profile = profile
        self.released = False

    def release(self) -> None:
        self.released = True


class _FakeFactory:
    """Records every build; hands out uniquely-indexed fake backends."""

    def __init__(self) -> None:
        self.builds: list = []

    def __call__(self, profile):
        backend = _FakeBackend(index=len(self.builds), profile=profile)
        self.builds.append(backend)
        return backend

    @property
    def build_count(self) -> int:
        return len(self.builds)

    @property
    def released_count(self) -> int:
        return sum(1 for b in self.builds if b.released)


def _mgr(lifecycle: SandboxLifecycle) -> tuple[SandboxLifecycleManager, _FakeFactory]:
    factory = _FakeFactory()
    return SandboxLifecycleManager(backend_factory=factory, lifecycle=lifecycle), factory


# --- PER_PROBLEM -------------------------------------------------------------
def test_per_problem_reuses_one_box_across_branches_and_calls() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_PROBLEM)
    a = mgr.acquire_for(problem_id="p1", branch_id="b0")
    b = mgr.acquire_for(problem_id="p1", branch_id="b1")
    c = mgr.acquire_for(problem_id="p1")
    assert a is b is c  # same box reused
    assert factory.build_count == 1
    assert mgr.live_count == 1


def test_per_problem_distinct_problems_get_distinct_boxes() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_PROBLEM)
    mgr.acquire_for(problem_id="p1")
    mgr.acquire_for(problem_id="p2")
    assert factory.build_count == 2
    assert mgr.live_count == 2


# --- PER_BRANCH --------------------------------------------------------------
def test_per_branch_builds_one_box_per_branch() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_BRANCH)
    b0a = mgr.acquire_for(problem_id="p1", branch_id="b0")
    b0b = mgr.acquire_for(problem_id="p1", branch_id="b0")  # reuse same branch
    b1 = mgr.acquire_for(problem_id="p1", branch_id="b1")
    assert b0a is b0b  # same branch reuses
    assert b0a is not b1  # different branch isolated
    assert factory.build_count == 2
    assert mgr.live_keys() == ["branch:p1:b0", "branch:p1:b1"]


def test_per_branch_requires_branch_id() -> None:
    mgr, _ = _mgr(SandboxLifecycle.PER_BRANCH)
    with pytest.raises(SandboxLifecycleError, match="branch_id"):
        mgr.acquire_for(problem_id="p1")


# --- PER_CALL ----------------------------------------------------------------
def test_per_call_builds_fresh_box_every_acquire() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_CALL)
    a = mgr.acquire_for(problem_id="p1", branch_id="b0")
    b = mgr.acquire_for(problem_id="p1", branch_id="b0")  # same args, still fresh
    assert a is not b
    assert factory.build_count == 2
    assert mgr.live_count == 2


def test_per_call_release_releases_matching_calls() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_CALL)
    mgr.acquire_for(problem_id="p1", branch_id="b0")
    mgr.acquire_for(problem_id="p1", branch_id="b0")
    mgr.acquire_for(problem_id="p1", branch_id="b1")
    mgr.release(problem_id="p1", branch_id="b0")
    assert mgr.live_count == 1
    assert factory.released_count == 2


# --- profile selection -------------------------------------------------------
def test_acquire_passes_chosen_profile_to_factory() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_PROBLEM)
    mgr.acquire_for(problem_id="p1", profile=VERIFIER_DEFAULT)
    assert factory.builds[0].profile is VERIFIER_DEFAULT


def test_acquire_defaults_to_default_profile() -> None:
    factory = _FakeFactory()
    mgr = SandboxLifecycleManager(
        backend_factory=factory,
        lifecycle=SandboxLifecycle.PER_PROBLEM,
        default_profile=PYTHON_DEFAULT,
    )
    mgr.acquire_for(problem_id="p1")
    assert factory.builds[0].profile is PYTHON_DEFAULT


def test_reuse_rejects_profile_mismatch() -> None:
    mgr, _ = _mgr(SandboxLifecycle.PER_PROBLEM)
    mgr.acquire_for(problem_id="p1", profile=PYTHON_DEFAULT)
    with pytest.raises(SandboxLifecycleError, match="profile mismatch"):
        mgr.acquire_for(problem_id="p1", profile=VERIFIER_DEFAULT)


def test_scope_ids_cannot_collide_when_they_contain_colons() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_BRANCH)
    first = mgr.acquire_for(problem_id="a:b", branch_id="c")
    second = mgr.acquire_for(problem_id="a", branch_id="b:c")
    assert first is not second
    assert factory.build_count == 2


# --- fork --------------------------------------------------------------------
def test_fork_derives_distinct_child_from_live_parent() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_BRANCH)
    parent = mgr.acquire_for(problem_id="p1", branch_id="b0")
    child_id, child = mgr.fork(problem_id="p1", parent_branch_id="b0")
    assert child is not parent
    assert child_id.startswith("b0.fork-")
    assert factory.build_count == 2
    assert f"branch:p1:{child_id}" in mgr.live_keys()


def test_fork_child_inherits_parent_profile() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_BRANCH)
    mgr.acquire_for(problem_id="p1", branch_id="b0", profile=VERIFIER_DEFAULT)
    _cid, child = mgr.fork(problem_id="p1", parent_branch_id="b0")
    assert child.profile is VERIFIER_DEFAULT


def test_fork_requires_live_parent() -> None:
    mgr, _ = _mgr(SandboxLifecycle.PER_BRANCH)
    with pytest.raises(SandboxLifecycleError, match="no live parent"):
        mgr.fork(problem_id="p1", parent_branch_id="ghost")


def test_fork_only_under_per_branch() -> None:
    mgr, _ = _mgr(SandboxLifecycle.PER_PROBLEM)
    mgr.acquire_for(problem_id="p1")
    with pytest.raises(SandboxLifecycleError, match="only supported under PER_BRANCH"):
        mgr.fork(problem_id="p1", parent_branch_id="b0")


# --- release -----------------------------------------------------------------
def test_per_problem_release_tears_down_box() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_PROBLEM)
    mgr.acquire_for(problem_id="p1")
    mgr.release(problem_id="p1")
    assert factory.released_count == 1
    assert mgr.live_count == 0


def test_release_is_idempotent() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_PROBLEM)
    mgr.acquire_for(problem_id="p1")
    mgr.release(problem_id="p1")
    mgr.release(problem_id="p1")  # second release is a no-op, no raise
    assert factory.released_count == 1


def test_per_branch_release_targets_one_branch() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_BRANCH)
    mgr.acquire_for(problem_id="p1", branch_id="b0")
    mgr.acquire_for(problem_id="p1", branch_id="b1")
    mgr.release(problem_id="p1", branch_id="b0")
    assert mgr.live_keys() == ["branch:p1:b1"]
    assert factory.released_count == 1


def test_release_all_tears_down_everything() -> None:
    mgr, factory = _mgr(SandboxLifecycle.PER_CALL)
    mgr.acquire_for(problem_id="p1", branch_id="b0")
    mgr.acquire_for(problem_id="p1", branch_id="b0")
    mgr.acquire_for(problem_id="p2", branch_id="b0")
    assert mgr.live_count == 3
    mgr.release_all()
    assert mgr.live_count == 0
    assert factory.released_count == 3


def test_release_all_survives_a_backend_that_raises_on_release() -> None:
    class _BadBackend:
        def release(self) -> None:
            raise RuntimeError("boom")

    def factory(_profile):
        return _BadBackend()

    mgr = SandboxLifecycleManager(backend_factory=factory, lifecycle=SandboxLifecycle.PER_PROBLEM)
    mgr.acquire_for(problem_id="p1")
    mgr.release_all()  # best-effort teardown must not raise
    # Failed cleanup remains tracked so a later release can retry it.
    assert mgr.live_count == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
