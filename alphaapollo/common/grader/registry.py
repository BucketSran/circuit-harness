"""Resolve a ``grader_id`` to its implementation.

``grader_id`` is carried in the private projection of every prepared task, so a
dataset declares how it wants to be scored and consumers dispatch on that value
instead of hard-coding one benchmark's rules.

Application-owned graders are registered as factories and initialized lazily,
so optional dependencies are loaded only when the selected grader needs them.
"""

from __future__ import annotations

from collections.abc import Callable

from alphaapollo.common.grader.base import Grader, Verdict
from alphaapollo.common.grader.environment import EnvironmentSuccessGrader
from alphaapollo.common.grader.exact import ExactMatchGrader

__all__ = [
    "available_graders",
    "get_grader",
    "grade",
    "register_grader",
    "require_offline_grader",
]


_FACTORIES: dict[str, Callable[[], Grader]] = {
    ExactMatchGrader.grader_id: ExactMatchGrader,
    EnvironmentSuccessGrader.grader_id: EnvironmentSuccessGrader,
}
_INSTANCES: dict[str, Grader] = {}


def register_grader(
    grader_id: str,
    factory: Callable[[], Grader],
    *,
    replace: bool = False,
) -> None:
    """Register a grader factory, refusing a silent override by default."""

    if not grader_id.strip():
        raise ValueError("grader_id must be a non-empty string")
    if not replace and grader_id in _FACTORIES:
        raise ValueError(f"grader {grader_id!r} is already registered")
    _FACTORIES[grader_id] = factory
    _INSTANCES.pop(grader_id, None)


def get_grader(grader_id: str) -> Grader:
    """Return the registered grader, or raise naming what is available."""

    cached = _INSTANCES.get(grader_id)
    if cached is not None:
        return cached
    try:
        factory = _FACTORIES[grader_id]
    except KeyError:
        raise ValueError(
            f"unknown grader_id {grader_id!r}; available: {sorted(_FACTORIES)}"
        ) from None
    grader = factory()
    _INSTANCES[grader_id] = grader
    return grader


def require_offline_grader(grader_id: str) -> Grader:
    """Resolve a grader for offline scoring, refusing environment-graded ids.

    ``environment_success`` is registered so a prepared robotics dataset
    resolves everywhere it is read, but its verdict comes from the execution
    backend's success predicate at rollout time. Configuration-time callers
    resolve through here so a matrix cannot run for an hour and only then
    report every cell as unscoreable.

    ``offline_scoreable`` is an opt-out: a grader that does not declare it is
    an ordinary text grader, which keeps ``register_grader`` open to callers
    outside this package.
    """

    grader = get_grader(grader_id)
    if not getattr(grader, "offline_scoreable", True):
        raise ValueError(
            f"grader {grader_id!r} is decided by the execution backend's success "
            "predicate at rollout time, not by the offline scorer; route these "
            "rows through the robotics environment instead of scoring them"
        )
    return grader


def available_graders() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def grade(candidate: str | None, gold: str | None, *, grader_id: str) -> Verdict:
    """Score one candidate against gold using the dataset's declared grader."""

    return get_grader(grader_id).grade(candidate, gold)
