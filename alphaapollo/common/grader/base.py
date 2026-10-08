"""The grading contract: one verdict type and the shape every grader implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Grader", "Verdict"]

# ``None`` is a third state, not a missing bool: it means the candidate could not
# be scored at all (no answer to compare, or the grader gave up). Collapsing it
# into ``False`` would report an unscoreable run as a wrong one and quietly
# understate a model, so every layer keeps the distinction end to end.
Verdict = bool | None


@runtime_checkable
class Grader(Protocol):
    """Score a candidate answer for one task family.

    A grader owns both halves of its domain: pulling the stated answer out of a
    full solution, and deciding whether it matches the gold. Callers dispatch on
    the ``grader_id`` a prepared task declares and never encode domain rules
    themselves.

    Implementations are pure and deterministic: the same input must always yield
    the same result, because scores are persisted and compared across runs.

    A grader whose verdict is not decided by comparing text at all -- the
    execution backend's success predicate decides it -- declares the optional
    ``offline_scoreable = False`` opt-out, which
    ``registry.require_offline_grader`` refuses at configuration time. The flag
    stays off this Protocol on purpose: it is an opt-out, so an ordinary grader
    (including one a caller registers) remains valid without it.
    """

    grader_id: str

    def extract(self, text: str | None) -> str | None:
        """Return the answer the solution puts forward, or None."""
        ...

    def grade(self, candidate: str | None, gold: str | None) -> Verdict:
        """Return True/False, or None when the pair cannot be scored."""
        ...
