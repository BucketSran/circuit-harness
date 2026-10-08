"""Task-agnostic extraction of a deliberately declared answer, for voting.

This is the consensus layer's key function: independent reasoning traces that
explicitly declare the same answer must vote together. It recognizes common
answer markers in English and Chinese and compares with exact rational
semantics, so ``204``, ``204.0`` and ``\\frac{408}{2}`` collapse to one vote.

Only explicit declarations are grouped. Presentation wrappers are stripped,
while units and expressions stay textual; numeric declarations compare as exact
rationals, without a symbolic solver or benchmark-specific grading policy.
"""

from __future__ import annotations

import re
from fractions import Fraction

from alphaapollo.common.grader.text import (
    ANSWER_STATEMENT_RE,
    after_thinking,
    answer_block,
    scan_boxed,
    strip_presentation,
)

__all__ = [
    "declared_answer_key",
    "declared_answers_equivalent",
    "extract_declared_answer",
]


def extract_declared_answer(text: str) -> str | None:
    """Extract a deliberately declared answer without importing an eval policy.

    Resolves *where* to look before *what* to look for, and widens only when the
    narrower region declares nothing: the last closed ``<answer>...</answer>``
    after the last ``</think>``, then everything after that ``</think>``, then
    the whole text. Each step is less deliberate than the one before -- the block
    is where the model said its answer is, the text after ``</think>`` is what it
    chose to say out loud, and the rest is scratch work it may have rejected.

    Widening rather than abstaining is what separates this from scoring: a vote
    needs a key more than it needs a scoreable answer, and a branch that declared
    nothing usable in its conclusion still belongs in a group. The whole-text
    scan is therefore kept as the last resort. When neither tag appears the first
    usable span already *is* the whole text, so behaviour is unchanged.

    Recognizes common, task-agnostic answer markers. Dataset-specific scoring
    stays in the graders; a consumer only needs a stable key so traces that
    declare the same answer group together.
    """

    visible = after_thinking(text)
    for span in (answer_block(visible), visible, text):
        if not span:
            continue
        declared = _scan_declaration(span)
        if declared is not None:
            return declared
    return None


def _scan_declaration(span: str) -> str | None:
    """Return the answer the span declares, most deliberate marker first."""

    boxed = scan_boxed(span)
    if boxed is not None:
        return _usable_declared_answer(boxed)
    matches = tuple(ANSWER_STATEMENT_RE.finditer(span))
    if not matches:
        return None
    match = matches[-1]
    answer = match.group(1) or match.group(2)
    return _usable_declared_answer(answer)


def declared_answers_equivalent(left: str | None, right: str | None) -> bool:
    """Compare two explicit answer values with the reducer's exact semantics."""

    if left is None or right is None:
        return False
    usable_left = _usable_declared_answer(left)
    usable_right = _usable_declared_answer(right)
    return (
        usable_left is not None
        and usable_right is not None
        and _normalize_answer(usable_left) == _normalize_answer(usable_right)
    )


def declared_answer_key(value: str) -> tuple[str, object]:
    """Return the grouping key two equivalent declarations must share.

    Exposed because voting groups branches by this key rather than by calling
    ``declared_answers_equivalent`` pairwise; both must agree, so they are one
    implementation.
    """

    return _normalize_answer(value)


def _usable_declared_answer(value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    kind, normalized = _normalize_answer(stripped)
    if kind == "text" and not normalized:
        return None
    return stripped


def _normalize_answer(value: str) -> tuple[str, object]:
    """Reduce a declaration to a comparable key, tagged by how it compares.

    Numbers are compared as exact rationals so ``204``, ``204.0`` and
    ``\\frac{408}{2}`` group together; everything else is compared as
    case-folded text. The tag keeps the two apart, so the string "204" and a
    genuinely textual answer never collide.
    """

    normalized = strip_presentation(value).casefold()
    numeric = _parse_number(normalized)
    return ("text", normalized) if numeric is None else ("number", numeric)


def _parse_number(value: str) -> Fraction | None:
    """Read a plain rational declaration; preserve units as text."""
    text = value.replace(",", "").strip()
    while len(text) >= 2 and (text[0], text[-1]) in {("(", ")"), ("[", "]"), ("{", "}")}:
        text = text[1:-1].strip()
    fraction = re.fullmatch(r"\\[dt]?frac\{([+-]?[\d.]+)\}\{([+-]?[\d.]+)\}", text)
    try:
        if fraction is not None:
            return Fraction(fraction[1]) / Fraction(fraction[2])
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
