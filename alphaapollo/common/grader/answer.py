"""Declared final-answer extraction shared by verifier and Environment clients."""

from __future__ import annotations

from alphaapollo.common.grader.declared import extract_declared_answer
from alphaapollo.common.grader.text import (
    after_thinking,
    answer_block,
    json_tail_answer,
    scan_boxed,
    strip_presentation,
)


def extract_boxed_answer(text: str | None) -> str | None:
    boxed = scan_boxed(text)
    return None if boxed is None else strip_presentation(boxed)


def extract_final_answer(text: str | None) -> str | None:
    """Read the conclusion, excluding rejected candidates in thinking blocks."""
    if not text:
        return None
    visible = after_thinking(text)
    block = answer_block(visible)
    span = visible if block is None else block
    boxed = extract_boxed_answer(span)
    if boxed is not None:
        return boxed
    keyed = json_tail_answer(span)
    if keyed is not None:
        return extract_boxed_answer(keyed) or strip_presentation(keyed) or None
    return extract_declared_answer(span)
