"""Domain-free text handling shared by every grader.

Nothing here interprets what an answer *means*: it scans for the marker a model
was asked to use, and strips presentation so two spellings of the same string
compare equal. Anything that needs to know the answer is a number, an
expression, or a set belongs in a domain package under this one.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "ANSWER_PREFIX_RE",
    "ANSWER_STATEMENT_RE",
    "after_thinking",
    "answer_block",
    "json_tail_answer",
    "scan_boxed",
    "strip_presentation",
    "tail",
]

BOXED_MARKER = "\\boxed{"
THINK_CLOSE_TAG = "</think>"
# The answer marker every prompt in this repository asks for. Scanning it is
# bracket matching, not mathematics, so it lives here; understanding what is
# inside the braces is the domain's job.
_EMPHASIS_EDGE_CHARS = "$*`'\" \t\r\n"
_TRAILING_PUNCT = ".,;:!?"
# "the final answer is 204", "answer = 204", "Answer: 204" -- as a leading
# wrapper on an already-extracted value.
ANSWER_PREFIX_RE = re.compile(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*", re.IGNORECASE)
# The same phrases used to *find* a value inside prose.
ANSWER_STATEMENT_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*([^\r\n]+)"
    r"|(?:最终答案|答案)\s*(?:是|=|:|：)\s*([^\r\n]+)",
    re.IGNORECASE,
)
# The block a prompt asks a model to put its conclusion in. Finding the block is
# tag scanning; reading what is inside it is the domain's job.
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
# A JSON object is a marker too: models copy one out of a sibling role's output
# format, and these are the keys they use for the value they stand behind.
_JSON_ANSWER_KEYS = ("final_answer", "answer")
_TRAILING_FENCE_RE = re.compile(r"\n?[`~]{3,}[A-Za-z0-9_+-]*[ \t]*$")
# How many candidate object starts a tail scan may try. Bounds the cost on a
# solution full of LaTeX braces; the object a text ends with is always within a
# few braces of its end.
_MAX_JSON_STARTS = 64


def scan_boxed(text: str | None) -> str | None:
    """Return the content of the last balanced ``\\boxed{...}``, or None.

    A brace-matching scan rather than a regex, so nested markup such as
    ``\\boxed{\\frac{408}{2}}`` comes back whole. An unclosed box returns None
    instead of a truncated fragment: a partial answer is worse than none, since
    it would be scored as a real one.
    """

    if not text:
        return None
    start = text.rfind(BOXED_MARKER)
    if start < 0:
        return None
    depth = 1
    chars: list[str] = []
    for char in text[start + len(BOXED_MARKER) :]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars)
        chars.append(char)
    return None


def strip_presentation(value: str) -> str:
    """Remove markdown emphasis, quoting, and trailing sentence punctuation.

    Punctuation is stripped only at the tail (``"yes."``) and never inside a
    value (``"0.5"``), and the two passes repeat until stable so ``**204.**``
    reduces the same way regardless of nesting order.
    """

    current = value.strip()
    while True:
        previous = current
        current = current.strip(_EMPHASIS_EDGE_CHARS)
        current = current.rstrip(_TRAILING_PUNCT)
        current = current.strip()
        if current == previous:
            return current


def after_thinking(text: str | None) -> str:
    """Return what the model said after it stopped thinking.

    A reasoning model wraps its scratch work in ``<think>...</think>`` and
    routinely writes candidate values there -- including boxed ones -- before
    rejecting them, so nothing inside the block is a stated answer. Only the text
    after the *last* closing tag counts. Text with no closing tag is returned
    unchanged: a generation cut off mid-thought never opened a conclusion to
    prefer, and refusing to look at it would lose answers that are really there.

    Finding a tag is scanning, not mathematics, so it belongs in this layer.
    """

    if not text:
        return ""
    end = text.lower().rfind(THINK_CLOSE_TAG)
    return text if end < 0 else text[end + len(THINK_CLOSE_TAG) :]


def answer_block(text: str | None) -> str | None:
    """Return the content of the last complete ``<answer>...</answer>``, or None.

    A model that used the block told us exactly where its answer is, so a caller
    should stop looking anywhere else. Only a closed block counts: an unclosed
    tag means the generation stopped inside it, and the enclosing text is then a
    better guess than a fragment.
    """

    if not text:
        return None
    matches = _ANSWER_BLOCK_RE.findall(text)
    return matches[-1] if matches else None


def json_tail_answer(text: str | None) -> str | None:
    """Return the value a trailing JSON object declares as its answer, or None.

    Prompts that hand a role a JSON output contract leak: in a multi-turn episode
    a sibling role's format enters the solver's context and the solver imitates
    it, ending its solution with ``{"reasoning": ..., "final_answer": 277}``
    instead of prose. Recognizing that key is marker scanning like any other, so
    it lives here; the value is returned verbatim for a domain to interpret.

    Deliberately strict, because a false positive is scored as a real answer:
    only a well-formed object that the text actually *ends* with, and only a
    scalar value. A number wrapped in a list or a nested object is a structure we
    were not promised, and prose that merely mentions a brace never parses.
    """

    obj = _json_object_tail(text)
    if obj is None:
        return None
    for key in _JSON_ANSWER_KEYS:
        if key not in obj:
            continue
        value = obj[key]
        if value is None or isinstance(value, bool):
            # A flag is a verdict, not an answer.
            continue
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _json_object_tail(text: str | None) -> dict[str, Any] | None:
    """Parse the JSON object a text ends with, ignoring a closing code fence."""

    if not text:
        return None
    trimmed = _TRAILING_FENCE_RE.sub("", text.rstrip()).rstrip()
    if not trimmed.endswith("}"):
        return None
    decoder = json.JSONDecoder()
    end = len(trimmed)
    starts = [index for index, char in enumerate(trimmed) if char == "{"]
    # Nearest-first: a nested object starts later but cannot reach ``end``, so the
    # first start that consumes the whole tail is the outermost one.
    for start in reversed(starts[-_MAX_JSON_STARTS:]):
        try:
            value, consumed = decoder.raw_decode(trimmed, start)
        except ValueError:
            continue
        if consumed == end and isinstance(value, dict):
            return value
    return None


def tail(text: str, characters: int) -> str:
    """Return the last ``characters`` of a solution.

    Fallback extraction is confined to the conclusion so a number mentioned
    mid-reasoning is never mistaken for the final result.
    """

    return text[-characters:] if characters > 0 else ""
