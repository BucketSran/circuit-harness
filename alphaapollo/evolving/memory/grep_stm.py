"""Deterministic lexical short-term memory over transcript, files, and entries.

This is the exact-match counterpart to semantic ``WorkingMemory``.  It performs
no network or model calls and returns stable locators suitable for replay.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .interface import MemoryEntry, MemoryKind

try:  # Python 3.11+ moved the old sre modules under ``re``.
    from re import _constants as _sre_constants
    from re import _parser as _sre_parser
except ImportError:  # pragma: no cover - supported Python 3.10
    import sre_constants as _sre_constants
    import sre_parse as _sre_parser

_OUTPUT_MODES = frozenset({"content", "locators_with_matches", "count"})
_DEFAULT_MAX_BYTES = 1_000_000
_MAX_PATTERN_CHARS = 512
_MAX_EXPLICIT_REPEAT = 10_000
_DEFAULT_IGNORE_DIRS = frozenset(
    {".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache", "node_modules"}
)


class GrepError(ValueError):
    """Raised for an invalid or intentionally bounded grep request."""


@dataclass(frozen=True, slots=True)
class GrepRecord:
    source: str
    locator: str
    text: str


@dataclass(frozen=True, slots=True)
class GrepHit:
    source: str
    locator: str
    line_no: int | None
    line: str
    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GrepResult:
    output_mode: str
    total: int
    truncated: bool
    hits: tuple[GrepHit, ...] = ()
    locators: tuple[str, ...] = ()
    counts: tuple[tuple[str, int], ...] = ()


@runtime_checkable
class GrepSource(Protocol):
    name: str

    def records(self) -> Iterable[GrepRecord]: ...


class TranscriptSource:
    """Expose one agent transcript with stable turn/role locators."""

    name = "transcript"

    def __init__(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        roles: Iterable[str] | None = None,
    ) -> None:
        self._messages = tuple(messages)
        self._roles = None if roles is None else frozenset(str(role) for role in roles)

    def records(self) -> Iterator[GrepRecord]:
        for index, message in enumerate(self._messages):
            role = str(message.get("role", ""))
            if self._roles is not None and role not in self._roles:
                continue
            yield GrepRecord(
                source=self.name,
                locator=f"turn:{index}:{role}",
                text=str(message.get("content", "")),
            )


class MemorySource:
    """Expose already scope-filtered memory entries to lexical retrieval."""

    name = "memory"

    def __init__(
        self,
        entries: Sequence[MemoryEntry],
        *,
        kinds: Iterable[MemoryKind | str] | None = None,
    ) -> None:
        self._entries = tuple(entries)
        self._kinds = None if kinds is None else frozenset(MemoryKind(kind) for kind in kinds)

    def records(self) -> Iterator[GrepRecord]:
        for index, entry in enumerate(self._entries):
            if self._kinds is not None and entry.kind not in self._kinds:
                continue
            yield GrepRecord(
                source=self.name,
                locator=f"entry:{index}:{entry.kind.value}",
                text=entry.content,
            )


class FileSource:
    """Expose bounded UTF-8 files beneath one root in deterministic order."""

    name = "files"

    def __init__(
        self,
        root: Path | str,
        *,
        glob: str | None = None,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        ignore_dirs: Iterable[str] = _DEFAULT_IGNORE_DIRS,
    ) -> None:
        self._root = Path(root).resolve()
        self._glob = glob
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise GrepError("max_bytes must be a positive integer")
        self._max_bytes = max_bytes
        self._ignore_dirs = frozenset(ignore_dirs)

    def records(self) -> Iterator[GrepRecord]:
        if not self._root.exists():
            return
        for path in sorted(self._root.rglob("*")):
            relative = path.relative_to(self._root)
            if any(part in self._ignore_dirs for part in relative.parts):
                continue
            if self._glob is not None and not fnmatch(relative.as_posix(), self._glob):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            try:
                if path.stat().st_size > self._max_bytes:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in data:
                continue
            yield GrepRecord(
                source=self.name,
                locator=relative.as_posix(),
                text=data.decode("utf-8", errors="replace"),
            )


def grep(
    pattern: str,
    sources: Iterable[GrepSource],
    *,
    fixed_string: bool = False,
    ignore_case: bool = False,
    output_mode: str = "content",
    before: int = 0,
    after: int = 0,
    context: int | None = None,
    multiline: bool = False,
    invert: bool = False,
    head_limit: int | None = None,
) -> GrepResult:
    """Search named corpora with deterministic ordering and bounded regexes."""

    if not isinstance(pattern, str) or not pattern:
        raise GrepError("pattern must be a non-empty string")
    if len(pattern) > _MAX_PATTERN_CHARS:
        raise GrepError(f"pattern too long (max {_MAX_PATTERN_CHARS} chars)")
    if output_mode not in _OUTPUT_MODES:
        raise GrepError(f"output_mode must be one of {sorted(_OUTPUT_MODES)}")
    for name, value in (("before", before), ("after", after)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GrepError(f"{name} must be a non-negative integer")
    if context is not None:
        if isinstance(context, bool) or not isinstance(context, int) or context < 0:
            raise GrepError("context must be a non-negative integer")
        before = after = context
    if head_limit is not None and (
        isinstance(head_limit, bool) or not isinstance(head_limit, int) or head_limit < 1
    ):
        raise GrepError("head_limit must be a positive integer or None")
    if multiline and invert:
        raise GrepError("invert is not supported with multiline")

    expression = re.escape(pattern) if fixed_string else pattern
    if not fixed_string:
        _validate_safe_regex(expression, flags=0)
    flags = re.IGNORECASE if ignore_case else 0
    if multiline:
        flags |= re.MULTILINE | re.DOTALL
    try:
        regex = re.compile(expression, flags)
    except re.error as exc:
        raise GrepError(f"invalid regular expression: {exc}") from exc

    matches = tuple(
        _iter_matches(
            regex,
            sources,
            before=before,
            after=after,
            multiline=multiline,
            invert=invert,
        )
    )
    total = len(matches)
    limited = matches if head_limit is None else matches[:head_limit]
    truncated = len(limited) < total
    if output_mode == "content":
        return GrepResult(output_mode, total, truncated, hits=limited)
    locators = tuple(dict.fromkeys(hit.locator for hit in limited))
    if output_mode == "locators_with_matches":
        return GrepResult(output_mode, total, truncated, locators=locators)
    counts: list[tuple[str, int]] = []
    for locator in locators:
        counts.append((locator, sum(hit.locator == locator for hit in limited)))
    return GrepResult(output_mode, total, truncated, counts=tuple(counts))


def _validate_safe_regex(pattern: str, *, flags: int) -> None:
    """Restrict model-authored regexes to a bounded, linear-ish subset."""

    try:
        parsed = _sre_parser.parse(pattern, flags)
    except re.error as exc:
        raise GrepError(f"invalid regular expression: {exc}") from exc
    _inspect_regex_sequence(parsed, inside_variable_repeat=False)


def _inspect_regex_sequence(sequence: Any, *, inside_variable_repeat: bool) -> int:
    repeat_ops = {_sre_constants.MAX_REPEAT, _sre_constants.MIN_REPEAT}
    if (possessive := getattr(_sre_constants, "POSSESSIVE_REPEAT", None)) is not None:
        repeat_ops.add(possessive)
    forbidden_ops = {
        _sre_constants.ASSERT,
        _sre_constants.ASSERT_NOT,
        _sre_constants.GROUPREF,
        _sre_constants.GROUPREF_EXISTS,
    }
    variable_repeats = 0
    for op, arg in sequence:
        if op in forbidden_ops:
            raise GrepError(
                "unsafe regular expression: lookarounds and backreferences are not allowed"
            )
        if op in repeat_ops:
            minimum, maximum, child = arg
            if maximum != _sre_constants.MAXREPEAT and maximum > _MAX_EXPLICIT_REPEAT:
                raise GrepError(
                    "unsafe regular expression: explicit repeat limit exceeds "
                    f"{_MAX_EXPLICIT_REPEAT}"
                )
            is_variable = minimum != maximum
            if is_variable and inside_variable_repeat:
                raise GrepError(
                    "unsafe regular expression: nested variable repeats are not allowed"
                )
            child_repeats = _inspect_regex_sequence(
                child,
                inside_variable_repeat=inside_variable_repeat or is_variable,
            )
            variable_repeats += child_repeats + int(is_variable)
            continue
        if op is _sre_constants.SUBPATTERN:
            variable_repeats += _inspect_regex_sequence(
                arg[-1],
                inside_variable_repeat=inside_variable_repeat,
            )
            continue
        if op is _sre_constants.BRANCH:
            if inside_variable_repeat:
                raise GrepError(
                    "unsafe regular expression: alternation inside a variable repeat is not allowed"
                )
            variable_repeats += max(
                (
                    _inspect_regex_sequence(
                        branch,
                        inside_variable_repeat=inside_variable_repeat,
                    )
                    for branch in arg[1]
                ),
                default=0,
            )
            continue
        atomic_group = getattr(_sre_constants, "ATOMIC_GROUP", None)
        if atomic_group is not None and op is atomic_group:
            _inspect_regex_sequence(arg, inside_variable_repeat=False)
    if variable_repeats > 1 and not inside_variable_repeat:
        raise GrepError(
            "unsafe regular expression: multiple variable repeats in one sequence are not allowed"
        )
    return variable_repeats


def _iter_matches(
    regex: re.Pattern[str],
    sources: Iterable[GrepSource],
    *,
    before: int,
    after: int,
    multiline: bool,
    invert: bool,
) -> Iterator[GrepHit]:
    for source in sources:
        if not isinstance(source, GrepSource):
            raise TypeError("sources must implement GrepSource")
        for record in source.records():
            if multiline:
                for found in regex.finditer(record.text):
                    line_no = record.text.count("\n", 0, found.start()) + 1
                    yield GrepHit(source.name, record.locator, line_no, found.group(0))
                continue
            lines = record.text.splitlines()
            for index, line in enumerate(lines):
                found = regex.search(line) is not None
                if found == invert:
                    continue
                yield GrepHit(
                    source.name,
                    record.locator,
                    index + 1,
                    line,
                    tuple(lines[max(0, index - before) : index]),
                    tuple(lines[index + 1 : index + 1 + after]),
                )


__all__ = [
    "FileSource",
    "GrepError",
    "GrepHit",
    "GrepRecord",
    "GrepResult",
    "GrepSource",
    "MemorySource",
    "TranscriptSource",
    "grep",
]
