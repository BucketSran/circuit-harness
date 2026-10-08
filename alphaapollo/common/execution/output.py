# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Backend-neutral bounded output capture and rendering primitives."""

from __future__ import annotations

import codecs
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# The host-side caps every bounded tool result is rendered through. The
# workspace worker restates them as its own ``MAX_LINES``/``MAX_BYTES`` because
# it runs inside the sandbox and cannot import this module; it then reserves
# room below them for its truncation notice, so a head-oriented workspace result
# is never tail-truncated here. Changing either value alone breaks that
# relation, so ``test_worker_output_caps_match_the_host_side_accumulator`` pins
# the two together.
DEFAULT_MAX_LINES = 2_000
DEFAULT_MAX_BYTES = 50 * 1024


@dataclass(frozen=True, slots=True)
class TruncationResult:
    """Describe a bounded tool-output view and the original output size."""

    content: str
    truncated: bool
    truncated_by: Literal["lines", "bytes"] | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    max_lines: int
    max_bytes: int


def _split_lines_for_counting(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _truncate_utf8_from_end(value: str, max_bytes: int) -> str:
    raw = value.encode("utf-8")
    if len(raw) <= max_bytes:
        return value
    start = len(raw) - max_bytes
    while start < len(raw) and raw[start] & 0xC0 == 0x80:
        start += 1
    return raw[start:].decode("utf-8")


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the last complete lines/bytes of output, matching Pi's bash behavior."""
    if not isinstance(content, str):
        raise TypeError("truncate_tail expects text")
    if not isinstance(max_lines, int) or isinstance(max_lines, bool) or max_lines <= 0:
        raise ValueError("max_lines must be a positive integer")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    total_bytes = len(content.encode("utf-8"))
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    last_line_partial = False
    for line in reversed(lines):
        line_bytes = len(line.encode("utf-8")) + (1 if output_lines else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines:
                partial = _truncate_utf8_from_end(line, max_bytes)
                output_lines.insert(0, partial)
                output_bytes = len(partial.encode("utf-8"))
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes += line_bytes
        if len(output_lines) >= max_lines:
            truncated_by = "lines"
            break

    output = "\n".join(output_lines)
    return TruncationResult(
        content=output,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=len(output.encode("utf-8")),
        last_line_partial=last_line_partial,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


@dataclass(frozen=True, slots=True)
class OutputSnapshot:
    content: str
    truncation: TruncationResult
    capture_path: Path


class OutputAccumulator:
    """Spool full bytes to disk while retaining a bounded decoded tail in memory."""

    def __init__(
        self,
        *,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        temp_file_prefix: str = "apollo-output-",
    ) -> None:
        if max_lines <= 0 or max_bytes <= 0:
            raise ValueError("output accumulator limits must be positive")
        self._max_lines = max_lines
        self._max_bytes = max_bytes
        self._max_rolling_bytes = max(max_bytes * 2, 1)
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        descriptor, path = tempfile.mkstemp(prefix=temp_file_prefix, suffix=".log")
        self._capture_path = Path(path)
        self._capture = os.fdopen(descriptor, "wb")
        self._tail_text = ""
        self._tail_bytes = 0
        self._tail_starts_at_line_boundary = True
        self._total_decoded_bytes = 0
        self._completed_lines = 0
        self._current_line_bytes = 0
        self._has_open_line = False
        self._finished = False

    @property
    def capture_path(self) -> Path:
        return self._capture_path

    def append(self, data: bytes) -> str:
        if self._finished:
            raise RuntimeError("cannot append to a finished output accumulator")
        if not isinstance(data, bytes):
            raise TypeError("output accumulator expects bytes")
        self._capture.write(data)
        decoded = self._decoder.decode(data, final=False)
        self._append_decoded(decoded)
        return decoded

    def finish(self) -> OutputSnapshot:
        if not self._finished:
            self._finished = True
            remainder = self._decoder.decode(b"", final=True)
            self._append_decoded(remainder)
            self._capture.flush()
            os.fsync(self._capture.fileno())
            self._capture.close()
        return self.snapshot()

    def snapshot(self) -> OutputSnapshot:
        snapshot_text = self._snapshot_text()
        tail = truncate_tail(
            snapshot_text,
            max_lines=self._max_lines,
            max_bytes=self._max_bytes,
        )
        total_lines = self._completed_lines + (1 if self._has_open_line else 0)
        truncated = total_lines > self._max_lines or self._total_decoded_bytes > self._max_bytes
        truncated_by = tail.truncated_by
        if truncated and truncated_by is None:
            truncated_by = "bytes" if self._total_decoded_bytes > self._max_bytes else "lines"
        result = TruncationResult(
            content=tail.content,
            truncated=truncated,
            truncated_by=truncated_by,
            total_lines=total_lines,
            total_bytes=self._total_decoded_bytes,
            output_lines=tail.output_lines,
            output_bytes=tail.output_bytes,
            last_line_partial=tail.last_line_partial,
            max_lines=self._max_lines,
            max_bytes=self._max_bytes,
        )
        return OutputSnapshot(
            content=result.content,
            truncation=result,
            capture_path=self._capture_path,
        )

    def discard(self) -> None:
        if not self._capture.closed:
            self._capture.close()
        self._capture_path.unlink(missing_ok=True)

    def _append_decoded(self, text: str) -> None:
        if not text:
            return
        encoded_bytes = len(text.encode("utf-8"))
        self._total_decoded_bytes += encoded_bytes
        self._tail_text += text
        self._tail_bytes += encoded_bytes
        if self._tail_bytes > self._max_rolling_bytes * 2:
            self._trim_tail()

        newline_count = text.count("\n")
        if newline_count == 0:
            self._current_line_bytes += encoded_bytes
            self._has_open_line = True
            return
        self._completed_lines += newline_count
        tail = text.rsplit("\n", maxsplit=1)[1]
        self._current_line_bytes = len(tail.encode("utf-8"))
        self._has_open_line = bool(tail)

    def _trim_tail(self) -> None:
        raw = self._tail_text.encode("utf-8")
        if len(raw) <= self._max_rolling_bytes:
            self._tail_bytes = len(raw)
            return
        start = len(raw) - self._max_rolling_bytes
        while start < len(raw) and raw[start] & 0xC0 == 0x80:
            start += 1
        self._tail_starts_at_line_boundary = (
            self._tail_starts_at_line_boundary if start == 0 else raw[start - 1] == 0x0A
        )
        self._tail_text = raw[start:].decode("utf-8")
        self._tail_bytes = len(self._tail_text.encode("utf-8"))

    def _snapshot_text(self) -> str:
        if self._tail_starts_at_line_boundary:
            return self._tail_text
        first_newline = self._tail_text.find("\n")
        return self._tail_text if first_newline == -1 else self._tail_text[first_newline + 1 :]


def format_truncated_output(
    stream: str,
    result: TruncationResult,
    artifact_id: str | None,
) -> str:
    """Add the model-facing Pi-style truncation footer."""
    if result.last_line_partial:
        shown_bytes = result.output_bytes
        total_bytes = result.total_bytes
        summary = f"showing the last {shown_bytes} bytes of a {total_bytes}-byte line"
    elif result.truncated_by == "lines":
        first = result.total_lines - result.output_lines + 1
        summary = f"showing lines {first}-{result.total_lines} of {result.total_lines}"
    else:
        summary = (
            f"showing the last {result.output_lines} lines "
            f"({result.output_bytes} of {result.total_bytes} bytes)"
        )
    artifact = f"; full output artifact: {artifact_id}" if artifact_id else ""
    separator = "\n\n" if result.content else ""
    return f"{result.content}{separator}[{stream} truncated: {summary}{artifact}]"
