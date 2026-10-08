"""Shared parsing rules for crash-tolerant memory JSONL journals."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JournalRecord:
    line_number: int
    value: Mapping[str, Any]


class JournalDecodeError(ValueError):
    def __init__(self, line_number: int) -> None:
        self.line_number = line_number
        super().__init__(f"invalid memory journal at line {line_number}")


def read_journal_records(path: Path) -> tuple[JournalRecord, ...]:
    """Parse complete JSONL records, ignoring only a torn final append."""

    raw_lines = path.read_bytes().splitlines(keepends=True)
    records: list[JournalRecord] = []
    for index, raw_line in enumerate(raw_lines):
        terminated = raw_line.endswith((b"\n", b"\r"))
        payload = raw_line.rstrip(b"\r\n")
        try:
            value = json.loads(payload.decode("utf-8"))
            if not isinstance(value, Mapping):
                raise TypeError("journal record must be an object")
        except (UnicodeDecodeError, TypeError, json.JSONDecodeError) as exc:
            if index == len(raw_lines) - 1 and not terminated:
                break
            raise JournalDecodeError(index + 1) from exc
        records.append(JournalRecord(line_number=index + 1, value=value))
    return tuple(records)


__all__ = ["JournalDecodeError", "JournalRecord", "read_journal_records"]
