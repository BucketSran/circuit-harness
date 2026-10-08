"""Branch-local scratchpad conventions and bounded update operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SCRATCHPAD_RELATIVE_PATH = ".alphaapollo/scratchpad.md"
SCRATCHPAD_VISIBLE_PATH = f"/workspace/{SCRATCHPAD_RELATIVE_PATH}"
SCRATCHPAD_READ_CAP = 6_000
SCRATCHPAD_FILE_CAP = 20_000


def read_scratchpad(owner: Any) -> str:
    """Read from a sandbox session or filesystem path; only absence means empty."""

    if isinstance(owner, (str, Path)):
        path = Path(owner)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")
    try:
        raw = owner.read_file(SCRATCHPAD_RELATIVE_PATH)
    except FileNotFoundError:
        return ""
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)


def bounded_scratchpad(text: str) -> tuple[str, bool]:
    if len(text) <= SCRATCHPAD_READ_CAP:
        return text, False
    return "[truncated]\n" + text[-SCRATCHPAD_READ_CAP:], True


def update_scratchpad(
    current: str,
    *,
    op: str,
    content: str = "",
    section: str = "",
) -> str:
    """Apply append/read/replace semantics without performing an I/O effect."""

    normalized = op.strip().lower()
    if normalized == "read":
        return current
    if normalized not in {"append", "replace"}:
        raise ValueError("op must be append, read, or replace")
    if not content.strip():
        raise ValueError("content is required")
    if normalized == "append":
        updated = f"{current.rstrip()}\n\n{content}" if current.strip() else content
    else:
        if not section.strip():
            raise ValueError("section is required")
        if any(line.strip().startswith("## ") for line in content.splitlines()):
            raise ValueError("replacement content must not contain '## ' headings")
        updated, found = _replace_section(current, section.strip(), content)
        if not found:
            raise ValueError(f"section {section!r} not found")
    if len(updated) > SCRATCHPAD_FILE_CAP:
        raise ValueError(f"scratchpad exceeds {SCRATCHPAD_FILE_CAP} characters")
    return updated


def write_scratchpad(owner: Any, content: str) -> None:
    """Write a validated complete scratchpad to a sandbox or local path."""

    if len(content) > SCRATCHPAD_FILE_CAP:
        raise ValueError(f"scratchpad exceeds {SCRATCHPAD_FILE_CAP} characters")
    if isinstance(owner, (str, Path)):
        path = Path(owner)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return
    owner.write_file(SCRATCHPAD_RELATIVE_PATH, content)


def _replace_section(text: str, section: str, content: str) -> tuple[str, bool]:
    lines = text.splitlines()
    heading = f"## {section}"
    start = next((index for index, line in enumerate(lines) if line.strip() == heading), None)
    if start is None:
        return text, False
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].strip().startswith("## ")),
        len(lines),
    )
    replaced = lines[: start + 1] + [content, ""] + lines[end:]
    return "\n".join(replaced).rstrip() + "\n", True


__all__ = [
    "SCRATCHPAD_FILE_CAP",
    "SCRATCHPAD_READ_CAP",
    "SCRATCHPAD_RELATIVE_PATH",
    "SCRATCHPAD_VISIBLE_PATH",
    "bounded_scratchpad",
    "read_scratchpad",
    "update_scratchpad",
    "write_scratchpad",
]
