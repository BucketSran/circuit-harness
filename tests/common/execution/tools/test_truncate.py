from __future__ import annotations

import pytest

from alphaapollo.common.execution.tools.builtins.shell import truncate_tail


def test_short_output_is_unchanged() -> None:
    result = truncate_tail("alpha\nbeta\n")

    assert result.content == "alpha\nbeta\n"
    assert result.truncated is False
    assert result.truncated_by is None
    assert result.total_lines == 2


def test_tail_truncation_keeps_the_last_complete_lines() -> None:
    result = truncate_tail("one\ntwo\nthree\nfour\n", max_lines=2, max_bytes=1_000)

    assert result.content == "three\nfour"
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.total_lines == 4
    assert result.output_lines == 2
    assert result.last_line_partial is False


def test_byte_truncation_keeps_valid_utf8_from_the_end() -> None:
    result = truncate_tail("prefix-你好世界", max_lines=10, max_bytes=7)

    assert result.content == "世界"
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.last_line_partial is True
    assert len(result.content.encode("utf-8")) <= 7


@pytest.mark.parametrize(
    ("max_lines", "max_bytes"),
    [(0, 10), (10, 0), (True, 10), (10, False)],
)
def test_invalid_limits_are_rejected(max_lines: int, max_bytes: int) -> None:
    with pytest.raises(ValueError):
        truncate_tail("output", max_lines=max_lines, max_bytes=max_bytes)
