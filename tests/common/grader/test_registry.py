from __future__ import annotations

import pytest

from alphaapollo.common.grader import (
    available_graders,
    check_exact_match,
    get_grader,
    grade,
    register_grader,
)
from alphaapollo.common.grader.base import Grader


def test_registry_dispatches_the_declared_grader() -> None:
    assert available_graders() == ("environment_success", "exact_match")
    assert grade("204", "204", grader_id="exact_match") is True
    assert grade("205", "204", grader_id="exact_match") is False


def test_exact_match_supports_custom_data_default() -> None:
    assert check_exact_match("Answer: Blue", "Blue") is False
    assert check_exact_match("Blue", "Blue") is True
    assert check_exact_match("blue", "Blue") is False
    assert check_exact_match(None, "Blue") is None
    assert get_grader("exact_match").extract("The final answer is Blue") == "Blue"


def test_unknown_grader_names_what_is_available() -> None:
    with pytest.raises(ValueError, match="unknown grader_id 'nope'"):
        grade("1", "1", grader_id="nope")


@pytest.mark.parametrize("grader_id", ["integer_answer", "math_expression"])
def test_removed_grader_is_rejected(grader_id: str) -> None:
    with pytest.raises(ValueError, match="unknown grader_id"):
        get_grader(grader_id)


def test_registered_graders_satisfy_the_protocol() -> None:
    for grader_id in available_graders():
        assert isinstance(get_grader(grader_id), Grader)


def test_lookup_is_cached_so_graders_are_reused() -> None:
    assert get_grader("exact_match") is get_grader("exact_match")


def test_registration_refuses_a_silent_override() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_grader("exact_match", lambda: get_grader("exact_match"))


def test_unscoreable_is_none_and_never_collapsed_into_false() -> None:
    assert check_exact_match(None, "204") is None
    assert check_exact_match("", "204") is None
    assert check_exact_match("205", "204") is False
