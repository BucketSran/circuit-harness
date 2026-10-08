"""The environment-graded grader id: resolvable, unscoreable, refused early."""

from __future__ import annotations

import pytest

from alphaapollo.common.grader import (
    available_graders,
    get_grader,
    grade,
    require_offline_grader,
)
from alphaapollo.common.grader.base import Grader
from alphaapollo.common.grader.environment import EnvironmentSuccessGrader


def test_the_id_resolves_like_any_other_grader() -> None:
    """A prepared robotics dataset must load wherever a grader_id is resolved."""

    assert "environment_success" in available_graders()
    grader = get_grader("environment_success")
    assert isinstance(grader, EnvironmentSuccessGrader)
    assert isinstance(grader, Grader)
    assert grader.grader_id == "environment_success"


def test_grading_is_unscoreable_and_never_collapses_into_a_verdict() -> None:
    """``None`` is the documented "could not be scored at all" state.

    Raising here instead put the refusal inside the per-cell
    ``except Exception`` in ``workflows/scoring.py`` (#256 review), which
    reports a category error as an ordinary cell failure.
    """

    assert get_grader("environment_success").extract("The final answer is done") is None
    assert grade("done", "environment", grader_id="environment_success") is None
    assert grade(None, None, grader_id="environment_success") is None


def test_offline_scoring_is_refused_at_configuration_time() -> None:
    with pytest.raises(ValueError, match="execution backend's success"):
        require_offline_grader("environment_success")


def test_ordinary_graders_pass_the_offline_gate() -> None:
    """The marker is an opt-out, so a grader without it stays scoreable."""

    for grader_id in ("exact_match", "exact_match"):
        assert require_offline_grader(grader_id) is get_grader(grader_id)


def test_the_offline_gate_still_names_an_unknown_id() -> None:
    with pytest.raises(ValueError, match=r"unknown grader_id 'nope'"):
        require_offline_grader("nope")
