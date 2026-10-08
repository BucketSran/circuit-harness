"""Vote-key extraction: where a declaration may come from, and what it accepts."""

from __future__ import annotations

from alphaapollo.common.grader import (
    declared_answer_key,
    declared_answers_equivalent,
    extract_declared_answer,
    extract_final_answer,
)

# A thinking model's answer as the ensemble roles now ask for it: scratch work
# with a rejected candidate inside <think>, then the conclusion inside <answer>.
THINKING_OUTPUT = (
    "<think>Let me try small cases. Suppose the count is \\boxed{17}. That "
    "double counts the diagonals, so 17 is wrong. Redo the inclusion-exclusion "
    "step.</think>\n"
    "Counting by inclusion-exclusion gives 204 arrangements.\n"
    "Final answer: 204\n"
    "<answer>\\boxed{204}</answer>"
)


def test_a_box_only_inside_thinking_loses_to_the_answer_that_follows() -> None:
    # A reasoning block is where candidate values go to be rejected, so a box
    # left there must never become the key a branch votes with.
    text = "<think>Maybe \\boxed{17}?</think>\nOn review, the final answer is 204."

    assert declared_answer_key(extract_declared_answer(text)) == ("number", 204)


def test_the_answer_block_wins_over_a_distractor_outside_it() -> None:
    text = "<answer>\\boxed{204}</answer>\n\nAside: an earlier attempt gave \\boxed{17}."

    assert extract_declared_answer(text) == "204"


def test_the_answer_block_is_read_before_prose_around_it() -> None:
    text = "The sub-case answer is 12.\n<answer>\\boxed{870}</answer>"

    assert extract_declared_answer(text) == "870"


def test_only_a_closed_answer_block_narrows_the_search() -> None:
    # A generation cut off inside the block never closed a conclusion; the text
    # around it is then the better guess, not a fragment.
    text = "<think>scratch \\boxed{17}</think>\nFinal answer: 204\n<answer>\\boxed{20"

    assert extract_declared_answer(text) == "204"


def test_untagged_text_behaves_exactly_as_before() -> None:
    assert extract_declared_answer("So the result is \\boxed{204}.") == "204"
    assert extract_declared_answer("Reasoning here. Final answer: 42") == "42"
    assert extract_declared_answer("First \\boxed{7}, on reflection \\boxed{42}") == "42"
    assert extract_declared_answer("no declaration anywhere") is None
    assert extract_declared_answer("\\boxed{}") is None


def test_chinese_markers_still_group_a_branch() -> None:
    # This marker set is wider than the math grader's on purpose: voting must
    # group a trace that declared its answer, whatever language it used.
    assert extract_declared_answer("推导过程如下。\n最终答案：42") == "42"
    assert extract_declared_answer("<think>试 \\boxed{17}</think>\n答案: 42") == "42"


def test_comparison_semantics_are_unchanged_by_the_span_rules() -> None:
    assert declared_answers_equivalent("204", "204.0")
    assert declared_answers_equivalent("204", "\\frac{408}{2}")
    assert not declared_answers_equivalent("204", "205")
    assert declared_answer_key("204") != declared_answer_key("blue")


def test_voting_and_scoring_agree_on_a_thinking_model_conclusion() -> None:
    # The regression this closes: voting used to group branches on text that
    # scoring deliberately excludes, so the two disagreed on the same output.
    voted = extract_declared_answer(THINKING_OUTPUT)
    scored = extract_final_answer(THINKING_OUTPUT)

    assert scored == "204"
    assert declared_answer_key(voted) == declared_answer_key(scored)


def test_two_branches_that_reason_differently_still_vote_together() -> None:
    other = (
        "<think>Brute force suggests \\boxed{17}.</think>\n"
        "A generating function gives the same count.\n"
        "<answer>\\boxed{204.0}</answer>"
    )

    assert declared_answer_key(extract_declared_answer(THINKING_OUTPUT)) == declared_answer_key(
        extract_declared_answer(other)
    )


def test_a_branch_whose_conclusion_declares_nothing_still_gets_a_key() -> None:
    # Voting needs a key more than it needs a scoreable answer, so the search
    # widens to the whole text rather than abstaining. Scoring refuses here.
    text = "<think>The count is \\boxed{17}.</think>"

    assert extract_declared_answer(text) == "17"
    assert extract_final_answer(text) is None


def test_numeric_declarations_keep_units_and_percent_signs() -> None:
    assert not declared_answers_equivalent("2 cm", "2")
    assert not declared_answers_equivalent("20%", "20")
    assert declared_answers_equivalent("0.5", r"\frac{1}{2}")
