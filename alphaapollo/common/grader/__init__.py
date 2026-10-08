"""Answer extraction, equivalence, and grader dispatch shared by all consumers.

The split is by domain, not by pipeline stage:

* ``base`` is the contract -- a verdict type and the grader shape.
* ``registry`` dispatches a prepared task's ``grader_id`` to its grader.
* ``text`` is domain-free: find the answer marker, strip presentation. Nothing
  here knows what an answer means.
* ``declared`` is the consensus layer's extractor, using exact declaration
  semantics; see its module docstring.
* ``answer`` reads explicit conclusions for verifier and Environment clients.

A grader owns both halves of its domain, extraction and comparison, so callers
dispatch on ``grader_id`` instead of encoding one benchmark's rules.

This package carries no product-line dependency, so reasoning verification,
workflow voting, workflow evaluation, and training reward can all reach it.
"""

from alphaapollo.common.grader.answer import extract_boxed_answer, extract_final_answer
from alphaapollo.common.grader.base import Grader, Verdict
from alphaapollo.common.grader.declared import (
    declared_answer_key,
    declared_answers_equivalent,
    extract_declared_answer,
)
from alphaapollo.common.grader.exact import check_exact_match
from alphaapollo.common.grader.registry import (
    available_graders,
    get_grader,
    grade,
    register_grader,
    require_offline_grader,
)
from alphaapollo.common.grader.text import scan_boxed, strip_presentation

__all__ = [
    "Grader",
    "Verdict",
    "available_graders",
    "check_exact_match",
    "declared_answer_key",
    "declared_answers_equivalent",
    "extract_boxed_answer",
    "extract_declared_answer",
    "extract_final_answer",
    "get_grader",
    "grade",
    "register_grader",
    "require_offline_grader",
    "scan_boxed",
    "strip_presentation",
]
