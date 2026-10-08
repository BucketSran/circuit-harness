# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact text grading for generic prepared datasets."""

from __future__ import annotations

from alphaapollo.common.grader.base import Verdict
from alphaapollo.common.grader.declared import extract_declared_answer
from alphaapollo.common.grader.text import strip_presentation

__all__ = ["ExactMatchGrader", "check_exact_match"]


def check_exact_match(candidate: str | None, gold: str | None) -> Verdict:
    """Compare non-empty answers after removing presentation-only wrappers."""

    if candidate is None or gold is None:
        return None
    normalized_candidate = strip_presentation(candidate)
    normalized_gold = strip_presentation(gold)
    if not normalized_candidate or not normalized_gold:
        return None
    return normalized_candidate == normalized_gold


class ExactMatchGrader:
    """Registry grader used by ``prepare_custom_data`` by default."""

    grader_id = "exact_match"

    def extract(self, text: str | None) -> str | None:
        if text is None:
            return None
        return extract_declared_answer(text)

    def grade(self, candidate: str | None, gold: str | None) -> Verdict:
        return check_exact_match(candidate, gold)
