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

"""Deterministic policies for selecting one output from Workflow branches."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from alphaapollo.common.grader import (
    declared_answers_equivalent,
    extract_declared_answer,
)
from alphaapollo.common.grader.declared import declared_answer_key
from alphaapollo.reasoning.runtime import AgentResult
from alphaapollo.reasoning.verification import VerificationResult

__all__ = [
    "declared_answers_equivalent",
    "extract_declared_answer",
    "majority_vote",
    "select_result",
]

T = TypeVar("T")
WorkflowOutput = AgentResult | VerificationResult


def majority_vote(
    values: Sequence[T],
    *,
    key: Callable[[T], object | None] | None = None,
) -> T:
    """Return the earliest value among those tied for the largest vote count.

    This stable tie break makes results independent of dict/set iteration order
    and preserves the input branch order chosen by ``WorkflowExecutor``.
    """

    if not values:
        raise ValueError("majority_vote requires at least one value")
    key_fn: Callable[[T], object | None] = key or (lambda value: value)
    keys: list[object] = []
    counts: list[int] = []
    eligible: list[tuple[T, object]] = []
    for value in values:
        vote = key_fn(value)
        if vote is None:
            continue
        eligible.append((value, vote))
        for index, existing in enumerate(keys):
            if vote == existing:
                counts[index] += 1
                break
        else:
            keys.append(vote)
            counts.append(1)
    if not eligible:
        raise ValueError("majority_vote requires at least one non-abstaining value")
    winning_key = keys[max(range(len(keys)), key=counts.__getitem__)]
    return next(value for value, vote in eligible if vote == winning_key)


def select_result(
    values: Sequence[WorkflowOutput],
    *,
    strategy: str,
    selection_key: str = "exact_text",
) -> WorkflowOutput:
    """Apply one closed, configuration-safe ensemble selection strategy."""

    if not values:
        raise ValueError("result selection requires at least one value")
    if strategy == "first":
        return values[0]
    if strategy == "majority_vote":
        return majority_vote(values, key=lambda value: _output_key(value, selection_key))
    raise ValueError(f"unsupported ensemble strategy {strategy!r}")


def _output_key(value: WorkflowOutput, selection_key: str) -> object | None:
    if isinstance(value, AgentResult):
        text = value.final_text.strip()
        kind = "agent"
    else:
        if value.verdict != "pass":
            return None
        # A terminal verifier result is a judgment *about* its bound candidate.
        # Only accepted candidates are eligible to vote; failed or inconclusive
        # judgments abstain rather than lending weight to a rejected answer.
        text = value.candidate.strip()
        kind = "verification-candidate"
    if not text:
        return None
    if selection_key == "exact_text":
        return (f"{kind}-text", text)
    if selection_key == "declared_answer":
        answer = extract_declared_answer(text)
        return None if answer is None else (f"{kind}-answer", declared_answer_key(answer))
    raise ValueError(f"unsupported ensemble selection key {selection_key!r}")
