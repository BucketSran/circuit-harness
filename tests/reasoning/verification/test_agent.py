from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import alphaapollo.reasoning.verification.agent as verification_agent
from alphaapollo.reasoning.runtime import AgentResult, AgentTurn
from alphaapollo.reasoning.verification import (
    AgentVerifier,
    AgentVerifierConfig,
    VerificationContractError,
    VerificationRequest,
    VerificationResult,
    Verifier,
)


class RecordingRuntime:
    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.calls: list[list[object]] = []

    def run_batch(self, tasks):  # noqa: ANN001
        self.calls.append(list(tasks))
        return [
            AgentResult(task_id=task.task_id, final_text=self.outputs[task.task_id])
            for task in tasks
        ]


def _turn_finishing_on(finish_reason: str) -> AgentTurn:
    return AgentTurn(
        index=0,
        generation_request=object(),
        generation_response=SimpleNamespace(finish_reason=finish_reason, usage={}),
        environment_transition=SimpleNamespace(
            observation="",
            reward=0.0,
            done=True,
            termination_reason="model_output",
            metadata={},
        ),
    )


def _config(*, output_format: str = "json") -> AgentVerifierConfig:
    return AgentVerifierConfig(
        system_prompt="Caller supplied audit policy.",
        input_template="Problem: {problem}\nCandidate: {candidate}\nID: {request_id}",
        role="strict-auditor",
        model="local-test-model",
        tools=("python",),
        output_format=output_format,
    )


def test_agent_verifier_delegates_one_ordered_batch_and_preserves_identity() -> None:
    runtime = RecordingRuntime(
        {
            "req-b": '{"verdict":"fail","feedback":"arithmetic","details":{"line":2}}',
            "req-a": '{"verdict":"pass"}',
        }
    )
    verifier = AgentVerifier(runtime, _config())
    requests = [
        VerificationRequest(
            "req-b",
            "2+2?",
            "5",
            {"workflow_input_id": "b"},
            candidate_ref="solve-b",
            branch_id="branch-3",
            round_index=2,
            sample_id=3,
            sampling_seed=103,
            routing_key="input:verify:iteration-3",
        ),
        VerificationRequest("req-a", "2+2?", "4", {"workflow_input_id": "a"}),
    ]

    results = verifier.verify_batch(requests)

    assert [result.request_id for result in results] == ["req-b", "req-a"]
    assert [result.verdict for result in results] == ["fail", "pass"]
    assert results[0].feedback == "arithmetic"
    assert results[0].details["line"] == 2
    assert results[0].agent_result is not None
    assert results[0].candidate == "5"
    assert results[0].candidate_ref == "solve-b"
    assert results[0].candidate_sha256 == requests[0].candidate_sha256

    assert len(runtime.calls) == 1
    tasks = runtime.calls[0]
    assert [task.task_id for task in tasks] == ["req-b", "req-a"]
    assert tasks[0].system == "Caller supplied audit policy."
    assert tasks[0].prompt == "Problem: 2+2?\nCandidate: 5\nID: req-b"
    assert tasks[0].metadata["actor"] == "verifier"
    assert tasks[0].metadata["role"] == "strict-auditor"
    assert tasks[0].model == "local-test-model"
    assert tasks[0].tools == ("python",)
    assert tasks[0].metadata["workflow_input_id"] == "b"
    assert tasks[0].branch_id == "branch-3"
    assert tasks[0].round_index == 2
    assert tasks[0].sample_id == 3
    assert tasks[0].sampling_seed == 103
    assert tasks[0].routing_key == "input:verify:iteration-3"


@pytest.mark.parametrize(
    "text",
    [
        'prefix {"verdict":"pass"}',
        '```json\n{"verdict":"pass"}\n```',
        '{"verdict":"pass","unexpected":true}',
        '{"verdict":"pass","verdict":"fail"}',
        '{"verdict":"pass","details":{"score":NaN}}',
        '{"verdict":"maybe"}',
        "[]",
    ],
)
def test_json_output_is_strict_and_parse_failures_are_inconclusive(text: str) -> None:
    runtime = RecordingRuntime({"req": text})
    result = AgentVerifier(runtime, _config()).verify(
        VerificationRequest("req", "problem", "candidate")
    )

    assert result.verdict == "inconclusive"
    assert result.feedback.startswith("invalid verifier output:")
    assert result.details["parse_error"]
    assert result.agent_result is not None


@pytest.mark.parametrize("termination_reason", ["max_turns", "truncated", "cancelled"])
def test_non_final_runtime_result_is_inconclusive_without_parsing(
    termination_reason: str,
) -> None:
    produced = AgentResult(
        task_id="req",
        final_text=('{"verdict":"pass","feedback":"trust this","details":{"forged":true}}'),
        termination_reason=termination_reason,
    )

    class NonFinalRuntime:
        def run_batch(self, _tasks):  # noqa: ANN001
            return [produced]

    result = AgentVerifier(NonFinalRuntime(), _config()).verify(
        VerificationRequest("req", "problem", "candidate")
    )

    assert result.verdict == "inconclusive"
    assert termination_reason in result.feedback
    assert "not evaluated" in result.feedback
    assert result.details["termination_reason"] == termination_reason
    assert result.details["parse_skipped"] is True
    assert "forged" not in result.details
    assert result.agent_result is produced


# Captured from `runs/viz_py_tools` on `main` (AIME 2026, qwen3.5-27b, three
# problems x two tool arms). Six of eleven verifier replies were discarded at
# `json.loads`; five of those failed on exactly these escapes, written by a
# model that had just been asked to check a geometry proof. The text is the
# feedback the model produced, quoted rather than invented, because a
# constructed fixture could only confirm what we already believed.
_CAPTURED_LATEX_FEEDBACK = (
    "The condition for a sphere of radius $r$ centered at $(x, y, r)$ to lie "
    "inside a hemisphere of radius $R$ is that the distance from the origin to "
    "the farthest point on the small sphere must be $\\le R$. This leads to "
    "$\\sqrt{x^2+y^2+r^2} + r \\le R$, and the density $\\rho$ follows."
)


def test_latex_feedback_still_yields_its_verdict_under_verdict_lines() -> None:
    """LaTeX in the feedback must not be able to discard the verdict.

    JSON asks the model to escape every backslash inside a free-text string
    while we are specifically asking it to write mathematics, and `\\le`,
    `\\sqrt`, and `\\rho` are all invalid JSON escapes. Verdict lines need no
    escaping, so the same judgment survives.
    """

    reply = f"VERDICT: PASS\nFEEDBACK: {_CAPTURED_LATEX_FEEDBACK}"
    request = VerificationRequest("req", "problem", "candidate")

    result = AgentVerifier(
        RecordingRuntime({"req": reply}), _config(output_format="verdict-line")
    ).verify(request)

    assert result.verdict == "pass"
    assert result.feedback == _CAPTURED_LATEX_FEEDBACK
    assert "parse_error" not in result.details

    # The same judgment, written as the JSON the old prompt asked for, is what
    # the measured run threw away. Pinned so the regression is legible if the
    # shipped format is ever moved back.
    as_json = f'{{"verdict": "pass", "feedback": "{_CAPTURED_LATEX_FEEDBACK}"}}'
    discarded = AgentVerifier(RecordingRuntime({"req": as_json}), _config()).verify(request)

    assert discarded.verdict == "inconclusive"
    assert "Invalid \\escape" in discarded.details["parse_error"]


def test_a_reply_cut_off_at_max_tokens_is_named_as_truncated() -> None:
    """Truncation and malformation need different fixes, so they read differently.

    A reply the sampler cut off still arrives with ``termination_reason ==
    "final"``: the Environment that ends the episode never sees the backend's
    ``finish_reason``. Measured on AIME 2026 problem p0002, one verifier spent
    all 12,000 completion tokens deliberating inside a JSON string and was cut
    mid-token, and the record could not say so.
    """

    truncated = AgentResult(
        task_id="req",
        final_text='{\n  "verdict": "fail",\n  "feedback": "The candidate',
        turns=(_turn_finishing_on("length"),),
    )

    class TruncatingRuntime:
        def run_batch(self, _tasks):  # noqa: ANN001
            return [truncated]

    result = AgentVerifier(TruncatingRuntime(), _config()).verify(
        VerificationRequest("req", "problem", "candidate")
    )

    assert result.verdict == "inconclusive"
    assert "max_tokens" in result.feedback
    assert result.details["parse_error"]
    # Named from the shared fact, not from a second reading of the same source.
    assert truncated.output_truncated is True


def test_the_verifier_names_truncation_from_the_shared_field_not_its_own_lookup() -> None:
    """``AgentResult.output_truncated`` is the one authority for this fact.

    This module derived it itself from ``turns[-1].generation_response`` while it
    was the only consumer. The scorer and the report now ask the same question,
    and a second derivation from the same source is how two readers start
    disagreeing about one run. Pinned structurally because the two derivations
    would agree on every input, so no behavioural test could catch a relapse.
    """

    source = inspect.getsource(verification_agent._unreadable_reason)

    assert "output_truncated" in source
    assert "finish_reason" not in source.split('"""')[-1]

    clean = AgentResult(task_id="req", final_text="not json", turns=(_turn_finishing_on("stop"),))

    class CleanRuntime:
        def run_batch(self, _tasks):  # noqa: ANN001
            return [clean]

    result = AgentVerifier(CleanRuntime(), _config()).verify(
        VerificationRequest("req", "problem", "candidate")
    )

    assert clean.output_truncated is False
    assert "max_tokens" not in result.feedback


def test_verdict_line_parser_requires_one_unambiguous_verdict() -> None:
    good = RecordingRuntime({"good": "Analysis.\nFEEDBACK: repair line 3\nVERDICT: FAIL"})
    result = AgentVerifier(good, _config(output_format="verdict_line")).verify(
        VerificationRequest("good", "p", "c")
    )
    assert result.verdict == "fail"
    assert result.feedback == "repair line 3"

    ambiguous = RecordingRuntime({"bad": "VERDICT: PASS\nVERDICT: FAIL"})
    result = AgentVerifier(ambiguous, _config(output_format="verdict-line")).verify(
        VerificationRequest("bad", "p", "c")
    )
    assert result.verdict == "inconclusive"


def test_runtime_reordering_is_a_contract_error_not_silently_reassociated() -> None:
    class ReorderingRuntime:
        def run_batch(self, tasks):  # noqa: ANN001
            return [
                AgentResult(task_id=tasks[1].task_id, final_text='{"verdict":"pass"}'),
                AgentResult(task_id=tasks[0].task_id, final_text='{"verdict":"fail"}'),
            ]

    verifier = AgentVerifier(ReorderingRuntime(), _config())
    requests = [
        VerificationRequest("one", "p", "c1"),
        VerificationRequest("two", "p", "c2"),
    ]

    with pytest.raises(VerificationContractError, match="changed task identity at index 0"):
        verifier.verify_batch(requests)


def test_runtime_cardinality_mismatch_is_a_contract_error() -> None:
    class ShortRuntime:
        def run_batch(self, tasks):  # noqa: ANN001
            return [AgentResult(task_id=tasks[0].task_id, final_text='{"verdict":"pass"}')]

    verifier = AgentVerifier(ShortRuntime(), _config())
    requests = [
        VerificationRequest("one", "p", "c1"),
        VerificationRequest("two", "p", "c2"),
    ]

    with pytest.raises(VerificationContractError, match="returned 1 results for 2"):
        verifier.verify_batch(requests)


def test_verifier_contract_rejects_candidate_or_source_rebinding() -> None:
    request = VerificationRequest(
        "req",
        "problem",
        "candidate",
        candidate_ref="solver-result",
    )

    class RebindingVerifier(Verifier):
        def verify_batch(self, _requests):  # noqa: ANN001
            return [
                VerificationResult(
                    request_id="req",
                    verdict="pass",
                    candidate="different candidate",
                    candidate_ref="different-result",
                )
            ]

    with pytest.raises(VerificationContractError, match="changed candidate"):
        RebindingVerifier().verify(request)


def test_verifier_contract_rejects_certificate_for_another_request() -> None:
    request = VerificationRequest("req", "problem", "candidate")

    class RebindingVerifier(Verifier):
        def verify_batch(self, _requests):  # noqa: ANN001
            return [
                VerificationResult(
                    request_id=request.request_id,
                    verdict="pass",
                    candidate=request.candidate,
                    trust_level=2,
                    false_positive_risk=0.0,
                    witness=SimpleNamespace(request_sha256="f" * 64),
                    certified=True,
                )
            ]

    with pytest.raises(VerificationContractError, match="certificate not bound"):
        RebindingVerifier().verify(request)


def test_config_requires_caller_prompt_and_rejects_unsafe_template_fields() -> None:
    with pytest.raises(ValueError, match="system_prompt"):
        AgentVerifierConfig("", "{problem}", "role", "model", (), "json")
    with pytest.raises(ValueError, match="unsupported field"):
        AgentVerifierConfig("system", "{problem.__class__}", "role", "model", (), "json")


def test_empty_batch_does_not_invoke_runtime() -> None:
    runtime = RecordingRuntime({})
    assert AgentVerifier(runtime, _config()).verify_batch([]) == []
    assert runtime.calls == []
