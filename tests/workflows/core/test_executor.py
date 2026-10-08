from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import pytest

from alphaapollo.common.prompts import prompt_digest
from alphaapollo.reasoning.runtime import (
    AgentResult,
    AgentRuntime,
    AgentTask,
)
from alphaapollo.reasoning.verification import (
    AgentVerifier,
    AgentVerifierConfig,
    VerificationRequest,
    VerificationResult,
    Verifier,
)
from alphaapollo.workflows.config import parse_workflow_config
from alphaapollo.workflows.executor import WorkflowExecutionError, WorkflowExecutor
from alphaapollo.workflows.records import (
    StepResult,
    Workflow,
    WorkflowInput,
    WorkflowResumeStep,
)


class FakeRuntime(AgentRuntime):
    def __init__(self, answer: Callable[[AgentTask], str]) -> None:
        self.answer = answer
        self.batches: list[tuple[AgentTask, ...]] = []

    def run_batch(self, tasks: Sequence[AgentTask]) -> list[AgentResult]:
        batch = tuple(tasks)
        self.batches.append(batch)
        return [
            AgentResult(
                task_id=task.task_id,
                final_text=self.answer(task),
                termination_reason="final",
                metadata=task.metadata,
            )
            for task in batch
        ]


class FakeVerifier(Verifier):
    def __init__(self, verdict: Callable[[VerificationRequest], str]) -> None:
        self.verdict = verdict
        self.batches: list[tuple[VerificationRequest, ...]] = []

    def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
        batch = tuple(requests)
        self.batches.append(batch)
        return [
            VerificationResult(
                request_id=request.request_id,
                verdict=self.verdict(request),
                candidate=request.candidate,
                candidate_ref=request.candidate_ref,
                feedback=f"feedback:{request.candidate}",
            )
            for request in batch
        ]


def _branching_config() -> dict[str, object]:
    return {
        "version": 1,
        "name": "branching",
        "roles": [
            {"id": "solver", "target": "runtime", "system_prompt": "propose"},
            {
                "id": "judge",
                "target": "judge",
                "system_prompt": "verify",
                "input_template": "{problem}\n{candidate}",
                "output_format": "json",
            },
            {"id": "accept", "target": "runtime", "system_prompt": "accept"},
            {"id": "repair", "target": "runtime", "system_prompt": "repair"},
        ],
        "steps": [
            {
                "id": "propose",
                "kind": "agent",
                "role": "solver",
                "input_template": "{problem}",
            },
            {"id": "verify", "kind": "verifier", "role": "judge"},
            {
                "id": "accepted",
                "kind": "agent",
                "role": "accept",
                "input_template": "accepted:{candidate}",
                "output": True,
            },
            {
                "id": "repaired",
                "kind": "agent",
                "role": "repair",
                "input_template": "repaired:{candidate}:{feedback}",
            },
        ],
        "entry_step": "propose",
        "transitions": [
            {"source": "propose", "target": "verify"},
            {"source": "verify", "target": "accepted", "condition": "passed"},
            {"source": "verify", "target": "repaired", "condition": "not_passed"},
            {"source": "repaired", "target": "accepted"},
        ],
    }


def test_batch_branching_routing_and_input_order() -> None:
    workflow = Workflow.from_config(parse_workflow_config(_branching_config()))
    runtime = FakeRuntime(
        lambda task: (
            f"candidate:{task.prompt}"
            if task.system == "propose"
            else f"{task.system}:{task.prompt}"
        )
    )
    verifier = FakeVerifier(
        lambda request: "pass" if request.problem.startswith("pass") else "fail"
    )
    executor = WorkflowExecutor(
        workflow, runtimes={"runtime": runtime}, verifiers={"judge": verifier}
    )

    results = executor.run_batch(
        [
            WorkflowInput(input_id="second", problem="fail problem"),
            WorkflowInput(input_id="first", problem="pass problem"),
        ]
    )

    assert [result.input_id for result in results] == ["second", "first"]
    assert [step.step_id for step in results[0].steps] == [
        "propose",
        "verify",
        "repaired",
        "accepted",
    ]
    assert [step.step_id for step in results[1].steps] == [
        "propose",
        "verify",
        "accepted",
    ]
    assert len(runtime.batches[0]) == 2
    assert all(task.metadata["actor"] == "solver" for task in runtime.batches[0])
    assert [task.metadata["role"] for task in runtime.batches[0]] == ["solver", "solver"]
    assert len(verifier.batches) == 1
    assert [request.problem for request in verifier.batches[0]] == [
        "fail problem",
        "pass problem",
    ]
    assert results[0].status == results[1].status == "completed"


def test_stateful_verifier_can_select_a_prior_branch_candidate() -> None:
    class ChampionSelectingVerifier(FakeVerifier):
        def __init__(self) -> None:
            super().__init__(lambda _request: "pass")
            self.champion_ref: str | None = None

        def select_candidate_ref(self, request: VerificationRequest) -> str | None:
            if request.metadata["workflow_step_id"] == "finish":
                return self.champion_ref
            return request.candidate_ref

        def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
            for request in requests:
                if request.metadata["workflow_step_id"] == "evaluate":
                    self.champion_ref = request.candidate_ref
            return super().verify_batch(requests)

    raw = {
        "version": 1,
        "name": "select_prior_candidate",
        "roles": [
            {"id": "solver", "target": "runtime", "system_prompt": "solve"},
            {"id": "judge", "target": "judge", "system_prompt": "judge"},
        ],
        "steps": [
            {"id": "propose", "kind": "agent", "role": "solver"},
            {"id": "evaluate", "kind": "verifier", "role": "judge"},
            {"id": "revise", "kind": "agent", "role": "solver"},
            {"id": "finish", "kind": "verifier", "role": "judge", "output": True},
        ],
        "entry_step": "propose",
        "transitions": [
            {"source": "propose", "target": "evaluate"},
            {"source": "evaluate", "target": "revise"},
            {"source": "revise", "target": "finish"},
        ],
    }
    candidates = iter(("public champion", "later regression"))
    verifier = ChampionSelectingVerifier()

    result = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"runtime": FakeRuntime(lambda _task: next(candidates))},
        verifiers={"judge": verifier},
    ).run(WorkflowInput(input_id="select", problem="P"))

    assert isinstance(result.output, VerificationResult)
    assert result.output.candidate == "public champion"
    assert result.output.candidate_ref == verifier.champion_ref
    assert [batch[0].candidate for batch in verifier.batches] == [
        "public champion",
        "public champion",
    ]


def test_stateful_verifier_cannot_select_outside_branch_history() -> None:
    class UnknownCandidateVerifier(FakeVerifier):
        def select_candidate_ref(self, request: VerificationRequest) -> str | None:
            del request
            return "another-branch:agent-result"

    raw = {
        "version": 1,
        "name": "reject_unknown_candidate",
        "roles": [
            {"id": "solver", "target": "runtime", "system_prompt": "solve"},
            {"id": "judge", "target": "judge", "system_prompt": "judge"},
        ],
        "steps": [
            {"id": "propose", "kind": "agent", "role": "solver"},
            {"id": "finish", "kind": "verifier", "role": "judge", "output": True},
        ],
        "entry_step": "propose",
        "transitions": [{"source": "propose", "target": "finish"}],
    }
    executor = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"runtime": FakeRuntime(lambda _task: "candidate")},
        verifiers={"judge": UnknownCandidateVerifier(lambda _request: "pass")},
    )

    with pytest.raises(WorkflowExecutionError, match="outside the current Workflow branch"):
        executor.run(WorkflowInput(input_id="select", problem="P"))


def test_shared_prompt_provenance_reaches_the_agent_task() -> None:
    workflow = Workflow.from_config(
        parse_workflow_config(
            {
                "version": 1,
                "name": "shared_prompt",
                "roles": [
                    {
                        "id": "solver",
                        "target": "runtime",
                        "prompt_ref": "roles.model_default",
                    }
                ],
                "steps": [{"id": "solve", "kind": "agent", "role": "solver", "output": True}],
                "entry_step": "solve",
                "transitions": [],
            }
        )
    )
    runtime = FakeRuntime(lambda task: "answer")

    WorkflowExecutor(workflow, runtimes={"runtime": runtime}, verifiers={}).run(
        WorkflowInput(input_id="one", problem="2 + 2")
    )

    task = runtime.batches[0][0]
    assert task.system == ""
    assert task.prompt == "2 + 2"
    assert task.metadata["prompt_ref"] == "roles.model_default"
    assert task.metadata["prompt_version"] == 1
    assert task.metadata["prompt_digest"] == prompt_digest("", "2 + 2")


def test_bounded_revise_loop_uses_fallback_after_budget() -> None:
    raw = {
        "version": 1,
        "name": "revise",
        "roles": [
            {"id": "solver", "target": "runtime", "system_prompt": "solve"},
            {
                "id": "judge",
                "target": "judge",
                "system_prompt": "judge",
                "input_template": "{problem}:{candidate}",
                "output_format": "json",
            },
        ],
        "steps": [
            {"id": "propose", "kind": "agent", "role": "solver"},
            {"id": "verify", "kind": "verifier", "role": "judge"},
            {
                "id": "revise",
                "kind": "agent",
                "role": "solver",
                "input_template": "{candidate}:{feedback}",
            },
            {
                "id": "final",
                "kind": "agent",
                "role": "solver",
                "input_template": "{candidate}",
                "output": True,
            },
        ],
        "entry_step": "propose",
        "transitions": [
            {"source": "propose", "target": "verify"},
            {"source": "verify", "target": "final", "condition": "passed"},
            {
                "source": "verify",
                "target": "revise",
                "condition": "not_passed",
                "max_iterations": 2,
            },
            {"source": "verify", "target": "final", "condition": "not_passed"},
            {"source": "revise", "target": "verify"},
        ],
    }
    runtime = FakeRuntime(lambda task: f"answer:{task.prompt}")
    verifier = FakeVerifier(lambda _request: "fail")
    executor = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"runtime": runtime},
        verifiers={"judge": verifier},
    )

    result = executor.run(WorkflowInput(input_id="loop", problem="problem"))

    counts = Counter(step.step_id for step in result.steps)
    assert counts == {"propose": 1, "verify": 3, "revise": 2, "final": 1}
    assert [step.iteration for step in result.steps if step.step_id == "verify"] == [1, 2, 3]
    assert result.status == "completed"


def test_explicit_empty_step_template_is_not_replaced_by_a_default() -> None:
    raw = {
        "version": 1,
        "name": "empty_template",
        "roles": [
            {
                "id": "solver",
                "target": "runtime",
                "system_prompt": "solve",
                "input_template": "role:{problem}",
            }
        ],
        "steps": [
            {
                "id": "answer",
                "kind": "agent",
                "role": "solver",
                "input_template": "",
                "output": True,
            }
        ],
        "entry_step": "answer",
        "transitions": [],
    }
    runtime = FakeRuntime(lambda task: f"prompt={task.prompt!r}")

    result = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"runtime": runtime},
        verifiers={},
    ).run(WorkflowInput(input_id="empty", problem="must not leak"))

    assert runtime.batches[0][0].prompt == ""
    assert isinstance(result.output, AgentResult)
    assert result.output.final_text == "prompt=''"


def test_verifier_target_routes_to_configured_implementation() -> None:
    raw = _branching_config()
    raw["roles"][1]["target"] = "lean"  # type: ignore[index]
    workflow = Workflow.from_config(parse_workflow_config(raw))
    runtime = FakeRuntime(lambda task: task.prompt)
    unused = FakeVerifier(lambda _request: "fail")
    lean = FakeVerifier(lambda _request: "pass")

    result = WorkflowExecutor(
        workflow,
        runtimes={"runtime": runtime},
        verifiers={"judge": unused, "lean": lean},
    ).run(WorkflowInput(input_id="route", problem="P"))

    assert not unused.batches
    assert len(lean.batches) == 1
    assert [step.step_id for step in result.steps] == ["propose", "verify", "accepted"]


def test_generic_verifier_step_interchanges_agent_and_custom_implementations() -> None:
    raw = _branching_config()
    raw["roles"][1]["model"] = "test-judge-model"  # type: ignore[index]
    workflow = Workflow.from_config(parse_workflow_config(raw))
    role = workflow.get_role("judge")
    assert role.input_template is not None
    assert role.model is not None
    assert role.output_format is not None

    agent_runtime = FakeRuntime(lambda _task: '{"verdict":"pass","feedback":"agent accepted"}')
    agent_verifier = AgentVerifier(
        agent_runtime,
        AgentVerifierConfig(
            system_prompt=role.system_prompt,
            input_template=role.input_template,
            role=role.id,
            model=role.model,
            tools=role.tools,
            output_format=role.output_format,
        ),
    )

    custom_verifier = FakeVerifier(lambda _request: "pass")
    workflow_input = WorkflowInput(
        input_id="swap", problem="Check the candidate circuit specification."
    )

    def execute_with(verifier: Verifier):
        runtime = FakeRuntime(lambda task: f"{task.system}:{task.prompt}")
        return WorkflowExecutor(
            workflow,
            runtimes={"runtime": runtime},
            verifiers={"judge": verifier},
        ).run(workflow_input)

    agent_result = execute_with(agent_verifier)
    custom_result = execute_with(custom_verifier)
    agent_judgment = next(step.output for step in agent_result.steps if step.step_id == "verify")
    custom_judgment = next(step.output for step in custom_result.steps if step.step_id == "verify")

    expected_request_id = "swap:branch-0:verify:iteration-1"
    assert type(agent_judgment) is type(custom_judgment) is VerificationResult
    assert agent_judgment.request_id == custom_judgment.request_id == expected_request_id
    assert agent_judgment.verdict == custom_judgment.verdict == "pass"
    expected_candidate_ref = "swap:branch-0:propose:iteration-1"
    assert agent_judgment.candidate_ref == custom_judgment.candidate_ref == expected_candidate_ref
    assert agent_judgment.candidate_sha256 == custom_judgment.candidate_sha256
    assert [step.step_id for step in agent_result.steps] == ["propose", "verify", "accepted"]
    assert [step.step_id for step in custom_result.steps] == ["propose", "verify", "accepted"]
    assert agent_runtime.batches[0][0].task_id == expected_request_id
    assert agent_judgment.agent_result is not None
    assert agent_judgment.agent_result.task_id == expected_request_id
    assert custom_judgment.agent_result is None
    expected_candidate = f"propose:{workflow_input.problem}"
    assert len(custom_verifier.batches) == 1
    assert expected_candidate == custom_verifier.batches[0][0].candidate
    assert agent_runtime.batches[0][0].prompt == (f"{workflow_input.problem}\n{expected_candidate}")


def test_ensemble_selection_is_stable_and_preserves_branch_steps() -> None:
    raw = {
        "version": 1,
        "name": "ensemble",
        "ensemble": {"replicas": 3, "strategy": "majority_vote"},
        "roles": [{"id": "solver", "target": "runtime", "system_prompt": "solve"}],
        "steps": [
            {
                "id": "answer",
                "kind": "agent",
                "role": "solver",
                "input_template": "{problem}:{branch_index}",
                "output": True,
            }
        ],
        "entry_step": "answer",
        "transitions": [],
    }
    runtime = FakeRuntime(lambda task: "A" if task.metadata["branch_index"] in (0, 2) else "B")
    result = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"runtime": runtime},
        verifiers={},
    ).run(WorkflowInput(input_id="ensemble-input", problem="P"))

    assert isinstance(result.output, AgentResult)
    assert result.output.final_text == "A"
    assert result.selected_branch_index == 0
    assert [step.branch_index for step in result.steps] == [0, 1, 2]
    assert len(runtime.batches) == 1
    assert len(runtime.batches[0]) == 3


def test_diverged_branches_keep_their_own_outputs_and_their_own_step_groups() -> None:
    """Once branches route differently, each output must return to its own branch.

    A scheduling round matches a group's outputs back to its slots by
    ``(input_index, branch_index)``. This is the case where that matters: branch 0
    is at ``accepted`` while branch 1 is at ``repaired``, so two groups run in the
    same round and a mismatched key would splice one branch's answer into the
    other's history rather than failing.

    It also pins the current batching contract: two slots of the same input, at
    different steps, on the same runtime, are two separate ``run_batch`` calls.
    """

    raw = {**_branching_config(), "ensemble": {"replicas": 2, "strategy": "first"}}
    runtime = FakeRuntime(lambda task: f"{task.system}:{task.metadata['branch_index']}")
    verifier = FakeVerifier(
        lambda request: "pass" if request.metadata["branch_index"] == 0 else "fail"
    )

    result = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"runtime": runtime},
        verifiers={"judge": verifier},
    ).run(WorkflowInput(input_id="diverging", problem="P"))

    by_branch: dict[int, list[str]] = {0: [], 1: []}
    for step in result.steps:
        by_branch[step.branch_index].append(step.step_id)
    assert by_branch[0] == ["propose", "verify", "accepted"]
    assert by_branch[1] == ["propose", "verify", "repaired", "accepted"]
    assert all(
        step.output.final_text.endswith(f":{step.branch_index}")
        for step in result.steps
        if isinstance(step.output, AgentResult)
    )
    # Round 3 splits: `accepted` for branch 0 and `repaired` for branch 1 reach the
    # same runtime as two calls, because the group key includes the step id.
    branch_groups = [
        sorted(task.metadata["branch_index"] for task in batch) for batch in runtime.batches
    ]
    assert branch_groups == [[0, 1], [0], [1], [1]]


def test_agent_ensemble_votes_on_declared_answer_and_empty_answers_abstain() -> None:
    raw = {
        "version": 1,
        "name": "answer_vote",
        "ensemble": {
            "replicas": 3,
            "strategy": "majority_vote",
            "selection_key": "declared_answer",
        },
        "roles": [{"id": "solver", "target": "runtime", "system_prompt": "solve"}],
        "steps": [
            {
                "id": "answer",
                "kind": "agent",
                "role": "solver",
                "input_template": "{branch_index}",
                "output": True,
            }
        ],
        "entry_step": "answer",
        "transitions": [],
    }
    outputs = {
        0: "wrong reasoning\nFinal answer: 7",
        1: "method A\nFinal answer: 42",
        2: "method B\nThe final answer is 42.0",
    }
    runtime = FakeRuntime(lambda task: outputs[task.sample_id])
    result = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"runtime": runtime},
        verifiers={},
    ).run(WorkflowInput(input_id="vote-answer", problem="P"))

    assert isinstance(result.output, AgentResult)
    assert result.output.final_text == outputs[1]
    assert result.selected_branch_index == 1
    assert [task.branch_id for task in runtime.batches[0]] == [
        "branch-0",
        "branch-1",
        "branch-2",
    ]
    assert [task.round_index for task in runtime.batches[0]] == [0, 0, 0]
    assert [task.sample_id for task in runtime.batches[0]] == [0, 1, 2]
    assert len({task.routing_key for task in runtime.batches[0]}) == 1

    empty_runtime = FakeRuntime(
        lambda task: (
            "Final answer: valid"
            if task.sample_id == 2
            else (r"\boxed{}" if task.sample_id == 0 else "Final answer: ***")
        )
    )
    empty_result = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"runtime": empty_runtime},
        verifiers={},
    ).run(WorkflowInput(input_id="vote-empty", problem="P"))
    assert isinstance(empty_result.output, AgentResult)
    assert empty_result.output.final_text == "Final answer: valid"
    assert empty_result.selected_branch_index == 2

    all_empty = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={
            "runtime": FakeRuntime(
                lambda task: (
                    r"\boxed{}"
                    if task.sample_id == 0
                    else ("Final answer: ." if task.sample_id == 1 else "")
                )
            )
        },
        verifiers={},
    )
    with pytest.raises(WorkflowExecutionError, match="non-abstaining"):
        all_empty.run(WorkflowInput(input_id="vote-all-empty", problem="P"))


def test_terminal_verifier_ensemble_only_votes_on_passed_candidates() -> None:
    class BranchVerifier(Verifier):
        def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
            outcomes = {
                0: ("fail", "first rejection"),
                1: ("fail", "second rejection"),
                2: ("pass", "accepted"),
            }
            return [
                VerificationResult(
                    request_id=request.request_id,
                    verdict=outcomes[request.metadata["branch_index"]][0],
                    candidate=request.candidate,
                    candidate_ref=request.candidate_ref,
                    feedback=outcomes[request.metadata["branch_index"]][1],
                )
                for request in requests
            ]

    raw = {
        "version": 1,
        "name": "verifier_ensemble",
        "ensemble": {
            "replicas": 3,
            "strategy": "majority_vote",
            "selection_key": "declared_answer",
        },
        "roles": [
            {"id": "solver", "target": "solver", "system_prompt": "solve"},
            {
                "id": "judge",
                "target": "judge",
                "system_prompt": "judge",
                "input_template": "{problem}\n{candidate}",
                "output_format": "json",
            },
        ],
        "steps": [
            {"id": "solve", "kind": "agent", "role": "solver"},
            {"id": "verify", "kind": "verifier", "role": "judge", "output": True},
        ],
        "entry_step": "solve",
        "transitions": [{"source": "solve", "target": "verify"}],
    }
    candidates = {
        0: "Rejected reasoning 1. Final answer: WRONG",
        1: "Unverified reasoning. Final answer: WRONG",
        2: "Accepted reasoning. Final answer: RIGHT",
    }
    result = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"solver": FakeRuntime(lambda task: candidates[task.sample_id])},
        verifiers={"judge": BranchVerifier()},
    ).run(WorkflowInput(input_id="vote", problem="P"))

    assert isinstance(result.output, VerificationResult)
    assert result.output.verdict == "pass"
    assert result.output.feedback == "accepted"
    assert result.output.candidate == candidates[2]
    assert result.selected_branch_index == 2


def test_terminal_verifier_ensemble_fails_when_every_judgment_abstains() -> None:
    class RejectingVerifier(Verifier):
        def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
            return [
                VerificationResult(
                    request_id=request.request_id,
                    verdict="fail" if index == 0 else "inconclusive",
                    candidate=request.candidate,
                    candidate_ref=request.candidate_ref,
                )
                for index, request in enumerate(requests)
            ]

    raw = {
        "version": 1,
        "name": "verifier_abstention",
        "ensemble": {
            "replicas": 2,
            "strategy": "majority_vote",
            "selection_key": "declared_answer",
        },
        "roles": [
            {"id": "solver", "target": "solver", "system_prompt": "solve"},
            {"id": "judge", "target": "judge", "system_prompt": "judge"},
        ],
        "steps": [
            {"id": "solve", "kind": "agent", "role": "solver"},
            {"id": "verify", "kind": "verifier", "role": "judge", "output": True},
        ],
        "entry_step": "solve",
        "transitions": [{"source": "solve", "target": "verify"}],
    }
    executor = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"solver": FakeRuntime(lambda task: f"Final answer: {task.sample_id}")},
        verifiers={"judge": RejectingVerifier()},
    )

    with pytest.raises(WorkflowExecutionError, match="non-abstaining"):
        executor.run(WorkflowInput(input_id="all-rejected", problem="P"))


def test_executor_rejects_runtime_reordering() -> None:
    class ReorderingRuntime(FakeRuntime):
        def run_batch(self, tasks: Sequence[AgentTask]) -> list[AgentResult]:
            return list(reversed(super().run_batch(tasks)))

    raw = {
        "version": 1,
        "name": "ordered",
        "roles": [{"id": "solver", "target": "runtime", "system_prompt": "solve"}],
        "steps": [{"id": "answer", "kind": "agent", "role": "solver", "output": True}],
        "entry_step": "answer",
        "transitions": [],
    }
    executor = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"runtime": ReorderingRuntime(lambda task: task.prompt)},
        verifiers={},
    )

    with pytest.raises(WorkflowExecutionError, match="changed task identity"):
        executor.run_batch(
            [
                WorkflowInput(input_id="a", problem="A"),
                WorkflowInput(input_id="b", problem="B"),
            ]
        )


def test_task_payload_reaches_runtime_dedicated_field_but_not_prompt_or_result() -> None:
    raw = {
        "version": 1,
        "name": "private-environment-payload",
        "roles": [
            {
                "id": "solver",
                "target": "runtime",
                "system_prompt": "solve",
                "input_template": "{problem}\npublic={metadata}",
            }
        ],
        "steps": [{"id": "answer", "kind": "agent", "role": "solver", "output": True}],
        "entry_step": "answer",
        "transitions": [],
    }
    runtime = FakeRuntime(lambda task: task.prompt)
    executor = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"runtime": runtime},
        verifiers={},
    )

    result = executor.run(
        WorkflowInput(
            input_id="robot-one",
            problem="pick up the cup",
            metadata={"source": "public"},
            task_payload={"scene_seed": 42},
        )
    )

    task = runtime.batches[0][0]
    assert "scene_seed" not in task.prompt
    assert task.task_payload == {"scene_seed": 42}
    assert "task_payload" not in task.metadata
    assert task.metadata["source"] == "public"
    assert isinstance(result.output, AgentResult)
    assert "task_payload" not in result.output.metadata
    assert "scene_seed" not in result.output.metadata


def test_task_payload_does_not_reach_verifier_request() -> None:
    raw = {
        "version": 1,
        "name": "private-payload-verification",
        "roles": [
            {"id": "solver", "target": "runtime", "system_prompt": "solve"},
            {
                "id": "judge",
                "target": "judge",
                "system_prompt": "judge",
                "input_template": "{problem}\n{candidate}",
                "output_format": "json",
            },
        ],
        "steps": [
            {"id": "solve", "kind": "agent", "role": "solver"},
            {"id": "verify", "kind": "verifier", "role": "judge", "output": True},
        ],
        "entry_step": "solve",
        "transitions": [{"source": "solve", "target": "verify"}],
    }
    runtime = FakeRuntime(lambda _task: "public candidate")
    verifier = FakeVerifier(lambda _request: "pass")
    executor = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(raw)),
        runtimes={"runtime": runtime},
        verifiers={"judge": verifier},
    )

    executor.run(
        WorkflowInput(
            input_id="private-one",
            problem="public problem",
            task_payload={"answer": "PRIVATE-VERIFIER-CANARY"},
        )
    )

    request = verifier.batches[0][0]
    assert "PRIVATE-VERIFIER-CANARY" not in repr(request)
    assert "task_payload" not in request.metadata


def test_workflow_result_retains_agent_and_verifier_trajectory_ids() -> None:
    class ProvenanceVerifier(FakeVerifier):
        def verify_batch(self, requests: Sequence[VerificationRequest]) -> list[VerificationResult]:
            self.batches.append(tuple(requests))
            return [
                VerificationResult(
                    request_id=request.request_id,
                    verdict="pass",
                    candidate=request.candidate,
                    candidate_ref=request.candidate_ref,
                    agent_result=AgentResult(
                        task_id=request.request_id,
                        final_text='{"verdict":"pass"}',
                    ),
                )
                for request in requests
            ]

    workflow = Workflow.from_config(parse_workflow_config(_branching_config()))
    runtime = FakeRuntime(lambda task: task.prompt)
    result = WorkflowExecutor(
        workflow,
        runtimes={"runtime": runtime},
        verifiers={"judge": ProvenanceVerifier(lambda _request: "pass")},
    ).run(WorkflowInput(input_id="trace", problem="P"))

    assert result.trajectory_refs == tuple(dict.fromkeys(result.trajectory_refs))
    assert len(result.trajectory_refs) == 3
    assert any(":verify:" in task_id for task_id in result.trajectory_refs)


def test_memory_completion_hook_must_return_bool() -> None:
    class InvalidCompletionMemory:
        def render_context(self, *_args: object, **_kwargs: object) -> str:
            return ""

        def task_metadata(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {}

        def observe_output(self, *_args: object, **_kwargs: object) -> None:
            return None

        def is_complete(self, *_args: object, **_kwargs: object) -> str:
            return "yes"

        def provenance_steps(self, *_args: object, **_kwargs: object) -> tuple[()]:
            return ()

    executor = WorkflowExecutor(
        Workflow.from_config(parse_workflow_config(_branching_config())),
        runtimes={"runtime": FakeRuntime(lambda _task: "answer")},
        verifiers={"judge": FakeVerifier(lambda _request: "pass")},
        memory=InvalidCompletionMemory(),  # type: ignore[arg-type]
    )

    with pytest.raises(WorkflowExecutionError, match="is_complete.*bool"):
        executor.run(WorkflowInput(input_id="one", problem="problem"))


def test_memory_resume_cannot_change_configured_execution_identity() -> None:
    class MismatchedResumeMemory:
        def resume_steps(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[WorkflowResumeStep, ...]:
            return (
                WorkflowResumeStep(
                    step_id="solve",
                    iteration=1,
                    output=AgentResult(task_id="different", final_text="invented"),
                ),
            )

        def provenance_steps(self, *_args: object, **_kwargs: object) -> tuple[()]:
            return ()

    workflow = Workflow.from_config(
        parse_workflow_config(
            {
                "version": 1,
                "name": "memory_resume",
                "roles": [{"id": "solver", "target": "runtime", "system_prompt": "solve"}],
                "steps": [{"id": "solve", "kind": "agent", "role": "solver", "output": True}],
                "entry_step": "solve",
                "transitions": [],
            }
        )
    )
    executor = WorkflowExecutor(
        workflow,
        runtimes={"runtime": FakeRuntime(lambda _task: "answer")},
        verifiers={},
        memory=MismatchedResumeMemory(),  # type: ignore[arg-type]
    )

    with pytest.raises(WorkflowExecutionError, match="output identity"):
        executor.run(WorkflowInput(input_id="one", problem="problem"))


def test_memory_finalizer_cannot_invent_an_output_outside_workflow_history() -> None:
    class InventingMemory:
        def render_context(self, *_args: object, **_kwargs: object) -> str:
            return ""

        def task_metadata(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {}

        def observe_output(self, *_args: object, **_kwargs: object) -> None:
            return None

        def is_complete(self, *_args: object, **_kwargs: object) -> bool:
            return False

        def provenance_steps(self, *_args: object, **_kwargs: object) -> tuple[()]:
            return ()

        def finalize_output(self, *_args: object, **_kwargs: object) -> AgentResult:
            return AgentResult(task_id="unrelated", final_text="invented")

    workflow = Workflow.from_config(
        parse_workflow_config(
            {
                "version": 1,
                "name": "memory_provenance",
                "roles": [{"id": "solver", "target": "runtime", "system_prompt": "solve"}],
                "steps": [{"id": "solve", "kind": "agent", "role": "solver", "output": True}],
                "entry_step": "solve",
                "transitions": [],
            }
        )
    )
    executor = WorkflowExecutor(
        workflow,
        runtimes={"runtime": FakeRuntime(lambda _task: "answer")},
        verifiers={},
        memory=InventingMemory(),  # type: ignore[arg-type]
    )

    with pytest.raises(WorkflowExecutionError, match="does not match Workflow history"):
        executor.run(WorkflowInput(input_id="one", problem="problem"))


def test_memory_provenance_cannot_impersonate_a_configured_step() -> None:
    class ImpersonatingMemory:
        def render_context(self, *_args: object, **_kwargs: object) -> str:
            return ""

        def task_metadata(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {}

        def observe_output(self, *_args: object, **_kwargs: object) -> None:
            return None

        def is_complete(self, *_args: object, **_kwargs: object) -> bool:
            return False

        def provenance_steps(
            self,
            workflow_input: WorkflowInput,
            *,
            branch_index: int,
        ) -> tuple[StepResult, ...]:
            return (
                StepResult(
                    input_id=workflow_input.input_id,
                    step_id="solve",
                    role="solver",
                    iteration=2,
                    branch_index=branch_index,
                    output=AgentResult(task_id="invented", final_text="invented"),
                ),
            )

        def finalize_output(self, *_args: object, **kwargs: object) -> object:
            return kwargs["output"]

    workflow = Workflow.from_config(
        parse_workflow_config(
            {
                "version": 1,
                "name": "memory_provenance",
                "roles": [{"id": "solver", "target": "runtime", "system_prompt": "solve"}],
                "steps": [{"id": "solve", "kind": "agent", "role": "solver", "output": True}],
                "entry_step": "solve",
                "transitions": [],
            }
        )
    )
    executor = WorkflowExecutor(
        workflow,
        runtimes={"runtime": FakeRuntime(lambda _task: "answer")},
        verifiers={},
        memory=ImpersonatingMemory(),  # type: ignore[arg-type]
    )

    with pytest.raises(WorkflowExecutionError, match="must not impersonate"):
        executor.run(WorkflowInput(input_id="one", problem="problem"))
