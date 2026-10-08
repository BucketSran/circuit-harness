from __future__ import annotations

import copy
import hashlib
import inspect
import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import asdict, astuple
from typing import Any

import pytest

from alphaapollo.reasoning._immutable import FrozenDict, _thaw_json
from alphaapollo.reasoning.runtime import AgentResult, AgentTask
from alphaapollo.reasoning.verification import (
    AgentVerifier,
    AgentVerifierConfig,
    DeterministicCheck,
    RecomputeResult,
    VerificationRequest,
    VerificationResult,
    WitnessRecorder,
    request_fingerprint,
)


def test_agent_records_recursively_freeze_metadata_and_isolate_aliases() -> None:
    nested = {"source": {"name": "original"}, "tags": ["math", {"level": 2}]}

    task = AgentTask("task", "system", "prompt", metadata=nested)
    result = AgentResult("task", "answer", metadata=nested)
    nested["source"]["name"] = "changed"
    nested["tags"].append("late")

    for metadata in (task.metadata, result.metadata):
        assert isinstance(metadata, FrozenDict)
        assert metadata["source"]["name"] == "original"
        assert metadata["tags"] == ("math", {"level": 2})
        with pytest.raises(TypeError, match="do not support mutation"):
            metadata["new"] = True  # type: ignore[index]
        with pytest.raises(TypeError, match="do not support mutation"):
            metadata["source"]["name"] = "mutated"  # type: ignore[index]


def test_agent_task_private_payload_is_keyword_only_isolated_and_hidden_from_repr() -> None:
    payload = {
        "gold_answer": "PRIVATE-ANSWER-CANARY",
        "nested": {"items": [1]},
    }

    task = AgentTask("task", "system", "prompt", task_payload=payload)
    payload["nested"]["items"].append(2)

    parameter = inspect.signature(AgentTask).parameters["task_payload"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert task.metadata == {}
    assert task.task_payload == {
        "gold_answer": "PRIVATE-ANSWER-CANARY",
        "nested": {"items": (1,)},
    }
    assert "PRIVATE-ANSWER-CANARY" not in repr(task)
    with pytest.raises(TypeError, match="do not support mutation"):
        task.task_payload["nested"]["items"] = ()


def test_agent_task_migrates_legacy_metadata_task_payload() -> None:
    task = AgentTask(
        "task",
        "system",
        "prompt",
        metadata={
            "source": "legacy-caller",
            "task_payload": {"answer": "PRIVATE-LEGACY-CANARY"},
        },
    )

    assert task.metadata == {"source": "legacy-caller"}
    assert task.task_payload == {"answer": "PRIVATE-LEGACY-CANARY"}
    assert "PRIVATE-LEGACY-CANARY" not in repr(task)


def test_agent_task_rejects_conflicting_private_payload_sources() -> None:
    with pytest.raises(ValueError, match="both explicitly and through metadata"):
        AgentTask(
            "task",
            "system",
            "prompt",
            metadata={"task_payload": {"answer": "legacy"}},
            task_payload={"answer": "explicit"},
        )


@pytest.mark.parametrize(
    ("record", "attribute"),
    (
        (
            VerificationRequest(
                "request", "problem", "candidate", metadata={"nested": {"items": [1, 2]}}
            ),
            "metadata",
        ),
        (
            VerificationResult(
                "request", "pass", "candidate", details={"nested": {"items": [1, 2]}}
            ),
            "details",
        ),
        (RecomputeResult(True, details={"nested": {"items": [1, 2]}}), "details"),
        (DeterministicCheck(True, details={"nested": {"items": [1, 2]}}), "details"),
    ),
)
def test_verification_records_recursively_freeze_audit_mappings(
    record: object,
    attribute: str,
) -> None:
    audit = getattr(record, attribute)

    assert isinstance(audit, FrozenDict)
    assert audit["nested"]["items"] == (1, 2)
    with pytest.raises(TypeError, match="do not support mutation"):
        audit["nested"]["items"] = ()


def test_frozen_json_mapping_supports_json_copy_deepcopy_pickle_and_thaw() -> None:
    metadata = AgentTask(
        "task",
        "system",
        "prompt",
        metadata={"nested": {"items": [1, True, None]}},
    ).metadata

    expected = {"nested": {"items": [1, True, None]}}
    assert json.loads(json.dumps(metadata)) == expected
    assert copy.copy(metadata) is metadata
    assert copy.deepcopy(metadata) is metadata
    restored = pickle.loads(pickle.dumps(metadata))
    assert isinstance(restored, FrozenDict)
    assert restored == metadata
    with pytest.raises(TypeError, match="do not support mutation"):
        restored["nested"]["items"] = ()

    thawed = _thaw_json(metadata)
    assert thawed == expected
    assert isinstance(thawed["nested"]["items"], list)
    thawed["nested"]["items"].append(2)
    assert metadata["nested"]["items"] == (1, True, None)


def test_frozen_json_mapping_preserves_dataclass_asdict_compatibility() -> None:
    task = AgentTask(
        "task",
        "system",
        "prompt",
        metadata={"nested": {"items": [1, 2]}},
    )

    projected = asdict(task)

    assert projected["metadata"] == {"nested": {"items": (1, 2)}}
    assert isinstance(projected["metadata"], FrozenDict)
    with pytest.raises(TypeError, match="do not support mutation"):
        projected["metadata"]["nested"] = {}
    with pytest.raises(TypeError, match="reinitialization"):
        task.metadata.__init__({"replacement": True})


@pytest.mark.parametrize(
    "record",
    (
        AgentResult("task", "answer", metadata={"nested": [1, 2]}),
        VerificationRequest("request", "problem", "candidate", metadata={"nested": [1, 2]}),
        VerificationResult("request", "pass", "candidate", details={"nested": [1, 2]}),
        RecomputeResult(True, details={"nested": [1, 2]}),
        DeterministicCheck(True, details={"nested": [1, 2]}),
    ),
)
def test_frozen_public_records_support_standard_dataclass_projection(record: object) -> None:
    assert json.loads(json.dumps(asdict(record)))
    assert astuple(record)


@pytest.mark.parametrize(
    "bad_metadata",
    (
        {1: "non-string key"},
        {"unsupported": {1, 2}},
        {"not_finite": float("nan")},
        {"not_finite": float("inf")},
    ),
)
def test_reasoning_records_reject_non_json_audit_values(
    bad_metadata: Mapping[object, object],
) -> None:
    with pytest.raises(ValueError, match="strings|JSON-compatible|NaN or infinity"):
        AgentTask(  # type: ignore[arg-type]
            "task",
            "system",
            "prompt",
            metadata=bad_metadata,
        )
    with pytest.raises(ValueError, match="strings|JSON-compatible|NaN or infinity"):
        VerificationResult(  # type: ignore[arg-type]
            "request",
            "pass",
            "candidate",
            details=bad_metadata,
        )


def test_reasoning_records_reject_recursive_audit_containers() -> None:
    recursive: dict[str, Any] = {}
    recursive["self"] = recursive

    with pytest.raises(ValueError, match="recursive"):
        VerificationRequest("request", "problem", "candidate", metadata=recursive)


class _MemoryWitnessStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def put_blob(self, content: bytes, *, type_: str, created_by: str) -> dict[str, str]:
        digest = hashlib.sha256(content).hexdigest()
        self.blobs[digest] = content
        return {
            "id": digest,
            "hash": digest,
            "type": type_,
            "created_by": created_by,
        }

    def get_blob(self, reference: Mapping[str, str]) -> bytes:
        return self.blobs[reference["id"]]


def test_request_fingerprint_and_certificate_readback_are_stable_after_construction() -> None:
    source = {"constraints": ["integer"], "provenance": {"dataset": "test"}}
    request = VerificationRequest(
        "request",
        "Compute 40 + 2.",
        "42",
        metadata=source,
    )
    before = request_fingerprint(request)
    recorder = WitnessRecorder(_MemoryWitnessStore(), checker_id="test.stable-request")
    recompute = RecomputeResult(
        True,
        recompute_log=("computed exact integer sum",),
        trust_level=2,
        false_positive_risk=0.0,
    )
    witness = recorder.record(request, recompute)

    source["constraints"].append("changed")
    source["provenance"]["dataset"] = "changed"

    assert request_fingerprint(request) == before
    assert witness.request_sha256 == before
    assert recorder.read(witness, request=request)["candidate"]["request_sha256"] == before
    with pytest.raises(TypeError, match="do not support mutation"):
        request.metadata["provenance"]["dataset"] = "changed"


class _CapturingRuntime:
    def __init__(self) -> None:
        self.tasks: list[AgentTask] = []

    def run_batch(self, tasks: Sequence[AgentTask]) -> list[AgentResult]:
        self.tasks.extend(tasks)
        return [
            AgentResult(task_id=task.task_id, final_text='{"verdict":"pass"}') for task in tasks
        ]


def test_agent_verifier_thaws_request_metadata_at_the_agent_task_boundary() -> None:
    runtime = _CapturingRuntime()
    verifier = AgentVerifier(
        runtime,  # type: ignore[arg-type]
        AgentVerifierConfig(
            system_prompt="Judge the candidate.",
            input_template="{metadata}",
            role="judge",
            model="fake",
            tools=(),
            output_format="json",
        ),
    )
    request = VerificationRequest(
        "request",
        "problem",
        "candidate",
        metadata={"nested": {"items": [1, 2]}},
    )

    result = verifier.verify(request)

    assert result.verdict == "pass"
    assert json.loads(runtime.tasks[0].prompt) == {"nested": {"items": [1, 2]}}
    assert runtime.tasks[0].metadata["nested"]["items"] == (1, 2)
    assert runtime.tasks[0].metadata["verification_request_id"] == "request"
