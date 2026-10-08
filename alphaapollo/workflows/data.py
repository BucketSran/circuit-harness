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

"""Prepared inputs, scoring-data alignment, trajectory projection, and persistence.

This is both ends of the pipeline with execution in between, and they stay
together because they are the two halves of one boundary: what the public
projection is allowed to contain on the way in, and what a durable artifact is
allowed to claim on the way out.  ``_PUBLIC_TASK_METADATA_KEYS`` and
``ScoredTask.public_view`` are only legible next to the loader that produced the
row they narrow.  Read it by section:

1. public input loading   -- prepared dataset -> ``WorkflowInput``, field allowlist
2. durable persistence    -- ``WorkflowResult`` -> ``workflow_results``/``trajectories``
3. cells and private gold -- binding gold after execution is planned, cell expansion
4. trajectory evidence    -- ``WorkflowResult`` -> the ``traj.jsonl`` metrics envelope
"""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphaapollo.common.grader import require_offline_grader
from alphaapollo.common.trajectory.metrics import TrajectoryEvidenceError
from alphaapollo.common.trajectory.schemas import (
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryEventType,
)
from alphaapollo.data_preprocess import read_prepared
from alphaapollo.reasoning.runtime import AgentResult, AgentTurn
from alphaapollo.reasoning.verification import VerificationResult
from alphaapollo.workflows.config import ConfigError, RunConfig
from alphaapollo.workflows.records import (
    ScoredTask,
    StepResult,
    WorkflowInput,
    WorkflowResult,
)

__all__ = [
    "Cell",
    "InputDataError",
    "IncrementalWorkflowRecorder",
    "align_public_inputs_with_private_gold",
    "expand_cells",
    "load_prepared_inputs",
    "persist_results",
    "validate_inputs",
    "write_trajectory_projection",
]

_WORKFLOW_RESULT_SCHEMA_VERSION = 1
_AGENT_TRAJECTORY_SCHEMA_VERSION = 1
_INCREMENTAL_STEP_SCHEMA_VERSION = 1


class InputDataError(ValueError):
    """A prepared dataset does not satisfy its explicit public-field mapping."""


class IncrementalWorkflowRecorder:
    """Atomically retain every completed step before a later step can fail.

    Each checkpoint is self-contained: it carries the full candidate/result and
    the producing Agent trajectory when one exists. The canonical aggregate
    files are still written by :func:`persist_results` after successful
    completion; these checkpoints are the failure-recovery evidence path.
    """

    _PREFIX = "workflow-step-"
    _MARKER = ".alphaapollo-workflow-run"

    def __init__(self, directory: Path, *, resume: bool = False) -> None:
        if not isinstance(directory, Path):
            raise TypeError("directory must be a Path")
        if not isinstance(resume, bool):
            raise TypeError("resume must be a bool")
        if directory.exists():
            if not directory.is_dir():
                raise ConfigError(f"workflow output path is not a directory: {directory}")
            entries = tuple(directory.iterdir())
            if entries and not resume:
                raise ConfigError(
                    "workflow output directory already contains evidence; choose a new "
                    f"directory instead of overwriting {directory}"
                )
        else:
            directory.mkdir(parents=True)
            entries = ()
        marker = directory / self._MARKER
        if not resume:
            try:
                with marker.open("x", encoding="utf-8") as handle:
                    handle.write("1\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as exc:
                raise ConfigError(
                    f"workflow output directory was claimed concurrently: {directory}"
                ) from exc
        elif not marker.exists():
            with marker.open("x", encoding="utf-8") as handle:
                handle.write("1\n")
                handle.flush()
                os.fsync(handle.fileno())
        self.directory = directory
        self._sequence = max(
            (
                number
                for path in entries
                if (number := self._checkpoint_number(path.name)) is not None
            ),
            default=0,
        )

    def record(self, step: StepResult) -> None:
        if not isinstance(step, StepResult):
            raise TypeError("step must be a StepResult")
        agent = step.output if isinstance(step.output, AgentResult) else step.output.agent_result
        payload = {
            "incremental_step_schema_version": _INCREMENTAL_STEP_SCHEMA_VERSION,
            "sequence": self._sequence + 1,
            "step": {
                "input_id": step.input_id,
                "step_id": step.step_id,
                "role": step.role,
                "iteration": step.iteration,
                "branch_index": step.branch_index,
                "output": _workflow_output_record(step.output, include_payload=True),
            },
            "agent_trajectory": None if agent is None else _agent_trajectory_record(agent),
        }
        encoded = _encode_json(payload)
        sequence = self._sequence + 1
        path = self.directory / f"{self._PREFIX}{sequence:06d}.json"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self._sequence = sequence

    @classmethod
    def _checkpoint_number(cls, name: str) -> int | None:
        if not name.startswith(cls._PREFIX) or not name.endswith(".json"):
            return None
        raw = name[len(cls._PREFIX) : -len(".json")]
        return int(raw) if raw.isdigit() else None


# ---- 1. Public input loading ----------------------------------------------


def load_prepared_inputs(config: RunConfig) -> tuple[WorkflowInput, ...]:
    """Load the public prepared dataset through ``RunConfig``'s field whitelist."""

    dataset = config.dataset
    records = _read_input_records(dataset.path, dataset.format, dataset.split)
    task_payloads: Mapping[str, Mapping[str, Any]] = {}
    if dataset.task_payload_path is not None:
        # A task-payload path already names one concrete private split file.
        # Prepared private rows intentionally have no public ``split`` column.
        private_records = _read_input_records(
            dataset.task_payload_path,
            dataset.task_payload_format or dataset.format,
            None,
        )
        task_payloads = _task_payloads(private_records, config)
    return _records_to_inputs(records, config, task_payloads=task_payloads)


def _read_input_records(path: Path, format_: str, split: str | None) -> list[Mapping[str, Any]]:
    if format_ == "json":
        return _read_json_records(path, split)
    if format_ == "jsonl":
        return _read_jsonl_records(path, split)
    return _read_parquet_records(path, split)


def _read_json_records(path: Path, split: str | None) -> list[Mapping[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputDataError(f"could not read prepared JSON dataset {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputDataError(f"invalid prepared JSON dataset {path}: {exc}") from exc
    selected_top_level_split = split is not None and isinstance(raw, Mapping)
    if selected_top_level_split:
        if split not in raw:
            raise InputDataError(f"prepared JSON dataset has no split {split!r}")
        raw = raw[split]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise InputDataError("prepared JSON dataset must contain an array of records")
    records = _record_mappings(raw, where=f"prepared JSON dataset {path}")
    return records if selected_top_level_split else _filter_split(records, split)


def _read_jsonl_records(path: Path, split: str | None) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputDataError(f"could not read prepared JSONL dataset {path}: {exc}") from exc
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputDataError(
                f"invalid JSON on line {line_number} of prepared dataset {path}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise InputDataError(f"line {line_number} of prepared dataset {path} must be an object")
        records.append(value)
    return _filter_split(records, split)


def _read_parquet_records(path: Path, split: str | None) -> list[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise InputDataError("Parquet input requires the optional 'pyarrow' dependency") from exc
    try:
        records = parquet.read_table(path).to_pylist()
    except Exception as exc:
        raise InputDataError(f"could not read prepared Parquet dataset {path}: {exc}") from exc
    return _filter_split(_record_mappings(records, where=f"prepared Parquet dataset {path}"), split)


def _record_mappings(values: Sequence[object], *, where: str) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise InputDataError(f"{where} record {index} must be an object")
        records.append(value)
    return records


def _filter_split(records: list[Mapping[str, Any]], split: str | None) -> list[Mapping[str, Any]]:
    if split is None:
        return records
    missing = [index for index, record in enumerate(records) if "split" not in record]
    if missing:
        raise InputDataError(
            f"prepared dataset declares split {split!r}, but records have no complete "
            f"split field (first missing record: {missing[0]})"
        )
    selected = [record for record in records if record.get("split") == split]
    if not selected:
        raise InputDataError(f"prepared dataset contains no records for split {split!r}")
    return selected


def _records_to_inputs(
    records: Sequence[Mapping[str, Any]],
    config: RunConfig,
    *,
    task_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[WorkflowInput, ...]:
    dataset = config.dataset
    payloads = task_payloads or {}
    inputs: list[WorkflowInput] = []
    for index, record in enumerate(records):
        required = {dataset.input_key, *dataset.metadata_keys}
        if dataset.id_key is not None:
            required.add(dataset.id_key)
        missing = sorted(key for key in required if key not in record)
        if missing:
            raise InputDataError(f"prepared dataset record {index} is missing fields: {missing}")
        problem = record[dataset.input_key]
        if not isinstance(problem, str) or not problem.strip():
            raise InputDataError(
                f"prepared dataset record {index} field {dataset.input_key!r} "
                "must be a non-empty string"
            )
        raw_id = index if dataset.id_key is None else record[dataset.id_key]
        if raw_id is None or (
            isinstance(raw_id, (Mapping, Sequence)) and not isinstance(raw_id, (str, bytes))
        ):
            raise InputDataError(f"prepared dataset record {index} id must be a string or scalar")
        input_id = str(raw_id)
        if not input_id.strip():
            raise InputDataError(f"prepared dataset record {index} has an empty id")
        # Explicit anti-leak boundary: labels, answers, and reference solutions
        # are discarded unless a field was deliberately whitelisted as metadata.
        metadata = {key: record[key] for key in dataset.metadata_keys}
        try:
            task_payload = payloads.get(input_id, {})
            if dataset.task_payload_path is not None and input_id not in payloads:
                raise InputDataError(
                    f"prepared dataset record {index} has no task payload for id {input_id!r}"
                )
            if dataset.task_payload_envelope == "robot_task":
                # Per #260 the private env_payload names the robot benchmark and
                # simulator checkout itself; source_id stays dataset provenance
                # and must not be treated as an alias for either. Both identity
                # fields are required non-empty at load time: tolerating an
                # empty value only defers the failure to episode init, where it
                # aborts the whole batch instead of one record.
                payload = dict(task_payload)
                benchmark = payload.pop("benchmark", None)
                if not isinstance(benchmark, str) or not benchmark.strip():
                    raise InputDataError(
                        f"prepared robotics record {index} needs a non-empty "
                        "'benchmark' in its env_payload"
                    )
                environment_version = payload.pop("environment_version", "")
                if not isinstance(environment_version, str) or not environment_version.strip():
                    raise InputDataError(
                        f"prepared robotics record {index} needs a non-empty "
                        "'environment_version' in its env_payload"
                    )
                task_payload = {
                    "instruction": problem,
                    "benchmark": benchmark.strip(),
                    "environment_version": environment_version,
                    "backend_metadata": payload,
                }
            inputs.append(
                WorkflowInput(
                    input_id=input_id,
                    problem=problem,
                    metadata=metadata,
                    task_payload=task_payload,
                )
            )
        except (TypeError, ValueError) as exc:
            raise InputDataError(f"invalid prepared dataset record {index}: {exc}") from exc
    return validate_inputs(inputs)


def _task_payloads(
    records: Sequence[Mapping[str, Any]],
    config: RunConfig,
) -> Mapping[str, Mapping[str, Any]]:
    """Join private envelope rows without inferring a schema from private data."""

    dataset = config.dataset
    assert dataset.id_key is not None
    payloads: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        missing = [key for key in (dataset.id_key, dataset.task_payload_key) if key not in record]
        if missing:
            raise InputDataError(f"task payload record {index} is missing fields: {missing}")
        raw_id = record[dataset.id_key]
        if raw_id is None or (
            isinstance(raw_id, (Mapping, Sequence)) and not isinstance(raw_id, (str, bytes))
        ):
            raise InputDataError(f"task payload record {index} id must be a string or scalar")
        input_id = str(raw_id)
        if not input_id.strip():
            raise InputDataError(f"task payload record {index} has an empty id")
        if input_id in payloads:
            raise InputDataError(f"duplicate task payload id {input_id!r}")
        payload = _decode_task_payload(
            record[dataset.task_payload_key],
            envelope=dataset.task_payload_envelope,
            index=index,
            key=dataset.task_payload_key,
        )
        payloads[input_id] = dict(payload)
    return payloads


# Envelopes this decoder can actually shape.  Kept separate from
# ``DatasetConfig``'s accepted set so a new envelope cannot reach the loader
# without a decoder for it.
_DECODABLE_ENVELOPES = frozenset({"direct", "robot_task"})


def _decode_task_payload(
    payload: Any,
    *,
    envelope: str,
    index: int,
    key: str,
) -> Mapping[str, Any]:
    """Decode the explicit private-row envelope at its single extension point."""

    # Guarded by DatasetConfig; defensive for direct callers.  ``robot_task``
    # shares ``direct``'s JSON-object wire shape and is reshaped against the
    # public problem text in ``_records_to_inputs``.
    if envelope not in _DECODABLE_ENVELOPES:
        raise InputDataError(f"unsupported task payload envelope {envelope!r}")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise InputDataError(
                f"task payload record {index} field {key!r} contains invalid JSON: {exc}"
            ) from exc
    if not isinstance(payload, Mapping):
        raise InputDataError(
            f"task payload record {index} field {key!r} must be an object or an encoded JSON object"
        )
    return payload


def validate_inputs(inputs: Sequence[WorkflowInput]) -> tuple[WorkflowInput, ...]:
    """Validate caller-injected inputs through the same public boundary."""

    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise TypeError("inputs must be a sequence of WorkflowInput objects")
    materialized = tuple(inputs)
    for index, item in enumerate(materialized):
        if not isinstance(item, WorkflowInput):
            raise TypeError(f"inputs[{index}] must be a WorkflowInput")
    ids = [item.input_id for item in materialized]
    if len(set(ids)) != len(ids):
        raise ValueError("WorkflowInput ids must be unique")
    return materialized


# ---- 2. Durable WorkflowResult persistence --------------------------------


def persist_results(config: RunConfig, results: Sequence[WorkflowResult]) -> None:
    # ``workflow_results`` is the versioned orchestration index.  It deliberately
    # contains only output summaries and stable task-id references; complete model
    # turns have exactly one owner in ``trajectories.jsonl`` below.
    result_records = [_workflow_result_record(result) for result in results]
    trajectories = [_agent_trajectory_record(result) for result in _trajectory_results(results)]
    if config.output.format == "jsonl":
        result_name = "workflow_results.jsonl"
        result_payload = _encode_jsonl(result_records)
    else:
        result_name = "workflow_results.json"
        result_payload = _encode_json(result_records)
    trajectory_payload = _encode_jsonl(trajectories)

    # Complete both projections and JSON encodings before the first filesystem
    # write.  A malformed provider/Environment record therefore cannot leave a
    # Workflow index that points at a missing trajectory file.
    directory = config.output.directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / result_name).write_text(result_payload, encoding="utf-8")
    (directory / "trajectories.jsonl").write_text(trajectory_payload, encoding="utf-8")


def _workflow_result_record(result: WorkflowResult) -> dict[str, Any]:
    """Project one in-process result onto the stable orchestration file shape."""

    return {
        "workflow_result_schema_version": _WORKFLOW_RESULT_SCHEMA_VERSION,
        "input_id": result.input_id,
        "workflow_name": result.workflow_name,
        "status": result.status,
        "selected_step_id": result.selected_step_id,
        "selected_branch_index": result.selected_branch_index,
        "trajectory_refs": list(result.trajectory_refs),
        "output": (
            None
            if result.output is None
            else _workflow_output_record(result.output, include_payload=True)
        ),
        "steps": [_step_result_record(step) for step in result.steps],
    }


def _step_result_record(step: StepResult) -> dict[str, Any]:
    # StepResult is intentionally not reflected: adding an in-process field must
    # not silently mutate the durable file format.
    return {
        "input_id": step.input_id,
        "step_id": step.step_id,
        "role": step.role,
        "iteration": step.iteration,
        "branch_index": step.branch_index,
        "output": _workflow_output_record(step.output, include_payload=False),
    }


def _workflow_output_record(
    output: AgentResult | VerificationResult,
    *,
    include_payload: bool,
) -> dict[str, Any]:
    if isinstance(output, AgentResult):
        record: dict[str, Any] = {
            "kind": "agent",
            "task_id": output.task_id,
            "trajectory_ref": output.task_id,
            "termination_reason": output.termination_reason,
        }
        if include_payload:
            record.update(
                {
                    "final_text": output.final_text,
                    "metadata": _json_mapping(output.metadata, where="AgentResult.metadata"),
                }
            )
        return record
    agent_ref = output.agent_result.task_id if output.agent_result is not None else None
    record = {
        "kind": "verification",
        "request_id": output.request_id,
        "verdict": output.verdict,
        "candidate_ref": output.candidate_ref,
        "candidate_sha256": output.candidate_sha256,
        "agent_trajectory_ref": agent_ref,
        "trust_level": output.trust_level,
        "false_positive_risk": output.false_positive_risk,
        "witness": _witness_record(output.witness),
        "certified": output.certified,
    }
    if include_payload:
        record.update(
            {
                "candidate": output.candidate,
                "feedback": output.feedback,
                "details": _json_mapping(
                    output.details,
                    where="VerificationResult.details",
                ),
            }
        )
    return record


def _trajectory_results(results: Sequence[WorkflowResult]) -> list[AgentResult]:
    ordered: list[AgentResult] = []
    seen: set[str] = set()
    for workflow_result in results:
        for step in workflow_result.steps:
            agent_result: AgentResult | None
            if isinstance(step.output, AgentResult):
                agent_result = step.output
            elif isinstance(step.output, VerificationResult):
                agent_result = step.output.agent_result
            else:  # pragma: no cover - protected by StepResult validation
                agent_result = None
            if agent_result is not None and agent_result.task_id not in seen:
                seen.add(agent_result.task_id)
                ordered.append(agent_result)
    return ordered


def _agent_trajectory_record(result: AgentResult) -> dict[str, Any]:
    """Project the complete Agent trajectory without reflecting runtime objects.

    ``output_truncated`` is projected beside ``termination_reason`` for the same
    reason that one is: this record is the complete mirror of one
    ``AgentResult``, and the two are its two independent answers to "why did this
    stop". The per-turn ``finish_reason`` it derives from is written verbatim by
    ``_generation_response_record`` below, so a reader can always check the
    derived value against its source in the same row.

    Additive, so ``_AGENT_TRAJECTORY_SCHEMA_VERSION`` does not move: this file is
    written and never read back by this repository, and nothing resumes a run
    from it. ``RESULT_SCHEMA_VERSION`` is the constant that had to move, because
    ``result.json`` *is* resumed.
    """

    return {
        "agent_trajectory_schema_version": _AGENT_TRAJECTORY_SCHEMA_VERSION,
        "task_id": result.task_id,
        "final_text": result.final_text,
        "termination_reason": result.termination_reason,
        "output_truncated": result.output_truncated,
        "metadata": _json_mapping(result.metadata, where="AgentResult.metadata"),
        "turns": [_agent_turn_record(turn) for turn in result.turns],
    }


def _agent_turn_record(turn: AgentTurn) -> dict[str, Any]:
    return {
        "index": turn.index,
        "generation_request": _generation_request_record(turn.generation_request),
        "generation_response": _generation_response_record(turn.generation_response),
        "environment_transition": _environment_transition_record(turn.environment_transition),
    }


def _generation_request_record(request: object) -> dict[str, Any]:
    sampling = getattr(request, "sampling", None)
    if sampling is None:
        raise TypeError("generation request has no sampling record")
    return {
        "request_id": _required_string_attribute(request, "request_id"),
        "model": _required_string_attribute(request, "model"),
        "messages": _json_value(getattr(request, "messages", ())),
        "sampling": {
            "temperature": _json_value(sampling.temperature),
            "max_tokens": _json_value(sampling.max_tokens),
            "top_p": _json_value(getattr(sampling, "top_p", 1.0)),
        },
        "group_id": _json_value(getattr(request, "group_id", "")),
        "sample_id": _json_value(getattr(request, "sample_id", 0)),
        "tools": _json_value(getattr(request, "tools", ())),
        "tool_choice": _json_value(getattr(request, "tool_choice", None)),
        "provider_options": _json_mapping(
            getattr(request, "provider_options", {}),
            where="GenerationRequest.provider_options",
        ),
        "routing_key": _json_value(getattr(request, "routing_key", "")),
    }


def _generation_response_record(response: object) -> dict[str, Any]:
    return {
        "request_id": _required_string_attribute(response, "request_id"),
        "group_id": _json_value(getattr(response, "group_id", "")),
        "sample_id": _json_value(getattr(response, "sample_id", 0)),
        "content": _json_value(response.content),
        "reasoning_content": _json_value(getattr(response, "reasoning_content", None)),
        "finish_reason": _json_value(getattr(response, "finish_reason", None)),
        "tool_calls": [
            _wire_generation_tool_call(call) for call in (getattr(response, "tool_calls", ()) or ())
        ],
        "usage": _json_mapping(getattr(response, "usage", {}), where="GenerationResponse.usage"),
        "backend_metadata": _json_mapping(
            getattr(response, "backend_metadata", {}),
            where="GenerationResponse.backend_metadata",
        ),
        "prompt_token_ids": _json_value(getattr(response, "prompt_token_ids", None)),
        "response_token_ids": _json_value(getattr(response, "response_token_ids", None)),
        "response_logprobs": _json_value(getattr(response, "response_logprobs", None)),
        "provenance": _generation_provenance_record(getattr(response, "provenance", None)),
    }


def _wire_generation_tool_call(call: object) -> dict[str, Any]:
    """Render one Generation tool call in its OpenAI wire shape.

    Two consumers need the same rendering and must not drift: this module writes
    it into the durable ``trajectories.jsonl`` turn, and ``resources`` puts it in
    the assistant message replayed back to the provider. A trajectory that
    disagrees with what was actually sent is worse than either alone, so this is
    the one owner and ``resources`` imports it.
    """

    renderer = getattr(call, "to_openai_tool_call", None)
    if callable(renderer):
        rendered = renderer()
        if not isinstance(rendered, Mapping):
            raise TypeError("tool call renderer must return a mapping")
        return dict(rendered)
    if isinstance(call, Mapping):
        return dict(call)
    call_id = getattr(call, "id", None)
    name = getattr(call, "name", None)
    arguments = getattr(call, "arguments", None)
    call_type = getattr(call, "type", "function")
    if not all(isinstance(value, str) for value in (call_id, name, arguments, call_type)):
        raise TypeError("Generation response tool call is malformed")
    return {
        "id": call_id,
        "type": call_type,
        "function": {"name": name, "arguments": arguments},
    }


def _generation_provenance_record(provenance: object | None) -> dict[str, Any] | None:
    if provenance is None:
        return None
    return {
        "policy_model": _required_string_attribute(provenance, "policy_model"),
        "tokenizer_id": _required_string_attribute(provenance, "tokenizer_id"),
        "weights_version": _json_value(getattr(provenance, "weights_version", "")),
        "tokenizer_fingerprint": _json_value(getattr(provenance, "tokenizer_fingerprint", "")),
        "chat_template_fingerprint": _json_value(
            getattr(provenance, "chat_template_fingerprint", "")
        ),
        "actor": _json_value(getattr(provenance, "actor", "")),
    }


def _environment_transition_record(transition: object) -> dict[str, Any]:
    return {
        "observation": _environment_observation_record(transition.observation),
        "reward": _json_value(transition.reward),
        "done": _json_value(transition.done),
        "success": _json_value(getattr(transition, "success", None)),
        "response_format_valid": _json_value(getattr(transition, "response_format_valid", True)),
        "env_action_valid": _json_value(getattr(transition, "env_action_valid", True)),
        "termination_reason": _json_value(transition.termination_reason),
        "metadata": _json_mapping(transition.metadata, where="EnvironmentTransition.metadata"),
    }


def _environment_observation_record(observation: object) -> Any:
    """Project #195's explicit observation record without reflecting objects."""

    to_dict = getattr(observation, "to_dict", None)
    if callable(to_dict):
        projected = to_dict()
        if not isinstance(projected, Mapping):
            raise TypeError("Environment observation to_dict() must return a mapping")
        return _json_mapping(projected, where="EnvironmentObservation.to_dict()")
    return _json_value(observation)


def _witness_record(witness: object | None) -> Any:
    if witness is None or isinstance(witness, (str, int, float, bool, Mapping, list, tuple)):
        return _json_value(witness)
    reference = getattr(witness, "reference", witness)
    reference_fields = {
        name: _json_value(getattr(reference, name, None))
        for name in (
            "id",
            "location",
            "version",
            "hash",
            "type",
            "created_by",
            "contamination_flags",
        )
    }
    if reference_fields["id"] is None:
        raise TypeError(f"cannot persist witness of type {type(witness).__name__}")
    return {
        "request_sha256": _json_value(getattr(witness, "request_sha256", None)),
        "checker_id": _json_value(getattr(witness, "checker_id", None)),
        "checker_version": _json_value(getattr(witness, "checker_version", None)),
        "reference": reference_fields,
    }


def _required_string_attribute(value: object, name: str) -> str:
    result = getattr(value, name, None)
    if not isinstance(result, str):
        raise TypeError(f"{type(value).__name__}.{name} must be a string")
    return result


def _json_mapping(value: object, *, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be a mapping")
    return {str(key): _json_value(item) for key, item in value.items()}


def _encode_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _encode_jsonl(values: Sequence[object]) -> str:
    lines = [
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) for value in values
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _json_value(value: object) -> Any:
    """Normalize already-structured JSON values without object reflection."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot persist non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_value(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError(f"cannot encode non-JSON value of type {type(value).__name__}")


# ---- 3. Cells and private-gold binding ------------------------------------


@dataclass(frozen=True, slots=True)
class Cell:
    """One public/scored task bound to one runtime seed.

    ``public_input`` is the prepared public row this cell was expanded from. It
    is absent only on the programmatic ``run.run_batch(problems=...)`` path; see
    :attr:`injected` for what that absence is allowed to mean.
    """

    problem: ScoredTask
    problem_index: int
    condition: str
    seed: int
    public_input: WorkflowInput | None = None

    @property
    def injected(self) -> bool:
        """True when a caller supplied this cell's task programmatically.

        Two unrelated decisions read this one fact, so it is named rather than
        spelled ``public_input is None`` at each site:

        * ``scoring.run_cell`` has to synthesize the model's ``WorkflowInput``
          from the gold-bearing ``ScoredTask``'s public projection, because
          there is no prepared public row to send instead.
        * ``scoring.run_cell`` scores the cell even without a ``scoring:``
          block, because an injected task's ``gold_answer`` came from the
          caller rather than from a prepared private dataset that a
          ``scoring:`` block would have had to name.
        """

        return self.public_input is None


# ``PreparedExample.extra`` is a PRIVATE column (see
# ``data_preprocess.core.PRIVATE_FIELDS``). ``prepare_custom_data`` fills it
# with whatever ``metadata_keys`` an operator named, including possible
# worked solutions and other answer-bearing metadata. Copying it wholesale
# into ``ScoredTask.metadata`` would therefore hand
# the model strictly more than the old ``year`` field did, and would do it
# silently the first time someone prepares a dataset with a richer source row.
# So the crossing is an allowlist: a key is model-visible only once it is named
# here, and naming one is the moment to check that it cannot imply the answer.
_PUBLIC_TASK_METADATA_KEYS = frozenset({"year"})


def _public_task_metadata(extra: Mapping[str, object]) -> dict[str, str]:
    """Project a private prepared row's ``extra`` onto its model-visible keys."""

    return {
        key: str(extra[key])
        for key in sorted(_PUBLIC_TASK_METADATA_KEYS)
        if extra.get(key) not in (None, "")
    }


def align_public_inputs_with_private_gold(
    config: RunConfig,
    inputs: Sequence[WorkflowInput],
) -> list[tuple[ScoredTask, WorkflowInput]]:
    """Bind prepared public rows to their private gold without leaking it.

    The public ``RunConfig.dataset`` is authoritative for statement, order, and
    selected split. The private projection of the same prepared build supplies
    the gold, joined strictly on the stable ``task_uid``. Statement equality is
    checked so a changed public file can never be scored against an unrelated
    answer key.
    """

    prepared = config.scoring
    if prepared is None:
        raise ConfigError(
            "scoring requires a private prepared dataset; set scoring.dataset_root, "
            "scoring.source_id, and scoring.version"
        )
    try:
        tasks = read_prepared(
            prepared.source_id,
            prepared.version,
            prepared.dataset_root,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigError(f"could not load prepared dataset {prepared.source_id!r}: {exc}") from exc
    # Fail before any cell runs if the dataset asks for a grader this build has
    # no implementation for, or for one whose verdict the offline scorer cannot
    # produce at all; discovering either per-cell would waste the run.
    for grader_id in sorted({task.grader_id for task in tasks}):
        try:
            require_offline_grader(grader_id)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    private_rows = [
        ScoredTask(
            id=task.task_uid,
            problem=task.statement,
            gold_answer=task.answer,
            metadata=_public_task_metadata(task.extra),
            grader_id=task.grader_id,
        )
        for task in tasks
    ]
    by_task_uid = {row.id: row for row in private_rows}
    input_ids = [item.input_id for item in inputs]
    if len(set(input_ids)) != len(input_ids):
        raise ConfigError("prepared public dataset contains duplicate task_uids")
    if set(input_ids) != set(by_task_uid):
        missing_private = sorted(set(input_ids) - set(by_task_uid))
        orphaned_private = sorted(set(by_task_uid) - set(input_ids))
        raise ConfigError(
            "public/private task_uid mismatch: "
            f"missing private={missing_private}, orphaned private={orphaned_private}"
        )
    aligned: list[tuple[ScoredTask, WorkflowInput]] = []
    for item in inputs:
        private = by_task_uid[item.input_id]
        if item.problem.strip() != private.problem.strip():
            raise ConfigError(
                f"prepared dataset input {item.input_id!r} does not match private "
                f"scoring statement {private.id!r}"
            )
        # Rebuilt on purpose, and this is load-bearing rather than redundant.
        # ``scoring.run_cell`` falls back to building a ``WorkflowInput`` out of
        # this task when no public input was injected, so whatever sits in these
        # fields can reach the model's process. Carrying the public ``problem``
        # (already checked equal above) means that path can only ever surface
        # public text; ``metadata`` was already narrowed to the allowlist above.
        # Collapsing this into ``private`` would look like a simplification and
        # would open a leak.
        aligned.append(
            (
                ScoredTask(
                    id=private.id,
                    problem=item.problem,
                    gold_answer=private.gold_answer,
                    metadata=private.metadata,
                    grader_id=private.grader_id,
                ),
                item,
            )
        )
    return aligned


def expand_cells(
    task_rows: Sequence[tuple[ScoredTask, WorkflowInput | None]],
    *,
    conditions: Sequence[str],
    k: int,
    seed_base: int,
    shard: tuple[int, int] | None = None,
    reverse: bool = False,
) -> list[Cell]:
    """Expand bound rows into the full ``problems x conditions x k`` cell list.

    ``reverse`` is applied before sharding so a helper instance walking the
    cells tail-first still covers a disjoint set from the primary instance
    rather than re-running the same shard backwards.
    """

    cells = [
        Cell(
            problem=task,
            problem_index=index,
            condition=condition,
            seed=seed_base + sample,
            public_input=public_input,
        )
        for index, (task, public_input) in enumerate(task_rows)
        for condition in conditions
        for sample in range(k)
    ]
    if reverse:
        cells = list(reversed(cells))
    if shard is not None:
        shard_index, shard_count = shard
        if (
            isinstance(shard_index, bool)
            or isinstance(shard_count, bool)
            or not isinstance(shard_index, int)
            or not isinstance(shard_count, int)
            or shard_count < 1
            or not 0 <= shard_index < shard_count
        ):
            raise ValueError(f"invalid shard {shard!r}: need 0 <= i < n")
        cells = [
            cell for position, cell in enumerate(cells) if position % shard_count == shard_index
        ]
    return cells


# ---- 4. Trajectory evidence projection ------------------------------------


def write_trajectory_projection(
    outcome: WorkflowResult,
    path: Path,
    *,
    session_id: str,
) -> None:
    """Project canonical child turns into the established metrics evidence shape."""

    events: list[dict[str, Any]] = []
    for step in outcome.steps:
        role = "verifier" if isinstance(step.output, VerificationResult) else "solver"
        agent = step.output if isinstance(step.output, AgentResult) else step.output.agent_result
        if agent is None:
            continue
        branch_id = f"branch-{step.branch_index}"
        round_index = step.iteration - 1
        for turn in agent.turns:
            raw_usage = getattr(turn.generation_response, "usage", None)
            usage = raw_usage if isinstance(raw_usage, Mapping) else {}
            prompt_tokens, prompt_error = _token_count(usage, "prompt_tokens")
            completion_tokens, completion_error = _token_count(usage, "completion_tokens")
            usage_errors = [
                error for error in (prompt_error, completion_error) if error is not None
            ]
            if raw_usage is not None and not isinstance(raw_usage, Mapping):
                usage_errors.insert(0, "model_usage: provider usage is not a mapping")
            payload: dict[str, Any] = {
                "role": role,
                "round": round_index,
                "turn": turn.index,
                "capture_kind": "model_interaction",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "workflow_step_id": step.step_id,
                "task_id": agent.task_id,
            }
            if usage_errors:
                # Counts remain conservative integer lower bounds for the current
                # report schema, while the durable gap prevents them from being
                # mistaken for provider-reported zero usage.
                payload["event_errors"] = usage_errors
            events.append(
                _trajectory_event(
                    session_id=session_id,
                    branch_id=branch_id,
                    event_type=TrajectoryEventType.EVIDENCE_ADDED,
                    payload=payload,
                )
            )
            tool_event = _tool_event_payload(
                turn.environment_transition,
                role=role,
                round_index=round_index,
                turn_index=turn.index,
            )
            if tool_event is not None:
                event_type, payload = tool_event
                events.append(
                    _trajectory_event(
                        session_id=session_id,
                        branch_id=branch_id,
                        event_type=event_type,
                        payload=payload,
                    )
                )
    if not events:
        raise TrajectoryEvidenceError("canonical WorkflowResult contains no model turns")
    encoded = (
        "\n".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
            for event in events
        )
        + "\n"
    )
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _trajectory_event(
    *,
    session_id: str,
    branch_id: str,
    event_type: TrajectoryEventType,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": event_type.value,
        "event_id": str(uuid.uuid4()),
        "session_id": session_id,
        "branch_id": branch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": dict(payload),
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
    }


def _token_count(usage: Mapping[str, Any], key: str) -> tuple[int, str | None]:
    if key not in usage:
        return 0, f"model_usage.{key}: provider did not report this count"
    value = usage[key]
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value, None
    return 0, f"model_usage.{key}: provider returned an invalid count"


def _tool_event_payload(
    transition: object,
    *,
    role: str,
    round_index: int,
    turn_index: int,
) -> tuple[TrajectoryEventType, dict[str, Any]] | None:
    metadata = getattr(transition, "metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    request = metadata.get("tool_request")
    if not isinstance(request, Mapping):
        return None
    error = metadata.get("tool_error")
    response = metadata.get("tool_response")
    failed = isinstance(error, Mapping)
    if isinstance(response, Mapping):
        exit_code = response.get("exit_code")
        failed = failed or (
            isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0
        )
    payload: dict[str, Any] = {
        "role": role,
        "round": round_index,
        "turn": turn_index,
        "request": dict(request),
        "tool_id": request.get("tool_id"),
        "response": dict(response) if isinstance(response, Mapping) else None,
    }
    if isinstance(error, Mapping):
        payload.update(
            {
                "stage": error.get("stage"),
                "code": error.get("code"),
                "attempted": error.get("attempted"),
            }
        )
        if request.get("tool_id") is None and error.get("stage") == "parse":
            payload["canonicalization_status"] = "rejected_before_resolution"
    return (
        TrajectoryEventType.TOOL_FAILED if failed else TrajectoryEventType.TOOL_CALLED,
        payload,
    )
