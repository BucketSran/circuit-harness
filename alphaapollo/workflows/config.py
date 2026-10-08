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

"""Strict, versioned configuration records for Workflow execution.

Workflow presets and runnable recipes deliberately use different records:

* :class:`WorkflowConfig` is a pure execution topology.  It cannot name a
  dataset, backend, Environment, or output directory.
* :class:`RunConfig` binds one validated topology to prepared input data and
  concrete resources.

Both loaders reject unknown keys.  Configuration is data only: transitions use
a closed condition vocabulary and prompt templates use simple named
placeholders.  Python expressions, imports, attribute traversal, and ``eval``
are never accepted.

The records, their parsers, and the graph checks stay in one module on purpose.
Two config schemas once both claimed the ``evaluation:`` key and the two entry
points scored against different gold; the fix was one recipe with one authority,
and splitting the record from the parser that is the only thing allowed to build
it would reopen the seam this package exists to close.  Read it by section:

1. records          -- the immutable ``*Config`` dataclasses and their invariants
2. loading          -- reading a mapping and resolving a Hydra defaults list
3. parsers          -- mapping -> record, rejecting unknown keys
4. graph validation -- routing, reachability, and bounded-cycle checks
5. primitives       -- frozen JSON values and the small ``_require_*`` guards
6. repeated-run     -- digest, per-cell binding, and the condition label
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import Any, cast

from alphaapollo.common.prompts import PromptConfigError, resolve_prompt

__all__ = [
    "CONFIG_VERSION",
    "ConfigError",
    "DatasetConfig",
    "ExecutionConfig",
    "EnsembleConfig",
    "MemoryConfig",
    "OutputConfig",
    "ResourceConfig",
    "RoleConfig",
    "RunConfig",
    "ScoringConfig",
    "StepConfig",
    "TransitionConfig",
    "WorkflowConfig",
    "compose_config_mapping",
    "config_digest",
    "execution_label",
    "load_run_config",
    "load_workflow_config",
    "parse_run_config",
    "parse_workflow_config",
    "read_config_mapping",
]

CONFIG_VERSION = 1
_STEP_KINDS = frozenset({"agent", "verifier"})
_TRANSITION_CONDITIONS = frozenset({"always", "passed", "failed", "inconclusive", "not_passed"})
_AGENT_VERIFIER_OUTPUT_FORMATS = frozenset({"json", "verdict-line"})
_ENSEMBLE_STRATEGIES = frozenset({"first", "majority_vote"})
_ENSEMBLE_SELECTION_KEYS = frozenset({"exact_text", "declared_answer"})
_DATASET_FORMATS = frozenset({"json", "jsonl", "parquet"})
# ``direct`` is the v1 wire shape: decode the selected field as a JSON object
# and pass that object through unchanged.  ``robot_task`` is the canonical
# robotics envelope validated at the loader boundary.  The explicit
# discriminator is kept so a further envelope can be added there without
# guessing a schema from private data.
_TASK_PAYLOAD_ENVELOPES = frozenset({"direct", "robot_task"})
_OUTPUT_FORMATS = frozenset({"json", "jsonl"})
_MAX_LOOP_ITERATIONS = 100
_MAX_SEMANTIC_ROUTE_STATES = 100_000
_MAX_ENSEMBLE_REPLICAS = 64
_TEMPLATE_FIELDS = frozenset(
    {
        "branch_index",
        "candidate",
        "feedback",
        "input_id",
        "iteration",
        "metadata",
        "model",
        "previous_output",
        "problem",
        "request_id",
        "role",
        "tools",
    }
)


# ---- 1. Configuration records ---------------------------------------------


class ConfigError(ValueError):
    """A Workflow or run configuration is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class RoleConfig:
    """Configuration shared by steps that execute the same semantic role.

    ``target`` is a key in ``RunConfig.runtimes`` for agent steps and in
    ``RunConfig.verifiers`` for verifier steps.  Verifier prompt rendering is
    configured here so verifier implementations contain no built-in Workflow
    prompts.  A role belongs to exactly one step kind: ``output_format`` is
    meaningful only for an agent-backed verifier and deterministic verifiers
    reject all AgentVerifier-only prompt/model fields during composition.
    """

    id: str
    target: str
    system_prompt: str | None = None
    model: str | None = None
    tools: tuple[str, ...] = ()
    input_template: str | None = None
    output_format: str | None = None
    prompt_ref: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.id, "role.id")
        _require_identifier(self.target, f"role {self.id!r}.target")
        if self.prompt_ref is not None:
            _require_non_empty_string(self.prompt_ref, f"role {self.id!r}.prompt_ref")
            try:
                prompt = resolve_prompt(self.prompt_ref)
            except PromptConfigError as exc:
                raise ConfigError(f"role {self.id!r}.prompt_ref is invalid: {exc}") from exc
            resolved_input = prompt.user_template if prompt.user_template else None
            if self.system_prompt is not None and self.system_prompt != prompt.system:
                raise ConfigError(
                    f"role {self.id!r}.system_prompt does not match {self.prompt_ref!r}"
                )
            if self.input_template is not None and self.input_template != resolved_input:
                raise ConfigError(
                    f"role {self.id!r}.input_template does not match {self.prompt_ref!r}"
                )
            object.__setattr__(self, "system_prompt", prompt.system)
            object.__setattr__(self, "input_template", resolved_input)
        if self.system_prompt is not None:
            _require_string(self.system_prompt, f"role {self.id!r}.system_prompt")
        if self.model is not None:
            _require_non_empty_string(self.model, f"role {self.id!r}.model")
        object.__setattr__(
            self,
            "tools",
            _string_tuple(self.tools, where=f"role {self.id!r}.tools", identifiers=True),
        )
        if len(set(self.tools)) != len(self.tools):
            raise ConfigError(f"role {self.id!r}.tools contains duplicate entries")
        if self.input_template is not None:
            _validate_template(self.input_template, where=f"role {self.id!r}.input_template")
        if self.output_format is not None:
            _require_non_empty_string(self.output_format, f"role {self.id!r}.output_format")
            object.__setattr__(
                self,
                "output_format",
                self.output_format.strip().lower().replace("_", "-"),
            )


@dataclass(frozen=True, slots=True)
class StepConfig:
    """One executable Workflow node."""

    id: str
    kind: str
    role: str
    input_template: str | None = None
    candidate_from: str | None = None
    output: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.id, "step.id")
        if self.kind not in _STEP_KINDS:
            raise ConfigError(
                f"step {self.id!r}.kind must be one of {sorted(_STEP_KINDS)}, got {self.kind!r}"
            )
        _require_identifier(self.role, f"step {self.id!r}.role")
        if self.input_template is not None:
            _validate_template(self.input_template, where=f"step {self.id!r}.input_template")
        if self.candidate_from is not None:
            _require_identifier(self.candidate_from, f"step {self.id!r}.candidate_from")
        if not isinstance(self.output, bool):
            raise ConfigError(f"step {self.id!r}.output must be a bool")
        if self.kind == "verifier" and self.input_template is not None:
            raise ConfigError(
                f"step {self.id!r}: verifier input_template belongs on its role so "
                "AgentVerifier receives the same validated template"
            )


@dataclass(frozen=True, slots=True)
class TransitionConfig:
    """One ordered, declarative route from a source step.

    Transitions are considered in declaration order.  A matching transition
    whose iteration budget is exhausted is skipped, which permits an explicit
    bounded retry edge followed by a terminal fallback.
    """

    source: str
    target: str | None
    condition: str = "always"
    max_iterations: int | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.source, "transition.source")
        if self.target is not None:
            _require_identifier(self.target, f"transition from {self.source!r}.target")
        if self.condition not in _TRANSITION_CONDITIONS:
            raise ConfigError(
                f"transition from {self.source!r}.condition must be one of "
                f"{sorted(_TRANSITION_CONDITIONS)}, got {self.condition!r}"
            )
        if self.max_iterations is not None:
            _require_int(
                self.max_iterations,
                f"transition from {self.source!r}.max_iterations",
                minimum=1,
                maximum=_MAX_LOOP_ITERATIONS,
            )


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """Repeat the whole topology and select one ordered branch result."""

    replicas: int
    strategy: str = "majority_vote"
    selection_key: str = "exact_text"

    def __post_init__(self) -> None:
        _require_int(
            self.replicas,
            "ensemble.replicas",
            minimum=2,
            maximum=_MAX_ENSEMBLE_REPLICAS,
        )
        if self.strategy not in _ENSEMBLE_STRATEGIES:
            raise ConfigError(
                f"ensemble.strategy must be one of {sorted(_ENSEMBLE_STRATEGIES)}, "
                f"got {self.strategy!r}"
            )
        if self.selection_key not in _ENSEMBLE_SELECTION_KEYS:
            raise ConfigError(
                "ensemble.selection_key must be one of "
                f"{sorted(_ENSEMBLE_SELECTION_KEYS)}, got {self.selection_key!r}"
            )


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """A complete immutable Workflow topology."""

    version: int
    name: str
    roles: tuple[RoleConfig, ...]
    steps: tuple[StepConfig, ...]
    entry_step: str
    transitions: tuple[TransitionConfig, ...] = ()
    ensemble: EnsembleConfig | None = None

    def __post_init__(self) -> None:
        if self.version != CONFIG_VERSION:
            raise ConfigError(
                f"unsupported workflow config version {self.version!r}; expected {CONFIG_VERSION}"
            )
        _require_identifier(self.name, "workflow.name")
        object.__setattr__(self, "roles", _typed_tuple(self.roles, RoleConfig, "roles"))
        object.__setattr__(self, "steps", _typed_tuple(self.steps, StepConfig, "steps"))
        object.__setattr__(
            self,
            "transitions",
            _typed_tuple(self.transitions, TransitionConfig, "transitions"),
        )
        if self.ensemble is not None and not isinstance(self.ensemble, EnsembleConfig):
            raise ConfigError("ensemble must be an EnsembleConfig")
        if not self.roles:
            raise ConfigError("workflow.roles must be non-empty")
        if not self.steps:
            raise ConfigError("workflow.steps must be non-empty")
        # A verifier judges a candidate, and the only producer of one is an agent
        # step: ``WorkflowInput`` carries the problem and nothing else, so with no
        # agent step ``_verification_request`` falls back to the problem statement
        # as the candidate and the run reports the problem back as its own answer.
        # Nothing downstream can read such a run either -- ``resource_summary``
        # names the solver runtime that a verifier-only topology never references.
        # Refuse the topology here, where the graph is first known, rather than
        # after the verifier calls have been paid for.
        if not any(step.kind == "agent" for step in self.steps):
            raise ConfigError(
                "workflow must declare at least one agent step; a verifier-only "
                "topology has no step that produces the candidate its verifiers judge"
            )

        roles = _unique_by_id(self.roles, "role")
        steps = _unique_by_id(self.steps, "step")
        _require_identifier(self.entry_step, "workflow.entry_step")
        if self.entry_step not in steps:
            raise ConfigError(f"entry_step references unknown step {self.entry_step!r}")

        for step in self.steps:
            if step.role not in roles:
                raise ConfigError(f"step {step.id!r} references unknown role {step.role!r}")
            role = roles[step.role]
            if role.prompt_ref is not None and step.input_template is not None:
                raise ConfigError(
                    f"step {step.id!r}.input_template cannot override prompt_ref "
                    f"{role.prompt_ref!r} owned by role {role.id!r}"
                )
            if step.candidate_from is not None:
                source = steps.get(step.candidate_from)
                if source is None:
                    raise ConfigError(
                        f"step {step.id!r}.candidate_from references unknown step "
                        f"{step.candidate_from!r}"
                    )
                if source.kind != "agent":
                    raise ConfigError(
                        f"step {step.id!r}.candidate_from must reference an agent step"
                    )

        role_kinds: dict[str, set[str]] = {role_id: set() for role_id in roles}
        for step in self.steps:
            role_kinds[step.role].add(step.kind)
        unused_roles = sorted(role_id for role_id, kinds in role_kinds.items() if not kinds)
        if unused_roles:
            raise ConfigError(f"workflow contains unused roles: {unused_roles}")
        for role_id, kinds in role_kinds.items():
            if len(kinds) != 1:
                raise ConfigError(f"role {role_id!r} cannot be shared by agent and verifier steps")
            role = roles[role_id]
            kind = next(iter(kinds))
            if kind == "agent":
                if role.system_prompt is None:
                    raise ConfigError(f"agent role {role_id!r}.system_prompt is required")
                if role.output_format is not None:
                    raise ConfigError(
                        f"agent role {role_id!r}.output_format has no agent-step consumer"
                    )
            if (
                kind == "verifier"
                and role.output_format is not None
                and role.output_format not in _AGENT_VERIFIER_OUTPUT_FORMATS
            ):
                raise ConfigError(
                    f"verifier role {role_id!r}.output_format must be one of "
                    f"{sorted(_AGENT_VERIFIER_OUTPUT_FORMATS)}, got {role.output_format!r}"
                )

        outgoing: dict[str, list[tuple[int, TransitionConfig]]] = {step_id: [] for step_id in steps}
        for index, transition in enumerate(self.transitions):
            if transition.source not in steps:
                raise ConfigError(
                    f"transition {index} references unknown source {transition.source!r}"
                )
            if transition.target is not None and transition.target not in steps:
                raise ConfigError(
                    f"transition {index} references unknown target {transition.target!r}"
                )
            source_step = steps[transition.source]
            if transition.condition != "always" and source_step.kind != "verifier":
                raise ConfigError(
                    f"transition {index}: condition {transition.condition!r} is only valid "
                    "after a verifier step"
                )
            outgoing[transition.source].append((index, transition))

        outputs = [step for step in self.steps if step.output]
        if len(outputs) != 1:
            raise ConfigError(
                f"workflow must declare exactly one output step, found {len(outputs)}"
            )
        output_step = outputs[0]
        if outgoing[output_step.id]:
            raise ConfigError(f"output step {output_step.id!r} must be terminal")

        reachable = _reachable_steps(self.entry_step, outgoing)
        unreachable = sorted(set(steps) - reachable)
        if unreachable:
            raise ConfigError(f"workflow contains unreachable steps: {unreachable}")
        if output_step.id not in reachable:
            raise ConfigError(f"output step {output_step.id!r} is not reachable")
        _validate_candidate_availability(self.entry_step, self.steps, outgoing)

        for source, indexed in outgoing.items():
            _validate_ordered_routes(source, indexed)
            if source != output_step.id:
                _validate_total_routing(steps[source], indexed)
        _validate_output_reachability(output_step.id, steps, outgoing)
        _validate_bounded_cycles(self.steps, self.transitions)
        _validate_semantic_route_reachability(self.entry_step, steps, outgoing)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Prepared public input plus an optional Environment-only private sidecar.

    ``split`` selects public records. ``task_payload_path`` instead names one
    concrete private split file whose rows need no ``split`` field. Under the
    ``direct`` envelope, ``task_payload_key`` must hold a JSON object (or a
    JSON-encoded object) that is passed through unchanged.
    """

    path: Path
    format: str = "jsonl"
    split: str | None = None
    input_key: str = "problem"
    id_key: str | None = "id"
    metadata_keys: tuple[str, ...] = ()
    task_payload_path: Path | None = None
    task_payload_format: str | None = None
    task_payload_key: str = "env_payload"
    task_payload_envelope: str = "direct"

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ConfigError("dataset.path must resolve to a pathlib.Path")
        if self.format not in _DATASET_FORMATS:
            raise ConfigError(
                f"dataset.format must be one of {sorted(_DATASET_FORMATS)}, got {self.format!r}"
            )
        if self.split is not None:
            _require_non_empty_string(self.split, "dataset.split")
        _require_non_empty_string(self.input_key, "dataset.input_key")
        if self.id_key is not None:
            _require_non_empty_string(self.id_key, "dataset.id_key")
        if self.task_payload_path is not None:
            if not isinstance(self.task_payload_path, Path):
                raise ConfigError("dataset.task_payload_path must resolve to a pathlib.Path")
            if self.id_key is None:
                raise ConfigError("dataset.task_payload_path requires dataset.id_key")
        if self.task_payload_format is not None:
            if self.task_payload_format not in _DATASET_FORMATS:
                raise ConfigError(
                    "dataset.task_payload_format must be one of "
                    f"{sorted(_DATASET_FORMATS)}, got {self.task_payload_format!r}"
                )
            if self.task_payload_path is None:
                raise ConfigError("dataset.task_payload_format requires dataset.task_payload_path")
        _require_non_empty_string(self.task_payload_key, "dataset.task_payload_key")
        if self.task_payload_envelope not in _TASK_PAYLOAD_ENVELOPES:
            raise ConfigError(
                f"dataset.task_payload_envelope must be one of {sorted(_TASK_PAYLOAD_ENVELOPES)}"
            )
        if self.task_payload_envelope != "direct" and self.task_payload_path is None:
            raise ConfigError("dataset.task_payload_envelope requires dataset.task_payload_path")
        object.__setattr__(
            self,
            "metadata_keys",
            _string_tuple(self.metadata_keys, where="dataset.metadata_keys"),
        )
        if len(set(self.metadata_keys)) != len(self.metadata_keys):
            raise ConfigError("dataset.metadata_keys contains duplicate entries")
        protected = {self.input_key}
        if self.id_key is not None:
            protected.add(self.id_key)
        overlap = protected.intersection(self.metadata_keys)
        if overlap:
            raise ConfigError(
                f"dataset.metadata_keys must not repeat input/id fields: {sorted(overlap)}"
            )


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    """A registry-selected resource with JSON-safe implementation options."""

    type: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.type, "resource.type")
        if not isinstance(self.options, Mapping):
            raise ConfigError("resource.options must be a mapping")
        object.__setattr__(self, "options", _freeze_json_mapping(self.options, "resource.options"))


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Application-owned WorkflowResult persistence settings."""

    directory: Path
    format: str = "jsonl"

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path):
            raise ConfigError("output.directory must resolve to a pathlib.Path")
        if self.format not in _OUTPUT_FORMATS:
            raise ConfigError(
                f"output.format must be one of {sorted(_OUTPUT_FORMATS)}, got {self.format!r}"
            )


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Application policy for repeated, resumable Workflow execution."""

    samples: int = 1
    seed: int = 0
    limit: int | None = None
    concurrency: int = 1
    resume: bool = True

    def __post_init__(self) -> None:
        _require_positive_int(self.samples, "execution.samples")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ConfigError("execution.seed must be an int")
        if self.limit is not None:
            if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit < 0:
                raise ConfigError("execution.limit must be a non-negative int")
        _require_positive_int(self.concurrency, "execution.concurrency")
        if not isinstance(self.resume, bool):
            raise ConfigError("execution.resume must be a bool")


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Opt-in adapter-driven memory for one Workflow execution."""

    adapter: str | None = "verification"
    mode: str = "working"
    top_k: int = 3
    profile: str = "semantic"
    cross_run: bool = False
    persistent_backend: str | None = None
    persistent_config: Mapping[str, Any] | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # adapter=None is the injection-only configuration: the session then
        # requires an adapter instance instead of a built-in name. Domain
        # adapters use it so the canonical runtime configuration never
        # names an adapter it does not honor. The YAML parser never produces
        # None; named selection stays validated below.
        if self.adapter is not None:
            _require_non_empty_string(self.adapter, "memory.adapter")
            if self.adapter not in {"verification", "robotics", "unified"}:
                raise ConfigError(
                    "memory.adapter must be one of the built-in adapters: "
                    "'verification', 'robotics', or 'unified'"
                )
        if not isinstance(self.options, Mapping):
            raise ConfigError("memory.options must be a mapping")
        options = _freeze_json_mapping(self.options, "memory.options")
        if options:
            raise ConfigError("memory.options has no supported settings in this branch")
        object.__setattr__(self, "options", options)
        if not isinstance(self.cross_run, bool):
            raise ConfigError("memory.cross_run must be a boolean")
        if self.cross_run and self.mode != "persistent":
            # A persistent lifetime in a process-local store silently buys
            # nothing; a loud contradiction beats a quiet no-op.
            raise ConfigError("memory.cross_run requires memory.mode='persistent'")
        if self.mode not in {"working", "persistent"}:
            raise ConfigError(
                "memory.mode must be 'working' or 'persistent'; omit memory to disable it"
            )
        _require_int(self.top_k, "memory.top_k", minimum=1, maximum=20)
        from alphaapollo.evolving.memory_profile import NAMED_PROFILES

        if self.profile not in NAMED_PROFILES or self.profile == "off":
            raise ConfigError(
                "memory.profile must be one of "
                f"{sorted(name for name in NAMED_PROFILES if name != 'off')}"
            )
        configured = self.persistent_backend is not None or self.persistent_config is not None
        if self.mode != "persistent" and configured:
            raise ConfigError("persistent backend configuration requires memory.mode='persistent'")
        if self.mode == "persistent":
            if self.persistent_backend is None or not self.persistent_backend.strip():
                raise ConfigError("memory.mode='persistent' requires memory.persistent_backend")
            if self.persistent_config is None:
                raise ConfigError("memory.mode='persistent' requires memory.persistent_config")
        if self.persistent_config is not None:
            object.__setattr__(
                self,
                "persistent_config",
                _freeze_json_mapping(self.persistent_config, "memory.persistent_config"),
            )


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Optional out-of-loop scoring against a private prepared-data projection."""

    dataset_root: Path
    source_id: str
    version: str = "v1"
    report_name: str = "report.json"

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_root, Path):
            raise ConfigError("scoring.dataset_root must resolve to a pathlib.Path")
        _require_identifier(self.source_id, "scoring.source_id")
        _require_non_empty_string(self.version, "scoring.version")
        _require_non_empty_string(self.report_name, "scoring.report_name")
        if Path(self.report_name).name != self.report_name:
            raise ConfigError("scoring.report_name must be a filename, not a path")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """A validated composition recipe for one prepared-dataset Workflow run."""

    version: int
    workflow: WorkflowConfig
    dataset: DatasetConfig
    runtimes: Mapping[str, ResourceConfig]
    verifiers: Mapping[str, ResourceConfig]
    environment: ResourceConfig
    output: OutputConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    scoring: ScoringConfig | None = None
    memory: MemoryConfig | None = None

    def __post_init__(self) -> None:
        if self.version != CONFIG_VERSION:
            raise ConfigError(
                f"unsupported run config version {self.version!r}; expected {CONFIG_VERSION}"
            )
        if not isinstance(self.workflow, WorkflowConfig):
            raise ConfigError("run.workflow must resolve to a WorkflowConfig")
        if not isinstance(self.dataset, DatasetConfig):
            raise ConfigError("run.dataset must be a DatasetConfig")
        if not isinstance(self.environment, ResourceConfig):
            raise ConfigError("run.environment must be a ResourceConfig")
        if not isinstance(self.output, OutputConfig):
            raise ConfigError("run.output must be an OutputConfig")
        if not isinstance(self.execution, ExecutionConfig):
            raise ConfigError("run.execution must be an ExecutionConfig")
        if self.scoring is not None and not isinstance(self.scoring, ScoringConfig):
            raise ConfigError("run.scoring must be a ScoringConfig")
        if self.memory is not None and not isinstance(self.memory, MemoryConfig):
            raise ConfigError("run.memory must be a MemoryConfig")
        runtimes = _resource_mapping(self.runtimes, "runtimes")
        verifiers = _resource_mapping(self.verifiers, "verifiers")
        object.__setattr__(self, "runtimes", runtimes)
        object.__setattr__(self, "verifiers", verifiers)

        roles = {role.id: role for role in self.workflow.roles}
        for step in self.workflow.steps:
            role = roles[step.role]
            available = runtimes if step.kind == "agent" else verifiers
            resource_kind = "runtime" if step.kind == "agent" else "verifier"
            if role.target not in available:
                raise ConfigError(
                    f"step {step.id!r} role {role.id!r} targets unknown {resource_kind} "
                    f"{role.target!r}"
                )


# ---- 2. Loading and Hydra composition -------------------------------------


def read_config_mapping(path: str | Path) -> dict[str, Any]:
    """Read a JSON/YAML object without interpreting it as a config record."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ConfigError(f"unsupported config extension {suffix!r}; use .json, .yaml, or .yml")
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read config {source}: {exc}") from exc
    try:
        if suffix == ".json":
            raw = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise ConfigError("PyYAML is required to load YAML configs") from exc
            raw = yaml.safe_load(text)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"could not parse config {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("config root must be a mapping")
    return dict(raw)


def compose_config_mapping(path: str | Path) -> dict[str, Any]:
    """Read one config and resolve its Hydra defaults list when present.

    Standalone JSON/YAML mappings keep their existing behavior. A YAML primary
    config containing ``defaults`` is composed relative to its own directory,
    with AlphaApollo's packaged config groups appended after that primary search
    path. Hydra control nodes are consumed before the strict RunConfig parser sees
    the resulting ordinary mapping.
    """

    source = Path(path).resolve()
    raw = read_config_mapping(source)
    if "defaults" not in raw:
        return raw
    if source.suffix.lower() != ".yaml":
        raise ConfigError("Hydra-composed configs must use the .yaml extension")

    try:
        from hydra import compose, initialize_config_dir
        from hydra.errors import HydraException
        from omegaconf import OmegaConf
        from omegaconf.errors import OmegaConfBaseException
    except ImportError as exc:  # pragma: no cover - packaging/install failure
        raise ConfigError("hydra-core and omegaconf are required to compose YAML configs") from exc

    hydra_node = raw.get("hydra", {})
    if not isinstance(hydra_node, Mapping):
        raise ConfigError("hydra must be a mapping when provided")
    configured_paths = hydra_node.get("searchpath", ())
    if isinstance(configured_paths, (str, bytes)) or not isinstance(configured_paths, Sequence):
        raise ConfigError("hydra.searchpath must be a sequence")
    search_paths = list(configured_paths)
    if any(not isinstance(item, str) or not item.strip() for item in search_paths):
        raise ConfigError("hydra.searchpath must contain non-empty strings")
    packaged_path = "pkg://alphaapollo.configs"
    if packaged_path not in search_paths:
        search_paths.append(packaged_path)

    try:
        with initialize_config_dir(
            config_dir=str(source.parent),
            job_name="alphaapollo-workflow",
            version_base=None,
        ):
            composed = compose(
                config_name=source.stem,
                overrides=[f"hydra.searchpath={json.dumps(search_paths)}"],
            )
            resolved = OmegaConf.to_container(composed, resolve=True, throw_on_missing=True)
    except (HydraException, OmegaConfBaseException, OSError, ValueError) as exc:
        raise ConfigError(f"could not compose config {source}: {exc}") from exc

    if not isinstance(resolved, Mapping):
        raise ConfigError("composed config root must be a mapping")
    return dict(resolved)


def parse_workflow_config(raw: Mapping[str, Any]) -> WorkflowConfig:
    """Validate a plain mapping as a topology-only Workflow config."""

    top = _mapping(raw, "workflow config")
    _reject_unknown(
        top,
        {"version", "name", "roles", "steps", "entry_step", "transitions", "ensemble"},
        "workflow config",
    )
    _require_keys(top, {"version", "name", "roles", "steps", "entry_step"}, "workflow config")
    roles_raw = _sequence(top["roles"], "workflow.roles")
    steps_raw = _sequence(top["steps"], "workflow.steps")
    transitions_raw = _sequence(top.get("transitions", ()), "workflow.transitions")
    return WorkflowConfig(
        version=_raw_int(top["version"], "workflow.version"),
        name=_raw_string(top["name"], "workflow.name"),
        roles=tuple(_parse_role(value, index) for index, value in enumerate(roles_raw)),
        steps=tuple(_parse_step(value, index) for index, value in enumerate(steps_raw)),
        entry_step=_raw_string(top["entry_step"], "workflow.entry_step"),
        transitions=tuple(
            _parse_transition(value, index) for index, value in enumerate(transitions_raw)
        ),
        ensemble=_parse_ensemble(top["ensemble"]) if "ensemble" in top else None,
    )


def load_workflow_config(path: str | Path) -> WorkflowConfig:
    """Load and validate one topology preset."""

    return parse_workflow_config(read_config_mapping(path))


def parse_run_config(
    raw: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> RunConfig:
    """Validate a complete run recipe.

    Relative dataset, Workflow preset, and output paths are resolved against
    ``base_dir``.  This makes a recipe independent of the caller's current
    working directory and behaves the same in a local checkout and on a remote
    worker.
    """

    top = _mapping(raw, "run config")
    _reject_unknown(
        top,
        {
            "version",
            "workflow",
            "dataset",
            "runtimes",
            "verifiers",
            "environment",
            "output",
            "execution",
            "scoring",
            "memory",
        },
        "run config",
    )
    _require_keys(
        top,
        {"version", "workflow", "dataset", "runtimes", "verifiers", "environment", "output"},
        "run config",
    )
    root = Path.cwd() if base_dir is None else Path(base_dir)
    root = root.resolve()

    workflow_raw = top["workflow"]
    if isinstance(workflow_raw, (str, Path)):
        workflow_path = _resolve_path(workflow_raw, root, "workflow")
        workflow = load_workflow_config(workflow_path)
    elif isinstance(workflow_raw, Mapping):
        workflow = parse_workflow_config(workflow_raw)
    else:
        raise ConfigError("run.workflow must be a preset path or an inline mapping")

    scoring_raw = top.get("scoring")
    if scoring_raw is not None:
        _reject_in_loop_grading(top.get("environment"))

    return RunConfig(
        version=_raw_int(top["version"], "run.version"),
        workflow=workflow,
        dataset=_parse_dataset(top["dataset"], root),
        runtimes=_parse_resources(top["runtimes"], "run.runtimes"),
        verifiers=_parse_resources(top["verifiers"], "run.verifiers"),
        environment=_parse_resource(top["environment"], "run.environment"),
        output=_parse_output(top["output"], root),
        execution=_parse_execution(top.get("execution")),
        scoring=_parse_scoring(scoring_raw, root) if scoring_raw is not None else None,
        memory=_parse_memory(top.get("memory"), root),
    )


def load_run_config(path: str | Path) -> RunConfig:
    """Load a run recipe, resolving its relative paths beside the recipe file."""

    source = Path(path).resolve()
    return parse_run_config(compose_config_mapping(source), base_dir=source.parent)


# ---- 3. Mapping-to-record parsers -----------------------------------------


def _parse_role(value: Any, index: int) -> RoleConfig:
    where = f"workflow.roles[{index}]"
    raw = _mapping(value, where)
    allowed = {
        "id",
        "target",
        "prompt_ref",
        "system_prompt",
        "model",
        "tools",
        "input_template",
        "output_format",
    }
    _reject_unknown(raw, allowed, where)
    _require_keys(raw, {"id", "target"}, where)
    if raw.get("prompt_ref") is not None and any(
        raw.get(name) is not None for name in ("system_prompt", "input_template")
    ):
        raise ConfigError(f"{where} cannot combine prompt_ref with system_prompt or input_template")
    return RoleConfig(
        id=_raw_string(raw["id"], f"{where}.id"),
        target=_raw_string(raw["target"], f"{where}.target"),
        prompt_ref=_optional_string(raw.get("prompt_ref"), f"{where}.prompt_ref"),
        system_prompt=_optional_string(raw.get("system_prompt"), f"{where}.system_prompt"),
        model=_optional_string(raw.get("model"), f"{where}.model"),
        tools=_raw_string_tuple(raw.get("tools", ()), f"{where}.tools"),
        input_template=_optional_string(raw.get("input_template"), f"{where}.input_template"),
        output_format=_optional_string(raw.get("output_format"), f"{where}.output_format"),
    )


def _parse_step(value: Any, index: int) -> StepConfig:
    where = f"workflow.steps[{index}]"
    raw = _mapping(value, where)
    allowed = {
        "id",
        "kind",
        "role",
        "input_template",
        "candidate_from",
        "output",
    }
    _reject_unknown(raw, allowed, where)
    _require_keys(raw, {"id", "kind", "role"}, where)
    output = raw.get("output", False)
    if not isinstance(output, bool):
        raise ConfigError(f"{where}.output must be a bool")
    return StepConfig(
        id=_raw_string(raw["id"], f"{where}.id"),
        kind=_raw_string(raw["kind"], f"{where}.kind"),
        role=_raw_string(raw["role"], f"{where}.role"),
        input_template=_optional_string(raw.get("input_template"), f"{where}.input_template"),
        candidate_from=_optional_string(raw.get("candidate_from"), f"{where}.candidate_from"),
        output=output,
    )


def _parse_transition(value: Any, index: int) -> TransitionConfig:
    where = f"workflow.transitions[{index}]"
    raw = _mapping(value, where)
    allowed = {"source", "target", "condition", "max_iterations"}
    _reject_unknown(raw, allowed, where)
    _require_keys(raw, {"source", "target"}, where)
    target = raw["target"]
    if target is not None and not isinstance(target, str):
        raise ConfigError(f"{where}.target must be a string or null")
    maximum = raw.get("max_iterations")
    if maximum is not None:
        maximum = _raw_int(maximum, f"{where}.max_iterations")
    return TransitionConfig(
        source=_raw_string(raw["source"], f"{where}.source"),
        target=target,
        condition=_raw_string(raw.get("condition", "always"), f"{where}.condition"),
        max_iterations=maximum,
    )


def _parse_ensemble(value: Any) -> EnsembleConfig:
    where = "workflow.ensemble"
    raw = _mapping(value, where)
    _reject_unknown(raw, {"replicas", "strategy", "selection_key"}, where)
    _require_keys(raw, {"replicas"}, where)
    return EnsembleConfig(
        replicas=_raw_int(raw["replicas"], f"{where}.replicas"),
        strategy=_raw_string(raw.get("strategy", "majority_vote"), f"{where}.strategy"),
        selection_key=_raw_string(
            raw.get("selection_key", "exact_text"),
            f"{where}.selection_key",
        ),
    )


def _parse_dataset(value: Any, base_dir: Path) -> DatasetConfig:
    where = "run.dataset"
    raw = _mapping(value, where)
    allowed = {
        "path",
        "format",
        "split",
        "input_key",
        "id_key",
        "metadata_keys",
        "task_payload_path",
        "task_payload_format",
        "task_payload_key",
        "task_payload_envelope",
    }
    _reject_unknown(raw, allowed, where)
    _require_keys(raw, {"path"}, where)
    raw_id_key = raw.get("id_key", "id")
    if raw_id_key is not None and not isinstance(raw_id_key, str):
        raise ConfigError(f"{where}.id_key must be a string or null")
    return DatasetConfig(
        path=_resolve_path(raw["path"], base_dir, f"{where}.path"),
        format=_raw_string(raw.get("format", "jsonl"), f"{where}.format"),
        split=_optional_string(raw.get("split"), f"{where}.split"),
        input_key=_raw_string(raw.get("input_key", "problem"), f"{where}.input_key"),
        id_key=raw_id_key,
        metadata_keys=_raw_string_tuple(raw.get("metadata_keys", ()), f"{where}.metadata_keys"),
        task_payload_path=(
            _resolve_path(raw["task_payload_path"], base_dir, f"{where}.task_payload_path")
            if raw.get("task_payload_path") is not None
            else None
        ),
        task_payload_format=_optional_string(
            raw.get("task_payload_format"),
            f"{where}.task_payload_format",
        ),
        task_payload_key=_raw_string(
            raw.get("task_payload_key", "env_payload"),
            f"{where}.task_payload_key",
        ),
        task_payload_envelope=_raw_string(
            raw.get("task_payload_envelope", "direct"),
            f"{where}.task_payload_envelope",
        ),
    )


def _parse_resources(value: Any, where: str) -> Mapping[str, ResourceConfig]:
    raw = _mapping(value, where)
    resources: dict[str, ResourceConfig] = {}
    for name, item in raw.items():
        _require_identifier(name, f"{where} key")
        resources[name] = _parse_resource(item, f"{where}.{name}")
    return MappingProxyType(resources)


def _parse_resource(value: Any, where: str) -> ResourceConfig:
    raw = _mapping(value, where)
    _reject_unknown(raw, {"type", "options"}, where)
    _require_keys(raw, {"type"}, where)
    options = raw.get("options", {})
    if not isinstance(options, Mapping):
        raise ConfigError(f"{where}.options must be a mapping")
    return ResourceConfig(
        type=_raw_string(raw["type"], f"{where}.type"),
        options=options,
    )


def _parse_output(value: Any, base_dir: Path) -> OutputConfig:
    where = "run.output"
    raw = _mapping(value, where)
    _reject_unknown(raw, {"directory", "format"}, where)
    _require_keys(raw, {"directory"}, where)
    return OutputConfig(
        directory=_resolve_path(raw["directory"], base_dir, f"{where}.directory"),
        format=_raw_string(raw.get("format", "jsonl"), f"{where}.format"),
    )


def _parse_execution(value: Any) -> ExecutionConfig:
    where = "run.execution"
    if value is None:
        return ExecutionConfig()
    raw = _mapping(value, where)
    _reject_unknown(raw, {"samples", "seed", "limit", "concurrency", "resume"}, where)
    return ExecutionConfig(
        samples=_raw_int(raw.get("samples", 1), f"{where}.samples"),
        seed=_raw_int(raw.get("seed", 0), f"{where}.seed"),
        limit=(_raw_int(raw["limit"], f"{where}.limit") if raw.get("limit") is not None else None),
        concurrency=_raw_int(raw.get("concurrency", 1), f"{where}.concurrency"),
        resume=raw.get("resume", True),
    )


def _parse_memory(value: Any, base_dir: Path | None = None) -> MemoryConfig | None:
    if value is None:
        return None
    where = "run.memory"
    raw = _mapping(value, where)
    _reject_unknown(
        raw,
        {
            "adapter",
            "mode",
            "profile",
            "top_k",
            "cross_run",
            "options",
            "persistent_backend",
            "persistent_config",
        },
        where,
    )
    cross_run = raw.get("cross_run", False)
    if not isinstance(cross_run, bool):
        raise ConfigError(f"{where}.cross_run must be a boolean")
    persistent_config = raw.get("persistent_config")
    if persistent_config is not None:
        persistent_config = _mapping(persistent_config, f"{where}.persistent_config")
    options = dict(_mapping(raw.get("options", {}), f"{where}.options"))
    return MemoryConfig(
        adapter=_raw_string(raw.get("adapter", "verification"), f"{where}.adapter"),
        mode=_raw_string(raw.get("mode", "working"), f"{where}.mode"),
        top_k=_raw_int(raw.get("top_k", 3), f"{where}.top_k"),
        profile=_raw_string(raw.get("profile", "semantic"), f"{where}.profile"),
        cross_run=cross_run,
        options=options,
        persistent_backend=_optional_string(
            raw.get("persistent_backend"), f"{where}.persistent_backend"
        ),
        persistent_config=persistent_config,
    )


def _parse_scoring(value: Any, base_dir: Path) -> ScoringConfig:
    where = "run.scoring"
    raw = _mapping(value, where)
    _reject_unknown(raw, {"dataset_root", "source_id", "version", "report_name"}, where)
    _require_keys(raw, {"dataset_root", "source_id"}, where)
    return ScoringConfig(
        dataset_root=_resolve_path(raw["dataset_root"], base_dir, f"{where}.dataset_root"),
        source_id=_raw_string(raw["source_id"], f"{where}.source_id"),
        version=_raw_string(raw.get("version", "v1"), f"{where}.version"),
        report_name=_raw_string(raw.get("report_name", "report.json"), f"{where}.report_name"),
    )


# ---- 4. Template and graph validation -------------------------------------


def _reject_in_loop_grading(environment: Any) -> None:
    """Refuse an Environment that grades inside the episode of a scored run.

    ``environment.options.grader_id`` makes the Environment produce reward from
    the gold answer, which means gold must be handed to the process running the
    model. That is correct for RL training and fatal next to ``scoring``: an
    evaluated run's whole guarantee is that the public projection reaches the
    model and the private one reaches only the scorer.

    ``resources._validate_environment_resource`` refuses the key for *every*
    Workflow config, scored or not, because this package composes only the
    out-of-loop path. This check exists so a scored recipe fails at parse time
    naming the boundary it crossed, rather than later with a generic unknown-key
    message; it is a better error, not the only one.
    """

    if not isinstance(environment, Mapping):
        return
    options = environment.get("options")
    if isinstance(options, Mapping) and options.get("grader_id") is not None:
        raise ConfigError(
            "environment.options.grader_id grades inside the episode, which would put "
            "gold answers in the model's process; a config with scoring grades out of "
            "loop from the private projection, so remove the option -- Workflow configs "
            "do not compose in-loop grading at all"
        )


def _validate_template(template: str, *, where: str) -> None:
    _require_string(template, where)
    try:
        parsed = tuple(Formatter().parse(template))
    except ValueError as exc:
        raise ConfigError(f"{where} is not a valid template: {exc}") from exc
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in _TEMPLATE_FIELDS:
            raise ConfigError(
                f"{where} uses unsupported placeholder {field_name!r}; expected one of "
                f"{sorted(_TEMPLATE_FIELDS)}"
            )
        if format_spec or conversion:
            raise ConfigError(f"{where} placeholders cannot use conversions or format specs")


def _validate_ordered_routes(source: str, indexed: Sequence[tuple[int, TransitionConfig]]) -> None:
    outcomes = ("agent", "pass", "fail", "inconclusive")
    permanently_covered: set[str] = set()
    for index, transition in indexed:
        matched = {
            outcome
            for outcome in outcomes
            if _config_condition_matches(transition.condition, outcome)
        }
        if matched <= permanently_covered:
            raise ConfigError(
                f"transition {index} from {source!r} is shadowed by earlier unbounded "
                f"routes for outcomes {sorted(matched)!r}"
            )
        if transition.max_iterations is None:
            permanently_covered.update(matched)


def _validate_semantic_route_reachability(
    entry: str,
    steps: Mapping[str, StepConfig],
    outgoing: Mapping[str, Sequence[tuple[int, TransitionConfig]]],
) -> None:
    """Require every declared route to be selectable under the executor contract.

    Ordered finite routes do not automatically make their fallbacks reachable:
    their budgets can only be exhausted if execution returns to the source.  We
    therefore explore the finite executor state (current step plus bounded-edge
    counts) over every possible typed step outcome.  This accepts genuine
    bounded revise loops while rejecting routes hidden behind a one-shot branch
    whose target can never return to its source.
    """

    all_routes = {index for indexed in outgoing.values() for index, _transition in indexed}
    if not all_routes:
        return
    pending: list[tuple[str, frozenset[tuple[int, int]]]] = [(entry, frozenset())]
    visited: set[tuple[str, frozenset[tuple[int, int]]]] = set()
    selected_routes: set[int] = set()

    while pending:
        step_id, counts = pending.pop()
        state = (step_id, counts)
        if state in visited:
            continue
        if len(visited) >= _MAX_SEMANTIC_ROUTE_STATES:
            raise ConfigError(
                "workflow ordered routing exceeds the semantic validation limit of "
                f"{_MAX_SEMANTIC_ROUTE_STATES} states; reduce bounded routes or their budgets"
            )
        visited.add(state)
        step = steps[step_id]
        count_by_index = dict(counts)
        outcomes = ("pass", "fail", "inconclusive") if step.kind == "verifier" else ("agent",)
        for outcome in outcomes:
            selected: tuple[int, TransitionConfig] | None = None
            for index, transition in outgoing[step_id]:
                if transition.max_iterations is not None and (
                    count_by_index.get(index, 0) >= transition.max_iterations
                ):
                    continue
                if _config_condition_matches(transition.condition, outcome):
                    selected = (index, transition)
                    break
            if selected is None:
                continue
            index, transition = selected
            selected_routes.add(index)
            if selected_routes == all_routes:
                return
            next_counts = counts
            if transition.max_iterations is not None:
                updated = dict(counts)
                updated[index] = updated.get(index, 0) + 1
                next_counts = frozenset(updated.items())
            if transition.target is not None:
                pending.append((transition.target, next_counts))

    for indexed in outgoing.values():
        for index, transition in indexed:
            if index not in selected_routes:
                raise ConfigError(
                    f"transition {index} from {transition.source!r} is semantically unreachable "
                    "under ordered routing and edge budgets"
                )


def _validate_total_routing(
    step: StepConfig,
    indexed: Sequence[tuple[int, TransitionConfig]],
) -> None:
    if not indexed:
        raise ConfigError(f"non-output step {step.id!r} must declare a transition")
    outcomes = ("pass", "fail", "inconclusive") if step.kind == "verifier" else ("agent",)
    for outcome in outcomes:
        covered = any(
            transition.max_iterations is None
            and _config_condition_matches(transition.condition, outcome)
            for _, transition in indexed
        )
        if not covered:
            raise ConfigError(
                f"step {step.id!r} has no unbounded fallback transition for outcome {outcome!r}"
            )


def _config_condition_matches(condition: str, outcome: str) -> bool:
    if condition == "always":
        return True
    if outcome == "agent":
        return False
    if condition == "passed":
        return outcome == "pass"
    if condition == "failed":
        return outcome == "fail"
    if condition == "inconclusive":
        return outcome == "inconclusive"
    if condition == "not_passed":
        return outcome != "pass"
    return False


def _validate_output_reachability(
    output_step_id: str,
    steps: Mapping[str, StepConfig],
    outgoing: Mapping[str, Sequence[tuple[int, TransitionConfig]]],
) -> None:
    reverse: dict[str, list[str]] = {step_id: [] for step_id in steps}
    for source, indexed in outgoing.items():
        for _, transition in indexed:
            if transition.target is not None:
                reverse[transition.target].append(source)
    pending = [output_step_id]
    reaches_output: set[str] = set()
    while pending:
        node = pending.pop()
        if node in reaches_output:
            continue
        reaches_output.add(node)
        pending.extend(reverse[node])
    stranded = sorted(set(steps) - reaches_output)
    if stranded:
        raise ConfigError(f"steps cannot reach output {output_step_id!r}: {stranded}")


def _validate_bounded_cycles(
    steps: Sequence[StepConfig], transitions: Sequence[TransitionConfig]
) -> None:
    """Reject cycles made entirely of unbounded transitions.

    A finite edge anywhere in a cycle bounds how often that cycle can be
    completed.  Requiring every edge in the same cycle to carry a duplicate
    bound would make the budget ambiguous and is intentionally avoided.
    """

    adjacency: dict[str, list[str]] = {step.id: [] for step in steps}
    for transition in transitions:
        if transition.target is not None and transition.max_iterations is None:
            adjacency[transition.source].append(transition.target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ConfigError("workflow contains a cycle without an explicit max_iterations edge")
        if node in visited:
            return
        visiting.add(node)
        for target in adjacency[node]:
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for step in steps:
        visit(step.id)


def _reachable_steps(
    entry: str,
    outgoing: Mapping[str, Sequence[tuple[int, TransitionConfig]]],
) -> set[str]:
    pending = [entry]
    visited: set[str] = set()
    while pending:
        step_id = pending.pop()
        if step_id in visited:
            continue
        visited.add(step_id)
        pending.extend(
            transition.target
            for _, transition in outgoing[step_id]
            if transition.target is not None
        )
    return visited


def _validate_candidate_availability(
    entry: str,
    steps: Sequence[StepConfig],
    outgoing: Mapping[str, Sequence[tuple[int, TransitionConfig]]],
) -> None:
    """Require explicit candidate sources to dominate their consuming step.

    A source that dominates a step has executed on every route from the entry
    before that step can be reached for the first time.  This rejects self and
    forward references as well as branch-local candidates consumed after a
    branch merge where the producing branch may not have run.
    """

    step_ids = {step.id for step in steps}
    predecessors: dict[str, set[str]] = {step_id: set() for step_id in step_ids}
    for source, indexed in outgoing.items():
        for _, transition in indexed:
            if transition.target is not None:
                predecessors[transition.target].add(source)

    dominators: dict[str, set[str]] = {
        step_id: ({entry} if step_id == entry else set(step_ids)) for step_id in step_ids
    }
    changed = True
    while changed:
        changed = False
        for step_id in step_ids:
            if step_id == entry:
                continue
            incoming = predecessors[step_id]
            common = set.intersection(*(dominators[source] for source in incoming))
            updated = {step_id, *common}
            if updated != dominators[step_id]:
                dominators[step_id] = updated
                changed = True

    for step in steps:
        source = step.candidate_from
        if source is not None and source not in dominators[step.id] - {step.id}:
            raise ConfigError(
                f"step {step.id!r}.candidate_from {source!r} must reference an agent step "
                "guaranteed to run before the step is first reached"
            )


# ---- 5. Frozen JSON values and primitive guards ---------------------------


def _resource_mapping(value: Any, where: str) -> Mapping[str, ResourceConfig]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    resources: dict[str, ResourceConfig] = {}
    for name, resource in value.items():
        _require_identifier(name, f"{where} key")
        if not isinstance(resource, ResourceConfig):
            raise ConfigError(f"{where}.{name} must be a ResourceConfig")
        resources[name] = resource
    return MappingProxyType(resources)


class _FrozenJsonDict(dict[str, Any]):
    """A recursively frozen mapping that remains JSON/deepcopy compatible."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("frozen JSON mappings do not support mutation")

    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __copy__(self) -> _FrozenJsonDict:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> _FrozenJsonDict:
        return self


def _freeze_json_mapping(value: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{where} keys must be strings")
        frozen[key] = _freeze_value(item, f"{where}.{key}")
    return _FrozenJsonDict(frozen)


def _freeze_value(value: Any, where: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{where} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, where)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(item, f"{where}[]") for item in value)
    raise ConfigError(f"{where} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    """Return mutable plain JSON containers at a Config-to-runtime boundary.

    Config records keep recursively immutable JSON values.  Consumers that hand
    options to mutable third-party SDKs should explicitly thaw them instead of
    relying on ``deepcopy`` implementation details.
    """

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw_json(item) for item in value]
    return value


def _resolve_path(value: Any, base_dir: Path, where: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"{where} must be a path string")
    if not str(value).strip():
        raise ConfigError(f"{where} must be non-empty")
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{where} keys must be strings")
    return dict(value)


def _sequence(value: Any, where: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError(f"{where} must be a list")
    return tuple(value)


def _closed_options(
    value: object,
    *,
    allowed: set[str],
    required: set[str],
    where: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    raw = dict(value)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{where} contains unknown keys: {unknown}")
    missing = sorted(required - set(raw))
    if missing:
        raise ConfigError(f"{where} is missing required keys: {missing}")
    return raw


def _non_empty_string(value: object, where: str) -> str:
    _require_non_empty_string(value, where)
    return cast(str, value)


def _positive_int(value: object, where: str) -> int:
    _require_positive_int(value, where)
    return cast(int, value)


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"unknown {where} keys: {unknown}")


def _require_keys(raw: Mapping[str, Any], required: set[str], where: str) -> None:
    missing = sorted(required - set(raw))
    if missing:
        raise ConfigError(f"missing {where} keys: {missing}")


def _unique_by_id(values: Sequence[Any], noun: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if value.id in result:
            raise ConfigError(f"duplicate {noun} id {value.id!r}")
        result[value.id] = value
    return result


def _typed_tuple(value: Any, expected: type, where: str) -> tuple[Any, ...]:
    if not isinstance(value, (tuple, list)):
        raise ConfigError(f"workflow.{where} must be a sequence")
    result = tuple(value)
    if not all(isinstance(item, expected) for item in result):
        raise ConfigError(f"workflow.{where} contains an invalid record")
    return result


def _raw_string(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string")
    return value


def _optional_string(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _raw_string(value, where)


def _raw_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer")
    return value


def _raw_string_tuple(value: Any, where: str) -> tuple[str, ...]:
    return _string_tuple(_sequence(value, where), where=where)


def _string_tuple(
    value: Any,
    *,
    where: str,
    identifiers: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ConfigError(f"{where} must be a list of strings")
    result = tuple(value)
    for item in result:
        if identifiers:
            _require_identifier(item, where)
        else:
            _require_non_empty_string(item, where)
    return result


def _require_string(value: Any, where: str) -> None:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string")


def _require_non_empty_string(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")


def _require_identifier(value: Any, where: str) -> None:
    _require_non_empty_string(value, where)
    if not value.replace("_", "").replace("-", "").isalnum():
        raise ConfigError(f"{where} must contain only letters, numbers, '_' or '-', got {value!r}")


def _require_positive_int(value: Any, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{where} must be a positive int")


def _require_int(
    value: Any,
    where: str,
    *,
    minimum: int,
    maximum: int,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer")
    if not minimum <= value <= maximum:
        raise ConfigError(f"{where} must be between {minimum} and {maximum}")


# ---- 6. Repeated-run helpers ----------------------------------------------


def coerce_config(value: object, *, base_dir: str | Path | None = None) -> RunConfig:
    """Coerce a RunConfig or a canonical mapping at a programmatic boundary."""

    if isinstance(value, RunConfig):
        return value
    if isinstance(value, Mapping):
        return parse_run_config(value, base_dir=base_dir)
    raise TypeError("config must be a RunConfig or canonical mapping")


def config_digest(config: RunConfig | object) -> str:
    """Return a stable run identity without credentials or machine-local paths."""

    payload = _jsonable(coerce_config(config))
    payload.pop("output", None)
    # Adding an opt-in capability must not invalidate every existing memory-off
    # resume key. Before this field existed, its absent representation was no key.
    memory = payload.get("memory")
    if memory is None:
        payload.pop("memory", None)
    elif isinstance(memory, dict) and memory.get("cross_run") is False:
        # Same contract for memory-on identities: the switch's default (off)
        # representation predates the field and must keep the historical
        # digest, or every existing memory-enabled resume key breaks. Enabled
        # cross-run is a genuinely different memory identity and keys freshly.
        memory.pop("cross_run", None)
    if isinstance(memory, dict) and not memory.get("options"):
        # Preserve historical identities that predate this optional field.
        # The empty default adds no semantic configuration.
        memory.pop("options", None)
    payload.get("dataset", {}).pop("path", None)
    payload.get("dataset", {}).pop("task_payload_path", None)
    scoring = payload.get("scoring")
    if isinstance(scoring, dict):
        scoring.pop("dataset_root", None)
    execution = payload.get("execution")
    if isinstance(execution, dict):
        for key in ("concurrency", "limit", "resume"):
            execution.pop(key, None)
    for resource in payload.get("runtimes", {}).values():
        backend_options = resource.get("options", {}).get("backend", {}).get("options", {})
        backend_options.pop("api_key", None)
        backend_options.pop("base_url", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def run_config_for_cell(
    config: RunConfig,
    *,
    run_dir: Path,
    condition: str,
    seed: int,
) -> RunConfig:
    """Bind one repeated cell to an isolated output directory and sampling seed."""

    expected_condition = execution_label(config)
    if condition != expected_condition:
        raise ConfigError(
            f"run condition {condition!r} does not match configured resources "
            f"({expected_condition!r})"
        )
    runtimes: dict[str, ResourceConfig] = {}
    for name, resource in config.runtimes.items():
        options = _thaw_json(resource.options)
        if resource.type == "alphaapollo":
            sampling = dict(options.get("sampling", {}))
            sampling["seed"] = seed
            options["sampling"] = sampling
        runtimes[name] = ResourceConfig(type=resource.type, options=options)
    return replace(
        config,
        runtimes=MappingProxyType(runtimes),
        output=OutputConfig(directory=run_dir / "canonical", format="jsonl"),
        execution=ExecutionConfig(),
        scoring=None,
    )


def execution_label(run: RunConfig) -> str:
    """Return a report label derived from the actually configured tool surface."""

    python_enabled = bool(run.environment.options.get("enable_python_code", False))
    declared = [str(tool_id) for role in run.workflow.roles for tool_id in role.tools]
    for resource in run.runtimes.values():
        configured = resource.options.get("tools", ())
        if isinstance(configured, (tuple, list)):
            declared.extend(str(tool_id) for tool_id in configured)
    try:
        tool_ids = set(declared)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"configured tools are invalid: {exc}") from exc
    if python_enabled and tool_ids:
        return "python_code_and_" + "_".join(sorted(tool_ids))
    if python_enabled:
        return "python_code"
    if tool_ids:
        return "_".join(sorted(tool_ids))
    return "no_tool"


def _jsonable(value: object) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    return value
