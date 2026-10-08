# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"Core Environment records, lifecycle, and Runtime adapter."

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, SupportsFloat, TypedDict, TypeVar

from alphaapollo.common.execution import ExecutionContext


@dataclass(frozen=True, slots=True)
class EnvironmentContext:
    """Resolved input required to initialize one environment episode."""

    session_id: str
    actor: str
    task_id: str | None = None
    branch_id: str = "main"
    round_index: int = 0
    seed: int | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    task_payload: dict[str, Any] = field(default_factory=dict, repr=False)
    environment_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: int = 0
    routing_key: str = ""

    def __post_init__(self) -> None:
        for name in ("session_id", "actor", "branch_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.task_id is not None and not isinstance(self.task_id, str):
            raise TypeError("task_id must be a string when provided")
        if (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index < 0
        ):
            raise ValueError("round_index must be a non-negative integer")
        if (
            isinstance(self.sample_id, bool)
            or not isinstance(self.sample_id, int)
            or self.sample_id < 0
        ):
            raise ValueError("sample_id must be a non-negative integer")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer when provided")
        for name in ("system_prompt", "user_prompt"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        if not isinstance(self.routing_key, str):
            raise TypeError("routing_key must be a string")
        for name in ("task_payload", "environment_config", "metadata"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a dict")
            object.__setattr__(self, name, copy.deepcopy(value))


@dataclass(frozen=True, slots=True)
class EnvironmentInitResult:
    """Model-facing prompt, raw task observation, and public episode metadata."""

    observation: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_observation: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    def __iter__(self) -> Iterator[Any]:
        """Preserve tuple unpacking at the current Runtime boundary."""

        yield self.observation
        yield self.metadata


@dataclass(frozen=True, slots=True)
class EnvironmentTransition(Mapping[str, Any]):
    """One raw environment transition, before Learning assigns credit."""

    observation: Any
    reward: float
    done: bool
    success: bool | None = None
    response_format_valid: bool = True
    env_action_valid: bool = True
    termination_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    previous_observation: Any | None = None
    raw_observation: Any | None = None
    executed_action: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.done, bool):
            raise TypeError("done must be a bool")
        if self.done and not self.termination_reason:
            raise ValueError("terminal transitions require termination_reason")
        if not self.done and self.termination_reason is not None:
            raise ValueError("non-terminal transitions cannot have termination_reason")
        if not self.done and self.success is not None:
            raise ValueError("success is defined only for terminal transitions")
        if self.success is not None and not isinstance(self.success, bool):
            raise TypeError("success must be a bool when provided")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        reward = self.reward
        if isinstance(reward, (bool, str, bytes)):
            raise TypeError("reward must be a finite number")
        try:
            normalized_reward = float(reward)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("reward must be a finite number") from exc
        if not math.isfinite(normalized_reward):
            raise ValueError("reward must be finite")
        object.__setattr__(self, "reward", normalized_reward)
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    @property
    def action_valid(self) -> bool:
        """Backward-compatible combined validity flag."""

        return self.response_format_valid and self.env_action_valid

    def to_legacy_output(self) -> dict[str, Any]:
        """Project the canonical transition onto the current Runtime Env seam."""

        metadata = copy.deepcopy(self.metadata)
        metadata.update(
            success=self.success,
            response_format_valid=self.response_format_valid,
            env_action_valid=self.env_action_valid,
            action_valid=self.action_valid,
            termination_reason=self.termination_reason,
            previous_observation=copy.deepcopy(self.previous_observation),
            raw_observation=copy.deepcopy(self.raw_observation),
            executed_action=copy.deepcopy(self.executed_action),
        )
        return {
            "observations": self.observation,
            "reward": self.reward,
            "done": self.done,
            "metadata": metadata,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_legacy_output()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("observations", "reward", "done", "metadata"))

    def __len__(self) -> int:
        return 4


class EnvironmentState(str, Enum):
    """Valid lifecycle states for one environment episode."""

    CREATED = "created"
    ACTIVE = "active"
    TERMINATED = "terminated"
    FAILED = "failed"
    CLOSED = "closed"


class EnvironmentLifecycleError(RuntimeError):
    """Raised when a caller violates the environment lifecycle."""


@dataclass(frozen=True, slots=True)
class EnvironmentObservation:
    """Stable, JSON-compatible observation returned to an agent runtime."""

    kind: str
    content: str
    call_id: str | None = None
    tool_id: str | None = None
    attempted: bool = False
    exit_code: int | None = None
    error_stage: str | None = None
    error_code: str | None = None
    contamination_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["contamination_flags"] = list(self.contamination_flags)
        return value


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class EnvStepOutput(TypedDict):
    """Result of one environment action.

    The field names and required-key behavior match the legacy v2/v3 contract.
    ``metadata`` may contain ``None`` but remains a required TypedDict key.
    """

    observations: Any
    reward: SupportsFloat
    done: bool
    metadata: dict[str, Any] | None


class Env(Generic[ObsType, ActType]):
    """Small lifecycle contract implemented by concrete environments.

    ``init`` initializes one episode. ``step`` executes one model action.
    ``close`` releases resources. The contract deliberately does not invent a
    Gym-style ``reset`` method that the legacy AlphaApollo interface never had.
    """

    def step(self, action: ActType) -> EnvStepOutput:
        """Execute ``action`` and return observation, reward, done, and metadata."""
        raise NotImplementedError

    def init(self, *args: Any, **kwargs: Any) -> tuple[ObsType, dict[str, Any]]:
        """Initialize an episode and return its first observation and metadata.

        Positional arguments preserve the legacy v2 contract. Keyword arguments
        support configuration-oriented environment implementations without
        forcing owners to encode named settings positionally.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release resources; calling close repeatedly should be harmless."""

    def __str__(self) -> str:
        return f"Env({type(self).__name__})"

    def __enter__(self) -> Env[ObsType, ActType]:
        return self

    def __exit__(self, *args: Any) -> bool:
        self.close()
        return False


class BaseEnvironment(Env[ObsType, ActType], ABC):
    """Canonical contract for one task-agnostic, single-episode environment.

    Environment implementations own action interpretation and termination.
    They return raw task reward and validity signals; batching, model decoding,
    credit assignment, and parameter updates live outside this layer.
    """

    @abstractmethod
    def init(self, context: EnvironmentContext) -> EnvironmentInitResult:
        """Initialize exactly one episode from resolved runtime context."""

    @abstractmethod
    def step(self, action: ActType) -> EnvironmentTransition:
        """Apply one model action; the concrete environment decides ``done``."""

    def terminate(self, reason: str = "runtime_requested") -> None:
        """Allow a runtime to stop an active episode before task termination."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("termination reason must be non-empty")

    @abstractmethod
    def close(self) -> None:
        """Release resources; repeated calls must be harmless."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class EnvironmentSession:
    """Mutable lifecycle record for one Solver or Verifier environment."""

    session_id: str
    actor: str
    branch_id: str = "main"
    round_index: int = 0
    workspace_snapshot_ref: str | None = None
    execution_mode: str = "default"
    state: EnvironmentState = EnvironmentState.CREATED
    step_index: int = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None
    termination_reason: str | None = None
    cleanup_errors: list[str] = field(default_factory=list)
    event_errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index < 0
        ):
            raise ValueError("round_index must be a non-negative integer")
        # Reuse the canonical execution validation rather than maintaining a
        # second set of rules for session, branch, actor, and workspace values.
        self.execution_context()

    def execution_context(self, *, timeout_s: float | None = None) -> ExecutionContext:
        return ExecutionContext(
            session_id=self.session_id,
            branch_id=self.branch_id,
            workspace_snapshot_ref=self.workspace_snapshot_ref,
            actor=self.actor,
            mode=self.execution_mode,
            timeout_s=timeout_s,
        )

    def activate(self) -> None:
        if self.state is not EnvironmentState.CREATED:
            raise EnvironmentLifecycleError(f"init requires created state, got {self.state.value}")
        self.state = EnvironmentState.ACTIVE
        self.started_at = _now()

    def require_active(self) -> None:
        if self.state is not EnvironmentState.ACTIVE:
            raise EnvironmentLifecycleError(f"step requires active state, got {self.state.value}")

    def advance(self) -> None:
        self.require_active()
        self.step_index += 1

    def terminate(self, reason: str) -> None:
        self.require_active()
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("termination reason must be non-empty")
        self.state = EnvironmentState.TERMINATED
        self.termination_reason = reason
        self.ended_at = _now()

    def fail(self, reason: str) -> None:
        if self.state is not EnvironmentState.ACTIVE:
            raise EnvironmentLifecycleError(
                f"failure requires active state, got {self.state.value}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failure reason must be non-empty")
        self.state = EnvironmentState.FAILED
        self.termination_reason = reason
        self.ended_at = _now()

    def close(self) -> None:
        if self.state is EnvironmentState.CLOSED:
            return
        self.state = EnvironmentState.CLOSED
        self.ended_at = self.ended_at or _now()

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "actor": self.actor,
            "branch_id": self.branch_id,
            "round_index": self.round_index,
            "workspace_snapshot_ref": self.workspace_snapshot_ref,
            "execution_mode": self.execution_mode,
            "state": self.state.value,
            "step_index": self.step_index,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "termination_reason": self.termination_reason,
            "cleanup_errors": list(self.cleanup_errors),
            "event_errors": list(self.event_errors),
        }


class EnvironmentRuntimeError(RuntimeError):
    """A terminal Environment failure that must not be scored as a final answer."""

    def __init__(self, termination_reason: str) -> None:
        self.termination_reason = termination_reason
        super().__init__(f"environment terminated with failure: {termination_reason}")


class RuntimeEnvironmentAdapter(Env[str, str]):
    """Bind episode identity to the MVP Runtime's ``init(system, prompt)`` seam.

    The adapter contains no task termination policy. A concrete Environment
    interprets the model action and decides whether the episode is complete.
    """

    def __init__(
        self,
        environment: BaseEnvironment[Any, str],
        *,
        session_id: str,
        actor: str,
        task_id: str | None = None,
        branch_id: str = "main",
        round_index: int = 0,
        seed: int | None = None,
        workspace_snapshot_ref: str | None = None,
        task_payload: dict[str, Any] | None = None,
        environment_config: dict[str, Any] | None = None,
        episode_context: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(environment, BaseEnvironment):
            raise TypeError("runtime adapter requires BaseEnvironment")
        self._environment = environment
        self._identity = {
            "session_id": session_id,
            "actor": actor,
            "task_id": task_id,
            "branch_id": branch_id,
            "round_index": round_index,
            "seed": seed,
        }
        self._workspace_snapshot_ref = workspace_snapshot_ref
        self._task_payload = copy.deepcopy(task_payload or {})
        self._environment_config = copy.deepcopy(environment_config or {})
        self._episode_context = copy.deepcopy(episode_context or {})

    @property
    def environment(self) -> BaseEnvironment[Any, str]:
        return self._environment

    def init(self, system: str, prompt: str) -> tuple[str, dict[str, Any]]:
        if not isinstance(system, str) or not isinstance(prompt, str):
            raise TypeError("system and prompt must be strings")
        metadata = copy.deepcopy(self._episode_context)
        if self._workspace_snapshot_ref is not None:
            metadata["workspace_snapshot_ref"] = self._workspace_snapshot_ref
        result = self._environment.init(
            EnvironmentContext(
                **self._identity,
                system_prompt=system,
                user_prompt=prompt,
                task_payload=self._task_payload,
                environment_config=self._environment_config,
                metadata=metadata,
            )
        )
        return str(result.observation), copy.deepcopy(result.metadata)

    def step(self, action: str) -> EnvStepOutput:
        transition = self._environment.step(action)
        if not isinstance(transition, EnvironmentTransition):
            raise TypeError("BaseEnvironment.step() must return EnvironmentTransition")
        if transition.done and transition.termination_reason == "internal_error":
            raise EnvironmentRuntimeError(transition.termination_reason)
        result = transition.to_legacy_output()
        if transition.done and transition.termination_reason == "model_output":
            # AgentRuntime already retains the raw model output as final_text.
            result["observations"] = ""
        return result

    def close(self) -> None:
        self._environment.close()

    def terminate(self, reason: str) -> None:
        self._environment.terminate(reason)
