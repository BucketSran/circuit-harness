# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"Complete episode records and trajectory persistence contracts."

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from alphaapollo.common.trajectory.schemas import TrajectoryEvent, TrajectoryEventType

if TYPE_CHECKING:
    from alphaapollo.common.environment.base import EnvironmentTransition


@dataclass(frozen=True, slots=True)
class EpisodeTurn:
    """One Generation request/response followed by its Environment transition.

    The original records are retained instead of projecting token evidence or
    Environment signals into a second schema.  Their semantic owners therefore
    remain the single source of truth.
    """

    index: int
    generation_request: object
    generation_response: object
    environment_transition: EnvironmentTransition

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("episode turn index must be a non-negative integer")
        if self.generation_request is None:
            raise TypeError("generation_request cannot be None")
        if self.generation_response is None:
            raise TypeError("generation_response cannot be None")
        if self.environment_transition is None:
            raise TypeError("environment_transition cannot be None")

    @property
    def env_reward(self) -> float:
        return self.environment_transition.reward


def validate_ordered_episode_turns(
    turns: Iterable[EpisodeTurn],
) -> tuple[EpisodeTurn, ...]:
    """Return validated turns in canonical episode order."""
    ordered = tuple(turns)
    if any(not isinstance(turn, EpisodeTurn) for turn in ordered):
        raise TypeError("turns must contain EpisodeTurn records")
    if tuple(turn.index for turn in ordered) != tuple(range(len(ordered))):
        raise ValueError("episode turn indices must be contiguous and start at zero")
    if any(turn.environment_transition.done for turn in ordered[:-1]):
        raise ValueError("a terminal Environment transition must be the final turn")
    return ordered


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Complete ordered episode, ending through Environment or Runtime policy."""

    episode_id: str
    final_text: str
    turns: tuple[EpisodeTurn, ...]
    termination_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        if not isinstance(self.final_text, str):
            raise TypeError("final_text must be a string")
        if not isinstance(self.termination_reason, str) or not self.termination_reason.strip():
            raise ValueError("termination_reason must be a non-empty string")
        object.__setattr__(self, "turns", validate_ordered_episode_turns(self.turns))

    @property
    def env_reward_total(self) -> float:
        return sum((turn.env_reward for turn in self.turns), 0.0)

    @property
    def terminal_success(self) -> bool | None:
        if not self.turns or not self.turns[-1].environment_transition.done:
            return None
        return self.turns[-1].environment_transition.success

    @property
    def env_steps(self) -> int:
        return len(self.turns)


@dataclass(frozen=True)
class TrajectoryQuery:
    """Neutral event filter; Learning applies DataProto field policy later."""

    session_id: str | None = None
    branch_id: str | None = None
    event_types: tuple[TrajectoryEventType, ...] = ()
    training_eligible_only: bool = False
    exclude_contaminated: bool = False


@runtime_checkable
class TrajectorySink(Protocol):
    def append(self, event: TrajectoryEvent) -> TrajectoryEvent: ...


@runtime_checkable
class TrajectoryReader(Protocol):
    def read_events(self, query: TrajectoryQuery | None = None) -> list[TrajectoryEvent]: ...

    def export(self, query: TrajectoryQuery | None = None) -> list[TrajectoryEvent]: ...


__all__ = [
    "EpisodeResult",
    "EpisodeTurn",
    "TrajectoryQuery",
    "TrajectoryReader",
    "TrajectorySink",
]
