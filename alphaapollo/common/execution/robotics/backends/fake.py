# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deterministic backend used for contract tests and local integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from alphaapollo.common.execution.robotics.schemas import (
    RobotAction,
    RobotObservation,
    RobotResetResult,
    RobotTask,
    RobotTransition,
)


class FakeBackend:
    """Replay a fixed transition script while enforcing backend lifecycle rules.

    Each scripted transition describes the delta caused by one action.
    ``steps_used`` exposed by this backend is the cumulative sum for the current
    episode, matching the state consumed by ``RobotEnvironment``.
    """

    def __init__(
        self,
        *,
        initial_observation: RobotObservation,
        transitions: Sequence[RobotTransition] = (),
        reset_info: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(initial_observation, RobotObservation):
            raise TypeError("initial_observation must be a RobotObservation")
        script = tuple(transitions)
        if any(not isinstance(transition, RobotTransition) for transition in script):
            raise TypeError("transitions must contain only RobotTransition values")
        for index, transition in enumerate(script[:-1]):
            if transition.done:
                raise ValueError(
                    f"transitions[{index}] is terminal, so later scripted transitions "
                    "would be unreachable"
                )

        self._initial_observation = initial_observation
        self._transitions = script
        self._reset_info = dict(reset_info or {})
        self._task: RobotTask | None = None
        self._observation: RobotObservation | None = None
        self._actions: list[RobotAction] = []
        self._next_transition = 0
        self._terminated = False
        self._truncated = False
        self._success: bool | None = None
        self._termination_reason: str | None = None
        self._steps_used = 0
        self._closed = False

    @property
    def task(self) -> RobotTask | None:
        """Return the task for the active episode, if reset has run."""

        return self._task

    @property
    def actions(self) -> tuple[RobotAction, ...]:
        """Return actions successfully matched to scripted transitions."""

        return tuple(self._actions)

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def success(self) -> bool | None:
        return self._success

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    @property
    def steps_used(self) -> int:
        return self._steps_used

    def reset(self, task: RobotTask) -> RobotResetResult:
        self._require_open()
        if not isinstance(task, RobotTask):
            raise TypeError("task must be a RobotTask")
        self._task = task
        self._observation = self._initial_observation
        self._actions.clear()
        self._next_transition = 0
        self._terminated = False
        self._truncated = False
        self._success = None
        self._termination_reason = None
        self._steps_used = 0
        return RobotResetResult(observation=self._observation, info=self._reset_info)

    def observe(self) -> RobotObservation:
        self._require_active("observe")
        assert self._observation is not None
        return self._observation

    def execute(self, action: RobotAction) -> RobotTransition:
        self._require_active("execute")
        if not isinstance(action, RobotAction):
            raise TypeError("action must be a RobotAction")
        if self._terminated or self._truncated:
            raise RuntimeError("cannot execute an action after the episode is terminal")
        if self._next_transition >= len(self._transitions):
            raise RuntimeError("fake backend transition script is exhausted")

        transition = self._transitions[self._next_transition]
        self._next_transition += 1
        self._actions.append(action)
        self._observation = transition.observation
        self._steps_used += transition.steps_used
        self._terminated = transition.terminated
        self._truncated = transition.truncated
        self._success = transition.success
        self._termination_reason = transition.termination_reason
        return transition

    def close(self) -> None:
        """Close the backend; repeated cleanup is harmless."""

        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("fake backend is closed")

    def _require_active(self, operation: str) -> None:
        self._require_open()
        if self._task is None:
            raise RuntimeError(f"reset must be called before {operation}")


__all__ = ["FakeBackend"]
