# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""In-memory assembly of one ordered environment episode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphaapollo.common.trajectory.episode import EpisodeResult, EpisodeTurn

if TYPE_CHECKING:
    from alphaapollo.common.environment.base import EnvironmentTransition


class EpisodeRecorder:
    """Assemble one ordered episode without owning its loop or stop policy."""

    def __init__(self, episode_id: str) -> None:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        self._episode_id = episode_id
        self._turns: list[EpisodeTurn] = []
        self._result: EpisodeResult | None = None

    @property
    def turns(self) -> tuple[EpisodeTurn, ...]:
        return tuple(self._turns)

    def record_turn(
        self,
        generation_request: object,
        generation_response: object,
        environment_transition: EnvironmentTransition,
    ) -> EpisodeTurn:
        if self._result is not None:
            raise RuntimeError("episode is already finished")
        if self._turns and self._turns[-1].environment_transition.done:
            raise RuntimeError("cannot record a turn after a terminal transition")
        turn = EpisodeTurn(
            index=len(self._turns),
            generation_request=generation_request,
            generation_response=generation_response,
            environment_transition=environment_transition,
        )
        self._turns.append(turn)
        return turn

    def finish(self, *, final_text: str, termination_reason: str) -> EpisodeResult:
        if self._result is not None:
            raise RuntimeError("episode is already finished")
        self._result = EpisodeResult(
            episode_id=self._episode_id,
            final_text=final_text,
            turns=tuple(self._turns),
            termination_reason=termination_reason,
        )
        return self._result
