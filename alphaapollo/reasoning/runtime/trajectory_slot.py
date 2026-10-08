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

"""Per-task mutable state used by the batched AlphaApollo agent loop."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from alphaapollo.reasoning._immutable import _thaw_json
from alphaapollo.reasoning.runtime.agent_runtime import (
    AgentResult,
    AgentTask,
    AgentTurn,
)

__all__: list[str] = []


@dataclass(slots=True)
class _TrajectorySlot:
    """Independent transcript, accounting, recorder, and outcome for one task."""

    task: AgentTask
    position: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: list[AgentTurn] = field(default_factory=list)
    environment_init: Any | None = None
    active: bool = False
    termination_requested: bool = False
    final_text: str = ""
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.position, bool) or not isinstance(self.position, int):
            raise TypeError("position must be an int")
        if self.position < 0:
            raise ValueError("position must be non-negative")

    @property
    def next_turn_index(self) -> int:
        return len(self.turns)

    def initialize(self, result: Any) -> None:
        if self.active or self.environment_init is not None:
            raise RuntimeError("trajectory slot is already initialized")
        if not hasattr(result, "observation") or not hasattr(result, "metadata"):
            raise TypeError("environment init result must expose observation and metadata")
        self.environment_init = result
        observation = result.observation
        if observation in (None, ""):
            # No environment-authored first message: `task.prompt` already
            # carries any workflow-memory context appended by the executor.
            initial_content: Any = self.task.prompt
        else:
            # The environment authors the first message and must never see
            # memory (its task definition is validated against the unaugmented
            # prompt), so the executor ships the agent-facing context through
            # metadata and it is attached here, after the environment content.
            initial_content = copy.deepcopy(observation)
            memory_context = (
                self.task.metadata.get("agent_memory_context")
                if isinstance(self.task.metadata, Mapping)
                else None
            )
            if isinstance(memory_context, str) and memory_context.strip():
                if isinstance(initial_content, str):
                    initial_content = f"{initial_content}\n\n{memory_context}"
                elif isinstance(initial_content, Sequence) and not isinstance(
                    initial_content, (str, bytes)
                ):
                    initial_content = [
                        *initial_content,
                        {"type": "text", "text": memory_context},
                    ]
                else:
                    raise TypeError(
                        "cannot attach agent memory context to an environment "
                        f"observation of type {type(initial_content).__name__}"
                    )
        messages: list[dict[str, Any]] = []
        if self.task.system:
            messages.append({"role": "system", "content": self.task.system})
        messages.append({"role": "user", "content": copy.deepcopy(initial_content)})
        self.messages = messages
        self.active = True

    def message_snapshot(self) -> tuple[Mapping[str, Any], ...]:
        if not self.active:
            raise RuntimeError("cannot build a Generation request for an inactive slot")
        return tuple(copy.deepcopy(message) for message in self.messages)

    def record(
        self,
        turn: AgentTurn,
        *,
        final_text: str,
        continuation_messages: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if not self.active:
            raise RuntimeError("cannot record a turn for an inactive slot")
        if turn.index != self.next_turn_index:
            raise RuntimeError(
                f"turn index {turn.index} does not match next index {self.next_turn_index}"
            )
        if not isinstance(final_text, str):
            raise TypeError("final_text must be a string")
        self.turns.append(turn)
        self.final_text = final_text
        for message in continuation_messages:
            if not isinstance(message, Mapping):
                raise TypeError("continuation messages must be mappings")
            self.messages.append(copy.deepcopy(dict(message)))

    def finish(self, reason: str) -> None:
        if not self.active:
            raise RuntimeError("cannot finish an inactive slot")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("termination reason must be non-empty")
        self.active = False
        self.termination_reason = reason

    def claim_termination(self) -> bool:
        """Claim the one allowed Runtime-driven terminate call for this slot."""

        if self.termination_requested:
            return False
        self.termination_requested = True
        return True

    def result(self) -> AgentResult:
        if self.active or self.termination_reason is None:
            raise RuntimeError("cannot produce a result for an active trajectory slot")
        metadata = _thaw_json(self.task.metadata)
        initial_observation = None
        if self.environment_init is not None and self.environment_init.metadata:
            metadata["environment_init"] = _thaw_json(self.environment_init.metadata)
        if self.environment_init is not None:
            initial_observation = self.environment_init.raw_observation
            if initial_observation is None:
                initial_observation = self.environment_init.observation
        return AgentResult(
            task_id=self.task.task_id,
            final_text=self.final_text,
            turns=tuple(self.turns),
            termination_reason=self.termination_reason,
            metadata=metadata,
            initial_observation=copy.deepcopy(initial_observation),
        )
