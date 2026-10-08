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

"""A deterministic external agent that needs no CLI, credentials, or network.

It exists so the Runtime, the composition root, and the shipped example can be
exercised end to end offline, and so a test can assert the projection of a tool
round trip without depending on a real agent choosing to call a tool.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alphaapollo.common.generation.base import ToolCall
from alphaapollo.reasoning.runtime.agent_runtime import AgentTask
from alphaapollo.reasoning.runtime.external_agent_runtime import (
    TERMINATION_REASONS,
    ExternalEvent,
    ExternalRunOutcome,
    encode_arguments,
)

__all__ = ["FakeExternalSession"]


class FakeExternalSession:
    """Replay one scripted external agent turn sequence for every task."""

    def __init__(
        self,
        *,
        answer: str = "fake external answer",
        reasoning: str | None = None,
        tool_id: str | None = None,
        tool_arguments: Any = None,
        tool_result: str = "",
        termination_reason: str = "final",
        usage: Mapping[str, Any] | None = None,
        echo_prompt: bool = False,
    ) -> None:
        if not isinstance(answer, str):
            raise TypeError("answer must be a string")
        if reasoning is not None and not isinstance(reasoning, str):
            raise TypeError("reasoning must be a string when provided")
        if tool_id is not None and (not isinstance(tool_id, str) or not tool_id.strip()):
            raise ValueError("tool_id must be a non-empty string when provided")
        if termination_reason not in TERMINATION_REASONS:
            raise ValueError(
                f"termination_reason must be one of {sorted(TERMINATION_REASONS)}, "
                f"got {termination_reason!r}"
            )
        if usage is not None and not isinstance(usage, Mapping):
            raise TypeError("usage must be a mapping when provided")
        self._answer = answer
        self._reasoning = reasoning
        self._tool_id = tool_id
        self._tool_arguments = tool_arguments if tool_arguments is not None else {}
        self._tool_result = tool_result
        self._termination_reason = termination_reason
        self._usage = dict(usage or {})
        self._echo_prompt = bool(echo_prompt)
        self.calls: list[str] = []
        self.workspaces: list[Path] = []
        self.closed = False

    def run(self, task: AgentTask, *, workspace: Path) -> ExternalRunOutcome:
        if self.closed:
            raise RuntimeError("FakeExternalSession is closed")
        if not isinstance(task, AgentTask):
            raise TypeError("task must be an AgentTask")
        self.calls.append(task.task_id)
        self.workspaces.append(workspace)

        events: list[ExternalEvent] = []
        if self._reasoning is not None:
            events.append(ExternalEvent(kind="reasoning", content=self._reasoning))
        if self._tool_id is not None:
            call_id = f"{task.task_id}-call-0"
            events.append(
                ExternalEvent(
                    kind="tool_call",
                    tool_call=ToolCall(
                        id=call_id,
                        name=self._tool_id,
                        arguments=encode_arguments(self._tool_arguments),
                    ),
                )
            )
            events.append(
                ExternalEvent(
                    kind="tool_result",
                    content=self._tool_result,
                    call_id=call_id,
                    tool_id=self._tool_id,
                )
            )
        answer = f"{self._answer}\n{task.prompt}" if self._echo_prompt else self._answer
        events.append(ExternalEvent(kind="message", content=answer))
        return ExternalRunOutcome(
            final_text=answer,
            events=tuple(events),
            termination_reason=self._termination_reason,
            usage=self._usage,
            # Host paths would end up in every persisted record; the workspace is
            # exposed through `workspaces` for assertions instead.
            provider_metadata={"scripted": True},
        )

    def close(self) -> None:
        self.closed = True
