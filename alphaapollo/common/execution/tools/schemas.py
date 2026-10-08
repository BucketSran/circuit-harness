"""Persistent records emitted by tool execution."""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from alphaapollo.common.artifacts.schemas import ArtifactRef, require_json_value

_STRICT = ConfigDict(extra="forbid", validate_assignment=True)


class CostProgressRecord(BaseModel):
    """Per-step resource delta or accumulated total."""

    model_config = _STRICT

    tokens: Annotated[StrictInt, Field(ge=0)] = 0
    reasoning_tokens: Annotated[StrictInt, Field(ge=0)] = 0
    wall_clock_s: Annotated[float, Field(ge=0.0)] = 0.0
    gpu_s: Annotated[float, Field(ge=0.0)] = 0.0
    cpu_s: Annotated[float, Field(ge=0.0)] = 0.0
    tool_calls: Annotated[StrictInt, Field(ge=0)] = 0
    verifier_calls: Annotated[StrictInt, Field(ge=0)] = 0
    sandbox_time_s: Annotated[float, Field(ge=0.0)] = 0.0
    cache_hits: Annotated[StrictInt, Field(ge=0)] = 0
    artifacts_produced: Annotated[StrictInt, Field(ge=0)] = 0
    trust_level_improvement: StrictInt = 0
    branch_progress_score: float = 0.0
    budget_consumed_fraction: float = 0.0

    @field_validator(
        "wall_clock_s",
        "gpu_s",
        "cpu_s",
        "sandbox_time_s",
        "branch_progress_score",
        "budget_consumed_fraction",
    )
    @classmethod
    def _reject_non_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("float fields must be finite")
        return value

    def __add__(self, other: CostProgressRecord) -> CostProgressRecord:
        return CostProgressRecord(
            tokens=self.tokens + other.tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            wall_clock_s=self.wall_clock_s + other.wall_clock_s,
            gpu_s=self.gpu_s + other.gpu_s,
            cpu_s=self.cpu_s + other.cpu_s,
            tool_calls=self.tool_calls + other.tool_calls,
            verifier_calls=self.verifier_calls + other.verifier_calls,
            sandbox_time_s=self.sandbox_time_s + other.sandbox_time_s,
            cache_hits=self.cache_hits + other.cache_hits,
            artifacts_produced=self.artifacts_produced + other.artifacts_produced,
            trust_level_improvement=(self.trust_level_improvement + other.trust_level_improvement),
            branch_progress_score=(self.branch_progress_score + other.branch_progress_score),
            budget_consumed_fraction=(
                self.budget_consumed_fraction + other.budget_consumed_fraction
            ),
        )


class ToolCallRecord(BaseModel):
    """Canonical audit record for one attempted tool execution."""

    model_config = _STRICT

    tool_id: str
    args: dict[str, Any] = Field(default_factory=dict)
    sandbox: dict[str, Any] | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    fs_diff: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    cost: CostProgressRecord = Field(default_factory=CostProgressRecord)

    @field_validator("args", "sandbox", "fs_diff")
    @classmethod
    def _json_execution_fields(cls, value: Any) -> Any:
        return require_json_value(value)


SCHEMA_MODELS: dict[str, type[BaseModel]] = {"tool_call_record": ToolCallRecord}

__all__ = ["CostProgressRecord", "SCHEMA_MODELS", "ToolCallRecord"]
