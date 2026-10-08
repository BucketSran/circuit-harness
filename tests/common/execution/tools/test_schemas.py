from __future__ import annotations

import pytest
from pydantic import ValidationError

from alphaapollo.common.artifacts.schemas import ArtifactRef
from alphaapollo.common.execution.tools.schemas import CostProgressRecord, ToolCallRecord


def test_tool_call_record_round_trip_preserves_artifacts_and_cost() -> None:
    record = ToolCallRecord(
        tool_id="bash",
        args={"cmd": "true"},
        artifacts=[ArtifactRef(id="artifact", hash="a" * 64)],
        cost=CostProgressRecord(tool_calls=1, wall_clock_s=0.5),
    )

    assert ToolCallRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_cost_record_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValidationError):
        CostProgressRecord(wall_clock_s=value)


def test_tool_call_record_rejects_non_json_arguments() -> None:
    with pytest.raises(ValidationError, match="JSON"):
        ToolCallRecord(tool_id="bash", args={"bad": ("tuple",)})
