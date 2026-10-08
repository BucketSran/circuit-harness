from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from alphaapollo.common.artifacts.schemas import ArrayArtifact, ArtifactRef
from alphaapollo.learning.adapters.schemas import (
    INFERENCE_SCHEMA_VERSION,
    CaptureDisposition,
    CaptureStatus,
    InferenceCaptureRecord,
    require_supported_inference_schema,
)


def test_inference_capture_record_round_trip() -> None:
    array = ArrayArtifact(
        artifact=ArtifactRef(id="artifact", hash="a" * 64),
        encoding="numpy-npy",
        dtype="int64",
        shape=[1],
    )
    record = InferenceCaptureRecord(
        record_id="capture",
        created_at=datetime.now(timezone.utc),
        model="model",
        batch_size=1,
        tensor_batch={"input_ids": array},
        dispositions=[
            CaptureDisposition(
                field="input_ids",
                status=CaptureStatus.CAPTURED,
                reason_code="selected",
            )
        ],
    )

    restored = InferenceCaptureRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert restored.schema_version == INFERENCE_SCHEMA_VERSION


def test_inference_schema_rejects_duplicate_dispositions_and_unknown_major() -> None:
    disposition = CaptureDisposition(
        field="input_ids",
        status=CaptureStatus.CAPTURED,
        reason_code="selected",
    )
    with pytest.raises(ValidationError, match="exactly once"):
        InferenceCaptureRecord(
            record_id="capture",
            created_at=datetime.now(timezone.utc),
            model="model",
            dispositions=[disposition, disposition],
        )
    with pytest.raises(ValueError, match="unsupported"):
        require_supported_inference_schema("2.0")
