"""Durable records consumed by Learning adapters."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from alphaapollo.common.artifacts.schemas import ArrayArtifact, ArtifactRef, require_json_value

INFERENCE_SCHEMA_VERSION = "1.0"
SUPPORTED_INFERENCE_SCHEMA_MAJOR = 1
_STRICT_FROZEN = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)


def inference_schema_major(version: str) -> int:
    try:
        return int(version.split(".", maxsplit=1)[0])
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid inference schema version {version!r}") from exc


def require_supported_inference_schema(version: str) -> None:
    major = inference_schema_major(version)
    if major != SUPPORTED_INFERENCE_SCHEMA_MAJOR:
        raise ValueError(
            "unsupported inference schema major "
            f"{major}; reader supports {SUPPORTED_INFERENCE_SCHEMA_MAJOR}"
        )


class CaptureStatus(str, Enum):
    CAPTURED = "captured"
    UNAVAILABLE = "unavailable"
    OMITTED = "omitted"
    REJECTED = "rejected"


class CaptureDisposition(BaseModel):
    model_config = _STRICT_FROZEN

    field: str
    status: CaptureStatus
    reason_code: str
    detail: str | None = None


class InferenceCaptureRecord(BaseModel):
    """Validated backend-neutral record restored into a Learning container."""

    model_config = _STRICT_FROZEN

    record_id: str
    created_at: AwareDatetime
    schema_version: str = INFERENCE_SCHEMA_VERSION
    hash_algorithm: Literal["sha256"] = "sha256"
    session_id: str | None = None
    request_id: str | None = None
    model: str
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    input_batch_size: int | None = Field(default=None, ge=0)
    input_tensor_batch: dict[str, ArrayArtifact] = Field(default_factory=dict)
    input_non_tensor_batch: dict[str, ArrayArtifact] = Field(default_factory=dict)
    input_meta_info: dict[str, Any] = Field(default_factory=dict)
    input_manifest_ref: ArtifactRef | None = None
    output_manifest_ref: ArtifactRef | None = None
    batch_size: int | None = Field(default=None, ge=0)
    output_to_input: ArrayArtifact | None = None
    tensor_batch: dict[str, ArrayArtifact] = Field(default_factory=dict)
    non_tensor_batch: dict[str, ArrayArtifact] = Field(default_factory=dict)
    meta_info: dict[str, Any] = Field(default_factory=dict)
    backend_metadata: dict[str, Any] = Field(default_factory=dict)
    raw_backend: dict[str, Any] = Field(default_factory=dict)
    dispositions: list[CaptureDisposition] = Field(default_factory=list)

    @model_validator(mode="after")
    def _dispositions_are_unique(self) -> InferenceCaptureRecord:
        for field_name in (
            "request",
            "response",
            "input_meta_info",
            "meta_info",
            "backend_metadata",
            "raw_backend",
        ):
            require_json_value(getattr(self, field_name), path=f"$.{field_name}")
        fields = [item.field for item in self.dispositions]
        if len(fields) != len(set(fields)):
            raise ValueError("capture dispositions must contain each field exactly once")
        if self.output_to_input is not None:
            if self.batch_size is None or self.input_batch_size is None:
                raise ValueError("output_to_input requires both batch sizes")
            if self.output_to_input.shape != [self.batch_size]:
                raise ValueError("output_to_input must contain one index per output row")
            if self.output_to_input.dtype != "int64":
                raise ValueError("output_to_input must use int64 indices")
        return self


SCHEMA_MODELS: dict[str, type[BaseModel]] = {"inference_capture_record": InferenceCaptureRecord}

__all__ = [
    "CaptureDisposition",
    "CaptureStatus",
    "INFERENCE_SCHEMA_VERSION",
    "InferenceCaptureRecord",
    "SCHEMA_MODELS",
    "SUPPORTED_INFERENCE_SCHEMA_MAJOR",
    "inference_schema_major",
    "require_supported_inference_schema",
]
