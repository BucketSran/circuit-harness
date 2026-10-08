"""Serializable artifact records and JSON-value validation."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid", validate_assignment=True)
_STRICT_FROZEN = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)


def require_json_value(value: Any, *, path: str = "$") -> Any:
    """Reject values whose JSON representation would change their meaning."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_json_value(item, path=f"{path}[{index}]")
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} must contain only string object keys")
            require_json_value(item, path=f"{path}.{key}")
        return value
    raise ValueError(
        f"{path} contains non-JSON value of type {type(value).__name__}; "
        "use null/bool/string/int/finite-float/list/object"
    )


class Ref(BaseModel):
    model_config = _STRICT

    id: str
    location: str | None = None
    version: str | None = None
    hash: str | None = None


class ProvenanceRef(Ref):
    pass


class ArtifactRef(Ref):
    """One durable occurrence of content-addressed artifact data."""

    type: str | None = None
    created_by: str | None = None
    provenance: ProvenanceRef | None = None
    contamination_flags: list[str] = Field(default_factory=list)


class Artifact(BaseModel):
    model_config = _STRICT

    id: str
    type: str
    content_ref: str | None = None
    hash: str | None = None
    provenance: ProvenanceRef | None = None
    contamination_flags: list[str] = Field(default_factory=list)
    created_by: str | None = None


class ArrayArtifact(BaseModel):
    """A safe array artifact; no pickle or executable object graph."""

    model_config = _STRICT_FROZEN

    artifact: ArtifactRef
    encoding: Literal["numpy-npy", "torch-raw", "json-array"]
    dtype: str
    shape: list[int]
    byte_order: Literal["little", "big"] = "little"


SCHEMA_MODELS: dict[str, type[BaseModel]] = {"artifact": Artifact}

__all__ = [
    "ArrayArtifact",
    "Artifact",
    "ArtifactRef",
    "ProvenanceRef",
    "Ref",
    "SCHEMA_MODELS",
    "require_json_value",
]
