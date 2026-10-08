from __future__ import annotations

import pytest
from pydantic import ValidationError

from alphaapollo.common.artifacts.schemas import (
    Artifact,
    ArtifactRef,
    ProvenanceRef,
    require_json_value,
)


def test_artifact_reference_preserves_occurrence_metadata() -> None:
    provenance = ProvenanceRef(id="solver", version="1")
    artifact = Artifact(
        id="a" * 64,
        type="text/plain",
        hash="a" * 64,
        provenance=provenance,
        contamination_flags=["private-input"],
        created_by="solver",
    )
    ref = ArtifactRef(
        id=artifact.id,
        hash=artifact.hash,
        type=artifact.type,
        provenance=artifact.provenance,
        contamination_flags=artifact.contamination_flags,
        created_by=artifact.created_by,
    )

    assert ArtifactRef.model_validate_json(ref.model_dump_json()) == ref


@pytest.mark.parametrize("unsafe", [float("nan"), float("inf"), ("tuple",), object()])
def test_json_value_validation_rejects_lossy_values(unsafe: object) -> None:
    with pytest.raises(ValueError, match="JSON|finite"):
        require_json_value({"unsafe": unsafe})


def test_artifact_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Artifact(id="artifact", type="text/plain", unexpected=True)  # type: ignore[call-arg]
