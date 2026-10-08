from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from alphaapollo.common.artifacts.schemas import ArtifactRef, ProvenanceRef
from alphaapollo.common.execution import ArtifactStore


def test_put_ref_get_round_trip_is_content_addressed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put(
        b"proof",
        type_="witness",
        created_by="verifier",
        provenance=ProvenanceRef(id="run-1"),
        contamination_flags=["b", "a", "a"],
    )
    digest = hashlib.sha256(b"proof").hexdigest()

    assert artifact.id == digest
    assert artifact.hash == digest
    assert artifact.contamination_flags == ["a", "b"]
    assert artifact.content_ref == f"{digest[:2]}/{digest}"
    assert (store.root / artifact.content_ref).is_file()
    ref = store.ref(artifact)
    assert ref.type == "witness"
    assert ref.created_by == "verifier"
    assert ref.provenance == ProvenanceRef(id="run-1")
    assert ref.contamination_flags == ["a", "b"]
    assert store.get(ref) == b"proof"


def test_put_is_write_once_and_get_detects_tampering(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = store.put(b"original", type_="output", created_by="tool")
    path = store.root / (artifact.content_ref or "")
    path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="refusing false-success put"):
        store.put(b"original", type_="output", created_by="tool")

    with pytest.raises(ValueError, match="hash mismatch"):
        store.get(store.ref(artifact))


@pytest.mark.parametrize(
    "bad_hash",
    [None, "../escape", "deadbeef", "g" * 64, "A" * 64],
)
def test_get_rejects_missing_or_malformed_hashes(tmp_path: Path, bad_hash: str | None) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.get(ArtifactRef(id=bad_hash or "missing", hash=bad_hash))


def test_get_ignores_untrusted_location_and_uses_digest_path(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "root")
    artifact = store.put(b"safe", type_="file", created_by="tool")
    real_ref = store.ref(artifact)
    untrusted = ArtifactRef(
        id=real_ref.id,
        hash=real_ref.hash,
        location="../../outside",
    )

    assert store.get(untrusted) == b"safe"


def test_reference_resolves_across_store_instances(tmp_path: Path) -> None:
    first = ArtifactStore(tmp_path)
    ref = first.ref(first.put(b"shared", type_="blob", created_by="one"))
    assert ArtifactStore(tmp_path).get(ref) == b"shared"


def test_put_rejects_non_bytes_and_empty_metadata(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(TypeError):
        store.put("text", type_="text", created_by="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.put(b"x", type_="", created_by="x")
    with pytest.raises(ValueError):
        store.put(b"x", type_="text", created_by="")


def test_artifact_store_uses_owner_only_permissions_and_tightens_existing_blob(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    artifact = store.put(b"private prompt", type_="text/plain", created_by="test")
    ref = store.ref(artifact)
    blob = root / artifact.content_ref

    assert root.stat().st_mode & 0o777 == 0o700
    assert blob.parent.stat().st_mode & 0o777 == 0o700
    assert blob.stat().st_mode & 0o777 == 0o600

    blob.chmod(0o644)
    assert store.get(ref) == b"private prompt"
    assert blob.stat().st_mode & 0o777 == 0o600


def test_put_file_streams_content_into_the_content_addressed_store(tmp_path: Path) -> None:
    source = tmp_path / "large-output.log"
    content = (b"0123456789abcdef" * (128 * 1024)) + b"\nend\n"
    source.write_bytes(content)
    store = ArtifactStore(tmp_path / "artifacts")

    artifact = store.put_file(
        source,
        type_="bash_stdout",
        created_by="bash",
    )
    ref = store.ref(artifact)

    assert store.get(ref) == content
    assert store.verify(ref) is None


def test_verify_streams_and_detects_tampering_without_returning_content(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.put(b"predicted cells", type_="matrix", created_by="state")
    ref = store.ref(artifact)

    assert store.verify(ref) is None

    (store.root / (artifact.content_ref or "")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.verify(ref)
