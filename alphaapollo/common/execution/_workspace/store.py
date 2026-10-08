# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Content-addressed artifact storage for execution outputs."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from alphaapollo.common.artifacts.schemas import Artifact, ArtifactRef, ProvenanceRef

_SHA256_HEX_LENGTH = 64
_HEX_CHARS = frozenset("0123456789abcdef")


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_sha256_digest(digest: str | None) -> str:
    if not isinstance(digest, str):
        raise ValueError("ArtifactRef hash must be a sha256 hex string")
    if len(digest) != _SHA256_HEX_LENGTH or any(ch not in _HEX_CHARS for ch in digest):
        raise ValueError("ArtifactRef hash must be a lowercase 64-character sha256 hex digest")
    return digest


# Content-addressed artifacts ------------------------------------------------
class ArtifactStore:
    """Local write-once store at ``root/<digest[:2]>/<digest>``."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._root.chmod(0o700)

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, digest: str) -> Path:
        return self._root / digest[:2] / digest

    def put(
        self,
        content: bytes,
        *,
        type_: str,
        created_by: str,
        provenance: ProvenanceRef | None = None,
        contamination_flags: list[str] | None = None,
    ) -> Artifact:
        if not isinstance(content, bytes):
            raise TypeError("ArtifactStore.put expects bytes")
        if not type_.strip():
            raise ValueError("artifact type must be non-empty")
        if not created_by.strip():
            raise ValueError("created_by must be non-empty")

        digest = hashlib.sha256(content).hexdigest()
        path = self._path_for(digest)
        shard_existed = path.parent.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        if not shard_existed:
            _fsync_directory(self._root)
        if not path.exists():
            with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                temporary_path = Path(tmp.name)
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
            try:
                if not path.exists():
                    os.replace(temporary_path, path)
                    _fsync_directory(path.parent)
            finally:
                temporary_path.unlink(missing_ok=True)

        path.chmod(0o600)
        stored = path.read_bytes()
        actual = hashlib.sha256(stored).hexdigest()
        if actual != digest:
            raise ValueError(
                f"artifact content hash mismatch for {digest}; refusing false-success put"
            )

        return Artifact(
            id=digest,
            type=type_,
            content_ref=f"{digest[:2]}/{digest}",
            hash=digest,
            provenance=provenance,
            contamination_flags=sorted(set(contamination_flags or [])),
            created_by=created_by,
        )

    def put_file(
        self,
        source: Path,
        *,
        type_: str,
        created_by: str,
        provenance: ProvenanceRef | None = None,
        contamination_flags: list[str] | None = None,
    ) -> Artifact:
        """Store a file without loading its full content into memory."""
        source = Path(source)
        if not source.is_file():
            raise ValueError("ArtifactStore.put_file source must be a file")
        if not type_.strip():
            raise ValueError("artifact type must be non-empty")
        if not created_by.strip():
            raise ValueError("created_by must be non-empty")

        digest_builder = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        path = self._path_for(digest)
        shard_existed = path.parent.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        if not shard_existed:
            _fsync_directory(self._root)
        if not path.exists():
            with (
                source.open("rb") as input_handle,
                NamedTemporaryFile(
                    dir=path.parent,
                    delete=False,
                ) as tmp,
            ):
                temporary_path = Path(tmp.name)
                shutil.copyfileobj(input_handle, tmp, length=1024 * 1024)
                tmp.flush()
                os.fsync(tmp.fileno())
            try:
                if not path.exists():
                    os.replace(temporary_path, path)
                    _fsync_directory(path.parent)
            finally:
                temporary_path.unlink(missing_ok=True)

        path.chmod(0o600)
        actual_builder = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                actual_builder.update(chunk)
        if actual_builder.hexdigest() != digest:
            raise ValueError(
                f"artifact content hash mismatch for {digest}; refusing false-success put_file"
            )

        return Artifact(
            id=digest,
            type=type_,
            content_ref=f"{digest[:2]}/{digest}",
            hash=digest,
            provenance=provenance,
            contamination_flags=sorted(set(contamination_flags or [])),
            created_by=created_by,
        )

    def ref(self, artifact: Artifact) -> ArtifactRef:
        digest = _validate_sha256_digest(artifact.hash)
        if artifact.id != digest:
            raise ValueError("Artifact id and hash must match")
        return ArtifactRef(
            id=artifact.id,
            location=artifact.content_ref,
            hash=digest,
            type=artifact.type,
            created_by=artifact.created_by,
            provenance=artifact.provenance,
            contamination_flags=list(artifact.contamination_flags),
        )

    def get(self, ref: ArtifactRef) -> bytes:
        digest = _validate_sha256_digest(ref.hash)
        if ref.id != digest:
            raise ValueError("ArtifactRef id and hash must match")
        path = self._path_for(digest)
        if not path.is_file():
            raise ValueError(f"artifact content is missing for {digest}")
        path.chmod(0o600)
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != digest:
            raise ValueError(f"artifact content hash mismatch for {digest}")
        return content

    def verify(self, ref: ArtifactRef) -> None:
        """Verify an artifact in bounded memory without returning its content."""

        digest = _validate_sha256_digest(ref.hash)
        if ref.id != digest:
            raise ValueError("ArtifactRef id and hash must match")
        path = self._path_for(digest)
        if not path.is_file():
            raise ValueError(f"artifact content is missing for {digest}")
        actual_builder = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                actual_builder.update(chunk)
        if actual_builder.hexdigest() != digest:
            raise ValueError(f"artifact content hash mismatch for {digest}")
