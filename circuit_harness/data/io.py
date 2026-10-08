"""Atomic publication and integrity helpers for trajectory datasets."""

import hashlib
import os
import shutil
import uuid
from pathlib import Path

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def commit_directory(staged: Path, destination: Path, *, replace: bool) -> None:
    """Atomically publish a staged directory, restoring the old build on failure."""

    if destination.exists() and not replace:
        raise FileExistsError(destination)
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    moved_old = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_old = True
        os.replace(staged, destination)
    except BaseException:
        if moved_old and not destination.exists() and backup.exists():
            os.replace(backup, destination)
        raise
    else:
        if moved_old:
            try:
                shutil.rmtree(backup)
            except BaseException:
                # The second replace is the commit point: reporting failure now
                # would contradict the visible destination. Keep the uniquely
                # named old generation quarantined for later cleanup instead.
                pass
