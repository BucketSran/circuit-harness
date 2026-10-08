"""Private, byte-verifiable Analog session and Pi trace archive."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from .archive import durable_json, private_root, sync_directory
from .journal import file_digest

_MAX_FILES = 5000
_MAX_BYTES = 256 * 1024 * 1024
_AGENT_FILES = (
    "operator.json",
    "pi-outcome.json",
    "pi-events.jsonl",
    "runtime-result.json",
    "model-budget.jsonl",
    "report.json",
    "launcher.json",
    "collection.json",
    "agent-input.json",
)


def _add_tree(members: dict[str, Path], prefix: str, root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{prefix} must be a regular directory")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symbolic link in episode evidence")
        if path.is_file():
            members[f"{prefix}/{path.relative_to(root).as_posix()}"] = path
        elif not path.is_dir():
            raise ValueError("unsupported episode evidence entry")


def _members(
    session: Path, agent: Path | None, final: Path | None
) -> tuple[tuple[dict, str | None], dict[str, Path]]:
    if session.is_symlink() or not session.is_dir():
        raise ValueError("session must be a regular directory")
    frozen_path = session / "frozen.json"
    digest = None
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        candidate = session / "frozen/circuit.spi"
        digest = frozen["candidate_sha256"]
        if candidate.is_symlink() or not candidate.is_file() or file_digest(candidate) != digest:
            raise ValueError("frozen candidate changed")
    else:
        from .analog_session import END_REASONS

        ended = json.loads((session / "episode-end.json").read_text())
        if (
            ended.get("state") != "missing_candidate"
            or ended.get("candidate_sha256") is not None
            or ended.get("collection_source") != "episode_end"
            or ended.get("agent_submitted") is not False
            or ended.get("termination_reason") not in END_REASONS
            or (session / "candidate.spi").exists()
            or (session / "frozen").exists()
        ):
            raise ValueError("missing candidate receipt contradicts session evidence")
        if final is not None:
            raise ValueError("cannot archive final scoring without a candidate")
    if final is not None:
        score_input = final / "inputs/circuit.spi"
        if (
            score_input.is_symlink()
            or not score_input.is_file()
            or file_digest(score_input) != digest
        ):
            raise ValueError("final scorer candidate differs from frozen submission")
    paths: dict[str, Path] = {}
    for name in (
        "session.json",
        "frozen.json",
        "candidate.spi",
        "original-instruction.md",
        "ending.json",
        "episode-end.json",
    ):
        path = session / name
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise ValueError("symbolic link in episode evidence")
            paths["session/" + name] = path
    for name in ("public", "frozen", "actions", "requests", "simulations", "candidates"):
        path = session / name
        if path.exists():
            _add_tree(paths, "session/" + name, path)
    # Refuse even an unselected link in session scratch; it may conceal evidence.
    if any(path.is_symlink() for path in session.rglob("*")):
        raise ValueError("symbolic link in episode evidence")
    if agent is not None:
        if agent.is_symlink() or not agent.is_dir():
            raise ValueError("agent evidence must be a regular directory")
        for name in _AGENT_FILES:
            path = agent / name
            if path.exists():
                if path.is_symlink() or not path.is_file():
                    raise ValueError("symbolic link in episode evidence")
                paths["agent/" + name] = path
        tools = agent / "tools"
        if tools.exists():
            if tools.is_symlink() or not tools.is_dir():
                raise ValueError("agent tools must be a regular directory")
            for path in tools.rglob("*"):
                if path.is_symlink():
                    raise ValueError("symbolic link in episode evidence")
                if path.is_file() and path.name in {
                    "request.json",
                    "response.json",
                    "timing.json",
                    "transport.json",
                }:
                    paths["agent/tools/" + path.relative_to(tools).as_posix()] = path
    if final is not None:
        _add_tree(paths, "final", final)
    members = {
        name: {"sha256": file_digest(path), "bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    if len(members) > _MAX_FILES or sum(value["bytes"] for value in members.values()) > _MAX_BYTES:
        raise ValueError("episode evidence exceeds archive limit")
    return (members, digest), paths


def _verify_package(package: Path, members: dict) -> None:
    seen = set()
    try:
        with tarfile.open(package, "r:gz") as archive:
            for member in archive:
                name = member.name
                if (
                    not member.isfile()
                    or name in seen
                    or name not in members
                    or member.size != members[name]["bytes"]
                ):
                    raise ValueError("unsafe episode archive member")
                with archive.extractfile(member) as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != members[name]["sha256"]:
                    raise ValueError("episode archive member changed")
                seen.add(name)
    except tarfile.TarError as error:
        raise ValueError("invalid episode archive") from error
    if seen != members.keys():
        raise ValueError("episode archive members missing")


def verify_episode(record: Path) -> dict:
    record = Path(record)
    receipt = json.loads((record / "receipt.json").read_text(encoding="utf-8"))
    package = record / "episode.tar.gz"
    if (
        package.is_symlink()
        or package.stat().st_size != receipt["bytes"]
        or file_digest(package) != receipt["sha256"]
    ):
        raise ValueError("episode archive package changed")
    _verify_package(package, receipt["members"])
    return receipt


def archive_episode(
    session: Path,
    archive_root: Path,
    episode_id: str,
    *,
    agent: Path | None = None,
    final: Path | None = None,
) -> dict:
    """Archive selected private evidence, never the model key or whole Pi home."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", episode_id):
        raise ValueError("invalid episode ID")
    session = Path(session).absolute()
    agent = Path(agent).absolute() if agent is not None else None
    final = Path(final).absolute() if final is not None else None
    (members, digest), paths = _members(session, agent, final)
    parent = private_root(Path(archive_root)) / "episodes"
    private_root(parent)
    record = private_root(parent / episode_id)
    receipt_path = record / "receipt.json"
    if receipt_path.exists():
        receipt = verify_episode(record)
        if receipt["members"] != members or receipt["candidate_sha256"] != digest:
            raise ValueError("episode archive differs from current evidence")
        return receipt
    if any(record.iterdir()):
        raise ValueError("incomplete episode archive exists")
    with tempfile.TemporaryDirectory(prefix="analog-archive-", dir=session) as temporary:
        packed = Path(temporary) / "episode.tar.gz"
        with tarfile.open(packed, "w:gz") as archive:
            for name in sorted(paths):
                archive.add(paths[name], arcname=name, recursive=False)
        _verify_package(packed, members)
        partial = record / "episode.tar.gz.partial"
        with packed.open("rb") as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output)
            output.flush()
            os.fsync(output.fileno())
        package_hash = file_digest(packed)
        if file_digest(partial) != package_hash:
            raise ValueError("episode archive copy changed")
        _verify_package(partial, members)
        partial.replace(record / "episode.tar.gz")
        sync_directory(record)
        receipt = {
            "schema_version": 1,
            "candidate_sha256": digest,
            "members": members,
            "sha256": package_hash,
            "bytes": packed.stat().st_size,
        }
        durable_json(receipt_path, receipt)
    return receipt
