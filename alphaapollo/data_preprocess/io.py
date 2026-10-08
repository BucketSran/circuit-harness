"""Input, serialization, cache, and integrity helpers for prepared datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from alphaapollo.data_preprocess.core import (
    MANIFEST_SCHEMA_VERSION,
    ROW_SCHEMA_VERSION,
    DatasetManifest,
    OutputFormat,
    PreparedDataset,
    PreparedExample,
    example_from_rows,
    sha256_hex,
)

__all__ = [
    "commit_directory",
    "dataset_dir",
    "default_root",
    "file_sha256",
    "load_source",
    "open_dataset",
    "read_manifest",
    "read_prepared",
    "read_records",
    "resolve_hub_revision",
    "source_fingerprint",
    "write_manifest",
    "write_records",
]


def default_root() -> Path:
    """Return the prepared-data root, honoring ``ALPHAAPOLLO_DATA_DIR``."""

    configured = os.environ.get("ALPHAAPOLLO_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".alphaapollo" / "data_preprocess").resolve()


def _safe_segment(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be a safe path segment, got {value!r}")
    return value


def dataset_dir(
    dataset_name: str,
    dataset_version: str,
    *,
    root: Path | str | None = None,
) -> Path:
    """Return the stable logical directory for a prepared dataset version."""

    base = Path(root).expanduser().resolve() if root else default_root()
    return (
        base
        / _safe_segment(dataset_name, "dataset_name")
        / _safe_segment(dataset_version, "dataset_version")
    )


def _format_from_path(path: Path) -> OutputFormat:
    try:
        return OutputFormat(path.suffix.removeprefix(".").lower())
    except ValueError as exc:
        raise ValueError(
            f"unsupported dataset extension {path.suffix!r}; expected .parquet, .jsonl, or .json"
        ) from exc


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support requires pyarrow; install AlphaApollo with `pip install -e '.[data]'`"
        ) from exc
    return pa, pq


def read_records(path: Path | str) -> list[dict[str, Any]]:
    """Read a Parquet, JSONL, or JSON record file based on its suffix."""

    path = Path(path).expanduser().resolve()
    format_ = _format_from_path(path)
    if format_ is OutputFormat.PARQUET:
        _, pq = _require_pyarrow()
        rows = pq.read_table(path).to_pylist()
    elif format_ is OutputFormat.JSONL:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"dataset file must contain a list of objects: {path}")
    return [dict(row) for row in rows]


def _canonical_dumps(value: Any) -> str:
    """Serialize deterministically so identical data yields identical bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_records(
    records: Iterable[Mapping[str, Any]],
    path: Path,
    output_format: OutputFormat,
) -> Path:
    """Write records deterministically in the selected output format."""

    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError(f"refusing to write an empty dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != output_format.suffix:
        raise ValueError(f"path suffix {path.suffix!r} does not match {output_format.value!r}")

    if output_format is OutputFormat.PARQUET:
        pa, pq = _require_pyarrow()
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif output_format is OutputFormat.JSONL:
        payload = "".join(f"{_canonical_dumps(row)}\n" for row in rows)
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(_canonical_dumps(rows) + "\n", encoding="utf-8")
    return path


def load_source(
    data_source: str,
    *,
    revision: str | None,
    splits: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Load requested splits from a local source or Hugging Face Hub."""

    local = Path(data_source).expanduser()
    if local.exists():
        local = local.resolve()
        if local.is_file():
            if len(splits) != 1:
                raise ValueError("a single source file can only supply one requested split")
            return {splits[0]: read_records(local)}
        loaded: dict[str, list[dict[str, Any]]] = {}
        for split in splits:
            matches = [
                local / f"{split}{suffix}"
                for suffix in (".parquet", ".jsonl", ".json")
                if (local / f"{split}{suffix}").is_file()
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected exactly one {split!r} file under {local}, found {matches}"
                )
            loaded[split] = read_records(matches[0])
        return loaded

    if revision is None or not revision.strip():
        raise ValueError("Hugging Face sources require an exact revision for reproducibility")
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face sources require datasets; install AlphaApollo with "
            "`pip install -e '.[data]'`"
        ) from exc

    dataset = load_dataset(data_source, revision=revision)
    if not isinstance(dataset, DatasetDict):
        if len(splits) != 1:
            raise ValueError("a single Hugging Face Dataset can only supply one requested split")
        return {splits[0]: [dict(row) for row in dataset]}
    missing = sorted(set(splits) - set(dataset))
    if missing:
        raise ValueError(f"Hugging Face dataset {data_source!r} has no splits {missing}")
    return {split: [dict(row) for row in dataset[split]] for split in splits}


def resolve_hub_revision(data_source: str) -> str:
    """Return the commit a Hub dataset currently resolves to.

    Callers should not have to paste a SHA to get a reproducible build. Resolving
    it here and recording it in the manifest keeps the guarantee -- a finished
    build always names the exact commit it read -- without making the caller look
    it up. Two builds a week apart will differ, and the fingerprint will say so.
    """

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "resolving a Hub revision requires huggingface_hub; install AlphaApollo with "
            "`pip install -e '.[data]'`"
        ) from exc
    revision = HfApi().dataset_info(data_source).sha
    if not revision:
        raise ValueError(f"Hugging Face dataset {data_source!r} reported no commit sha")
    return str(revision)


def source_fingerprint(
    data_source: str,
    *,
    revision: str | None,
    splits: Sequence[str],
) -> str:
    """Fingerprint local bytes or return the pinned Hub revision."""

    local = Path(data_source).expanduser()
    if not local.exists():
        if revision is None or not revision.strip():
            raise ValueError("Hugging Face sources require an exact revision for reproducibility")
        return revision
    local = local.resolve()
    if local.is_file():
        return file_sha256(local)
    entries: list[tuple[str, str]] = []
    for split in splits:
        matches = [
            local / f"{split}{suffix}"
            for suffix in (".parquet", ".jsonl", ".json")
            if (local / f"{split}{suffix}").is_file()
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {split!r} file under {local}, found {matches}")
        entries.append((matches[0].name, file_sha256(matches[0])))
    return sha256_hex(json.dumps(entries, separators=(",", ":")))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(manifest: DatasetManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def read_manifest(path: Path | str) -> DatasetManifest:
    """Parse one manifest, refusing a shape this code did not write."""

    source = Path(path)
    # A missing file already raises ``FileNotFoundError`` naming the path. A
    # truncated or schema-drifted one otherwise surfaces as a bare JSON or
    # pydantic error the caller cannot trace back to a build directory.
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"malformed prepared dataset manifest {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"malformed prepared dataset manifest {source}: expected a JSON object")
    # Read the declared version before validating against the model, so an
    # earlier manifest is refused by version rather than by whichever field it
    # happens to be missing. Every build predating `row_schema_version` is
    # rejected outright and deliberately: such a manifest cannot state which row
    # shape its files hold, so nothing can establish that they match this code.
    # Prepared builds are rebuilt from a recorded source in seconds and no
    # consumer outside this repository reads one, so refusing costs a rebuild
    # while guessing would risk reading rows against the wrong shape.
    recorded = payload.get("schema_version")
    if recorded != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"prepared dataset manifest {source} states schema_version {recorded!r}, "
            f"but this code writes schema_version {MANIFEST_SCHEMA_VERSION}; "
            "rebuild the dataset (prepare with if_exists='refresh')"
        )
    try:
        return DatasetManifest.model_validate(payload)
    except ValueError as exc:
        raise ValueError(f"malformed prepared dataset manifest {source}: {exc}") from exc


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


def open_dataset(path: Path | str) -> PreparedDataset:
    """Open and integrity-check one concrete prepared dataset directory."""

    path = Path(path).expanduser().resolve()
    manifest_path = path / "manifest.json"
    manifest = read_manifest(manifest_path)
    # Same class of check as the file digests below: both answer "can this code
    # read what is on disk?" before any row is handed back. The digests catch
    # corrupted bytes; this catches intact rows written under a different shape
    # than ``example_from_rows`` expects, which would otherwise surface as a
    # per-row schema mismatch pointing at a field rather than at the build.
    #
    # A mismatch is refused, not adapted. No flow reads an older build on
    # purpose: there is no migration path, no in-repository consumer pins a row
    # schema, and `read_prepared` is the only reader. ``ExistingPolicy.REFRESH``
    # rebuilds over a stale directory without opening it, so this refusal never
    # blocks the rebuild it asks for.
    if manifest.row_schema_version != ROW_SCHEMA_VERSION:
        raise ValueError(
            f"prepared dataset {path} was written under row schema version "
            f"{manifest.row_schema_version}, but this code reads row schema version "
            f"{ROW_SCHEMA_VERSION}; rebuild the dataset (prepare with if_exists='refresh')"
        )
    public: dict[str, Path] = {}
    private: dict[str, Path] = {}
    for split, files in manifest.splits.items():
        public_path = path / files.public_path
        private_path = path / files.private_path
        for candidate, expected in (
            (public_path, files.public_sha256),
            (private_path, files.private_sha256),
        ):
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            actual = file_sha256(candidate)
            if actual != expected:
                raise ValueError(
                    f"prepared dataset digest mismatch for {candidate}: {actual} != {expected}"
                )
        public[split] = public_path
        private[split] = private_path
    return PreparedDataset(
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        build_id=manifest.build_id,
        root=path,
        public_splits=public,
        private_splits=private,
        manifest_path=manifest_path,
    )


def read_prepared(
    dataset_name: str,
    dataset_version: str = "v1",
    root: Path | str | None = None,
    *,
    split: str | None = None,
) -> list[PreparedExample]:
    """Read joined public/private records after manifest integrity checks."""

    prepared = open_dataset(dataset_dir(dataset_name, dataset_version, root=root))
    selected = [split] if split is not None else list(prepared.public_splits)
    unknown = sorted(set(selected) - set(prepared.public_splits))
    if unknown:
        raise ValueError(f"prepared dataset has no splits {unknown}")
    examples: list[PreparedExample] = []
    for split_name in selected:
        public_rows = read_records(prepared.public_splits[split_name])
        private_rows = read_records(prepared.private_splits[split_name])
        if len(public_rows) != len(private_rows):
            raise ValueError(f"public/private row count mismatch for split {split_name!r}")
        private_by_uid = {row.get("task_uid"): row for row in private_rows}
        if len(private_by_uid) != len(private_rows):
            raise ValueError(f"duplicate private task_uid in split {split_name!r}")
        public_path = prepared.public_splits[split_name]
        for position, public in enumerate(public_rows):
            uid = public.get("task_uid")
            private = private_by_uid.pop(uid, None)
            if private is None:
                raise ValueError(
                    f"missing private row for task_uid {uid!r} "
                    f"(split {split_name!r}, row {position} of {public_path})"
                )
            # A schema or content mismatch is reported by the row it came from.
            # ``example_from_rows`` only sees two mappings, so on its own it can
            # say what is wrong but not which of thousands of rows is wrong.
            try:
                examples.append(example_from_rows(public, private))
            except ValueError as exc:
                raise ValueError(
                    f"malformed prepared row in split {split_name!r} at row {position} "
                    f"(task_uid {uid!r}, {public_path}): {exc}"
                ) from exc
        if private_by_uid:
            raise ValueError(
                f"orphaned private task_uids in split {split_name!r} "
                f"({prepared.private_splits[split_name]}): {sorted(private_by_uid)}"
            )
    return examples
