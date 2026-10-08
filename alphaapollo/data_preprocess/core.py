"""Shared contracts and build orchestration for prepared datasets.

Dataset-specific modules only normalize raw rows into :class:`PreparedExample`.
This module owns identity, public/private separation, validation, manifests, and
the deterministic build lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ROW_SCHEMA_VERSION",
    "BuildRequest",
    "DatasetManifest",
    "ExistingPolicy",
    "NormalizeContext",
    "OutputFormat",
    "PreparedDataset",
    "PreparedExample",
    "canonical_json",
    "prepare_dataset",
    "private_row",
    "public_row",
    "raw_digest",
    "sha256_hex",
    "task_uid",
]


_STRICT = ConfigDict(extra="forbid", frozen=True)
PUBLIC_FIELDS = frozenset(
    {
        "task_uid",
        "source_id",
        "source_record_id",
        "source_order",
        "split",
        "group_id",
        "domain",
        "statement",
        "constraints",
        "prompt",
    }
)
PRIVATE_FIELDS = frozenset(
    {
        "task_uid",
        "answer",
        "answer_type",
        "reference_solution",
        "env_payload",
        "extra",
        "grader_id",
        "source_uri",
        "source_revision",
        "dataset_license",
        "original_content_owner",
        "original_content_license",
        "raw_digest",
        "builder_version",
    }
)
_PRIVATE_KEY_NAMES = frozenset(
    {"answer", "solution", "ground_truth", "gold_answer", "reference_answer"}
)
# Free-form mappings are persisted as canonical JSON strings rather than native
# nested values. Parquet cannot represent an empty struct at all, and inferring a
# struct type from content would let the physical schema drift with the data (a
# key present in one build but not the next, or an int where a string was). A
# string column is stable across builds and identical in all three formats.
_JSON_FIELDS = ("env_payload", "extra")
# Bump when the persisted row shape changes so stale builds are never reused.
ROW_SCHEMA_VERSION = 2
# Bump when the manifest's own field set changes. 2 added `row_schema_version`,
# without which a manifest cannot say which row shape its files hold.
MANIFEST_SCHEMA_VERSION = 2


class OutputFormat(str, Enum):
    """Supported prepared-data serialization formats."""

    PARQUET = "parquet"
    JSONL = "jsonl"
    JSON = "json"

    @property
    def suffix(self) -> str:
        return f".{self.value}"


class ExistingPolicy(str, Enum):
    """How a prepare call handles an existing build directory."""

    REUSE = "reuse"
    REFRESH = "refresh"
    ERROR = "error"


@dataclass(frozen=True)
class BuildRequest:
    """Dataset-independent inputs to a deterministic prepare operation."""

    dataset_name: str
    dataset_version: str
    data_source: str
    revision: str | None
    splits: tuple[str, ...]
    output_root: Path
    output_format: OutputFormat = OutputFormat.PARQUET
    if_exists: ExistingPolicy = ExistingPolicy.REUSE

    def __post_init__(self) -> None:
        for value, label in (
            (self.dataset_name, "dataset_name"),
            (self.dataset_version, "dataset_version"),
            (self.data_source, "data_source"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        if "/" in self.dataset_name or "\\" in self.dataset_name:
            raise ValueError("dataset_name must be a single path segment")
        if "/" in self.dataset_version or "\\" in self.dataset_version:
            raise ValueError("dataset_version must be a single path segment")
        if not self.splits or any(not split.strip() for split in self.splits):
            raise ValueError("splits must contain at least one non-empty split name")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("splits contains duplicate entries")


@dataclass(frozen=True)
class NormalizeContext:
    """Stable context supplied to a dataset-specific row normalizer."""

    source_id: str
    split: str
    source_order: int
    builder_version: str


class PreparedExample(BaseModel):
    """Complete in-memory example; it is never persisted as one record."""

    model_config = _STRICT

    task_uid: str
    source_id: str
    source_record_id: str
    source_order: int = Field(ge=0)
    split: str
    group_id: str | None = None

    statement: str
    domain: str = "math"
    constraints: tuple[str, ...] = ()
    prompt: str = ""

    answer: str
    answer_type: str = "string"
    # RESERVED, not forgotten. No shipped preparer writes `reference_solution`
    # today, and the LIBERO preparer is the first to write `env_payload`; both
    # are part of the persisted private row on purpose -- they were kept when
    # `answer_space` and `grader_params` were dropped because each has a named
    # consumer. Do not delete them as dead fields; see README.md, "Private
    # extension fields".
    #
    # `reference_solution`: the worked gold derivation, distinct from the final
    # `answer`. A preparer for a dataset that ships solutions would fill it; the
    # SFT export is what read it, filtering out tasks whose reference solution was
    # empty. That export no longer exists, so nothing reads it today.
    reference_solution: str = ""
    # `env_payload`: the channel for state only an Environment should consume,
    # including gold needed for in-loop RL reward. A Workflow can opt into the
    # exact private split file through `dataset.task_payload_path`; the loader
    # joins by id and carries this value through `WorkflowInput.task_payload` to
    # `EnvironmentContext.task_payload`. `DefaultEnvironment._gold_from_payload`
    # consumes its `answer` field when an in-loop grader is configured. The
    # LIBERO preparer writes the benchmark and environment-version fields its
    # owning integration layer maps into the robot task contract; the other
    # shipped normalizers still leave it empty.
    env_payload: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    grader_id: str

    source_uri: str
    source_revision: str
    dataset_license: str
    original_content_owner: str = "unverified"
    original_content_license: str = "unverified"
    # FORENSIC, not protective, and deliberately never verified automatically.
    # `raw_digest` is the SHA-256 of the canonical JSON of the source row this
    # example was normalized from. Nothing reads it back, and that is by design,
    # not an oversight: verifying it would require the original row, which a
    # prepared build does not keep -- only the digest survives. Re-fetching the
    # source to compare is exactly what a rebuild already does, so an automatic
    # check would be a rebuild wearing a different name. There is no cheap check
    # to add here; do not add one.
    #
    # What it does support is after-the-fact evidence. If an upstream dataset is
    # suspected of having silently edited rows while keeping the same revision --
    # the one failure `source_revision` and `build_id` cannot detect -- rebuild
    # the dataset into a fresh output root and diff the `raw_digest` column
    # against the old build. Rows whose digest changed are the rows that changed,
    # named exactly. Without this column a rebuild would only show that some
    # answer somewhere differs, not which upstream rows moved.
    #
    # Contrast with the manifest's per-split `public_sha256`/`private_sha256`,
    # which `io.open_dataset` does verify on every open and raises on mismatch.
    # File digests are enforced and protect against local corruption; this row
    # digest is evidence and protects against nothing on its own. Both are
    # intentional; see README.md, "Row digests are evidence, file digests are
    # enforced".
    raw_digest: str
    builder_version: str

    @model_validator(mode="after")
    def _validate_content(self) -> PreparedExample:
        required = {
            "task_uid": self.task_uid,
            "source_id": self.source_id,
            "source_record_id": self.source_record_id,
            "split": self.split,
            "statement": self.statement,
            "answer": self.answer,
            "grader_id": self.grader_id,
            "source_uri": self.source_uri,
            "source_revision": self.source_revision,
            "dataset_license": self.dataset_license,
            "raw_digest": self.raw_digest,
            "builder_version": self.builder_version,
        }
        empty = [name for name, value in required.items() if not value.strip()]
        if empty:
            raise ValueError(f"prepared example has empty required fields: {empty}")
        if self.task_uid != task_uid(self.source_id, self.source_record_id):
            raise ValueError("task_uid does not match source_id/source_record_id")
        canonical_json(self.env_payload)
        canonical_json(self.extra)
        return self


class PreparedSplit(BaseModel):
    model_config = _STRICT

    rows: int = Field(ge=0)
    public_path: str
    private_path: str
    public_sha256: str
    private_sha256: str


class DatasetManifest(BaseModel):
    """Integrity and provenance record stored next to every prepared build.

    Both version fields are stated by the writer rather than defaulted, so an
    older manifest cannot be parsed into current values it never recorded.
    ``io.read_manifest`` refuses a `schema_version` this code did not write, and
    ``io.open_dataset`` refuses a `row_schema_version` it cannot read.
    """

    model_config = _STRICT

    schema_version: int
    # The shape the `public/` and `private/` files were written under. `build_id`
    # already folds it in, so a schema bump yields a different fingerprint -- but
    # a reader holding a prepared directory learns that only by recomputing the
    # fingerprint from the source it no longer has. Recording it lets any open
    # answer "do these rows match the code about to read them?" directly.
    row_schema_version: int
    dataset_name: str
    dataset_version: str
    build_id: str
    builder_name: str
    builder_version: str
    data_source: str
    source_revision: str | None
    source_fingerprint: str
    output_format: OutputFormat
    splits: dict[str, PreparedSplit]


@dataclass(frozen=True)
class PreparedDataset:
    """Resolved paths for one validated prepared dataset."""

    dataset_name: str
    dataset_version: str
    build_id: str
    root: Path
    public_splits: Mapping[str, Path]
    private_splits: Mapping[str, Path]
    manifest_path: Path


Normalizer = Callable[[Mapping[str, Any], NormalizeContext], PreparedExample]


def canonical_json(value: Any) -> str:
    """Return stable, compact JSON or fail for non-JSON values."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_hex(data: str | bytes) -> str:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()


def task_uid(source_id: str, source_record_id: str) -> str:
    return sha256_hex(f"{source_id}\0{source_record_id}")[:32]


def raw_digest(record: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(dict(record)))


def public_row(example: PreparedExample) -> dict[str, Any]:
    """Serialize the model-visible projection."""

    row = {
        "task_uid": example.task_uid,
        "source_id": example.source_id,
        "source_record_id": example.source_record_id,
        "source_order": example.source_order,
        "split": example.split,
        "group_id": example.group_id,
        "domain": example.domain,
        "statement": example.statement,
        "constraints": list(example.constraints),
        "prompt": example.prompt,
    }
    if set(row) != PUBLIC_FIELDS:
        raise AssertionError("public row schema drifted from PUBLIC_FIELDS")
    _reject_private_keys(row, where="public row")
    return row


def private_row(example: PreparedExample) -> dict[str, Any]:
    """Serialize the evaluator-only projection."""

    row = {
        "task_uid": example.task_uid,
        "answer": example.answer,
        "answer_type": example.answer_type,
        "reference_solution": example.reference_solution,
        "env_payload": canonical_json(dict(example.env_payload)),
        "extra": canonical_json(dict(example.extra)),
        "grader_id": example.grader_id,
        "source_uri": example.source_uri,
        "source_revision": example.source_revision,
        "dataset_license": example.dataset_license,
        "original_content_owner": example.original_content_owner,
        "original_content_license": example.original_content_license,
        "raw_digest": example.raw_digest,
        "builder_version": example.builder_version,
    }
    if set(row) != PRIVATE_FIELDS:
        raise AssertionError("private row schema drifted from PRIVATE_FIELDS")
    return row


def example_from_rows(
    public: Mapping[str, Any],
    private: Mapping[str, Any],
) -> PreparedExample:
    """Rebuild an example from its persisted public/private projections."""

    if set(public) != PUBLIC_FIELDS:
        unexpected = sorted(set(public) - PUBLIC_FIELDS)
        missing = sorted(PUBLIC_FIELDS - set(public))
        raise ValueError(f"public row schema mismatch: unexpected={unexpected}, missing={missing}")
    if set(private) != PRIVATE_FIELDS:
        unexpected = sorted(set(private) - PRIVATE_FIELDS)
        missing = sorted(PRIVATE_FIELDS - set(private))
        raise ValueError(f"private row schema mismatch: unexpected={unexpected}, missing={missing}")
    if public["task_uid"] != private["task_uid"]:
        raise ValueError("public/private task_uid mismatch")
    decoded = dict(private)
    for field in _JSON_FIELDS:
        raw = decoded[field]
        if isinstance(raw, str):
            decoded[field] = json.loads(raw)
        if not isinstance(decoded[field], Mapping):
            raise ValueError(f"private field {field!r} must decode to an object")
    return PreparedExample.model_validate({**dict(public), **decoded})


def _reject_private_keys(value: Any, *, where: str) -> None:
    if isinstance(value, Mapping):
        leaked = sorted(str(key) for key in value if str(key).lower() in _PRIVATE_KEY_NAMES)
        if leaked:
            raise ValueError(f"{where} contains private keys: {leaked}")
        for child in value.values():
            _reject_private_keys(child, where=where)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_private_keys(child, where=where)


def _reject_unread_data_source(request: BuildRequest) -> None:
    """Refuse a generated build whose ``data_source`` names something unread.

    With ``source_rows`` the pipeline skips both ``io.source_fingerprint`` and
    ``io.load_source``, so nothing checks ``data_source`` or ``revision``. A
    path or bare Hub repo id would then be recorded in the manifest as the
    source of bytes the build never opened, and a Hub-shaped id would bypass
    the exact-revision guard both helpers enforce. A generated build must
    instead name its generator with a scheme-qualified pseudo-URI, for example
    ``libero://checkout/<revision>/<suite>/<order>``.
    """

    if Path(request.data_source).expanduser().exists():
        raise ValueError(
            f"source_rows was given with data_source {request.data_source!r}, which "
            "resolves to an existing path; generated rows are fingerprinted from the "
            "rows themselves, so that path would be recorded as a source the build "
            "never read"
        )
    scheme, separator, remainder = request.data_source.partition("://")
    if not separator or not scheme.strip() or not remainder.strip():
        raise ValueError(
            f"source_rows was given with data_source {request.data_source!r}; a "
            "generated build is not loaded from a file or the Hub, so data_source must "
            "be a scheme-qualified pseudo-URI naming the generator, for example "
            "'libero://checkout/<revision>/<suite>/<order>'"
        )


def _build_spec(
    request: BuildRequest,
    *,
    builder_name: str,
    builder_version: str,
    source_fingerprint: str,
    build_options: Mapping[str, Any],
) -> dict[str, Any]:
    spec = {
        "dataset_name": request.dataset_name,
        "dataset_version": request.dataset_version,
        "data_source": request.data_source,
        "revision": request.revision,
        "source_fingerprint": source_fingerprint,
        "splits": list(request.splits),
        "output_format": request.output_format.value,
        "builder_name": builder_name,
        "builder_version": builder_version,
        "row_schema_version": ROW_SCHEMA_VERSION,
    }
    if build_options:
        spec["build_options"] = dict(build_options)
    return spec


def prepare_dataset(
    request: BuildRequest,
    normalizer: Normalizer,
    *,
    builder_name: str,
    builder_version: str,
    build_options: Mapping[str, Any] | None = None,
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> PreparedDataset:
    """Load, normalize, validate, and atomically persist a prepared dataset.

    ``build_options`` names JSON-stable normalizer inputs that can change rows
    without changing the source or generic request. They participate in the
    build id so ``reuse`` cannot return an artifact built under other options.

    ``source_rows`` supports a source whose canonical records are generated by
    a domain adapter rather than read from a file or Hub dataset. The shared
    pipeline fingerprints those rows itself; ordinary file/Hub callers keep the
    normal source-loading path.
    """

    if not builder_name.strip() or not builder_version.strip():
        raise ValueError("builder_name and builder_version must be non-empty")
    if build_options is None:
        normalized_build_options: dict[str, Any] = {}
    elif not isinstance(build_options, Mapping):
        raise TypeError("build_options must be a mapping")
    else:
        normalized_build_options = dict(build_options)
    if any(not isinstance(key, str) or not key.strip() for key in normalized_build_options):
        raise ValueError("build_options keys must be non-empty strings")
    # Validate before source loading. These values affect normalized rows, so
    # they must be JSON-stable inputs to the same build identity as the source.
    canonical_json(normalized_build_options)
    normalized_source_rows: dict[str, list[dict[str, Any]]] | None = None
    if source_rows is not None:
        if not isinstance(source_rows, Mapping):
            raise TypeError("source_rows must be a mapping")
        normalized_source_rows = {}
        for split, rows in source_rows.items():
            if not isinstance(split, str) or not split.strip():
                raise ValueError("source_rows split names must be non-empty strings")
            if not isinstance(rows, Sequence):
                raise TypeError(f"source_rows[{split!r}] must be a sequence")
            normalized: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError(f"source_rows[{split!r}] contains a non-mapping row")
                normalized.append(dict(row))
            normalized_source_rows[split] = normalized
        if set(normalized_source_rows) != set(request.splits):
            raise ValueError(
                "source_rows splits must exactly match request.splits: "
                f"rows={sorted(normalized_source_rows)}, request={sorted(request.splits)}"
            )
        canonical_json(normalized_source_rows)
        _reject_unread_data_source(request)

    # Import lazily so ``io`` can use the public models without a module cycle.
    from alphaapollo.data_preprocess import io

    destination = io.dataset_dir(
        request.dataset_name,
        request.dataset_version,
        root=request.output_root,
    )
    # ``error`` is answerable from the destination alone, so refuse before
    # reading the source. Fingerprinting hashes every local split file, which is
    # work spent on a build that was never allowed to be published.
    if destination.exists() and request.if_exists is ExistingPolicy.ERROR:
        raise FileExistsError(f"prepared dataset already exists: {destination}")

    source_fingerprint = (
        sha256_hex(canonical_json(normalized_source_rows))
        if normalized_source_rows is not None
        else io.source_fingerprint(
            request.data_source,
            revision=request.revision,
            splits=request.splits,
        )
    )
    build_id = sha256_hex(
        canonical_json(
            _build_spec(
                request,
                builder_name=builder_name,
                builder_version=builder_version,
                source_fingerprint=source_fingerprint,
                build_options=normalized_build_options,
            )
        )
    )[:16]

    if destination.exists() and request.if_exists is ExistingPolicy.REUSE:
        prepared = io.open_dataset(destination)
        if prepared.build_id != build_id:
            raise ValueError(
                f"existing dataset at {destination} was built from a different spec; "
                "use a new dataset_version or if_exists='refresh'"
            )
        return prepared

    raw_splits = (
        normalized_source_rows
        if normalized_source_rows is not None
        else io.load_source(
            request.data_source,
            revision=request.revision,
            splits=request.splits,
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{request.dataset_name}-", dir=destination.parent))
    seen_uids: set[str] = set()
    split_manifests: dict[str, PreparedSplit] = {}

    try:
        for split in request.splits:
            raw_rows = raw_splits.get(split)
            if raw_rows is None:
                raise ValueError(f"source {request.data_source!r} has no split {split!r}")
            examples: list[PreparedExample] = []
            for source_order, raw in enumerate(raw_rows):
                context = NormalizeContext(
                    source_id=request.dataset_name,
                    split=split,
                    source_order=source_order,
                    builder_version=builder_version,
                )
                example = normalizer(raw, context)
                if example.source_id != request.dataset_name:
                    raise ValueError("normalizer changed the configured source_id")
                if example.split != split or example.source_order != source_order:
                    raise ValueError("normalizer changed split or source order")
                if example.builder_version != builder_version:
                    raise ValueError("normalizer changed builder_version")
                if example.task_uid in seen_uids:
                    raise ValueError(f"duplicate task_uid {example.task_uid!r}")
                seen_uids.add(example.task_uid)
                examples.append(example)
            if not examples:
                raise ValueError(f"split {split!r} is empty")

            public_path = Path("public") / f"{split}{request.output_format.suffix}"
            private_path = Path("private") / f"{split}{request.output_format.suffix}"
            io.write_records(
                (public_row(example) for example in examples),
                temp_root / public_path,
                request.output_format,
            )
            io.write_records(
                (private_row(example) for example in examples),
                temp_root / private_path,
                request.output_format,
            )
            split_manifests[split] = PreparedSplit(
                rows=len(examples),
                public_path=public_path.as_posix(),
                private_path=private_path.as_posix(),
                public_sha256=io.file_sha256(temp_root / public_path),
                private_sha256=io.file_sha256(temp_root / private_path),
            )

        manifest = DatasetManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            row_schema_version=ROW_SCHEMA_VERSION,
            dataset_name=request.dataset_name,
            dataset_version=request.dataset_version,
            build_id=build_id,
            builder_name=builder_name,
            builder_version=builder_version,
            data_source=request.data_source,
            source_revision=request.revision,
            source_fingerprint=source_fingerprint,
            output_format=request.output_format,
            splits=split_manifests,
        )
        io.write_manifest(manifest, temp_root / "manifest.json")
        io.commit_directory(temp_root, destination, replace=destination.exists())
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    return io.open_dataset(destination)
