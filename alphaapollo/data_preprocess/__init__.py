"""Prepare reproducible public/private datasets from Hub or local sources.

The default serialized format is Parquet. JSON and JSONL are available for
small datasets and debugging. General entry points live in flat ``prepare_*.py``
modules; benchmark-specific robotics preparers live under ``robotics/``.
"""

from alphaapollo.data_preprocess.core import (
    MANIFEST_SCHEMA_VERSION,
    ROW_SCHEMA_VERSION,
    BuildRequest,
    DatasetManifest,
    ExistingPolicy,
    NormalizeContext,
    OutputFormat,
    PreparedDataset,
    PreparedExample,
    canonical_json,
    prepare_dataset,
    raw_digest,
    sha256_hex,
    task_uid,
)
from alphaapollo.data_preprocess.io import (
    dataset_dir,
    default_root,
    open_dataset,
    read_manifest,
    read_prepared,
    read_records,
)

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
    "dataset_dir",
    "default_root",
    "open_dataset",
    "prepare_dataset",
    "raw_digest",
    "read_manifest",
    "read_prepared",
    "read_records",
    "sha256_hex",
    "task_uid",
]
