# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Stable imports for execution workspace contracts and implementations.

Implementations are split by responsibility under ``execution._workspace``;
this module remains the supported compatibility and public import surface.
"""

from alphaapollo.common.execution._workspace.export import (
    RUNS_DIR_ENV,
    RUNS_MAX_ENV,
    ExportResult,
    WorkspaceExporter,
    WorkspaceExportError,
)
from alphaapollo.common.execution._workspace.lease import WorkspaceLease, WorkspaceProvider
from alphaapollo.common.execution._workspace.snapshot import SnapshotError, WorkspaceSnapshotter
from alphaapollo.common.execution._workspace.store import ArtifactStore
from alphaapollo.common.execution._workspace.verifier import (
    VerifierWorkspace,
    VerifierWorkspaceError,
)

__all__ = [
    "ArtifactStore",
    "ExportResult",
    "RUNS_DIR_ENV",
    "RUNS_MAX_ENV",
    "SnapshotError",
    "VerifierWorkspace",
    "VerifierWorkspaceError",
    "WorkspaceExportError",
    "WorkspaceExporter",
    "WorkspaceLease",
    "WorkspaceProvider",
    "WorkspaceSnapshotter",
]
