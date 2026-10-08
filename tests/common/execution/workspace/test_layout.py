"""Compatibility tests for the split workspace implementation."""

from alphaapollo.common.execution._workspace.export import WorkspaceExporter as SplitExporter
from alphaapollo.common.execution._workspace.lease import WorkspaceLease as SplitLease
from alphaapollo.common.execution._workspace.snapshot import (
    WorkspaceSnapshotter as SplitSnapshotter,
)
from alphaapollo.common.execution._workspace.store import ArtifactStore as SplitArtifactStore
from alphaapollo.common.execution._workspace.verifier import (
    VerifierWorkspace as SplitVerifierWorkspace,
)
from alphaapollo.common.execution.workspace import (
    ArtifactStore,
    VerifierWorkspace,
    WorkspaceExporter,
    WorkspaceLease,
    WorkspaceSnapshotter,
)


def test_workspace_facade_reexports_split_implementations() -> None:
    assert ArtifactStore is SplitArtifactStore
    assert WorkspaceLease is SplitLease
    assert WorkspaceSnapshotter is SplitSnapshotter
    assert WorkspaceExporter is SplitExporter
    assert VerifierWorkspace is SplitVerifierWorkspace
