# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Workspace acquisition and release contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable


class WorkspaceLease:
    """Opaque workspace reference plus an idempotent release callback."""

    def __init__(self, ref: str, release: Callable[[], None]) -> None:
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError("workspace lease ref must be non-empty")
        if not callable(release):
            raise TypeError("workspace lease release must be callable")
        self._ref = ref
        self._release = release
        self._released = False

    @property
    def ref(self) -> str:
        return self._ref

    @property
    def released(self) -> bool:
        return self._released

    def close(self) -> None:
        if self._released:
            return
        self._release()
        self._released = True


@runtime_checkable
class WorkspaceProvider(Protocol):
    """Acquire a distinct workspace lease for one Environment session."""

    def acquire(
        self,
        *,
        session_id: str,
        actor: str,
        branch_id: str,
    ) -> WorkspaceLease: ...
