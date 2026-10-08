"""Abstract base for pluggable pretrain model architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModelBuilder(ABC):
    """A pluggable model-architecture component.

    ``build`` returns a Megatron model (e.g. ``megatron.core.models.gpt.GPTModel``) for the
    live ``args`` and parallelism placement — the same contract the package's ``--arch``
    builders always had, now formalized. Subclasses set ``name``.
    """

    name: str

    @abstractmethod
    def build(
        self,
        args: Any,
        pre_process: bool,
        post_process: bool,
        vp_stage: Any = None,
        config: Any = None,
        pg_collection: Any = None,
    ) -> Any:
        """Return a Megatron model for this architecture."""
        ...
