"""Dependency-light switches for independent memory channels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryProfile:
    """Enable semantic recall, lexical recall, and scratchpad independently."""

    semantic_recall: bool = False
    lexical_memory: bool = False
    scratchpad: bool = False

    @property
    def any_enabled(self) -> bool:
        return self.semantic_recall or self.lexical_memory or self.scratchpad

    @property
    def retrieval_enabled(self) -> bool:
        return self.semantic_recall or self.lexical_memory

    @property
    def tier_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.semantic_recall:
            names.append("semantic")
        if self.lexical_memory:
            names.append("lexical")
        return tuple(names)


MEMORY_OFF = MemoryProfile()
MEMORY_FULL = MemoryProfile(semantic_recall=True, lexical_memory=True, scratchpad=True)

NAMED_PROFILES: dict[str, MemoryProfile] = {
    "off": MEMORY_OFF,
    "semantic": MemoryProfile(semantic_recall=True),
    "semantic_lexical": MemoryProfile(semantic_recall=True, lexical_memory=True),
    "scratchpad": MemoryProfile(scratchpad=True),
    "full": MEMORY_FULL,
}


def resolve_memory_profile(name: str) -> MemoryProfile:
    try:
        return NAMED_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown memory profile {name!r}; expected one of {sorted(NAMED_PROFILES)}"
        ) from exc


__all__ = [
    "MEMORY_FULL",
    "MEMORY_OFF",
    "NAMED_PROFILES",
    "MemoryProfile",
    "resolve_memory_profile",
]
