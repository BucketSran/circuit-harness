"""Resolve packaged prompt references without exposing filesystem paths."""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import cache
from importlib import resources
from typing import Any

import yaml

from alphaapollo.common.prompts.config import PromptConfigError, PromptSpec

__all__ = ["resolve_prompt"]

_PROMPT_REF = re.compile(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\Z")


def resolve_prompt(prompt_ref: str) -> PromptSpec:
    """Load one immutable packaged prompt by ``category.name`` reference."""

    if not isinstance(prompt_ref, str) or not _PROMPT_REF.fullmatch(prompt_ref):
        raise PromptConfigError(
            "prompt_ref must have the form 'category.name' using lowercase identifiers"
        )
    return _resolve_prompt(prompt_ref)


@cache
def _resolve_prompt(prompt_ref: str) -> PromptSpec:
    category, name = prompt_ref.split(".", 1)
    resource = resources.files("alphaapollo.common.prompts.resources").joinpath(
        category, f"{name}.yaml"
    )
    try:
        text = resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise PromptConfigError(f"unknown prompt_ref {prompt_ref!r}") from exc
    try:
        raw: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PromptConfigError(f"could not parse prompt {prompt_ref!r}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise PromptConfigError(f"prompt {prompt_ref!r} root must be a mapping")
    return PromptSpec.from_mapping(prompt_ref, raw)
