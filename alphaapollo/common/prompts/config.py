"""Immutable prompt resources with strict, minimal template rendering."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from string import Formatter
from types import MappingProxyType
from typing import Any

__all__ = ["PromptConfigError", "PromptSpec", "prompt_digest"]

_TEMPLATE_NAMES = ("user_template", "continuation_template", "visual_template")


class PromptConfigError(ValueError):
    """A prompt resource or render request is malformed."""


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One packaged prompt revision with optional continuation and visual variants.

    ``version`` is an author-maintained revision number, not a selector for
    historical content. Reconstructing an older resource requires its Git revision;
    the rendered first-turn digest verifies the exact system/user text used.
    """

    prompt_ref: str
    version: int
    system: str = ""
    user_template: str = ""
    continuation_template: str | None = None
    visual_template: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_ref, str) or not self.prompt_ref.strip():
            raise PromptConfigError("prompt_ref must be a non-empty string")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise PromptConfigError("prompt version must be a positive integer")
        if not isinstance(self.system, str):
            raise PromptConfigError("prompt system must be a string")
        for name in _TEMPLATE_NAMES:
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise PromptConfigError(f"prompt {name} must be a string when provided")
            if value is not None:
                _template_fields(value, where=f"prompt {self.prompt_ref!r}.{name}")

    @classmethod
    def from_mapping(cls, prompt_ref: str, value: Mapping[str, Any]) -> PromptSpec:
        """Validate one strict YAML/JSON mapping."""

        if not isinstance(value, Mapping):
            raise PromptConfigError(f"prompt {prompt_ref!r} root must be a mapping")
        raw = dict(value)
        allowed = {"version", "system", *_TEMPLATE_NAMES}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise PromptConfigError(f"prompt {prompt_ref!r} has unknown keys: {unknown}")
        if "version" not in raw:
            raise PromptConfigError(f"prompt {prompt_ref!r} is missing required key 'version'")
        return cls(
            prompt_ref=prompt_ref,
            version=raw["version"],
            system=raw.get("system", ""),
            user_template=raw.get("user_template", ""),
            continuation_template=raw.get("continuation_template"),
            visual_template=raw.get("visual_template"),
        )

    @property
    def variables(self) -> Mapping[str, frozenset[str]]:
        """Return template variables derived from the templates themselves."""

        result: dict[str, frozenset[str]] = {}
        for name in _TEMPLATE_NAMES:
            template = getattr(self, name)
            if template is not None:
                result[name] = _template_fields(
                    template, where=f"prompt {self.prompt_ref!r}.{name}"
                )
        return MappingProxyType(result)

    def render_user(self, **variables: object) -> str:
        return self._render("user_template", self.user_template, variables)

    def render_continuation(self, **variables: object) -> str:
        return self._render("continuation_template", self.continuation_template, variables)

    def render_visual(self, **variables: object) -> str:
        return self._render("visual_template", self.visual_template, variables)

    def render_messages(self, **variables: object) -> tuple[dict[str, str], ...]:
        """Render a first-turn chat, omitting an empty system message."""

        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.render_user(**variables)})
        return tuple(messages)

    def digest(self, **variables: object) -> str:
        """Identify the exact first-turn messages rendered by this resource."""

        return prompt_digest(self.system, self.render_user(**variables))

    def _render(
        self,
        name: str,
        template: str | None,
        variables: Mapping[str, object],
    ) -> str:
        if template is None:
            raise PromptConfigError(f"prompt {self.prompt_ref!r} has no {name}")
        expected = _template_fields(template, where=f"prompt {self.prompt_ref!r}.{name}")
        provided = frozenset(variables)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            raise PromptConfigError(
                f"prompt {self.prompt_ref!r}.{name} variables are invalid: " + ", ".join(details)
            )
        return template.format(**variables)


def _template_fields(template: str, *, where: str) -> frozenset[str]:
    try:
        parsed = tuple(Formatter().parse(template))
    except ValueError as exc:
        raise PromptConfigError(f"{where} is not a valid template: {exc}") from exc
    fields: set[str] = set()
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if not field_name.isidentifier():
            raise PromptConfigError(f"{where} uses invalid placeholder {field_name!r}")
        if format_spec or conversion:
            raise PromptConfigError(f"{where} placeholders cannot use conversions or format specs")
        fields.add(field_name)
    return frozenset(fields)


def prompt_digest(system_prompt: str | None, user_prompt: str) -> str:
    """Identify exact rendered first-turn system/user text across prompt consumers.

    This deliberately is not a complete Generation request fingerprint: tool
    schemas, continuation/visual prompts, provider chat templates, model identity,
    and sampling parameters are outside its scope.
    """

    if system_prompt is not None and not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string when provided")
    if not isinstance(user_prompt, str):
        raise TypeError("user_prompt must be a string")
    payload = json.dumps(
        {"system_prompt": system_prompt or "", "user_prompt": user_prompt},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
