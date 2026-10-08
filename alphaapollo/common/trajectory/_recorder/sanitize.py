# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bounded capture copying, redaction, and environment-input validation."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from alphaapollo.common.artifacts.schemas import ProvenanceRef, require_json_value

if TYPE_CHECKING:
    from alphaapollo.common.environment.base import EnvironmentSession
    from alphaapollo.common.trajectory._recorder.contracts import EnvironmentCapture


class SensitiveEnvironmentInputError(ValueError):
    """Raised when evaluator-only or credential material reaches Environment input."""


_CAPTURE_VERSION = 1
_CAPTURE_MEDIA_TYPE = "application/vnd.alphaapollo.environment-capture+json"
_CAPTURE_PROVENANCE = ProvenanceRef(
    id="common.environment.trajectory_capture",
    version=str(_CAPTURE_VERSION),
)
_REDACTED = "[REDACTED]"
_CAPTURE_CYCLE = "[TRUNCATED: cyclic reference]"
_CAPTURE_DEPTH_LIMIT = "[TRUNCATED: maximum depth exceeded]"
_CAPTURE_NODE_LIMIT = "[TRUNCATED: maximum node count exceeded]"
_MAX_CAPTURE_DEPTH = 64
_MAX_CAPTURE_NODES = 10_000
_GROUND_TRUTH_KEYS = frozenset(
    {
        "gold",
        "groundtruth",
        "expectedanswer",
        "referenceanswer",
        "answerkey",
        "evaluatoronlyanswer",
        "evaluatoranswer",
    }
)
_PROTECTED_ENVIRONMENT_INPUT_KEYS = _GROUND_TRUTH_KEYS | frozenset(
    {
        "answer",
        "goldanswer",
        "referencesolution",
        "solution",
        "taskpayload",
    }
)
_SECRET_KEYS = frozenset(
    {
        "apikey",
        "authorization",
        "authtoken",
        "accesstoken",
        "bearertoken",
        "clientsecret",
        "credential",
        "password",
        "privatekey",
        "refreshtoken",
        "secret",
        "sessionkey",
        "token",
        "awssecretaccesskey",
    }
)
_GROUND_TRUTH_TEXT = re.compile(
    r"(?i)\b(?:ground[_ -]?truth[\w-]*|expected[_ -]?answer|reference[_ -]?answer|"
    r"answer[_ -]?key|evaluator[_ -]?(?:only[_ -]?)?answer|gold)\s*[:=]"
)
_SECRET_TEXT = re.compile(
    r"(?i)\b(?:api[_ -]?key|authorization|auth[_ -]?token|access[_ -]?token|"
    r"bearer[_ -]?token|client[_ -]?secret|credential|password|private[_ -]?key|"
    r"refresh[_ -]?token|session[_ -]?key|aws[_ -]?secret[_ -]?access[_ -]?key)"
    r"\s*[:=]\s*\S+"
)


def build_capture_envelope(
    session: EnvironmentSession,
    capture: EnvironmentCapture,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    sanitized, detected_flags, redactions = sanitize_capture_value(capture.content)
    flags = tuple(sorted(set(detected_flags).union(capture.contamination_flags)))
    envelope = {
        "capture_version": _CAPTURE_VERSION,
        "kind": capture.kind.value,
        "session_id": session.session_id,
        "branch_id": session.branch_id,
        "actor": session.actor,
        "round_index": session.round_index,
        "step_index": session.step_index,
        "content": sanitized,
        "redactions": list(redactions),
        "source_contamination_flags": list(capture.contamination_flags),
    }
    require_json_value(envelope, path="$.capture")
    return envelope, flags


def sanitize_capture_value(
    value: Any,
) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    """Recursively redact evaluator-only and credential-shaped values."""

    flags: set[str] = set()
    redactions: list[str] = []
    visited_nodes = 0

    def truncate(flag: str, path: str, sentinel: str) -> str:
        flags.add(flag)
        redactions.append(path)
        return sentinel

    def visit(item: Any, path: str, depth: int, ancestors: frozenset[int]) -> Any:
        nonlocal visited_nodes
        visited_nodes += 1
        if visited_nodes > _MAX_CAPTURE_NODES:
            return truncate("capture_node_limit", path, _CAPTURE_NODE_LIMIT)
        if depth > _MAX_CAPTURE_DEPTH:
            return truncate("capture_depth_limit", path, _CAPTURE_DEPTH_LIMIT)
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in ancestors:
                return truncate("capture_cycle", path, _CAPTURE_CYCLE)
            child_ancestors = ancestors.union((identity,))
            sanitized: dict[str, Any] = {}
            for raw_key, raw_value in item.items():
                if visited_nodes >= _MAX_CAPTURE_NODES:
                    flags.add("capture_node_limit")
                    redactions.append(path)
                    break
                if not isinstance(raw_key, str):
                    flags.add("unsafe_non_string_key")
                    redactions.append(path)
                    continue
                normalized = _normalize_key(raw_key)
                if normalized in _GROUND_TRUTH_KEYS:
                    flags.add("ground_truth")
                    redactions.append(f"{path}.{raw_key}")
                    sanitized[raw_key] = _REDACTED
                elif normalized in _SECRET_KEYS:
                    flags.add("credential")
                    redactions.append(f"{path}.{raw_key}")
                    sanitized[raw_key] = _REDACTED
                else:
                    sanitized[raw_key] = visit(
                        raw_value,
                        f"{path}.{raw_key}",
                        depth + 1,
                        child_ancestors,
                    )
            return sanitized
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in ancestors:
                return truncate("capture_cycle", path, _CAPTURE_CYCLE)
            child_ancestors = ancestors.union((identity,))
            sanitized_items: list[Any] = []
            for index, child in enumerate(item):
                if visited_nodes >= _MAX_CAPTURE_NODES:
                    sanitized_items.append(
                        truncate("capture_node_limit", path, _CAPTURE_NODE_LIMIT)
                    )
                    break
                sanitized_items.append(visit(child, f"{path}[{index}]", depth + 1, child_ancestors))
            return sanitized_items
        if isinstance(item, str):
            matched_flags: list[str] = []
            if _GROUND_TRUTH_TEXT.search(item):
                matched_flags.append("ground_truth")
            if _SECRET_TEXT.search(item):
                matched_flags.append("credential")
            if matched_flags:
                flags.update(matched_flags)
                redactions.append(path)
                return _REDACTED
        return copy.deepcopy(item)

    result = visit(value, "$.content", 0, frozenset())
    require_json_value(result, path="$.capture.content")
    return result, tuple(sorted(flags)), tuple(sorted(set(redactions)))


def ensure_safe_environment_input(value: Any) -> None:
    """Fail closed before evaluator-only material can enter the model loop."""

    protected_paths = _protected_environment_input_paths(value)
    if protected_paths:
        raise SensitiveEnvironmentInputError(
            "environment input contains protected material at " + ", ".join(protected_paths)
        )


def _bounded_capture_copy(value: Any) -> tuple[Any, tuple[str, ...]]:
    """Copy capture content without allowing hostile graphs to break persistence."""

    flags: set[str] = set()
    visited_nodes = 0

    def visit(item: Any, depth: int, ancestors: frozenset[int]) -> Any:
        nonlocal visited_nodes
        visited_nodes += 1
        if visited_nodes > _MAX_CAPTURE_NODES:
            flags.add("capture_node_limit")
            return _CAPTURE_NODE_LIMIT
        if depth > _MAX_CAPTURE_DEPTH:
            flags.add("capture_depth_limit")
            return _CAPTURE_DEPTH_LIMIT
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in ancestors:
                flags.add("capture_cycle")
                return _CAPTURE_CYCLE
            child_ancestors = ancestors.union((identity,))
            copied: dict[Any, Any] = {}
            for raw_key, raw_value in item.items():
                if visited_nodes >= _MAX_CAPTURE_NODES:
                    flags.add("capture_node_limit")
                    break
                copied[copy.deepcopy(raw_key)] = visit(raw_value, depth + 1, child_ancestors)
            return copied
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in ancestors:
                flags.add("capture_cycle")
                return _CAPTURE_CYCLE
            child_ancestors = ancestors.union((identity,))
            copied_items: list[Any] = []
            for child in item:
                if visited_nodes >= _MAX_CAPTURE_NODES:
                    flags.add("capture_node_limit")
                    copied_items.append(_CAPTURE_NODE_LIMIT)
                    break
                copied_items.append(visit(child, depth + 1, child_ancestors))
            return copied_items
        return copy.deepcopy(item)

    copied = visit(value, 0, frozenset())
    require_json_value(copied, path="$.capture.content")
    return copied, tuple(sorted(flags))


def _protected_environment_input_paths(value: Any) -> tuple[str, ...]:
    """Find structural secrets while allowing ordinary free-form problem text."""

    protected: set[str] = set()

    def visit(item: Any, path: str, depth: int, ancestors: frozenset[int]) -> None:
        if depth > _MAX_CAPTURE_DEPTH:
            protected.add(path)
            return
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in ancestors:
                protected.add(path)
                return
            child_ancestors = ancestors.union((identity,))
            for raw_key, raw_value in item.items():
                if not isinstance(raw_key, str):
                    protected.add(path)
                    continue
                child_path = f"{path}.{raw_key}"
                normalized = _normalize_key(raw_key)
                if normalized in _PROTECTED_ENVIRONMENT_INPUT_KEYS or normalized in _SECRET_KEYS:
                    protected.add(child_path)
                    continue
                # Problem statements are untrusted free-form text. Heuristic phrases
                # in that field are redacted from durable capture, not treated as
                # proof that evaluator-only data crossed the structural boundary.
                if path == "$.model_input" and normalized == "content":
                    continue
                visit(raw_value, child_path, depth + 1, child_ancestors)
            return
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in ancestors:
                protected.add(path)
                return
            child_ancestors = ancestors.union((identity,))
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", depth + 1, child_ancestors)
            return
        if isinstance(item, str) and (_GROUND_TRUTH_TEXT.search(item) or _SECRET_TEXT.search(item)):
            protected.add(path)

    visit(value, "$", 0, frozenset())
    return tuple(sorted(protected))


def _normalize_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
