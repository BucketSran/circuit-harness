# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"Immutable tool registry and argument validation."

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from alphaapollo.common.execution.tools.base import (
    ToolError,
    ToolRequest,
    ToolSpec,
    list_tool_specs,
)


@dataclass(frozen=True, slots=True, init=False)
class ToolCatalog:
    """Immutable catalog containing public tools and internal compatibility entries."""

    _specs: Mapping[str, ToolSpec]

    def __init__(self, specs: Iterable[ToolSpec] | None = None) -> None:
        selected = tuple(specs) if specs is not None else list_tool_specs(include_internal=True)
        by_id: dict[str, ToolSpec] = {}
        for spec in selected:
            if spec.tool_id in by_id:
                raise ValueError(f"duplicate tool id: {spec.tool_id}")
            _validate_schema(spec.parameters, path=f"{spec.tool_id}.parameters")
            by_id[spec.tool_id] = spec
        object.__setattr__(self, "_specs", MappingProxyType(by_id))

    def resolve(self, request: ToolRequest) -> ToolSpec | ToolError:
        """Resolve and validate one request without acquiring an execution backend."""
        spec = self._specs.get(request.tool_id)
        if spec is None:
            return _catalog_error(
                request,
                "unknown_tool",
                f"unknown tool: {request.tool_id}",
            )
        problem = _validate_value(request.arguments, spec.parameters, path="$.arguments")
        if problem is not None:
            return _catalog_error(request, "invalid_arguments", problem)
        return spec

    def get(self, tool_id: str) -> ToolSpec:
        """Return one registered spec or raise a stable lookup error."""
        try:
            return self._specs[tool_id]
        except KeyError as exc:
            raise KeyError(f"unknown tool id: {tool_id}") from exc

    def list_specs(self, *, include_internal: bool = False) -> tuple[ToolSpec, ...]:
        """List specs in registration order, hiding internal entries by default."""
        return tuple(
            spec for spec in self._specs.values() if include_internal or spec.model_visible
        )


# Schema keywords _validate_value actually enforces, per declared type. A spec
# using anything else (pattern, multipleOf, minLength, ...) would validate as a
# silent no-op, so registration rejects it instead.
_COMMON_SCHEMA_KEYWORDS = frozenset({"type", "description"})
_NUMERIC_SCHEMA_KEYWORDS = _COMMON_SCHEMA_KEYWORDS | {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
}
_SCHEMA_KEYWORDS_BY_TYPE: Mapping[str, frozenset[str]] = {
    "object": _COMMON_SCHEMA_KEYWORDS | {"properties", "required", "additionalProperties"},
    "array": _COMMON_SCHEMA_KEYWORDS | {"items", "minItems", "maxItems"},
    "string": _COMMON_SCHEMA_KEYWORDS | {"enum"},
    "number": frozenset(_NUMERIC_SCHEMA_KEYWORDS),
    "integer": frozenset(_NUMERIC_SCHEMA_KEYWORDS),
    "boolean": _COMMON_SCHEMA_KEYWORDS,
}


def _validate_schema(schema: object, *, path: str) -> None:
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path} must be a schema mapping")
    expected = schema.get("type")
    allowed = _SCHEMA_KEYWORDS_BY_TYPE.get(expected) if isinstance(expected, str) else None
    if allowed is None:
        raise ValueError(f"{path} uses unsupported schema type {expected!r}")
    unknown = sorted(set(map(str, schema)) - allowed)
    if unknown:
        raise ValueError(
            f"{path} uses schema keyword(s) {unknown} that argument validation does not "
            f"enforce; supported for type {expected!r}: {sorted(allowed)}"
        )
    # Keyword values that _validate_value would silently skip (a string bound,
    # a schema-valued additionalProperties, draft-4 boolean exclusive bounds)
    # are exactly the no-op class this check exists to reject.
    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if keyword in schema:
            bound = schema[keyword]
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise ValueError(f"{path}.{keyword} must be a number")
    for keyword in ("minItems", "maxItems"):
        if keyword in schema:
            count = schema[keyword]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{path}.{keyword} must be a non-negative integer")
    if "enum" in schema:
        options = schema["enum"]
        if (
            isinstance(options, (str, bytes))
            or not isinstance(options, (list, tuple))
            or not options
        ):
            raise ValueError(f"{path}.enum must be a non-empty list")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise ValueError(
            f"{path}.additionalProperties must be a boolean; "
            "schema-valued additionalProperties is not enforced"
        )
    if expected == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path}.properties must be a mapping")
        required = schema.get("required", ())
        if isinstance(required, (str, bytes)) or not isinstance(required, (list, tuple)):
            raise ValueError(f"{path}.required must be a list of field names")
        for name, item in properties.items():
            _validate_schema(item, path=f"{path}.{name}")
    elif expected == "array":
        items = schema.get("items")
        if items is not None:
            _validate_schema(items, path=f"{path}[]")


def _validate_value(value: Any, schema: Mapping[str, Any], *, path: str) -> str | None:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"
        properties = schema.get("properties", {})
        required = schema.get("required", ())
        if not isinstance(properties, Mapping):
            return f"{path} has an invalid object schema"
        if isinstance(required, (str, bytes)) or not isinstance(required, (list, tuple)):
            return f"{path} has an invalid required-fields schema"
        for name in required:
            if not isinstance(name, str):
                return f"{path} has an invalid required field name"
            if name not in value:
                return f"{path}.{name} is required"
        for name, item in value.items():
            item_schema = properties.get(name)
            if item_schema is None:
                if schema.get("additionalProperties") is False:
                    return f"{path}.{name} is not allowed"
                continue
            problem = _validate_value(item, item_schema, path=f"{path}.{name}")
            if problem is not None:
                return problem
        return None
    if expected == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        minimum = schema.get("minItems")
        if isinstance(minimum, int) and not isinstance(minimum, bool) and len(value) < minimum:
            return f"{path} must contain at least {minimum} item(s)"
        maximum = schema.get("maxItems")
        if isinstance(maximum, int) and not isinstance(maximum, bool) and len(value) > maximum:
            return f"{path} must contain at most {maximum} item(s)"
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                problem = _validate_value(item, item_schema, path=f"{path}[{index}]")
                if problem is not None:
                    return problem
        return None
    if expected == "string":
        if not isinstance(value, str):
            return f"{path} must be a string"
        allowed = schema.get("enum")
        if isinstance(allowed, (list, tuple)) and value not in allowed:
            return f"{path} must be one of {list(allowed)!r}"
        return None
    if expected == "number":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return f"{path} must be a finite number"
        return _numeric_bounds_problem(value, schema, path=path)
    if expected == "integer":
        # JSON Schema counts integral floats (25.0) as integers, and models
        # routinely emit them; rejecting each one burns a planner turn.
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{path} must be an integer"
        return _numeric_bounds_problem(value, schema, path=path)
    if expected == "boolean":
        return None if isinstance(value, bool) else f"{path} must be a boolean"
    return f"{path} uses unsupported schema type {expected!r}"


def _numeric_bounds_problem(value: float, schema: Mapping[str, Any], *, path: str) -> str | None:
    def _bound(keyword: str) -> int | float | None:
        candidate = schema.get(keyword)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return candidate
        return None

    minimum = _bound("minimum")
    if minimum is not None and value < minimum:
        return f"{path} must be greater than or equal to {minimum}"
    exclusive_minimum = _bound("exclusiveMinimum")
    if exclusive_minimum is not None and value <= exclusive_minimum:
        return f"{path} must be greater than {exclusive_minimum}"
    maximum = _bound("maximum")
    if maximum is not None and value > maximum:
        return f"{path} must be less than or equal to {maximum}"
    exclusive_maximum = _bound("exclusiveMaximum")
    if exclusive_maximum is not None and value >= exclusive_maximum:
        return f"{path} must be less than {exclusive_maximum}"
    return None


def _catalog_error(request: ToolRequest, code: str, message: str) -> ToolError:
    return ToolError(
        stage="catalog",
        code=code,
        message=message,
        call_id=request.call_id,
        tool_id=request.tool_id,
    )
