from __future__ import annotations

from collections.abc import Mapping

import pytest

from alphaapollo.common.execution import ToolError, ToolRequest
from alphaapollo.common.execution.tools import (
    INTERNAL_PYTHON_TOOL_ID,
    PUBLIC_TOOL_IDS,
    ToolCatalog,
    ToolSpec,
    get_tool_spec,
)


def _request(tool_id: str, arguments: dict[str, object]) -> ToolRequest:
    return ToolRequest(call_id=f"call-{tool_id}", tool_id=tool_id, arguments=arguments)


def _assert_catalog_error(result: object, code: str) -> ToolError:
    assert isinstance(result, ToolError)
    assert result.stage == "catalog"
    assert result.code == code
    return result


@pytest.mark.parametrize(
    ("tool_id", "arguments"),
    [
        ("read", {"path": "README.md"}),
        ("bash", {"command": "python -V"}),
        (
            "edit",
            {
                "path": "notes.txt",
                "edits": [{"oldText": "before", "newText": "after"}],
            },
        ),
        ("write", {"path": "notes.txt", "content": "hello"}),
        ("grep", {"pattern": "ToolRequest"}),
        ("find", {"pattern": "**/*.py"}),
        ("ls", {}),
    ],
)
def test_catalog_resolves_all_public_tools_with_minimal_valid_arguments(
    tool_id: str, arguments: dict[str, object]
) -> None:
    result = ToolCatalog().resolve(_request(tool_id, arguments))

    assert result is get_tool_spec(tool_id)


def test_catalog_contains_internal_python_but_hides_it_from_public_listing() -> None:
    catalog = ToolCatalog()
    request = _request(INTERNAL_PYTHON_TOOL_ID, {"code": "print(1)"})

    assert catalog.resolve(request) is get_tool_spec(INTERNAL_PYTHON_TOOL_ID)
    assert tuple(spec.tool_id for spec in catalog.list_specs()) == PUBLIC_TOOL_IDS
    assert catalog.list_specs(include_internal=True)[-1].tool_id == INTERNAL_PYTHON_TOOL_ID


def test_unknown_tool_returns_typed_catalog_error_with_request_identity() -> None:
    error = _assert_catalog_error(ToolCatalog().resolve(_request("missing", {})), "unknown_tool")

    assert error.call_id == "call-missing"
    assert error.tool_id == "missing"


@pytest.mark.parametrize(
    ("tool_id", "arguments", "message"),
    [
        ("read", {}, "path is required"),
        ("read", {"path": 3}, "path must be a string"),
        ("bash", {"command": "pwd", "timeout": True}, "timeout must be a finite number"),
        ("edit", {"path": "x", "edits": []}, "at least 1 item"),
        (
            "edit",
            {"path": "x", "edits": [{"oldText": "x"}]},
            "newText is required",
        ),
        ("grep", {"pattern": "x", "ignoreCase": "yes"}, "ignoreCase must be a boolean"),
        ("find", {"pattern": "x", "limit": "many"}, "limit must be an integer"),
        ("read", {"path": "x", "offset": 1.5}, "offset must be an integer"),
        ("grep", {"pattern": "x", "context": -1}, "greater than or equal to 0"),
    ],
)
def test_invalid_arguments_return_typed_catalog_errors(
    tool_id: str, arguments: dict[str, object], message: str
) -> None:
    error = _assert_catalog_error(
        ToolCatalog().resolve(_request(tool_id, arguments)),
        "invalid_arguments",
    )

    assert message in error.message


def test_additional_arguments_remain_allowed_by_the_canonical_schema() -> None:
    request = _request("read", {"path": "README.md", "provider_extension": {"value": 1}})

    assert ToolCatalog().resolve(request) is get_tool_spec("read")


def test_catalog_rejects_duplicate_ids_and_unknown_lookup() -> None:
    read = get_tool_spec("read")

    with pytest.raises(ValueError, match="duplicate tool id"):
        ToolCatalog((read, read))
    with pytest.raises(KeyError, match="unknown tool id"):
        ToolCatalog().get("missing")


@pytest.mark.parametrize(
    "parameters",
    [
        {"type": "object", "properties": []},
        {"type": "object", "properties": {}, "required": "path"},
    ],
)
def test_malformed_custom_object_schemas_are_rejected_at_registration(
    parameters: Mapping[str, object],
) -> None:
    spec = ToolSpec(tool_id="custom", description="custom", parameters=parameters)

    with pytest.raises(ValueError, match="must be a"):
        ToolCatalog((spec,))


def test_malformed_required_field_name_fails_closed_at_resolve() -> None:
    spec = ToolSpec(
        tool_id="custom",
        description="custom",
        parameters={"type": "object", "properties": {}, "required": [1]},
    )

    error = _assert_catalog_error(
        ToolCatalog((spec,)).resolve(_request("custom", {})),
        "invalid_arguments",
    )

    assert "invalid" in error.message
