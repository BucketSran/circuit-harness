from __future__ import annotations

from collections.abc import Mapping

import pytest

from alphaapollo.common.execution.tools import (
    INTERNAL_PYTHON_TOOL_ID,
    PI_SOURCE_COMMIT,
    PUBLIC_TOOL_IDS,
    ToolSpec,
    export_openai_tools,
    get_tool_spec,
    list_tool_specs,
)


def _parameters(tool_id: str) -> Mapping[str, object]:
    return get_tool_spec(tool_id).parameters


def test_public_specs_are_exactly_the_seven_pi_tools_in_stable_order() -> None:
    specs = list_tool_specs()

    assert tuple(spec.tool_id for spec in specs) == PUBLIC_TOOL_IDS
    assert PUBLIC_TOOL_IDS == ("read", "bash", "edit", "write", "grep", "find", "ls")
    assert len(specs) == 7
    assert all(spec.model_visible for spec in specs)
    assert all(spec.source.endswith(PI_SOURCE_COMMIT) for spec in specs)


def test_public_parameter_shapes_preserve_pi_names_and_required_fields() -> None:
    expected = {
        "read": ({"path", "offset", "limit"}, {"path"}),
        "bash": ({"command", "timeout"}, {"command"}),
        "edit": ({"path", "edits"}, {"path", "edits"}),
        "write": ({"path", "content"}, {"path", "content"}),
        "grep": (
            {"pattern", "path", "glob", "ignoreCase", "literal", "context", "limit"},
            {"pattern"},
        ),
        "find": ({"pattern", "path", "limit"}, {"pattern"}),
        "ls": ({"path", "limit"}, set()),
    }

    for tool_id, (property_names, required_names) in expected.items():
        parameters = _parameters(tool_id)
        assert parameters["type"] == "object"
        assert set(parameters["properties"]) == property_names
        assert set(parameters.get("required", ())) == required_names

    grep_properties = _parameters("grep")["properties"]
    assert "ignoreCase" in grep_properties
    assert "ignore_case" not in grep_properties

    for tool_id, field, minimum in (
        ("read", "offset", 1),
        ("read", "limit", 1),
        ("grep", "context", 0),
        ("grep", "limit", 1),
        ("find", "limit", 1),
        ("ls", "limit", 1),
    ):
        field_schema = _parameters(tool_id)["properties"][field]
        assert field_schema["type"] == "integer"
        assert field_schema["minimum"] == minimum


def test_edit_schema_uses_pi_multi_hunk_shape_with_explicit_nonempty_adaptation() -> None:
    edit = get_tool_spec("edit")
    edits = edit.parameters["properties"]["edits"]
    item_schema = edits["items"]

    assert edits["type"] == "array"
    assert edits["minItems"] == 1
    assert set(item_schema["properties"]) == {"oldText", "newText"}
    assert set(item_schema["required"]) == {"oldText", "newText"}
    assert edit.adaptations


def test_internal_python_compatibility_spec_is_catalogued_but_not_exported() -> None:
    public_specs = list_tool_specs()
    all_specs = list_tool_specs(include_internal=True)
    python_spec = get_tool_spec(INTERNAL_PYTHON_TOOL_ID)

    assert tuple(spec.tool_id for spec in all_specs[:-1]) == tuple(
        spec.tool_id for spec in public_specs
    )
    assert all_specs[-1] is python_spec
    assert python_spec.model_visible is False
    assert python_spec.source == "AlphaApollo v2 compatibility contract"
    assert set(python_spec.parameters["required"]) == {"code"}
    with pytest.raises(ValueError, match="not model-visible"):
        python_spec.to_openai_tool()


def test_openai_export_has_function_shape_and_returns_independent_mutable_copies() -> None:
    first = export_openai_tools()
    second = export_openai_tools()

    assert [entry["function"]["name"] for entry in first] == list(PUBLIC_TOOL_IDS)
    assert all(entry["type"] == "function" for entry in first)
    assert INTERNAL_PYTHON_TOOL_ID not in {entry["function"]["name"] for entry in first}

    first[0]["function"]["parameters"]["properties"]["path"]["description"] = "changed"
    assert second[0]["function"]["parameters"]["properties"]["path"]["description"] != "changed"
    assert get_tool_spec("read").parameters["properties"]["path"]["description"] != "changed"


def test_canonical_parameters_are_recursively_immutable() -> None:
    parameters = get_tool_spec("edit").parameters

    with pytest.raises(TypeError):
        parameters["type"] = "array"  # type: ignore[index]
    with pytest.raises(TypeError):
        parameters["properties"]["path"]["type"] = "number"  # type: ignore[index]


def test_unknown_tool_id_fails_loudly() -> None:
    with pytest.raises(KeyError, match="unknown tool id"):
        get_tool_spec("missing")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tool_id": "", "description": "description", "parameters": {"type": "object"}},
        {"tool_id": "x", "description": "", "parameters": {"type": "object"}},
        {"tool_id": "x", "description": "description", "parameters": []},
        {"tool_id": "x", "description": "description", "parameters": {"type": "array"}},
        {
            "tool_id": "x",
            "description": "description",
            "parameters": {"type": "object"},
            "model_visible": "yes",
        },
        {
            "tool_id": "x",
            "description": "description",
            "parameters": {"type": "object"},
            "adaptations": ("",),
        },
    ],
)
def test_tool_spec_rejects_invalid_contracts(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ToolSpec(**kwargs)  # type: ignore[arg-type]
