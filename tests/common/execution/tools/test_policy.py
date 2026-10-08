from __future__ import annotations

import pytest

from alphaapollo.common.execution import ExecutionContext, ToolError, ToolRequest
from alphaapollo.common.execution.tools import ExecutionPolicy, get_tool_spec


def _request(
    tool_id: str = "read",
    arguments: dict[str, object] | None = None,
    *,
    source: str = "openai_tool_call",
) -> ToolRequest:
    return ToolRequest(
        call_id=f"call-{tool_id}",
        tool_id=tool_id,
        arguments={"path": "README.md"} if arguments is None else arguments,
        source=source,
    )


def _authorize(
    request: ToolRequest,
    context: ExecutionContext | None = None,
    policy: ExecutionPolicy | None = None,
) -> ToolError | None:
    return (policy or ExecutionPolicy()).authorize(
        request,
        context or ExecutionContext(session_id="session"),
        get_tool_spec(request.tool_id),
    )


def _assert_policy_error(result: object, code: str) -> ToolError:
    assert isinstance(result, ToolError)
    assert result.stage == "policy"
    assert result.code == code
    return result


@pytest.mark.parametrize(
    "path",
    [
        ".",
        "src/main.py",
        "/workspace",
        "/workspace/src/main.py",
    ],
)
def test_relative_and_workspace_absolute_paths_are_allowed(path: str) -> None:
    assert _authorize(_request(arguments={"path": path})) is None


@pytest.mark.parametrize(
    "path",
    [
        "",
        " ",
        "\x00secret",
        "../secret",
        "src/../../secret",
        "/etc/passwd",
        "/workspace/../etc/passwd",
    ],
)
def test_blank_traversal_and_outside_paths_are_forbidden(path: str) -> None:
    _assert_policy_error(_authorize(_request(arguments={"path": path})), "path_forbidden")


@pytest.mark.parametrize(
    ("tool_id", "arguments"),
    [
        ("grep", {"pattern": "x"}),
        ("find", {"pattern": "*.py"}),
        ("ls", {}),
    ],
)
def test_optional_paths_default_to_workspace_root(
    tool_id: str, arguments: dict[str, object]
) -> None:
    assert _authorize(_request(tool_id, arguments)) is None


def test_custom_workspace_root_is_used_for_checks_and_error_messages() -> None:
    policy = ExecutionPolicy(workspace_root="/repo")

    assert _authorize(_request(arguments={"path": "/repo/src/a.py"}), policy=policy) is None
    error = _assert_policy_error(
        _authorize(_request(arguments={"path": "/workspace/a.py"}), policy=policy),
        "path_forbidden",
    )
    assert "/repo" in error.message


def test_policy_freezes_caller_owned_allowlists() -> None:
    actors = {"solver"}
    modes = {"default"}
    policy = ExecutionPolicy(allowed_actors=actors, allowed_modes=modes)  # type: ignore[arg-type]

    actors.add("guest")
    modes.add("host")

    assert policy.allowed_actors == frozenset({"solver"})
    assert policy.allowed_modes == frozenset({"default"})


@pytest.mark.parametrize(
    ("context", "code"),
    [
        (ExecutionContext(session_id="session", actor="guest"), "actor_forbidden"),
        (ExecutionContext(session_id="session", mode="host"), "mode_forbidden"),
    ],
)
def test_disallowed_actor_and_mode_fail_closed(context: ExecutionContext, code: str) -> None:
    _assert_policy_error(_authorize(_request(), context), code)


def test_internal_python_requires_python_code_ingress() -> None:
    direct = _request("python", {"code": "print(1)"})
    compat = _request("python", {"code": "print(1)"}, source="python_code")

    _assert_policy_error(_authorize(direct), "internal_tool_forbidden")
    assert _authorize(compat) is None


@pytest.mark.parametrize(
    ("system_timeout", "requested_timeout"),
    [
        (None, None),
        (10, None),
        (None, 3),
        (10, 3),
        (3, 10),
    ],
)
def test_valid_timeout_combinations_are_authorized(
    system_timeout: float | None, requested_timeout: float | None
) -> None:
    arguments: dict[str, object] = {"command": "pwd"}
    if requested_timeout is not None:
        arguments["timeout"] = requested_timeout
    context = ExecutionContext(session_id="session", timeout_s=system_timeout)

    assert _authorize(_request("bash", arguments), context) is None


@pytest.mark.parametrize("timeout", [0, -1])
def test_non_positive_tool_timeout_is_a_policy_error(timeout: float) -> None:
    request = _request("bash", {"command": "pwd", "timeout": timeout})

    _assert_policy_error(_authorize(request), "invalid_timeout")


def test_policy_can_require_an_enforceable_timeout() -> None:
    policy = ExecutionPolicy(require_timeout=True)

    _assert_policy_error(
        _authorize(_request("bash", {"command": "pwd"}), policy=policy),
        "timeout_required",
    )
    assert (
        _authorize(
            _request("bash", {"command": "pwd", "timeout": 5}),
            policy=policy,
        )
        is None
    )
    assert (
        _authorize(
            _request("read", {"path": "README.md"}),
            context=ExecutionContext(session_id="session", timeout_s=5),
            policy=policy,
        )
        is None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"workspace_root": "relative"},
        {"workspace_root": "/workspace/../etc"},
        {"workspace_root": "/workspace/"},
        {"workspace_root": "//workspace"},
        {"allowed_actors": frozenset()},
        {"allowed_actors": frozenset({""})},
        {"allowed_actors": "solver"},
        {"allowed_modes": frozenset()},
        {"allowed_modes": frozenset({1})},
        {"require_timeout": 1},
    ],
)
def test_invalid_policy_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ExecutionPolicy(**kwargs)  # type: ignore[arg-type]
