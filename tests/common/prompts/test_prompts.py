from __future__ import annotations

from importlib import resources

import pytest

from alphaapollo.common.prompts import (
    PromptConfigError,
    PromptSpec,
    prompt_digest,
    resolve_prompt,
)

_CONSUMER_VARIABLES = (
    ("roles.model_default", "user_template", "render_user", {"problem"}),
    ("roles.problem_analyst", "user_template", "render_user", {"problem"}),
    ("roles.bash_problem_analyst", "user_template", "render_user", {"problem"}),
    ("roles.python_code_problem_analyst", "user_template", "render_user", {"problem"}),
    ("roles.proposer", "user_template", "render_user", {"problem", "candidate"}),
    ("roles.ensemble_proposer", "user_template", "render_user", {"problem", "candidate"}),
    ("roles.bash_proposer", "user_template", "render_user", {"problem", "candidate"}),
    ("roles.python_code_proposer", "user_template", "render_user", {"problem", "candidate"}),
    ("roles.verifier", "user_template", "render_user", {"request_id", "problem", "candidate"}),
    ("roles.reviser", "user_template", "render_user", {"problem", "candidate", "feedback"}),
    (
        "roles.ensemble_reviser",
        "user_template",
        "render_user",
        {"problem", "candidate", "feedback"},
    ),
    ("roles.bash_reviser", "user_template", "render_user", {"problem", "candidate", "feedback"}),
    (
        "roles.python_code_reviser",
        "user_template",
        "render_user",
        {"problem", "candidate", "feedback"},
    ),
    ("roles.finalizer", "user_template", "render_user", {"problem", "candidate", "feedback"}),
    (
        "roles.ensemble_finalizer",
        "user_template",
        "render_user",
        {"problem", "candidate", "feedback"},
    ),
    ("roles.bash_finalizer", "user_template", "render_user", {"problem", "candidate", "feedback"}),
    (
        "roles.python_code_finalizer",
        "user_template",
        "render_user",
        {"problem", "candidate", "feedback"},
    ),
)


def test_resolver_returns_one_cached_immutable_resource() -> None:
    first = resolve_prompt("roles.proposer")
    second = resolve_prompt("roles.proposer")

    assert first is second
    assert first.prompt_ref == "roles.proposer"
    assert first.version == 1
    assert first.variables["user_template"] == {"problem", "candidate"}

    with pytest.raises(TypeError):
        first.variables["user_template"] = frozenset()  # type: ignore[index]


def test_model_default_omits_the_system_message() -> None:
    prompt = resolve_prompt("roles.model_default")

    assert prompt.render_messages(problem="2 + 2") == ({"role": "user", "content": "2 + 2"},)
    assert prompt.digest(problem="2 + 2") == prompt_digest("", "2 + 2")


@pytest.mark.parametrize(("prompt_ref", "template", "renderer", "variables"), _CONSUMER_VARIABLES)
def test_builtin_prompt_variants_match_their_consumer_variables(
    prompt_ref: str,
    template: str,
    renderer: str,
    variables: set[str],
) -> None:
    prompt = resolve_prompt(prompt_ref)

    assert prompt.variables[template] == variables
    assert isinstance(getattr(prompt, renderer)(**dict.fromkeys(variables, "value")), str)


# The answer-format contract is what extraction reads. Adding tool text to a
# role must not reword it: an unextractable answer scores as unscoreable, not
# wrong, so a drifted sentence silently removes the run from the numerator.
_ANSWER_CONTRACT = (
    "Put the reasoning first, then give the final answer only inside "
    "<answer>...</answer>, formatted in LaTeX as \\boxed{...}. Do not put code, "
    "reasoning, or any other format inside <answer>."
)

_TOOL_VARIANTS = (
    ("problem_analyst", False),
    ("proposer", True),
    ("reviser", True),
    ("finalizer", True),
)


@pytest.mark.parametrize(("role", "answers"), _TOOL_VARIANTS)
@pytest.mark.parametrize("prefix", ["bash", "python_code"])
def test_tool_role_variants_add_tool_text_without_touching_the_contract(
    prefix: str, role: str, answers: bool
) -> None:
    """The variant differs from its vanilla twin by tool text and nothing else.

    `aime26_cli.yaml` and `aime26_tool.yaml` granted a tool that no role prompt
    mentioned, so both measured what `aime26_vanilla.yaml` measures. These
    variants say so; the vanilla resources they mirror stay untouched, which is
    what keeps the two arms comparable.
    """

    vanilla = resolve_prompt(f"roles.{role}")
    variant = resolve_prompt(f"roles.{prefix}_{role}")

    assert variant.user_template == vanilla.user_template
    assert (_ANSWER_CONTRACT in vanilla.system) is answers
    assert (_ANSWER_CONTRACT in variant.system) is answers
    if answers:
        # Last, so it is the final instruction the model reads before answering.
        assert variant.system.endswith(_ANSWER_CONTRACT)
    marker = "`bash` tool" if prefix == "bash" else "<python_code>...</python_code>"
    assert marker in variant.system
    assert marker not in vanilla.system


def test_every_packaged_prompt_has_a_consumer_variable_contract() -> None:
    root = resources.files("alphaapollo.common.prompts.resources")
    packaged = {
        f"{category.name}.{resource.name.removesuffix('.yaml')}"
        for category in root.iterdir()
        if category.is_dir() and not category.name.startswith("__")
        for resource in category.iterdir()
        if resource.name.endswith(".yaml")
    }

    assert packaged == {case[0] for case in _CONSUMER_VARIABLES}


@pytest.mark.parametrize(
    "variables, match",
    [
        ({"problem": "p"}, "missing"),
        ({"problem": "p", "candidate": "c", "extra": "x"}, "unexpected"),
    ],
)
def test_render_rejects_variable_drift(variables: dict[str, str], match: str) -> None:
    with pytest.raises(PromptConfigError, match=match):
        resolve_prompt("roles.proposer").render_user(**variables)


def test_prompt_mapping_is_strict() -> None:
    with pytest.raises(PromptConfigError, match="unknown keys"):
        PromptSpec.from_mapping("roles.bad", {"version": 1, "extends": "roles.proposer"})

    with pytest.raises(PromptConfigError, match="invalid placeholder"):
        PromptSpec.from_mapping(
            "roles.bad",
            {"version": 1, "user_template": "{problem.__class__}"},
        )


@pytest.mark.parametrize(
    "prompt_ref",
    ["missing.prompt", "roles/solver", "Roles.solver", []],
)
def test_resolver_rejects_unknown_or_malformed_references(prompt_ref: object) -> None:
    with pytest.raises(PromptConfigError):
        resolve_prompt(prompt_ref)  # type: ignore[arg-type]
