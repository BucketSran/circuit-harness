"""Configuration-only tests: real Harbor factory, no agent installs or API calls."""

import os

import pytest

pytest.importorskip("harbor")

from harbor.agents.factory import AgentFactory

from alphaapollo.workflows.harbor_chips.profiles import compile_agent


@pytest.fixture(autouse=True)
def fixture_environment_only(monkeypatch):
    # Constructors and CLI subprocesses receive only explicitly supplied fixture env.
    monkeypatch.setattr(os, "environ", {})


def catalog():
    return {
        "agents": {"code": {"name": "codex", "kwargs": {"reasoning_effort": "medium"}}},
        "models": {
            "glm": {
                "connections": [
                    {
                        "protocol": "openai-responses",
                        "model": "glm-5",
                        "base_url": "https://fixture.invalid/v1",
                        "key_env": "FIXTURE_KEY",
                    }
                ]
            },
        },
    }


def test_codex_selection_reaches_stock_connection_without_resolving_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("FIXTURE_KEY", "fixture-secret")
    config = compile_agent(catalog(), "code", "glm")
    assert config.env == {
        "OPENAI_API_KEY": "${FIXTURE_KEY}",
        "OPENAI_BASE_URL": "https://fixture.invalid/v1",
    }
    assert "fixture-secret" not in config.model_dump_json()
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    assert agent.model_name == "openai/glm-5"
    assert agent.model_connection.api_key == "fixture-secret"
    assert agent.model_connection.configured_base_url == "https://fixture.invalid/v1"
    assert agent.options.reasoning_effort == "medium"


@pytest.mark.parametrize(
    "name,protocol,model_name,key_var,url_var,model_api",
    [
        (
            "claude-code",
            "anthropic-messages",
            "glm-5",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            None,
        ),
        (
            "pi",
            "openai-responses",
            "openai/glm-5",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "openai-responses",
        ),
        (
            "pi",
            "openai-chat-completions",
            "openai/glm-5",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "openai-completions",
        ),
        (
            "pi",
            "anthropic-messages",
            "anthropic/glm-5",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "anthropic-messages",
        ),
        (
            "mini-swe-agent",
            "openai-chat-completions",
            "openai/glm-5",
            "MSWEA_API_KEY",
            "OPENAI_BASE_URL",
            None,
        ),
        (
            "mini-swe-agent",
            "anthropic-messages",
            "anthropic/glm-5",
            "MSWEA_API_KEY",
            "ANTHROPIC_BASE_URL",
            None,
        ),
    ],
)
def test_stock_agents_receive_explicit_protocol_connection(
    tmp_path, monkeypatch, name, protocol, model_name, key_var, url_var, model_api
):
    monkeypatch.setenv("FIXTURE_KEY", "fixture-secret")
    data = catalog()
    data["agents"]["code"] = {"name": name}
    data["models"]["glm"]["connections"][0]["protocol"] = protocol
    config = compile_agent(data, "code", "glm")
    assert config.env[key_var] == "${FIXTURE_KEY}"
    assert config.env[url_var] == "https://fixture.invalid/v1"
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    assert agent.model_name == model_name
    assert agent.model_connection.api_key == "fixture-secret"
    assert agent.model_connection.configured_base_url == "https://fixture.invalid/v1"
    if model_api:
        assert agent.options.model_api == model_api


def test_agent_and_model_selection_are_independent_and_protocol_is_explicit(tmp_path, monkeypatch):
    from copy import deepcopy

    monkeypatch.setenv("FIXTURE_KEY", "fixture-secret")
    data = catalog()
    data["agents"]["pi"] = {"name": "pi", "kwargs": {"thinking": "low"}}
    data["models"]["deepseek"] = deepcopy(data["models"]["glm"])
    data["models"]["deepseek"]["connections"][0].update(
        model="deepseek-future", base_url="https://other-fixture.invalid/v2"
    )
    data["models"]["glm"]["connections"].append(
        {
            "protocol": "openai-chat-completions",
            "model": "glm-chat",
            "base_url": "https://chat-fixture.invalid/v1",
            "key_env": "FIXTURE_KEY",
        }
    )
    original = deepcopy(data)
    for agent_key in ("code", "pi"):
        config = compile_agent(data, agent_key, "deepseek")
        agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path / agent_key)
        assert agent.model_name == "openai/deepseek-future"
        assert agent.model_connection.configured_base_url == "https://other-fixture.invalid/v2"
    assert compile_agent(data, "code", "glm").model_name == "openai/glm-5"
    with pytest.raises(ValueError, match="ambiguous"):
        compile_agent(data, "pi", "glm")
    assert (
        compile_agent(data, "pi", "glm", "openai-chat-completions").model_name == "openai/glm-chat"
    )
    assert data == original


@pytest.mark.parametrize(
    "name,protocol",
    [
        ("codex", "openai-chat-completions"),
        ("codex", "anthropic-messages"),
        ("claude-code", "openai-responses"),
        ("mini-swe-agent", "openai-responses"),
    ],
)
def test_incompatible_protocol_fails_before_harbor_trial(name, protocol):
    data = catalog()
    data["agents"]["code"] = {"name": name}
    data["models"]["glm"]["connections"][0]["protocol"] = protocol
    with pytest.raises(ValueError, match="compatible"):
        compile_agent(data, "code", "glm")


@pytest.mark.parametrize(
    "agent",
    [
        {"name": "codex", "unknown": 1},
        {"name": "codex", "kwargs": {"unknown": 1}},
        {"name": "oracle"},
        {"kwargs": {}},
        {"name": "codex", "model_name": "other/model"},
        {"name": "codex", "import_path": "other:Agent"},
        {"name": "codex", "env": {"OPENAI_API_BASE": "https://wrong.invalid"}},
        {"name": "claude-code", "kwargs": {"config": {"env": {"ANTHROPIC_MODEL": "wrong"}}}},
        {"name": "codex", "kwargs": {"config": {"model_provider": "wrong"}}},
        {
            "name": "mini-swe-agent",
            "kwargs": {
                "config": {"model": {"model_kwargs": {"api_base": "https://wrong.invalid"}}}
            },
        },
        {"name": "mini-swe-agent", "kwargs": {"config_file": "/not/read/a/file"}},
        {"name": "pi", "kwargs": {"model_api": "wrong"}},
    ],
)
def test_unknown_missing_or_conflicting_agent_settings_fail(agent):
    data = catalog()
    data["agents"]["code"] = agent
    with pytest.raises(ValueError):
        compile_agent(data, "code", "glm")


def test_cli_writes_stock_job_and_harbor_resolves_reference_only_at_factory(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys

    from harbor.models.job.config import JobConfig

    monkeypatch.setenv("FIXTURE_KEY", "fixture-secret")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog()))
    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "job_name": "fixture",
                "tasks": [{"path": "local-fixture-task"}],
                "environment": {"type": "docker"},
                "n_attempts": 2,
                "verifier": {"override_timeout_sec": 321},
            }
        )
    )
    output = tmp_path / "generated.json"
    command = [
        sys.executable,
        "-m",
        "alphaapollo.workflows.harbor_chips.profiles",
        "--catalog",
        str(catalog_path),
        "--agent",
        "code",
        "--model",
        "glm",
        "--job",
        str(job_path),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
    assert completed.returncode == 0, completed.stderr
    assert "fixture-secret" not in output.read_text()
    job = JobConfig.model_validate_json(output.read_text())
    assert job.n_attempts == 2
    assert job.tasks[0].path.name == "local-fixture-task"
    assert job.verifier.override_timeout_sec == 321
    assert job.agents[0].env["OPENAI_API_KEY"] == "${FIXTURE_KEY}"
    agent = AgentFactory.create_agent_from_config(job.agents[0], logs_dir=tmp_path / "logs")
    assert agent.model_connection.api_key == "fixture-secret"
    assert agent.model_name == "openai/glm-5"
    assert agent.model_connection.base_url == "https://fixture.invalid/v1"


def test_compile_job_rejects_unknown_stock_job_and_nested_settings():
    from alphaapollo.workflows.harbor_chips.profiles import compile_job

    for job in (
        {"typo": True},
        {"environment": {"typo": True}},
        {"tasks": [{"path": "fixture", "typo": True}]},
    ):
        with pytest.raises(ValueError, match="unknown"):
            compile_job(catalog(), "code", "glm", job)


@pytest.mark.parametrize(
    "connection",
    [
        {
            "protocol": "openai",
            "model": "glm",
            "base_url": "https://fixture.invalid",
            "key_env": "KEY",
        },
        {
            "protocol": "openai-responses",
            "model": "glm",
            "base_url": "https://key@fixture.invalid",
            "key_env": "KEY",
        },
        {
            "protocol": "openai-responses",
            "model": "glm",
            "base_url": "https://fixture.invalid",
            "key_env": "literal-secret!",
        },
        {
            "protocol": "openai-responses",
            "model": "glm",
            "base_url": "https://fixture.invalid",
            "key_env": "KEY",
            "api_key": "literal-secret",
        },
    ],
)
def test_invalid_model_connection_is_rejected(connection):
    data = catalog()
    data["models"]["glm"]["connections"] = [connection]
    with pytest.raises(ValueError):
        compile_agent(data, "code", "glm")


def test_duplicate_connection_protocol_and_unknown_alias_are_rejected():
    from copy import deepcopy

    data = catalog()
    data["models"]["glm"]["connections"] *= 2
    with pytest.raises(ValueError, match="unique"):
        compile_agent(data, "code", "glm")
    for agent_key, model_key in (("unknown", "glm"), ("code", "unknown")):
        with pytest.raises(ValueError, match="unknown"):
            compile_agent(deepcopy(catalog()), agent_key, model_key)


def test_published_catalog_schema_and_example(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from jsonschema import Draft202012Validator

    from alphaapollo.workflows.harbor_chips.profiles import catalog_schema

    root = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (root / "alphaapollo/workflows/harbor_chips/profiles.schema.json").read_text()
    )
    assert schema == catalog_schema()
    Draft202012Validator.check_schema(schema)
    example = json.loads((root / "examples/chips/harbor/profiles.example.json").read_text())
    Draft202012Validator(schema).validate(example)
    for model_key in example["models"]:
        config = compile_agent(
            example, "pi", model_key, example["models"][model_key]["connections"][0]["protocol"]
        )
        key = example["models"][model_key]["connections"][0]["key_env"]
        monkeypatch.setenv(key, "fixture-secret")
        agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path / model_key)
        assert agent.model_connection.api_key == "fixture-secret"


@pytest.mark.parametrize("protocol", ["openai-chat-completions", "anthropic-messages"])
def test_mini_swe_native_model_config_owns_endpoint_and_chat_protocol(protocol):
    data = catalog()
    data["agents"]["code"] = {
        "name": "mini-swe-agent",
        "kwargs": {
            "config": {"agent": {"step_limit": 12}, "model": {"model_kwargs": {"temperature": 0.2}}}
        },
    }
    data["models"]["glm"]["connections"][0]["protocol"] = protocol
    config = compile_agent(data, "code", "glm")
    assert config.kwargs["config"] == {
        "agent": {"step_limit": 12},
        "model": {
            "model_class": "litellm",
            "model_kwargs": {"temperature": 0.2, "api_base": "https://fixture.invalid/v1"},
        },
    }


def test_mini_swe_chat_rejects_reasoning_option_that_harbor_routes_to_responses():
    data = catalog()
    data["agents"]["code"] = {"name": "mini-swe-agent", "kwargs": {"reasoning_effort": "high"}}
    data["models"]["glm"]["connections"][0]["protocol"] = "openai-chat-completions"
    with pytest.raises(ValueError, match="switches OpenAI to Responses"):
        compile_agent(data, "code", "glm")


def test_circuit_job_wraps_only_stock_lifecycle_and_preserves_selected_connection():
    from alphaapollo.workflows.harbor_chips.profiles import compile_job

    job = compile_job(
        catalog(),
        "code",
        "glm",
        {
            "environment": {
                "import_path": (
                    "alphaapollo.workflows.harbor_chips.docker_environment:CircuitDockerEnvironment"
                ),
                "kwargs": {"config": {"fixture": True}},
            }
        },
    )
    selected = job.agents[0]
    assert selected.name is None
    assert selected.import_path == "alphaapollo.workflows.harbor_chips.installed_agent:CircuitAgent"
    assert selected.kwargs == {
        "agent_name": "codex",
        "agent_kwargs": {"reasoning_effort": "medium"},
    }
    assert selected.model_name == "openai/glm-5"
    assert selected.env["OPENAI_API_KEY"] == "${FIXTURE_KEY}"


def test_circuit_job_rejects_unimplemented_resume_contract():
    from alphaapollo.workflows.harbor_chips.profiles import compile_job

    data = catalog()
    data["agents"]["code"]["resume_trajectory"] = True
    with pytest.raises(ValueError, match="single-step"):
        compile_job(
            data,
            "code",
            "glm",
            {
                "environment": {
                    "import_path": (
                        "alphaapollo.workflows.harbor_chips.docker_environment:"
                        "CircuitDockerEnvironment"
                    )
                }
            },
        )


@pytest.mark.parametrize(
    "native",
    [
        {"model": {"model_kwargs": {"custom_llm_provider": "wrong"}}},
        {"model": {"model_kwargs": None}},
    ],
)
def test_mini_swe_rejects_native_connection_override_and_invalid_model_settings(native):
    data = catalog()
    data["agents"]["code"] = {"name": "mini-swe-agent", "kwargs": {"config": native}}
    data["models"]["glm"]["connections"][0]["protocol"] = "openai-chat-completions"
    with pytest.raises(ValueError):
        compile_agent(data, "code", "glm")


def test_mutated_catalog_is_revalidated_before_composition():
    from alphaapollo.workflows.harbor_chips.profiles import ProfileCatalog

    data = ProfileCatalog.model_validate(catalog())
    data.agents["code"]["kwargs"]["config"] = {"model_provider": "wrong"}
    with pytest.raises(ValueError):
        compile_agent(data, "code", "glm")


@pytest.mark.parametrize(
    "name,key,value",
    [
        ("codex", "CODEX_AUTH_JSON_PATH", "/fixture/auth.json"),
        ("codex", "CODEX_FORCE_AUTH_JSON", "true"),
        ("claude-code", "CLAUDE_FORCE_OAUTH", "true"),
        ("claude-code", "AWS_BEARER_TOKEN_BEDROCK", "fixture-token"),
    ],
)
def test_agent_cannot_override_connection_auth(name, key, value):
    data = catalog()
    data["agents"]["code"] = {"name": name, "env": {key: value}}
    data["models"]["glm"]["connections"][0]["protocol"] = (
        "anthropic-messages" if name == "claude-code" else "openai-responses"
    )
    with pytest.raises(ValueError, match="conflicts"):
        compile_agent(data, "code", "glm")


@pytest.mark.parametrize(
    "name,key",
    [
        ("codex", "CODEX_AUTH_JSON_PATH"),
        ("codex", "CODEX_FORCE_AUTH_JSON"),
        ("claude-code", "CLAUDE_FORCE_OAUTH"),
        ("claude-code", "AWS_BEARER_TOKEN_BEDROCK"),
        ("claude-code", "CLAUDE_CODE_USE_BEDROCK"),
    ],
)
def test_runtime_rejects_ambient_connection_override_after_compilation(
    tmp_path, monkeypatch, name, key
):
    from harbor.models.trial.config import AgentConfig

    data = catalog()
    data["agents"]["code"] = {"name": name}
    data["models"]["glm"]["connections"][0]["protocol"] = (
        "anthropic-messages" if name == "claude-code" else "openai-responses"
    )
    selected = compile_agent(data, "code", "glm")
    monkeypatch.setenv(key, "true")
    monkeypatch.setenv("FIXTURE_KEY", "fixture-secret")
    wrapped = selected.model_copy(
        update={
            "name": None,
            "import_path": "alphaapollo.workflows.harbor_chips.installed_agent:CircuitAgent",
            "kwargs": {"agent_name": name, "agent_kwargs": selected.kwargs},
        }
    )
    # Test persisted JobConfig semantics via the actual factory, not just a dict.
    wrapped = AgentConfig.model_validate_json(wrapped.model_dump_json())
    with pytest.raises(ValueError, match=key):
        AgentFactory.create_agent_from_config(wrapped, logs_dir=tmp_path)


def test_codex_rejects_model_id_that_pinned_harbor_would_truncate():
    data = catalog()
    data["models"]["glm"]["connections"][0]["model"] = "vendor/model-id"
    with pytest.raises(ValueError, match="slash"):
        compile_agent(data, "code", "glm")
    data["agents"]["code"] = {"name": "pi"}
    assert compile_agent(data, "code", "glm").model_name == "openai/vendor/model-id"
