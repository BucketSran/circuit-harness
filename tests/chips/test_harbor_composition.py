"""Stock Harbor agents share one circuit environment; no real model calls."""

import asyncio
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("harbor")
from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from circuit_harness.harbor.docker_environment import CircuitDockerEnvironment
from circuit_harness.harbor.verifier import FrozenCandidateVerifier


def task_config(tmp_path, agent):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True, exist_ok=True)
    (task / "tests").mkdir(exist_ok=True)
    (task / "instruction.md").write_text("Use harness-public to edit dut.va.\n")
    (task / "tests/test.sh").write_text("exit 0\n")
    (task / "task.toml").write_text('[environment]\ndocker_image="python:3.12.12-slim-bookworm"\n')
    config = tmp_path / "operator.json"
    config.write_text(
        json.dumps(
            {
                "task": {"task_id": "fixture", "task_version": "1"},
                "materials": str(tmp_path / "materials"),
                "checkout": str(tmp_path / "source"),
                "kernel": str(tmp_path / "kernel"),
                "image": "sha256:" + "a" * 64,
            }
        )
    )
    return TrialConfig(
        task={"path": task},
        trial_name=agent,
        trials_dir=tmp_path / "trials",
        agent={"name": agent, "model_name": "openai/fixture"},
        environment={
            "import_path": ("circuit_harness.harbor.docker_environment:CircuitDockerEnvironment"),
            "kwargs": {
                "session_config": str(config),
                "gateway_bind_host": "127.0.0.1",
                "gateway_host": "host.docker.internal",
            },
        },
    )


@pytest.mark.parametrize("agent", ["codex", "claude-code", "pi", "mini-swe-agent"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_stock_agents_share_environment(tmp_path, agent, wrapped):
    config = task_config(tmp_path, agent)
    if wrapped:
        config.agent.name = None
        config.agent.import_path = "circuit_harness.harbor.installed_agent:CircuitAgent"
        config.agent.kwargs = {"agent_name": agent}
    trial = asyncio.run(Trial.create(config))
    assert trial.agent.to_agent_info().name == agent
    assert trial.agent.model_name == "openai/fixture"
    assert trial.agent_environment.capabilities.mounted


@pytest.mark.skipif(
    not os.environ.get("CHIPS_TEST_DOCKER_IMAGE"), reason="needs local Docker image"
)
def test_stock_docker_environment_exposes_only_public_session_and_freezes(tmp_path):
    from tests.chips.test_current_evas_session import make_session

    fixture = make_session(tmp_path, image=os.environ["CHIPS_TEST_DOCKER_IMAGE"])
    session = json.loads((fixture / "session.json").read_text())
    config = task_config(tmp_path, "nop")
    config.environment.kwargs["gateway_bind_host"] = "0.0.0.0"
    config.environment.kwargs["gateway_host"] = os.environ.get(
        "CHIPS_TEST_GATEWAY_HOST", "host.docker.internal"
    )
    operator = Path(config.environment.kwargs["session_config"])
    operator.write_text(
        json.dumps(
            {
                "task": session["task"],
                "materials": str(tmp_path / "materials"),
                "checkout": str(tmp_path / "engine"),
                "kernel": str(tmp_path / "kernel"),
                "image": os.environ["CHIPS_TEST_DOCKER_IMAGE"],
            }
        )
    )

    async def run():
        trial = await Trial.create(config)
        trial.paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (trial.paths.verifier_dir / "private-score").write_text("private")
        env = trial.agent_environment
        try:
            await env.start(force_build=False)
            result = await env.exec("harness-public info")
            assert result.return_code == 0, result
            assert json.loads(result.stdout)["task_id"] == "synthetic"
            hidden = await env.exec("test ! -e /logs/verifier/private-score")
            assert hidden.return_code == 0
            write = await env.exec(
                'echo \'{"action_id":"first","tool":"evas_write",'
                '"arguments":{"path":"dut.va","content":"last complete"}}\' | harness-public action'
            )
            assert write.return_code == 0, write
            assert json.loads(write.stdout)["ok"]
            receipt = await env.freeze("completed")
            assert (
                Path(receipt["candidate_directory"]) / "files/dut.va"
            ).read_text() == "last complete"
            late = await env.exec("harness-public info")
            assert late.return_code != 0
        finally:
            await env.stop(delete=True)

    asyncio.run(run())


class LifecycleEnvironment(CircuitDockerEnvironment):
    """Real session/gateway with no container, for phase and cancellation tests."""

    async def start(self, force_build):
        from circuit_harness.harbor.public_gateway import PublicSessionGateway
        from tests.chips.test_current_evas_session import make_session

        self.session_directory = make_session(self.trial_paths.trial_dir)
        self.gateway = await PublicSessionGateway(
            self.session_directory, bind_host="127.0.0.1", advertised_host="127.0.0.1"
        ).start()
        self.started = asyncio.Event()
        self.late = None


class LifecycleAgent(BaseAgent):
    @staticmethod
    def name():
        return "lifecycle-fixture"

    def version(self):
        return "fixture"

    async def setup(self, environment):
        assert self.context_id == environment.context_id
        assert self.session_id

    async def run(self, instruction, environment, context):
        from tests.chips.test_current_evas_session import call

        assert call(
            environment.session_directory, "first", "evas_write", path="dut.va", content="first"
        )["ok"]
        environment.started.set()
        if instruction == "error":
            raise NonZeroAgentExitCodeError("fixture unavailable API")
        if instruction in ("timeout", "cancel"):
            await asyncio.sleep(10)
        context.n_input_tokens = 17
        context.n_output_tokens = 3

    def populate_context_post_run(self, context):
        context.n_input_tokens = 17
        context.n_output_tokens = 3


class LifecycleVerifier(FrozenCandidateVerifier):
    async def evaluate(self, environment, candidate):
        assert environment.frozen is not None
        assert (Path(candidate["candidate_directory"]) / "files/dut.va").read_text() == "first"
        return {
            "state": "completed",
            "score": 0.5,
            "execution": "ok",
            "verdict": "fail",
            "candidate_sha256": candidate["candidate_sha256"],
            "task_id": "fixture",
            "task_version": "1",
            "purpose": "final",
        }


def lifecycle_config(tmp_path, mode):
    config = task_config(tmp_path, "nop")
    (tmp_path / "task/instruction.md").write_text(mode)
    config.agent.name = None
    config.agent.import_path = "circuit_harness.harbor.installed_agent:CircuitAgent"
    config.agent.kwargs = {"agent_name": __name__ + ":LifecycleAgent"}
    config.agent.override_timeout_sec = 0.15 if mode == "timeout" else 2
    config.environment.import_path = __name__ + ":LifecycleEnvironment"
    config.verifier.import_path = __name__ + ":LifecycleVerifier"
    return config


@pytest.mark.parametrize("mode", ["completed", "timeout", "error", "cancel"])
def test_real_trial_delegates_identity_usage_and_freezes_at_phase_end(tmp_path, mode):
    from harbor.trial.hooks import TrialEvent

    async def run():
        trial = await Trial.create(lifecycle_config(tmp_path, mode))
        seen = []

        async def on_end(event):
            seen.append(trial.agent_environment.frozen)

        trial.add_hook(TrialEvent.AGENT_END, on_end)
        running = asyncio.create_task(trial.run())
        if mode == "cancel":
            for _ in range(200):
                if (
                    getattr(trial.agent_environment, "started", None)
                    and trial.agent_environment.started.is_set()
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("fixture agent failed to start")
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            result = trial.result
        else:
            result = await running
        assert seen and all(item and item.get("candidate_sha256") for item in seen)
        if mode in ("error", "cancel"):
            assert result.verifier_result is None
        else:
            assert result.verifier_result.rewards == {"reward": 0.5}
            assert result.agent_result.n_input_tokens == 17
            assert result.agent_result.n_output_tokens == 3
        if mode == "error":
            assert result.exception_info.exception_type == "RuntimeError"
        if mode == "timeout":
            assert result.exception_info.exception_type == "AgentTimeoutError"
        assert result.agent_info.name == "lifecycle-fixture"

    asyncio.run(run())


@pytest.mark.parametrize("where", ["task", "trials/nop"])
def test_private_session_config_cannot_enter_task_or_trial_exports(tmp_path, where):
    config = task_config(tmp_path, "nop")
    old = Path(config.environment.kwargs["session_config"])
    private = tmp_path / where / "private.json"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_bytes(old.read_bytes())
    config.environment.kwargs["session_config"] = str(private)
    with pytest.raises(ValueError, match="outside the task and trial"):
        asyncio.run(Trial.create(config))


def test_private_verifier_config_cannot_enter_public_task(tmp_path):
    config = task_config(tmp_path, "nop")
    private = tmp_path / "task/private.json"
    private.write_text('{"task_package":"/private/final","remote":{}}')
    config.verifier.import_path = "circuit_harness.harbor.verifier:FrozenCandidateVerifier"
    config.verifier.kwargs = {"config_path": str(private)}
    # Harbor creates its verifier at verification time, so construct that actual boundary.
    trial = asyncio.run(Trial.create(config))
    with pytest.raises(ValueError, match="outside task and trial"):
        FrozenCandidateVerifier(
            task=trial.task,
            trial_paths=trial.paths,
            environment=trial.agent_environment,
            config_path=str(private),
        )


@pytest.mark.parametrize(
    "filename, model_name",
    [
        ("config.schema.json", "HarborChipsConfig"),
        ("public_session.schema.json", "PublicSessionConfig"),
        ("final_evaluation.schema.json", "FinalEvaluationConfig"),
    ],
)
def test_operator_schema_snapshots(filename, model_name):
    from jsonschema import Draft202012Validator

    from circuit_harness.harbor import config

    schema = json.loads(Path(config.__file__).with_name(filename).read_text())
    assert schema == getattr(config, model_name).model_json_schema()
    Draft202012Validator.check_schema(schema)
