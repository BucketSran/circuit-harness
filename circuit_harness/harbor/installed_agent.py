"""One lifecycle adapter around Harbor agents, independent of model/provider."""

import asyncio
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.oracle import OracleAgent
from harbor.models.trial.config import AgentConfig
from harbor.models.trial.paths import TrialPaths

from .config import require_harbor_version
from .docker_environment import CircuitDockerEnvironment
from .profiles import validate_connection_environment


class CircuitAgent(BaseAgent):
    """Delegate all agent behavior; revoke public writes when its phase ends.

    Harbor has no CLI-configurable Trial hook for this custom environment.
    This adapter adds no agent loop, CLI parser or model transport. Its only
    policy is phase-end collection and keeping CLI infrastructure failure unscored.
    """

    def __init__(self, *args, agent_name, agent_kwargs=None, **kwargs):
        require_harbor_version()
        validate_connection_environment(agent_name, kwargs.get("extra_env"))
        super().__init__(*args, **kwargs)
        config = AgentConfig(name=agent_name)
        agent_class = AgentFactory.get_agent_class_from_config(config)
        if issubclass(agent_class, CircuitAgent):
            raise ValueError("CircuitAgent cannot wrap itself")
        delegate_kwargs = dict(agent_kwargs or {})
        self.oracle_task_dir = None
        if issubclass(agent_class, OracleAgent):
            if "trial_paths" in delegate_kwargs:
                raise ValueError("Oracle trial paths come from Harbor's agent logs")
            task_dir = delegate_kwargs.get("task_dir")
            if task_dir is None:
                raise ValueError("CircuitAgent oracle requires agent_kwargs.task_dir")
            self.oracle_task_dir = Path(task_dir).resolve()
            delegate_kwargs["task_dir"] = self.oracle_task_dir
            delegate_kwargs["trial_paths"] = TrialPaths(trial_dir=self.logs_dir.parent)
        self.delegate = agent_class(*args, **kwargs, **delegate_kwargs)

    @staticmethod
    def name():
        return "harness-circuit"

    @classmethod
    def preflight(cls, kwargs=None, env=None):
        options = dict(kwargs or {})
        name = options.pop("agent_name", None)
        nested = options.pop("agent_kwargs", {})
        if name is None or options:
            raise ValueError("declare agent_name and agent_kwargs for CircuitAgent")
        agent_class = AgentFactory.get_agent_class_from_config(AgentConfig(name=name))
        if issubclass(agent_class, CircuitAgent):
            raise ValueError("CircuitAgent cannot wrap itself")
        validate_connection_environment(name, env)
        agent_class.preflight(kwargs=nested, env=env)

    def version(self):
        return self.delegate.version()

    def to_agent_info(self):
        return self.delegate.to_agent_info()

    async def setup(self, environment):
        if not isinstance(environment, CircuitDockerEnvironment):
            raise TypeError("CircuitAgent requires the public circuit environment")
        if self.oracle_task_dir is not None:
            if self.oracle_task_dir != Path(environment.environment_dir).parent.resolve():
                raise ValueError("Oracle task_dir differs from the current Harbor task")
            if self.logs_dir.parent.resolve() != environment.trial_paths.trial_dir.resolve():
                raise ValueError("Oracle logs differ from the current Harbor trial")
        self.delegate.context_id = self.context_id
        self.delegate.session_id = self.session_id
        await self.delegate.setup(environment)

    async def run(self, instruction, environment, context):
        reason = "agent_error"
        try:
            await self.delegate.run(
                instruction=instruction, environment=environment, context=context
            )
            if self.oracle_task_dir is not None:
                exit_code = self.logs_dir / "exit-code.txt"
                if exit_code.exists() and exit_code.read_text().strip() != "0":
                    raise RuntimeError(
                        "Harbor oracle reference solution failed; retained without grading"
                    )
            reason = "completed"
        except NonZeroAgentExitCodeError as error:
            # Harbor normally grades these exits. A failed CLI/API is not a
            # circuit attempt with an independently valid score.
            raise RuntimeError(
                "Harbor agent execution failed; candidate retained without grading"
            ) from error
        except asyncio.CancelledError:
            reason = "cancelled"
            raise
        finally:
            await environment.freeze(reason)

    def populate_context_post_run(self, context):
        self.delegate.populate_context_post_run(context)
