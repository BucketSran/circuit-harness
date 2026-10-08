"""One lifecycle adapter around Harbor agents, independent of model/provider."""

import asyncio

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.trial.config import AgentConfig

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
        self.delegate = agent_class(*args, **kwargs, **(agent_kwargs or {}))

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
        self.delegate.context_id = self.context_id
        self.delegate.session_id = self.session_id
        await self.delegate.setup(environment)

    async def run(self, instruction, environment, context):
        reason = "agent_error"
        try:
            await self.delegate.run(
                instruction=instruction, environment=environment, context=context
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
