"""Circuit session policy over Harbor's existing Podman runtime."""

from harbor.environments.podman import PodmanEnvironment

from .container_commands import BoundedContainerCommands
from .docker_environment import CircuitDockerEnvironment


class CircuitPodmanEnvironment(
    BoundedContainerCommands, CircuitDockerEnvironment, PodmanEnvironment
):
    """Reuse the circuit gateway/freeze and Harbor Podman file/container lifecycle."""

    def __init__(self, *args, **kwargs):
        import os
        from pathlib import Path

        from harbor.environments.docker.docker import _sanitize_docker_compose_project_name

        from circuit_harness.execution.journal import atomic_json

        super().__init__(*args, **kwargs)
        self._podman_project = _sanitize_docker_compose_project_name(self.session_id)
        control = os.environ.get("CHIPS_PODMAN_CONTROL_DIR")
        if control:
            atomic_json(
                Path(control) / "projects" / (self._podman_project + ".json"),
                {"project": self._podman_project},
            )

    async def stop(self, delete):
        # Harbor logs and swallows Compose down errors. Never certify cleanup
        # from that return alone; query the exact owned project afterwards.
        await super().stop(delete)
        if delete:
            import asyncio

            from .container_commands import _spawn_command

            for command in (("ps", "-aq"), ("network", "ls", "-q")):
                process = await _spawn_command(
                    *type(self)._engine_cmd(*command),
                    "--filter",
                    "label=com.docker.compose.project=" + self._podman_project,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                result = await self._collect_buffered_output(process, timeout_sec=10)
                if result.return_code or (result.stdout or "").strip():
                    raise RuntimeError("Podman project cleanup not confirmed; retain the trial")
