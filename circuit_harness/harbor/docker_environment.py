"""Circuit tools beside an unmodified Harbor installed-agent environment."""

import json
import tempfile
import tomllib
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment

from .config import PublicSessionConfig, require_harbor_version
from .session import freeze_session, prepare_session


class CircuitDockerEnvironment(DockerEnvironment):
    """Keep private sessions on the runner; expose only an authenticated public client.

    Harbor owns container execution, agent installation, model access and logs.
    Candidate writes go through the gateway, so closing it also revokes writes
    from background agent processes before the verifier receives the candidate.
    """

    def __init__(
        self,
        *args,
        session_config=None,
        task_bindings=None,
        task_bindings_sha256=None,
        task_binding_receipts=None,
        gateway_bind_host,
        gateway_host,
        **kwargs,
    ):
        require_harbor_version()
        if session_config is not None and task_bindings is not None:
            raise ValueError("scalar session_config and task_bindings are mutually exclusive")
        if task_bindings is not None:
            from .task_bindings import resolve_task_binding, write_binding_receipt

            task_dir = Path(kwargs["environment_dir"]).parent.resolve()
            if task_bindings_sha256 is not None and str(task_dir) not in (
                task_binding_receipts or {}
            ):
                raise ValueError("compiled task binding pins are incomplete")
            selection = resolve_task_binding(
                task_bindings,
                task_dir,
                public_roots=[kwargs["trial_paths"].trial_dir.parent],
                expected_manifest_sha256=task_bindings_sha256,
                expected_receipt=(task_binding_receipts or {}).get(str(task_dir)),
            )
            session_config = selection.session_config_path
            self.task_binding = selection.receipt
            write_binding_receipt(selection, kwargs["trial_paths"].trial_dir)
        if session_config is None:
            raise ValueError("session_config or task_bindings is required")
        self.settings = PublicSessionConfig.read(Path(session_config))
        self.gateway_bind_host = gateway_bind_host
        self.gateway_host = gateway_host
        self.gateway = None
        self.frozen = None
        self._container_started = False
        paths = kwargs["trial_paths"]
        self.session_directory = paths.trial_dir / "public-session"
        task_dir = Path(kwargs["environment_dir"]).parent.resolve()
        config_path = Path(session_config).resolve()
        if config_path.is_relative_to(task_dir) or config_path.is_relative_to(
            paths.trial_dir.resolve()
        ):
            raise ValueError("session_config must be outside the task and trial export directories")

        # Harbor normally shares verifier logs with the agent. This verifier is
        # private, and never runs inside the agent container.
        permitted = {
            (paths.agent_dir.resolve(), "/logs/agent"),
            ((paths.artifacts_dir / "logs/artifacts").resolve(), "/logs/artifacts"),
        }
        mounts = []
        for mount in kwargs.get("mounts", []) or []:
            source = Path(mount["source"]).resolve()
            if source == paths.verifier_dir.resolve() and mount["target"] == "/logs/verifier":
                continue
            if mount["type"] != "bind" or (source, mount["target"]) not in permitted:
                raise ValueError(
                    "circuit environment accepts only Harbor agent-log and artifact mounts"
                )
            mounts.append(mount)
        kwargs["mounts"] = mounts
        super().__init__(*args, **kwargs)

    def _validate_definition(self):
        super()._validate_definition()
        if self._is_windows_container:
            raise ValueError("circuit tools require a Linux agent container with Python 3")
        if (
            self.extra_docker_compose_paths
            or (self.environment_dir / "docker-compose.yaml").exists()
        ):
            raise ValueError("circuit environment requires a single Dockerfile or prebuilt image")
        if (self.environment_dir / "harness.json").exists():
            raise ValueError("move private harness.json outside the task and use session_config")
        task_file = self.environment_dir.parent / "task.toml"
        if task_file.exists() and tomllib.loads(task_file.read_text()).get("steps"):
            raise ValueError("circuit candidate collection requires one single-step Harbor Trial")

    async def start(self, force_build):
        from .public_gateway import PublicSessionGateway

        await prepare_session(self.settings, self.session_directory, self.trial_paths.trial_dir)
        self.gateway = await PublicSessionGateway(
            self.session_directory,
            bind_host=self.gateway_bind_host,
            advertised_host=self.gateway_host,
        ).start()
        # Set before start so cancellation/partial setup still uses Harbor cleanup.
        self._container_started = True
        await super().start(force_build)
        result = await self.exec("mkdir -p /opt/harness", user="root")
        if result.return_code:
            raise RuntimeError("cannot prepare the public tool client")
        with tempfile.TemporaryDirectory(prefix="chips-client-") as directory:
            root = Path(directory)
            config = root / "public.json"
            config.write_text(
                json.dumps(
                    {
                        "url": self.gateway.url,
                        "token": self.gateway.token,
                        "timeout": self.settings.simulation_timeout_s + 15,
                    }
                )
            )
            wrapper = root / "harness-public"
            wrapper.write_text('#!/bin/sh\nexec python3 /opt/harness/public_client.py "$@"\n')
            await self.upload_file(
                Path(__file__).with_name("public_client.py"), "/opt/harness/public_client.py"
            )
            await self.upload_file(config, "/opt/harness/public.json")
            await self.upload_file(wrapper, "/usr/local/bin/harness-public")
        result = await self.exec(
            "chmod 755 /usr/local/bin/harness-public && chmod 444 /opt/harness/* "
            "&& harness-public info",
            user="root",
            timeout_sec=15,
        )
        if result.return_code:
            (self.trial_paths.trial_dir / "public-client-preflight.log").write_text(
                (result.stdout or "") + (result.stderr or "")
            )
            raise RuntimeError(
                "agent container cannot reach the public gateway; "
                "inspect public-client-preflight.log"
            )

    async def freeze(self, reason):
        if self.frozen is None:
            if self.gateway is not None:
                await self.gateway.close()
            self.frozen = await freeze_session(
                self.session_directory, reason, self.settings.simulation_timeout_s
            )
        return self.frozen

    async def stop(self, delete):
        try:
            if (self.session_directory / "session.json").exists():
                await self.freeze("cancelled")
            elif self.gateway is not None:
                await self.gateway.close()
        finally:
            if self._container_started:
                await super().stop(delete)
                self._container_started = False
