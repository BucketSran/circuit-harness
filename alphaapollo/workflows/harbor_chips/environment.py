"""Harbor environment owning the public session, never a second Agent controller."""

import asyncio
import os
import re
import shutil
import sys
from pathlib import Path

from harbor.environments.base import BaseEnvironment
from harbor.environments.capabilities import EnvironmentCapabilities

from .config import HarborChipsConfig, require_harbor_version


class HarborChipsEnvironment(BaseEnvironment):
    def __init__(self, *args, **kwargs):
        require_harbor_version()
        super().__init__(*args, **kwargs)
        self.settings = HarborChipsConfig.read(self.environment_dir / "harness.json")
        self.session_directory = self.trial_paths.trial_dir / "public-session"
        self.workspace = self.trial_paths.trial_dir / "native-public"
        self.frozen = None
        self.broker = None
        self.sandbox = None
        self.native_home = self.workspace / ".codex-home"
        self.public_tools = ()

    @staticmethod
    def type():
        return "harness-native-codex"

    @property
    def capabilities(self):
        return EnvironmentCapabilities(mounted=True)

    def _validate_definition(self):
        import tomllib

        task_file = self.environment_dir.parent / "task.toml"
        if task_file.is_file() and tomllib.loads(task_file.read_text()).get("steps"):
            raise ValueError("native Harness plugins require one single-step Harbor Trial")
        if not (self.environment_dir / "harness.json").is_file():
            raise FileNotFoundError("harness.json operator configuration is required")

    async def start(self, force_build):
        from alphaapollo.common.execution.chips.benchmark_remote import RemoteBenchmarkSpectre
        from alphaapollo.common.execution.chips.benchmark_spectre import package_identity
        from alphaapollo.common.execution.chips.native_sandbox import NativeSandbox

        config = self.settings
        codex = config.executable
        if not codex.is_absolute() or not codex.is_file():
            raise ValueError("declare the absolute native Codex binary")
        if config.auth_file is None or not config.auth_file.is_file():
            raise ValueError("declare an explicit operator auth file; user homes are not imported")
        if not config.runtime_readonly_paths or not config.allowed_hosts:
            raise ValueError("declare exact runtime read grants and model endpoint hosts")
        package = package_identity(config.final_task_package, purpose="final")
        # Validate the operator endpoint without submitting a job or contacting SSH.
        if config.final_backend == "remote_spectre":
            RemoteBenchmarkSpectre(config.final_remote, self.trial_paths.verifier_dir / "preflight")
        for key in ("task_id", "task_version"):
            if package["manifest"][key] != config.task.get(key):
                raise ValueError("public and final task declarations differ")
        process = await asyncio.create_subprocess_exec(
            str(codex), "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=10)
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
        match = re.search(rb"(\d+)\.(\d+)\.(\d+)", output)
        if process.returncode or not match or tuple(map(int, match.groups())) < (0, 154, 0):
            raise RuntimeError("native sandbox requires Codex 0.154.0 or newer")
        self.native_version = output.decode().strip()
        self.workspace.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.native_home.mkdir(mode=0o700, exist_ok=False)
        shutil.copyfile(config.auth_file, self.native_home / "auth.json")
        (self.native_home / "auth.json").chmod(0o400)
        await self._create_public_session()
        from alphaapollo.common.execution.chips.current_evas_session import session_info

        self.public_tools = tuple(
            schema["function"]["name"] for schema in session_info(self.session_directory)["tools"]
        )
        from alphaapollo.workflows.chips_public_mcp import PublicMCPBroker

        self.broker = await PublicMCPBroker(
            self.session_directory,
            self.trial_paths.trial_dir / "public-mcp",
            Path(sys.executable).resolve(),
        ).start()
        self.sandbox = NativeSandbox(
            codex=codex,
            directory=self.trial_paths.trial_dir / "native-policy",
            workspace=self.workspace,
            readonly_paths=(
                *config.runtime_readonly_paths,
                self.broker.client_path,
                self.native_home / "auth.json",
            ),
            socket_paths=(self.broker.socket_path,),
            protected_paths=(
                self.session_directory,
                self.trial_paths.verifier_dir,
                self.environment_dir.parent,
                config.materials,
                *((config.checkout,) if config.checkout is not None else ()),
                *((config.kernel,) if config.kernel is not None else ()),
                config.final_task_package,
                *((config.public_task_package,) if config.public_task_package is not None else ()),
                config.auth_file.parent,
            ),
            allowed_hosts=config.allowed_hosts,
        )
        self.sandbox.probe()

    def command(self, model: str) -> tuple[list[str], dict[str, str]]:
        from alphaapollo.reasoning.runtime.external.bridge.mcp_config import (
            codex_approval_overrides,
            codex_config_overrides,
        )

        if self.sandbox is None or self.broker is None:
            raise RuntimeError("An enforced native sandbox and public broker are required")
        client = self.broker.client_command()
        argv = [
            "/usr/bin/env",
            f"CODEX_HOME={self.native_home}",
            str(self.settings.executable),
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "--json",
            "--sandbox",
            "workspace-write",
            "-C",
            str(self.workspace),
            "-m",
            model,
            "--enable",
            "skip_host_skill_discovery",
        ]
        overrides = [
            'web_search="disabled"',
            "project_doc_max_bytes=0",
            f'model_reasoning_effort="{self.settings.reasoning_effort}"',
            *codex_config_overrides({"harness_public": {"command": client[0], "args": client[1:]}}),
            *codex_approval_overrides({"harness_public": self.public_tools}),
        ]
        for feature in (
            "apps",
            "plugins",
            "hooks",
            "browser_use",
            "computer_use",
            "multi_agent",
            "unbounded_connection_retries",
        ):
            argv.extend(("--disable", feature))
        for override in overrides:
            argv.extend(("-c", override))
        argv.append("-")
        wrapped, env = self.sandbox.command(argv)
        return wrapped, {"PATH": os.defpath, **env}

    def native_conditions(self) -> dict:
        import hashlib

        return {
            "configuration_sha256": hashlib.sha256(
                self.settings.model_dump_json().encode()
            ).hexdigest(),
            "native_cli_version": getattr(self, "native_version", None),
            "sandbox": dict(self.sandbox.receipt) if self.sandbox is not None else None,
            "native_tool_catalog": "not independently observed",
            "public_mcp_tools": list(self.public_tools),
            "task_id": self.settings.task.get("task_id"),
            "task_version": self.settings.task.get("task_version"),
            "public_backend": self.settings.public_backend,
            "simulation_limit": self.settings.max_simulations,
            "action_limit": self.settings.max_actions,
            "reasoning_effort": self.settings.reasoning_effort,
        }

    async def freeze(self, reason: str) -> dict:
        if self.frozen is None:
            from .session import freeze_session

            self.frozen = await freeze_session(
                self.session_directory, reason, self.settings.simulation_timeout_s
            )
        return self.frozen

    async def stop(self, delete: bool):
        # Retain receipts and raw evidence regardless of Harbor's environment-delete flag.
        try:
            if self.session_directory.exists() and self.frozen is None:
                await self.freeze("cancelled")
        finally:
            try:
                if self.broker is not None:
                    await self.broker.close()
            finally:
                if self.native_home.exists():
                    shutil.rmtree(self.native_home)

    async def exec(self, *args, **kwargs):
        raise RuntimeError("Harbor environment does not expose an unrestricted host shell")

    async def upload_file(self, *args, **kwargs):
        raise RuntimeError("Use declared public session materials, not arbitrary host uploads")

    async def upload_dir(self, *args, **kwargs):
        raise RuntimeError("Use declared public session materials, not arbitrary host uploads")

    async def download_file(self, *args, **kwargs):
        raise RuntimeError("Read the frozen candidate receipt, not arbitrary host files")

    async def download_dir(self, *args, **kwargs):
        raise RuntimeError("Read retained trial evidence directly")

    async def _create_public_session(self):
        from .session import prepare_session

        await prepare_session(self.settings, self.session_directory, self.trial_paths.trial_dir)
