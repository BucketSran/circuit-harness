"""Cancellation-safe commands for the pinned Harbor container lifecycle."""

import asyncio
import os
import signal

from harbor.environments.docker.docker import DockerEnvironment as CircuitDockerEnvironment


async def _kill_command(process):
    # Only signal a group created by this preflight; never the runner's group.
    try:
        owned_group = (
            getattr(process, "_preflight_owned_group", False)
            or os.getpgid(process.pid) == process.pid
        )
    except ProcessLookupError:
        owned_group = False

    def terminate(sig):
        try:
            if owned_group:
                os.killpg(process.pid, sig)
            elif process.returncode is None:
                process.send_signal(sig)
        except ProcessLookupError:
            pass

    async def finish():
        terminate(signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), 1)
        except TimeoutError:
            terminate(signal.SIGKILL)
            await asyncio.wait_for(process.wait(), 5)
        finally:
            # A CLI can exit before an ignoring plugin child; revoke the group.
            terminate(signal.SIGKILL)

    waiting = asyncio.create_task(finish())
    while True:
        try:
            return await asyncio.shield(waiting)
        except asyncio.CancelledError:
            if waiting.done():
                return waiting.result()


async def _spawn_command(*args, **kwargs):
    launch = asyncio.create_task(
        asyncio.create_subprocess_exec(*args, start_new_session=True, **kwargs)
    )
    try:
        process = await asyncio.shield(launch)
        process._preflight_owned_group = True
        return process
    except asyncio.CancelledError:
        while True:
            try:
                process = await asyncio.shield(launch)
                break
            except asyncio.CancelledError:
                continue
        await _kill_command(process)
        raise


class BoundedContainerCommands:
    """Use existing startup while closing Harbor CLI processes on cancellation."""

    async def _run_docker_compose_command(
        self, command, check=True, timeout_sec=None, stdin_data=None, on_output=None
    ):
        # Version-locked Harbor 0.23 command construction. The only launch change
        # is a private process group, so Compose plugins are cancelled together.
        from harbor.environments.docker.docker import _sanitize_docker_compose_project_name

        runtime = type(self).runtime()
        full_command = [
            *runtime.compose,
            "--project-name",
            _sanitize_docker_compose_project_name(self.session_id),
        ]
        if runtime.supports_compose_project_directory:
            full_command.extend(
                ["--project-directory", str(self.environment_dir.resolve().absolute())]
            )
        for path in self._docker_compose_paths:
            full_command.extend(["-f", str(path.resolve().absolute())])
        full_command.extend(command)
        process = await _spawn_command(
            *full_command,
            env=self._compose_env_vars(include_os_env=True),
            cwd=str(self.environment_dir.resolve().absolute())
            if self.environment_dir.is_dir()
            else None,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        if on_output is not None:
            result = await self._collect_streamed_output(
                process, timeout_sec=timeout_sec, stdin_data=stdin_data, on_output=on_output
            )
        else:
            result = await self._collect_buffered_output(
                process, timeout_sec=timeout_sec, stdin_data=stdin_data
            )
        if check and result.return_code:
            raise RuntimeError("container compose command failed")
        return result

    @staticmethod
    async def _collect_buffered_output(process, *, timeout_sec, stdin_data=None):
        try:
            return await CircuitDockerEnvironment._collect_buffered_output(
                process, timeout_sec=timeout_sec, stdin_data=stdin_data
            )
        except BaseException:
            await _kill_command(process)
            raise

    @staticmethod
    async def _collect_streamed_output(process, *, timeout_sec, stdin_data=None, on_output):
        try:
            return await CircuitDockerEnvironment._collect_streamed_output(
                process, timeout_sec=timeout_sec, stdin_data=stdin_data, on_output=on_output
            )
        except BaseException:
            await _kill_command(process)
            raise

    async def _validate_image_os(self, image_name):
        # Harbor's separate image-inspect process has no cancellation cleanup.
        # Use the same public docker command with the bounded collector here.
        process = await _spawn_command(
            *type(self)._engine_cmd("inspect"),
            "--format",
            "{{.Os}}",
            image_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        result = await self._collect_buffered_output(process, timeout_sec=None)
        if result.return_code or (result.stdout or "").strip() != "linux":
            raise RuntimeError("existing image must be Linux")
