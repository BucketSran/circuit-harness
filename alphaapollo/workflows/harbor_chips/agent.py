"""One cancellable host process per Harbor Agent phase."""

import asyncio
import json
import os
import signal
import time

from harbor.agents.base import BaseAgent

from .config import HARBOR_VERSION, require_harbor_version
from .environment import HarborChipsEnvironment


class NativeAgentExecutionError(RuntimeError):
    """Native CLI failed without a valid completed model attempt."""


class NativeOutputLimitError(NativeAgentExecutionError):
    """Native process exceeded its bounded evidence budget."""


class NativeCodexAgent(BaseAgent):
    @staticmethod
    def name():
        return "harness-native-codex"

    def version(self):
        return f"1/harbor-{HARBOR_VERSION}"

    async def setup(self, environment):
        require_harbor_version()
        if not isinstance(environment, HarborChipsEnvironment):
            raise TypeError("Native Codex requires the Harness native environment")
        if not self.model_name:
            raise ValueError("Native Codex requires an explicit model")

    async def run(self, instruction, environment, context):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        reason = "agent_error"
        process = None
        context.metadata = {
            "schema_version": 1,
            "harbor_version": HARBOR_VERSION,
            "trial_id": str(self.context_id),
            "model_requested": self.model_name,
            "protocol": "native-codex-public-mcp",
            "usage": None,
            "cost": None,
            "conditions": environment.native_conditions(),
        }
        try:
            command, env = environment.command(self.model_name)
            with (
                (self.logs_dir / "native-events.jsonl").open("wb") as out,
                (self.logs_dir / "native-stderr.log").open("wb") as err,
            ):
                launch = asyncio.create_task(
                    asyncio.create_subprocess_exec(
                        *command,
                        cwd=environment.workspace,
                        env=env,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True,
                    )
                )
                try:
                    process = await asyncio.shield(launch)
                except asyncio.CancelledError:
                    process = await launch
                    raise
                await self._capture(
                    process, instruction.encode(), out, err, environment.settings.max_output_bytes
                )
            if process.returncode:
                raise NativeAgentExecutionError(
                    f"Native Codex exited with status {process.returncode}"
                )
            reason = "completed"
        except NativeAgentExecutionError as error:
            context.metadata["execution"] = "infrastructure_error"
            context.metadata["output_limit_exceeded"] = isinstance(error, NativeOutputLimitError)
            raise
        except asyncio.CancelledError:
            reason = "cancelled"
            raise
        finally:
            if process is not None:
                # Kill the complete process group even if its leader exited while children remain.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
                context.metadata["returncode"] = process.returncode
            context.metadata["elapsed_s"] = time.monotonic() - started
            self._record_native_metrics(context, completed=reason == "completed")
            receipt = await environment.freeze(reason)
            context.metadata.update(receipt)
            (self.logs_dir / "attempt.json").write_text(
                json.dumps(context.metadata, indent=2) + "\n"
            )

    async def _capture(self, process, instruction, out, err, limit):
        written = 0

        async def pump(stream, destination):
            nonlocal written
            while chunk := await stream.read(65536):
                remaining = limit - written
                destination.write(chunk[:remaining])
                written += min(len(chunk), remaining)
                if len(chunk) > remaining:
                    raise NativeOutputLimitError(f"Native output exceeds {limit} bytes")

        async def send():
            try:
                process.stdin.write(instruction)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()

        tasks = [
            asyncio.create_task(pump(process.stdout, out)),
            asyncio.create_task(pump(process.stderr, err)),
            asyncio.create_task(send()),
            asyncio.create_task(process.wait()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _record_native_metrics(self, context, *, completed):
        path = self.logs_dir / "native-events.jsonl"
        if not path.exists():
            return
        malformed = 0
        observed_tools = set()
        usages = []
        final_turn_completed = False
        with path.open(errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    malformed += 1
                    continue
                if not isinstance(event, dict):
                    malformed += 1
                    continue
                if not isinstance(event.get("type"), str):
                    malformed += 1
                    continue
                if event["type"].startswith("turn."):
                    final_turn_completed = event.get("type") == "turn.completed"
                item = event.get("item")
                if isinstance(item, dict):
                    kind = item.get("type")
                    if kind in ("mcp_tool_call", "command_execution", "file_change", "web_search"):
                        observed_tools.add(str(item.get("tool", kind)))
                if event.get("type") == "thread.started":
                    context.metadata["native_thread_id"] = event.get("thread_id")
                elif event.get("type") == "turn.completed":
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        usages.append(usage)
        context.metadata["turn_usage"] = usages
        complete_usage = completed and not malformed and final_turn_completed and len(usages) == 1
        context.metadata["usage_completeness"] = (
            "single completed turn" if complete_usage else "unknown"
        )
        if complete_usage:
            usage = usages[0]
            context.metadata["usage"] = usage
            context.metadata["usage_source"] = "native turn.completed"
            for source, target in (
                ("input_tokens", "n_input_tokens"),
                ("output_tokens", "n_output_tokens"),
                ("cached_input_tokens", "n_cache_tokens"),
            ):
                value = usage.get(source)
                if type(value) is int and value >= 0:
                    setattr(context, target, value)
        context.metadata["malformed_native_event_lines"] = malformed
        context.metadata["observed_tools"] = sorted(observed_tools)
