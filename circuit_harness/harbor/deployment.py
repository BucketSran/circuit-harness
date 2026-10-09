"""Bounded operator preflight for the stock Harbor circuit runner.

Static mode reads declarations and local resource metadata/identity bytes. Environment mode reuses
CircuitDockerEnvironment startup, including its public-client info check.
"""

import argparse
import asyncio
import json
import math
import os
import re
import tempfile
import tomllib
from pathlib import Path

from .config import (
    FinalEvaluationConfig,
    PublicSessionConfig,
    require_harbor_version,
    validate_final_separation,
)
from .container_commands import BoundedContainerCommands, _kill_command, _spawn_command
from .docker_environment import CircuitDockerEnvironment
from .podman_environment import CircuitPodmanEnvironment
from .profiles import _read_json, _reject_unknown_fields

ENVIRONMENT = "circuit_harness.harbor.docker_environment:CircuitDockerEnvironment"
PODMAN_ENVIRONMENT = "circuit_harness.harbor.podman_environment:CircuitPodmanEnvironment"
VERIFIER = "circuit_harness.harbor.verifier:FrozenCandidateVerifier"
AGENT = "circuit_harness.harbor.installed_agent:CircuitAgent"


class PreflightEnvironment(BoundedContainerCommands, CircuitDockerEnvironment):
    """Bound Docker preflight command cancellation."""


class PodmanPreflightEnvironment(CircuitPodmanEnvironment):
    """Podman uses bounded commands during both preflight and normal trials."""


def check(name, status, code):
    return {"name": name, "status": status, "code": code}


def static_preflight(path):
    """Validate local scalar declarations without contacting external services.

    Binding mode reads package identity metadata and resource bytes through the
    shared resolver. Public manifest semantics and credentials remain unchecked.
    """
    from harbor.models.job.config import JobConfig
    from harbor.models.task.task import Task

    require_harbor_version()
    raw = _read_json(path)
    job = JobConfig.model_validate(raw)
    _reject_unknown_fields(raw, job, "job")
    if (
        job.environment.import_path not in {ENVIRONMENT, PODMAN_ENVIRONMENT}
        or job.verifier.import_path != VERIFIER
    ):
        raise ValueError("use the stock circuit environment and frozen verifier")
    if job.verifier.disable or job.user_agent or len(job.agents) != 1:
        raise ValueError("one circuit agent and an enabled independent verifier required")
    agent = job.agents[0]
    if (
        agent.import_path != AGENT
        or agent.resume_trajectory
        or agent.load_trajectory
        or not agent.model_name
    ):
        raise ValueError("use the compiled single-step CircuitAgent profile")
    if agent.kwargs.keys() - {"agent_name", "agent_kwargs"} or agent.kwargs.get(
        "agent_name"
    ) not in {"codex", "pi", "claude-code", "mini-swe-agent"}:
        raise ValueError("unknown circuit agent options")
    from harbor.agents.factory import AgentFactory
    from harbor.models.trial.config import AgentConfig

    native = AgentConfig(
        name=agent.kwargs["agent_name"], kwargs=agent.kwargs.get("agent_kwargs", {})
    )
    options = AgentFactory.get_agent_class_from_config(native).options_model.model_validate(
        native.kwargs
    )
    _reject_unknown_fields(native.kwargs, options, "agent_kwargs")
    if job.datasets or not job.tasks or any(task.git_url for task in job.tasks):
        raise ValueError("preflight requires explicit local task paths")
    tasks = [Task(task.path) for task in job.tasks]
    for task in tasks:
        _reject_unknown_fields(
            type(task.config).handle_version_rename(
                tomllib.loads(task.paths.config_path.read_text())
            ),
            task.config,
            "task",
        )
    env = job.environment.kwargs
    verifier = job.verifier.kwargs
    binding_mode = "task_bindings" in env or "task_bindings" in verifier
    pins = {"task_bindings_sha256", "task_binding_receipts"}
    expected_env = {
        "gateway_host",
        "gateway_bind_host",
        "task_bindings" if binding_mode else "session_config",
    }
    expected_verifier = {"task_bindings" if binding_mode else "config_path"}
    if set(env) - pins != expected_env or set(verifier) - pins != expected_verifier:
        raise ValueError("unknown or missing circuit adapter options")
    if not binding_mode and (len(tasks) != 1 or set(env) & pins or set(verifier) & pins):
        raise ValueError("scalar preflight requires one task without binding pins")
    if binding_mode and any(env.get(key) != verifier.get(key) for key in ("task_bindings", *pins)):
        raise ValueError("environment/verifier task bindings must agree")
    for host in (env["gateway_host"], env["gateway_bind_host"]):
        if (
            not isinstance(host, str)
            or not host.strip()
            or host.startswith("REPLACE_")
            or any(c.isspace() for c in host)
            or "/" in host
        ):
            raise ValueError("declare explicit gateway hosts")
    if any(job.jobs_dir.resolve().is_relative_to(task.paths.task_dir.resolve()) for task in tasks):
        raise ValueError("job exports belong outside public task directories")
    roots = [task.paths.task_dir.resolve() for task in tasks] + [job.jobs_dir.resolve()]
    configs = []
    for task in tasks:
        if binding_mode:
            from .task_bindings import resolve_task_binding

            selected = resolve_task_binding(
                env["task_bindings"],
                task.paths.task_dir,
                public_roots=roots,
                expected_manifest_sha256=env.get("task_bindings_sha256"),
                expected_receipt=(env.get("task_binding_receipts") or {}).get(
                    str(task.paths.task_dir.resolve())
                ),
            )
            private = [selected.session_config_path, selected.final_config_path, path]
        else:
            private = [Path(env["session_config"]), Path(verifier["config_path"]), path]
        if any(not value.is_absolute() for value in private):
            raise ValueError("operator config paths must be absolute")
        if any(value.resolve().is_relative_to(root) for value in private for root in roots):
            raise ValueError("private configuration belongs outside task and job exports")
        configs.append(_validate_resources(private, roots))
    return job, tasks, configs


def _validate_resources(private, roots):
    public = PublicSessionConfig.model_validate(_read_json(private[0]))
    final = FinalEvaluationConfig.model_validate(_read_json(private[1]))
    validate_final_separation(public, final.task_package, final.remote)
    if final.remote is not None and (
        set(final.remote)
        != {
            "host",
            "python",
            "bundle",
            "profile",
            "run_root",
            "archive_root",
            "upload_root",
        }
        or any(not isinstance(value, str) or not value for value in final.remote.values())
    ):
        raise ValueError("declare complete final connection metadata")
    declaration = public.task
    required = {
        "task_id",
        "task_version",
        "public_files",
        "candidate_files",
        "feedback_fields",
        "manifest",
    }
    if not required <= declaration.keys() or declaration.keys() - required - {"experiments"}:
        raise ValueError("unknown or missing public task declaration fields")
    if any(
        not isinstance(declaration[key], str) or not declaration[key]
        for key in ("task_id", "task_version")
    ):
        raise ValueError("missing public task identity")
    resources = [public.materials, final.task_package]
    resources.extend(
        value
        for value in (
            public.checkout,
            public.kernel,
            public.public_task_package,
            public.public_codex,
            public.public_python,
        )
        if value is not None
    )
    if any(not value.is_absolute() or not value.exists() for value in resources):
        raise FileNotFoundError("missing declared local resource")
    if any(value.resolve().is_relative_to(root) for value in resources for root in roots):
        raise ValueError("private resources belong outside task and job exports")
    return public


async def check_environment(environment, *, timeout_s, cleanup_timeout_s):
    """Run only environment startup and always attempt finite cleanup.

    The environment's startup owns the public-client info request. No Agent or
    verifier is constructed. A cleanup failure reports retained resources.
    """
    results = []
    try:
        await asyncio.wait_for(environment.start(force_build=False), timeout_s)
        results.append(check("environment", "passed", "public_client_info_ok"))
    except TimeoutError:
        results.append(check("environment", "failed", "environment_timeout"))
    except Exception:
        results.append(check("environment", "failed", "environment_unavailable"))
    finally:
        try:
            await asyncio.wait_for(environment.stop(delete=True), cleanup_timeout_s)
            results.append(check("cleanup", "passed", "created_resources_closed"))
        except TimeoutError:
            results.append(check("cleanup", "failed", "cleanup_timeout"))
        except Exception:
            results.append(check("cleanup", "failed", "cleanup_failed"))
    return results


async def _require_local_docker(timeout_s):
    host = os.environ.get("DOCKER_HOST")
    if not host or os.environ.get("DOCKER_CONTEXT"):
        process = await _spawn_command(
            "docker",
            "context",
            "inspect",
            "--format",
            "{{.Endpoints.docker.Host}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        result = await asyncio.wait_for(
            PreflightEnvironment._collect_buffered_output(process, timeout_sec=None), timeout_s
        )
        if result.return_code:
            raise RuntimeError("Docker endpoint metadata unavailable")
        host = (result.stdout or "").strip()
    if not host.startswith("unix:///"):
        raise ValueError("preflight requires a local Unix Docker endpoint")


async def _inspect_image(image, timeout_s, backend="docker"):
    process = await _spawn_command(
        backend,
        "image",
        "inspect",
        image,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(process.wait(), timeout_s)
        if process.returncode:
            raise RuntimeError("local image unavailable")
    finally:
        await _kill_command(process)


async def _probe_public_runtime(public, directory, timeout_s):
    import threading

    from circuit_harness.execution.sessions.current_evas_public import run_isolated_container

    cancel = threading.Event()
    running = asyncio.create_task(
        asyncio.to_thread(
            run_isolated_container,
            backend=public.public_backend,
            cpu_limit=public.public_cpu_limit,
            image=public.image,
            readonly={},
            writable={},
            command=["python3", "-I", "-S", "-c", "print('public-runtime-ready')"],
            directory=directory / "public-runtime",
            action_id="preflight",
            timeout_s=min(timeout_s, 10),
            max_output_bytes=65536,
            workdir="/tmp",
            cancel=cancel,
        )
    )
    try:
        return await asyncio.shield(running)
    except asyncio.CancelledError:
        cancel.set()
        while not running.done():
            try:
                await asyncio.shield(running)
            except asyncio.CancelledError:
                continue
        raise


async def _environment_preflight(job, task, public, timeout_s, cleanup_timeout_s):
    from harbor.models.trial.paths import TrialPaths

    if public.public_backend not in {"docker", "podman"}:
        return [
            check("environment", "not_checked", "unsupported_public_backend"),
            check("cleanup", "not_checked", "not_started"),
        ]
    # Image IDs cannot name a registry image to pull. Compose policies/sidecars
    # would add installations or network changes outside this bounded check.
    image = task.config.environment.docker_image
    directory = task.paths.environment_dir
    if (
        not isinstance(image, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", image)
        or job.environment.force_build
        or job.environment.mounts
        or job.environment.extra_docker_compose
        or job.environment.extra_allowed_hosts
        or job.environment.env
        or task.has_steps
        or task.config.environment.os.value != "linux"
        or task.config.environment.network_mode.value != "public"
        or any(
            (directory / name).exists()
            for name in ("Dockerfile", "docker-compose.yaml", "harness.json")
        )
    ):
        return [
            check("environment", "failed", "existing_image_only"),
            check("cleanup", "not_checked", "not_started"),
        ]
    backend = "podman" if job.environment.import_path == PODMAN_ENVIRONMENT else "docker"
    try:
        async with asyncio.timeout(timeout_s):
            for engine in {backend, public.public_backend}:
                if engine == "docker":
                    await _require_local_docker(timeout_s)
                elif not os.environ.get("DOCKER_HOST", "").startswith("unix:///"):
                    raise ValueError("Podman preflight requires a local Unix API endpoint")
            await _inspect_image(image, timeout_s, backend)
            await _inspect_image(public.image, timeout_s, public.public_backend)
    except TimeoutError:
        return [
            check("environment", "failed", "image_inspection_timeout"),
            check("cleanup", "not_checked", "not_started"),
        ]
    except ValueError:
        return [
            check(
                "environment",
                "failed",
                "local_docker_required" if backend == "docker" else "local_podman_required",
            ),
            check("cleanup", "not_checked", "not_started"),
        ]
    except (OSError, RuntimeError):
        return [
            check("environment", "failed", "local_image_unavailable"),
            check("cleanup", "not_checked", "not_started"),
        ]
    # All files are private and removable only after cleanup succeeds.
    job.jobs_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="preflight-", dir=job.jobs_dir))
    paths = TrialPaths(directory)
    paths.mkdir()
    if public.public_backend == "podman":
        probe = await _probe_public_runtime(public, directory, timeout_s)
        if probe["execution"] != "ok" or probe["returncode"] != 0:
            return [
                check("environment", "failed", "public_runtime_unavailable"),
                check(
                    "cleanup",
                    "passed" if probe["cleanup_confirmed"] else "failed",
                    "public_probe_closed" if probe["cleanup_confirmed"] else "cleanup_failed",
                ),
            ]
    try:
        from harbor.environments.factory import EnvironmentFactory

        environment_class = (
            PodmanPreflightEnvironment if backend == "podman" else PreflightEnvironment
        )
        environment = EnvironmentFactory.create_environment_from_config(
            config=job.environment.model_copy(
                update={"import_path": __name__ + ":" + environment_class.__name__}
            ),
            environment_dir=task.paths.environment_dir,
            environment_name="circuit-preflight",
            session_id=directory.name,
            trial_paths=paths,
            task_env_config=task.config.environment,
        )
    except Exception:
        return [
            check("environment", "failed", "environment_configuration_invalid"),
            check("cleanup", "not_checked", "not_started"),
        ]
    results = await check_environment(
        environment, timeout_s=timeout_s, cleanup_timeout_s=cleanup_timeout_s
    )
    if results[-1]["status"] == "passed":
        import shutil

        shutil.rmtree(directory)
    return results


def _seconds(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 300:
        raise argparse.ArgumentTypeError("seconds must be finite and in (0, 300]")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--timeout-s", type=_seconds, default=60)
    parser.add_argument("--cleanup-timeout-s", type=_seconds, default=30)
    args = parser.parse_args(argv)
    checks = []
    try:
        job, tasks, publics = static_preflight(args.job.resolve())
        checks.extend(
            [
                check("configuration", "passed", "declarations_valid"),
                check("local_resources", "passed", "declared_paths_exist"),
            ]
        )
    except FileNotFoundError:
        checks.append(check("configuration", "failed", "missing_local_resource"))
    except (ImportError, RuntimeError):
        checks.append(check("configuration", "failed", "dependency_unavailable"))
    except (OSError, ValueError, TypeError, KeyError):
        checks.append(check("configuration", "failed", "configuration_invalid"))
    if checks[0]["status"] == "passed" and args.check_environment:
        for task, public in zip(tasks, publics, strict=True):
            results = asyncio.run(
                _environment_preflight(job, task, public, args.timeout_s, args.cleanup_timeout_s)
            )
            if len(tasks) > 1:
                for result in results:
                    result["task_id"] = public.task["task_id"]
            checks.extend(results)

    else:
        code = "not_requested" if not args.check_environment else "configuration_failed"
        checks.extend(
            [
                check("environment", "not_checked", code),
                check("cleanup", "not_checked", "not_started"),
            ]
        )
    checks.extend(
        check(name, "not_checked", "outside_preflight")
        for name in (
            "model_auth",
            "commercial_license",
            "final_evaluation",
            "public_manifest_semantics",
        )
    )
    print(
        json.dumps({"schema_version": 1, "acceptance": "not_evaluated", "checks": checks}, indent=2)
    )
    return int(any(item["status"] == "failed" for item in checks))


if __name__ == "__main__":
    raise SystemExit(main())
