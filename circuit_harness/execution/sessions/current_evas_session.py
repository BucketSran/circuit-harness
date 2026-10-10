"""Public fixed-manifest EVAS sessions; benchmark correctness remains private."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import sys
from pathlib import Path

from circuit_harness.execution.backends.current_evas import (
    EVAS_SOURCE,
    _validate_manifest,
    snapshot_repository,
)
from circuit_harness.execution.evaluation.candidate_bundle import (
    MAX_CANDIDATE_BYTES,
    declared_files,
    freeze_candidate,
    regular_file,
    verify_candidate,
)
from circuit_harness.execution.runtime.journal import atomic_json, file_digest
from circuit_harness.execution.sessions import public_observations
from circuit_harness.execution.sessions.action_store import ActionRecord, ActionStore
from circuit_harness.execution.sessions.current_evas_public import run_public

TOOLS = {
    "evas_read": {"path": {"type": "string"}},
    "evas_write": {"path": {"type": "string"}, "content": {"type": "string"}},
    "evas_simulate": {},
    "evas_testbench": {"spec": {"type": "string"}},
    "evas_submit": {},
    "evas_experiment": {"analysis": {"type": "string"}, "script": {"type": "string"}},
}
END_REASONS = {
    "completed",
    "timeout",
    "agent_error",
    "cancelled",
    "deadline",
    "final",
    "model_request_limit",
    "output_token_limit",
    "request_bytes_limit",
}


def tool_schemas(*, experiments=False, observations=False, testbench=False):
    return [
        {
            "type": "function",
            "function": {
                "name": tool,
                "description": {
                    **public_observations.READ_DESCRIPTIONS,
                    "evas_read": "Read a public file or candidate; empty path lists files.",
                    "evas_write": "Replace complete bytes of one declared candidate file.",
                    "evas_simulate": "Run the fixed public manifest on frozen input; no score.",
                    "evas_testbench": (
                        "Run a temporary Spectre netlist and support VA files; spec is a JSON "
                        "string with netlist and support_files. "
                        "No score or change to final submission."
                    ),
                    "evas_experiment": "Run a declared Python measurement in Docker; no score.",
                    "evas_submit": "Freeze the last complete candidate and end editing; no score.",
                }[tool],
                "parameters": {
                    "type": "object",
                    "properties": fields,
                    "required": ["artifact_id"]
                    if tool in public_observations.READ_TOOLS
                    else list(fields),
                    "additionalProperties": False,
                },
            },
        }
        for tool, fields in {
            **TOOLS,
            **(public_observations.READ_TOOLS if observations else {}),
        }.items()
        if (experiments or tool != "evas_experiment") and (testbench or tool != "evas_testbench")
    ]


def create_session(
    *,
    task: dict,
    materials: Path,
    checkout: Path | None = None,
    kernel: Path | None = None,
    directory: Path,
    image: str | None = None,
    backend: str = "docker",
    cpu_limit: float | None = 1,
    codex: Path | None = None,
    public_task_package: Path | None = None,
    public_remote: dict | None = None,
    python: Path | str = sys.executable,
    max_actions=24,
    max_simulations=4,
    timeout_s=120,
    max_output_bytes=16 * 1024 * 1024,
) -> dict:
    """Pin operator-declared public materials and one explicit isolated EVAS runtime."""
    from circuit_harness.execution.sessions.current_evas_public import validate_cpu_limit

    validate_cpu_limit(cpu_limit)
    if backend not in {"docker", "podman"} and cpu_limit != 1:
        raise ValueError("cpu_limit applies only to container backends")
    if not isinstance(task, dict) or set(task) - {"experiments", "testbench"} != {
        "task_id",
        "task_version",
        "public_files",
        "candidate_files",
        "feedback_fields",
        "manifest",
    }:
        raise ValueError("invalid public task declaration")
    if any(not isinstance(task[key], str) or not task[key] for key in ("task_id", "task_version")):
        raise ValueError("invalid task identity")
    fields = task["feedback_fields"]
    if (
        not isinstance(fields, list)
        or any(field not in {"diagnostics", "observations"} for field in fields)
        or len(fields) != len(set(fields))
    ):
        raise ValueError("declare only supported public feedback fields")
    public_names, candidate_names = (
        declared_files(task["public_files"]),
        declared_files(task["candidate_files"]),
    )
    declared_files(public_names + candidate_names)
    if "testbench" in task:
        if (
            backend != "remote_spectre"
            or task["testbench"]
            != {"version": "spectre-netlist-v1", "payload_file": ".public-testbench.json"}
            or ".public-testbench.json" in [*public_names, *candidate_names]
        ):
            raise ValueError("invalid remote Spectre testbench declaration")
    experiments = task.get("experiments")
    if "experiments" in task:
        if (
            backend not in {"docker", "podman"}
            or not isinstance(experiments, dict)
            or set(experiments) != {"version", "files", "analyses"}
            or not isinstance(experiments["version"], str)
            or not experiments["version"]
            or experiments["analyses"] != ["python_measurement"]
        ):
            raise ValueError("declare a versioned container-only Python measurement capability")
        experiment_names = declared_files(experiments["files"])
        if any(
            Path(name).suffix not in {".py", ".json", ".va", ".txt"} for name in experiment_names
        ):
            raise ValueError("unsupported public experiment file type")
        declared_files(public_names + candidate_names + experiment_names)
        if "manifest.json" in [*public_names, *experiment_names]:
            raise ValueError("fixed manifest is not editable")
    if backend != "remote_spectre":
        _validate_manifest(task["manifest"])
        if set(task["manifest"]["models"]) != set(candidate_names):
            raise ValueError("fixed manifest must reference exactly the declared candidates")
    if backend not in {"docker", "podman", "native_codex_sandbox", "remote_spectre"}:
        raise ValueError("unsupported public backend")
    if backend in {"docker", "podman"} and (
        not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image)
    ):
        raise ValueError("declare an immutable local container image ID")
    if (
        type(max_actions) is not int
        or not 1 <= max_actions <= 1000
        or type(max_simulations) is not int
        or not 1 <= max_simulations <= 1000
    ):
        raise ValueError("invalid session budgets")
    if (
        type(timeout_s) not in (int, float)
        or not 0 < timeout_s <= 300
        or type(max_output_bytes) is not int
        or not 1 <= max_output_bytes <= MAX_CANDIDATE_BYTES
    ):
        raise ValueError("invalid execution limits")
    if backend != "remote_spectre" and (checkout is None or kernel is None):
        raise ValueError("local EVAS backend requires checkout and kernel")
    kernel = Path(kernel).absolute() if kernel is not None else None
    if (
        backend != "remote_spectre"
        and (kernel.is_symlink() or not kernel.is_file())
        or (backend in {"docker", "podman"} and kernel.read_bytes()[:4] != b"\x7fELF")
    ):
        raise ValueError("Container EVAS requires a Linux ELF kernel, not a host Mach-O binary")
    python = Path(python).absolute()
    if backend == "native_codex_sandbox":
        if image is not None or codex is None or not Path(codex).is_file() or not python.is_file():
            raise ValueError("native sandbox requires Codex/Python executables and no image")
        codex = Path(codex).absolute()
    public_package = None
    if backend == "remote_spectre":
        from circuit_harness.execution.evaluation.benchmark_spectre import package_identity
        from circuit_harness.execution.transport.benchmark_remote import RemoteBenchmarkSpectre

        if image is not None or public_task_package is None or public_remote is None:
            raise ValueError("remote public backend requires package/endpoint and image=None")
        public_package = package_identity(Path(public_task_package), purpose="public")
        declared = public_package["manifest"]
        if any(declared[k] != task[k] for k in ("task_id", "task_version", "feedback_fields")):
            raise ValueError("public package and session declarations differ")
        if task["manifest"] != {"condition_id": declared["condition_id"]}:
            raise ValueError("remote public manifest must declare the package condition_id")
        if declared["candidate_file"] not in candidate_names:
            raise ValueError("public package candidate differs")
        # Syntax validation only; constructing transport never contacts SSH.
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            RemoteBenchmarkSpectre(public_remote, Path(temporary))
    elif public_task_package is not None or public_remote is not None:
        raise ValueError("remote public declarations require remote_spectre")
    contents = {name: regular_file(Path(materials), name).read_bytes() for name in public_names}
    if sum(map(len, contents.values())) > MAX_CANDIDATE_BYTES:
        raise ValueError("public materials exceed limit")
    directory = Path(directory).absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in ("public", "submission", "runtime", "actions", "experiments"):
        (directory / name).mkdir(mode=0o700)
    for name, data in contents.items():
        target = directory / "public" / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_bytes(data)
    source = None
    frozen_kernel = None
    if backend != "remote_spectre":
        source = snapshot_repository(Path(checkout), directory / "runtime/source", EVAS_SOURCE)
        frozen_kernel = directory / "runtime/evas-kernel"
        shutil.copy2(kernel, frozen_kernel)
    if public_package is not None:
        shutil.copytree(public_task_package, directory / "runtime/public-task")
    atomic_json(
        directory / "session.json",
        {
            "schema_version": 1,
            "observation_view_version": public_observations.VIEW_VERSION,
            "backend_config_version": 1,
            "backend": backend,
            "cpu_limit": cpu_limit,
            "public_remote": public_remote,
            "public_package": public_package,
            "codex": str(codex) if codex else None,
            "codex_sha256": file_digest(codex) if codex else None,
            "python": str(python),
            "python_sha256": file_digest(python),
            "task": task,
            "public_files": {name: file_digest(directory / "public" / name) for name in contents},
            "source": source,
            "kernel_sha256": file_digest(frozen_kernel) if frozen_kernel else None,
            "image": image,
            "max_actions": max_actions,
            "max_simulations": max_simulations,
            "timeout_s": timeout_s,
            "max_output_bytes": max_output_bytes,
        },
    )
    return session_info(directory)


def session_info(directory: Path) -> dict:
    directory = Path(directory).absolute()
    config = json.loads((directory / "session.json").read_text())
    previous = list((directory / "actions").iterdir())
    simulations = sum(
        json.loads((item / "request.json").read_text())["tool"]
        in {"evas_simulate", "evas_experiment", "evas_testbench"}
        for item in previous
    )
    return {
        "state": "frozen" if (directory / "frozen.json").exists() else "ready",
        "task_id": config["task"]["task_id"],
        "task_version": config["task"]["task_version"],
        "public_files": list(config["public_files"]),
        "candidate_files": config["task"]["candidate_files"],
        "public_workspace": str(directory / "public"),
        "candidate_workspace": str(directory / "submission"),
        "tools": tool_schemas(
            experiments=bool(config["task"].get("experiments")),
            testbench=bool(config["task"].get("testbench")),
            observations=config.get("observation_view_version") == 1,
        ),
        "experiments": config["task"].get("experiments"),
        "max_actions": config["max_actions"],
        "max_simulations": config["max_simulations"],
        "remaining_actions": max(0, config["max_actions"] - len(previous)),
        "remaining_simulations": max(0, config["max_simulations"] - simulations),
        "backend_config_version": config["backend_config_version"],
        "backend": config["backend"],
        "feedback_fields": config["task"]["feedback_fields"],
        "manifest": config["task"]["manifest"],
        **(
            public_observations.manifest_view(config["task"]["manifest"])
            if config.get("observation_view_version") == 1
            else {}
        ),
        "authority": "public_diagnostic",
    }


def _request(request):
    if not isinstance(request, dict) or set(request) != {"action_id", "tool", "arguments"}:
        raise ValueError("invalid action envelope")
    action_id, tool, args = request["action_id"], request["tool"], request["arguments"]
    if not isinstance(action_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", action_id):
        raise ValueError("invalid action ID")
    if isinstance(tool, str) and tool in public_observations.READ_TOOLS:
        public_observations.validate_read(tool, args)
    elif (
        not isinstance(tool, str)
        or tool not in TOOLS
        or not isinstance(args, dict)
        or set(args) != set(TOOLS[tool])
        or any(not isinstance(value, str) for value in args.values())
    ):
        raise ValueError("invalid tool arguments")
    if len(json.dumps(request).encode()) > MAX_CANDIDATE_BYTES * 6 + 4096:
        raise ValueError("request exceeds limit")
    return action_id, tool, args


def session_action(directory: Path, request: dict) -> dict:
    action_id, tool, args = _request(request)
    directory = Path(directory).absolute()
    store = ActionStore(directory)
    with store.lock() as lock:
        config = json.loads((directory / "session.json").read_text())
        record = store.lookup(action_id, request)
        if record is not None:
            action = record.path
            response = action / "response.json"
            if (
                not response.exists()
                and config["backend"] == "remote_spectre"
                and (action / "snapshot.json").exists()
            ):
                fcntl.flock(lock, fcntl.LOCK_UN)
                return _finish_simulation(
                    directory,
                    config,
                    action,
                    tool,
                    args,
                    {"ok": True, "result": json.loads((action / "snapshot.json").read_text())},
                )
            return record.cached_response()
        if (directory / "frozen.json").exists() or (directory / "episode-end.json").exists():
            return {"ok": False, "error": "submission_frozen"}
        previous = list((directory / "actions").iterdir())
        if any(
            not (item / "response.json").exists() and not (item / "snapshot.json").exists()
            for item in previous
        ):
            return {"ok": False, "error": "unresolved_previous_action", "retry_safe": False}
        if tool in {"evas_simulate", "evas_experiment", "evas_testbench"} and any(
            not (item / "response.json").exists()
            and json.loads((item / "request.json").read_text())["tool"]
            in {"evas_simulate", "evas_experiment", "evas_testbench"}
            for item in previous
        ):
            return {"ok": False, "error": "unresolved_previous_simulation", "retry_safe": False}
        if len(previous) >= config["max_actions"] or (
            len(previous) == config["max_actions"] - 1 and tool != "evas_submit"
        ):
            return {"ok": False, "error": "action_budget_exhausted"}
        simulations = sum(
            json.loads((item / "request.json").read_text())["tool"]
            in {"evas_simulate", "evas_experiment", "evas_testbench"}
            for item in previous
        )
        if (
            tool in {"evas_simulate", "evas_experiment", "evas_testbench"}
            and simulations >= config["max_simulations"]
        ):
            return {"ok": False, "error": "simulation_budget_exhausted"}
        record = store.begin(action_id, request)
        action = record.path
        try:
            result = _execute(directory, config, action, tool, args)
            response = {"ok": True, "result": result}
        except (ValueError, OSError) as error:
            response = {"ok": False, "error": type(error).__name__}
            if isinstance(error, ValueError):
                response["detail"] = str(error)
        if tool not in {"evas_simulate", "evas_experiment", "evas_testbench"} or not response["ok"]:
            return record.complete(response)
    # Snapshot and budget reservation are durable before releasing the edit lock.
    return _finish_simulation(directory, config, action, tool, args, response)


def _finish_simulation(directory, config, action, tool, args, response):
    try:
        if config["backend"] == "remote_spectre":
            from circuit_harness.execution.transport.remote_public import run_remote_public

            runner = run_remote_public
        else:
            runner = run_public
        result = runner(
            directory=directory,
            config=config,
            candidate=action / "candidate",
            output=action / "execution",
            action_id=action.name,
            **(
                {"experiment": action / "experiment", "script": args["script"]}
                if tool == "evas_experiment"
                else {}
            ),
        )
        if result.get("execution") == "unknown_execution":
            return {
                "ok": False,
                "error": "unknown_execution",
                "retry_safe": False,
                "wait_elapsed_s": result.get("wait_elapsed_s"),
            }
        public_fields = {
            "execution",
            "backend",
            "image",
            "cleanup_confirmed",
            "wait_elapsed_s",
            *config["task"]["feedback_fields"],
        }
        projected = {key: value for key, value in result.items() if key in public_fields}
        if config.get("observation_view_version") == 1:
            projected = public_observations.project_observations(
                action,
                projected,
                response["result"],
                waveform=tool == "evas_simulate" and config["backend"] != "remote_spectre",
            )
        response = {"ok": True, "result": {**projected, **response["result"]}}
    except (ValueError, OSError) as error:
        response = {"ok": False, "error": type(error).__name__}
    return ActionRecord(action).complete(response)


def _freeze(directory, config, reason):
    receipt = directory / "frozen.json"
    if receipt.exists():
        result = json.loads(receipt.read_text())
        verify_candidate(Path(result["candidate_directory"]))
        return result
    task = config["task"]
    # A crash after a complete bundle but before the receipt must not lose its bytes.
    bundle = directory / "candidate"
    if bundle.exists():
        frozen = verify_candidate(bundle)
    else:
        frozen = freeze_candidate(
            directory / "submission",
            bundle,
            task["candidate_files"],
            task_id=task["task_id"],
            task_version=task["task_version"],
            reason=reason,
        )
    result = {
        "state": "submitted" if frozen["reason"] == "agent_submit" else "collected",
        "candidate_directory": str(bundle),
        "candidate_sha256": frozen["candidate_sha256"],
        "collection_source": frozen["reason"],
        "task_correctness": "not_evaluated",
    }
    atomic_json(receipt, result)
    return result


def _execute(directory, config, action, tool, args):
    task = config["task"]
    if tool in public_observations.READ_TOOLS:
        if config.get("observation_view_version") != 1:
            raise ValueError("observation reads require a versioned session")
        if tool == "evas_observe":
            return public_observations.observe(directory, config, args)
        return public_observations.read_artifact(directory, config, args)
    experiment_files = task.get("experiments", {}).get("files", [])
    if tool == "evas_read":
        name = args["path"]
        if not name:
            return {
                "public_files": list(config["public_files"]),
                "candidate_files": task["candidate_files"],
                "experiment_files": experiment_files,
            }
        root = directory / (
            "public"
            if name in config["public_files"]
            else "experiments"
            if name in experiment_files
            else "submission"
        )
        if name not in {*config["public_files"], *task["candidate_files"], *experiment_files}:
            raise ValueError("undeclared public path")
        path = regular_file(root, name)
        if path.stat().st_size > MAX_CANDIDATE_BYTES:
            raise ValueError("file exceeds read limit")
        if name in config["public_files"] and file_digest(path) != config["public_files"][name]:
            raise ValueError("public material changed")
        return {"path": name, "content": path.read_text()}
    if tool == "evas_write":
        name, data = args["path"], args["content"].encode()
        if (
            name not in [*task["candidate_files"], *experiment_files]
            or len(data) > MAX_CANDIDATE_BYTES
        ):
            raise ValueError("undeclared or oversized candidate")
        parent = directory / ("experiments" if name in experiment_files else "submission")
        target = parent / name
        for part in Path(name).parts[:-1]:
            parent /= part
            if parent.is_symlink():
                raise ValueError("symlink candidate parent")
            parent.mkdir(mode=0o700, exist_ok=True)
        if target.is_symlink():
            raise ValueError("symlink candidate")
        temporary = action / "candidate.tmp"
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        return {"path": name, "sha256": file_digest(target), "bytes": len(data)}
    if tool == "evas_submit":
        return _freeze(directory, config, "agent_submit")
    if tool == "evas_experiment" and (
        not experiment_files
        or args["analysis"] != "python_measurement"
        or args["script"] not in experiment_files
        or not args["script"].endswith(".py")
    ):
        raise ValueError("unsupported or undeclared public experiment")
    source = directory / "submission"
    files = task["candidate_files"]
    if tool == "evas_testbench":
        if config["backend"] != "remote_spectre" or not task.get("testbench"):
            raise ValueError("temporary Spectre testbenches are not declared")
        spec = json.loads(args["spec"])
        if (
            not isinstance(spec, dict)
            or set(spec) != {"netlist", "support_files"}
            or not isinstance(spec["netlist"], str)
            or not spec["netlist"]
            or not isinstance(spec["support_files"], dict)
            or len(spec["support_files"]) > 16
            or any(not isinstance(v, str) for v in spec["support_files"].values())
            or len(args["spec"].encode()) > MAX_CANDIDATE_BYTES
        ):
            raise ValueError("invalid or oversized testbench specification")
        names = declared_files(list(spec["support_files"])) if spec["support_files"] else []
        if any(not name.endswith(".va") or name in files for name in names):
            raise ValueError("support models require separate declared VA paths")
        source = action / "diagnostic-source"
        source.mkdir(mode=0o700)
        for name in files:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(regular_file(directory / "submission", name), target)
        payload = task["testbench"]["payload_file"]
        atomic_json(source / payload, spec)
        files = [*files, payload]
    snapshot = action / "candidate"
    frozen = freeze_candidate(
        source,
        snapshot,
        files,
        task_id=task["task_id"],
        task_version=task["task_version"],
        reason="simulation",
    )
    result = {
        "candidate_sha256": frozen["candidate_sha256"],
        "authority": "public_diagnostic",
        "task_correctness": "not_evaluated",
    }
    if tool == "evas_experiment":
        workspace = action / "workspace"
        workspace.mkdir(mode=0o700)
        for root, names in (
            (directory / "submission", task["candidate_files"]),
            (directory / "public", list(config["public_files"])),
            (directory / "experiments", experiment_files),
        ):
            for name in names:
                source = regular_file(root, name)
                if root.name == "public" and file_digest(source) != config["public_files"][name]:
                    raise ValueError("public material changed")
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        experiment = freeze_candidate(
            workspace,
            action / "experiment",
            [*task["candidate_files"], *config["public_files"], *experiment_files],
            task_id=task["task_id"],
            task_version=json.dumps(
                [task["task_version"], task["experiments"]["version"]], separators=(",", ":")
            ),
            reason="public_experiment",
        )
        result["experiment_sha256"] = experiment["candidate_sha256"]
        result["authority"] = "agent_measurement"
    atomic_json(action / "snapshot.json", result)
    return result


def close_session(directory: Path, reason: str) -> dict:
    """Collect the last complete candidate after the Agent has stopped."""
    if reason not in END_REASONS:
        raise ValueError("unsupported episode end reason")
    directory = Path(directory).absolute()
    with (directory / ".session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"state": "awaiting_action_recovery"}
        receipt = directory / "episode-end.json"
        if receipt.exists():
            return json.loads(receipt.read_text())
        if any(
            not (item / "response.json").exists() and not (item / "snapshot.json").exists()
            for item in (directory / "actions").iterdir()
        ):
            return {"state": "awaiting_action_recovery"}
        config = json.loads((directory / "session.json").read_text())
        try:
            result = _freeze(directory, config, reason)
        except ValueError:
            result = {
                "state": "candidate_integrity_error"
                if (directory / "candidate").exists()
                else "missing_candidate",
                "task_correctness": "not_evaluated",
            }
        result = {**result, "termination_reason": reason}
        atomic_json(receipt, result)
        return result
