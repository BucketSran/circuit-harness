"""Bounded Analog RLC episode: public files, candidate edits, diagnostics and freeze."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import time
from pathlib import Path

from circuit_harness.execution.evaluation.analog_design_bench import (
    TASKS,
    rlc_contract,
    run_case,
    tree_digest,
)
from circuit_harness.execution.runtime.journal import atomic_json, file_digest
from circuit_harness.execution.sessions.action_store import ActionStore
from circuit_harness.execution.sessions.analog_public import (
    TASK_ID,
    run_public_rlc,
    validate_rlc_candidate,
)
from circuit_harness.execution.sessions.session_budget import budget_rejection, with_budget

# Compatibility surface for callers that inspect the original default task.
PUBLIC_FILES = rlc_contract(TASK_ID).public_files
TOOLS = {
    "analog_read": (
        "Read a public task file; an empty path lists public files.",
        {"path": {"type": "string"}},
    ),
    "analog_write": (
        "Replace the one candidate circuit.spi with a bounded passive RLC subcircuit.",
        {"content": {"type": "string"}},
    ),
    "analog_simulate": (
        "Run public ngspice benches on the current candidate; this is not a score.",
        {},
    ),
    "analog_history": (
        "List saved candidate hashes and their public simulation feedback; no final scores.",
        {},
    ),
    "analog_restore": (
        "Restore exact bytes of a candidate saved in this session by its SHA-256. "
        "Compare public feedback before choosing; this consumes an action, not a simulation.",
        {"candidate_sha256": {"type": "string"}},
    ),
    "analog_submit": ("Freeze the candidate; no final score is shown.", {}),
}

END_REASONS = frozenset(
    {"final", "model_request_limit", "output_token_limit", "request_bytes_limit", "deadline"}
)


def _adapt_instruction(text: str) -> str:
    return (
        "## Harness execution adapter\nThe original upstream instruction is preserved below. "
        "Use analog_read for public files and analog_write for the declared circuit.spi. "
        "Use analog_simulate for the upstream public diagnostic commands; do not execute "
        "those commands yourself or edit testbenches. The following harness completion "
        "rules replace the upstream workspace and timeout-collection mechanics. "
        "The upstream circuit interface and numerical requirements still apply.\n\n"
        "## Original upstream instruction\n"
        + text
        + "\n\n## Harness completion rules\nUse analog_submit to finish early. Otherwise the "
        "harness collects the last complete candidate at a normal or budget stop. "
        "Automatic collection is recorded separately from your explicit submission. "
        "A missing candidate is recorded as missing_candidate without an invented score. "
        "Uncertain server actions must be recovered before collection. "
        "Use analog_history to compare saved candidates and analog_restore to recover a "
        "previous version before submitting. The harness does not choose the best version. "
        "Final scores are never returned to the Agent.\n"
    )


def tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
        }
        for name, (description, properties) in TOOLS.items()
    ]


def create_session(
    source_root: Path,
    directory: Path,
    *,
    task_id: str = TASK_ID,
    podman: str = "podman",
    podman_root: Path | None = None,
    podman_runroot: Path | None = None,
    runtime_image: str | None = None,
    offline_image_archive: Path | None = None,
    podman_single_id: bool = False,
    podman_no_cpu_limit: bool = False,
    timeout_s: int = 120,
    max_actions: int = 24,
    max_simulations: int = 4,
) -> dict:
    """Copy only the pinned public slice into a new private session directory."""
    if not 1 <= max_actions <= 100 or not 1 <= max_simulations <= 10 or not 1 <= timeout_s <= 300:
        raise ValueError("invalid session limits")
    contract = rlc_contract(task_id)
    source_root = Path(source_root).absolute()
    source = source_root / "tasks" / task_id
    if tree_digest(source) != TASKS[task_id].source_sha256:
        raise ValueError("public task source pin mismatch")
    contents = {
        name: (source / relative).read_bytes() for name, relative in contract.public_files.items()
    }
    if any(len(data) > 100_000 for data in contents.values()):
        raise ValueError("public task file exceeds read limit")
    directory = Path(directory).absolute()
    if directory.exists() or directory.is_relative_to(source_root):
        raise ValueError("use a new session directory outside the task source")
    directory.mkdir(parents=True, mode=0o700)
    public = directory / "public"
    public.mkdir(mode=0o700)
    original = directory / "original-instruction.md"
    original.write_bytes(contents["instruction.md"])
    original.chmod(0o600)
    contents["instruction.md"] = _adapt_instruction(
        contents["instruction.md"].decode("utf-8")
    ).encode("utf-8")
    for name, data in contents.items():
        target = public / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_bytes(data)
        target.chmod(0o600)
    atomic_json(
        directory / "session.json",
        {
            "schema_version": 3,
            "collection_policy": "episode_end",
            "budget_feedback": True,
            "task_id": task_id,
            "task_contract_sha256": contract.sha256,
            "source_root": str(source_root),
            "source_sha256": TASKS[task_id].source_sha256,
            "public_files": {name: file_digest(public / name) for name in contract.public_files},
            "podman": podman,
            "podman_root": str(podman_root) if podman_root is not None else None,
            "podman_runroot": str(podman_runroot) if podman_runroot is not None else None,
            "runtime_image": runtime_image,
            "offline_image_archive": (
                str(offline_image_archive) if offline_image_archive is not None else None
            ),
            "podman_single_id": podman_single_id,
            "podman_no_cpu_limit": podman_no_cpu_limit,
            "timeout_s": timeout_s,
            "max_actions": max_actions,
            "max_simulations": max_simulations,
        },
    )
    return {"state": "ready", "task_id": task_id, "tools": list(TOOLS)}


def session_task_id(config: dict) -> str:
    """Resolve the stored task; old sessions belong only to the original bandpass task."""
    task_id = config.get("task_id", TASK_ID)
    contract = rlc_contract(task_id)
    if config.get("schema_version", 1) >= 3:
        if config.get("task_contract_sha256") != contract.sha256:
            raise ValueError("session task contract changed")
    elif task_id != TASK_ID:
        raise ValueError("session task contract changed: a new task requires schema version 3")
    return task_id


def session_action(directory: Path, request: dict) -> dict:
    """Persist each call before executing it; an uncertain call is never replayed."""
    action_id, tool, arguments = _validate_request(request)
    directory = Path(directory).absolute()
    store = ActionStore(directory)
    with store.lock():
        actions = store.actions
        actions.mkdir(mode=0o700, exist_ok=True)
        record = store.lookup(action_id, request)
        if record is not None:
            return record.cached_response()
        config = json.loads((directory / "session.json").read_text())
        session_task_id(config)
        if (directory / "ending.json").exists():
            return {"ok": False, "error": "episode_closed"}
        if len(list(actions.iterdir())) >= config["max_actions"]:
            return with_budget({"ok": False, "error": "action_budget_exhausted"}, directory, config)
        if (directory / "frozen.json").exists():
            return {"ok": False, "error": "submission_frozen"}
        if any(not (previous / "response.json").exists() for previous in actions.iterdir()):
            return {"ok": False, "error": "unresolved_previous_action", "retry_safe": False}
        rejected = budget_rejection(directory, config, tool)
        if rejected is not None:
            return rejected
        record = store.begin(action_id, request)
        started = time.monotonic()
        try:
            reply = {"ok": True, "result": _execute(directory, config, action_id, tool, arguments)}
        except (ValueError, OSError) as error:
            reply = {"ok": False, "error": type(error).__name__}
        reply = with_budget(reply, directory, config)
        return record.complete(reply, started=started)


def _validate_request(request: dict) -> tuple[str, str, dict]:
    if not isinstance(request, dict) or set(request) != {"id", "tool", "arguments"}:
        raise ValueError("invalid action envelope")
    action_id, tool, arguments = request["id"], request["tool"], request["arguments"]
    if not isinstance(action_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", action_id):
        raise ValueError("invalid action ID")
    if (
        not isinstance(tool, str)
        or tool not in TOOLS
        or not isinstance(arguments, dict)
        or set(arguments) != set(TOOLS[tool][1])
    ):
        raise ValueError("invalid tool or arguments")
    if any(not isinstance(value, str) for value in arguments.values()):
        raise ValueError("arguments must be strings")
    if len(json.dumps(request).encode("utf-8")) > 150_000:
        raise ValueError("action exceeds request limit")
    return action_id, tool, arguments


def enqueue_action(directory: Path, request: dict) -> dict:
    """Start a standalone worker before acknowledging an action over SSH."""
    import subprocess
    import sys

    from circuit_harness.execution.runtime.bundle import build_cli

    action_id, _, _ = _validate_request(request)
    directory = Path(directory).absolute()
    with (directory / ".enqueue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = directory / "requests" / action_id
        if target.exists():
            if json.loads((target / "request.json").read_text()) != request:
                raise ValueError("action ID belongs to another request")
            if not (target / "started.json").exists():
                return {"state": "unknown_execution", "id": action_id, "retry_safe": False}
            return {"state": "accepted", "id": action_id}
        if (directory / "ending.json").exists():
            return {"state": "episode_closed", "id": action_id}
        target.mkdir(mode=0o700, parents=True)
        atomic_json(target / "request.json", request)
        worker_sha256 = build_cli(target / "worker.pyz")
        allowed = {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "LANG",
            "LC_ALL",
            "XDG_RUNTIME_DIR",
            "XDG_CONFIG_HOME",
            "DBUS_SESSION_BUS_ADDRESS",
        }
        worker_env = {key: value for key, value in os.environ.items() if key in allowed}
        worker_env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        with (target / "worker.log").open("wb") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(target / "worker.pyz"),
                    "analog-action",
                    "--session",
                    str(directory),
                    "--request",
                    str(target / "request.json"),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env=worker_env,
            )
        atomic_json(target / "started.json", {"pid": child.pid, "worker_sha256": worker_sha256})
        return {"state": "accepted", "id": action_id}


def action_response(directory: Path, action_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", action_id):
        raise ValueError("invalid action ID")
    response = Path(directory) / "actions" / action_id / "response.json"
    if response.exists():
        return json.loads(response.read_text())
    request = Path(directory) / "requests" / action_id
    # Rejections before action creation (frozen session, exhausted budget) are
    # terminal worker replies too. The CLI persists them in the request spool.
    if (request / "response.json").is_file():
        return json.loads((request / "response.json").read_text())
    if request.exists() and not (request / "started.json").exists():
        return {"state": "unknown_execution", "id": action_id, "retry_safe": False}
    return {"state": "pending"}


def _execute(directory: Path, config: dict, action_id: str, tool: str, arguments: dict) -> dict:
    task_id = session_task_id(config)
    public_files = rlc_contract(task_id).public_files
    public = directory / "public"
    if tool == "analog_read":
        name = arguments["path"]
        if name == "":
            return {"files": sorted(public_files)}
        if name not in public_files:
            raise ValueError("only declared public files are readable")
        target = public / name
        if target.is_symlink() or file_digest(target) != config["public_files"][name]:
            raise ValueError("public task file changed")
        return {"path": name, "content": target.read_text(encoding="utf-8")}
    if tool == "analog_history":
        return _candidate_history(directory)
    candidate = directory / "candidate.spi"
    if tool == "analog_write":
        return _save_candidate(directory, arguments["content"], task_id=task_id)
    if tool == "analog_restore":
        digest = arguments["candidate_sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("invalid candidate SHA-256")
        snapshot = directory / "candidates" / (digest + ".spi")
        if snapshot.is_symlink() or not snapshot.is_file() or file_digest(snapshot) != digest:
            raise ValueError("saved candidate missing or changed")
        return _save_candidate(directory, snapshot.read_bytes().decode("utf-8"), task_id=task_id)
    if not candidate.is_file() or candidate.is_symlink():
        return {"state": "candidate_missing"}
    validate_rlc_candidate(candidate.read_text(encoding="utf-8"), task_id=task_id)
    if tool == "analog_submit":
        frozen = directory / "frozen"
        frozen.mkdir(mode=0o700)
        snapshot = frozen / "circuit.spi"
        shutil.copyfile(candidate, snapshot)
        snapshot.chmod(0o600)
        digest = file_digest(snapshot)
        atomic_json(directory / "frozen.json", {"candidate_sha256": digest, "action_id": action_id})
        return {"state": "submitted", "candidate_sha256": digest, "final_score": "not_exposed"}
    calls = [
        json.loads(path.read_text()) for path in (directory / "actions").glob("*/request.json")
    ]
    if sum(call["tool"] == "analog_simulate" for call in calls) > config["max_simulations"]:
        return {"state": "simulation_budget_exhausted"}
    for name, digest in config["public_files"].items():
        target = public / name
        if target.is_symlink() or file_digest(target) != digest:
            raise ValueError("public task file changed")
    before = file_digest(candidate)
    result = run_public_rlc(
        Path(config["source_root"]),
        candidate,
        directory / "simulations" / action_id,
        task_id=task_id,
        podman=config["podman"],
        podman_root=Path(config["podman_root"]) if config["podman_root"] else None,
        podman_runroot=Path(config["podman_runroot"]) if config["podman_runroot"] else None,
        runtime_image=config["runtime_image"],
        offline_image_archive=(
            Path(config["offline_image_archive"]) if config["offline_image_archive"] else None
        ),
        podman_single_id=config["podman_single_id"],
        podman_no_cpu_limit=config["podman_no_cpu_limit"],
        timeout_s=config["timeout_s"],
    )
    if file_digest(candidate) != before:
        raise ValueError("simulation changed candidate")
    excerpt = result["diagnostic_excerpt"][-2000:]
    # Agent feedback contains circuit diagnostics, never absolute host paths.
    excerpt = re.sub(r"/[^\s:\"']+", "<path>", excerpt)
    return {
        "state": result["state"],
        "authority": "public_diagnostic",
        "task_correctness": "not_evaluated",
        "candidate_sha256": before,
        "measurements": result["measurements"],
        "missing_measurements": result["missing_measurements"],
        "diagnostic_excerpt": excerpt,
        **({"sweep": result["sweep"]} if "sweep" in result else {}),
    }


def _save_candidate(directory: Path, content: str, *, task_id: str) -> dict:
    validate_rlc_candidate(content, task_id=task_id)
    temporary = directory / "candidate.spi.tmp"
    with temporary.open("wb") as stream:
        stream.write(content.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    digest = file_digest(temporary)
    versions = directory / "candidates"
    versions.mkdir(mode=0o700, exist_ok=True)
    snapshot = versions / (digest + ".spi")
    if snapshot.exists() or snapshot.is_symlink():
        if snapshot.is_symlink() or not snapshot.is_file() or file_digest(snapshot) != digest:
            raise ValueError("saved candidate changed")
    else:
        shutil.copyfile(temporary, snapshot)
        snapshot.chmod(0o600)
    temporary.replace(directory / "candidate.spi")
    return {"candidate_sha256": digest}


def _candidate_history(directory: Path) -> dict:
    candidates = []
    for snapshot in sorted((directory / "candidates").glob("*.spi")):
        digest = snapshot.stem
        if (
            not re.fullmatch(r"[a-f0-9]{64}", digest)
            or snapshot.is_symlink()
            or not snapshot.is_file()
            or file_digest(snapshot) != digest
        ):
            raise ValueError("saved candidate changed")
        feedback = []
        for path in sorted((directory / "actions").glob("*/response.json")):
            request = json.loads(path.with_name("request.json").read_text(encoding="utf-8"))
            reply = json.loads(path.read_text(encoding="utf-8"))
            result = reply.get("result", {})
            if request["tool"] == "analog_simulate" and result.get("candidate_sha256") == digest:
                feedback.append({"action_id": request["id"], **result})
        candidates.append({"candidate_sha256": digest, "public_simulations": feedback})
    current = directory / "candidate.spi"
    return {
        "current_candidate_sha256": file_digest(current) if current.is_file() else None,
        "candidates": candidates,
        "authority": "public_diagnostic",
        "ordering": "sha256_not_quality",
    }


def close_session(directory: Path, reason: str) -> dict:
    """Operator-only collection after the Agent stops; never manufactures a tool call."""
    if reason not in END_REASONS:
        raise ValueError("unsupported episode termination reason")
    directory = Path(directory).absolute()
    with (
        (directory / ".enqueue.lock").open("a") as enqueue_lock,
        (directory / ".session.lock").open("a") as session_lock,
    ):
        try:
            fcntl.flock(enqueue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"state": "awaiting_action_recovery"}
        receipt_path = directory / "episode-end.json"
        if receipt_path.exists():
            return json.loads(receipt_path.read_text())
        for root in (directory / "actions", directory / "requests"):
            for action in root.glob("*"):
                reply = action_response(directory, action.name)
                if reply.get("state") in {"pending", "unknown_execution"} or reply.get("error") in {
                    "unknown_execution",
                    "unresolved_previous_action",
                }:
                    return {"state": "awaiting_action_recovery"}
        config = json.loads((directory / "session.json").read_text())
        task_id = session_task_id(config)
        frozen_path = directory / "frozen.json"
        if not frozen_path.exists() and config.get("collection_policy") != "episode_end":
            return {"state": "unsubmitted", "collection_policy": "explicit_submit"}
        intent_path = directory / "ending.json"
        if intent_path.exists():
            reason = json.loads(intent_path.read_text())["termination_reason"]
        else:
            atomic_json(intent_path, {"termination_reason": reason})
        receipt = {
            "state": "missing_candidate",
            "termination_reason": reason,
            "collection_source": "episode_end",
            "agent_submitted": False,
            "candidate_sha256": None,
        }
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text())
            source = frozen.get("collection_source", "agent_submit")
            receipt.update(
                state="submitted" if source == "agent_submit" else "collected",
                collection_source=source,
                agent_submitted=source == "agent_submit",
                candidate_sha256=frozen["candidate_sha256"],
            )
        else:
            candidate = directory / "candidate.spi"
            if candidate.is_symlink():
                raise ValueError("candidate must be a regular file")
            if candidate.is_file():
                validate_rlc_candidate(candidate.read_text(encoding="utf-8"), task_id=task_id)
                frozen_dir = directory / "frozen"
                frozen_dir.mkdir(mode=0o700, exist_ok=True)
                temporary = frozen_dir / "circuit.spi.tmp"
                shutil.copyfile(candidate, temporary)
                temporary.chmod(0o600)
                temporary.replace(frozen_dir / "circuit.spi")
                digest = file_digest(frozen_dir / "circuit.spi")
                atomic_json(
                    frozen_path,
                    {
                        "candidate_sha256": digest,
                        "collection_source": "episode_end",
                        "termination_reason": reason,
                    },
                )
                receipt.update(state="collected", candidate_sha256=digest)
        atomic_json(receipt_path, receipt)
        return receipt


def session_info(directory: Path) -> dict:
    """Public execution contract, without host paths or final evaluator details."""
    config = json.loads((Path(directory) / "session.json").read_text())
    return {
        "task_id": session_task_id(config),
        "collection_policy": config.get("collection_policy", "explicit_submit"),
        "max_actions": config["max_actions"],
        "max_simulations": config["max_simulations"],
    }


def finalize_session(
    directory: Path,
    output: Path,
    *,
    archive_root: Path | None = None,
    timeout_s: int = 900,
) -> dict:
    """Operator-only original scoring of an acknowledged, unchanged frozen submission."""
    directory = Path(directory).absolute()
    frozen_path = directory / "frozen.json"
    if not frozen_path.is_file():
        raise ValueError("final scoring requires an acknowledged submission")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("collection_source") == "episode_end":
        receipt = json.loads((directory / "episode-end.json").read_text())
        if (
            receipt.get("state") != "collected"
            or receipt.get("collection_source") != "episode_end"
            or receipt.get("agent_submitted") is not False
            or receipt.get("candidate_sha256") != frozen.get("candidate_sha256")
            or receipt.get("termination_reason") not in END_REASONS
            or receipt.get("termination_reason") != frozen.get("termination_reason")
        ):
            raise ValueError("final scoring requires an acknowledged collection")
    else:
        _verify_agent_submission(directory, frozen)
    candidate = directory / "frozen/circuit.spi"
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or file_digest(candidate) != frozen["candidate_sha256"]
    ):
        raise ValueError("frozen candidate changed")
    return _score_frozen(directory, candidate, frozen, output, archive_root, timeout_s)


def _verify_agent_submission(directory: Path, frozen: dict) -> None:
    action_id = frozen.get("action_id")
    if not isinstance(action_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", action_id):
        raise ValueError("invalid frozen action ID")
    action = directory / "actions" / action_id
    if not (action / "request.json").is_file() or not (action / "response.json").is_file():
        raise ValueError("final scoring requires an acknowledged submission")
    request = json.loads((action / "request.json").read_text(encoding="utf-8"))
    response = json.loads((action / "response.json").read_text(encoding="utf-8"))
    if (
        request.get("tool") != "analog_submit"
        or not response.get("ok")
        or response.get("result", {}).get("state") != "submitted"
        or response["result"].get("candidate_sha256") != frozen.get("candidate_sha256")
    ):
        raise ValueError("final scoring requires an acknowledged submission")


def _score_frozen(directory, candidate, frozen, output, archive_root, timeout_s):
    config = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    task_id = session_task_id(config)
    validate_rlc_candidate(candidate.read_text(encoding="utf-8"), task_id=task_id)
    output = Path(output).absolute()
    result = run_case(
        task_id,
        Path(config["source_root"]),
        candidate,
        output,
        backend="podman",
        podman=config.get("podman", "podman"),
        podman_root=Path(config["podman_root"]) if config.get("podman_root") else None,
        podman_runroot=Path(config["podman_runroot"]) if config.get("podman_runroot") else None,
        runtime_image=config.get("runtime_image"),
        offline_image_archive=(
            Path(config["offline_image_archive"]) if config.get("offline_image_archive") else None
        ),
        podman_single_id=config.get("podman_single_id", False),
        podman_no_cpu_limit=config.get("podman_no_cpu_limit", False),
        timeout_s=timeout_s,
        archive_root=archive_root,
    )
    if (
        file_digest(output / "inputs/circuit.spi") != frozen["candidate_sha256"]
        or file_digest(candidate) != frozen["candidate_sha256"]
    ):
        raise ValueError("final scorer candidate differs from frozen submission")
    return {**result, "frozen_candidate_sha256": frozen["candidate_sha256"]}
