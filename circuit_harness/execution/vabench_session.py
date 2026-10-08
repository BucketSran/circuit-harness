"""Bounded public VABench tools. Final scoring remains an operator-only job.

The coordinator is trusted and owns the session exclusively. Candidate code is
executed only by the upstream OS sandbox, never by a model-selected host command.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
from importlib.resources import files
from pathlib import Path, PurePosixPath

from .journal import atomic_json, file_digest
from .session_budget import budget_rejection, with_budget
from .vabench import candidate_files, export_vabench, runtime_env, verify_pin

TOOLS = {
    "vabench_read": (
        "Read a task/... or submission/... file; empty path lists public files.",
        {"path": {"type": "string"}},
    ),
    "vabench_write": (
        "Write one declared candidate artifact, relative to submission.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
    ),
    "vabench_simulate": (
        "Run the fixed public EVAS simulation. Diagnostic feedback is not a final score.",
        {},
    ),
    "vabench_submit": ("Freeze the candidate and end editing. No final score is returned.", {}),
}


class InvalidPublicPath(ValueError):
    """A read path omitted its public task/submission prefix."""


def tool_schemas():
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


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("expected a public relative path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("expected a public relative path")
    return path


def _public_file(root, value):
    relative = _relative(value)
    if relative.parts[0] not in ("task", "submission"):
        raise InvalidPublicPath("only task and submission files are readable")
    target = root / relative
    if any(p.is_symlink() for p in (target, *target.parents) if p != root.parent):
        raise ValueError("links are forbidden")
    return target


def create_session(pin, directory, *, max_actions=24, max_simulations=4, timeout_s=120):
    if not 1 <= max_actions <= 100 or not 1 <= max_simulations <= 10 or not 1 <= timeout_s <= 300:
        raise ValueError("invalid session limits")
    directory = Path(directory).absolute()
    directory.mkdir(mode=0o700)
    export_vabench(pin, directory / "public")
    contract = json.loads((directory / "public/task/public_contract.json").read_text())
    artifacts = contract["target_artifacts"]
    if not artifacts or any(_relative(p).as_posix() != p for p in artifacts):
        raise ValueError("invalid candidate artifact contract")
    config = {
        "schema_version": 1,
        "budget_feedback": True,
        "task_id": pin["task_id"],
        "pin": pin,
        "artifacts": artifacts,
        "max_actions": max_actions,
        "max_simulations": max_simulations,
        "timeout_s": timeout_s,
        "public_files": {
            p.relative_to(directory / "public/task").as_posix(): file_digest(p)
            for p in (directory / "public/task").rglob("*")
            if p.is_file()
        },
    }
    atomic_json(directory / "session.json", config)
    return {"task_id": pin["task_id"], "artifacts": artifacts, "state": "ready"}


def _tree(root):
    return candidate_files(root) if any(root.iterdir()) else {}


def session_action(directory, request):
    """Persist an action before execution; the same ID never repeats a side effect."""
    directory = Path(directory)
    if set(request) != {"id", "tool", "arguments"} or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,80}", request["id"]
    ):
        raise ValueError("invalid action envelope")
    tool, arguments = request["tool"], request["arguments"]
    if (
        tool not in TOOLS
        or not isinstance(arguments, dict)
        or set(arguments) != set(TOOLS[tool][1])
    ):
        raise ValueError("invalid tool or arguments")
    if any(not isinstance(value, str) for value in arguments.values()):
        raise ValueError("arguments must be strings")
    with (directory / ".session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        actions = directory / "actions"
        actions.mkdir(mode=0o700, exist_ok=True)
        action = actions / request["id"]
        if action.exists():
            if json.loads((action / "request.json").read_text()) != request:
                raise ValueError("action ID belongs to another request")
            if (action / "response.json").exists():
                return json.loads((action / "response.json").read_text())
            return {"ok": False, "error": "unknown_execution", "retry_safe": False}
        config = json.loads((directory / "session.json").read_text())
        if len(list(actions.iterdir())) >= config["max_actions"]:
            return with_budget({"ok": False, "error": "action_budget_exhausted"}, directory, config)
        if (directory / "frozen.json").exists():
            return {"ok": False, "error": "submission_frozen"}
        if any(not (a / "response.json").exists() for a in actions.iterdir()):
            return {"ok": False, "error": "unresolved_previous_action", "retry_safe": False}
        rejected = budget_rejection(directory, config, tool)
        if rejected is not None:
            return rejected
        action.mkdir(mode=0o700)
        atomic_json(action / "request.json", request)
        started = time.monotonic()
        try:
            result = _execute(directory, config, tool, arguments, action)
            reply = {"ok": True, "result": result}
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            # Exception text may contain operator paths. Keep the full type only.
            reply = {
                "ok": False,
                "error": "ValueError"
                if isinstance(error, InvalidPublicPath)
                else type(error).__name__,
            }
            if isinstance(error, InvalidPublicPath):
                reply["hint"] = "Read task/... or submission/...; use an empty path to list files."
        reply = with_budget(reply, directory, config)
        atomic_json(action / "measurement.json", {"elapsed_s": time.monotonic() - started})
        atomic_json(action / "response.json", reply)
        return reply


def _execute(directory, config, tool, arguments, action):
    public = directory / "public"
    submission = public / "submission"
    if tool == "vabench_read":
        if arguments["path"] == "":
            return {
                "files": sorted(
                    p.relative_to(public).as_posix()
                    for prefix in ("task", "submission")
                    for p in (public / prefix).rglob("*")
                    if p.is_file() and not p.is_symlink()
                )
            }
        target = _public_file(public, arguments["path"])
        if not target.is_file() or target.stat().st_size > 100_000:
            raise ValueError("file absent or over read limit")
        return {"path": arguments["path"], "content": target.read_text()}
    if tool == "vabench_write":
        name = arguments["path"]
        if name not in config["artifacts"] or len(arguments["content"].encode()) > 100_000:
            raise ValueError("undeclared or oversized candidate")
        target = _public_file(public, "submission/" + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(arguments["content"])
        return {"path": name, "sha256": file_digest(target)}
    candidate = _tree(submission)
    if set(candidate) != set(config["artifacts"]):
        return {"status": "candidate_incomplete", "required": config["artifacts"]}
    if tool == "vabench_submit":
        from .vabench import freeze_candidate

        freeze_candidate(submission, directory / "candidate", {"candidate": candidate})
        atomic_json(directory / "frozen.json", {"candidate": candidate})
        return {"status": "submitted", "candidate": candidate, "final_score": "not_exposed"}
    calls = [json.loads(p.read_text()) for p in (directory / "actions").glob("*/request.json")]
    if sum(p["tool"] == "vabench_simulate" for p in calls) > config["max_simulations"]:
        return {"status": "simulation_budget_exhausted"}
    verify_pin(config["pin"])
    for name, digest in config["public_files"].items():
        if file_digest(public / "task" / name) != digest:
            raise ValueError("public task changed")
    _run_public_worker(directory, config, action)
    if _tree(submission) != candidate:
        raise ValueError("simulation modified candidate")
    verify_pin(config["pin"])
    feedback = json.loads((action / "feedback.json").read_text())
    return {
        **feedback,
        "candidate": candidate,
        "authority": "public_diagnostic",
        "task_correctness": "not_evaluated",
    }


def _run_public_worker(directory, config, action, *, preflight=False):
    worker = action / "public_worker.py"
    worker.write_bytes(files(__package__).joinpath("vabench_public_worker.py").read_bytes())
    python = config["pin"]["python"]
    (action / "scratch").mkdir()
    with (action / "worker.log").open("w") as log:
        process = subprocess.run(
            [python, "-I", "-B", str(worker), str(directory), str(action)]
            + (["preflight"] if preflight else []),
            env=runtime_env(python, action),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=config["timeout_s"] + 30,
            check=False,
        )
    if process.returncode != 0:
        raise ValueError("public simulation worker failed")


def preflight_session(directory):
    import uuid

    directory = Path(directory)
    if (directory / "frozen.json").exists():
        raise ValueError("session is already frozen; create a new attempt")
    config = json.loads((directory / "session.json").read_text())
    verify_pin(config["pin"])
    action = directory / "preflights" / uuid.uuid4().hex
    action.mkdir(mode=0o700, parents=True)
    with (directory / ".session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _run_public_worker(directory, config, action, preflight=True)
    reply = json.loads((action / "feedback.json").read_text())
    reply["task_id"] = config["task_id"]
    reply["session_sha256"] = file_digest(directory / "session.json")
    reply["limits"] = {
        name: config[name] for name in ("max_actions", "max_simulations", "timeout_s")
    }
    return reply


def enqueue_action(directory, request):
    """Detach before acknowledging; reconnect with the same request ID."""
    import sys

    from .bundle import build_cli

    directory = Path(directory).absolute()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(request.get("id", ""))):
        raise ValueError("invalid action ID")
    if len(json.dumps(request).encode()) > 150_000:
        raise ValueError("request too large")
    with (directory / ".enqueue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        spool = directory / "requests"
        spool.mkdir(mode=0o700, exist_ok=True)
        target = spool / request["id"]
        if target.exists():
            if json.loads((target / "request.json").read_text()) != request:
                raise ValueError("action ID belongs to another request")
            return {"state": "accepted", "id": request["id"]}
        target.mkdir(mode=0o700)
        atomic_json(target / "request.json", request)
        worker_sha256 = build_cli(target / "worker.pyz")
        with (target / "worker.log").open("wb") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(target / "worker.pyz"),
                    "vabench-action",
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
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        atomic_json(target / "started.json", {"pid": child.pid, "worker_sha256": worker_sha256})
        return {"state": "accepted", "id": request["id"]}


def action_response(directory, action_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", action_id):
        raise ValueError("invalid action ID")
    response = Path(directory) / "actions" / action_id / "response.json"
    if response.exists():
        return json.loads(response.read_text())
    response = Path(directory) / "requests" / action_id / "response.json"
    return json.loads(response.read_text()) if response.exists() else {"state": "pending"}


def finalize_session(directory, root, job_id, archive_root=None):
    """Operator boundary: freeze must precede a separately archived final replay."""
    from .jobs import submit_vabench

    directory = Path(directory)
    with (directory / ".session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frozen = json.loads((directory / "frozen.json").read_text())
        if candidate_files(directory / "candidate") != frozen["candidate"]:
            raise ValueError("frozen submission changed")
        config = json.loads((directory / "session.json").read_text())
        if archive_root is not None:
            archive_episode(directory, Path(archive_root), job_id)
        return submit_vabench(
            config["pin"], directory / "candidate", root, job_id, archive_root=archive_root
        )


def archive_episode(directory, archive_root, job_id):
    """Keep the frozen public trace beside final-job archives, before scratch can expire."""
    import shutil
    import tarfile

    from .archive import durable_json, private_root, sync_directory

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job_id):
        raise ValueError("invalid archive ID")
    private_root(archive_root)
    parent = archive_root / "episodes"
    private_root(parent)
    target = parent / job_id
    private_root(target)
    paths = [directory / "session.json", directory / "frozen.json"]
    for prefix in ("public/task", "candidate", "actions", "preflights"):
        paths.extend(
            p for p in (directory / prefix).rglob("*") if p.is_file() and not p.is_symlink()
        )
    for action in (directory / "actions").iterdir():
        for name in ("request.json", "started.json", "worker.pyz"):
            path = directory / "requests" / action.name / name
            if path.is_file() and not path.is_symlink():
                paths.append(path)
    members = {
        p.relative_to(directory).as_posix(): {"sha256": file_digest(p), "bytes": p.stat().st_size}
        for p in paths
    }
    if len(members) > 5000 or sum(p["bytes"] for p in members.values()) > 256 * 1024 * 1024:
        raise ValueError("episode evidence exceeds archive limit")
    receipt_path = target / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["members"] != members
            or file_digest(target / "episode.tar.gz") != receipt["sha256"]
        ):
            raise ValueError("episode archive changed")
        return receipt
    packed = directory / "episode.tar.gz.partial"
    with tarfile.open(packed, "w:gz") as tar:
        for name in sorted(members):
            tar.add(directory / name, arcname=name, recursive=False)
    partial = target / "episode.tar.gz.partial"
    with packed.open("rb") as source, partial.open("wb") as output:
        shutil.copyfileobj(source, output)
        output.flush()
        os.fsync(output.fileno())
    digest = file_digest(packed)
    if file_digest(partial) != digest:
        raise ValueError("episode archive copy changed")
    os.replace(partial, target / "episode.tar.gz")
    sync_directory(target)
    receipt = {
        "schema_version": 1,
        "members": members,
        "sha256": digest,
        "bytes": packed.stat().st_size,
        "candidate": json.loads((directory / "frozen.json").read_text())["candidate"],
    }
    durable_json(receipt_path, receipt)
    packed.unlink()
    return receipt


def verify_episode_archive(record):
    """Verify a downloaded public trace without extracting or running code."""
    import hashlib
    import tarfile

    record = Path(record)
    receipt = json.loads((record / "receipt.json").read_text())
    packed = record / "episode.tar.gz"
    if packed.stat().st_size != receipt["bytes"] or file_digest(packed) != receipt["sha256"]:
        raise ValueError("episode archive package changed")
    seen = set()
    with tarfile.open(packed, "r:gz") as tar:
        for member in tar:
            name = member.name
            if (
                not member.isfile()
                or name in seen
                or name not in receipt["members"]
                or _relative(name).as_posix() != name
            ):
                raise ValueError("unsafe episode archive member")
            expected = receipt["members"][name]
            if member.size != expected["bytes"]:
                raise ValueError("episode archive member size changed")
            with tar.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected["sha256"]:
                raise ValueError("episode archive member changed")
            seen.add(name)
    if seen != receipt["members"].keys():
        raise ValueError("episode archive members missing")
    return receipt
