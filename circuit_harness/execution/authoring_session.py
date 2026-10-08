"""Server-owned public simulation budget and frozen testbench submission.

This is a cooperative operator protocol, not an authorization boundary against
another process running as the same account.
"""

import fcntl
import json
import re
from pathlib import Path

from .jobs import inspect_job, submit_spectre_gain
from .journal import atomic_json, digest
from .spectre_testbench import prepare_testbench, testbench_identity


def _read(path):
    return json.loads(Path(path).read_text())


def _candidate_digest(candidate):
    return digest(json.dumps(candidate, sort_keys=True, allow_nan=False).encode())


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", value):
        raise ValueError("action ID must contain 1-48 safe characters")


def create_session(directory, payload, profile_path, max_simulations=3):
    """Operator-only creation; the model cannot change conditions or budgets."""
    if type(max_simulations) is not int or not 1 <= max_simulations <= 3:
        raise ValueError("max_simulations must be 1-3")
    identity = testbench_identity(payload, profile_path)
    directory = Path(directory).absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    (directory / "actions").mkdir(mode=0o700)
    atomic_json(
        directory / "session.json",
        {
            "schema_version": 1,
            "identity": identity,
            "max_simulations": max_simulations,
            "job_prefix": "gain-" + digest(str(directory).encode())[:16],
        },
    )
    return {"status": "ready"}


def request_action(directory, request):
    """Reserve once before job submission; an uncertain action is never relaunched."""
    directory = Path(directory)
    if not isinstance(request, dict) or set(request) != {"id", "tool", "arguments"}:
        raise ValueError("invalid action envelope")
    _identifier(request["id"])
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = directory / "actions" / request["id"]
        if record.exists():
            if _read(record / "request.json") != request:
                raise ValueError("action ID belongs to a different request")
            return {"state": "accepted"}
        record.mkdir(mode=0o700)
        atomic_json(record / "request.json", request)

        def reject(reason):
            atomic_json(record / "response.json", {"ok": False, "error": reason})
            return {"state": "accepted"}

        config = _read(directory / "session.json")
        identity = config["identity"]
        task = identity["task"]
        if (directory / "submission.json").exists():
            return reject("submission_frozen")
        if request["tool"] not in {"gain_simulate", "gain_submit"}:
            return reject("unknown_tool")
        args = request["arguments"]
        if not isinstance(args, dict) or set(args) != {"candidate"}:
            return reject("invalid_arguments")
        candidate = args["candidate"]
        try:
            if testbench_identity(task, identity["profile_path"]) != identity:
                raise ValueError("session deployment changed")
            prepare_testbench(
                task["draft"], task["materials"], task["confirmation"], candidate, task["reference"]
            )
        except (ValueError, OSError) as error:
            return reject(str(error))
        simulations = sorted((directory / "actions").glob("*/simulation.json"))
        for simulation in simulations:
            if _response(directory, simulation.parent.name).get("ok") is None:
                return reject("recover_previous_simulation")
        if request["tool"] == "gain_submit":
            signature = _candidate_digest(candidate)
            validated = any(
                _read(p)["candidate_sha256"] == signature
                and _response(directory, p.parent.name).get("ok") is True
                for p in simulations
            )
            if not validated:
                return reject("candidate_requires_public_simulation")
            submission = {
                "candidate": candidate,
                "candidate_sha256": signature,
                "confirmation": task["confirmation"],
                "action_id": request["id"],
            }
            atomic_json(directory / "submission.json", submission)
            atomic_json(
                record / "response.json",
                {"ok": True, "result": {"status": "submitted", "candidate_sha256": signature}},
            )
            return {"state": "accepted"}
        if len(simulations) >= config["max_simulations"]:
            return reject("simulation_budget_exhausted")
        job_id = config["job_prefix"] + "-" + request["id"]
        job = str(Path(identity["profile"]["run_root"]) / job_id)
        atomic_json(
            record / "simulation.json",
            {"job": job, "candidate_sha256": _candidate_digest(candidate)},
        )
        # A failure after reservation remains visible and consumes a slot.
        try:
            submit_spectre_gain({**task, "candidate": candidate}, identity["profile_path"], job_id)
        except (ValueError, OSError) as error:
            return reject(f"submission_failed: {error}")
        return {"state": "accepted"}


def _response(directory, action_id):
    record = directory / "actions" / action_id
    if (record / "response.json").exists():
        return _read(record / "response.json")
    if not (record / "simulation.json").exists():
        return {"state": "unknown_execution"}
    state = inspect_job(Path(_read(record / "simulation.json")["job"]))
    if state["state"] == "running":
        return {"state": "pending"}
    if state["state"] != "finished":
        return {"state": "unknown_execution"}
    result = state["result"]
    if result["execution"] != "ok":
        reply = {"ok": False, "error": result["execution"], "reason": result["reason"]}
    else:
        metrics = result["metrics"]
        reply = {
            "ok": True,
            "result": {
                "measurement": {
                    key: metrics[key]
                    for key in ("gain_db", "phase_deg", "input_magnitude_v", "frequency_hz")
                },
                "public_testbench_valid": metrics["testbench_valid"],
                "dut_pass": metrics["dut_pass"],
            },
        }
    atomic_json(record / "response.json", reply)
    return reply


def action_response(directory, action_id):
    _identifier(action_id)
    directory = Path(directory)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _response(directory, action_id)


def tool_schemas():
    """Only bounded candidate data enters the server; approval stays operator-side."""
    node = {"type": "string", "enum": ["vip", "vin", "out", "vdd", "0"]}
    pair = {"type": "array", "items": node, "minItems": 2, "maxItems": 2}
    ac = {
        "type": "object",
        "additionalProperties": False,
        "required": ["magnitude", "phase_deg"],
        "properties": {
            "magnitude": {"type": "number", "minimum": 0, "maximum": 1},
            "phase_deg": {"type": "number", "minimum": -360, "maximum": 360},
        },
    }
    candidate = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ports", "positive_ac", "negative_ac", "measurement"],
        "properties": {
            "ports": {"type": "array", "items": node, "minItems": 5, "maxItems": 5},
            "positive_ac": ac,
            "negative_ac": ac,
            "measurement": {
                "type": "object",
                "additionalProperties": False,
                "required": ["numerator", "denominator"],
                "properties": {"numerator": pair, "denominator": pair},
            },
        },
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["candidate"],
                    "properties": {"candidate": candidate},
                },
            },
        }
        for name, description in (
            (
                "gain_simulate",
                "Run the candidate in Spectre with confirmed conditions. At most three "
                "simulations; inspect the measurement and public validation before revising.",
            ),
            (
                "gain_submit",
                "Freeze an explicitly simulated candidate. No edits or simulations after "
                "submission. Independent evaluation is performed later by the operator.",
            ),
        )
    ]
