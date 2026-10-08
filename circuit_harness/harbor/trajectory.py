"""Export one saved Pi trial to ATIF without Agent, environment or simulator execution."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from harbor.models.trial.config import TrialConfig

from circuit_harness.data.io import commit_directory
from circuit_harness.execution.candidate_bundle import regular_file, verify_candidate
from circuit_harness.execution.journal import file_digest

from .config import require_harbor_version
from .evidence import checked_evaluation, read_json
from .pi_trajectory import read_pi_trajectory


def export_pi_trial(trial_dir, output, *, redact=()):
    """Write a new private trajectory directory; raw inputs remain untouched.

    Scores require the same independent archive checks used by offline reports.
    Only score identities enter metadata; verifier contents never enter messages.
    """
    require_harbor_version()
    root = Path(trial_dir).absolute()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if root.is_symlink() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("output must be outside the source trial")
    sources = {}

    def read(name):
        path = regular_file(root, name)
        sources[name] = file_digest(path)
        return read_json(path)

    config = read("config.json")
    result = read("result.json")
    if (
        TrialConfig.model_validate(result.get("config")) != TrialConfig.model_validate(config)
        or result.get("trial_name") != root.name
    ):
        raise ValueError("trial result/config identity mismatch")
    if not result.get("finished_at") or result.get("exception_info"):
        raise ValueError("only completed Agent trials are supported")
    info = result.get("agent_info") or {}
    if info.get("name") != "pi" or not info.get("version"):
        raise ValueError("expected a versioned Pi trial")
    native_files = sorted((root / "agent/pi/sessions").glob("*.jsonl"))
    if len(native_files) != 1:
        raise ValueError("expected exactly one native Pi session")
    native = regular_file(root, str(native_files[0].relative_to(root)))
    sources[str(native.relative_to(root))] = file_digest(native)
    kwargs = config["agent"].get("kwargs", {})
    options = kwargs.get("agent_kwargs", kwargs)
    trajectory = read_pi_trajectory(
        native,
        agent_version=info["version"],
        requested_thinking=options.get("thinking"),
        redact=redact,
    )
    model = config["agent"].get("model_name")
    if not model or model.split("/", 1)[-1] != trajectory.agent.model_name:
        raise ValueError("native model differs from trial configuration")
    observed_model = (info.get("model_info") or {}).get("name")
    if observed_model and observed_model != trajectory.agent.model_name:
        raise ValueError("native model differs from trial result")
    session = read("public-session/session.json")
    frozen = verify_candidate(root / "public-session/candidate")
    read("public-session/candidate/manifest.json")
    collected = read("public-session/frozen.json")
    if collected.get("candidate_sha256") != frozen["candidate_sha256"] or any(
        session["task"].get(key) != frozen[key] for key in ("task_id", "task_version")
    ):
        raise ValueError("frozen candidate/session identity mismatch")
    circuit = {key: frozen[key] for key in ("task_id", "task_version", "candidate_sha256")}
    circuit.update(
        trial_id=result["id"],
        task_checksum=result["task_checksum"],
        grade_status="unscored",
        reward=None,
        actions=[],
    )
    evaluation = root / "verifier/evaluation.json"
    has_receipt = (
        bool(list((root / "verifier/transport").glob("*/archive/receipt.json")))
        or (root / "verifier/replay/receipt.json").exists()
    )
    native_reward = (result.get("verifier_result") or {}).get("rewards")
    if evaluation.exists() or has_receipt or native_reward is not None:
        final, digests, _ = checked_evaluation(root, frozen, circuit)
        sources.update(digests)
        if evaluation.exists():
            read("verifier/evaluation.json")
        if final.get("execution") != "ok" or final.get("score") is None:
            if native_reward is not None:
                raise ValueError("native reward without successful independent evaluation")
        else:
            score = final["score"]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0 <= score <= 1
            ):
                raise ValueError("invalid independent reward")
            if native_reward != {"reward": score}:
                raise ValueError("native reward differs from independent evaluation")
            circuit.update(
                grade_status="verified",
                reward=score,
                final_backend=final["backend"],
                criteria_sha256=final.get("criteria_sha256"),
            )
    actions = root / "public-session/actions"
    circuit["public_actions_available"] = actions.is_dir()
    for action in sorted(actions.iterdir()) if actions.is_dir() else []:
        if not action.is_dir():
            continue
        request_name = f"public-session/actions/{action.name}/request.json"
        response_name = f"public-session/actions/{action.name}/response.json"
        request, _ = read(request_name), read(response_name)
        if request.get("action_id") != action.name:
            raise ValueError("action identity mismatch")
        # These are textual references, not reconstructed model tool calls.
        pattern = r"""["']action_id["']\s*:\s*["']""" + re.escape(action.name) + r"""["']"""
        references = [
            call.tool_call_id
            for step in trajectory.steps
            for call in step.tool_calls or []
            if isinstance(call.arguments.get("command"), str)
            and re.search(pattern, call.arguments["command"])
        ]
        circuit["actions"].append(
            {
                "action_id": action.name,
                "tool": request["tool"],
                "native_call_references": references,
                "request_sha256": sources[request_name],
                "response_sha256": sources[response_name],
            }
        )
    trajectory.extra.update(circuit=circuit, sources=sources)
    if any(file_digest(regular_file(root, name)) != digest for name, digest in sources.items()):
        raise ValueError("source evidence changed during export")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".trajectory-", dir=output.parent))
    try:
        target = staged / "trajectory.json"
        target.write_text(
            trajectory.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
        )
        target.chmod(0o600)
        commit_directory(staged, output, replace=False)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return output / "trajectory.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--redact-file", type=Path, help="private JSON array of literal secrets to replace"
    )
    args = parser.parse_args()
    redact = json.loads(args.redact_file.read_text()) if args.redact_file else []
    if not isinstance(redact, list):
        raise ValueError("redaction file must contain an array")
    print(export_pi_trial(args.trial, args.output, redact=redact))


if __name__ == "__main__":
    main()
