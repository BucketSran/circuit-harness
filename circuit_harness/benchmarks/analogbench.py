"""Prepare pinned Analog Design Bench tasks for stock Harbor trials.

Writes configuration only. Harbor owns Agent, Trial, Docker and grading.
Upstream sources remain in the operator's external cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from circuit_harness.execution.analog_design_bench import TASKS, tree_digest
from circuit_harness.harbor.profiles import compile_job

EXAMPLES = ("rlc-rf-bandpass-100mhz", "sky130-ota-5t-gain40-pm60-noise50uv-pvt")
REPOSITORY = "https://github.com/Arcadia-1/analog-design-bench"


def prepare(
    task_id: str,
    source_root: Path,
    output: Path,
    jobs_dir: Path,
    *,
    oracle: bool = False,
    catalog: dict | None = None,
    agent: str | None = None,
    model: str | None = None,
    protocol: str | None = None,
    cpus: int | None = None,
) -> Path:
    """Validate original task bytes and write a fresh private Harbor job directory.

    Oracle copies the upstream reference solution. It is an infrastructure control,
    never evidence that a model solved the circuit. Model mode compiles the selected
    stock Harbor agent using the existing independent Agent/Model catalog.
    """
    if task_id not in EXAMPLES:
        raise ValueError(f"unsupported onboarding task: {task_id}")
    if cpus is not None and (type(cpus) is not int or cpus < 1):
        raise ValueError("cpus must be a positive integer")
    if oracle:
        if any(value is not None for value in (catalog, agent, model, protocol)):
            raise ValueError("oracle cannot be combined with agent/model settings")
    elif catalog is None or agent is None or model is None:
        raise ValueError("select oracle or supply catalog, agent and model")
    task = TASKS[task_id]
    root = Path(source_root).resolve()
    source = root / "tasks" / task_id
    actual_digest = tree_digest(source)
    if actual_digest != task.source_sha256:
        raise ValueError(f"task source pin mismatch for {task_id}: {actual_digest}")
    output = Path(output).absolute()
    job = {
        "job_name": output.name,
        "jobs_dir": str(Path(jobs_dir).absolute()),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "retry": {"max_retries": 0},
        "tasks": [{"path": str(source)}],
        "environment": {"type": "docker", "override_cpus": cpus},
    }
    job["verifier"] = {
        "import_path": "circuit_harness.harbor.analogbench:OriginalAnalogVerifier",
        "kwargs": {"skip_tests_upload": True} if task_id == "rlc-rf-bandpass-100mhz" else {},
    }
    if task_id.startswith("sky130-"):
        job["environment"]["import_path"] = (
            "circuit_harness.harbor.analogbench:AnalogDockerEnvironment"
        )
    if oracle:
        from harbor.models.job.config import JobConfig

        prepared = JobConfig.model_validate({**job, "agents": [{"name": "oracle"}]})
    else:
        prepared = compile_job(catalog, agent, model, job, protocol)
        # Preserve the upstream no-network verifier. Only the agent environment
        # gains access to the explicitly selected provider endpoint.
        endpoint = prepared.agents[0].env.get("OPENAI_BASE_URL") or prepared.agents[0].env.get(
            "ANTHROPIC_BASE_URL"
        )
        prepared.agents[0].extra_allowed_hosts = [urlsplit(endpoint).hostname]
    provenance = {
        "repository": REPOSITORY,
        "source_commit": task.commit,
        "task_id": task_id,
        "task_sha256": actual_digest,
        "base_image": task.image,
        "root_license_at_pin": "present" if (root / "LICENSE").is_file() else "absent",
        "license_notice": (
            "Consult upstream licensing before reuse. Current upstream LICENSE separates "
            "Apache-2.0 software and CC BY-NC 4.0 benchmark content; historical pins may "
            "have no root LICENSE. This preparation does not grant or resolve rights."
        ),
        "mode": "reference_solution_control" if oracle else "stock_harbor_agent",
        "runtime_overrides": {
            "cpus": cpus,
            "timeout": None,
            "image": None,
            "agent_extra_allowed_hosts": prepared.agents[0].extra_allowed_hosts,
            "agent_network_at_source": "public" if task_id.startswith("sky130-") else "no-network",
            "verifier_phase": "no-network",
            "verifier_layout_adapter": task_id.startswith("sky130-"),
            "verifier_evidence_guard": True,
        },
    }
    # All validation precedes writes. An existing directory is never overwritten.
    output.mkdir(mode=0o700, parents=True)
    for name, value in (
        ("job.json", prepared.model_dump(mode="json", context={"redact_sensitive_env": False})),
        ("provenance.json", provenance),
    ):
        destination = output / name
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        destination.chmod(0o600)
    return output / "job.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=EXAMPLES, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--agent")
    parser.add_argument("--model")
    parser.add_argument(
        "--protocol", choices=("openai-responses", "openai-chat-completions", "anthropic-messages")
    )
    parser.add_argument("--cpus", type=int)
    args = parser.parse_args(argv)
    try:
        catalog = json.loads(args.catalog.read_text()) if args.catalog else None
        destination = prepare(
            args.task,
            args.source_root,
            args.output,
            args.jobs_dir,
            oracle=args.oracle,
            catalog=catalog,
            agent=args.agent,
            model=args.model,
            protocol=args.protocol,
            cpus=args.cpus,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
