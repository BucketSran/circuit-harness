"""Executed with the pinned external Python. Never imported into the agent runtime."""

import json
import os
import shlex
import shutil
import sys
from pathlib import Path


def main():
    action = sys.argv[1]
    identity_path, candidate = map(Path, sys.argv[2:])
    identity = json.loads(identity_path.read_text())
    pin = identity["pin"]
    source = Path(pin["source"])
    package = source / "benchmark-vabench-release-v4"
    release = package / "release/benchmarkv4-r53"
    calibration = package / "operations/calibration_pilot"
    sys.path[:0] = [
        str(calibration),
        str(package / "operations/tri_form_derivation_prep"),
        str(source),
    ]
    import export_tri_form_runtime as exporter

    record, task = exporter.task_record(release, pin["task_id"])
    if action == "export":
        exporter.install_public(task, candidate, record["form"], "G2")
        (candidate / "submission").mkdir(exist_ok=True)
        shutil.copy2(task / "public_contract.json", candidate / "task/public_contract.json")
        return
    if action != "replay":
        raise ValueError("unsupported VABench worker action")
    import final_replay
    import result_protocol
    import submission_contract
    from runners.agent_harness import EpisodeContext

    runtime = identity_path.parent / "runtime"
    # These are operator-owned evaluation files, never an agent working directory.
    exporter.install_public(task, runtime / "public", record["form"], "G2")
    exporter.install_evaluator(task, runtime / "evaluator", record)
    shutil.copy2(task / "public_contract.json", runtime / "public/task/public_contract.json")
    submission = runtime / "public/submission"
    if submission.exists():
        shutil.rmtree(submission)
    shutil.copytree(candidate, submission)
    gate = submission_contract.submission_artifact_gate(runtime)
    frozen = result_protocol.snapshot_submission(runtime, gate)
    evidence = runtime / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "artifact_gate.json").write_text(json.dumps(gate, indent=2))
    (evidence / "frozen_submission.json").write_text(json.dumps(frozen, indent=2))
    if not gate["passed"]:
        replay = {"status": "no_submission", "artifact_gate": gate}
    else:
        os.environ["VABENCH_RELEASE_DIR"] = str(release)
        evas = str(Path(pin["python"]).parent / "evas")
        command = shlex.join([pin["python"], "-B", str(calibration / "trusted_replay_adapter.py")])
        profile = final_replay.build_final_test_profile(
            runtime=runtime,
            release=release,
            campaign_config_sha256=result_protocol.canonical_sha256(identity),
            command=command,
            timeout_s=identity["timeout_s"],
            evas_command=evas,
        )
        context = EpisodeContext(
            episode_id=identity_path.parent.parent.name,
            attempt_id="frozen-replay-1",
            task_id=pin["task_id"],
            condition="chips-fixture-replay",
            max_steps=1,
        )
        replay = final_replay.run_trusted_replay(
            runtime,
            command,
            identity["timeout_s"],
            evas,
            frozen,
            final_test_profile=profile,
            episode_context=context,
        )
    (identity_path.parent / "replay.json").write_text(json.dumps(replay, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
