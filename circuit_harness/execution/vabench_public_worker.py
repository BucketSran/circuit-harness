"""Pinned vaEVAS interpreter entry point for public-only Bubblewrap execution."""

import json
import shlex
import shutil
import sys
from pathlib import Path


def main():
    directory, action = map(Path, sys.argv[1:3])
    config = json.loads((directory / "session.json").read_text())
    source = Path(config["pin"]["source"])
    calibration = source / "benchmark-vabench-release-v4/operations/calibration_pilot"
    sys.path[:0] = [str(calibration), str(source)]
    from mini_swe_vabench import VaBenchBashEnvironment
    from public_validation import public_execution_contract
    from runners.agent_harness.tools.waveform_summary import summarize_waveform

    runtime = action / "sandbox"
    public = runtime / "public"
    public.mkdir(parents=True)
    shutil.copytree(directory / "public/task", public / "task")
    shutil.copytree(directory / "public/submission", public / "submission")

    class PublicEnvironment(VaBenchBashEnvironment):
        def _sandbox_argv(self, command):
            argv = super()._sandbox_argv(command)
            # Older lab Bubblewrap lacks --clearenv. Empty the environment before
            # exec instead; retain all upstream namespace/mount/setenv arguments.
            argv.remove("--clearenv")
            return ["/usr/bin/env", "-i", *argv]

    environment = PublicEnvironment(
        runtime,
        timeout_s=config["timeout_s"],
        sandbox_backend="bubblewrap",
        evas_command=str(Path(config["pin"]["python"]).parent / "evas"),
        submission_gate=lambda _: {"passed": False},
        candidate_artifacts=config["artifacts"],
    )
    try:
        environment.preflight()
        # Check actual namespace visibility, never mount the evaluator or the source.
        boundary = environment.execute(
            {
                "command": "test ! -e "
                + shlex.quote(str(source))
                + " && test ! -e "
                + shlex.quote(str(directory / "session.json"))
                + ' && ! env | /bin/grep -E "API_KEY|TOKEN|SECRET"'
            }
        )
        if boundary["returncode"] != 0:
            raise ValueError("public isolation preflight failed")
        if sys.argv[3:] == ["preflight"]:
            (action / "feedback.json").write_text(
                json.dumps(
                    {"state": "ready", "sandbox": "bubblewrap", "isolation_preflight": "passed"}
                )
            )
            return
        contract = json.loads((public / "task/evas_runtime.json").read_text())
        command, scope = public_execution_contract(contract)
        # The upstream non-Docker sandbox has a writable public .tmp, not /tmp.
        # Output semantics and input deck stay unchanged; record the resolved command.
        command = command.replace("/tmp/vabench-visible", "/workspace/public/.tmp/vabench-visible")
        command = "export EVAS_ENGINE=evas2 VABENCH_EVAS_PROFILE=r53; " + command
        result = environment.execute({"command": command})
        (action / "execution.json").write_text(json.dumps(result, indent=2))
        output = public / ".tmp/vabench-visible/evas-output"
        if scope == "reference_dut_only":
            output /= "reference"
        visible_output = str(result.get("output", ""))[-12000:]
        for private_path in (
            str(Path(config["pin"]["python"]).parent.parent),
            str(source),
            str(directory),
        ):
            visible_output = visible_output.replace(private_path, "[operator-runtime]")
        feedback = {
            "status": "succeeded" if result["returncode"] == 0 else "failed",
            "returncode": result["returncode"],
            "feedback_scope": scope,
            "output": visible_output,
            "waveform": summarize_waveform(output),
            "sandbox": "bubblewrap",
        }
        (action / "feedback.json").write_text(json.dumps(feedback, indent=2))
        (action / "public-profile.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "isolation_preflight": "passed",
                    "upstream_environment": environment.serialize(),
                },
                indent=2,
            )
        )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
