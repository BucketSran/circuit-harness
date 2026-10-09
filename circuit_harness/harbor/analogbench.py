"""Harbor adapters for original pinned Analog Design Bench grading."""

from __future__ import annotations

import json
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.verifier.verifier import Verifier

from circuit_harness.benchmarks.analogbench import EXAMPLES
from circuit_harness.execution.analog_design_bench import TASKS, tree_digest


class AnalogDockerEnvironment(DockerEnvironment):
    """Enable Harbor's phase-scoped network control for the historical shared verifier."""

    def __init__(self, *args, phase_network_policies=(), **kwargs):
        super().__init__(
            *args,
            phase_network_policies=[
                *phase_network_policies,
                NetworkPolicy(network_mode=NetworkMode.NO_NETWORK),
            ],
            **kwargs,
        )


class OriginalAnalogVerifier(Verifier):
    """Check original grading evidence and supply historical SKY130 checker paths.

    The original test.sh and checker bytes remain unchanged. Tests are uploaded
    only after the agent has finished, using Harbor's verifier lifecycle.
    """

    async def verify(self):
        task_dir = self.task.paths.task_dir
        task_id = task_dir.name
        if task_id not in EXAMPLES:
            raise ValueError("original checker guard applies only to pinned onboarding tasks")
        if tree_digest(task_dir) != TASKS[task_id].source_sha256:
            raise ValueError("task source pin mismatch before verification")
        if task_id == "rlc-rf-bandpass-100mhz":
            # Harbor has already built the upstream independent verifier image.
            # Keep its tests and no-network policy; never upload them to the agent.
            probe = await self.environment.exec(
                command="command -v ngspice && test -s /app/analog_arena_tests/verify.py",
                user="root",
            )
            if probe.return_code != 0:
                raise RuntimeError("upstream RLC verifier prerequisites unavailable")
            candidate = await self.environment.exec(command="test -s /app/circuit.spi", user="root")
            if candidate.return_code not in (0, 1):
                raise RuntimeError("upstream RLC candidate check failed to execute")
            result = await super().verify()
            validate_grading_evidence(
                result.rewards,
                candidate.return_code,
                self.trial_paths.verifier_dir / "new-ctrf.json",
                expected_total=15,
            )
            return result
        await self.environment.set_network_policy(
            NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
        )
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.verifier_dir / "network-policy.json").write_text(
            json.dumps({"phase": "verifier", "applied_network_mode": "no-network"}) + "\n"
        )
        self.task.paths.validate_input_tree(self.task.paths.tests_dir)
        await self.environment.upload_dir(
            source_dir=self.task.paths.tests_dir, target_dir="/app/analog_arena_tests"
        )
        probe = await self.environment.exec(
            command=(
                "command -v ngspice && test -x /opt/analog-arena/check_circuit.py "
                "&& test -s /opt/sky130/continuous/sky130.lib.spice "
                "&& test -s /app/analog_arena_tests/verify.py"
            ),
            user="root",
        )
        if probe.return_code != 0:
            raise RuntimeError("upstream SKY130 verifier prerequisites unavailable")
        legality = await self.environment.exec(
            command="/opt/analog-arena/check_circuit.py /app/circuit.spi", user="root"
        )
        if legality.return_code not in (0, 1):
            raise RuntimeError("upstream circuit legality checker failed to execute")
        result = await super().verify()
        validate_grading_evidence(
            result.rewards, legality.return_code, self.trial_paths.verifier_dir / "new-ctrf.json"
        )
        return result


def validate_grading_evidence(
    rewards: dict,
    legality_exit: int,
    checker_report: Path,
    *,
    expected_total: int = 7,
) -> None:
    """Reject the historical shell trap's fallback zero after checker execution failure.

    The original script can reject an absent RLC candidate or invalid SKY130 netlist
    before simulation. All other cases must produce its complete CTRF report.
    """
    if (
        rewards.get("tests_total") != expected_total
        or not 0 <= rewards.get("tests_passed", -1) <= expected_total
    ):
        raise RuntimeError("upstream verifier reward has invalid test counts")
    if legality_exit == 1:
        if rewards.get("tests_passed") != 0 or rewards.get("reward") != 0:
            raise RuntimeError("rejected netlist has inconsistent upstream reward")
        return
    if legality_exit != 0:
        raise RuntimeError("upstream circuit legality checker failed to execute")
    try:
        summary = json.loads(checker_report.read_text())["results"]["summary"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("upstream checker did not produce a valid report") from error
    if summary.get("tests") != expected_total or summary.get("passed") != rewards["tests_passed"]:
        raise RuntimeError("upstream checker report and reward disagree")
