"""Onboarding prepares stock Harbor jobs and refuses unpinned benchmark content."""

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from circuit_harness.execution.analog_design_bench import TASKS
from circuit_harness.harbor.analog_example import prepare


def test_prepare_rejects_changed_task_before_writing_job(tmp_path):
    source = tmp_path / "source"
    task = source / "tasks/rlc-rf-bandpass-100mhz"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("changed task")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="pin mismatch"):
        prepare("rlc-rf-bandpass-100mhz", source, output, tmp_path / "jobs", oracle=True)
    assert not output.exists()


@pytest.mark.parametrize(
    "task_id",
    [
        "rlc-rf-bandpass-100mhz",
        "sky130-ota-5t-gain40-pm60-noise50uv-pvt",
    ],
)
@pytest.mark.parametrize(
    "entrypoint",
    ["circuit_harness.harbor.analog_example", "circuit_harness.benchmarks.analogbench"],
)
def test_original_source_prepares_stock_oracle_job(tmp_path, task_id, entrypoint):
    roots = os.environ.get("ANALOG_EXAMPLE_SOURCES", "").split(os.pathsep)
    source = next(
        (Path(root) for root in roots if root and Path(root).name.endswith(TASKS[task_id].commit)),
        None,
    )
    if source is None:
        pytest.skip("set ANALOG_EXAMPLE_SOURCES to externally downloaded pinned source roots")
    output = tmp_path / "prepared"
    import_module(entrypoint).prepare(task_id, source, output, tmp_path / "jobs", oracle=True)
    job = json.loads((output / "job.json").read_text())
    assert job["agents"][0]["name"] == "oracle"
    assert job["environment"]["type"] == "docker"
    if not task_id.startswith("sky130-"):
        assert job["environment"]["import_path"] is None
    if task_id.startswith("sky130-"):
        assert job["environment"]["import_path"] == (
            "circuit_harness.harbor.analogbench:AnalogDockerEnvironment"
        )
    assert job["verifier"]["import_path"] == (
        "circuit_harness.harbor.analogbench:OriginalAnalogVerifier"
    )
    assert job["tasks"][0]["path"] == str(source.resolve() / "tasks" / task_id)


def test_original_source_compiles_agent_model_without_resolving_credentials(tmp_path, monkeypatch):
    roots = os.environ.get("ANALOG_EXAMPLE_SOURCES", "").split(os.pathsep)
    source = next(
        (
            Path(root)
            for root in roots
            if root and Path(root).name.endswith(TASKS["rlc-rf-bandpass-100mhz"].commit)
        ),
        None,
    )
    if source is None:
        pytest.skip("requires externally downloaded pinned source")
    monkeypatch.setenv("EXAMPLE_KEY", "never-serialize-this-key")
    catalog = {
        "agents": {"pi": {"name": "pi"}},
        "models": {
            "model": {
                "connections": [
                    {
                        "protocol": "openai-chat-completions",
                        "model": "test-model",
                        "base_url": "https://model.example.invalid/v1",
                        "key_env": "EXAMPLE_KEY",
                    }
                ]
            }
        },
    }
    output = tmp_path / "prepared"
    prepare(
        "rlc-rf-bandpass-100mhz",
        source,
        output,
        tmp_path / "jobs",
        catalog=catalog,
        agent="pi",
        model="model",
        cpus=2,
    )
    text = (output / "job.json").read_text()
    job = json.loads(text)
    assert job["agents"][0]["name"] == "pi"
    assert job["agents"][0]["env"]["OPENAI_API_KEY"] == "${EXAMPLE_KEY}"
    assert "never-serialize-this-key" not in text
    assert job["environment"]["override_cpus"] == 2
    assert "model.example.invalid" in job["agents"][0]["extra_allowed_hosts"]
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["task_sha256"] == TASKS["rlc-rf-bandpass-100mhz"].source_sha256
    assert provenance["root_license_at_pin"] == "absent"


def test_valid_netlist_without_checker_report_is_infrastructure_error(tmp_path):
    from circuit_harness.harbor.analog_example import validate_grading_evidence

    with pytest.raises(RuntimeError, match="checker did not produce"):
        validate_grading_evidence(
            {"reward": 0, "tests_total": 7, "tests_passed": 0}, 0, tmp_path / "new-ctrf.json"
        )


def test_original_starter_rejection_remains_valid_zero(tmp_path):
    from circuit_harness.harbor.analog_example import validate_grading_evidence

    validate_grading_evidence(
        {"reward": 0, "tests_total": 7, "tests_passed": 0}, 1, tmp_path / "new-ctrf.json"
    )


def test_missing_uploaded_checker_is_error_before_grading(tmp_path):
    import asyncio

    from harbor.environments.base import ExecResult
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths

    from circuit_harness.harbor.analog_example import OriginalAnalogVerifier

    roots = os.environ.get("ANALOG_EXAMPLE_SOURCES", "").split(os.pathsep)
    task_id = "sky130-ota-5t-gain40-pm60-noise50uv-pvt"
    source = next(
        (Path(root) for root in roots if root and Path(root).name.endswith(TASKS[task_id].commit)),
        None,
    )
    if source is None:
        pytest.skip("requires externally downloaded pinned source")

    class MissingCheckerEnvironment:
        # Protocol fixture: uploaded files are unavailable in the remote environment.
        async def set_network_policy(self, policy):
            pass

        async def upload_dir(self, source_dir, target_dir):
            pass

        async def exec(self, command, user):
            return ExecResult(return_code=1, stderr="verify.py missing")

    verifier = OriginalAnalogVerifier(
        Task(source / "tasks" / task_id),
        TrialPaths(tmp_path / "trial"),
        MissingCheckerEnvironment(),
    )
    with pytest.raises(RuntimeError, match="prerequisites unavailable"):
        asyncio.run(verifier.verify())


@pytest.mark.parametrize("plugin_module", ["analog_example", "analogbench"])
def test_rlc_checker_exit_trap_zero_is_error_at_real_verifier_boundary(tmp_path, plugin_module):
    import asyncio

    from harbor.environments.base import ExecResult
    from harbor.models.job.config import JobConfig
    from harbor.models.task.config import TaskOS
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths
    from harbor.verifier.factory import VerifierFactory

    task_id = "rlc-rf-bandpass-100mhz"
    roots = os.environ.get("ANALOG_EXAMPLE_SOURCES", "").split(os.pathsep)
    source = next(
        (Path(root) for root in roots if root and Path(root).name.endswith(TASKS[task_id].commit)),
        None,
    )
    if source is None:
        pytest.skip("requires externally downloaded pinned source")
    paths = TrialPaths(tmp_path / "trial")
    paths.verifier_dir.mkdir(parents=True)

    class CrashedCheckerEnvironment:
        # Simulates the separate verifier container, retaining real Harbor Verifier.
        os = TaskOS.LINUX
        capabilities = type("Capabilities", (), {"mounted": True})()

        async def exec(self, command, **kwargs):
            if "test.sh" in command and not command.startswith("chmod"):
                paths.reward_json_path.write_text(
                    json.dumps(
                        {
                            "reward": 0,
                            "tests_total": 15,
                            "tests_passed": 0,
                            "partial": 0.0,
                        }
                    )
                )
                return ExecResult(return_code=2, stderr="checker process crashed")
            return ExecResult(return_code=0)

        async def upload_dir(self, **kwargs):
            raise AssertionError("A uses its prebuilt separate verifier tests")

        async def set_network_policy(self, policy):
            raise AssertionError("A upstream verifier network must remain unchanged")

    config_dir = tmp_path / "prepared"
    prepare(task_id, source, config_dir, tmp_path / "jobs", oracle=True)
    config = JobConfig.model_validate_json((config_dir / "job.json").read_text())
    # Historical saved jobs retain their plugin names when reopened.
    config.verifier.import_path = f"circuit_harness.harbor.{plugin_module}:OriginalAnalogVerifier"
    verifier = VerifierFactory.create_verifier_from_config(
        config.verifier,
        task=Task(source / "tasks" / task_id),
        trial_paths=paths,
        environment=CrashedCheckerEnvironment(),
        skip_tests_upload=True,
    )
    with pytest.raises(RuntimeError, match="checker did not produce"):
        asyncio.run(verifier.verify())


@pytest.mark.parametrize(
    "entrypoint",
    ["circuit_harness.harbor.analog_example", "circuit_harness.benchmarks.analogbench"],
)
def test_cli_rejects_changed_task_before_creating_job(tmp_path, entrypoint):
    source = tmp_path / "source"
    task = source / "tasks/rlc-rf-bandpass-100mhz"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("changed task")
    output = tmp_path / "prepared"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            entrypoint,
            "--task",
            "rlc-rf-bandpass-100mhz",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--oracle",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 2
    assert "pin mismatch" in run.stderr
    assert not output.exists()


@pytest.mark.parametrize("plugin_module", ["analog_example", "analogbench"])
def test_saved_sky130_environment_loads_through_harbor_factory(tmp_path, plugin_module):
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
    from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from circuit_harness.harbor.analogbench import AnalogDockerEnvironment

    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM scratch\n")
    environment = EnvironmentFactory.create_environment_from_config(
        TrialEnvironmentConfig(
            type="docker",
            import_path=f"circuit_harness.harbor.{plugin_module}:AnalogDockerEnvironment",
        ),
        environment_dir=environment_dir,
        environment_name="saved-sky130",
        session_id="layout-migration",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        phase_network_policies=[NetworkPolicy(network_mode=NetworkMode.PUBLIC)],
    )
    assert isinstance(environment, AnalogDockerEnvironment)
    assert environment.capabilities.dynamic_network_policy
    assert environment.capabilities.disable_internet
