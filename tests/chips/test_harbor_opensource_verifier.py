"""Harbor final replay with controlled local checkers, not EVAS acceptance."""

import json
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from test_harbor_chips_trial import ControlledEnvironment

from circuit_harness.harbor.config import FinalEvaluationConfig


def test_final_configuration_selects_replay_and_rejects_mixed_backends(tmp_path):
    from test_benchmark_replay import replay_inputs

    _, package, config = replay_inputs(tmp_path)
    declaration = {
        "backend": "benchmark_opensource",
        "task_package": str(package),
        "opensource": config,
    }
    selected = FinalEvaluationConfig.model_validate(declaration)
    assert selected.backend == "benchmark_opensource"
    with pytest.raises(ValueError):
        FinalEvaluationConfig.model_validate({**declaration, "remote": {}})
    with pytest.raises(ValueError):
        FinalEvaluationConfig.model_validate({**declaration, "unknown": True})
    with pytest.raises(ValueError):
        FinalEvaluationConfig.model_validate(
            {**declaration, "opensource": {**config, "timeout_s": "3"}}
        )
    assert FinalEvaluationConfig(task_package=package, remote={}).backend == "remote_spectre"


def verifier_for(tmp_path, package, config, candidate):
    from types import SimpleNamespace

    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths

    from circuit_harness.execution.candidate_bundle import verify_candidate
    from circuit_harness.harbor.verifier import FrozenCandidateVerifier

    taskdir = tmp_path / "harbor-task"
    (taskdir / "environment").mkdir(parents=True)
    (taskdir / "tests").mkdir()
    (taskdir / "tests/test.sh").write_text("exit 0\n")
    (taskdir / "instruction.md").write_text("Controlled candidate\n")
    (taskdir / "task.toml").write_text("[verifier]\ntimeout_sec=2\n")
    task = Task(taskdir)
    paths = TrialPaths(tmp_path / "trial")
    private = tmp_path / "private-final.json"
    private.write_text(
        json.dumps(
            {"backend": "benchmark_opensource", "task_package": str(package), "opensource": config}
        )
    )
    frozen = verify_candidate(candidate)
    environment = SimpleNamespace(
        frozen={
            "candidate_directory": str(candidate),
            "candidate_sha256": frozen["candidate_sha256"],
        },
        settings=SimpleNamespace(
            public_backend="docker", task={k: frozen[k] for k in ("task_id", "task_version")}
        ),
        context_id="fixture",
    )
    return FrozenCandidateVerifier(
        task=task, trial_paths=paths, environment=environment, config_path=private
    )


def test_harbor_verifier_reuses_sealed_replay_for_unsupported_task(tmp_path):
    import asyncio

    from test_benchmark_replay import replay_inputs

    from circuit_harness.execution.benchmark_replay import verify_replay

    candidate, package, config = replay_inputs(tmp_path)
    config["unsupported"] = ["controlled unavailable analysis"]
    verifier = verifier_for(tmp_path, package, config, candidate)
    with pytest.raises(RuntimeError, match="valid task score"):
        asyncio.run(verifier.verify())
    receipt = verify_replay(verifier.trial_paths.verifier_dir / "replay")
    assert receipt["result"]["execution"] == "unsupported_analysis"
    assert receipt["result"]["score"] is None


@pytest.fixture
def docker_image():
    import os

    image = os.environ.get("CHIPS_TEST_DOCKER_IMAGE")
    if not image:
        pytest.skip("set an already installed digest-pinned Python Docker image")
    return image


def docker_inputs(tmp_path, image, *, score=1, mode="graded"):
    import shlex

    from test_benchmark_replay import replay_inputs

    code = """
import hashlib,json,os,pathlib,subprocess,time
candidate=pathlib.Path(os.environ['CANDIDATE'])
out=pathlib.Path(os.environ['VERIFY_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
assert not pathlib.Path('/var/run/docker.sock').exists()
assert not any(k.endswith('API_KEY') for k in os.environ)
assert json.loads(os.environ['CHIPS_SOLVER_OPTIONS']) == {'controlled': True}
"""
    if mode == "slow":
        code += """
(out/'started').write_text('running')
child="import time,pathlib;time.sleep(2);pathlib.Path('survived').write_text('bad')"
subprocess.Popen(['python3','-c',child])
time.sleep(30)
"""
    elif mode == "broken":
        code += "raise RuntimeError('controlled checker crash')\n"
    elif mode == "malformed":
        code += "(out/'report.json').write_text('{broken')\n"
    else:
        code += (
            f"report={{'status':'completed','reward':{score},"
            f"'cases':[{{'status':'graded','passed':{bool(score)!r}}}]}}\n"
        )
        code += "report['candidate_sha256']=hashlib.sha256(candidate.read_bytes()).hexdigest()\n"
        if mode == "mismatched":
            code += "report['candidate_sha256']='0'*64\n"
        code += "report['solver_options']=json.loads(os.environ['CHIPS_SOLVER_OPTIONS'])\n"
        code += "(out/'report.json').write_text(json.dumps(report))\n"
    args = replay_inputs(tmp_path, checker=shlex.join(["python3", "-c", code]) + "\n")
    args[2]["image"] = image
    args[2]["solver_options"] = {"controlled": True}
    args[2]["timeout_s"] = 10
    return args


@pytest.mark.parametrize("score", [1, 0])
def test_real_docker_checker_final_score_through_harbor_verifier(tmp_path, docker_image, score):
    import asyncio

    from circuit_harness.execution.benchmark_replay import verify_replay

    candidate, package, config = docker_inputs(tmp_path, docker_image, score=score)
    verifier = verifier_for(tmp_path, package, config, candidate)
    assert asyncio.run(verifier.verify()).rewards == {"reward": score}
    receipt = verify_replay(verifier.trial_paths.verifier_dir / "replay")
    assert receipt["result"]["backend"] == "opensource"
    assert receipt["result"]["process"]["cleanup_confirmed"] is True
    with pytest.raises(FileExistsError):
        asyncio.run(verifier.verify())


@pytest.mark.parametrize("mode", ["broken", "malformed", "mismatched"])
def test_real_checker_failure_cannot_produce_reward(tmp_path, docker_image, mode):
    import asyncio

    from circuit_harness.execution.benchmark_replay import verify_replay

    candidate, package, config = docker_inputs(tmp_path, docker_image, mode=mode)
    verifier = verifier_for(tmp_path, package, config, candidate)
    with pytest.raises(RuntimeError, match="valid task score"):
        asyncio.run(verifier.verify())
    assert verify_replay(verifier.trial_paths.verifier_dir / "replay")["result"]["score"] is None


@pytest.mark.parametrize("deadline", [False, True])
def test_harbor_cancel_or_timeout_awaits_real_process_and_container_cleanup(
    tmp_path, docker_image, deadline
):
    import asyncio
    import subprocess

    from circuit_harness.execution.benchmark_replay import verify_replay

    candidate, package, config = docker_inputs(tmp_path, docker_image, mode="slow")
    verifier = verifier_for(tmp_path, package, config, candidate)
    replay = verifier.trial_paths.verifier_dir / "replay"

    async def run():
        task = asyncio.create_task(verifier.verify())
        async with asyncio.timeout(10):
            while not (replay / "run/work/verifier/started").exists():
                if task.done():
                    await task
                await asyncio.sleep(0.05)
        if deadline:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=0.01)
        else:
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()  # A second Harbor teardown signal must not detach cleanup.
            with pytest.raises(asyncio.CancelledError):
                await task
        await asyncio.sleep(2.2)

    asyncio.run(run())
    receipt = verify_replay(replay)
    assert receipt["result"]["execution"] == "cancelled"
    assert receipt["result"]["score"] is None
    assert receipt["result"]["process"]["cleanup_confirmed"] is True
    cleanup = json.loads((replay / "run/cleanup.json").read_text())
    assert cleanup["confirmed"] is True
    inspected = subprocess.run(["docker", "inspect", cleanup["container"]], capture_output=True)
    assert inspected.returncode != 0 and b"no such object" in inspected.stderr.lower()
    assert not (replay / "run/work/survived").exists()
    assert candidate.is_dir()
    assert (
        json.loads((verifier.trial_paths.verifier_dir / "local-execution.json").read_text())[
            "worker_finished"
        ]
        is True
    )


def test_final_schema_examples_match_runtime_and_backend_constraints():
    from jsonschema import Draft202012Validator

    from circuit_harness.harbor.config import HarborChipsConfig

    root = Path(__file__).resolve().parents[2]
    for name, model in (("config", HarborChipsConfig), ("final_evaluation", FinalEvaluationConfig)):
        schema = json.loads((root / f"circuit_harness/harbor/{name}.schema.json").read_text())
        assert schema == model.model_json_schema()
        Draft202012Validator.check_schema(schema)
    schema = FinalEvaluationConfig.model_json_schema()
    for name in ("final-evaluation", "final-evaluation-opensource"):
        value = json.loads((root / f"examples/chips/harbor/{name}.example.json").read_text())
        Draft202012Validator(schema).validate(value)
        FinalEvaluationConfig.model_validate(value)
        value["remote" if value["backend"] == "benchmark_opensource" else "opensource"] = None
        assert list(Draft202012Validator(schema).iter_errors(value))
        with pytest.raises(ValueError):
            FinalEvaluationConfig.model_validate(value)


def test_final_configuration_round_trip_preserves_selected_backend(tmp_path):
    from test_benchmark_replay import replay_inputs

    _, package, config = replay_inputs(tmp_path)
    for values in (
        {"task_package": package, "remote": {}},
        {"task_package": package, "backend": "benchmark_opensource", "opensource": config},
    ):
        selected = FinalEvaluationConfig.model_validate(values)
        assert FinalEvaluationConfig.model_validate_json(selected.model_dump_json()) == selected


class ReplayTrialEnvironment(ControlledEnvironment):
    async def freeze(self, reason):
        from circuit_harness.execution.candidate_bundle import freeze_candidate

        if self.frozen is None:
            directory = self.trial_paths.trial_dir / "frozen"
            self.frozen = freeze_candidate(
                self.workspace,
                directory,
                ["dut.va"],
                task_id="fixture-task",
                task_version="fixture-v1",
                reason=reason,
            )
            self.frozen["candidate_directory"] = str(directory)
        return self.frozen


@pytest.mark.parametrize("score", [1, 0])
def test_real_harbor_trial_uses_legacy_native_config_optional_replay(tmp_path, docker_image, score):
    import asyncio
    import sys

    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial

    from circuit_harness.execution.benchmark_replay import verify_replay

    source = tmp_path / "input"
    source.mkdir()
    _, package, replay_config = docker_inputs(source, docker_image, score=score)
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text("exit 0\n")
    (task / "instruction.md").write_text("Controlled candidate\n")
    (task / "task.toml").write_text("[agent]\ntimeout_sec=5\n[verifier]\ntimeout_sec=5\n")
    (task / "environment/agent.py").write_text(
        "import sys,pathlib\n"
        "(pathlib.Path(sys.argv[1])/'dut.va').write_text('controlled candidate')\n"
    )
    declaration = {
        "task": {
            "task_id": "fixture-task",
            "task_version": "fixture-v1",
            "public_files": [],
            "candidate_files": ["dut.va"],
            "manifest": {},
        },
        "materials": str(task),
        "checkout": str(task),
        "kernel": str(task / "kernel"),
        "image": "controlled-public-only",
        "executable": sys.executable,
        "final_task_package": str(package),
        "final_backend": "benchmark_opensource",
        "final_opensource": replay_config,
    }
    (task / "environment/harness.json").write_text(json.dumps(declaration))
    config = TrialConfig(
        task={"path": task},
        trial_name="trial",
        trials_dir=tmp_path / "trials",
        agent={
            "import_path": "circuit_harness.harbor.agent:NativeCodexAgent",
            "model_name": "controlled-model",
        },
        environment={"import_path": __name__ + ":ReplayTrialEnvironment"},
        verifier={"import_path": "circuit_harness.harbor.verifier:FrozenCandidateVerifier"},
    )
    trial = asyncio.run(Trial.create(config))
    result = asyncio.run(trial.run())
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": score}
    receipt = verify_replay(trial.paths.verifier_dir / "replay")
    assert (
        receipt["result"]["candidate_sha256"] == trial.agent_environment.frozen["candidate_sha256"]
    )


def test_task_mismatch_cannot_start_final_replay(tmp_path):
    import asyncio

    from test_benchmark_replay import replay_inputs

    candidate, package, config = replay_inputs(tmp_path)
    verifier = verifier_for(tmp_path, package, config, candidate)
    verifier.environment.settings.task["task_version"] = "different"
    with pytest.raises(ValueError, match="declarations differ"):
        asyncio.run(verifier.verify())
    assert not (verifier.trial_paths.verifier_dir / "replay").exists()


def test_cancel_before_local_runner_starts_is_sealed_without_execution(tmp_path, docker_image):
    import threading

    from circuit_harness.execution.benchmark_replay import replay_candidate, verify_replay

    candidate, package, config = docker_inputs(tmp_path, docker_image, mode="slow")
    cancel = threading.Event()
    cancel.set()
    output = tmp_path / "early-cancel"
    replay_candidate(candidate, package, config, output, cancel=cancel)
    result = verify_replay(output)["result"]
    assert result["execution"] == "cancelled"
    assert result["process"]["cleanup_confirmed"] is True
    assert not (output / "run/work/verifier/started").exists()
    assert '"process_started"' not in (output / "run/events.jsonl").read_text()
