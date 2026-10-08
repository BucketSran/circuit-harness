"""Task routing at the public Harbor profile and Trial seams."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("harbor")
from harbor.agents.nop import NopAgent
from test_harbor_profiles import catalog

from circuit_harness.harbor.docker_environment import CircuitDockerEnvironment
from circuit_harness.harbor.profiles import compile_job
from circuit_harness.harbor.verifier import FrozenCandidateVerifier


def suite(tmp_path):
    tasks = []
    entries = []
    for name in ("a", "b"):
        task = tmp_path / name / "same-name"
        (task / "environment").mkdir(parents=True)
        (task / "environment/Dockerfile").write_text("FROM fixture\n")
        (task / "instruction.md").write_text(name)
        (task / "task.toml").write_text("[agent]\ntimeout_sec=2\n")
        (task / "tests").mkdir()
        (task / "tests/test.sh").write_text("exit 0\n")
        private = tmp_path / "private" / name
        private.mkdir(parents=True)
        (private / "materials").mkdir()
        engine = private / "engine"
        engine.mkdir()
        (engine / ".gitignore").write_text("__pycache__/\n")
        source = engine / "evas/src/evas/__init__.py"
        source.parent.mkdir(parents=True)
        source.write_text("VERSION = 1\n")
        subprocess.run(["git", "-C", str(engine), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(engine), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(engine),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        (private / "kernel").write_text("kernel fixture")
        package = private / "final"
        package.mkdir()
        (package / "manifest.json").write_text(json.dumps({"task_id": name, "task_version": "v1"}))
        public = private / "public.json"
        public.write_text(
            json.dumps(
                dict(
                    task={"task_id": name, "task_version": "v1"},
                    materials=str(private / "materials"),
                    checkout=str(private / "engine"),
                    kernel=str(private / "kernel"),
                    image="fixture",
                )
            )
        )
        final = private / "final.json"
        final.write_text(json.dumps(dict(task_package=str(package), remote={})))
        entries.append(
            dict(
                task_path=str(task),
                task_id=name,
                task_version="v1",
                session_config=str(public),
                final_config=str(final),
            )
        )
        tasks.append({"path": str(task)})
    manifest = tmp_path / "private/bindings.json"
    manifest.write_text(json.dumps({"schema_version": 1, "tasks": entries}))
    job = dict(
        tasks=tasks,
        environment={
            "import_path": ("circuit_harness.harbor.docker_environment:CircuitDockerEnvironment"),
            "kwargs": {"task_bindings": str(manifest)},
        },
        verifier={
            "import_path": "circuit_harness.harbor.verifier:FrozenCandidateVerifier",
            "kwargs": {"task_bindings": str(manifest)},
        },
    )
    return manifest, entries, job


def test_compile_pins_each_actual_task_and_rejects_missing_binding(tmp_path):
    manifest, entries, job = suite(tmp_path)
    compiled = compile_job(catalog(), "code", "glm", job)
    assert (
        compiled.environment.kwargs["task_bindings_sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert set(compiled.environment.kwargs["task_binding_receipts"]) == {
        e["task_path"] for e in entries
    }
    manifest.write_text(json.dumps({"schema_version": 1, "tasks": entries[:1]}))
    with pytest.raises(ValueError, match="missing"):
        compile_job(catalog(), "code", "glm", job)


def test_environment_and_verifier_select_the_actual_task_and_detect_drift(tmp_path):
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths

    from circuit_harness.harbor.docker_environment import CircuitDockerEnvironment
    from circuit_harness.harbor.verifier import FrozenCandidateVerifier

    manifest, entries, job = suite(tmp_path)
    compiled = compile_job(catalog(), "code", "glm", job)
    task = Path(entries[1]["task_path"])
    paths = TrialPaths(trial_dir=tmp_path / "trials/b")
    environment = CircuitDockerEnvironment(
        environment_dir=task / "environment",
        trial_paths=paths,
        environment_name="fixture",
        session_id="binding-test",
        task_env_config=Task(task).config.environment,
        gateway_bind_host="127.0.0.1",
        gateway_host="localhost",
        **compiled.environment.kwargs,
    )
    assert environment.settings.task["task_id"] == "b"
    verifier = FrozenCandidateVerifier(
        task=Task(task), trial_paths=paths, environment=environment, **compiled.verifier.kwargs
    )
    assert verifier.final_config.task_package.name == "final"
    receipt = json.loads((paths.trial_dir / "task-binding.json").read_text())
    assert receipt["task_id"] == "b"
    assert "/private/" not in json.dumps(receipt)
    Path(entries[1]["session_config"]).write_text("{}")
    with pytest.raises(ValueError):
        FrozenCandidateVerifier(
            task=Task(task), trial_paths=paths, environment=environment, **compiled.verifier.kwargs
        )


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "unknown",
        "scalar",
        "identity",
        "cross-task-private",
        "unsupported",
        "remote",
        "task-drift",
        "package-drift",
    ],
)
def test_compile_rejects_bad_bindings_without_starting_tasks(tmp_path, failure):
    manifest, entries, job = suite(tmp_path)
    if failure == "duplicate":
        entries.append(entries[0])
    elif failure == "unknown":
        job["tasks"] = job["tasks"][:1]
    elif failure == "scalar":
        job["environment"]["kwargs"]["session_config"] = entries[0]["session_config"]
    elif failure == "identity":
        entries[0]["task_version"] = "wrong"
    elif failure == "cross-task-private":
        public = Path(entries[0]["session_config"])
        config = json.loads(public.read_text())
        config["materials"] = entries[1]["task_path"]
        public.write_text(json.dumps(config))
    elif failure == "unsupported":
        entries[0].update(support="unsupported", reason="no calibrated oracle")
        del entries[0]["session_config"]
        del entries[0]["final_config"]
    elif failure == "remote":
        job["tasks"][0]["git_url"] = "https://example.invalid/tasks.git"
    manifest.write_text(json.dumps({"schema_version": 1, "tasks": entries}))
    if failure.endswith("drift"):
        compiled = compile_job(catalog(), "code", "glm", job)
        from circuit_harness.harbor.task_bindings import resolve_task_binding

        changed = (
            Path(entries[0]["task_path"]) / "instruction.md"
            if failure == "task-drift"
            else Path(entries[0]["final_config"]).parent / "final/manifest.json"
        )
        changed.write_text(changed.read_text() + " ")
        with pytest.raises(ValueError, match="drift"):
            resolve_task_binding(
                manifest,
                entries[0]["task_path"],
                expected_receipt=compiled.environment.kwargs["task_binding_receipts"][
                    entries[0]["task_path"]
                ],
            )
    else:
        with pytest.raises(ValueError):
            compile_job(catalog(), "code", "glm", job)


class ControlledBindingEnvironment(CircuitDockerEnvironment):
    async def start(self, force_build):
        self.candidate_dir = self.trial_paths.trial_dir / "controlled-candidate"
        self.candidate_dir.mkdir()

    async def stop(self, delete):
        pass


class ControlledBindingAgent(NopAgent):
    async def run(self, instruction, environment, context):
        content = instruction.encode()
        (environment.candidate_dir / "dut.va").write_bytes(content)
        environment.frozen = {
            "candidate_directory": str(environment.candidate_dir),
            "candidate_sha256": hashlib.sha256(content).hexdigest(),
        }


class ControlledBindingVerifier(FrozenCandidateVerifier):
    async def evaluate(self, environment, candidate):
        # This controlled checker distinguishes bytes through its own private config.
        expected = self.final_config.task_package.parent.name.encode()
        matched = (Path(candidate["candidate_directory"]) / "dut.va").read_bytes() == expected
        return dict(
            state="completed",
            execution="ok",
            verdict="pass" if matched else "fail",
            score=int(matched),
            candidate_sha256=candidate["candidate_sha256"],
            task_id=self.task_binding["task_id"],
            task_version="v1",
            purpose="final",
        )


def test_two_real_harbor_trials_keep_candidates_and_checker_routing_distinct(tmp_path):
    import asyncio

    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial

    manifest, entries, job = suite(tmp_path)
    compiled = compile_job(catalog(), "code", "glm", job)

    async def run_suite():
        trials = []
        for entry in entries:
            config = TrialConfig(
                task={"path": entry["task_path"]},
                trial_name=entry["task_id"],
                trials_dir=tmp_path / "trials",
                agent={"import_path": __name__ + ":ControlledBindingAgent"},
                environment={
                    "import_path": __name__ + ":ControlledBindingEnvironment",
                    "kwargs": {
                        **compiled.environment.kwargs,
                        "gateway_bind_host": "127.0.0.1",
                        "gateway_host": "localhost",
                    },
                },
                verifier={
                    "import_path": __name__ + ":ControlledBindingVerifier",
                    "kwargs": compiled.verifier.kwargs,
                },
            )
            trials.append(await Trial.create(config))
        return trials, await asyncio.gather(*(trial.run() for trial in trials))

    trials, results = asyncio.run(run_suite())
    assert all(result.exception_info is None for result in results)
    assert [result.verifier_result.rewards for result in results] == [{"reward": 1}, {"reward": 1}]
    assert [
        (trial.agent_environment.candidate_dir / "dut.va").read_bytes() for trial in trials
    ] == [b"a", b"b"]
    receipts = [
        json.loads((trial.paths.trial_dir / "task-binding.json").read_text()) for trial in trials
    ]
    assert [receipt["task_id"] for receipt in receipts] == ["a", "b"]
    assert receipts[0]["binding_sha256"] != receipts[1]["binding_sha256"]


def test_adapter_rejects_private_resources_inside_another_trial_export(tmp_path):
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths

    manifest, entries, job = suite(tmp_path)
    public = Path(entries[0]["session_config"])
    data = json.loads(public.read_text())
    leaked = tmp_path / "trials/other/private-materials"
    leaked.mkdir(parents=True)
    data["materials"] = str(leaked)
    public.write_text(json.dumps(data))
    task = Task(Path(entries[0]["task_path"]))
    with pytest.raises(ValueError, match="export roots"):
        CircuitDockerEnvironment(
            environment_dir=task.paths.environment_dir,
            trial_paths=TrialPaths(trial_dir=tmp_path / "trials/active"),
            environment_name="fixture",
            session_id="fixture",
            task_env_config=task.config.environment,
            task_bindings=manifest,
            gateway_bind_host="127.0.0.1",
            gateway_host="localhost",
        )


def test_trial_rejects_incomplete_compilation_pins(tmp_path):
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths

    manifest, entries, job = suite(tmp_path)
    compiled = compile_job(catalog(), "code", "glm", job)
    task = Task(Path(entries[0]["task_path"]))
    kwargs = compiled.environment.kwargs.copy()
    kwargs["task_binding_receipts"] = {}
    with pytest.raises(ValueError, match="pins"):
        CircuitDockerEnvironment(
            environment_dir=task.paths.environment_dir,
            trial_paths=TrialPaths(trial_dir=tmp_path / "trials/a"),
            environment_name="fixture",
            session_id="fixture",
            task_env_config=task.config.environment,
            gateway_bind_host="127.0.0.1",
            gateway_host="localhost",
            **kwargs,
        )


def test_compilation_ignores_evas_history_and_cache_but_pins_captured_source(tmp_path):

    from circuit_harness.harbor.task_bindings import resolve_task_binding

    manifest, entries, job = suite(tmp_path)
    compiled = compile_job(catalog(), "code", "glm", job)
    checkout = Path(entries[0]["session_config"]).parent / "engine"
    source = checkout / "evas/src/evas/__init__.py"
    # Compile source pin before unrelated output is added.
    compiled = compile_job(catalog(), "code", "glm", job)
    expected = compiled.environment.kwargs["task_binding_receipts"][entries[0]["task_path"]]
    (checkout / "runs").mkdir()
    (checkout / "runs/history.log").write_text("unrelated old run")
    (checkout / "runs/runtime-link").symlink_to("/unrelated/nonexistent/runtime")
    cache = source.parent / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"cached bytes")
    selection = resolve_task_binding(manifest, entries[0]["task_path"], expected_receipt=expected)
    assert selection.receipt == expected
    source.write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="drift"):
        resolve_task_binding(manifest, entries[0]["task_path"], expected_receipt=expected)


def test_real_python_link_is_accepted_and_linked_executable_bytes_are_pinned(tmp_path):
    import shutil
    import sys

    from circuit_harness.harbor.task_bindings import resolve_task_binding

    assert Path(sys.executable).is_symlink(), "This controlled host uses a linked Python runtime"
    manifest, entries, job = suite(tmp_path)
    public_path = Path(entries[0]["session_config"])
    executable = public_path.parent / "controlled-python"
    shutil.copyfile(Path(sys.executable).resolve(), executable)
    link = public_path.parent / "python-link"
    link.symlink_to(executable)
    public = json.loads(public_path.read_text())
    public.update(
        public_backend="native_codex_sandbox",
        image=None,
        public_python=sys.executable,
        public_codex=str(link),
    )
    public_path.write_text(json.dumps(public))
    compiled = compile_job(catalog(), "code", "glm", job)
    expected = compiled.environment.kwargs["task_binding_receipts"][entries[0]["task_path"]]
    assert (
        resolve_task_binding(manifest, entries[0]["task_path"], expected_receipt=expected).receipt
        == expected
    )
    with executable.open("ab") as stream:
        stream.write(b"changed controlled executable bytes")
    with pytest.raises(ValueError, match="drift"):
        resolve_task_binding(manifest, entries[0]["task_path"], expected_receipt=expected)


def test_bindings_accept_independent_opensource_final_configuration(tmp_path):
    from circuit_harness.harbor.task_bindings import resolve_task_binding

    manifest, entries, job = suite(tmp_path)
    final_path = Path(entries[0]["final_config"])
    final = json.loads(final_path.read_text())
    final.pop("remote")
    final.update(
        backend="benchmark_opensource",
        opensource={
            "schema_version": 1,
            "image": "sha256:" + "a" * 64,
            "task_package_sha256": "b" * 64,
            "solver_options": {},
            "unsupported": [],
            "timeout_s": 2.0,
            "max_output_bytes": 1024,
        },
    )
    final_path.write_text(json.dumps(final))
    compiled = compile_job(catalog(), "code", "glm", job)
    selection = resolve_task_binding(
        manifest,
        entries[0]["task_path"],
        expected_receipt=compiled.environment.kwargs["task_binding_receipts"][
            entries[0]["task_path"]
        ],
    )
    assert selection.final_config_path == final_path
