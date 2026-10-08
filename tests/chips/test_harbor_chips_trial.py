"""Real Harbor Trial tests with a controlled local process, not a model or sandbox."""

import asyncio
import json
import sys

import pytest

pytest.importorskip("harbor")
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from alphaapollo.workflows.harbor_chips.agent import NativeCodexAgent as NativeCodexAgent
from alphaapollo.workflows.harbor_chips.environment import HarborChipsEnvironment
from alphaapollo.workflows.harbor_chips.verifier import FrozenCandidateVerifier


class ControlledEnvironment(HarborChipsEnvironment):
    """Exercise real process cancellation without asserting isolation."""

    async def start(self, force_build):
        self.events = []
        self.workspace.mkdir(parents=True, exist_ok=True)

    def command(self, model):
        return [sys.executable, str(self.environment_dir / "agent.py"), str(self.workspace)], {}

    async def freeze(self, reason):
        if self.frozen is None:
            candidate = self.workspace / "dut.va"
            self.frozen = {
                "candidate_directory": str(self.workspace),
                "candidate_sha256": "fixture",
            }
            self.events.append(("freeze", reason, candidate.read_bytes()))
        return self.frozen


class ControlledVerifier(FrozenCandidateVerifier):
    async def evaluate(self, environment, candidate):
        environment.events.append(("verify", candidate["candidate_sha256"]))
        return {
            "state": "completed",
            "score": 0.5,
            "execution": "ok",
            "verdict": "fail",
            "candidate_sha256": "fixture",
            "task_id": "fixture",
            "task_version": "1",
            "purpose": "final",
        }


def make_trial(
    tmp_path,
    script,
    timeout=2,
    *,
    verifier_class=ControlledVerifier,
    final_timeout=600,
    environment_class=ControlledEnvironment,
    max_output_bytes=16 * 1024 * 1024,
):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text("exit 0\n")
    (task / "instruction.md").write_text("Produce candidate bytes.\n")
    (task / "task.toml").write_text("[agent]\ntimeout_sec=2\n[verifier]\ntimeout_sec=2\n")
    (task / "environment/agent.py").write_text(script)
    (task / "environment/harness.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": {
                    "task_id": "fixture",
                    "task_version": "1",
                    "public_files": [],
                    "candidate_files": ["dut.va"],
                    "manifest": {},
                },
                "materials": str(task),
                "checkout": str(task),
                "kernel": str(task / "kernel"),
                "image": "fixture-only",
                "executable": sys.executable,
                "final_task_package": str(task),
                "final_remote": {},
                "final_timeout_s": final_timeout,
                "max_output_bytes": max_output_bytes,
            }
        )
    )
    config = TrialConfig(
        task={"path": task},
        trial_name="trial",
        trials_dir=tmp_path / "trials",
        agent={
            "import_path": __name__ + ":NativeCodexAgent",
            "model_name": "fixture-model",
            "override_timeout_sec": timeout,
        },
        environment={"import_path": __name__ + ":" + environment_class.__name__},
        verifier={"import_path": __name__ + ":" + verifier_class.__name__},
    )
    return asyncio.run(Trial.create(config))


def test_real_trial_freezes_candidate_once_and_grades_same_bytes(tmp_path):
    trial = make_trial(
        tmp_path,
        "import pathlib,sys\np=pathlib.Path(sys.argv[1]);"
        "(p/'dut.va').write_bytes(b'last complete\\n')\n",
    )
    result = asyncio.run(trial.run())
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": 0.5}
    assert trial.agent_environment.events == [
        ("freeze", "completed", b"last complete\n"),
        ("verify", "fixture"),
    ]
    assert (trial.paths.agent_dir / "native-events.jsonl").exists()


def test_real_trial_deadline_stops_descendants_before_freezing_and_verifies(tmp_path):
    script = """import pathlib,subprocess,sys,time
p=pathlib.Path(sys.argv[1]); (p/'dut.va').write_bytes(b'last complete\\n')
child="import pathlib,time,sys;time.sleep(0.5);p=pathlib.Path(sys.argv[1]);p.write_bytes(b'late')"
subprocess.Popen([sys.executable,'-c',child,str(p/'dut.va')])
time.sleep(10)
"""
    trial = make_trial(tmp_path, script, timeout=0.25)

    async def run():
        result = await trial.run()
        await asyncio.sleep(0.6)
        return result

    result = asyncio.run(run())
    assert result.exception_info.exception_type == "AgentTimeoutError"
    assert result.verifier_result.rewards == {"reward": 0.5}
    assert (trial.agent_environment.workspace / "dut.va").read_bytes() == b"last complete\n"
    assert trial.agent_environment.events[0] == ("freeze", "cancelled", b"last complete\n")


def test_external_trial_cancellation_freezes_without_grading_or_restart(tmp_path):
    trial = make_trial(
        tmp_path,
        "import pathlib,sys,time\np=pathlib.Path(sys.argv[1]);"
        "(p/'dut.va').write_bytes(b'cancelled candidate');time.sleep(10)\n",
    )

    async def run():
        running = asyncio.create_task(trial.run())
        for _ in range(100):
            if (trial.agent_environment.workspace / "dut.va").exists():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("controlled agent did not start")
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    asyncio.run(run())
    assert trial.agent_environment.events == [("freeze", "cancelled", b"cancelled candidate")]
    assert trial.paths.result_path.exists()


class BrokenVerifier(FrozenCandidateVerifier):
    async def evaluate(self, environment, candidate):
        return {"state": "failed", "score": None, "candidate_sha256": "fixture"}


def test_real_trial_infrastructure_failure_has_no_reward(tmp_path):
    trial = make_trial(
        tmp_path,
        "import pathlib,sys\n(pathlib.Path(sys.argv[1])/'dut.va').write_bytes(b'candidate')\n",
        verifier_class=BrokenVerifier,
    )
    result = asyncio.run(trial.run())
    assert result.verifier_result is None
    assert result.exception_info.exception_type == "RuntimeError"
    assert "valid task score" in result.exception_info.exception_message


def test_trial_retains_native_usage_without_inventing_missing_metrics(tmp_path):
    script = """import pathlib,sys,json
(pathlib.Path(sys.argv[1])/'dut.va').write_bytes(b'candidate')
print(json.dumps({'type':'thread.started','thread_id':'native-fixture'}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':19,'output_tokens':7}}))
"""
    result = asyncio.run(make_trial(tmp_path, script).run())
    assert result.agent_result.n_input_tokens == 19
    assert result.agent_result.n_output_tokens == 7
    assert result.agent_result.metadata["native_thread_id"] == "native-fixture"
    assert result.agent_result.metadata["cost"] is None


class WrongCandidateVerifier(FrozenCandidateVerifier):
    async def evaluate(self, environment, candidate):
        return {
            "state": "completed",
            "score": 1,
            "candidate_sha256": "another-candidate",
            "execution": "ok",
            "verdict": "pass",
        }


def test_trial_rejects_grade_for_another_frozen_candidate(tmp_path):
    trial = make_trial(
        tmp_path,
        "import pathlib,sys\n(pathlib.Path(sys.argv[1])/'dut.va').write_bytes(b'candidate')\n",
        verifier_class=WrongCandidateVerifier,
    )
    result = asyncio.run(trial.run())
    assert result.verifier_result is None
    assert "identity mismatch" in result.exception_info.exception_message


class SlowVerifier(FrozenCandidateVerifier):
    async def evaluate(self, environment, candidate):
        await asyncio.sleep(10)
        return {"state": "completed", "score": 1, "candidate_sha256": "fixture"}


def test_trial_final_deadline_retains_candidate_without_a_fake_reward(tmp_path):
    trial = make_trial(
        tmp_path,
        "import pathlib,sys\n(pathlib.Path(sys.argv[1])/'dut.va').write_bytes(b'candidate')\n",
        verifier_class=SlowVerifier,
        final_timeout=0.05,
    )
    result = asyncio.run(trial.run())
    assert result.verifier_result is None
    assert result.exception_info.exception_type == "VerifierTimeoutError"
    assert trial.agent_environment.events == [("freeze", "completed", b"candidate")]


class WrongTaskVerifier(FrozenCandidateVerifier):
    async def evaluate(self, environment, candidate):
        return {
            "state": "completed",
            "score": 1,
            "candidate_sha256": "fixture",
            "execution": "ok",
            "verdict": "pass",
            "task_id": "another-task",
            "task_version": "1",
            "purpose": "final",
        }


def test_trial_rejects_a_score_from_another_task_with_identical_candidate_bytes(tmp_path):
    trial = make_trial(
        tmp_path,
        "import pathlib,sys\n(pathlib.Path(sys.argv[1])/'dut.va').write_bytes(b'candidate')\n",
        verifier_class=WrongTaskVerifier,
    )
    result = asyncio.run(trial.run())
    assert result.verifier_result is None
    assert "task identity" in result.exception_info.exception_message


def test_production_trial_refuses_implicit_user_credentials_before_agent_launch(tmp_path):
    trial = make_trial(
        tmp_path,
        "raise AssertionError('must not launch')\n",
        environment_class=HarborChipsEnvironment,
    )
    result = asyncio.run(trial.run())
    assert result.verifier_result is None
    assert result.agent_execution is None
    assert result.exception_info.exception_type == "ValueError"
    assert "explicit operator auth file" in result.exception_info.exception_message
    assert not (trial.paths.agent_dir / "native-events.jsonl").exists()


class InfrastructureZeroVerifier(FrozenCandidateVerifier):
    async def evaluate(self, environment, candidate):
        return {
            "state": "completed",
            "execution": "infrastructure_error",
            "score": 0,
            "candidate_sha256": "fixture",
            "task_id": "fixture",
            "task_version": "1",
            "purpose": "final",
            "verdict": "fail",
        }


def test_trial_does_not_turn_infrastructure_failure_into_model_zero(tmp_path):
    trial = make_trial(
        tmp_path,
        "import pathlib,sys\n(pathlib.Path(sys.argv[1])/'dut.va').write_bytes(b'candidate')\n",
        verifier_class=InfrastructureZeroVerifier,
    )
    result = asyncio.run(trial.run())
    assert result.verifier_result is None
    assert "valid task score" in result.exception_info.exception_message


def test_trial_output_limit_kills_process_and_freezes_without_grading(tmp_path):
    script = """import pathlib,sys,time
(pathlib.Path(sys.argv[1])/'dut.va').write_bytes(b'candidate')
sys.stderr.write('x'*100000);sys.stderr.flush();time.sleep(10)
"""
    trial = make_trial(tmp_path, script, max_output_bytes=1024)
    result = asyncio.run(trial.run())
    assert result.verifier_result is None
    assert result.exception_info.exception_type == "NativeOutputLimitError"
    assert trial.agent_environment.events == [("freeze", "agent_error", b"candidate")]
    assert (
        sum(
            (trial.paths.agent_dir / name).stat().st_size
            for name in ("native-events.jsonl", "native-stderr.log")
        )
        <= 1024
    )


def test_multiple_native_turns_do_not_report_last_turn_as_trial_usage(tmp_path):
    script = """import pathlib,sys,json
(pathlib.Path(sys.argv[1])/'dut.va').write_bytes(b'candidate')
for count in (19, 3):
    print(json.dumps({'type':'turn.completed','usage':{'input_tokens':count,'output_tokens':7}}))
"""
    result = asyncio.run(make_trial(tmp_path, script).run())
    assert result.agent_result.n_input_tokens is None
    assert result.agent_result.metadata["usage"] is None
    assert len(result.agent_result.metadata["turn_usage"]) == 2


@pytest.mark.parametrize("public_backend", ["docker", "native_codex_sandbox"])
def test_production_trial_installation_uses_real_sandbox_and_public_broker_without_model(
    tmp_path, public_backend
):
    import hashlib
    import os
    import pathlib
    import shutil

    from test_current_evas_session import make_session

    executable = shutil.which("codex")
    if sys.platform != "darwin" or not executable:
        pytest.skip("requires native Codex macOS sandbox; no model is invoked")
    package = pathlib.Path(executable).resolve().parents[1]
    binaries = list(package.glob("node_modules/@openai/codex-*/vendor/*/bin/codex"))
    if not binaries:
        pytest.skip("declare native Codex runtime for this installation")
    codex = binaries[0]
    trial = make_trial(tmp_path, "", environment_class=HarborChipsEnvironment)
    fixture_root = tmp_path / "public-fixture"
    fixture_root.mkdir()
    make_session(fixture_root)
    config_path = trial.agent_environment.environment_dir / "harness.json"
    config = json.loads(config_path.read_text())
    session = json.loads((fixture_root / "session/session.json").read_text())
    config.update(
        materials=str(fixture_root / "materials"),
        checkout=str(fixture_root / "engine"),
        kernel=str(fixture_root / "kernel"),
        image="sha256:" + "a" * 64,
        executable=str(codex),
        allowed_hosts=["example.invalid"],
        runtime_readonly_paths=[str(codex.parent.parent), str(pathlib.Path(sys.base_prefix))],
    )
    config["task"] = session["task"]
    if public_backend == "native_codex_sandbox":
        (fixture_root / "kernel").write_bytes(b"\xcf\xfa\xed\xfe synthetic Mach-O; not runnable")
        config.update(
            public_backend=public_backend,
            public_codex=str(codex),
            public_python=str(pathlib.Path(sys.executable).resolve()),
            image=None,
        )
    auth = tmp_path / "operator-auth"
    auth.mkdir()
    (auth / "auth.json").write_text("{}")
    config["auth_file"] = str(auth / "auth.json")
    final_package = tmp_path / "final-package"
    (final_package / "tests").mkdir(parents=True)
    checker = final_package / "tests/test.sh"
    checker.write_text("exit 0\n")
    config["final_task_package"] = str(final_package)
    manifest = {
        "schema_version": 1,
        "task_id": config["task"]["task_id"],
        "task_version": config["task"]["task_version"],
        "purpose": "final",
        "entrypoint": "tests/test.sh",
        "candidate_file": "dut.va",
        "report_path": "verifier/report.json",
        "feedback_fields": [],
        "criteria_sha256": "a" * 64,
        "condition_id": "fixture",
        "task_set": "public",
        "files": {
            "tests/test.sh": {
                "sha256": hashlib.sha256(checker.read_bytes()).hexdigest(),
                "bytes": checker.stat().st_size,
            }
        },
    }
    (checker.parents[1] / "manifest.json").write_text(json.dumps(manifest))
    config["final_remote"] = {
        "host": "fixture-no-connect",
        **{
            key: "/fixture/" + key
            for key in ("python", "bundle", "profile", "run_root", "archive_root", "upload_root")
        },
    }
    config_path.write_text(json.dumps(config))
    trial.agent_environment.settings = trial.agent_environment.settings.read(config_path)

    async def install_and_probe():
        environment = trial.agent_environment
        try:
            await environment.start(False)
            await trial.agent.setup(environment)
            argv, overlays = environment.sandbox.command(
                ["/usr/bin/env", f"CODEX_HOME={environment.native_home}", str(codex), "--version"]
            )
            process = await asyncio.create_subprocess_exec(
                *argv,
                env={"PATH": os.defpath, **overlays},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            output, error = await asyncio.wait_for(process.communicate(), 10)
            assert process.returncode == 0, error.decode()
            assert b"codex-cli" in output
            assert environment.sandbox.receipt["access_probe"] == "synthetic_file_boundary_passed"
            assert environment.broker.socket_path.is_socket()
            created = json.loads((environment.session_directory / "session.json").read_text())
            assert created["backend"] == public_backend
            if public_backend == "native_codex_sandbox":
                assert created["image"] is None
                assert created["codex"] == str(codex)
                assert created["python"] == str(pathlib.Path(sys.executable).resolve())
                assert (
                    (environment.session_directory / "runtime/evas-kernel")
                    .read_bytes()
                    .startswith(b"\xcf\xfa\xed\xfe")
                )
            from alphaapollo.common.execution.chips.current_evas_session import session_info

            assert environment.native_conditions()["public_mcp_tools"] == [
                schema["function"]["name"]
                for schema in session_info(environment.session_directory)["tools"]
            ]
            # Native cache writes work while private session reads stay denied.
            argv, overlays = environment.sandbox.command(
                [
                    "/bin/sh",
                    "-c",
                    'echo cache > "$1/cache"; ! cat "$2/session.json"',
                    "sh",
                    str(environment.native_home),
                    str(environment.session_directory),
                ]
            )
            process = await asyncio.create_subprocess_exec(
                *argv,
                env={"PATH": os.defpath, **overlays},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
            assert process.returncode == 0
            assert (environment.native_home / "cache").read_text() == "cache\n"
        finally:
            await environment.stop(True)
        assert not environment.native_home.exists()
        assert not environment.broker.socket_path.exists()

    asyncio.run(install_and_probe())


def test_native_abnormal_exit_is_unscored_and_retains_candidate(tmp_path):
    script = """import pathlib,sys
(pathlib.Path(sys.argv[1])/'dut.va').write_bytes(b'candidate')
sys.exit(7)
"""
    trial = make_trial(tmp_path, script)
    result = asyncio.run(trial.run())
    assert result.verifier_result is None
    assert result.exception_info.exception_type == "NativeAgentExecutionError"
    assert trial.agent_environment.events == [("freeze", "agent_error", b"candidate")]


@pytest.mark.parametrize(
    "tail",
    ["print('malformed')", "print(json.dumps({'type':'turn.started'}),flush=True);time.sleep(10)"],
)
def test_partial_native_event_evidence_does_not_claim_complete_usage(tmp_path, tail):
    script = (
        """import pathlib,sys,json,time
(pathlib.Path(sys.argv[1])/'dut.va').write_bytes(b'candidate')
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':19,'output_tokens':7}}),flush=True)
"""
        + tail
    )
    result = asyncio.run(make_trial(tmp_path, script, timeout=0.25).run())
    assert result.agent_result.n_input_tokens is None
    assert result.agent_result.metadata["usage"] is None
    assert len(result.agent_result.metadata["turn_usage"]) == 1
    assert result.agent_result.metadata["usage_completeness"] == "unknown"


def test_broker_cleanup_failure_still_removes_native_credentials(tmp_path):
    class BrokenBroker:
        async def close(self):
            raise RuntimeError("fixture cleanup failure")

    trial = make_trial(tmp_path, "")
    environment = trial.agent_environment
    environment.native_home.mkdir(parents=True)
    (environment.native_home / "auth.json").write_text("fixture credential")
    environment.broker = BrokenBroker()
    with pytest.raises(RuntimeError, match="fixture cleanup failure"):
        asyncio.run(environment.stop(True))
    assert not environment.native_home.exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"image": "sha256:" + "a" * 64}, "image=null"),
        ({"public_python": None}, "explicit absolute"),
        ({"experiments": {}}, "experiments require a container"),
    ],
)
def test_native_public_backend_rejects_ambiguous_operator_conditions(tmp_path, change, message):
    from pathlib import Path

    trial = make_trial(tmp_path, "")
    config = trial.agent_environment.settings.model_dump(mode="json")
    config.update(
        public_backend="native_codex_sandbox",
        image=None,
        public_codex=str(Path(sys.executable).resolve()),
        public_python=str(Path(sys.executable).resolve()),
    )
    if "experiments" in change:
        config["task"]["experiments"] = change["experiments"]
    else:
        config.update(change)
    with pytest.raises(ValueError, match=message):
        trial.agent_environment.settings.model_validate(config)


class RemotePublicFixtureEnvironment(ControlledEnvironment):
    async def start(self, force_build):
        await super().start(force_build)
        await self._create_public_session()

    def command(self, model):
        from pathlib import Path

        return [
            sys.executable,
            str(self.environment_dir / "agent.py"),
            str(self.session_directory),
        ], {"PYTHONPATH": str(Path(__file__).resolve().parents[2])}

    async def freeze(self, reason):
        return await HarborChipsEnvironment.freeze(self, reason)


class RemotePublicFixtureVerifier(FrozenCandidateVerifier):
    async def evaluate(self, environment, candidate):
        from pathlib import Path

        assert (
            Path(candidate["candidate_directory"]) / "files/dut.va"
        ).read_text() == "after feedback"
        return {
            "state": "completed",
            "score": 0.5,
            "execution": "ok",
            "verdict": "fail",
            "candidate_sha256": candidate["candidate_sha256"],
            "task_id": "fixture-task",
            "task_version": "fixture-v1",
            "purpose": "final",
        }


def test_configuration_b_uses_harbor_session_preparation_and_final_freeze(tmp_path, monkeypatch):
    from test_benchmark_spectre import task_package

    from alphaapollo.workflows.harbor_chips.config import HarborChipsConfig

    script = """
import sys
from pathlib import Path
from alphaapollo.common.execution.chips.current_evas_session import session_action
from alphaapollo.common.execution.chips import remote_public
class FixtureRemote:
    submissions = 0
    def __init__(self, config, evidence, **kwargs): pass
    def submit(self, candidate, package, job_id):
        FixtureRemote.submissions += 1
    def query(self, job_id): return {'state':'finished'}
    def retrieve(self, job_id):
        return {'purpose':'public','execution':'ok','candidate_sha256':'fixture',
                'feedback':{'diagnostics':['fixture public diagnostic']}}
remote_public.RemotePublicSpectre = FixtureRemote
root=Path(sys.argv[1])
assert session_action(root,{'action_id':'initial','tool':'evas_write',
    'arguments':{'path':'dut.va','content':'before feedback'}})['ok']
request={'action_id':'simulation','tool':'evas_simulate','arguments':{}}
reply=session_action(root,request)
assert reply['result']['diagnostics'] == ['fixture public diagnostic']
assert 'reward' not in reply['result'] and 'score' not in reply['result']
assert session_action(root,request) == reply
assert FixtureRemote.submissions == 1
assert session_action(root,{'action_id':'edit','tool':'evas_write',
    'arguments':{'path':'dut.va','content':'after feedback'}})['ok']
assert session_action(root,{'action_id':'submit','tool':'evas_submit','arguments':{}})['ok']
"""
    trial = make_trial(
        tmp_path,
        script,
        environment_class=RemotePublicFixtureEnvironment,
        verifier_class=RemotePublicFixtureVerifier,
    )
    package = task_package(tmp_path / "public-package", purpose="public")
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["feedback_fields"] = ["diagnostics"]
    (package / "manifest.json").write_text(json.dumps(manifest))
    config = trial.agent_environment.settings.model_dump(mode="json")
    config.update(
        public_backend="remote_spectre",
        image=None,
        checkout=None,
        kernel=None,
        public_task_package=str(package),
        task={
            "task_id": "fixture-task",
            "task_version": "fixture-v1",
            "public_files": ["instruction.md"],
            "candidate_files": ["dut.va"],
            "feedback_fields": ["diagnostics"],
            "manifest": {"condition_id": "fixture-condition-v1"},
        },
        public_remote=dict(
            host="fixture",
            python="/python",
            bundle="/bundle",
            profile="/public/profile",
            run_root="/public/jobs",
            archive_root="/public/archive",
            upload_root="/public/upload",
        ),
    )
    trial.agent_environment.settings = HarborChipsConfig.model_validate(config)
    result = asyncio.run(trial.run())
    assert result.verifier_result.rewards == {"reward": 0.5}
    assert trial.agent_environment.frozen["state"] == "submitted"


def test_configuration_b_rejects_overlapping_final_material_and_storage(tmp_path):
    from alphaapollo.workflows.harbor_chips.config import HarborChipsConfig

    data = dict(
        task={},
        materials=tmp_path,
        image=None,
        public_backend="remote_spectre",
        public_task_package=tmp_path / "public-package",
        final_task_package=tmp_path / "final-package",
        public_remote=dict(
            host="fixture",
            profile="/public/profile",
            run_root="/public/jobs",
            archive_root="/public/archive",
            upload_root="/public/upload",
        ),
        final_remote=dict(
            host="fixture",
            profile="/final/profile",
            run_root="/final/jobs",
            archive_root="/final/archive",
            upload_root="/final/upload",
        ),
    )
    assert HarborChipsConfig.model_validate(data).kernel is None
    with pytest.raises(ValueError, match="separate"):
        HarborChipsConfig.model_validate(
            {**data, "final_task_package": data["public_task_package"]}
        )
    with pytest.raises(ValueError, match="separate"):
        HarborChipsConfig.model_validate({**data, "final_remote": data["public_remote"]})
    cross_overlap = {**data["public_remote"], "archive_root": data["final_remote"]["run_root"]}
    with pytest.raises(ValueError, match="separate"):
        HarborChipsConfig.model_validate({**data, "public_remote": cross_overlap})


@pytest.mark.parametrize(
    "path", ["/public/../final/jobs", "/public//jobs", "/public/./jobs", "relative/jobs"]
)
def test_configuration_b_rejects_noncanonical_remote_roots(tmp_path, path):
    from alphaapollo.workflows.harbor_chips.config import HarborChipsConfig

    data = dict(
        task={},
        materials=tmp_path,
        image=None,
        public_backend="remote_spectre",
        public_task_package=tmp_path / "public-package",
        final_task_package=tmp_path / "final-package",
        public_remote=dict(
            host="fixture",
            profile="/public/profile",
            run_root=path,
            archive_root="/public/archive",
            upload_root="/public/upload",
        ),
        final_remote=dict(
            host="fixture",
            profile="/final/profile",
            run_root="/final/jobs",
            archive_root="/final/archive",
            upload_root="/final/upload",
        ),
    )
    with pytest.raises(ValueError, match="canonical"):
        HarborChipsConfig.model_validate(data)
