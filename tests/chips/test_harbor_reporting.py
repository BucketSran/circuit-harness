"""Offline public report CLI with native Harbor records and sealed local fixtures."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("harbor")
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from test_benchmark_spectre import finish, inputs

from alphaapollo.common.execution.chips.jobs import submit_benchmark_spectre


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture_job(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    task = tmp_path / "harbor-task"
    task.mkdir()
    public = tmp_path / "public.json"
    write(
        public,
        dict(
            task={
                "task_id": "fixture-task",
                "task_version": "fixture-v1",
                "public_files": ["instruction.md"],
                "candidate_files": ["dut.va"],
            },
            materials=str(task),
            checkout=str(task),
            kernel=str(task / "kernel"),
            image="fixture@sha256:" + "b" * 64,
            max_actions=8,
            max_simulations=2,
        ),
    )
    config = JobConfig(
        job_name="fixture",
        n_attempts=6,
        tasks=[{"path": str(task)}],
        agents=[{"name": "pi", "model_name": "openai/fixture-model"}],
        environment={
            "import_path": (
                "alphaapollo.workflows.harbor_chips.docker_environment:CircuitDockerEnvironment"
            ),
            "kwargs": {"session_config": str(public)},
        },
    )
    write(job / "config.json", config.model_dump(mode="json"))
    result = JobResult(
        id=uuid4(), started_at=datetime(2026, 1, 1), n_total_trials=6, stats=JobStats()
    )
    write(job / "result.json", result.model_dump(mode="json"))
    for index, status in enumerate(("positive", "zero", "failed", "missing", "running")):
        directory = job / f"trial-{index}"
        trial_config = TrialConfig(
            task=config.tasks[0],
            agent=config.agents[0],
            environment=config.environment,
            verifier=config.verifier,
            job_id=result.id,
            trial_name=directory.name,
            trials_dir=job,
        )
        write(directory / "config.json", trial_config.model_dump(mode="json"))
        if status == "missing":
            continue
        trial = TrialResult(
            task_name=task.name,
            trial_name=directory.name,
            trial_uri=directory.as_uri(),
            task_id=config.tasks[0].get_task_id(),
            task_checksum="c" * 64,
            config=trial_config,
            agent_info=AgentInfo(name="pi", version="fixture-v1"),
            started_at=datetime(2026, 1, 1),
            finished_at=None if status == "running" else datetime(2026, 1, 1, 0, 1),
        )
        if status in ("positive", "zero"):
            setup = tmp_path / f"checker-{index}"
            setup.mkdir()
            code = (
                "import os,json,pathlib,hashlib; p=pathlib.Path(os.environ['CANDIDATE']); "
                "ok=b'zero' not in p.read_bytes(); out=pathlib.Path(os.environ['VERIFY_OUTPUT']); "
                "out.mkdir(parents=True,exist_ok=True); "
                "(out/'report.json').write_text(json.dumps(dict(status='completed',reward=int(ok),"
                "cases=[dict(status='graded',passed=ok)],"
                "candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))"
            )
            candidate, package, profile = inputs(
                setup, checker=shlex.join([sys.executable, "-c", code]) + "\n"
            )
            if status == "zero":
                from alphaapollo.common.execution.chips.candidate_bundle import freeze_candidate

                shutil.rmtree(candidate)
                (setup / "source/dut.va").write_text("module zero; endmodule\n")
                freeze_candidate(
                    setup / "source",
                    candidate,
                    ["dut.va"],
                    task_id="fixture-task",
                    task_version="fixture-v1",
                    reason="fixture",
                )
            shared_profile = tmp_path / "checker-0/profile.json"
            job_id = f"report-fixture-{index}"
            submit_benchmark_spectre(candidate, package, shared_profile, job_id)
            final = finish(tmp_path / "checker-0/jobs" / job_id)["result"]
            archive = directory / "verifier/transport/report-fixture/archive"
            archive.mkdir(parents=True)
            for name in ("receipt.json", "job.tar.gz"):
                shutil.copyfile(tmp_path / "checker-0/archive" / job_id / name, archive / name)

            shutil.copytree(candidate, directory / "public-session/candidate")
            write(
                directory / "public-session/frozen.json",
                dict(
                    state="submitted",
                    candidate_directory=str(directory / "public-session/candidate"),
                    candidate_sha256=final["candidate_sha256"],
                    collection_source="agent_submit",
                ),
            )
            write(
                directory / "public-session/session.json",
                dict(
                    task={"task_id": "fixture-task", "task_version": "fixture-v1"},
                    backend="docker",
                    image="fixture@sha256:" + "b" * 64,
                    max_actions=8,
                    max_simulations=2,
                    timeout_s=120,
                    max_output_bytes=16777216,
                ),
            )
            write(directory / "verifier/evaluation.json", {"state": "completed", **final})
            trial.verifier_result = VerifierResult(rewards={"reward": final["score"]})
        elif status == "failed":
            trial.exception_info = ExceptionInfo(
                exception_type="EnvironmentError",
                exception_message="fixture unavailable",
                exception_traceback="",
                occurred_at=datetime(2026, 1, 1),
            )
        write(directory / "result.json", trial.model_dump(mode="json"))
    final_config = tmp_path / "final.json"
    write(
        final_config,
        {
            "task_package": str(tmp_path / "checker-0/task"),
            "remote": {"host": "fixture", "profile": str(tmp_path / "checker-0/profile.json")},
        },
    )
    config.verifier.kwargs["config_path"] = str(final_config)
    write(job / "config.json", config.model_dump(mode="json"))
    for path in job.glob("*/config.json"):
        raw = json.loads(path.read_text())
        raw["verifier"] = config.verifier.model_dump(mode="json")
        write(path, raw)
        result_path = path.parent / "result.json"
        if result_path.exists():
            raw = json.loads(result_path.read_text())
            raw["config"]["verifier"] = config.verifier.model_dump(mode="json")
            write(result_path, raw)
    return job


def report(job, output):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "alphaapollo.workflows.harbor_chips.reporting",
            "--job",
            str(job),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )


def test_report_keeps_all_planned_attempts_and_valid_zero(tmp_path):
    job = fixture_job(tmp_path)
    output = tmp_path / "report"
    result = report(job, output)
    assert result.returncode == 0, result.stderr
    data = json.loads((output / "report.json").read_text())
    assert [row["status"] for row in data["records"]] == [
        "graded",
        "graded",
        "failed",
        "missing",
        "running",
        "not_run",
    ]
    assert [row["score"] for row in data["records"]] == [1, 0, None, None, None, None]
    assert data["groups"][0]["planned"] == 6
    assert data["groups"][0]["started"] == 4
    assert data["groups"][0]["metric_valid"] == 2
    assert data["groups"][0]["coverage"] == pytest.approx(1 / 3)
    assert data["groups"][0]["score_denominator"] == 2
    assert data["groups"][0]["mean_score"] == 0.5
    assert data["records"][0]["usage"]["cost_usd"] is None


def test_report_rejects_final_package_drift_before_using_scores(tmp_path):
    job = fixture_job(tmp_path)
    from test_benchmark_spectre import task_package

    other = tmp_path / "other-final"
    other.mkdir()
    package = task_package(other, checker="echo changed\n")
    write(tmp_path / "final.json", {"task_package": str(package), "remote": {"host": "fixture"}})
    output = tmp_path / "report"
    result = report(job, output)
    assert result.returncode == 1
    data = json.loads((output / "report.json").read_text())
    assert data["groups"][0]["metric_valid"] == 0
    assert all(row["status"] == "invalid" for row in data["records"][:2])


def test_report_never_reads_host_keys_or_starts_a_process(tmp_path):
    job = fixture_job(tmp_path)
    for path in [job / "config.json", *job.glob("*/config.json")]:
        raw = json.loads(path.read_text())
        agents = raw["agents"] if path.parent == job else [raw["agent"]]
        for agent in agents:
            agent["env"] = {"OPENAI_API_KEY": "fixture-literal-never-use"}
        raw["environment"]["env"] = {"PRIVATE_AUTH_TOKEN": "fixture-literal-never-use"}
        write(path, raw)
        trial_result = path.parent / "result.json"
        if path.parent != job and trial_result.exists():
            result = json.loads(trial_result.read_text())
            result["config"] = raw
            write(trial_result, result)
    before = {
        str(path.relative_to(job)): path.read_bytes() for path in job.rglob("*") if path.is_file()
    }
    code = """
import os,socket,subprocess,sys
from alphaapollo.workflows.harbor_chips.reporting import main
original_get=os.environ.get
original_item=type(os.environ).__getitem__
def get(key,*args):
    if key in ('OPENAI_API_KEY','PRIVATE_AUTH_TOKEN'):
        raise AssertionError('report read host credential')
    return original_get(key,*args)
def item(self,key):
    if key in ('OPENAI_API_KEY','PRIVATE_AUTH_TOKEN'):
        raise AssertionError('report read host credential')
    return original_item(self,key)
def forbidden(*args,**kwargs):
    raise AssertionError('report attempted process/network execution')
os.environ.get=get
type(os.environ).__getitem__=item
socket.socket=forbidden
subprocess.Popen=forbidden
sys.exit(main(sys.argv[1:]))
"""
    output = tmp_path / "report"
    result = subprocess.run(
        [sys.executable, "-c", code, "--job", str(job), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    after = {
        str(path.relative_to(job)): path.read_bytes() for path in job.rglob("*") if path.is_file()
    }
    assert after == before
    assert "fixture-literal-never-use" not in (output / "report.json").read_text()


@pytest.mark.parametrize(
    "damage",
    ["stale-evaluation", "candidate-task", "corrupt-archive", "duplicate-id", "duplicate-trial"],
)
def test_report_fails_closed_on_bad_evidence(tmp_path, damage):
    job = fixture_job(tmp_path)
    if damage == "stale-evaluation":
        path = job / "trial-0/verifier/evaluation.json"
        value = json.loads(path.read_text())
        value["score"] = 0
        write(path, value)
    elif damage == "candidate-task":
        path = job / "trial-0/public-session/session.json"
        value = json.loads(path.read_text())
        value["task"]["task_version"] = "wrong"
        write(path, value)
    elif damage == "corrupt-archive":
        path = job / "trial-0/verifier/transport/report-fixture/archive/job.tar.gz"
        path.write_bytes(b"corrupt")
    elif damage == "duplicate-id":
        first = json.loads((job / "trial-0/result.json").read_text())
        path = job / "trial-1/result.json"
        value = json.loads(path.read_text())
        value["id"] = first["id"]
        write(path, value)
    else:
        source = job / "trial-0"
        target = job / "trial-duplicate"
        shutil.copytree(source, target)
        path = target / "config.json"
        value = json.loads(path.read_text())
        value["trial_name"] = target.name
        write(path, value)
        path = target / "result.json"
        result = json.loads(path.read_text())
        result["trial_name"] = target.name
        result["config"] = value
        write(path, result)
        # More copies than n_attempts cannot be a native planned attempt.
        for i in range(2):
            shutil.copytree(target, job / f"trial-excess-{i}")
    result = report(job, tmp_path / "report")
    assert result.returncode == 1, result.stderr
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["groups"][0]["metric_valid"] < 2
    assert any(row["status"] == "invalid" and row["score"] is None for row in data["records"])


def test_report_rejects_unplanned_cpu_policy_and_preserves_legacy_default(tmp_path):
    job = fixture_job(tmp_path)
    path = job / "trial-1/public-session/session.json"
    session = json.loads(path.read_text())
    session["cpu_limit"] = None
    write(path, session)

    output = tmp_path / "report"
    assert report(job, output).returncode == 1
    data = json.loads((output / "report.json").read_text())
    legacy, unconstrained = data["records"][:2]
    assert legacy["status"] == "graded"
    assert legacy["actual_conditions"]["public"]["cpu_limit"] == 1
    assert unconstrained["status"] == "invalid"
    assert unconstrained["score"] is None
    assert unconstrained["actual_conditions"]["public"]["cpu_limit"] is None
    assert "actual public budget differs from plan" in unconstrained["errors"][0]
    assert data["groups"][0]["metric_valid"] == 1


@pytest.mark.parametrize("cpu_limit", [None, 0.5])
def test_report_accepts_and_records_explicit_cpu_policy(tmp_path, cpu_limit):
    job = fixture_job(tmp_path)
    path = tmp_path / "public.json"
    public = json.loads(path.read_text())
    public["public_cpu_limit"] = cpu_limit
    write(path, public)
    for path in job.glob("*/public-session/session.json"):
        session = json.loads(path.read_text())
        session["cpu_limit"] = cpu_limit
        write(path, session)

    output = tmp_path / "report"
    assert report(job, output).returncode == 0
    data = json.loads((output / "report.json").read_text())
    assert data["groups"][0]["metric_valid"] == 2
    assert data["groups"][0]["mean_score"] == 0.5
    for row in data["records"][:2]:
        assert row["conditions"]["public_budget"]["cpu_limit"] == cpu_limit
        assert row["actual_conditions"]["public"]["cpu_limit"] == cpu_limit


def test_evaluation_json_alone_is_not_a_trusted_score(tmp_path):
    job = fixture_job(tmp_path)
    shutil.rmtree(job / "trial-0/verifier/transport")
    result = report(job, tmp_path / "report")
    assert result.returncode == 1
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["records"][0]["status"] == "invalid"
    assert data["records"][0]["score"] is None
    assert "independent" in data["records"][0]["errors"][0]


def test_report_retains_agent_timeout_with_independently_valid_score(tmp_path):
    job = fixture_job(tmp_path)
    path = job / "trial-0/result.json"
    value = json.loads(path.read_text())
    value["exception_info"] = dict(
        exception_type="AgentTimeoutError",
        exception_message="deadline",
        exception_traceback="",
        occurred_at="2026-01-01T00:00:00",
    )
    write(path, value)
    assert report(job, tmp_path / "report").returncode == 0
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["records"][0]["status"] == "failed"
    assert data["records"][0]["agent_exception"] == "AgentTimeoutError"
    assert data["records"][0]["score"] == 1
    assert data["groups"][0]["metric_valid"] == 2


def test_report_separates_models_and_rejects_output_overwrite(tmp_path):
    job = fixture_job(tmp_path)
    config = json.loads((job / "config.json").read_text())
    config["agents"].append({"name": "pi", "model_name": "openai/other-model"})
    write(job / "config.json", config)
    result = json.loads((job / "result.json").read_text())
    result["n_total_trials"] = 12
    write(job / "result.json", result)
    output = tmp_path / "report"
    assert report(job, output).returncode == 0
    data = json.loads((output / "report.json").read_text())
    assert len(data["groups"]) == 2
    assert [group["planned"] for group in data["groups"]] == [6, 6]
    assert [group["metric_valid"] for group in data["groups"]] == [2, 0]
    before = (output / "report.json").read_bytes()
    assert report(job, output).returncode == 2
    assert (output / "report.json").read_bytes() == before
    assert report(job, job / "report").returncode == 2
    assert not (job / "report").exists()


def test_report_reuses_task_bindings_for_multitask_plan_and_unsupported_entries(tmp_path):
    from test_benchmark_spectre import task_package
    from test_harbor_task_bindings import suite

    manifest, entries, raw = suite(tmp_path)
    for entry in entries:
        private = Path(entry["final_config"]).parent
        generated = private / "generated"
        generated.mkdir()
        package = task_package(generated)
        data = json.loads((package / "manifest.json").read_text())
        data["task_id"] = entry["task_id"]
        data["task_version"] = "v1"
        write(package / "manifest.json", data)
        write(Path(entry["final_config"]), {"task_package": str(package), "remote": {}})
    third = tmp_path / "unsupported-task"
    third.mkdir()
    entries.append(
        dict(
            task_path=str(third),
            task_id="unsupported-task",
            task_version="v1",
            support="unsupported",
            reason="no calibrated checker",
        )
    )
    write(manifest, {"schema_version": 1, "tasks": entries})
    raw["tasks"].append({"path": str(third)})
    raw.update(
        job_name="multitask", agents=[{"name": "pi", "model_name": "openai/fixture"}], n_attempts=2
    )
    job = tmp_path / "jobs/multitask"
    write(job / "config.json", raw)
    output = tmp_path / "report"
    result = report(job, output)
    assert result.returncode == 0, result.stderr
    data = json.loads((output / "report.json").read_text())
    assert [group["task"]["task_id"] for group in data["groups"]] == ["a", "b", "unsupported-task"]
    assert [group["planned"] for group in data["groups"]] == [2, 2, 2]
    assert data["groups"][2]["conditions"]["unsupported_reason"] == "no calibrated checker"
    assert len(data["records"]) == 6


def test_report_verifies_replay_receipt_even_if_verifier_wrapper_did_not_finish(tmp_path):
    from alphaapollo.common.execution.chips.benchmark_replay import replay_candidate
    from alphaapollo.common.execution.chips.benchmark_spectre import package_identity

    job = fixture_job(tmp_path)
    package = tmp_path / "checker-0/task"
    config = dict(
        schema_version=1,
        image="sha256:" + "a" * 64,
        task_package_sha256=package_identity(package, purpose="final")["sha256"],
        solver_options={},
        unsupported=["controlled unavailable analysis"],
        timeout_s=3,
        max_output_bytes=1048576,
    )
    write(
        tmp_path / "final.json",
        {"backend": "benchmark_opensource", "task_package": str(package), "opensource": config},
    )
    for directory in (job / "trial-0", job / "trial-1"):
        shutil.rmtree(directory / "verifier/transport")
        (directory / "verifier/evaluation.json").unlink()
        replay_candidate(
            directory / "public-session/candidate", package, config, directory / "verifier/replay"
        )
        path = directory / "result.json"
        value = json.loads(path.read_text())
        value["verifier_result"] = None
        value["exception_info"] = dict(
            exception_type="VerifierTimeoutError",
            exception_message="controlled deadline",
            exception_traceback="",
            occurred_at="2026-01-01T00:00:00",
        )
        write(path, value)
    result = report(job, tmp_path / "report")
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["records"][0]["status"] == "failed"
    assert data["records"][0]["evaluation"]["execution"] == "unsupported_analysis"
    assert data["groups"][0]["conditions"]["final_backend"] == "opensource"
    assert data["groups"][0]["metric_valid"] == 0


def test_report_does_not_pool_legacy_and_summary_observation_conditions(tmp_path):
    job = fixture_job(tmp_path)
    path = job / "trial-1/public-session/session.json"
    data = json.loads(path.read_text())
    data["observation_view_version"] = 1
    write(path, data)
    assert report(job, tmp_path / "report").returncode == 1
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["groups"][0]["metric_valid"] == 0
    assert data["groups"][0]["mean_score"] is None
    old, new = [row["actual_conditions"]["public"] for row in data["records"][:2]]
    assert old["observation_view_version"] is None
    assert new["observation_view_version"] == 1
    assert "evas_observe" not in old["public_tools"]
    assert "evas_observe" in new["public_tools"]


def test_report_rejects_conflicting_actual_public_conditions(tmp_path):
    job = fixture_job(tmp_path)
    path = job / "trial-1/public-session/session.json"
    data = json.loads(path.read_text())
    data["image"] = "fixture@sha256:" + "c" * 64
    write(path, data)
    assert report(job, tmp_path / "report").returncode == 1
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["records"][1]["score"] is None
    assert data["records"][1]["status"] == "invalid"


def test_report_refuses_to_merge_different_actual_agents_in_one_planned_cell(tmp_path):
    job = fixture_job(tmp_path)
    for index in (0, 1):
        path = job / f"trial-{index}/result.json"
        data = json.loads(path.read_text())
        data["agent_info"] = {
            "name": "pi",
            "version": f"{index + 1}.0",
            "model_info": {"name": f"model-{chr(65 + index)}", "provider": "openai"},
        }
        write(path, data)
    result = report(job, tmp_path / "report")
    assert result.returncode == 1
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["groups"][0]["planned"] == 6
    assert data["groups"][0]["metric_valid"] == 0
    assert data["groups"][0]["mean_score"] is None
    assert all(row["status"] == "invalid" for row in data["records"][:2])
    assert all(row["score"] is None for row in data["records"][:2])


@pytest.mark.parametrize("field", ["name", "version", "model", "provider", "unknown-model"])
def test_report_checks_each_actual_agent_identity_field(tmp_path, field):
    job = fixture_job(tmp_path)
    for index in (0, 1):
        path = job / f"trial-{index}/result.json"
        data = json.loads(path.read_text())
        actual = {
            "name": "pi",
            "version": "1.0",
            "model_info": {"name": "model-A", "provider": "openai"},
        }
        if index == 1:
            if field in {"name", "version"}:
                actual[field] = "different"
            elif field == "unknown-model":
                actual["model_info"] = None
            else:
                actual["model_info"]["name" if field == "model" else "provider"] = "different"
        data["agent_info"] = actual
        write(path, data)
    result = report(job, tmp_path / "report")
    assert result.returncode == 1
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["groups"][0]["metric_valid"] == 0
    for row in data["records"][:2]:
        assert row["actual_conditions"]["agent"] == row["agent_info"]
        assert row["status"] == "invalid"


def test_report_preserves_consistent_actual_model_names_and_unknown_provider(tmp_path):
    job = fixture_job(tmp_path)
    actual = {
        "name": "pi",
        "version": "1.0",
        "model_info": {"name": "fixture-model", "provider": None},
    }
    for index in (0, 1):
        path = job / f"trial-{index}/result.json"
        data = json.loads(path.read_text())
        data["agent_info"] = actual
        write(path, data)
    result = report(job, tmp_path / "report")
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["groups"][0]["metric_valid"] == 2
    assert data["groups"][0]["mean_score"] == 0.5
    for row in data["records"][:2]:
        assert row["agent_info"] == actual
        assert row["actual_conditions"]["agent"] == actual


def test_report_refuses_to_merge_different_runtime_identity_in_one_planned_cell(tmp_path):
    job = fixture_job(tmp_path)
    for index in (0, 1):
        path = job / f"trial-{index}/public-session/session.json"
        data = json.loads(path.read_text())
        data["kernel_sha256"] = str(index) * 64
        write(path, data)
    result = report(job, tmp_path / "report")
    assert result.returncode == 1
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["groups"][0]["planned"] == 6
    assert data["groups"][0]["metric_valid"] == 0
    assert all(row["status"] == "invalid" for row in data["records"][:2])


@pytest.mark.parametrize(
    "damage", ["job-count", "embedded-result", "corrupt-config", "duplicate-json-key"]
)
def test_report_keeps_corrupt_native_evidence_explicit_and_unscored(tmp_path, damage):
    job = fixture_job(tmp_path)
    if damage == "job-count":
        path = job / "result.json"
        data = json.loads(path.read_text())
        data["n_total_trials"] = 99
        write(path, data)
    elif damage == "embedded-result":
        path = job / "result.json"
        data = json.loads(path.read_text())
        trial = json.loads((job / "trial-0/result.json").read_text())
        trial["task_checksum"] = "z" * 64
        data["trial_results"] = [trial]
        write(path, data)
    elif damage == "corrupt-config":
        (job / "trial-0/config.json").write_text("{corrupt")
    else:
        path = job / "trial-0/result.json"
        text = path.read_text()
        path.write_text(
            text.replace('"task_checksum":', '"task_checksum": "ignored", "task_checksum":', 1)
        )
    result = report(job, tmp_path / "report")
    assert result.returncode == 1, result.stderr
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert any(row["status"] == "invalid" and row["score"] is None for row in data["records"])
    if damage in ("job-count", "embedded-result"):
        assert data["groups"][0]["metric_valid"] == 0


def test_missing_trial_result_keeps_started_unknown(tmp_path):
    job = fixture_job(tmp_path)
    assert report(job, tmp_path / "report").returncode == 0
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["records"][3]["started"] is None
    assert data["groups"][0]["started_unknown"] == 1


def test_report_rejects_trial_budget_drift_from_native_job_plan(tmp_path):
    job = fixture_job(tmp_path)
    path = job / "trial-0/config.json"
    config = json.loads(path.read_text())
    config["agent_timeout_multiplier"] = 2
    write(path, config)
    path = job / "trial-0/result.json"
    data = json.loads(path.read_text())
    data["config"] = config
    write(path, data)
    assert report(job, tmp_path / "report").returncode == 1
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["records"][0]["status"] == "invalid"
    assert data["records"][0]["score"] is None


def test_copied_independent_evaluation_is_not_two_attempts(tmp_path):
    job = fixture_job(tmp_path)
    for name in ("public-session", "verifier"):
        shutil.rmtree(job / "trial-1" / name)
        shutil.copytree(job / "trial-0" / name, job / "trial-1" / name)
    path = job / "trial-1/result.json"
    data = json.loads(path.read_text())
    data["verifier_result"] = {"rewards": {"reward": 1}}
    write(path, data)
    assert report(job, tmp_path / "report").returncode == 1
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["groups"][0]["metric_valid"] == 0
    assert all(row["status"] == "invalid" for row in data["records"][:2])


def test_report_keeps_termination_reason_separate_from_candidate_and_score(tmp_path):
    job = fixture_job(tmp_path)
    frozen = json.loads((job / "trial-0/public-session/frozen.json").read_text())
    write(
        job / "trial-0/public-session/episode-end.json",
        {**frozen, "termination_reason": "model_request_limit"},
    )
    assert report(job, tmp_path / "report").returncode == 0
    data = json.loads((tmp_path / "report/report.json").read_text())
    assert data["records"][0]["termination_reason"] == "model_request_limit"
    assert data["records"][0]["collection_source"] == "agent_submit"
    assert data["records"][0]["score"] == 1
