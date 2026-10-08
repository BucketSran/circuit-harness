"""Real local checker processes, constructed task/simulator fixtures; no lab claims."""

import json
import shlex
import sys
import time

import pytest

from alphaapollo.common.execution.chips import benchmark_spectre
from alphaapollo.common.execution.chips.journal import file_digest


def task_package(root, *, purpose="final", checker="exit 0\n"):
    package = root / "task"
    (package / "tests").mkdir(parents=True)
    script = package / "tests/test.sh"
    script.write_text("#!/bin/sh\n" + checker)
    manifest = {
        "schema_version": 1,
        "task_id": "fixture-task",
        "task_version": "fixture-v1",
        "criteria_sha256": "a" * 64,
        "condition_id": "fixture-condition-v1",
        "task_set": "public",
        "purpose": purpose,
        "entrypoint": "tests/test.sh",
        "candidate_file": "dut.va",
        "report_path": "verifier/report.json",
        "files": {"tests/test.sh": {"sha256": file_digest(script), "bytes": script.stat().st_size}},
        "feedback_fields": ["measurement"] if purpose == "public" else [],
    }
    (package / "manifest.json").write_text(json.dumps(manifest))
    return package


def test_package_rejects_final_material_for_public_execution(tmp_path):
    package = task_package(tmp_path)
    with pytest.raises(ValueError, match="purpose"):
        benchmark_spectre.package_identity(package, purpose="public")


def test_task_package_hashes_and_feedback_policy(tmp_path):
    package = task_package(tmp_path)
    assert benchmark_spectre.package_identity(package, purpose="final")["sha256"]
    (package / "tests/test.sh").write_text("changed")
    with pytest.raises(ValueError, match="modified"):
        benchmark_spectre.package_identity(package, purpose="final")


def test_ambiguous_checker_failure_is_not_a_model_zero():
    report = {
        "status": "completed",
        "reward": 0,
        "cases": [{"status": "simulation_timeout", "passed": False}],
    }
    result = benchmark_spectre.project_report(report, purpose="final", feedback_fields=[])
    assert result["execution"] == "unclassified_failure"
    assert result["score"] is None


def inputs(root, *, report=None, checker=None, purpose="final", timeout=3):
    from alphaapollo.common.execution.chips.candidate_bundle import freeze_candidate

    source = root / "source"
    source.mkdir()
    (source / "dut.va").write_text("module dut; endmodule\n")
    candidate = root / "frozen"
    freeze_candidate(
        source,
        candidate,
        ["dut.va"],
        task_id="fixture-task",
        task_version="fixture-v1",
        reason="operator request",
    )
    report = report or {
        "status": "completed",
        "reward": 1,
        "cases": [{"status": "graded", "passed": True}],
    }
    if checker is None:
        code = "\n".join(
            [
                "import os,json,hashlib,pathlib",
                "out=pathlib.Path(os.environ['VERIFY_OUTPUT']);out.mkdir(parents=True,exist_ok=True)",
                "candidate=pathlib.Path(os.environ['CANDIDATE'])",
                f"report=json.loads({json.dumps(report)!r})",
                "report['candidate_sha256']=hashlib.sha256(candidate.read_bytes()).hexdigest()",
                "(out/'report.json').write_text(json.dumps(report))",
            ]
        )
        checker = shlex.join([sys.executable, "-c", code]) + "\n"
    package = task_package(root, purpose=purpose, checker=checker)
    for name in ("jobs", "archive"):
        (root / name).mkdir(mode=0o700)
    simulator = root / "spectre-fixture"
    simulator.write_text("#!/bin/sh\nexit 0\n")
    simulator.chmod(0o700)
    preflight = root / "preflight-fixture.sh"
    preflight_code = "\n".join(
        [
            "import os,json,pathlib",
            "report={'schema_version':1,'checks':[]}",
            "for kind in ('license','dependency'):",
            "    report['checks'].append(dict(kind=kind,name='fixture '+kind,"
            "status='passed',detail='constructed evidence'))",
            "pathlib.Path(os.environ['PREFLIGHT_OUTPUT']).write_text(json.dumps(report))",
        ]
    )
    preflight.write_text("#!/bin/sh\n" + shlex.join([sys.executable, "-c", preflight_code]) + "\n")
    profile = root / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shell": "/bin/sh",
                "setup_scripts": [],
                "spectre": str(simulator),
                "preflight_script": str(preflight),
                "run_root": str(root / "jobs"),
                "archive_root": str(root / "archive"),
                "timeout_s": timeout,
                "max_output_bytes": 1024 * 1024,
            }
        )
    )
    profile.chmod(0o600)
    return candidate, package, profile


def finish(directory):
    from alphaapollo.common.execution.chips.jobs import inspect_job

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = inspect_job(directory)
        if state["state"] == "finished" and state.get("archive", {}).get("state") == "verified":
            return state
        time.sleep(0.05)
    pytest.fail(f"job did not complete: {state}")


def test_detached_frozen_checker_and_deduplicated_archive(tmp_path):
    from alphaapollo.common.execution.chips.archive import verify_archive
    from alphaapollo.common.execution.chips.jobs import submit_benchmark_spectre, verify_job

    candidate, package, profile = inputs(tmp_path)
    ack = submit_benchmark_spectre(candidate, package, profile, "attempt")
    assert ack["state"] in {"running", "finished"}
    directory = tmp_path / "jobs/attempt"
    state = finish(directory)
    assert state["result"]["score"] == 1
    assert (
        state["result"]["candidate_sha256"]
        == json.loads((candidate / "manifest.json").read_text())["candidate_sha256"]
    )
    assert verify_job(directory)["result"] == state["result"]
    assert verify_archive(tmp_path / "archive/attempt")["completion"]["result"] == state["result"]
    assert submit_benchmark_spectre(candidate, package, profile, "attempt") == state | {
        "directory": str(directory)
    }


@pytest.mark.parametrize(
    "report,checker,timeout,execution,score",
    [
        (
            {"status": "completed", "reward": 0, "cases": [{"status": "graded", "passed": False}]},
            None,
            3,
            "ok",
            0,
        ),
        ({"status": "infrastructure_error", "reward": 0}, None, 3, "infrastructure_error", None),
        (
            {
                "status": "completed",
                "reward": 0,
                "cases": [{"status": "compile_or_simulation_failure", "passed": False}],
            },
            None,
            3,
            "unclassified_failure",
            None,
        ),
        (None, "sleep 5\n", 0.2, "timeout", None),
        (
            None,
            'mkdir -p "$VERIFY_OUTPUT"; echo broken > "$VERIFY_OUTPUT/report.json"\n',
            3,
            "invalid_result",
            None,
        ),
    ],
)
def test_real_process_failure_classification(tmp_path, report, checker, timeout, execution, score):
    from alphaapollo.common.execution.chips.jobs import submit_benchmark_spectre, verify_job

    args = inputs(tmp_path, report=report, checker=checker, timeout=timeout)
    submit_benchmark_spectre(*args, "classification")
    state = finish(tmp_path / "jobs/classification")
    assert state["result"]["execution"] == execution
    assert state["result"]["score"] == score
    assert verify_job(tmp_path / "jobs/classification")["result"] == state["result"]


def test_frozen_drift_and_unknown_never_restart(tmp_path):
    from alphaapollo.common.execution.chips.archive import reserve_archive
    from alphaapollo.common.execution.chips.jobs import submit_benchmark_spectre
    from alphaapollo.common.execution.chips.journal import atomic_json

    args = inputs(tmp_path)
    identity = benchmark_spectre.benchmark_identity(*args)
    directory = tmp_path / "jobs/lost"
    reserve_archive(tmp_path / "archive/lost", directory, identity)
    directory.mkdir(mode=0o700)
    atomic_json(
        directory / "request.json",
        {"identity": identity, "archive_directory": str(tmp_path / "archive/lost")},
    )
    assert submit_benchmark_spectre(*args, "lost")["state"] == "unknown"
    assert not (directory / "worker.json").exists()
    (args[0] / "files/dut.va").write_text("changed")
    with pytest.raises(ValueError):
        submit_benchmark_spectre(*args, "another")
    assert not (tmp_path / "jobs/another").exists()


def test_public_feedback_does_not_include_final_fields(tmp_path):
    from alphaapollo.common.execution.chips.jobs import submit_benchmark_spectre

    args = inputs(
        tmp_path,
        purpose="public",
        report={
            "status": "completed",
            "reward": 1,
            "measurement": {"gain": 2},
            "cases": ["hidden"],
        },
    )
    submit_benchmark_spectre(*args, "public", purpose="public")
    result = finish(tmp_path / "jobs/public")["result"]
    assert result["score"] is None
    assert benchmark_spectre.public_feedback(result)["feedback"] == {"measurement": {"gain": 2}}
    with pytest.raises(ValueError):
        benchmark_spectre.public_feedback({"purpose": "final"})


def test_archive_tamper_is_rejected(tmp_path):
    from alphaapollo.common.execution.chips.archive import verify_archive
    from alphaapollo.common.execution.chips.jobs import submit_benchmark_spectre

    submit_benchmark_spectre(*inputs(tmp_path), "sealed")
    finish(tmp_path / "jobs/sealed")
    archive = tmp_path / "archive/sealed"
    with (archive / "job.tar.gz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="modified"):
        verify_archive(archive)


def test_public_nested_final_report_is_rejected():
    report = {"status": "completed", "measurement": {"report": {"reward": 1}}}
    result = benchmark_spectre.project_report(
        report, purpose="public", feedback_fields=["measurement"]
    )
    assert result["execution"] == "invalid_result"


def test_bundle_operator_transfer_submit_query_retrieve(tmp_path):
    from alphaapollo.common.execution.chips.benchmark_remote import RemoteBenchmarkSpectre
    from alphaapollo.common.execution.chips.bundle import build_cli
    from alphaapollo.common.execution.chips.session_transport import LocalSessionTransport

    class FixtureTransport(RemoteBenchmarkSpectre):
        cli = LocalSessionTransport.cli
        download = LocalSessionTransport.download

    args = inputs(tmp_path)
    bundle = tmp_path / "cli.pyz"
    build_cli(bundle)
    config = {
        "host": "fixture",
        "python": sys.executable,
        "bundle": str(bundle),
        "profile": str(args[2]),
        "run_root": str(tmp_path / "jobs"),
        "archive_root": str(tmp_path / "archive"),
        "upload_root": str(tmp_path / "uploads"),
    }
    transport = FixtureTransport(config, tmp_path / "operator-evidence")
    assert transport.submit(*args[:2], "remote")["state"] in {"running", "finished"}
    finish(tmp_path / "jobs/remote")
    assert transport.query("remote")["state"] == "finished"
    assert transport.retrieve("remote")["score"] == 1
    assert transport.submit(*args[:2], "remote")["state"] == "finished"
    transport.interrupt_wait()
    with pytest.raises(ConnectionError):
        transport.query("remote")


def test_transfer_rejects_traversal_before_execution(tmp_path):
    from alphaapollo.common.execution.chips.benchmark_remote import stage_transfer

    with pytest.raises(ValueError, match="unsafe"):
        stage_transfer(
            tmp_path / "uploads", {"candidate": {"../escape": "YQ=="}, "task-package": {}}
        )
    assert not (tmp_path / "escape").exists()


def test_package_requires_explicit_comparison_identity(tmp_path):
    package = task_package(tmp_path)
    manifest = json.loads((package / "manifest.json").read_text())
    del manifest["criteria_sha256"]
    (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        benchmark_spectre.package_identity(package, purpose="final")


@pytest.mark.parametrize(
    "report,execution,score",
    [
        ({"status": "submission_contract_violation", "reward": 0}, "ok", 0),
        (
            {"status": "completed", "reward": 0, "cases": [{"status": "graded", "passed": False}]},
            "ok",
            0,
        ),
        (
            {
                "status": "completed",
                "reward": 0,
                "cases": [{"status": "compile_or_simulation_failure", "passed": False}],
            },
            "unclassified_failure",
            None,
        ),
    ],
)
def test_nonzero_checker_exit_preserves_explicit_task_judgment(tmp_path, report, execution, score):
    from alphaapollo.common.execution.chips.jobs import submit_benchmark_spectre

    args = inputs(tmp_path, report=report)
    script = args[1] / "tests/test.sh"
    script.write_text(script.read_text() + "exit 1\n")
    manifest = json.loads((args[1] / "manifest.json").read_text())
    manifest["files"]["tests/test.sh"] = {
        "sha256": file_digest(script),
        "bytes": script.stat().st_size,
    }
    (args[1] / "manifest.json").write_text(json.dumps(manifest))
    submit_benchmark_spectre(*args, "nonzero")
    result = finish(tmp_path / "jobs/nonzero")["result"]
    assert result["process"]["returncode"] == 1
    assert result["execution"] == execution
    assert result["score"] == score


@pytest.mark.parametrize(
    "kind,status,exit_code,execution",
    [
        ("license", "failed", 0, "license_unavailable"),
        ("dependency", "failed", 0, "dependency_unavailable"),
        ("license", "unknown", 0, "preflight_unknown"),
        ("license", "passed", 1, "invalid_preflight"),
    ],
)
def test_preflight_failure_blocks_checker_without_zero(
    tmp_path, kind, status, exit_code, execution
):
    from alphaapollo.common.execution.chips.jobs import submit_benchmark_spectre, verify_job

    args = inputs(tmp_path, checker="echo checker-ran\n")
    report = {
        "schema_version": 1,
        "checks": [
            {
                "kind": name,
                "name": "fixture " + name,
                "status": status if name == kind else "passed",
                "detail": "explicit failure",
            }
            for name in ("license", "dependency")
        ],
    }
    code = (
        "import os,pathlib;pathlib.Path(os.environ['PREFLIGHT_OUTPUT']).write_text("
        + repr(json.dumps(report))
        + ")"
    )
    (tmp_path / "preflight-fixture.sh").write_text(
        shlex.join([sys.executable, "-c", code]) + f"\nexit {exit_code}\n"
    )
    submit_benchmark_spectre(*args, "preflight-fails")
    directory = tmp_path / "jobs/preflight-fails"
    state = finish(directory)
    assert state["result"]["execution"] == execution
    assert state["result"]["score"] is None
    assert not (directory / "run/checker.stdout.log").exists()
    if exit_code == 0:
        assert state["result"]["preflight"]["checks"] == report["checks"]
    assert verify_job(directory)["result"] == state["result"]


@pytest.mark.parametrize(
    "script,timeout,execution",
    [
        ("exit 0\n", 3, "invalid_preflight"),
        ('echo broken > "$PREFLIGHT_OUTPUT"\n', 3, "invalid_preflight"),
        ("sleep 5\n", 0.25, "timeout"),
        (shlex.join([sys.executable, "-c", "print('x'*2000000)"]) + "\n", 3, "output_limit"),
    ],
)
def test_preflight_incomplete_or_bounded_failure_blocks_checker(
    tmp_path, script, timeout, execution
):
    from alphaapollo.common.execution.chips.jobs import submit_benchmark_spectre

    args = inputs(tmp_path, checker="echo checker-ran\n", timeout=timeout)
    (tmp_path / "preflight-fixture.sh").write_text(script)
    submit_benchmark_spectre(*args, "preflight-bounds")
    directory = tmp_path / "jobs/preflight-bounds"
    result = finish(directory)["result"]
    assert result["execution"] == execution
    assert result["score"] is None
    assert result["preflight_process"]["cleanup_confirmed"]
    assert not (directory / "run/checker.stdout.log").exists()


def test_preflight_and_checker_share_one_budget(tmp_path):
    from alphaapollo.common.execution.chips.jobs import submit_benchmark_spectre

    args = inputs(tmp_path, checker="sleep .6\n", timeout=1)
    script = tmp_path / "preflight-fixture.sh"
    script.write_text("sleep .6\n" + script.read_text())
    submit_benchmark_spectre(*args, "shared-budget")
    result = finish(tmp_path / "jobs/shared-budget")["result"]
    assert result["preflight"]["execution"] == "ok"
    assert result["execution"] == "timeout"
    assert result["process"]["elapsed_s"] < 0.6
    assert result["score"] is None


def test_preflight_cancellation_and_pinned_probe_source(tmp_path):
    from alphaapollo.common.execution.chips.jobs import cancel_job, submit_benchmark_spectre

    args = inputs(tmp_path, checker="echo checker-ran\n")
    script = tmp_path / "preflight-fixture.sh"
    script.write_text("sleep 5\n")
    identity = benchmark_spectre.benchmark_identity(*args)
    assert identity["profile"]["tools"][str(script)] == file_digest(script)
    submit_benchmark_spectre(*args, "cancel-preflight")
    directory = tmp_path / "jobs/cancel-preflight"
    cancel_job(directory)
    result = finish(directory)["result"]
    assert result["execution"] == "cancelled"
    assert result["score"] is None
    assert not (directory / "run/checker.stdout.log").exists()
    staging = tmp_path / "staging"
    staging.mkdir()
    benchmark_spectre.stage_inputs(*args[:2], staging, identity)
    script.write_text("changed probe")
    with pytest.raises(ValueError, match="changed"):
        benchmark_spectre.run_benchmark(identity, staging / "run")


def test_public_agent_profile_requires_container_not_host_scripts(tmp_path):
    candidate, package, profile = inputs(tmp_path, purpose="public")
    with pytest.raises(ValueError, match="isolated Docker"):
        benchmark_spectre.benchmark_identity(
            candidate, package, profile, purpose="public", isolated_public=True
        )


@pytest.mark.parametrize(
    "mode", ["completed", "preflight_cancelled", "checker_cancelled", "output_limit"]
)
def test_public_job_runs_real_container_with_only_declared_inputs(tmp_path, mode):
    import os

    from alphaapollo.common.execution.chips.jobs import (
        cancel_job,
        inspect_job,
        submit_benchmark_spectre,
    )

    image = os.environ.get("CHIPS_TEST_DOCKER_IMAGE")
    if not image:
        pytest.skip("declare immutable local Docker image; no pulls")
    secret = tmp_path / "private-final-secret"
    secret.write_text("not mounted into public execution")
    code = "\n".join(
        [
            "import os,json,pathlib,hashlib,socket",
            "candidate=pathlib.Path(os.environ['CANDIDATE'])",
            f"assert not pathlib.Path({str(secret)!r}).exists()",
            "try: candidate.write_text('changed')",
            "except OSError: pass",
            "else: raise AssertionError('candidate writable')",
            "try: pathlib.Path('/task/tests/test.sh').write_text('changed')",
            "except OSError: pass",
            "else: raise AssertionError('task writable')",
            "assert len(list(pathlib.Path('/sys/class/net').iterdir())) == 1",
            "out=pathlib.Path(os.environ['VERIFY_OUTPUT']);out.mkdir(parents=True,exist_ok=True)",
            "report={'status':'completed','measurement':42,'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()}",
            "(out/'report.json').write_text(json.dumps(report))",
        ]
    )
    if mode == "checker_cancelled":
        code += "\npathlib.Path('/output/started').touch();import time;time.sleep(60)"
    if mode == "output_limit":
        code += "\nfor i in range(100): pathlib.Path('/output/file'+str(i)).write_bytes(b'x'*65536)"
    candidate, package, profile = inputs(
        tmp_path,
        purpose="public",
        checker=shlex.join(["/usr/local/bin/python3", "-c", code]) + "\n",
        timeout=30,
    )
    probe = package / "tests/probe.sh"
    probe.write_text(
        "#!/bin/sh\n"
        + shlex.join(
            [
                "/usr/local/bin/python3",
                "-c",
                "import json,os,pathlib;"
                f"assert not pathlib.Path({str(secret)!r}).exists();"
                "pathlib.Path(os.environ['PREFLIGHT_OUTPUT']).write_text(json.dumps("
                "{'schema_version':1,'checks':[{'kind':kind,'name':'synthetic '+kind,"
                "'status':'passed','detail':'fixture only'} "
                "for kind in ('license','dependency')]}))",
            ]
        )
        + "\n"
    )
    if mode == "preflight_cancelled":
        probe.write_text(
            probe.read_text() + "/usr/local/bin/python3 -c 'import time;time.sleep(60)'\n"
        )
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["files"]["tests/probe.sh"] = {
        "sha256": file_digest(probe),
        "bytes": probe.stat().st_size,
    }
    (package / "manifest.json").write_text(json.dumps(manifest))
    profile.write_text(
        json.dumps(
            dict(
                schema_version=1,
                backend="docker",
                image=image,
                spectre="/bin/true",
                preflight_script="/task/tests/probe.sh",
                run_root=str(tmp_path / "jobs"),
                archive_root=str(tmp_path / "archive"),
                timeout_s=30,
                max_output_bytes=1024 * 1024,
            )
        )
    )
    profile.chmod(0o600)
    reply = submit_benchmark_spectre(
        candidate, package, profile, "public-docker-fixture", purpose="public", isolated_public=True
    )
    job = tmp_path / "jobs/public-docker-fixture"
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        reply = inspect_job(job)
        if mode == "preflight_cancelled" and (job / "run/backend.json").exists():
            cancel_job(job)
        elif mode == "checker_cancelled" and (job / "run/work/started").exists():
            cancel_job(job)
        if reply.get("state") == "finished":
            break
        time.sleep(0.1)
    assert reply["state"] == "finished", reply
    result = json.loads((job / "completion.json").read_text())["result"]
    expected = "cancelled" if mode.endswith("cancelled") else "ok" if mode == "completed" else mode
    assert result["execution"] == expected, result
    if mode == "completed":
        assert result["feedback"] == {"measurement": 42}
    assert result["score"] is None
    assert result["process"]["cleanup_confirmed"]
