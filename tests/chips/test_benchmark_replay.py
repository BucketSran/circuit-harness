"""Saved candidate comparison contracts; constructed records are not calibration."""

import pytest

from alphaapollo.common.execution.chips import benchmark_replay


def result(score, *, execution="ok"):
    return {
        "candidate_sha256": "c" * 64,
        "task_id": "fixture-task",
        "task_version": "fixture-v1",
        "criteria_sha256": "a" * 64,
        "condition_id": "fixture-conditions",
        "task_set": "public",
        "execution": execution,
        "score": score,
    }


@pytest.mark.parametrize(
    "opensource,spectre,classification",
    [
        (result(1), result(1), "match"),
        (result(1), result(0), "false_accept"),
        (result(0), result(1), "false_reject"),
        (result(None, execution="infrastructure_error"), result(1), "infra"),
        (result(None, execution="invalid_result"), result(1), "unevaluable"),
        (result(1), None, "not_compared"),
    ],
)
def test_comparison_preserves_denominators(opensource, spectre, classification):
    assert benchmark_replay.compare_results(opensource, spectre) == classification


def test_comparison_rejects_different_criteria():
    other = result(1)
    other["criteria_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="criteria_sha256"):
        benchmark_replay.compare_results(result(1), other)


def replay_inputs(tmp_path, *, checker=None):
    from test_benchmark_spectre import inputs

    from alphaapollo.common.execution.chips.benchmark_spectre import package_identity

    args = inputs(tmp_path, checker=checker)
    package = package_identity(args[1], purpose="final")
    config = {
        "schema_version": 1,
        "image": "sha256:" + "a" * 64,
        "task_package_sha256": package["sha256"],
        "solver_options": {},
        "unsupported": [],
        "timeout_s": 3,
        "max_output_bytes": 1024 * 1024,
    }
    return args[0], args[1], config


def test_unsupported_replay_never_probes_any_backend(tmp_path, monkeypatch):
    import subprocess

    candidate, package, config = replay_inputs(tmp_path)
    config["unsupported"] = ["benchmark-declared unavailable analysis"]

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported replay must not probe Docker, SSH, Spectre or models")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    receipt = benchmark_replay.replay_candidate(candidate, package, config, tmp_path / "replay")
    assert receipt["result"]["score"] is None
    assert receipt["classification"] == "unevaluable"
    assert benchmark_replay.verify_replay(tmp_path / "replay") == receipt
    summary = benchmark_replay.summarize_replays([tmp_path / "replay"])
    assert summary["public"]["records"] == 1
    assert summary["public"]["unevaluable"] == 1
    assert summary["extension"]["records"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        benchmark_replay.summarize_replays([tmp_path / "replay", tmp_path / "replay"])
    with pytest.raises(FileExistsError):
        benchmark_replay.replay_candidate(candidate, package, config, tmp_path / "replay")


def test_replay_rejects_unpinned_mapping_before_output(tmp_path):
    candidate, package, config = replay_inputs(tmp_path)
    config["image"] = "python:latest"
    with pytest.raises(ValueError, match="digest-pinned"):
        benchmark_replay.replay_candidate(candidate, package, config, tmp_path / "replay")
    assert not (tmp_path / "replay").exists()


@pytest.mark.skipif(
    not __import__("os").environ.get("CHIPS_TEST_DOCKER_IMAGE"),
    reason="set immutable local Docker image for a real constructed checker",
)
def test_real_saved_candidate_replay_is_isolated_and_preserves_old_record(tmp_path):
    import json
    import os
    import shlex

    host_secret = tmp_path / "operator-secret"
    host_secret.write_text("must not be mounted")
    code = f"""
import hashlib,json,os,pathlib,socket
candidate=pathlib.Path(os.environ['CANDIDATE'])
assert not pathlib.Path({str(host_secret)!r}).exists()
assert not pathlib.Path('/var/run/docker.sock').exists()
assert not any(k.endswith('API_KEY') for k in os.environ)
try:
    candidate.write_text('changed')
except OSError:
    pass
else:
    raise AssertionError('candidate writable')
s=socket.socket();s.settimeout(.2)
try:
    s.connect(('192.0.2.1',9))
except OSError:
    pass
else:
    raise AssertionError('network available')
out=pathlib.Path(os.environ['VERIFY_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
report={{'status':'completed','reward':1,'cases':[{{'status':'graded','passed':True}}],
        'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
        'solver_options':json.loads(os.environ['CHIPS_SOLVER_OPTIONS'])}}
(out/'report.json').write_text(json.dumps(report))
"""
    checker = shlex.join(["python3", "-c", code]) + "\n"
    candidate, package, config = replay_inputs(tmp_path, checker=checker)
    config["image"] = os.environ["CHIPS_TEST_DOCKER_IMAGE"]
    config["solver_options"] = {"fixture_option": "explicit"}
    receipt = benchmark_replay.replay_candidate(candidate, package, config, tmp_path / "replay")
    assert receipt["result"]["execution"] == "ok", receipt["result"]
    assert receipt["result"]["score"] == 1
    assert receipt["classification"] == "not_compared"
    assert benchmark_replay.verify_replay(tmp_path / "replay") == receipt
    # Exercise the distributable entrypoint as well as source imports.
    import subprocess
    import sys

    from alphaapollo.common.execution.chips.bundle import build_cli

    bundle = tmp_path / "cli.pyz"
    build_cli(bundle)
    configuration = tmp_path / "replay.json"
    configuration.write_text(json.dumps(config))
    bundled_output = tmp_path / "bundled-replay"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(bundle),
            "replay-benchmark",
            "--candidate",
            str(candidate),
            "--task-package",
            str(package),
            "--config",
            str(configuration),
            "--output",
            str(bundled_output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    bundled = json.loads(completed.stdout)
    assert bundled["result"]["score"] == 1
    assert benchmark_replay.verify_replay(bundled_output) == bundled
    original = (tmp_path / "replay/receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        benchmark_replay.replay_candidate(candidate, package, config, tmp_path / "replay")
    assert (tmp_path / "replay/receipt.json").read_bytes() == original
    (tmp_path / "replay/run/work/verifier/report.json").write_text(json.dumps({"reward": 0}))
    with pytest.raises(ValueError, match="modified"):
        benchmark_replay.verify_replay(tmp_path / "replay")


def test_replay_consumes_only_verified_matching_spectre_archive(tmp_path):
    import json

    from test_benchmark_spectre import finish, inputs

    from alphaapollo.common.execution.chips.benchmark_spectre import package_identity
    from alphaapollo.common.execution.chips.jobs import submit_benchmark_spectre

    original = tmp_path / "original"
    original.mkdir()
    args = inputs(original)
    submit_benchmark_spectre(*args, "spectre-fixture")
    finish(original / "jobs/spectre-fixture")
    saved = original / "archive/spectre-fixture"
    new = tmp_path / "new"
    new.mkdir()
    candidate, package, config = replay_inputs(new)
    config["unsupported"] = ["declared unsupported"]
    before = (saved / "receipt.json").read_bytes()
    receipt = benchmark_replay.replay_candidate(
        candidate, package, config, tmp_path / "compared", spectre_record=saved
    )
    assert receipt["classification"] == "unevaluable"
    assert receipt["spectre"]["score"] == 1
    assert (saved / "receipt.json").read_bytes() == before
    assert benchmark_replay.verify_replay(tmp_path / "compared") == receipt
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["condition_id"] = "different-behavioral-condition"
    (package / "manifest.json").write_text(json.dumps(manifest))
    config["task_package_sha256"] = package_identity(package, purpose="final")["sha256"]
    with pytest.raises(ValueError, match="condition_id"):
        benchmark_replay.replay_candidate(
            candidate, package, config, tmp_path / "mismatched", spectre_record=saved
        )


def test_zipapp_replay_returns_sealed_unscored_backend_failure(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys

    from alphaapollo.common.execution.chips.bundle import build_cli

    candidate, package, config = replay_inputs(tmp_path)
    # Exercise the missing-Docker outcome deterministically. An ambient daemon
    # can instead hit the three-second budget before reporting a missing image.
    # The absolute Python executable still runs the real standard-library bundle.
    monkeypatch.setenv("PATH", str(tmp_path / "no-docker-bin"))
    bundle = tmp_path / "cli.pyz"
    build_cli(bundle)
    configuration = tmp_path / "replay.json"
    configuration.write_text(json.dumps(config))
    output = tmp_path / "zipapp-replay"
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(bundle),
            "replay-benchmark",
            "--candidate",
            str(candidate),
            "--task-package",
            str(package),
            "--config",
            str(configuration),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert process.returncode == 1, process.stderr
    assert (output / "receipt.json").is_file(), process.stderr
    receipt = json.loads(process.stdout)
    assert receipt["result"]["execution"] in {"infrastructure_error", "cleanup_failed"}
    assert receipt["result"]["score"] is None
    assert benchmark_replay.verify_replay(output) == receipt
