"""Real verifier and archive transfer with an external SSH process fixture."""

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")
from harbor.models.trial.paths import TrialPaths
from test_benchmark_spectre import inputs

from circuit_harness.execution.bundle import build_cli
from circuit_harness.execution.candidate_bundle import verify_candidate
from circuit_harness.harbor.config import HarborChipsConfig
from circuit_harness.harbor.verifier import FrozenCandidateVerifier


def verifier_fixture(tmp_path, monkeypatch, archive_state="delayed", *, final_timeout_s=10):
    candidate, package, profile = inputs(tmp_path)
    bundle = tmp_path / "cli.pyz"
    build_cli(bundle)
    remote = {
        "host": "fixture",
        "python": sys.executable,
        "bundle": str(bundle),
        "profile": str(profile),
        "run_root": str(tmp_path / "jobs"),
        "archive_root": str(tmp_path / "archive"),
        "upload_root": str(tmp_path / "uploads"),
    }
    # The server publishes completion before the archive. Model that ordering
    # on the SSH wire, while using real staging, jobs and archive validation.
    ssh = tmp_path / "ssh"
    ssh.write_text(
        f"#!{sys.executable}\n"
        "import json,pathlib,shlex,subprocess,sys\n"
        f"root=pathlib.Path({str(tmp_path)!r})\n"
        f"mode={archive_state!r}\n"
        "argv=shlex.split(sys.argv[-1])\n"
        "with (root/'commands.jsonl').open('a') as log:\n"
        "    log.write(json.dumps(argv)+'\\n')\n"
        "if 'verify-archive' in argv and (root/'waiting').exists():\n"
        "    sys.exit(2)\n"
        "result=subprocess.run(argv,stdin=sys.stdin,stdout=subprocess.PIPE)\n"
        "data=result.stdout\n"
        "if 'job-status' in argv:\n"
        "    state=json.loads(data)\n"
        "    if state.get('state')=='finished':\n"
        "        count_path=root/'queries'\n"
        "        count=int(count_path.read_text())+1 if count_path.exists() else 1\n"
        "        count_path.write_text(str(count))\n"
        "        pending=('pending' if count==1 else 'running')\n"
        "        if mode=='delayed' and count<=2:\n"
        "            state['archive']={'state':pending}\n"
        "        elif mode!='delayed':\n"
        "            state['archive']={'state':mode}\n"
        "        if state.get('archive',{}).get('state')=='verified':\n"
        "            (root/'waiting').unlink(missing_ok=True)\n"
        "        else:\n"
        "            (root/'waiting').touch()\n"
        "        data=json.dumps(state).encode()\n"
        "sys.stdout.buffer.write(data)\n"
        "sys.exit(result.returncode)\n"
    )
    ssh.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    frozen = verify_candidate(candidate)
    environment = SimpleNamespace(
        context_id="fixture",
        frozen={
            "candidate_directory": str(candidate),
            "candidate_sha256": frozen["candidate_sha256"],
        },
        settings=HarborChipsConfig(
            materials=tmp_path,
            checkout=tmp_path,
            kernel=tmp_path / "kernel",
            image="controlled-public-unused",
            task={"task_id": "fixture-task", "task_version": "fixture-v1"},
            public_backend="docker",
            final_task_package=package,
            final_remote=remote,
            final_timeout_s=final_timeout_s,
        ),
    )
    paths = TrialPaths(trial_dir=tmp_path / "trial")
    verifier = FrozenCandidateVerifier(task=None, trial_paths=paths, environment=environment)
    return verifier


def commands(tmp_path, name):
    return [
        command
        for line in (tmp_path / "commands.jsonl").read_text().splitlines()
        if name in (command := json.loads(line))
    ]


def test_verifier_waits_for_published_archive_before_retrieving(tmp_path, monkeypatch):
    verifier = verifier_fixture(tmp_path, monkeypatch)
    result = asyncio.run(verifier.verify())
    assert result.rewards == {"reward": 1}
    assert int((tmp_path / "queries").read_text()) >= 3
    assert len(commands(tmp_path, "submit-benchmark-spectre")) == 1
    assert len(commands(tmp_path, "verify-archive")) == 1
    assert (tmp_path / "trial/verifier/evaluation.json").is_file()


@pytest.mark.parametrize("state", ["failed", "unexpected"])
def test_verifier_does_not_grade_unavailable_archive(tmp_path, monkeypatch, state):
    verifier = verifier_fixture(tmp_path, monkeypatch, state)
    with pytest.raises(RuntimeError, match="archive"):
        asyncio.run(verifier.verify())
    assert not commands(tmp_path, "verify-archive")
    assert len(commands(tmp_path, "submit-benchmark-spectre")) == 1
    assert not (tmp_path / "trial/verifier/evaluation.json").exists()


def test_archive_wait_obeys_verifier_deadline_without_resubmitting(tmp_path, monkeypatch):
    verifier = verifier_fixture(tmp_path, monkeypatch, "running", final_timeout_s=2)
    with pytest.raises(TimeoutError):
        asyncio.run(verifier.verify())
    assert not commands(tmp_path, "verify-archive")
    assert len(commands(tmp_path, "submit-benchmark-spectre")) == 1
    assert (tmp_path / "jobs/harbor-fixture/request.json").exists()
    assert (tmp_path / "frozen/manifest.json").exists()
