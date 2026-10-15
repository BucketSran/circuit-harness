"""Native Harbor transfers with real files/processes; no model or circuit grading."""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.models.task.config import TaskOS
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import RewardFileNotFoundError, Verifier, VerifierOutputParseError

from circuit_harness.harbor.artifact_delivery import ArtifactDelivery


class FilesystemEnvironment:
    """A local provider boundary. It does not claim container isolation."""

    os = TaskOS.LINUX
    capabilities = SimpleNamespace(mounted=False)

    def __init__(self, root, session_id):
        self.root = root
        self.root.mkdir(parents=True)
        self.session_id = session_id
        self.transfers = []

    def path(self, value):
        return self.root.joinpath(*PurePosixPath(str(value)).parts[1:])

    async def service_is_dir(self, source, **kwargs):
        return self.path(source).is_dir()

    async def service_download_file(self, source_path, target_path, **kwargs):
        shutil.copy2(self.path(source_path), target_path, follow_symlinks=False)

    async def service_download_dir(self, source_dir, target_dir, **kwargs):
        shutil.copytree(self.path(source_dir), target_dir, dirs_exist_ok=True, symlinks=True)

    async def ensure_dirs(self, paths, **kwargs):
        for path in paths:
            self.path(path).mkdir(parents=True, exist_ok=True)

    async def empty_dirs(self, paths, **kwargs):
        for path in paths:
            target = self.path(path)
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)

    async def upload_file(self, source_path, target_path):
        self.transfers.append(str(target_path))
        target = self.path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)

    async def upload_dir(self, source_dir, target_dir):
        self.transfers.append(str(target_dir))
        shutil.copytree(source_dir, self.path(target_dir), dirs_exist_ok=True, symlinks=True)

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(self.path(source_dir), target_dir, dirs_exist_ok=True, symlinks=True)

    async def exec(self, command, env=None, **kwargs):
        self.path("/logs/verifier").mkdir(parents=True, exist_ok=True)
        for name in ("/logs/verifier", "/tests", "/app"):
            command = command.replace(name, str(self.path(name)))
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.root,
            env={
                "PATH": str(Path(sys.executable).parent) + ":" + os.defpath,
                "DELIVERY_APP": str(self.path("/app")),
                "DELIVERY_LOGS": str(self.path("/logs/verifier")),
                **(env or {}),
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return SimpleNamespace(
            return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode()
        )


def source_fixture(tmp_path):
    baseline = tmp_path / "starter"
    (baseline / "testbench").mkdir(parents=True)
    (baseline / "circuit.va").write_bytes(b"STARTER\n")
    (baseline / "testbench/public.py").write_bytes(b"TRUSTED PUBLIC BENCH\n")
    source = FilesystemEnvironment(tmp_path / "agent", "agent-instance")
    shutil.copytree(baseline, source.path("/app"))
    return baseline, source


def test_single_file_collects_last_bytes_without_git_and_freezes_them(tmp_path):
    baseline, source = source_fixture(tmp_path)
    source.path("/app/circuit.va").write_bytes(b"SAVED FINAL BYTES\n")
    source.path("/app/scratch.log").write_text("not submitted")
    delivery = ArtifactDelivery(
        baseline, ["/app/circuit.va"], task_id="synthetic", task_version="v1"
    )
    frozen = asyncio.run(delivery.collect(source, tmp_path / "frozen"))
    source.path("/app/circuit.va").write_bytes(b"LATE CHANGE\n")
    receipt = delivery.verify(frozen)
    assert set(receipt["files"]) == {"circuit.va"}
    assert (frozen / "files/circuit.va").read_bytes() == b"SAVED FINAL BYTES\n"
    assert not (frozen / "files/scratch.log").exists()
    assert receipt["source_session"] == "agent-instance"


def test_explicit_nested_files_are_all_frozen_and_missing_file_stops_collection(tmp_path):
    baseline, source = source_fixture(tmp_path)
    (baseline / "modules").mkdir()
    (baseline / "modules/helper.va").write_text("OLD HELPER")
    source.path("/app/modules").mkdir()
    source.path("/app/modules/helper.va").write_text("NEW HELPER")
    delivery = ArtifactDelivery(
        baseline,
        ["/app/circuit.va", "/app/modules/helper.va"],
        task_id="synthetic",
        task_version="v1",
    )
    frozen = asyncio.run(delivery.collect(source, tmp_path / "first"))
    first = delivery.verify(frozen)
    assert set(first["files"]) == {"circuit.va", "modules/helper.va"}
    source.path("/app/modules/helper.va").write_text("SECOND HELPER")
    second = asyncio.run(delivery.collect(source, tmp_path / "second"))
    assert first["candidate_sha256"] != delivery.verify(second)["candidate_sha256"]
    source.path("/app/modules/helper.va").unlink()
    with pytest.raises(RuntimeError, match="collection incomplete"):
        asyncio.run(delivery.collect(source, tmp_path / "missing"))
    assert not (tmp_path / "missing/delivery.json").exists()


def test_directory_snapshot_preserves_new_renamed_and_empty_directories(tmp_path):
    baseline, source = source_fixture(tmp_path)
    (baseline / "modules").mkdir()
    (baseline / "modules/old.va").write_text("OLD")
    source.path("/app/modules/empty").mkdir(parents=True)
    source.path("/app/modules/renamed.va").write_text("RENAMED")
    source.path("/app/modules/new.va").write_text("NEW WITHOUT GIT")
    delivery = ArtifactDelivery(
        baseline,
        ["/app/circuit.va", "/app/modules/"],
        task_id="synthetic",
        task_version="v1",
    )
    frozen = asyncio.run(delivery.collect(source, tmp_path / "frozen"))
    receipt = delivery.verify(frozen)
    assert set(receipt["files"]) == {"circuit.va", "modules/renamed.va", "modules/new.va"}
    assert "modules/empty" in receipt["directories"]
    assert not (frozen / "files/modules/old.va").exists()


def test_an_entirely_empty_directory_is_a_valid_snapshot(tmp_path):
    baseline, source = source_fixture(tmp_path)
    (baseline / "modules").mkdir()
    source.path("/app/modules").mkdir()
    delivery = ArtifactDelivery(baseline, ["/app/modules"], task_id="synthetic", task_version="v1")
    frozen = asyncio.run(delivery.collect(source, tmp_path / "empty"))
    receipt = delivery.verify(frozen)
    assert receipt["files"] == {}
    assert receipt["directories"] == ["modules"]


def test_restore_uses_frozen_files_and_restores_fixed_material_even_inside_open_directory(tmp_path):
    baseline, source = source_fixture(tmp_path)
    (baseline / "modules").mkdir()
    (baseline / "modules/old.va").write_text("OLD")
    (baseline / "modules/fixed.inc").write_text("TRUSTED FIXED")
    source.path("/app/modules").mkdir()
    source.path("/app/modules/fixed.inc").write_text("TAMPERED FIXED")
    source.path("/app/modules/new.va").write_text("NEW")
    source.path("/app/circuit.va").write_text("FINAL")
    source.path("/app/testbench/public.py").write_text("TAMPERED PUBLIC")
    delivery = ArtifactDelivery(
        baseline,
        ["/app/circuit.va", "/app/modules"],
        task_id="synthetic",
        task_version="v1",
        fixed_paths=["/app/modules/fixed.inc"],
    )
    target = FilesystemEnvironment(tmp_path / "verifier", "verifier-instance")
    shutil.copytree(baseline, target.path("/app"))
    target.path("/app/modules/stale.va").write_text("STALE")

    async def run():
        frozen = await delivery.collect(source, tmp_path / "frozen")
        assert "modules/fixed.inc" not in delivery.verify(frozen)["files"]
        source.path("/app/modules/new.va").write_text("LATE CHANGE")
        await delivery.restore(target, frozen)
        return frozen

    asyncio.run(run())
    assert target.path("/app/circuit.va").read_text() == "FINAL"
    assert target.path("/app/modules/new.va").read_text() == "NEW"
    assert not target.path("/app/modules/old.va").exists()
    assert not target.path("/app/modules/stale.va").exists()
    assert target.path("/app/modules/fixed.inc").read_text() == "TRUSTED FIXED"
    assert target.path("/app/testbench/public.py").read_bytes() == b"TRUSTED PUBLIC BENCH\n"
    assert "/app/modules" in target.transfers  # Native directory upload, not per-file merging.


def test_empty_directory_restore_clears_old_baseline_files(tmp_path):
    baseline, source = source_fixture(tmp_path)
    (baseline / "modules").mkdir()
    (baseline / "modules/old.va").write_text("OLD")
    source.path("/app/modules").mkdir()
    delivery = ArtifactDelivery(baseline, ["/app/modules"], task_id="synthetic", task_version="v1")
    target = FilesystemEnvironment(tmp_path / "verifier", "verifier-instance")
    shutil.copytree(baseline, target.path("/app"))

    async def run():
        frozen = await delivery.collect(source, tmp_path / "frozen")
        await delivery.restore(target, frozen)

    asyncio.run(run())
    assert list(target.path("/app/modules").iterdir()) == []
    assert target.path("/app/circuit.va").read_bytes() == b"STARTER\n"


def test_restored_fixed_and_submitted_scripts_keep_executable_permissions(tmp_path):
    baseline, source = source_fixture(tmp_path)
    (baseline / "modules").mkdir()
    (baseline / "modules/fixed.sh").write_text("#!/bin/sh\nprintf TRUSTED\n")
    (baseline / "modules/fixed.sh").chmod(0o755)
    (baseline / "run.sh").write_text("#!/bin/sh\nprintf OLD\n")
    source.path("/app/modules").mkdir()
    source.path("/app/modules/fixed.sh").write_text("WRONG")
    source.path("/app/run.sh").write_text("#!/bin/sh\nprintf SUBMITTED\n")
    source.path("/app/run.sh").chmod(0o755)
    target = FilesystemEnvironment(tmp_path / "verifier", "verifier-instance")
    delivery = ArtifactDelivery(
        baseline,
        ["/app/run.sh", "/app/modules"],
        task_id="synthetic",
        task_version="v1",
        fixed_paths=["/app/modules/fixed.sh"],
    )

    async def run():
        frozen = await delivery.collect(source, tmp_path / "frozen")
        await delivery.restore(target, frozen)
        for name, expected in (("/app/run.sh", "SUBMITTED"), ("/app/modules/fixed.sh", "TRUSTED")):
            result = await target.exec(name)
            assert result.return_code == 0, result.stderr
            assert result.stdout == expected

    asyncio.run(run())


def verifier_fixture(tmp_path, environment, expected, *, paths=None):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment/Dockerfile").write_text("FROM fixture\n")
    (task / "instruction.md").write_text("Synthetic file-delivery check.\n")
    (task / "task.toml").write_text('schema_version = "1.4"\n[verifier]\ntimeout_sec = 10\n')
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text('#!/bin/sh\nset -eu\npython3 "$(dirname "$0")/verify.py"\n')
    (task / "tests/verify.py").write_text(
        "import json,os,pathlib\n"
        "app=pathlib.Path(os.environ.get('DELIVERY_APP','/app'))\n"
        "out=pathlib.Path(os.environ.get('DELIVERY_LOGS','/logs/verifier'));out.mkdir(parents=True,exist_ok=True)\n"
        f"expected={expected!r}\n"
        "actual={p.relative_to(app).as_posix():p.read_text() "
        "for p in app.rglob('*') if p.is_file()}\n"
        "(out/'reward.json').write_text(json.dumps({'reward':float(actual==expected)}))\n"
        "(out/'measurements.json').write_text(json.dumps({'files':actual,'matched':actual==expected}))\n"
    )
    paths = paths or TrialPaths(tmp_path / "trial")
    return Verifier(task=Task(task), trial_paths=paths, environment=environment)


@pytest.mark.parametrize("candidate,score", [("CORRECT", 1), ("WRONG", 0)])
@pytest.mark.parametrize("preloaded", [False, True])
def test_original_native_verifier_grades_restored_frozen_bytes(
    tmp_path, candidate, score, preloaded
):
    baseline, source = source_fixture(tmp_path)
    source.path("/app/circuit.va").write_text(candidate)
    target = FilesystemEnvironment(tmp_path / "independent", "verifier-instance")
    verifier = verifier_fixture(
        tmp_path,
        target,
        {
            "circuit.va": "CORRECT",
            "testbench/public.py": "TRUSTED PUBLIC BENCH\n",
        },
    )
    if preloaded:
        shutil.copytree(verifier.task.paths.tests_dir, target.path("/tests"))
        verifier._skip_tests_upload = True
    delivery = ArtifactDelivery(
        baseline,
        ["/app/circuit.va"],
        task_id="synthetic",
        task_version="v1",
        verifier_task=verifier.task,
    )
    assert not source.path("/tests").exists()

    async def run():
        frozen = await delivery.collect(source, tmp_path / "frozen")
        result = await delivery.grade(verifier, frozen)
        return frozen, result

    frozen, result = asyncio.run(run())
    assert result.rewards == {"reward": score}
    report = json.loads((verifier.trial_paths.verifier_dir / "delivery-result.json").read_text())
    assert report["candidate_sha256"] == delivery.verify(frozen)["candidate_sha256"]
    assert report["rewards"] == {"reward": score}
    assert not source.path("/tests").exists()
    assert not source.path("/logs/verifier/reward.json").exists()


def test_a_different_verifier_task_cannot_grade_the_frozen_delivery(tmp_path):
    baseline, source = source_fixture(tmp_path)
    target = FilesystemEnvironment(tmp_path / "verifier", "verifier-instance")
    approved = verifier_fixture(tmp_path / "approved", target, {"circuit.va": "APPROVED"})
    unrelated = verifier_fixture(tmp_path / "unrelated", target, {"circuit.va": "UNRELATED"})
    delivery = ArtifactDelivery(
        baseline,
        ["/app/circuit.va"],
        task_id="synthetic",
        task_version="v1",
        verifier_task=approved.task,
    )

    async def run():
        frozen = await delivery.collect(source, tmp_path / "frozen")
        with pytest.raises(RuntimeError, match="verifier task"):
            await delivery.grade(unrelated, frozen)

    asyncio.run(run())
    assert not unrelated.trial_paths.reward_json_path.exists()


@pytest.mark.parametrize("override", ["preloaded", "step"])
def test_actual_checker_must_match_bound_tests_before_it_can_run(tmp_path, override):
    baseline, source = source_fixture(tmp_path)
    target = FilesystemEnvironment(tmp_path / "verifier", "verifier-instance")
    verifier = verifier_fixture(tmp_path, target, {"circuit.va": "WILL NOT MATCH"})
    if override == "preloaded":
        verifier._skip_tests_upload = True
        rogue = target.path("/tests")
    else:
        verifier.step_name = "alternate"
        rogue = verifier.task.paths.task_dir / "steps/alternate/tests"
    rogue.mkdir(parents=True)
    (rogue / "test.sh").write_text('#!/bin/sh\nprintf 1 > "$DELIVERY_LOGS/reward.txt"\n')
    delivery = ArtifactDelivery(
        baseline,
        ["/app/circuit.va"],
        task_id="synthetic",
        task_version="v1",
        verifier_task=verifier.task,
    )

    async def run():
        frozen = await delivery.collect(source, tmp_path / "frozen")
        with pytest.raises(RuntimeError, match="checker.*differs"):
            await delivery.grade(verifier, frozen)

    asyncio.run(run())
    assert not target.path("/logs/verifier/reward.txt").exists()
    assert not (verifier.trial_paths.verifier_dir / "delivery-result.json").exists()


def test_checker_crash_after_writing_reward_is_not_a_completed_score(tmp_path):
    baseline, source = source_fixture(tmp_path)
    target = FilesystemEnvironment(tmp_path / "verifier", "verifier-instance")
    verifier = verifier_fixture(tmp_path, target, {})
    checker = verifier.task.paths.tests_dir / "verify.py"
    with checker.open("a") as stream:
        stream.write("raise SystemExit(7)\n")
    delivery = ArtifactDelivery(
        baseline,
        ["/app/circuit.va"],
        task_id="synthetic",
        task_version="v1",
        verifier_task=verifier.task,
    )

    async def run():
        frozen = await delivery.collect(source, tmp_path / "frozen")
        with pytest.raises(RuntimeError, match="entrypoint"):
            await delivery.grade(verifier, frozen)

    asyncio.run(run())
    assert not (verifier.trial_paths.verifier_dir / "delivery-result.json").exists()


@pytest.mark.parametrize(
    "changed", ["files/circuit.va", "baseline/testbench/public.py", "receipt", "mode"]
)
def test_snapshot_or_receipt_tampering_is_rejected_before_restoration(tmp_path, changed):
    baseline, source = source_fixture(tmp_path)
    delivery = ArtifactDelivery(
        baseline, ["/app/circuit.va"], task_id="synthetic", task_version="v1"
    )
    frozen = asyncio.run(delivery.collect(source, tmp_path / "frozen"))
    if changed == "receipt":
        path = frozen / "delivery.json"
        data = json.loads(path.read_text())
        data["source_session"] = "another-source"
        path.write_text(json.dumps(data))
    elif changed == "mode":
        (frozen / "files/circuit.va").chmod(0o755)
    else:
        (frozen / changed).write_text("TAMPERED")
    target = FilesystemEnvironment(tmp_path / "verifier", "verifier-instance")
    with pytest.raises(RuntimeError, match="changed"):
        asyncio.run(delivery.restore(target, frozen))
    assert target.transfers == []


def test_transfer_failure_cannot_fall_back_to_the_starter(tmp_path):
    class FailedTransfer(FilesystemEnvironment):
        async def service_download_file(self, source_path, target_path, **kwargs):
            raise ConnectionError("synthetic transfer failure")

    baseline, source = source_fixture(tmp_path)
    failing = FailedTransfer(tmp_path / "failed-agent", "failed-transfer")
    shutil.copytree(source.root, failing.root, dirs_exist_ok=True)
    delivery = ArtifactDelivery(
        baseline, ["/app/circuit.va"], task_id="synthetic", task_version="v1"
    )
    with pytest.raises(RuntimeError, match="collection incomplete"):
        asyncio.run(delivery.collect(failing, tmp_path / "failed"))
    entries = json.loads((tmp_path / "failed/collected/manifest.json").read_text())
    assert next(row for row in entries if row["source"] == "/app/circuit.va")["status"] == "failed"
    assert not (tmp_path / "failed/delivery.json").exists()


@pytest.mark.parametrize("kind", ["symlink", "directory", "missing"])
def test_invalid_single_file_source_is_not_frozen(tmp_path, kind):
    baseline, source = source_fixture(tmp_path)
    path = source.path("/app/circuit.va")
    path.unlink()
    if kind == "symlink":
        path.symlink_to(source.path("/app/testbench/public.py"))
    elif kind == "directory":
        path.mkdir()
    delivery = ArtifactDelivery(
        baseline, ["/app/circuit.va"], task_id="synthetic", task_version="v1"
    )
    with pytest.raises(RuntimeError, match="regular file|collection incomplete"):
        asyncio.run(delivery.collect(source, tmp_path / "frozen"))
    assert not (tmp_path / "frozen/delivery.json").exists()


def test_wrong_plan_and_agent_environment_cannot_receive_the_submission(tmp_path):
    baseline, source = source_fixture(tmp_path)
    delivery = ArtifactDelivery(
        baseline, ["/app/circuit.va"], task_id="synthetic", task_version="v1"
    )
    frozen = asyncio.run(delivery.collect(source, tmp_path / "frozen"))
    other = ArtifactDelivery(baseline, ["/app/circuit.va"], task_id="other-task", task_version="v1")
    with pytest.raises(RuntimeError, match="another task/plan"):
        other.verify(frozen)
    with pytest.raises(RuntimeError, match="independent verifier"):
        asyncio.run(delivery.restore(source, frozen))


@pytest.mark.parametrize("failure", ["stale-reward", "missing-reward", "nonfinite-reward"])
def test_incomplete_or_stale_grading_cannot_publish_a_delivery_score(tmp_path, failure):
    baseline, source = source_fixture(tmp_path)
    target = FilesystemEnvironment(tmp_path / "verifier", "verifier-instance")
    verifier = verifier_fixture(tmp_path, target, {})
    checker = verifier.task.paths.tests_dir / "verify.py"
    if failure == "missing-reward":
        checker.write_text("print('no reward')\n")
    elif failure == "nonfinite-reward":
        with checker.open("a") as stream:
            stream.write("(out/'reward.json').write_text('{\"reward\":NaN}')\n")
    else:
        verifier.trial_paths.verifier_dir.mkdir(parents=True)
        verifier.trial_paths.reward_json_path.write_text('{"reward":1}')
    delivery = ArtifactDelivery(
        baseline,
        ["/app/circuit.va"],
        task_id="synthetic",
        task_version="v1",
        verifier_task=verifier.task,
    )

    async def run():
        frozen = await delivery.collect(source, tmp_path / "frozen")
        with pytest.raises((RuntimeError, RewardFileNotFoundError, VerifierOutputParseError)):
            await delivery.grade(verifier, frozen)

    asyncio.run(run())
    assert not (verifier.trial_paths.verifier_dir / "delivery-result.json").exists()


@pytest.mark.parametrize(
    "artifacts",
    [
        ["/app/circuit.va", "/app/circuit.va"],
        [{"source": "/app/circuit.va", "destination": "../escape"}],
        ["/app/../circuit.va"],
    ],
)
def test_invalid_declarations_are_rejected_before_collection(tmp_path, artifacts):
    baseline, _ = source_fixture(tmp_path)
    with pytest.raises((RuntimeError, ValueError)):
        ArtifactDelivery(baseline, artifacts, task_id="synthetic", task_version="v1")


@pytest.mark.parametrize("change_input", [False, True])
def test_grader_may_create_outputs_but_cannot_change_restored_input_bytes(tmp_path, change_input):
    baseline, source = source_fixture(tmp_path)
    target = FilesystemEnvironment(tmp_path / "verifier", "verifier-instance")
    verifier = verifier_fixture(
        tmp_path,
        target,
        {
            "circuit.va": "STARTER\n",
            "testbench/public.py": "TRUSTED PUBLIC BENCH\n",
        },
    )
    with (verifier.task.paths.tests_dir / "verify.py").open("a") as stream:
        if change_input:
            stream.write("(app/'circuit.va').write_text('CHANGED BY CHECKER')\n")
        else:
            stream.write(
                "(app/'testbench/psf').mkdir();(app/'testbench/psf/wave.dat').write_text('OUTPUT')\n"
            )
    delivery = ArtifactDelivery(
        baseline,
        ["/app/circuit.va"],
        task_id="synthetic",
        task_version="v1",
        verifier_task=verifier.task,
    )

    async def run():
        frozen = await delivery.collect(source, tmp_path / "frozen")
        if change_input:
            with pytest.raises(RuntimeError, match="input|differs"):
                await delivery.grade(verifier, frozen)
        else:
            result = await delivery.grade(verifier, frozen)
            assert result.rewards == {"reward": 1}

    asyncio.run(run())
    assert (verifier.trial_paths.verifier_dir / "delivery-result.json").exists() != change_input
