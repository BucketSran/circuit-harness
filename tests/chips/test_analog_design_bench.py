"""Analog Design Bench source pin and verifier-result contract."""

import json
from pathlib import Path

import pytest

from alphaapollo.common.execution.chips import analog_design_bench as adb


def test_tree_digest_detects_changed_or_extra_verifier_file(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "verify.py").write_text("original\n")
    original = adb.tree_digest(tmp_path)

    (tests / "verify.py").write_text("modified\n")
    assert adb.tree_digest(tmp_path) != original

    (tests / "verify.py").write_text("original\n")
    (tests / "extra.py").write_text("extra\n")
    assert adb.tree_digest(tmp_path) != original


def test_tree_digest_rejects_symlink(tmp_path: Path) -> None:
    (tmp_path / "outside").write_text("data")
    (tmp_path / "alias").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symbolic link"):
        adb.tree_digest(tmp_path)


@pytest.mark.parametrize(
    ("reward", "expected"),
    [
        ({"reward": 1.0, "tests_total": 7, "tests_passed": 7}, "graded"),
        ({"reward": 0.0, "tests_total": 7, "tests_passed": 0}, "graded"),
        ({"reward": 0.0, "tests_total": 0, "tests_passed": 0}, "verifier_error"),
        ({"reward": 1.1, "tests_total": 7, "tests_passed": 7}, "verifier_error"),
    ],
)
def test_result_distinguishes_design_failure_from_verifier_failure(
    tmp_path: Path, reward: dict, expected: str
) -> None:
    path = tmp_path / "reward.json"
    path.write_text(json.dumps(reward))
    result = adb.read_result(path, container_exit=0)
    assert result["state"] == expected
    if expected == "graded":
        assert result["score"] == reward["reward"]
    else:
        assert result["score"] is None


def test_nonzero_container_exit_cannot_publish_score(tmp_path: Path) -> None:
    path = tmp_path / "reward.json"
    path.write_text('{"reward":1,"tests_total":7,"tests_passed":7}')
    result = adb.read_result(path, container_exit=125)
    assert result["state"] == "verifier_error"
    assert result["score"] is None


def test_ngspice_banner_selects_actual_version() -> None:
    assert (
        adb.ngspice_version("******\n** ngspice-47 : Circuit level simulation program\n")
        == "ngspice-47"
    )


def test_run_records_candidate_and_excludes_model_key_from_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = "rlc-rf-bandpass-100mhz"
    source = tmp_path / "source/tasks" / task_id
    (source / "tests").mkdir(parents=True)
    (source / "tests/test.sh").write_text("#!/bin/sh\n")
    monkeypatch.setitem(
        adb.TASKS,
        task_id,
        adb.Task("test-commit", adb.tree_digest(source), "test-image@sha256:" + "a" * 64),
    )
    candidate = tmp_path / "candidate.spi"
    candidate.write_text("R1 IN OUT 100\n")
    fake_podman = tmp_path / "fake-podman"
    fake_podman.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "mount = next(x for x in args if x.endswith(':/logs/verifier:rw')).split(':')[0]\n"
        "reward = {'reward': 0.5, 'tests_total': 2, 'tests_passed': 1}\n"
        "pathlib.Path(mount, 'reward.json').write_text(json.dumps(reward))\n"
        "print('key_present=' + str(bool(os.environ.get('CHIPS_MODEL_KEY'))))\n"
        "print('other_secret_present=' + str(bool(os.environ.get('UNLISTED_SECRET'))))\n"
        "print('runtime_image=' + args[-4])\n"
        "print('single_id=' + str('ignore_chown_errors=true' in args))\n"
        "print('cpu_quota=' + str('--cpus=4' in args))\n"
    )
    fake_podman.chmod(0o700)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "private-test-value")
    monkeypatch.setenv("UNLISTED_SECRET", "also-private")
    output = tmp_path / "result"
    result = adb.run_case(
        task_id,
        tmp_path / "source",
        candidate,
        output,
        podman=str(fake_podman),
        archive_root=tmp_path / "archive",
    )
    assert result["state"] == "graded"
    assert result["score"] == 0.5
    assert (output / "inputs/circuit.spi").read_text() == candidate.read_text()
    assert json.loads((output / "manifest.json").read_text())["source_commit"] == "test-commit"
    assert "key_present=False" in (output / "verifier.log").read_text()
    assert "other_secret_present=False" in (output / "verifier.log").read_text()
    assert (tmp_path / "archive/result/inputs/circuit.spi").read_text() == candidate.read_text()
    assert json.loads((tmp_path / "archive/result/result.json").read_text())["archive"] == str(
        tmp_path / "archive/result"
    )
    another_output = tmp_path / "another/result"
    with pytest.raises(ValueError, match="archive target already exists"):
        adb.run_case(
            task_id,
            tmp_path / "source",
            candidate,
            another_output,
            podman=str(fake_podman),
            archive_root=tmp_path / "archive",
        )
    assert not another_output.exists()

    image_archive = tmp_path / "sky130-tools.tar"
    image_archive.write_bytes(b"public image archive fixture")
    local_image = "sha256:" + "b" * 64
    offline_output = tmp_path / "offline"
    adb.run_case(
        task_id,
        tmp_path / "source",
        candidate,
        offline_output,
        podman=str(fake_podman),
        runtime_image=local_image,
        offline_image_archive=image_archive,
        podman_single_id=True,
        podman_no_cpu_limit=True,
    )
    offline_manifest = json.loads((offline_output / "manifest.json").read_text())
    assert offline_manifest["image"] == "test-image@sha256:" + "a" * 64
    assert offline_manifest["runtime_image"] == local_image
    assert offline_manifest["offline_image_archive_sha256"] == adb.file_digest(image_archive)
    assert "runtime_image=" + local_image in (offline_output / "verifier.log").read_text()
    assert "single_id=True" in (offline_output / "verifier.log").read_text()
    assert "cpu_quota=False" in (offline_output / "verifier.log").read_text()
