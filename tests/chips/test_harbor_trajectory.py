"""Offline export joins native context with independently verified fixture evidence."""

import json
import shutil

import pytest

pytest.importorskip("harbor")
from test_harbor_reporting import fixture_job
from test_pi_trajectory import session

from alphaapollo.workflows.harbor_chips.trajectory import export_pi_trial


def trial(tmp_path, index=0):
    directory = fixture_job(tmp_path) / f"trial-{index}"
    path, _ = session(tmp_path)
    target = directory / "agent/pi/sessions/session.jsonl"
    target.parent.mkdir(parents=True)
    shutil.copyfile(path, target)
    return directory


def test_export_attaches_checked_score_without_hidden_messages(tmp_path):
    directory = trial(tmp_path)
    output = tmp_path / "export"
    export_pi_trial(directory, output)
    atif = json.loads((output / "trajectory.json").read_text())
    circuit = atif["extra"]["circuit"]
    assert circuit["reward"] == 1
    assert circuit["grade_status"] == "verified"
    assert circuit["task_id"] == "fixture-task"
    assert len(circuit["candidate_sha256"]) == 64
    assert [s["source"] for s in atif["steps"]] == ["system", "user", "agent", "agent"]
    assert atif["steps"][-1]["message"] == "Done."
    assert (output.stat().st_mode & 0o777) == 0o700
    with pytest.raises(FileExistsError):
        export_pi_trial(directory, output)


def test_native_config_default_elision_is_accepted_but_conflict_is_not(tmp_path):
    from harbor.models.trial.config import TrialConfig

    directory = trial(tmp_path)
    result_path = directory / "result.json"
    raw = json.loads(result_path.read_text())
    raw["config"] = TrialConfig.model_validate(raw["config"]).model_dump(
        mode="json", exclude_defaults=True
    )
    result_path.write_text(json.dumps(raw))
    export_pi_trial(directory, tmp_path / "elided")
    raw["config"]["agent"]["model_name"] = "openai/another-model"
    result_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="identity mismatch"):
        export_pi_trial(directory, tmp_path / "conflicting")


@pytest.mark.parametrize(
    "damage", ["candidate", "score", "archive", "ambiguous_session", "unfinished"]
)
def test_invalid_trial_evidence_is_not_exported(tmp_path, damage):
    directory = trial(tmp_path)
    if damage == "candidate":
        path = directory / "public-session/candidate/files/dut.va"
        # Use the actual fixture's declared candidate name.
        manifest = json.loads((directory / "public-session/candidate/manifest.json").read_text())
        path = directory / "public-session/candidate/files" / next(iter(manifest["files"]))
        path.write_text("tampered")
    elif damage in {"score", "unfinished"}:
        path = directory / "result.json"
        raw = json.loads(path.read_text())
        if damage == "score":
            raw["verifier_result"]["rewards"]["reward"] = 0.25
        else:
            raw["finished_at"] = None
        path.write_text(json.dumps(raw))
    elif damage == "archive":
        next((directory / "verifier/transport").glob("*/archive/job.tar.gz")).write_bytes(
            b"broken archive"
        )
    else:
        native = directory / "agent/pi/sessions/session.jsonl"
        shutil.copyfile(native, native.with_name("extra.jsonl"))
    with pytest.raises((ValueError, OSError)):
        export_pi_trial(directory, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_valid_zero_and_absent_grading_remain_distinct(tmp_path):
    directory = trial(tmp_path, index=1)
    zero = export_pi_trial(directory, tmp_path / "zero")
    assert json.loads(zero.read_text())["extra"]["circuit"]["reward"] == 0
    shutil.rmtree(directory / "verifier")
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    result["verifier_result"] = None
    result_path.write_text(json.dumps(result))
    unscored = export_pi_trial(directory, tmp_path / "unscored")
    circuit = json.loads(unscored.read_text())["extra"]["circuit"]
    assert circuit["grade_status"] == "unscored"
    assert circuit["reward"] is None
