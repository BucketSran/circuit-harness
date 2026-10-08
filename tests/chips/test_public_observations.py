"""Public session observations, with synthetic data and no simulator/model execution."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from circuit_harness.execution import current_evas_session as session
from tests.chips.test_current_evas_session import call, make_session


def docker_output(tmp_path, monkeypatch, data):
    """External Docker CLI double; production snapshot/process/result code still runs."""
    payload = tmp_path / "docker-output.json"
    payload.write_text(json.dumps(data))
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport pathlib,sys\n"
        f"if sys.argv[1] == 'run': print(pathlib.Path({str(payload)!r}).read_text())\n"
    )
    docker.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    return payload


def waveform():
    return {
        "engine": "synthetic",
        "nodes": ["z", "count"],
        "solutions": [{"voltages": [i / 1000, i // 500]} for i in range(1001)],
        "transient": {"times": [i / 1000 for i in range(1001)], "events": []},
    }


def waveform_session(tmp_path, monkeypatch, **options):
    data = waveform()
    docker_output(tmp_path, monkeypatch, data)
    manifest = {
        "models": ["dut.va"],
        "instances": [],
        "transient": {
            "sources": {},
            "output_times": data["transient"]["times"],
            "stop": 1,
            "max_step": 0.001,
        },
        "tolerances": {"vabstol": 1e-8, "reltol": 0},
    }
    directory = make_session(tmp_path, manifest=manifest, **options)
    call(directory, "write", "evas_write", path="dut.va", content="synthetic candidate")
    return directory


def test_info_is_compact_and_full_manifest_is_readable_without_simulating(tmp_path):
    manifest = {
        "models": ["dut.va"],
        "instances": [],
        "transient": {
            "sources": {"ctl": [[0, 1], [3, 1]]},
            "output_times": [i / 200 for i in range(601)],
            "stop": 3,
            "max_step": 0.001,
        },
        "tolerances": {"vabstol": 1e-8, "reltol": 0},
    }
    directory = make_session(tmp_path, manifest=manifest)
    info = session.session_info(directory)
    assert info["observation_view_version"] == 1
    assert "output_times" not in json.dumps(info["manifest"])
    assert info["manifest"]["sample_count"] == 601
    assert info["manifest"]["time_range"] == [0, 3]
    artifact = info["manifest_artifact"]
    pieces, offset = [], 0
    while offset is not None:
        result = call(
            directory,
            f"page-{offset}",
            "evas_read_artifact",
            artifact_id=artifact["id"],
            offset=offset,
            limit=4096,
        )["result"]
        pieces.append(result["content"])
        offset = result["next_offset"]
    content = "".join(pieces)
    assert json.loads(content) == manifest
    assert hashlib.sha256(content.encode()).hexdigest() == artifact["sha256"]
    assert session.session_info(directory)["remaining_simulations"] == 1


def test_simulation_returns_summary_but_keeps_exact_public_data_and_candidate(
    tmp_path, monkeypatch
):
    directory = waveform_session(tmp_path, monkeypatch, feedback_fields=["observations"])
    response = call(directory, "sim", "evas_simulate")
    result = response["result"]
    summary = result["observations"]
    assert summary["kind"] == "waveform_summary"
    assert summary["sample_count"] == 1001
    assert summary["time_range"] == [0, 1]
    assert summary["signals"] == [
        {"name": "z", "min": 0, "max": 1},
        {"name": "count", "min": 0, "max": 2},
    ]
    assert len(json.dumps(response).encode()) < 4096
    assert result["task_correctness"] == "not_evaluated"
    raw = json.loads((directory / "actions/sim/execution/result.json").read_text())
    assert raw["observations"] == waveform()
    artifact = result["observation_artifact"]
    assert artifact["candidate_sha256"] == result["candidate_sha256"]
    page = call(directory, "read", "evas_read_artifact", artifact_id=artifact["id"], limit=4096)
    assert page["result"]["artifact"] == artifact
    assert page["result"]["content"].startswith('{"engine":"synthetic"')
    assert call(directory, "sim", "evas_simulate") == response


def test_agent_can_inspect_signal_window_without_rerunning_or_changing_evidence(
    tmp_path, monkeypatch
):
    directory = waveform_session(tmp_path, monkeypatch, feedback_fields=["observations"])
    artifact = call(directory, "sim", "evas_simulate")["result"]["observation_artifact"]
    call(directory, "edit", "evas_write", path="dut.va", content="new candidate")
    response = call(
        directory,
        "window",
        "evas_observe",
        artifact_id=artifact["id"],
        signals=["count", "z"],
        start=0.49,
        end=0.51,
        max_points=3,
    )
    result = response["result"]
    assert result["artifact"] == artifact
    assert result["times"] == [0.49, 0.5, 0.51]
    assert result["values"] == [[0, 0.49], [1, 0.5], [1, 0.51]]
    assert result["signals"] == ["count", "z"]
    assert result["matched_points"] == 21
    assert result["sampled"] is True
    assert result["sampling"] == "uniform_index_endpoints"
    assert (
        call(
            directory,
            "window",
            "evas_observe",
            artifact_id=artifact["id"],
            signals=["count", "z"],
            start=0.49,
            end=0.51,
            max_points=3,
        )
        == response
    )
    exact = call(
        directory,
        "exact",
        "evas_observe",
        artifact_id=artifact["id"],
        signals=["z"],
        start=0.499,
        end=0.501,
    )["result"]
    assert exact["times"] == [0.499, 0.5, 0.501]
    assert exact["sampled"] is False
    assert session.session_info(directory)["remaining_simulations"] == 0
    assert call(directory, "again", "evas_simulate")["error"] == "simulation_budget_exhausted"


@pytest.mark.parametrize("large", [False, True])
def test_measurements_keep_small_metrics_inline_and_large_results_are_lossless(
    tmp_path, monkeypatch, large
):
    data = {"error": 0.001, "comment": "测量🌊" * (1800 if large else 1)}
    docker_output(tmp_path, monkeypatch, data)
    directory = make_session(
        tmp_path,
        feedback_fields=["observations"],
        experiments={
            "version": "v1",
            "files": ["probe.py"],
            "analyses": ["python_measurement"],
        },
    )
    call(directory, "write", "evas_write", path="dut.va", content="candidate")
    call(directory, "script", "evas_write", path="probe.py", content="synthetic measurement")
    result = call(
        directory, "measure", "evas_experiment", analysis="python_measurement", script="probe.py"
    )["result"]
    assert result["authority"] == "agent_measurement"
    if large:
        assert result["observations"] == {"kind": "json_artifact", "inline_omitted": True}
    else:
        assert result["observations"] == data
    artifact = result["observation_artifact"]
    pieces, offset = [], 0
    while offset is not None:
        page = call(
            directory,
            f"read-{offset}",
            "evas_read_artifact",
            artifact_id=artifact["id"],
            offset=offset,
        )["result"]
        pieces.append(page["content"])
        offset = page["next_offset"]
    content = "".join(pieces)
    assert json.loads(content) == data
    assert hashlib.sha256(content.encode()).hexdigest() == artifact["sha256"]
    assert not call(directory, "wave", "evas_observe", artifact_id=artifact["id"])["ok"]


def test_undeclared_observations_never_gain_an_artifact_read_path(tmp_path, monkeypatch):
    directory = waveform_session(tmp_path, monkeypatch, feedback_fields=[])
    result = call(directory, "sim", "evas_simulate")["result"]
    assert "observations" not in result and "observation_artifact" not in result
    assert not call(directory, "read", "evas_read_artifact", artifact_id="observation:sim")["ok"]
    assert not (directory / "actions/sim/observation.json").exists()


@pytest.mark.parametrize("corruption", ["changed", "symlink", "pending"])
def test_corrupt_or_unfinished_artifacts_cannot_be_read(tmp_path, monkeypatch, corruption):
    directory = waveform_session(tmp_path, monkeypatch, feedback_fields=["observations"])
    artifact = call(directory, "sim", "evas_simulate")["result"]["observation_artifact"]
    path = directory / "actions/sim/observation.json"
    if corruption == "changed":
        path.write_text('{"secret":"changed"}')
    elif corruption == "symlink":
        path.unlink()
        path.symlink_to(directory / "session.json")
    else:
        (directory / "actions/sim/response.json").unlink()
    response = call(directory, "read", "evas_read_artifact", artifact_id=artifact["id"])
    assert not response["ok"]
    assert "content" not in response and "secret" not in json.dumps(response)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_digest",
        "missing_id",
        "list_format",
        "null_artifact",
        "null_result",
        "list_response",
    ],
)
def test_corrupt_artifact_metadata_does_not_leave_an_unfinished_read(
    tmp_path, monkeypatch, corruption
):
    directory = waveform_session(tmp_path, monkeypatch, feedback_fields=["observations"])
    response = call(directory, "sim", "evas_simulate")
    if corruption == "missing_digest":
        response["result"]["observation_artifact"].pop("sha256")
    elif corruption == "missing_id":
        response["result"]["observation_artifact"].pop("id")
    elif corruption == "list_format":
        response["result"]["observation_artifact"]["format"] = []
    elif corruption == "null_artifact":
        response["result"]["observation_artifact"] = None
    elif corruption == "null_result":
        response["result"] = None
    else:
        response = []
    (directory / "actions/sim/response.json").write_text(json.dumps(response))
    failed = call(directory, "read", "evas_read_artifact", artifact_id="observation:sim")
    assert failed["ok"] is False
    assert call(directory, "read", "evas_read_artifact", artifact_id="observation:sim") == failed
    assert call(directory, "next", "evas_read", path="instruction.md")["ok"]
    assert call(directory, "submit", "evas_submit")["ok"]


@pytest.mark.parametrize(
    "artifact_id",
    ["../session.json", "observation:../sim", "observation:missing", "final", "/etc/passwd"],
)
def test_only_current_session_public_artifacts_are_readable(tmp_path, artifact_id):
    directory = make_session(tmp_path)
    assert not call(directory, "read", "evas_read_artifact", artifact_id=artifact_id)["ok"]


@pytest.mark.parametrize(
    "tool,args",
    [
        ("evas_read_artifact", {"offset": -1}),
        ("evas_read_artifact", {"limit": 4097}),
        ("evas_read_artifact", {"offset": True}),
        ("evas_read_artifact", {"limit": 0}),
        ("evas_observe", {"signals": []}),
        ("evas_observe", {"signals": ["z", "z"]}),
        ("evas_observe", {"signals": [str(i) for i in range(9)]}),
        ("evas_observe", {"start": float("nan")}),
        ("evas_observe", {"start": 10**1000}),
        ("evas_observe", {"end": float("inf")}),
        ("evas_observe", {"start": 1, "end": 0}),
        ("evas_observe", {"max_points": 129}),
        ("evas_observe", {"max_points": 1}),
        ("evas_observe", {"max_points": True}),
        ("evas_observe", {"unknown": 1}),
    ],
)
def test_invalid_read_queries_are_rejected_before_reserving_an_action(tmp_path, tool, args):
    directory = make_session(tmp_path)
    with pytest.raises(ValueError):
        call(directory, "bad", tool, artifact_id="manifest", **args)
    assert not (directory / "actions/bad").exists()


def test_existing_sessions_keep_legacy_info_feedback_and_completed_responses(tmp_path, monkeypatch):
    directory = waveform_session(tmp_path, monkeypatch, feedback_fields=["observations"])
    path = directory / "session.json"
    config = json.loads(path.read_text())
    config.pop("observation_view_version")
    path.write_text(json.dumps(config))
    info = session.session_info(directory)
    assert "output_times" in info["manifest"]["transient"]
    assert "observation_view_version" not in info
    assert "evas_observe" not in {item["function"]["name"] for item in info["tools"]}
    response = call(directory, "sim", "evas_simulate")
    assert response["result"]["observations"] == waveform()
    assert "observation_artifact" not in response["result"]
    assert call(directory, "sim", "evas_simulate") == response
    assert not call(directory, "read", "evas_read_artifact", artifact_id="manifest")["ok"]


def test_reads_spend_only_action_budget_and_reserve_submission(tmp_path):
    directory = make_session(tmp_path, max_actions=3)
    call(directory, "write", "evas_write", path="dut.va", content="candidate")
    response = call(directory, "read", "evas_read_artifact", artifact_id="manifest")
    assert response["ok"]
    assert call(directory, "read", "evas_read_artifact", artifact_id="manifest") == response
    assert session.session_info(directory)["remaining_simulations"] == 1
    assert (
        call(directory, "read2", "evas_read_artifact", artifact_id="manifest")["error"]
        == "action_budget_exhausted"
    )
    assert call(directory, "submit", "evas_submit")["ok"]
    assert (
        call(directory, "read3", "evas_read_artifact", artifact_id="manifest")["error"]
        == "submission_frozen"
    )


def test_offline_bundle_serves_the_same_session_observation_contract(tmp_path):
    from circuit_harness.execution.bundle import build_cli

    directory = make_session(tmp_path)
    bundle = tmp_path / "chips.pyz"
    build_cli(bundle)
    code = """import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from circuit_harness.execution.current_evas_session import session_info,session_action
assert session_info(Path(sys.argv[2]))['observation_view_version']==1
print(json.dumps(session_action(Path(sys.argv[2]), {
 'action_id':'bundled','tool':'evas_read_artifact','arguments':{'artifact_id':'manifest'}})))
"""
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(bundle), str(directory)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["result"]["artifact"]["id"] == "manifest"
