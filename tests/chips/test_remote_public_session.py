"""Configuration B protocol fixtures, with no SSH or licensed simulator."""

import json

import pytest
from test_benchmark_spectre import task_package
from test_current_evas_session import call, make_session

from circuit_harness.execution import current_evas_session as session


def test_remote_session_reuses_public_budget_and_freezes(tmp_path, monkeypatch):
    package = task_package(tmp_path, purpose="public")
    manifest = json.loads((package / "manifest.json").read_text())
    manifest.update(task_id="synthetic", feedback_fields=["diagnostics"])
    (package / "manifest.json").write_text(json.dumps(manifest))
    remote = dict(
        host="fixture",
        python="/bin/python",
        bundle="/bundle.pyz",
        profile="/public.json",
        run_root="/public/jobs",
        archive_root="/public/archive",
        upload_root="/public/upload",
    )
    directory = make_session(
        tmp_path,
        backend="remote_spectre",
        image=None,
        public_task_package=package,
        public_remote=remote,
        manifest={"condition_id": "fixture-condition-v1"},
    )
    assert call(directory, "edit", "evas_write", path="dut.va", content="original")["ok"]
    from circuit_harness.execution import remote_public

    seen = []

    def execute(**kwargs):
        seen.append(kwargs["action_id"])
        return {
            "execution": "ok",
            "backend": "remote_spectre",
            "diagnostics": ["public"],
            "reward": 1,
            "score": 1,
            "wait_elapsed_s": 0.1,
        }

    monkeypatch.setattr(remote_public, "run_remote_public", execute)
    result = call(directory, "sim", "evas_simulate")
    assert result["ok"] and result["result"]["diagnostics"] == ["public"]
    assert "score" not in result["result"] and "reward" not in result["result"]
    assert result["result"]["wait_elapsed_s"] == 0.1
    assert call(directory, "sim", "evas_simulate") == result
    assert seen == ["sim"]
    assert call(directory, "extra", "evas_simulate")["error"] == "simulation_budget_exhausted"
    frozen = session.close_session(directory, "timeout")
    assert (directory / "candidate/files/dut.va").read_text() == "original"
    assert frozen["task_correctness"] == "not_evaluated"


def test_unknown_remote_job_resumes_without_submit_and_late_result_cannot_change_freeze(
    tmp_path, monkeypatch
):
    from circuit_harness.execution import remote_public

    package = task_package(tmp_path, purpose="public")
    manifest = json.loads((package / "manifest.json").read_text())
    manifest.update(task_id="synthetic", feedback_fields=["diagnostics"])
    (package / "manifest.json").write_text(json.dumps(manifest))
    remote = dict(
        host="fixture",
        python="/bin/python",
        bundle="/bundle.pyz",
        profile="/public.json",
        run_root="/public/jobs",
        archive_root="/public/archive",
        upload_root="/public/upload",
    )
    directory = make_session(
        tmp_path,
        backend="remote_spectre",
        image=None,
        public_task_package=package,
        public_remote=remote,
        manifest={"condition_id": "fixture-condition-v1"},
    )
    submitted, queried = [], []
    ready = False

    class Transport:
        def __init__(self, config, evidence, **kwargs):
            pass

        def submit(self, candidate, package, job_id):
            submitted.append(job_id)
            raise ConnectionError("unknown SSH handoff")

        def query(self, job_id):
            queried.append(job_id)
            return {"state": "finished" if ready else "missing"}

        def retrieve(self, job_id):
            return {
                "purpose": "public",
                "execution": "ok",
                "candidate_sha256": "old",
                "feedback": {"diagnostics": ["old result"]},
            }

    monkeypatch.setattr(remote_public, "RemotePublicSpectre", Transport)
    assert call(directory, "write", "evas_write", path="dut.va", content="before")["ok"]
    assert call(directory, "sim", "evas_simulate")["error"] == "unknown_execution"
    assert call(directory, "sim", "evas_simulate")["retry_safe"] is False
    assert call(directory, "new", "evas_simulate")["error"] == "unresolved_previous_simulation"
    assert session.session_info(directory)["remaining_simulations"] == 0
    assert call(directory, "edit", "evas_write", path="dut.va", content="after")["ok"]
    frozen = session.close_session(directory, "timeout")
    ready = True
    reply = call(directory, "sim", "evas_simulate")
    assert reply["result"]["diagnostics"] == ["old result"]
    assert submitted == queried[:1] and len(submitted) == 1
    assert session.close_session(directory, "timeout") == frozen
    assert (directory / "candidate/files/dut.va").read_text() == "after"


def test_public_transport_bounds_every_network_operation_by_remaining_wait(tmp_path, monkeypatch):
    import time

    from circuit_harness.execution.remote_public import RemotePublicSpectre
    from circuit_harness.execution.session_transport import RemoteSessionTransport

    remote = dict(
        host="fixture",
        python="/bin/python",
        bundle="/bundle.pyz",
        profile="/public.json",
        run_root="/public/jobs",
        archive_root="/public/archive",
        upload_root="/public/upload",
    )
    observed = []

    def cli(self, *args, timeout=30, **kwargs):
        observed.append(timeout)
        time.sleep(0.02)
        return {}

    def download(self, remote, destination, *, timeout=90, **kwargs):
        observed.append(timeout)

    monkeypatch.setattr(RemoteSessionTransport, "cli", cli)
    monkeypatch.setattr(RemoteSessionTransport, "download", download)
    transport = RemotePublicSpectre(remote, tmp_path / "evidence", deadline=time.monotonic() + 0.04)
    transport.cli("stage")
    transport.download("/archive", tmp_path / "receipt")
    assert 0 < observed[1] < observed[0] <= 0.04
    time.sleep(0.03)
    import pytest

    with pytest.raises(TimeoutError):
        transport.cli("query")


@pytest.mark.parametrize("receipt", ["{}", "[]", '{"package":null}'])
def test_malformed_downloaded_archive_is_durable_invalid_result(tmp_path, monkeypatch, receipt):
    from circuit_harness.execution.remote_public import RemotePublicSpectre

    package = task_package(tmp_path, purpose="public")
    manifest = json.loads((package / "manifest.json").read_text())
    manifest.update(task_id="synthetic", feedback_fields=["diagnostics"])
    (package / "manifest.json").write_text(json.dumps(manifest))
    remote = dict(
        host="fixture",
        python="/python",
        bundle="/bundle",
        profile="/public/profile",
        run_root="/public/jobs",
        archive_root="/public/archive",
        upload_root="/public/upload",
    )
    directory = make_session(
        tmp_path,
        backend="remote_spectre",
        image=None,
        public_task_package=package,
        public_remote=remote,
        manifest={"condition_id": "fixture-condition-v1"},
    )
    operations = []

    def cli(self, *arguments, **kwargs):
        operations.append(arguments[0])
        if arguments[0] == "stage-benchmark":
            return {"candidate": "/stage/candidate", "task_package": "/stage/task-package"}
        return {"state": "finished"}

    def download(self, remote_path, destination, **kwargs):
        destination.write_text(receipt)

    monkeypatch.setattr(RemotePublicSpectre, "cli", cli)
    monkeypatch.setattr(RemotePublicSpectre, "download", download)
    assert call(directory, "write", "evas_write", path="dut.va", content="candidate")["ok"]
    reply = call(directory, "simulation", "evas_simulate")
    assert reply["result"]["execution"] == "invalid_result"
    assert "score" not in reply["result"]
    assert call(directory, "simulation", "evas_simulate") == reply
    assert operations.count("submit-benchmark-spectre") == 1
    assert json.loads((directory / "actions/simulation/response.json").read_text()) == reply
