"""Public temporary testbench actions freeze inputs without changing submission."""

import json

from test_benchmark_spectre import task_package
from test_current_evas_session import call

from circuit_harness.execution import current_evas_session as session


def test_temporary_netlist_and_va_do_not_enter_final_candidate(tmp_path, monkeypatch):
    package = task_package(tmp_path, purpose="public")
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["feedback_fields"] = ["diagnostics"]
    (package / "manifest.json").write_text(json.dumps(manifest))
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "instruction.md").write_text("Public fixture")
    task = {
        "task_id": "fixture-task",
        "task_version": "fixture-v1",
        "public_files": ["instruction.md"],
        "candidate_files": ["dut.va"],
        "feedback_fields": ["diagnostics"],
        "manifest": {"condition_id": manifest["condition_id"]},
        "testbench": {"version": "spectre-netlist-v1", "payload_file": ".public-testbench.json"},
    }
    directory = tmp_path / "session"
    session.create_session(
        task=task,
        materials=materials,
        checkout=None,
        kernel=None,
        directory=directory,
        image=None,
        backend="remote_spectre",
        public_task_package=package,
        public_remote={
            "host": "fixture",
            "python": "/python",
            "bundle": "/bundle",
            "profile": "/profile",
            "run_root": "/jobs",
            "archive_root": "/archive",
            "upload_root": "/upload",
        },
        max_simulations=2,
    )
    assert call(directory, "write", "evas_write", path="dut.va", content="model")["ok"]
    from circuit_harness.execution import remote_public

    observed = []

    def execute(**kwargs):
        payload = kwargs["candidate"] / "files/.public-testbench.json"
        observed.append(json.loads(payload.read_text()))
        return {"execution": "ok", "backend": "remote_spectre", "diagnostics": ["fixture"]}

    monkeypatch.setattr(remote_public, "run_remote_public", execute)
    spec = {
        "netlist": 'simulator lang=spectre\nahdl_include "probe.va"\n',
        "support_files": {"probe.va": "module probe; endmodule"},
    }
    response = call(directory, "custom", "evas_testbench", spec=json.dumps(spec))
    assert response["ok"] and observed == [spec]
    assert (
        call(
            directory,
            "bad",
            "evas_testbench",
            spec=json.dumps({"netlist": "", "support_files": {"../truth": "bad"}}),
        )["ok"]
        is False
    )
    session.close_session(directory, "completed")
    assert sorted(p.name for p in (directory / "candidate/files").iterdir()) == ["dut.va"]
