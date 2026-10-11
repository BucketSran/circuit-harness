"""Opt-in current EVAS consumption; acceptance controls are not benchmark scores."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from circuit_harness.execution.backends.current_evas import read_diagnostic, run_evas
from circuit_harness.execution.evaluation.benchmark_replay import (
    replay_candidate,
    summarize_replays,
    verify_replay,
)
from circuit_harness.execution.evaluation.benchmark_spectre import package_identity
from circuit_harness.execution.evaluation.candidate_bundle import freeze_candidate
from circuit_harness.execution.sessions.current_evas_session import (
    close_session,
    create_session,
    session_action,
)

# Explicit operator-selected source and Linux kernel; no fixture engine substitution.
_REQUIRED = ("CHIPS_TEST_EVAS_CHECKOUT", "CHIPS_TEST_EVAS_LINUX_KERNEL", "CHIPS_TEST_DOCKER_IMAGE")
pytestmark = pytest.mark.skipif(
    not all(os.environ.get(key) for key in _REQUIRED),
    reason="requires real EVAS checkout, built Linux kernel and immutable Docker image",
)

CASES = [
    (
        "timer",
        "integer n;",
        "@(initial_step) n=0; @(timer(V(u,r),0,1e-12)) n=n+1; V(y,r)<+n;",
        1,
        "unsupported_timer_dependency",
        "unsupported",
        "lowering",
        "TIMER",
    ),
    (
        "dae",
        "real q;",
        "@(initial_step) q=1; @(timer(0.5,0,1e-12)) q=2; V(y,r)<+idt(q+2*V(y,r),0)-pow(V(y,r),2);",
        1,
        "kernel.unsupported_implicit_dynamics",
        "unsupported",
        "kernel",
        "DYNAMICS",
    ),
    (
        "overflow",
        "",
        "V(y,r)<+pow(V(u,r),3);",
        1e200,
        "kernel.nonfinite_arithmetic",
        "numerical",
        "kernel",
        None,
    ),
    (
        "integer",
        "integer n;",
        "@(initial_step) n=2147483647; @(timer(0,0,1e-12)) n=n+1; V(y,r)<+n;",
        1,
        "kernel.state_range",
        "unsupported",
        "kernel",
        None,
    ),
    ("success", "", "V(y,r)<+2*V(u,r);", 1, None, None, None, None),
    ("infrastructure", "", "V(y,r)<+2*V(u,r);", 1, None, None, None, None),
]


def _json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _package(directory, checkout, kernel, manifest, name):
    """Benchmark-declared transport probe, with no correctness claim on failures."""
    directory.mkdir()
    shutil.copytree(checkout / "evas/src", directory / "engine")
    shutil.copyfile(kernel, directory / "evas-kernel")
    (directory / "evas-kernel").chmod(0o755)
    _json(directory / "simulation.json", manifest)
    (directory / "test.sh").write_text("#!/bin/sh\nexec python3 -I -S -B /task/probe.py\n")
    # Real EVAS emits stderr unchanged. The maintained consumer parses those saved
    # bytes on the host after verify_replay, rather than interpreting checker text.
    (directory / "probe.py").write_text("""import hashlib,json,os,pathlib,subprocess,sys
out=pathlib.Path(os.environ['VERIFY_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
candidate=pathlib.Path(os.environ['CANDIDATE'])
manifest=json.loads(pathlib.Path('/task/simulation.json').read_text())
manifest['models']=[str(candidate)]
(out/'simulation.json').write_text(json.dumps(manifest))
code=("import runpy,sys;sys.path.insert(0,'/task/engine');"
      "runpy.run_module('evas',run_name='__main__')")
if pathlib.Path('/task/lint-control').exists():
    lint=subprocess.run([sys.executable,'-I','-S','-B','-c',code,'lint',str(out/'simulation.json')],capture_output=True,timeout=30)
    (out/'lint.stdout.log').write_bytes(lint.stdout)
    (out/'lint.stderr.log').write_bytes(lint.stderr)
    assert lint.returncode==0
with (out/'evas.stdout.log').open('wb') as stdout,(out/'evas.stderr.log').open('wb') as stderr:
    p=subprocess.run([sys.executable,'-I','-S','-B','-c',code,'transient',str(out/'simulation.json'),'--kernel','/task/evas-kernel','--timeout','30'],stdout=stdout,stderr=stderr,timeout=35)
report={'status':'completed','reward':1 if p.returncode==0 else 0,
        'cases':[{'status':'graded' if p.returncode==0 else 'simulation_failed',
                  'passed':p.returncode==0}],
        'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
        'solver_options':json.loads(os.environ['CHIPS_SOLVER_OPTIONS']),
        'simulation_returncode':p.returncode,'acceptance_control':True}
(out/'report.json').write_text(json.dumps(report))
""")
    if name == "dae":
        (directory / "lint-control").write_text(
            "Require static precheck success before dynamic failure.\n"
        )
    files = {
        path.relative_to(directory).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    _json(
        directory / "manifest.json",
        dict(
            schema_version=1,
            task_id=name,
            task_version="diagnostic-acceptance-v1",
            criteria_sha256=hashlib.sha256(
                b"diagnostic transport control, no circuit grading"
            ).hexdigest(),
            condition_id=name,
            task_set="extension",
            purpose="final",
            entrypoint="test.sh",
            candidate_file="dut.va",
            report_path="verifier/report.json",
            files=files,
            feedback_fields=[],
        ),
    )


def run_acceptance(output, checkout, linux_kernel, image):
    output.mkdir(parents=True, exist_ok=False)
    replays = []
    records = []
    for name, declarations, body, level, code, category, stage, capability in CASES:
        root = output / name
        inputs = root / "inputs"
        inputs.mkdir(parents=True)
        source = (
            '`include "disciplines.vams"\nmodule m(u,y,r); input u; output y; inout r; '
            f"electrical u,y,r; {declarations} analog begin {body} end endmodule\n"
        )
        (inputs / "dut.va").write_text(source)
        manifest = dict(
            models=["dut.va"],
            instances=[dict(name="dut", module="m", connections=dict(u="u", y="y", r="0"))],
            transient=dict(
                sources={"u": [[0, level], [1, level]]}, output_times=[0, 1], stop=1, max_step=1
            ),
            tolerances=dict(vabstol=1e-8, reltol=0),
        )
        _json(inputs / "manifest.json", manifest)
        infra = name == "infrastructure"
        # No image pull: the deliberately absent local image makes the actual
        # Docker launch fail. The same source/kernel and budgets remain recorded.
        selected_image = "sha256:" + "0" * 64 if infra else image
        direct = run_evas(
            checkout=checkout,
            kernel=root / "absent-kernel"
            if infra
            else checkout / "evas/rust_core/target/debug/evas-kernel",
            manifest=inputs / "manifest.json",
            output=root / "direct",
            timeout_s=30,
            max_output_bytes=2**20,
            python=sys.executable,
        )
        materials = root / "materials"
        materials.mkdir()
        (materials / "instruction.md").write_text("Diagnostic transport acceptance control.\n")
        session = root / "session"
        create_session(
            task=dict(
                task_id=name,
                task_version="diagnostic-acceptance-v1",
                public_files=["instruction.md"],
                candidate_files=["dut.va"],
                feedback_fields=["diagnostic"],
                manifest=manifest,
            ),
            materials=materials,
            checkout=checkout,
            kernel=linux_kernel,
            directory=session,
            image=selected_image,
            timeout_s=45,
            max_simulations=1,
        )

        def call(action, tool, session=session, **arguments):
            return session_action(session, dict(action_id=action, tool=tool, arguments=arguments))

        assert call("write", "evas_write", path="dut.va", content=source)["ok"]
        response = call("simulate", "evas_simulate")
        assert response == call("simulate", "evas_simulate")
        assert response["ok"], response
        public = response["result"]
        close = close_session(session, "completed")
        assert close["state"] == "collected"
        package = root / "package"
        _package(package, checkout, linux_kernel, manifest, name)
        freeze_candidate(
            inputs,
            root / "candidate",
            ["dut.va"],
            task_id=name,
            task_version="diagnostic-acceptance-v1",
            reason="fixed diagnostic acceptance set",
        )
        config = dict(
            schema_version=1,
            image=selected_image,
            task_package_sha256=package_identity(package, purpose="final")["sha256"],
            solver_options={},
            unsupported=[],
            timeout_s=45,
            max_output_bytes=2**20,
        )
        receipt = replay_candidate(root / "candidate", package, config, root / "replay")
        assert verify_replay(root / "replay") == receipt
        replays.append(root / "replay")
        replay_log = root / "replay/run/work/verifier/evas.stderr.log"
        if name == "dae":
            lint = json.loads((replay_log.parent / "lint.stdout.log").read_text())
            assert lint["status"] == "lint_passed"
        metadata = read_diagnostic(replay_log) if code else None
        record = dict(
            case=name,
            direct=direct,
            public=public,
            close=close,
            replay_result=receipt["result"],
            classification=receipt["classification"],
            diagnostic=metadata,
        )
        records.append(record)
        _json(output / "records.json", records)
        expected = "infrastructure_error" if infra else "backend_error" if code else "ok"
        assert direct["execution"] == public["execution"] == expected, record
        assert public["task_correctness"] == "not_evaluated"
        assert "diagnostics" not in public
        if code:
            expected_metadata = dict(
                diagnostic_version=1,
                code=code,
                category=category,
                stage=stage,
                capability=capability,
            )
            assert direct["diagnostic"] == public["diagnostic"] == metadata == expected_metadata
            raw = json.loads(replay_log.read_text())
            assert raw["message"]
            if name == "timer":
                assert raw["location"]["line"] == 2
                assert raw["location"]["column"] == 117
                assert raw["instance"] == "dut"
            if name == "overflow":
                assert raw["sample"] == 0
            assert receipt["classification"] == "unevaluable"
            assert receipt["result"]["execution"] == "unclassified_failure"
            assert receipt["result"]["score"] is None
        else:
            assert "diagnostic" not in public
            assert receipt["classification"] == ("infra" if infra else "not_compared")
            assert receipt["result"]["score"] == (None if infra else 1)
    summary = summarize_replays(replays)
    _json(output / "summary.json", summary)
    assert summary["extension"] == dict(
        records=6,
        candidates=6,
        match=0,
        false_accept=0,
        false_reject=0,
        infra=1,
        unevaluable=4,
        not_compared=1,
    )
    session_config = json.loads((output / "success/session/session.json").read_text())
    lint = json.loads((output / "dae/replay/run/work/verifier/lint.stdout.log").read_text())
    harness_root = Path(__file__).resolve().parents[2]
    identity = dict(
        harness_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=harness_root, text=True
        ).strip(),
        harness_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=harness_root)
        ).hexdigest(),
        acceptance_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source=session_config["source"],
        linux_kernel_sha256=hashlib.sha256(linux_kernel.read_bytes()).hexdigest(),
        native_kernel_sha256=hashlib.sha256(
            (checkout / "evas/rust_core/target/debug/evas-kernel").read_bytes()
        ).hexdigest(),
        image=image,
        ir_schema_version=lint["schema_version"],
        diagnostic_version=1,
        summary=summary,
        scope="six diagnostic transport controls, no circuit grading or Spectre comparison",
        replay_receipts={
            name: hashlib.sha256((output / name / "replay/receipt.json").read_bytes()).hexdigest()
            for name, *_ in CASES
        },
    )
    _json(output / "identity.json", identity)
    return summary


def test_real_docker_diagnostics_keep_all_six_replay_records(tmp_path):
    run_acceptance(
        tmp_path / "acceptance",
        Path(os.environ[_REQUIRED[0]]),
        Path(os.environ[_REQUIRED[1]]),
        os.environ[_REQUIRED[2]],
    )
