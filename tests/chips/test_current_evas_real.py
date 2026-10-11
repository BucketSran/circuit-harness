"""Opt-in real EVAS diagnostic consumption, not waveform or benchmark scoring."""

import json
import os
import sys
from pathlib import Path

import pytest

from circuit_harness.execution.backends.current_evas import run_evas
from circuit_harness.execution.sessions.current_evas_session import (
    close_session,
    create_session,
    session_action,
)

pytestmark = pytest.mark.skipif(
    not (os.environ.get("CHIPS_TEST_EVAS_CHECKOUT") and os.environ.get("CHIPS_TEST_NATIVE_CODEX")),
    reason="requires selected EVAS checkout/built native kernel and Codex sandbox binary",
)


@pytest.mark.parametrize(
    "declarations,body,level,code,category",
    [
        (
            "integer n;",
            "@(initial_step) n=0; @(timer(V(u,r),0,1e-12)) n=n+1; V(y,r)<+n;",
            1,
            "unsupported_timer_dependency",
            "unsupported",
        ),
        (
            "real q;",
            "@(initial_step) q=1; @(timer(0.5,0,1e-12)) q=2; "
            "V(y,r)<+idt(q+2*V(y,r),0)-pow(V(y,r),2);",
            1,
            "kernel.unsupported_implicit_dynamics",
            "unsupported",
        ),
        ("", "V(y,r)<+pow(V(u,r),3);", 1e200, "kernel.nonfinite_arithmetic", "numerical"),
        (
            "integer n;",
            "@(initial_step) n=2147483647; @(timer(0,0,1e-12)) n=n+1; V(y,r)<+n;",
            1,
            "kernel.state_range",
            "unsupported",
        ),
        ("", "V(y,r)<+2*V(u,r);", 1, None, None),
    ],
    ids=["timer-dependency", "dae-event", "overflow", "integer-range", "success"],
)
def test_real_error_reaches_public_session_without_changing_execution(
    tmp_path, declarations, body, level, code, category
):
    checkout = Path(os.environ["CHIPS_TEST_EVAS_CHECKOUT"]).resolve()
    kernel = checkout / "evas/rust_core/target/debug/evas-kernel"
    source = (
        '`include "disciplines.vams"\nmodule m(u,y,r); input u; output y; inout r; '
        f"electrical u,y,r; {declarations} analog begin {body} end endmodule\n"
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "dut.va").write_text(source)
    manifest = dict(
        models=["dut.va"],
        instances=[dict(name="dut", module="m", connections=dict(u="u", y="y", r="0"))],
        transient=dict(
            sources={"u": [[0, level], [1, level]]}, output_times=[0, 1], stop=1, max_step=1
        ),
        tolerances=dict(vabstol=1e-8, reltol=0),
    )
    (inputs / "manifest.json").write_text(json.dumps(manifest))
    direct = run_evas(
        checkout=checkout,
        kernel=kernel,
        manifest=inputs / "manifest.json",
        output=tmp_path / "direct",
        timeout_s=30,
        max_output_bytes=2**20,
        python=sys.executable,
    )
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "instruction.md").write_text("Diagnostic acceptance, no task score.\n")
    directory = tmp_path / "session"
    create_session(
        task=dict(
            task_id="diagnostic",
            task_version="v1",
            public_files=["instruction.md"],
            candidate_files=["dut.va"],
            feedback_fields=["diagnostic"],
            manifest=manifest,
        ),
        materials=materials,
        checkout=checkout,
        kernel=kernel,
        directory=directory,
        backend="native_codex_sandbox",
        codex=Path(os.environ["CHIPS_TEST_NATIVE_CODEX"]),
        python=sys.executable,
        timeout_s=30,
        max_simulations=1,
    )

    def call(action, tool, **arguments):
        return session_action(directory, dict(action_id=action, tool=tool, arguments=arguments))

    assert call("write", "evas_write", path="dut.va", content=source)["ok"]
    response = call("simulate", "evas_simulate")
    assert response == call("simulate", "evas_simulate")
    assert response["ok"], response
    public = response["result"]
    assert direct["execution"] == public["execution"] == ("backend_error" if code else "ok")
    if code:
        assert direct["diagnostic"] == public["diagnostic"]
        assert public["diagnostic"]["code"] == code
        assert public["diagnostic"]["category"] == category
    else:
        assert "diagnostic" not in public
    assert public["task_correctness"] == "not_evaluated"
    assert "diagnostics" not in public
    assert close_session(directory, "completed")["state"] == "collected"
