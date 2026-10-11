"""Constructed EVAS CLI processes, not real simulator evidence."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from circuit_harness.execution.current_evas import run_evas


@pytest.fixture
def evas_request(tmp_path):
    checkout = tmp_path / "checkout"
    package = checkout / "evas/src/evas"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text("""
import json, pathlib, subprocess, sys
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
kernel = sys.argv[sys.argv.index('--kernel')+1]
subprocess.run([kernel], check=True)
print(json.dumps({'engine':'constructed-evas', 'nodes':['0','z'],
 'solutions':[{'voltages':[0,t]} for t in manifest['transient']['output_times']],
 'transient':{'times':manifest['transient']['output_times'], 'events':[]}}))
""")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "constructed source",
        ],
        check=True,
    )
    kernel = checkout / "constructed-kernel"
    kernel.write_text(f"#!{sys.executable}\n")
    kernel.chmod(0o755)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "dut.va").write_text("// constructed input\n")
    manifest = inputs / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "models": ["dut.va"],
                "instances": [
                    {"name": "dut", "module": "example", "connections": {"z": "z", "r": "0"}}
                ],
                "transient": {"sources": {}, "output_times": [0, 1], "stop": 1, "max_step": 1},
                "tolerances": {"vabstol": 1e-8, "reltol": 0},
            }
        )
    )
    return dict(
        checkout=checkout,
        kernel=kernel,
        manifest=manifest,
        output=tmp_path / "run",
        timeout_s=5,
        max_output_bytes=1024 * 1024,
        python=sys.executable,
    )


def test_current_checkout_runs_frozen_input_and_preserves_evidence(evas_request):
    result = run_evas(**evas_request)
    output = evas_request["output"]
    assert result["execution"] == "ok"
    assert result["verdict"] == "not_evaluated"
    assert result["process"]["returncode"] == 0
    assert result["process"]["cleanup_confirmed"]
    assert json.loads((output / result["raw_result"]).read_text())["engine"] == "constructed-evas"
    receipt = json.loads((output / "request.json").read_text())
    assert receipt["checkout"] == str(evas_request["checkout"])
    assert receipt["kernel"]["sha256"]
    assert receipt["source"]["commit"]
    assert (output / "inputs/dut.va").read_text() == "// constructed input\n"
    assert (output / "execution/evas.stderr.log").is_file()
    assert result["artifacts"]["execution/evas.stdout.log"]["sha256"]
    with pytest.raises(FileExistsError):
        run_evas(**evas_request)


def test_missing_kernel_is_execution_failure_with_retained_receipt(evas_request):
    evas_request["kernel"] = evas_request["checkout"] / "missing-kernel"
    result = run_evas(**evas_request)
    assert result["execution"] == "infrastructure_error"
    assert result["verdict"] == "not_evaluated"
    assert result["process"]["returncode"] is None
    assert "missing-kernel" in result["reason"]
    assert (evas_request["output"] / "request.json").exists()


@pytest.mark.parametrize(
    "payload", ["{}", '{"engine":"x","nodes":["z"],"solutions":[],"transient":{"times":[0,1]}}']
)
def test_exit_zero_with_incomplete_output_cannot_be_graded(evas_request, payload):
    (evas_request["checkout"] / "evas/src/evas/__main__.py").write_text(f"print({payload!r})\n")
    result = run_evas(**evas_request)
    assert result["process"]["returncode"] == 0
    assert result["execution"] == "invalid_result"
    assert result["verdict"] == "not_evaluated"


def test_timeout_kills_kernel_group_and_keeps_logs(evas_request):
    import os
    import time

    evas_request["kernel"].write_text(
        f"#!{sys.executable}\nimport os,pathlib,time\n"
        "pathlib.Path('kernel.pid').write_text(str(os.getpid()))\ntime.sleep(20)\n"
    )
    evas_request["timeout_s"] = 2
    result = run_evas(**evas_request)
    assert result["execution"] == "timeout"
    assert result["verdict"] == "not_evaluated"
    pid = int((evas_request["output"] / "execution/kernel.pid").read_text())
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("kernel survived its process group's timeout")


def test_unknown_solver_option_is_rejected_before_execution(evas_request):
    path = evas_request["manifest"]
    config = json.loads(path.read_text())
    config["tolerances"]["iabstol"] = 1e-18
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="iabstol"):
        run_evas(**evas_request)
    assert not evas_request["output"].exists()


def test_ambient_evas_controls_cannot_change_request_or_write_outside_run(
    evas_request, monkeypatch
):
    outside = evas_request["output"].parent / "ambient-diagnostics.json"
    monkeypatch.setenv("EVAS_DIAGNOSTICS_PATH", str(outside))
    monkeypatch.setenv("EVAS_STATIC_THREADS", "invalid-ambient-setting")
    evas_request["kernel"].write_text(
        f"#!{sys.executable}\nimport os,pathlib\n"
        "if 'EVAS_DIAGNOSTICS_PATH' in os.environ:\n"
        " pathlib.Path(os.environ['EVAS_DIAGNOSTICS_PATH']).write_text('outside output')\n"
        "assert os.environ.get('EVAS_STATIC_THREADS', '1') == '1'\n"
    )
    result = run_evas(**evas_request)
    assert result["execution"] == "ok"
    assert not outside.exists()
    receipt = json.loads((evas_request["output"] / "request.json").read_text())
    assert receipt["environment"]["EVAS_STATIC_THREADS"] == "1"


@pytest.mark.parametrize("dependency", ["journal", "process"])
def test_dependency_cached_before_adapter_import_cannot_misidentify_source(
    evas_request, tmp_path, dependency
):
    from circuit_harness.execution.current_evas import HARNESS_SOURCE

    original = Path(__file__).resolve().parents[2]
    isolated = tmp_path / "harness"
    for name in (
        *HARNESS_SOURCE,
        "circuit_harness/execution/simulator.py",
        "circuit_harness/execution/runtime/simulator.py",
    ):
        target = isolated / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, target)
    subprocess.run(["git", "init", "-q", str(isolated)], check=True)
    subprocess.run(["git", "-C", str(isolated), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(isolated),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "isolated fixture",
        ],
        check=True,
    )
    code = """
import importlib, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
dependency = importlib.import_module('circuit_harness.execution.' + sys.argv[2])
path = pathlib.Path(dependency.__file__)
path.write_text(path.read_text() + '\\n# constructed edit before adapter load\\n')
from circuit_harness.execution.current_evas import run_evas
try:
    run_evas(**json.loads(sys.argv[3]))
except ValueError as error:
    print(str(error))
else:
    raise AssertionError('cached dependency was recorded as the new source')
"""
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            code,
            str(isolated),
            dependency,
            json.dumps(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in evas_request.items()
                }
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stderr
    assert "source identity changed after module load" in child.stdout
    assert not (evas_request["output"] / "execution").exists()


def test_loaded_harness_identity_uses_actual_source_module_bytes():
    from circuit_harness.execution import current_evas
    from circuit_harness.execution.journal import file_digest

    root = Path(current_evas.__file__).resolve().parents[3]
    assert current_evas._LOADED_HARNESS == {
        name: file_digest(root / name) for name in current_evas.HARNESS_SOURCE
    }


def test_zipimport_harness_identity_uses_actual_packaged_module_bytes(tmp_path):
    import hashlib
    from zipfile import ZipFile

    from circuit_harness.execution.bundle import build_cli
    from circuit_harness.execution.current_evas import HARNESS_SOURCE

    bundle = tmp_path / "cli.pyz"
    build_cli(bundle)
    code = "\n".join(
        [
            "import json, sys",
            "sys.path.insert(0, sys.argv[1])",
            "from circuit_harness.execution.current_evas import _LOADED_HARNESS",
            "print(json.dumps(_LOADED_HARNESS))",
        ]
    )
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(bundle)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    with ZipFile(bundle) as archive:
        expected = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in HARNESS_SOURCE}
    assert json.loads(process.stdout) == expected


def test_versioned_error_survives_untruncated_with_safe_metadata(evas_request):
    payload = dict(
        diagnostic_version=1,
        kind="compile_error",
        code="unsupported_timer_dependency",
        category="unsupported",
        stage="lowering",
        capability="TIMER",
        message="private/" + "x" * 5000,
        location={"source": "/private/model.va", "line": 3},
        raw_payload={"secret": 1},
    )
    (evas_request["checkout"] / "evas/src/evas/__main__.py").write_text(
        "import sys\nsys.stderr.write(" + repr(json.dumps(payload)) + ")\nraise SystemExit(2)\n"
    )
    result = run_evas(**evas_request)
    assert result["execution"] == "backend_error"
    assert result["verdict"] == "not_evaluated"
    assert result["diagnostic"] == dict(
        diagnostic_version=1,
        code="unsupported_timer_dependency",
        category="unsupported",
        stage="lowering",
        capability="TIMER",
    )
    raw = evas_request["output"] / "execution/evas.stderr.log"
    assert json.loads(raw.read_text()) == payload
    assert result["artifacts"]["execution/evas.stderr.log"]["sha256"]


@pytest.mark.parametrize(
    "payload, version",
    [
        ("plain diagnostic", None),
        ("{}", None),
        ('{"diagnostic_version":2,"category":"unsupported","kind":"unsupported_timer"}', 2),
        ('{"diagnostic_version":true}', None),
        ('{"diagnostic_version":1,"diagnostic_version":2}', None),
        (
            json.dumps(
                dict(
                    diagnostic_version=1,
                    kind="compile_error",
                    code="code",
                    stage="/private/path",
                    category="unsupported",
                    message="hidden",
                )
            ),
            1,
        ),
        (
            json.dumps(
                dict(
                    diagnostic_version=1,
                    kind="compile_error",
                    code="code",
                    stage="compile",
                    category=["unsupported"],
                    message="hidden",
                )
            ),
            1,
        ),
        pytest.param("x" * (1024 * 1024 + 1), None, id="oversized"),
    ],
)
def test_unknown_diagnostic_does_not_change_failed_execution(evas_request, payload, version):
    evas_request["max_output_bytes"] = 4 * 1024 * 1024
    (evas_request["checkout"] / "evas/src/evas/__main__.py").write_text(
        "import sys\nsys.stderr.write(" + repr(payload) + ")\nraise SystemExit(2)\n"
    )
    result = run_evas(**evas_request)
    assert result["execution"] == "backend_error"
    assert result["verdict"] == "not_evaluated"
    assert result["diagnostic"] == dict(
        diagnostic_version=version, code=None, category="unknown", stage=None, capability=None
    )
    assert (evas_request["output"] / "execution/evas.stderr.log").read_text() == payload
