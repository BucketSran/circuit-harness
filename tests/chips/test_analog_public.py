"""Public Analog RLC candidate boundary; constructed netlists, no simulator."""

import json

import pytest

from alphaapollo.common.execution.chips import analog_design_bench as adb
from alphaapollo.common.execution.chips.analog_public import (
    run_public_rlc,
    validate_rlc_candidate,
)


def test_accepts_bounded_passive_subcircuit() -> None:
    candidate = """* user circuit
.subckt rlc_rf_bandpass IN OUT COM
L1 IN mid 10n
C1 mid COM 2.5p
R1 mid OUT 50
.ends rlc_rf_bandpass
"""
    assert validate_rlc_candidate(candidate) == {"components": 3}


@pytest.mark.parametrize(
    ("task_id", "subcircuit"),
    [
        ("rlc-rf-bandpass-100mhz", "rlc_rf_bandpass"),
        ("rlc-broadband-50-to-200-match", "rlc_broadband_match"),
    ],
)
def test_candidate_interface_is_owned_by_the_selected_task(task_id, subcircuit):
    candidate = f".subckt {subcircuit} IN OUT COM\nR1 IN OUT 50\n.ends {subcircuit}\n"
    assert validate_rlc_candidate(candidate, task_id=task_id) == {"components": 1}
    wrong = candidate.replace("IN OUT COM", "OUT IN COM")
    with pytest.raises(ValueError, match="declared"):
        validate_rlc_candidate(wrong, task_id=task_id)
    other = "rlc_broadband_match" if subcircuit == "rlc_rf_bandpass" else "rlc_rf_bandpass"
    with pytest.raises(ValueError, match="declared"):
        validate_rlc_candidate(candidate.replace(subcircuit, other), task_id=task_id)


@pytest.mark.parametrize("task_id", ["unknown", "sky130-ota-5t-gain40-pm60-noise50uv-pvt"])
def test_passive_session_refuses_tasks_without_a_public_rlc_contract(task_id):
    with pytest.raises(ValueError, match="public RLC"):
        validate_rlc_candidate("* no candidate", task_id=task_id)


@pytest.mark.parametrize(
    "candidate",
    [
        ".subckt rlc_rf_bandpass IN OUT COM\n.include /private/gold.spi\n.ends rlc_rf_bandpass\n",
        ".subckt rlc_rf_bandpass IN OUT COM\n.control\nshell id\n.endc\n.ends rlc_rf_bandpass\n",
        ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT -10\n.ends rlc_rf_bandpass\n",
        ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 0\n.ends rlc_rf_bandpass\n",
        ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 1e999\n.ends rlc_rf_bandpass\n",
        ".subckt rlc_rf_bandpass IN OUT COM\nV1 IN OUT 1\n.ends rlc_rf_bandpass\n",
        ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 10\nR1 OUT COM 10\n.ends rlc_rf_bandpass\n",
        ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 10\n.ends wrong_name\n",
        ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 10\n",
        ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 10\n.ends rlc_rf_bandpass\nR2 OUT COM 10\n",
    ],
)
def test_rejects_non_passive_or_malformed_candidate(candidate: str) -> None:
    with pytest.raises(ValueError):
        validate_rlc_candidate(candidate)


def test_public_runner_exposes_only_public_benches_and_no_model_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = "rlc-rf-bandpass-100mhz"
    source = tmp_path / "source/tasks" / task_id
    public = source / "environment/starter/testbench"
    public.mkdir(parents=True)
    (public / "tb_ac.spi").write_text(
        '.include "/app/circuit.spi"\nmeas ac gain_100mhz_db find x at=100Meg\n'
    )
    (public / "tb_stopband.spi").write_text(
        '.include "/app/circuit.spi"\nmeas ac gain_90mhz_db find x at=90Meg\n'
    )
    (source / "tests").mkdir()
    (source / "tests/secret.txt").write_text("hidden score")
    (source / "solution").mkdir()
    (source / "solution/circuit.spi").write_text("hidden reference")
    monkeypatch.setitem(
        adb.TASKS,
        task_id,
        adb.Task(
            "test-commit",
            adb.tree_digest(source),
            "test-image@sha256:" + "a" * 64,
            adb.TASKS[task_id].public_rlc,
        ),
    )
    candidate = tmp_path / "candidate.spi"
    candidate.write_text(
        ".subckt rlc_rf_bandpass IN OUT COM\nR1 IN OUT 50\n.ends rlc_rf_bandpass\n"
    )
    fake_podman = tmp_path / "fake-podman"
    fake_podman.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, re, sys\n"
        "args = sys.argv[1:]\n"
        "assert not any('/solution' in a or '/tests' in a for a in args)\n"
        "assert not os.environ.get('CHIPS_MODEL_KEY')\n"
        "mount = next(a for a in args if a.endswith(':/app/public:ro')).split(':')[0]\n"
        "for bench in pathlib.Path(mount).glob('*.spi'):\n"
        "    for name in re.findall(r'meas ac ([a-z_0-9]+)', bench.read_text()):\n"
        "        print(f'{name} = -1.0e+00')\n"
    )
    fake_podman.chmod(0o700)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "private-fixture-key")
    output = tmp_path / "public-run"
    result = run_public_rlc(tmp_path / "source", candidate, output, podman=str(fake_podman))
    assert result["state"] == "simulated"
    assert result["authority"] == "public_diagnostic"
    assert result["task_correctness"] == "not_evaluated"
    assert result["measurements"] == {"gain_100mhz_db": -1.0, "gain_90mhz_db": -1.0}
    assert (output / "inputs/circuit.spi").read_text() == candidate.read_text()
    assert not (output / "tests").exists()
    assert not (output / "solution").exists()
    fake_podman.write_text("#!/usr/bin/env python3\nprint('simulator returned no measurements')\n")
    missing = run_public_rlc(
        tmp_path / "source", candidate, tmp_path / "missing", podman=str(fake_podman)
    )
    assert missing["state"] == "simulation_error"
    assert missing["missing_measurements"] == ["gain_100mhz_db", "gain_90mhz_db"]
    assert "score" not in missing
    (source / "tests/secret.txt").write_text("changed hidden scorer")
    with pytest.raises(ValueError, match="source pin mismatch"):
        run_public_rlc(tmp_path / "source", candidate, tmp_path / "drift", podman=str(fake_podman))
    assert not (tmp_path / "drift").exists()


def test_public_runner_rejects_directive_before_creating_output(tmp_path) -> None:
    candidate = tmp_path / "candidate.spi"
    candidate.write_text(
        ".subckt rlc_rf_bandpass IN OUT COM\n.include /private/gold.spi\n.ends rlc_rf_bandpass\n"
    )
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="only R/L/C"):
        run_public_rlc(tmp_path / "source", candidate, output)
    assert not output.exists()


def test_broadband_runner_uses_pinned_analyzer_and_requires_complete_sweep(tmp_path, monkeypatch):
    """Constructed Podman/log boundary; this does not execute ngspice."""
    task_id = "rlc-broadband-50-to-200-match"
    source = tmp_path / "source/tasks" / task_id
    bench = source / "environment/starter/testbench"
    bench.mkdir(parents=True)
    (bench / "tb_ac.spi").write_text(
        '.include "circuit_finite_q.spi"\n.control\nac lin 11 3.30G 3.80G\n'
        "print gamma\nprint transducer_gain\n.endc\n.end\n"
    )
    analyzer = "# constructed analyzer fixture\n"
    (bench / "analyze_broadband.py").write_text(analyzer)
    (source / "tests").mkdir()
    (source / "tests/private.txt").write_text("HIDDEN_SCORER")
    task = adb.TASKS[task_id]
    monkeypatch.setitem(
        adb.TASKS,
        task_id,
        adb.Task("fixture", adb.tree_digest(source), task.image, task.public_rlc),
    )
    candidate = tmp_path / "candidate.spi"
    candidate.write_text(
        ".subckt rlc_broadband_match IN OUT COM\nL1 IN OUT 1n\n.ends rlc_broadband_match\n"
    )
    transcript = ""
    for name, value in (("gamma", "0.05"), ("transducer_gain", "0.9")):
        transcript += f"Index frequency {name}\n----------------\n"
        transcript += "\n".join(f"{i} {3.3e9 + i * 5e7:.6e} {value}" for i in range(11)) + "\n\n"
    podman = tmp_path / "podman"
    podman.write_text(
        "#!/usr/bin/env python3\nimport os, pathlib, sys\n"
        "assert not os.environ.get('CHIPS_MODEL_KEY')\n"
        "args = sys.argv[1:]\n"
        "assert args[-1] == 'command -v ngspice && python3 /app/public/analyze_broadband.py'\n"
        "mount = next(a for a in args if a.endswith(':/app/public:ro')).split(':')[0]\n"
        "assert sorted(p.name for p in pathlib.Path(mount).iterdir()) == "
        "['analyze_broadband.py', 'tb_ac.spi']\n"
        f"print({transcript!r})\n"
    )
    podman.chmod(0o700)
    monkeypatch.setenv("CHIPS_MODEL_KEY", "PRIVATE_FIXTURE")
    output = tmp_path / "run"
    result = run_public_rlc(
        tmp_path / "source", candidate, output, task_id=task_id, podman=str(podman)
    )
    assert result["state"] == "simulated"
    assert result["task_id"] == task_id
    assert result["measurements"]["worst_gamma"] == 0.05
    assert result["measurements"]["min_transducer_gain"] == 0.9
    assert result["measurements"]["max_insertion_loss_db"] == pytest.approx(0.4575749)
    assert result["sweep"]["frequency_hz"] == [3.3e9 + i * 5e7 for i in range(11)]
    assert result["missing_measurements"] == []
    assert "score" not in result
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["task_id"] == task_id
    assert set(manifest["public_benches"]) == {"tb_ac.spi", "analyze_broadband.py"}
    assert (output / "public/analyze_broadband.py").read_text() == analyzer

    for index, broken in enumerate(
        (
            transcript.replace("10 3.800000e+09 0.9", ""),
            transcript.replace("5 3.550000e+09 0.9", "5 3.550000e+09 nan"),
            transcript.replace("5 3.550000e+09 0.9", "5 3.000000e+09 0.9"),
            transcript.replace("10 3.800000e+09 0.9", "9 3.800000e+09 0.9"),
        )
    ):
        podman.write_text(f"#!/usr/bin/env python3\nprint({broken!r})\n")
        failed = run_public_rlc(
            tmp_path / "source",
            candidate,
            tmp_path / f"invalid-{index}",
            task_id=task_id,
            podman=str(podman),
        )
        assert failed["state"] == "simulation_error"
        assert "transducer_gain" in failed["missing_measurements"]
        assert failed["measurements"] == {}
        assert failed["task_correctness"] == "not_evaluated"
