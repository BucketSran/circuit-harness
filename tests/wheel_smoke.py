"""Install a wheel outside the checkout and exercise its offline server bundle."""

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def check_wheel(wheel: Path) -> None:
    wheel = wheel.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="circuit-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        clean_env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

        def run(*args):
            result = subprocess.run(
                [str(python), *map(str, args)],
                cwd=root,
                env=clean_env,
                timeout=120,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            return result.stdout

        run("-m", "pip", "install", "--no-deps", "--no-index", wheel)
        output = run("-I", "-m", "circuit_harness", "--help")
        assert "vabench" in output and "analog" in output
        run(
            "-I",
            "-c",
            """
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path
from circuit_harness.execution.bundle import build_cli
from circuit_harness.execution.current_evas import HARNESS_SOURCE, _LOADED_HARNESS
from circuit_harness.execution.journal import file_digest
import circuit_harness

installed = distribution('circuit-harness')
assert not any(str(p).startswith(('alphaapollo/', 'verl/')) for p in installed.files)
assert files('circuit_harness.harbor').joinpath('profiles.schema.json').is_file()
assert files('circuit_harness.data').joinpath('json/atif_dataset_manifest.json').is_file()
root = Path(circuit_harness.__file__).parent.parent
assert _LOADED_HARNESS == {name: file_digest(root / name) for name in HARNESS_SOURCE}
build_cli(Path('circuit.pyz'))
""",
        )
        output = run("-I", "-S", root / "circuit.pyz", "--help")
        assert "vabench" in output and "analog" in output
        # The VA07 recipe remains usable without the optional Harbor dependency.
        for entrypoint in (
            "circuit_harness.benchmarks.evas_va07",
            "circuit_harness.harbor.evas_example",
        ):
            output = run("-I", "-m", entrypoint, "--help")
            assert "prepare" in output and "result" in output
            existing = root / entrypoint.rsplit(".", 1)[1]
            existing.mkdir()
            (existing / "keep.txt").write_text("previous evidence")
            rejected = subprocess.run(
                [
                    str(python),
                    "-I",
                    "-m",
                    entrypoint,
                    "run",
                    "--workspace",
                    str(root / "absent"),
                    "--image",
                    "absent",
                    "--output",
                    str(existing),
                ],
                cwd=root,
                env=clean_env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert rejected.returncode == 2, rejected.stdout + rejected.stderr
            assert "output already exists" in rejected.stderr
            assert (existing / "keep.txt").read_text() == "previous evidence"
        # Exercise a recorded failure, not only argparse/imports.
        (root / "input.json").write_text('{"resistance_ohm":1000,"capacitance_f":1e-9}')
        failed = subprocess.run(
            [
                str(python),
                "-I",
                "-S",
                str(root / "circuit.pyz"),
                "ngspice-rc",
                "--input",
                "input.json",
                "--output",
                "run",
                "--ngspice",
                str(root / "absent"),
            ],
            cwd=root,
            env=clean_env,
            capture_output=True,
            text=True,
        )
        assert failed.returncode == 1, failed.stderr
        import json

        result = json.loads((root / "run/result.json").read_text())
        assert result["execution"] == "infrastructure_error"
        assert result["reason"] == "dependency_missing"
        assert result["verdict"] == "not_evaluated"
    print("Installed wheel, schemas, source identity and stdlib zipapp verified")


if __name__ == "__main__":
    check_wheel(Path(sys.argv[1]))
