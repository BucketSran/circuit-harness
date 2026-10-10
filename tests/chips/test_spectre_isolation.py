"""Launcher protocol fixtures; actual Linux namespace checks are separate evidence."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("offline_bundle", [False, True])
def test_operator_launcher_scrubs_model_credentials_before_runtime(tmp_path, offline_bundle):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    simulator = runtime / "spectre"
    simulator.write_text("#!/bin/sh\nexit 0\n")
    simulator.chmod(0o700)
    bubblewrap = tmp_path / "bwrap-fixture"
    bubblewrap.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        'print(json.dumps({"argv":sys.argv[1:],"token_present":'
        'bool(os.environ.get("BENCHMARK_MODEL_KEY"))}))\n'
    )
    bubblewrap.chmod(0o700)
    config = tmp_path / "isolation.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bubblewrap": str(bubblewrap),
                "spectre": str(simulator),
                "runtime_readonly_paths": [str(runtime)],
                "network": "shared_license",
                "license_env": [],
            }
        )
    )
    config.chmod(0o600)
    if offline_bundle:
        from circuit_harness.execution.runtime.bundle import build_cli

        bundle = tmp_path / "harness.pyz"
        build_cli(bundle)
        entry = [str(bundle)]
    else:
        entry = ["-m", "circuit_harness.cli"]
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            *entry,
            "spectre-isolate",
            "--config",
            str(config),
            "--",
            "-W",
        ],
        env={**os.environ, "BENCHMARK_MODEL_KEY": "fixture-token"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["token_present"] is False
    assert report["argv"][-2:] == [str(simulator), "-W"]
    assert "--unshare-pid" in report["argv"]


def test_condition_inputs_are_readonly_and_preserve_relative_include_files(tmp_path):
    from circuit_harness.execution.backends.spectre_isolation import isolated_spectre_command

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    simulator = runtime / "spectre"
    simulator.write_text("#!/bin/sh\nexit 0\n")
    simulator.chmod(0o700)
    work = tmp_path / "condition"
    work.mkdir()
    (work / "dut.va").write_text('`include "helper.va"\nmodule dut; endmodule\n')
    (work / "helper.va").write_text("module helper; endmodule\n")
    config = tmp_path / "isolation.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bubblewrap": str(simulator),
                "spectre": str(simulator),
                "runtime_readonly_paths": [str(runtime)],
                "network": "disabled",
                "license_env": [],
            }
        )
    )
    config.chmod(0o600)
    argv, env = isolated_spectre_command(config, ["-64", "tb.scs"], cwd=work)
    for name in ["dut.va", "helper.va"]:
        source = str(work / name)
        assert ["--ro-bind", source, source] == argv[
            argv.index(source) - 1 : argv.index(source) + 2
        ]
    assert "--unshare-net" in argv
    assert env == {"PATH": "/usr/bin:/bin"}


def test_operator_cannot_declare_model_key_or_broad_filesystem_grant(tmp_path):
    import pytest

    from circuit_harness.execution.backends.spectre_isolation import isolated_spectre_command

    config = tmp_path / "isolation.json"
    settings = {
        "schema_version": 1,
        "bubblewrap": sys.executable,
        "spectre": sys.executable,
        "runtime_readonly_paths": ["/"],
        "network": "disabled",
        "license_env": ["BENCHMARK_MODEL_KEY"],
    }
    config.write_text(json.dumps(settings))
    config.chmod(0o600)
    with pytest.raises(ValueError, match="simulator/license"):
        isolated_spectre_command(config, ["-W"])
    settings["license_env"] = []
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="filesystem root"):
        isolated_spectre_command(config, ["-W"])


def test_condition_cannot_mount_private_configuration_or_symlink_inputs(tmp_path):
    from circuit_harness.execution.backends.spectre_isolation import isolated_spectre_command

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    simulator = runtime / "spectre"
    simulator.write_text("#!/bin/sh\nexit 0\n")
    simulator.chmod(0o700)
    work = tmp_path / "condition"
    work.mkdir()
    config = work / "isolation.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bubblewrap": str(simulator),
                "spectre": str(simulator),
                "runtime_readonly_paths": [str(runtime)],
                "network": "disabled",
                "license_env": [],
            }
        )
    )
    config.chmod(0o600)
    with pytest.raises(ValueError, match="configuration must remain outside"):
        isolated_spectre_command(config, ["tb.scs"], cwd=work)
    external = tmp_path / "isolation.json"
    config.rename(external)
    (work / "dut.va").symlink_to(simulator)
    with pytest.raises(ValueError, match="without symlinks"):
        isolated_spectre_command(external, ["tb.scs"], cwd=work)
