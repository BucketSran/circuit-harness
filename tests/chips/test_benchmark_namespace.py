"""Namespace public profiles must select the enforced child launcher."""

import json

import pytest

from circuit_harness.execution.evaluation.benchmark_spectre import _private_profile


def test_namespace_profile_accepts_valid_private_isolation_declaration(tmp_path):
    for name in ["jobs", "archives"]:
        (tmp_path / name).mkdir(mode=0o700)
    preflight = tmp_path / "preflight.sh"
    preflight.write_text("#!/bin/sh\nexit 0\n")
    config = tmp_path / "isolation.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bubblewrap": "/usr/bin/true",
                "spectre": "/usr/bin/true",
                "runtime_readonly_paths": ["/usr/bin"],
                "network": "disabled",
                "license_env": [],
            }
        )
    )
    config.chmod(0o600)
    profile = {
        "schema_version": 1,
        "backend": "spectre_namespace",
        "shell": "/bin/sh",
        "setup_scripts": [],
        "spectre": "/usr/bin/true",
        "preflight_script": str(preflight),
        "run_root": str(tmp_path / "jobs"),
        "archive_root": str(tmp_path / "archives"),
        "timeout_s": 30,
        "max_output_bytes": 1048576,
        "isolation_config": str(config),
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    path.chmod(0o600)
    assert _private_profile(path)["backend"] == "spectre_namespace"
    config.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        _private_profile(path)
