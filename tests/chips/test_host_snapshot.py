"""Public CLI checks; real local files, no lab host or model requests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).parent / "probes" / "host_snapshot.py"


def run_probe(*args, env=None):
    return subprocess.run(
        [sys.executable, str(PROBE), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )


def test_private_directory_snapshot_does_not_read_contents_or_change_permissions(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    secret = private / "operator-secret"
    secret.write_text("CANARY_MUST_NOT_APPEAR")
    before = private.stat().st_mode

    result = run_probe("--path", private)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["acceptance"] == "not_evaluated"
    assert report["paths"][0]["mode"] == "0700"
    assert report["network"] == []
    assert "CANARY_MUST_NOT_APPEAR" not in result.stdout + result.stderr
    assert private.stat().st_mode == before
    assert list(private.iterdir()) == [secret]


def test_private_child_does_not_hide_a_writable_parent(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    private = shared / "private"
    private.mkdir(mode=0o700)

    result = run_probe("--path", private)

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)["paths"][0]
    ancestor = next(row for row in observed["ancestors"] if row["path"] == str(shared))
    assert ancestor["mode"] == "0777"
    assert not ancestor["sticky"]
    assert "acl" in ancestor
    assert observed["mode"] == "0700"


def test_missing_path_keeps_other_observations_and_returns_failure(tmp_path):
    missing = tmp_path / "not-created"

    result = run_probe("--path", tmp_path, "--path", missing)

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["paths"][0]["status"] == "observed"
    assert report["paths"][1]["status"] == "error"
    assert report["paths"][1]["reason"] == "FileNotFoundError"
    assert report["acceptance"] == "not_evaluated"
    assert not missing.exists()


def curl_fixture(tmp_path, *, http_code="403", returncode=0):
    """Constructed transport boundary; does not contact an HTTPS service."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    marker = tmp_path / "curl-called"
    curl = binaries / "curl"
    payload = json.dumps(
        {
            "http_code": http_code,
            "dns_s": "0.01",
            "tcp_s": "0.02",
            "tls_s": "0.03",
            "total_s": "0.04",
        }
    )
    curl.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\nprint({payload!r})\nraise SystemExit({returncode})\n"
    )
    curl.chmod(0o700)
    return {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"]}, marker


def test_https_denial_is_observed_without_claiming_model_authentication(tmp_path):
    env, marker = curl_fixture(tmp_path)

    offline = run_probe("--path", tmp_path, env=env)
    assert offline.returncode == 0
    assert not marker.exists()

    result = run_probe("--url", "https://model.example.invalid/", env=env)

    assert result.returncode == 0, result.stderr
    assert marker.exists()
    report = json.loads(result.stdout)
    assert report["network"][0]["http_status"] == 403
    assert report["network"][0]["status"] == "observed"
    assert report["network"][0]["authentication"] == "not_tested"
    assert report["acceptance"] == "not_evaluated"


def test_empty_network_measurement_is_not_a_successful_observation(tmp_path):
    env, _ = curl_fixture(tmp_path)
    (tmp_path / "bin" / "curl").write_text(f"#!{sys.executable}\n")

    result = run_probe("--url", "https://model.example.invalid/", env=env)

    assert result.returncode == 1
    assert json.loads(result.stdout)["network"][0]["reason"] == "invalid_curl_measurements"


@pytest.mark.parametrize(
    "url",
    [
        "https://user:SECRET@example.invalid/",
        "https://example.invalid/?key=SECRET",
        "https://example.invalid/#SECRET",
        "http://example.invalid/SECRET",
    ],
)
def test_credential_bearing_or_plain_http_urls_are_rejected_before_network(tmp_path, url):
    env, marker = curl_fixture(tmp_path)

    result = run_probe("--url", url, env=env)

    assert result.returncode == 2
    assert not marker.exists()
    assert "SECRET" not in result.stdout + result.stderr


def test_transport_failure_keeps_measurements_and_returns_nonzero(tmp_path):
    env, _ = curl_fixture(tmp_path, http_code="000", returncode=28)

    result = run_probe("--url", "https://model.example.invalid/", env=env)

    assert result.returncode == 1
    row = json.loads(result.stdout)["network"][0]
    assert row["status"] == "error"
    assert row["returncode"] == 28
    assert row["http_status"] == 0


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf", "31"])
def test_timeout_must_be_positive_finite_and_bounded(tmp_path, seconds):
    env, marker = curl_fixture(tmp_path)

    result = run_probe("--timeout", seconds, "--url", "https://example.invalid/", env=env)

    assert result.returncode == 2
    assert not marker.exists()
