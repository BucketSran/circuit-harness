"""Collect host observations without reading project contents or changing settings.

Standalone Python 3.10+ CLI, also runnable through SSH stdin. No network by default.
This is an audit probe, not the production Harness preflight or an acceptance grader.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit


def command(argv: list[str], timeout: float = 5) -> dict:
    if shutil.which(argv[0]) is None:
        return {"status": "unavailable", "reason": "command_not_found"}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": "timeout"}
    except OSError:
        return {"status": "error", "reason": "launch_failed"}
    return {
        "status": "observed" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "stdout": result.stdout[:16384],
        "truncated": len(result.stdout) > 16384,
    }


def path_metadata(path: Path) -> dict:
    info = path.stat()
    return {
        "path": str(path),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "owner_uid": info.st_uid,
        "owner_gid": info.st_gid,
        "is_symlink": path.is_symlink(),
        "sticky": bool(info.st_mode & stat.S_ISVTX),
        "acl": command(["getfacl", "-c", "-p", "-n", "--", str(path)]),
    }


def inspect_path(value: str) -> dict:
    path = Path(value).expanduser().absolute()
    try:
        disk = shutil.disk_usage(path)
        resolved = path.resolve()
        parents = dict.fromkeys([*reversed(path.parents), *reversed(resolved.parents)])
        return {
            **path_metadata(path),
            "status": "observed",
            "resolved_path": str(resolved),
            "disk_free_bytes": disk.free,
            "ancestors": [path_metadata(parent) for parent in parents],
        }
    except (OSError, RuntimeError) as exc:
        return {"path": str(path), "status": "error", "reason": type(exc).__name__}


def https_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not any(c.isspace() or ord(c) < 32 for c in value)
        )
        _ = parsed.port  # Access validates malformed or out-of-range ports.
    except ValueError:
        valid = False
    if not valid:
        raise argparse.ArgumentTypeError("use an HTTPS URL without credentials, query or fragment")
    return value


def timeout_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number") from exc
    if not math.isfinite(seconds) or not 0 < seconds <= 30:
        raise argparse.ArgumentTypeError("timeout must be > 0 and <= 30 seconds")
    return seconds


def inspect_https(url: str, timeout: float) -> dict:
    fields = {
        "http_code": "%{http_code}",
        "dns_s": "%{time_namelookup}",
        "tcp_s": "%{time_connect}",
        "tls_s": "%{time_appconnect}",
        "total_s": "%{time_total}",
    }
    result = command(
        [
            "curl",
            "-q",
            "--head",
            "--silent",
            "--output",
            os.devnull,
            "--connect-timeout",
            str(timeout),
            "--max-time",
            str(timeout),
            "--write-out",
            json.dumps(fields),
            "--url",
            url,
        ],
        timeout=timeout + 1,
    )
    row = {"url": url, "authentication": "not_tested", **result}
    payload = row.pop("stdout", "")
    if payload:
        try:
            measured = json.loads(payload)
            row["http_status"] = int(measured["http_code"])
            row["timings_s"] = {key: float(measured[key]) for key in fields if key != "http_code"}
        except (ValueError, KeyError, TypeError):
            row.update(status="error", reason="invalid_curl_measurements")
    elif result["status"] == "observed":
        row.update(status="error", reason="invalid_curl_measurements")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", default=[], help="Inspect metadata only")
    parser.add_argument(
        "--url",
        action="append",
        type=https_endpoint,
        default=[],
        help="Opt in to an unauthenticated HTTPS HEAD request",
    )
    parser.add_argument(
        "--timeout",
        type=timeout_seconds,
        default=5.0,
        help="Per-request HTTPS timeout, in seconds (maximum 30)",
    )
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "acceptance": "not_evaluated",
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
        "paths": [inspect_path(path) for path in args.path],
        "network": [inspect_https(url, args.timeout) for url in args.url],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(any(row["status"] != "observed" for row in report["paths"] + report["network"]))


if __name__ == "__main__":
    raise SystemExit(main())
