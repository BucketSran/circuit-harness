#!/usr/bin/env python3
"""Bounded, explicit storage comparison; never run automatically against a lab host."""

import argparse
import hashlib
import io
import json
import os
import signal
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

MAX_BYTES = 32 * 1024 * 1024
MAX_MEMBERS = 2000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inspect_archive(payload):
    """Validate the complete bounded input before creating either destination."""
    expected = {}
    names = set()
    total = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or not (member.isfile() or member.isdir())
                or str(path) in names
            ):
                raise ValueError(f"unsafe or duplicate archive member: {member.name}")
            names.add(str(path))
            total += member.size
            if len(names) > MAX_MEMBERS or total > MAX_BYTES:
                raise ValueError("archive exceeds probe limits")
            if member.isfile():
                expected[str(path)] = digest(archive.extractfile(member).read())
    if not expected:
        raise ValueError("archive has no regular files")
    return expected, total


def verify_tree(root, expected):
    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in extracted tree: {path}")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = digest(path.read_bytes())
    if actual != expected:
        raise ValueError("extracted file set or contents differ from the input archive")


def filesystem(path):
    info = os.statvfs(path)
    try:
        mount = subprocess.run(
            ["findmnt", "-T", str(path), "-n", "-o", "TARGET,FSTYPE"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        mount_info = {"returncode": mount.returncode, "output": mount.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        mount_info = {"unavailable": type(exc).__name__}
    return {"free_bytes": info.f_bavail * info.f_frsize, "mount": mount_info}


def save_report(report, roots):
    encoded = json.dumps(report, indent=2) + "\n"
    for root in roots.values():
        temporary = root / "report.pending.json"
        with temporary.open("w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(root / "report.json")


def timed(sample, name, operation, *args):
    start = time.perf_counter()
    operation(*args)
    sample["seconds"][name] = time.perf_counter() - start


def extract(payload, target):
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(target, filter="data")


def deadline(_signum, _frame):
    raise TimeoutError("probe exceeded its 600-second wall-time budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--nfs-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    os.umask(0o077)
    report = {
        "status": "failed",
        "samples": [],
        "cache_policy": "system caches retained; fresh destinations",
    }
    roots = {}
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    try:
        if not 1 <= args.repeats <= 5:
            raise ValueError("repeats must be in [1, 5]")
        for path in (args.nfs_root, args.local_root):
            if path.exists() or path.is_symlink() or not path.parent.is_dir():
                raise ValueError("roots must be new directories with existing parents")
        destinations = {"nfs": args.nfs_root.resolve(), "local": args.local_root.resolve()}
        nfs, local = destinations.values()
        if nfs == local or nfs in local.parents or local in nfs.parents:
            raise ValueError("roots must be distinct and non-nested")
        if args.archive.stat().st_size > MAX_BYTES:
            raise ValueError("compressed archive exceeds probe limit")
        payload = args.archive.read_bytes()
        expected, total = inspect_archive(payload)
        report.update(
            archive_sha256=digest(payload),
            file_count=len(expected),
            expanded_bytes=total,
            repeats=args.repeats,
            uid=os.getuid(),
            started_at=time.time(),
            source=str(args.archive.resolve()),
        )
        for label, root in destinations.items():
            root.mkdir(mode=0o700)
            roots[label] = root
        report["filesystems"] = {label: filesystem(root) for label, root in roots.items()}
        if any(
            row["free_bytes"] < total * args.repeats * 2 for row in report["filesystems"].values()
        ):
            raise ValueError("insufficient reported free space for bounded probe")
        save_report(report, roots)
        for round_id in range(args.repeats):
            order = ("nfs", "local") if round_id % 2 == 0 else ("local", "nfs")
            for label in order:
                target = roots[label] / f"round-{round_id}"
                sample = {
                    "round": round_id,
                    "storage": label,
                    "directory": str(target),
                    "seconds": {},
                    "status": "running",
                    "started_at": time.time(),
                    "load_before": os.getloadavg(),
                }
                report["samples"].append(sample)
                print(
                    json.dumps({"event": "sample_start", "round": round_id, "storage": label}),
                    flush=True,
                )
                target.mkdir(mode=0o700)

                timed(sample, "extract", extract, payload, target)
                timed(sample, "verify_first", verify_tree, target, expected)
                timed(sample, "verify_repeated", verify_tree, target, expected)
                sample.update(status="passed", load_after=os.getloadavg())
                save_report(report, roots)
                print(json.dumps({"event": "sample_end", **sample}), flush=True)
        report["source_after_sha256"] = digest(args.archive.read_bytes())
        if report["source_after_sha256"] != report["archive_sha256"]:
            raise ValueError("input archive changed during probe")
        report["status"] = "passed"
    except (OSError, ValueError, tarfile.TarError, TimeoutError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if report["samples"] and report["samples"][-1]["status"] == "running":
            report["samples"][-1]["status"] = "failed"
    finally:
        signal.alarm(0)
    report["finished_at"] = time.time()
    try:
        save_report(report, roots)
    except OSError as exc:
        report.update(status="failed", report_write_error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
