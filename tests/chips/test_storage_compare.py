"""Real local I/O checks for the opt-in probe; neither test root is a lab NFS mount."""

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

PROBE = Path(__file__).parent / "probes/storage_compare.py"


def make_archive(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))


def run_probe(archive, nfs, local, repeats=2):
    return subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--archive",
            str(archive),
            "--nfs-root",
            str(nfs),
            "--local-root",
            str(local),
            "--repeats",
            str(repeats),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_identical_payloads_alternating_order_and_separate_timings(tmp_path):
    archive = tmp_path / "input.tar.gz"
    make_archive(archive, {"source/a.py": b"print('a')\n", "data/case.json": b'{"v": 1}'})
    nfs, local = tmp_path / "nfs-labelled", tmp_path / "local-labelled"
    result = run_probe(archive, nfs, local)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["status"] == "passed"
    assert report["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert report["file_count"] == 2
    assert [(s["round"], s["storage"]) for s in report["samples"]] == [
        (0, "nfs"),
        (0, "local"),
        (1, "local"),
        (1, "nfs"),
    ]
    for sample in report["samples"]:
        assert sample["status"] == "passed"
        assert set(sample["seconds"]) == {"extract", "verify_first", "verify_repeated"}
        assert all(value >= 0 for value in sample["seconds"].values())
        assert (Path(sample["directory"]) / "data/case.json").read_bytes() == b'{"v": 1}'
    assert report["cache_policy"] == "system caches retained; fresh destinations"
    assert report["source_after_sha256"] == report["archive_sha256"]


@pytest.mark.parametrize("kind", ["traversal", "symlink", "duplicate", "too_large", "empty"])
def test_unsafe_or_unbounded_archive_creates_no_roots(tmp_path, kind):
    path = tmp_path / "input.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        if kind != "empty":
            entry = tarfile.TarInfo("../escape" if kind == "traversal" else "a")
            if kind == "symlink":
                entry.type = tarfile.SYMTYPE
                entry.linkname = "/tmp"
            if kind == "too_large":
                entry.size = 33 * 1024 * 1024
                archive.addfile(entry, io.BytesIO(b"x" * entry.size))
            else:
                archive.addfile(entry)
            if kind == "duplicate":
                archive.addfile(entry)
    nfs, local = tmp_path / "nfs", tmp_path / "local"
    result = run_probe(path, nfs, local)
    assert result.returncode == 1
    assert json.loads(result.stdout.splitlines()[-1])["status"] == "failed"
    assert not nfs.exists() and not local.exists() and not (tmp_path / "escape").exists()


def test_existing_directory_is_not_modified(tmp_path):
    archive = tmp_path / "input.tar.gz"
    make_archive(archive, {"a": b"hello"})
    nfs = tmp_path / "nfs"
    nfs.mkdir()
    marker = nfs / "keep"
    marker.write_bytes(b"original")
    result = run_probe(archive, nfs, tmp_path / "local")
    assert result.returncode == 1
    assert sorted(p.name for p in nfs.iterdir()) == ["keep"]
    assert marker.read_bytes() == b"original"
    assert not (tmp_path / "local").exists()


def test_hash_verification_rejects_corruption_and_extra_files(tmp_path):
    spec = importlib.util.spec_from_file_location("storage_compare", PROBE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    file = tmp_path / "source"
    file.write_bytes(b"original")
    expected = {"source": hashlib.sha256(b"original").hexdigest()}
    probe.verify_tree(tmp_path, expected)
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differ"):
        probe.verify_tree(tmp_path, expected)
    file.write_bytes(b"original")
    (tmp_path / "unexpected").write_bytes(b"extra")
    with pytest.raises(ValueError, match="differ"):
        probe.verify_tree(tmp_path, expected)
