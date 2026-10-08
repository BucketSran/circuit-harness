"""Reproducible standard-library CLI bundle for offline simulator hosts."""

from importlib.resources import files
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .journal import file_digest


def build_cli(destination: Path) -> str:
    sources = {
        f"circuit_harness/execution/{name}.py": files(__package__)
        .joinpath(f"{name}.py")
        .read_bytes()
        for name in (
            "bundle",
            "candidate_bundle",
            "current_evas",
            "current_evas_public",
            "current_evas_session",
            "public_observations",
            "native_sandbox",
            "benchmark_spectre",
            "benchmark_remote",
            "benchmark_replay",
            "session_transport",
            "analog_design_bench",
            "analog_public",
            "analog_session",
            "session_budget",
            "analog_episode",
            "archive",
            "emx",
            "ssh_worker",
            "journal",
            "process",
            "simulator",
            "ngspice",
            "spectre",
            "spectre_testbench",
            "task_authoring",
            "authoring_session",
            "jobs",
            "vabench",
            "vabench_worker",
            "vabench_session",
            "vabench_public_worker",
            "rc_validation",
        )
    }
    sources["circuit_harness/cli.py"] = files("circuit_harness").joinpath("cli.py").read_bytes()
    for prefix in (
        "circuit_harness",
        "circuit_harness/execution",
    ):
        sources[f"{prefix}/__init__.py"] = b""
    sources["__main__.py"] = b"from circuit_harness.cli import main\nraise SystemExit(main())\n"
    with ZipFile(destination, "w") as archive:
        for name, content in sorted(sources.items()):
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, content)
    return file_digest(destination)
