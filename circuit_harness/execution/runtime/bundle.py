"""Reproducible standard-library CLI bundle for offline simulator hosts."""

from importlib.resources import files
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from circuit_harness.execution.runtime.journal import file_digest


def build_cli(destination: Path) -> str:
    sources = {
        f"circuit_harness/execution/{name}": files("circuit_harness.execution")
        .joinpath(name)
        .read_bytes()
        for name in (
            "process.py",
            "journal.py",
            "jobs.py",
            "simulator.py",
            "native_sandbox.py",
            "bundle.py",
            "ngspice.py",
            "spectre.py",
            "emx.py",
            "spectre_testbench.py",
            "current_evas.py",
            "vabench.py",
            "vabench_worker.py",
            "vabench_public_worker.py",
            "analog_session.py",
            "authoring_session.py",
            "current_evas_session.py",
            "vabench_session.py",
            "analog_public.py",
            "current_evas_public.py",
            "public_observations.py",
            "session_budget.py",
            "task_authoring.py",
            "archive.py",
            "candidate_bundle.py",
            "analog_episode.py",
            "analog_design_bench.py",
            "benchmark_spectre.py",
            "benchmark_replay.py",
            "rc_validation.py",
            "session_transport.py",
            "ssh_worker.py",
            "benchmark_remote.py",
            "runtime/process.py",
            "runtime/journal.py",
            "runtime/jobs.py",
            "runtime/simulator.py",
            "runtime/native_sandbox.py",
            "runtime/bundle.py",
            "backends/ngspice.py",
            "backends/spectre.py",
            "backends/emx.py",
            "backends/spectre_testbench.py",
            "backends/current_evas.py",
            "backends/vabench.py",
            "backends/vabench_worker.py",
            "backends/vabench_public_worker.py",
            "sessions/action_store.py",
            "sessions/analog_session.py",
            "sessions/authoring_session.py",
            "sessions/current_evas_session.py",
            "sessions/vabench_session.py",
            "sessions/analog_public.py",
            "sessions/current_evas_public.py",
            "sessions/public_observations.py",
            "sessions/session_budget.py",
            "sessions/task_authoring.py",
            "evaluation/archive.py",
            "evaluation/candidate_bundle.py",
            "evaluation/analog_episode.py",
            "evaluation/analog_design_bench.py",
            "evaluation/benchmark_spectre.py",
            "evaluation/benchmark_replay.py",
            "evaluation/rc_validation.py",
            "transport/session_transport.py",
            "transport/ssh_worker.py",
            "transport/benchmark_remote.py",
            "runtime/__init__.py",
            "backends/__init__.py",
            "sessions/__init__.py",
            "evaluation/__init__.py",
            "transport/__init__.py",
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
