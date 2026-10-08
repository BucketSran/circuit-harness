"""Chips harness CLI: run, resume, inspect live events and export a report."""

from __future__ import annotations

import argparse
import html
import json
import signal
import subprocess
import sys
import threading
from pathlib import Path

from alphaapollo.common.execution.chips.analog_design_bench import TASKS as ANALOG_BENCH_TASKS
from alphaapollo.common.execution.chips.analog_design_bench import run_case as run_analog_bench
from alphaapollo.common.execution.chips.analog_episode import (
    archive_episode as archive_analog_episode,
)
from alphaapollo.common.execution.chips.analog_episode import (
    verify_episode as verify_analog_episode,
)
from alphaapollo.common.execution.chips.analog_public import TASK_ID, run_public_rlc
from alphaapollo.common.execution.chips.analog_session import (
    action_response as analog_action_response,
)
from alphaapollo.common.execution.chips.analog_session import (
    close_session as close_analog_session,
)
from alphaapollo.common.execution.chips.analog_session import (
    create_session as create_analog_session,
)
from alphaapollo.common.execution.chips.analog_session import (
    enqueue_action as enqueue_analog_action,
)
from alphaapollo.common.execution.chips.analog_session import (
    finalize_session as finalize_analog_session,
)
from alphaapollo.common.execution.chips.analog_session import (
    session_action as analog_session_action,
)
from alphaapollo.common.execution.chips.analog_session import (
    session_info as analog_session_info,
)
from alphaapollo.common.execution.chips.archive import archived_job_status, verify_archive
from alphaapollo.common.execution.chips.authoring_session import (
    action_response as gain_response,
)
from alphaapollo.common.execution.chips.authoring_session import (
    create_session as create_gain_session,
)
from alphaapollo.common.execution.chips.authoring_session import (
    request_action as gain_request,
)
from alphaapollo.common.execution.chips.benchmark_remote import stage_transfer
from alphaapollo.common.execution.chips.benchmark_replay import (
    replay_candidate,
    summarize_replays,
    verify_replay,
)
from alphaapollo.common.execution.chips.bundle import build_cli
from alphaapollo.common.execution.chips.emx import EmxConfig, run_emx
from alphaapollo.common.execution.chips.jobs import (
    cancel_job,
    cleanup_job,
    execute_job,
    inspect_job,
    job_timings,
    retry_archive,
    submit_benchmark_spectre,
    submit_rc,
    submit_spectre_gain,
    submit_spectre_rc,
    submit_vabench,
    verify_job,
)
from alphaapollo.common.execution.chips.journal import atomic_json, read_events, status
from alphaapollo.common.execution.chips.ngspice import run_rc, verify_rc
from alphaapollo.common.execution.chips.simulator import SimulationRequest, simulate
from alphaapollo.common.execution.chips.vabench import export_vabench, pin_vabench


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    session = commands.add_parser("vabench-session")
    session.add_argument("--pin", type=Path, required=True)
    session.add_argument("--output", type=Path, required=True)
    session.add_argument("--max-actions", type=int, default=24)
    session.add_argument("--max-simulations", type=int, default=4)
    session.add_argument("--timeout-s", type=int, default=120)
    for name in ("vabench-action", "vabench-request"):
        action = commands.add_parser(name)
        action.add_argument("--session", type=Path, required=True)
        action.add_argument("--request", required=True)
    preflight = commands.add_parser("vabench-preflight")
    preflight.add_argument("--session", type=Path, required=True)
    response = commands.add_parser("vabench-response")
    response.add_argument("--session", type=Path, required=True)
    response.add_argument("--id", required=True)
    final = commands.add_parser("vabench-finalize")
    final.add_argument("--session", type=Path, required=True)
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--job-id", required=True)
    final.add_argument("--archive-root", type=Path)
    bundle = commands.add_parser("bundle", help="build an offline standard-library CLI zipapp")
    bundle.add_argument("--output", type=Path, required=True)
    analog = commands.add_parser("analog-bench", help="run a pinned Analog Design Bench verifier")
    analog.add_argument("--task-id", choices=ANALOG_BENCH_TASKS, required=True)
    analog.add_argument("--source-root", type=Path, required=True)
    analog.add_argument("--candidate", type=Path, required=True)
    analog.add_argument("--output", type=Path, required=True)
    analog.add_argument("--backend", choices=("podman", "bubblewrap"), default="podman")
    analog.add_argument("--podman-root", type=Path)
    analog.add_argument("--podman-runroot", type=Path)
    analog.add_argument("--runtime-image")
    analog.add_argument("--offline-image-archive", type=Path)
    analog.add_argument("--podman-single-id", action="store_true")
    analog.add_argument("--podman-no-cpu-limit", action="store_true")
    analog.add_argument("--ngspice", type=Path)
    analog.add_argument("--timeout-s", type=int, default=900)
    analog.add_argument("--archive-root", type=Path)
    public = commands.add_parser("analog-public", help="run pinned public RLC diagnostics")
    public.add_argument(
        "--task-id",
        default=TASK_ID,
        choices=[key for key, task in ANALOG_BENCH_TASKS.items() if task.public_rlc is not None],
    )
    public.add_argument("--source-root", type=Path, required=True)
    public.add_argument("--candidate", type=Path, required=True)
    public.add_argument("--output", type=Path, required=True)
    public.add_argument("--podman", default="podman")
    public.add_argument("--podman-root", type=Path)
    public.add_argument("--podman-runroot", type=Path)
    public.add_argument("--runtime-image")
    public.add_argument("--offline-image-archive", type=Path)
    public.add_argument("--podman-single-id", action="store_true")
    public.add_argument("--podman-no-cpu-limit", action="store_true")
    public.add_argument("--timeout-s", type=int, default=120)
    analog_session = commands.add_parser(
        "analog-session", help="create a bounded public RLC session"
    )
    analog_session.add_argument(
        "--task-id",
        default=TASK_ID,
        choices=[key for key, task in ANALOG_BENCH_TASKS.items() if task.public_rlc is not None],
    )
    analog_session.add_argument("--source-root", type=Path, required=True)
    analog_session.add_argument("--output", type=Path, required=True)
    analog_session.add_argument("--podman", default="podman")
    analog_session.add_argument("--podman-root", type=Path)
    analog_session.add_argument("--podman-runroot", type=Path)
    analog_session.add_argument("--runtime-image")
    analog_session.add_argument("--offline-image-archive", type=Path)
    analog_session.add_argument("--podman-single-id", action="store_true")
    analog_session.add_argument("--podman-no-cpu-limit", action="store_true")
    analog_session.add_argument("--timeout-s", type=int, default=120)
    analog_session.add_argument("--max-actions", type=int, default=24)
    analog_session.add_argument("--max-simulations", type=int, default=4)
    info = commands.add_parser("analog-info", help="read the public session contract")
    info.add_argument("--session", type=Path, required=True)
    close = commands.add_parser("analog-close", help="operator-only episode end collection")
    close.add_argument("--session", type=Path, required=True)
    close.add_argument("--reason", required=True)
    for name in ("analog-action", "analog-request"):
        action = commands.add_parser(name, help="run or enqueue one bounded session action")
        action.add_argument("--session", type=Path, required=True)
        action.add_argument("--request", required=True)
    analog_response = commands.add_parser("analog-response", help="read one action response")
    analog_response.add_argument("--session", type=Path, required=True)
    analog_response.add_argument("--id", required=True)
    analog_final = commands.add_parser("analog-finalize", help="operator-only frozen RLC scoring")
    analog_final.add_argument("--session", type=Path, required=True)
    analog_final.add_argument("--output", type=Path, required=True)
    analog_final.add_argument("--archive-root", type=Path)
    analog_final.add_argument("--timeout-s", type=int, default=900)
    analog_archive = commands.add_parser("analog-archive", help="seal private Analog evidence")
    analog_archive.add_argument("--session", type=Path, required=True)
    analog_archive.add_argument("--archive-root", type=Path, required=True)
    analog_archive.add_argument("--episode-id", required=True)
    analog_archive.add_argument("--agent-evidence", type=Path)
    analog_archive.add_argument("--final-output", type=Path)
    analog_verify = commands.add_parser("verify-analog-episode")
    analog_verify.add_argument("directory", type=Path)
    emx = commands.add_parser("emx")
    emx.add_argument("--config", type=Path, required=True)
    emx.add_argument("--input", type=Path, required=True)
    emx.add_argument("--output", type=Path, required=True)
    emx.add_argument("--resume", action="store_true")
    rc = commands.add_parser("ngspice-rc")
    rc.add_argument("--input", type=Path, required=True)
    rc.add_argument("--output", type=Path, required=True)
    rc.add_argument("--ngspice", default="ngspice")
    rc.add_argument("--timeout", type=float, default=60)
    rc.add_argument("--resume", action="store_true")
    submit = commands.add_parser("submit-rc", help="submit an independent server-side RC job")
    submit.add_argument("--input", type=Path, required=True)
    submit.add_argument("--root", type=Path, required=True)
    submit.add_argument("--archive-root", type=Path)
    submit.add_argument("--job-id", required=True)
    submit.add_argument("--ngspice", default="ngspice")
    submit.add_argument("--timeout", type=float, default=60)
    spectre = commands.add_parser(
        "submit-spectre-rc", help="submit the bounded RC-001 task using an operator Spectre profile"
    )
    spectre.add_argument("--input", type=Path, required=True)
    spectre.add_argument("--profile", type=Path, required=True)
    spectre.add_argument("--job-id", required=True)
    gain = commands.add_parser(
        "submit-spectre-gain", help="submit a confirmed bounded differential-gain testbench"
    )
    gain.add_argument("--input", type=Path, required=True)
    gain.add_argument("--profile", type=Path, required=True)
    gain.add_argument("--job-id", required=True)
    replay = commands.add_parser(
        "replay-benchmark", help="saved candidate open-source replay, no commercial execution"
    )
    replay.add_argument("--candidate", type=Path, required=True)
    replay.add_argument("--task-package", type=Path, required=True)
    replay.add_argument("--config", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--spectre-record", type=Path)
    replay_verify = commands.add_parser("verify-benchmark-replay")
    replay_verify.add_argument("directory", type=Path)
    replay_summary = commands.add_parser("summarize-benchmark-replays")
    replay_summary.add_argument("directories", type=Path, nargs="+")
    stage_benchmark = commands.add_parser(
        "stage-benchmark", help="operator-only frozen input transfer"
    )
    stage_benchmark.add_argument("--root", type=Path, required=True)
    stage_benchmark.add_argument("--request", required=True)
    stage_benchmark.add_argument("--purpose", choices=("final", "public"), default="final")
    benchmark = commands.add_parser(
        "submit-benchmark-spectre", help="operator-only frozen benchmark job"
    )
    benchmark.add_argument("--candidate", type=Path, required=True)
    benchmark.add_argument("--task-package", type=Path, required=True)
    benchmark.add_argument("--profile", type=Path, required=True)
    benchmark.add_argument("--job-id", required=True)
    benchmark.add_argument("--isolated-public", action="store_true")
    benchmark.add_argument("--purpose", choices=("final", "public"), default="final")
    gain_session = commands.add_parser("gain-session", help="operator-only confirmed gain session")
    gain_session.add_argument("--input", type=Path, required=True)
    gain_session.add_argument("--profile", type=Path, required=True)
    gain_session.add_argument("--session", type=Path, required=True)
    gain_session.add_argument("--max-simulations", type=int, default=3)
    gain_action = commands.add_parser("gain-request")
    gain_action.add_argument("--session", type=Path, required=True)
    gain_action.add_argument("--request", required=True)
    gain_result = commands.add_parser("gain-response")
    gain_result.add_argument("--session", type=Path, required=True)
    gain_result.add_argument("--id", required=True)
    pin = commands.add_parser(
        "pin-vabench", help="pin one r53 task, original scorer and EVAS environment"
    )
    pin.add_argument("--source", type=Path, required=True)
    pin.add_argument("--python", type=Path, required=True)
    pin.add_argument("--task-id", required=True)
    pin.add_argument("--output", type=Path, required=True)
    export = commands.add_parser(
        "export-vabench", help="export original public-only task materials"
    )
    export.add_argument("--pin", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    va = commands.add_parser(
        "submit-vabench", help="operator-only detached frozen submission replay"
    )
    va.add_argument("--pin", type=Path, required=True)
    va.add_argument("--submission", type=Path, required=True)
    va.add_argument("--root", type=Path, required=True)
    va.add_argument("--archive-root", type=Path)
    va.add_argument("--job-id", required=True)
    va.add_argument("--timeout", type=float, default=300)
    for name in (
        "job-status",
        "job-cancel",
        "verify-job",
        "job-execute",
        "verify-archive",
        "archive-status",
        "job-archive",
        "job-cleanup",
        "job-timings",
    ):
        commands.add_parser(name).add_argument("directory", type=Path)
    rtl = commands.add_parser("rtl-smoke")
    rtl.add_argument("--design", type=Path, required=True)
    rtl.add_argument("--testbench", type=Path, required=True)
    rtl.add_argument("--output", type=Path, required=True)
    rtl.add_argument("--timeout", type=float, default=30)
    rtl.add_argument("--resume", action="store_true")
    for name in ("status", "watch", "report", "verify-rc"):
        sub = commands.add_parser(name)
        sub.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command in (
        "vabench-session",
        "vabench-action",
        "vabench-request",
        "vabench-response",
        "vabench-finalize",
        "vabench-preflight",
    ):
        from alphaapollo.common.execution.chips.vabench_session import (
            action_response,
            create_session,
            enqueue_action,
            finalize_session,
            preflight_session,
            session_action,
        )

        if args.command == "vabench-session":
            reply = create_session(
                json.loads(args.pin.read_text()),
                args.output,
                max_actions=args.max_actions,
                max_simulations=args.max_simulations,
                timeout_s=args.timeout_s,
            )
        elif args.command == "vabench-preflight":
            reply = preflight_session(args.session)
        elif args.command == "vabench-response":
            reply = action_response(args.session, args.id)
        elif args.command == "vabench-finalize":
            reply = finalize_session(args.session, args.root, args.job_id, args.archive_root)
        else:
            request = json.loads(
                sys.stdin.read(150_001) if args.request == "-" else Path(args.request).read_text()
            )
            handler = session_action if args.command == "vabench-action" else enqueue_action
            reply = handler(args.session, request)
            if args.command == "vabench-action":
                spool = args.session / "requests" / request["id"]
                if spool.is_dir():
                    atomic_json(spool / "response.json", reply)
        print(json.dumps(reply))
        return 0 if reply.get("ok", True) else 1
    if args.command == "bundle":
        print(json.dumps({"path": str(args.output), "sha256": build_cli(args.output)}))
        return 0
    if args.command == "analog-session":
        reply = create_analog_session(
            args.source_root,
            args.output,
            task_id=args.task_id,
            podman=args.podman,
            podman_root=args.podman_root,
            podman_runroot=args.podman_runroot,
            runtime_image=args.runtime_image,
            offline_image_archive=args.offline_image_archive,
            podman_single_id=args.podman_single_id,
            podman_no_cpu_limit=args.podman_no_cpu_limit,
            timeout_s=args.timeout_s,
            max_actions=args.max_actions,
            max_simulations=args.max_simulations,
        )
        print(json.dumps(reply, ensure_ascii=False))
        return 0
    if args.command == "analog-response":
        print(json.dumps(analog_action_response(args.session, args.id), ensure_ascii=False))
        return 0
    if args.command == "analog-info":
        print(json.dumps(analog_session_info(args.session)))
        return 0
    if args.command == "analog-close":
        print(json.dumps(close_analog_session(args.session, args.reason)))
        return 0
    if args.command == "analog-finalize":
        reply = finalize_analog_session(
            args.session,
            args.output,
            archive_root=args.archive_root,
            timeout_s=args.timeout_s,
        )
        print(json.dumps(reply, ensure_ascii=False))
        return 0 if reply["state"] == "graded" else 1
    if args.command == "analog-archive":
        receipt = archive_analog_episode(
            args.session,
            args.archive_root,
            args.episode_id,
            agent=args.agent_evidence,
            final=args.final_output,
        )
        print(
            json.dumps(
                {
                    "path": str(args.archive_root / "episodes" / args.episode_id),
                    "sha256": receipt["sha256"],
                    "members": len(receipt["members"]),
                    "candidate_sha256": receipt["candidate_sha256"],
                }
            )
        )
        return 0
    if args.command == "verify-analog-episode":
        receipt = verify_analog_episode(args.directory)
        print(json.dumps({"state": "verified", "sha256": receipt["sha256"]}))
        return 0
    if args.command in {"analog-action", "analog-request"}:
        if args.request == "-":
            raw = sys.stdin.read(150_001)
        else:
            path = Path(args.request)
            if path.stat().st_size > 150_000:
                raise ValueError("action exceeds request limit")
            raw = path.read_text(encoding="utf-8")
        request = json.loads(raw)
        reply = (
            analog_session_action(args.session, request)
            if args.command == "analog-action"
            else enqueue_analog_action(args.session, request)
        )
        if args.command == "analog-action":
            spool = args.session / "requests" / request["id"]
            if spool.is_dir():
                atomic_json(spool / "response.json", reply)
        print(json.dumps(reply, ensure_ascii=False))
        return 0 if reply.get("ok") or reply.get("state") == "accepted" else 1
    if args.command == "analog-public":
        reply = run_public_rlc(
            args.source_root,
            args.candidate,
            args.output,
            task_id=args.task_id,
            podman=args.podman,
            podman_root=args.podman_root,
            podman_runroot=args.podman_runroot,
            runtime_image=args.runtime_image,
            offline_image_archive=args.offline_image_archive,
            podman_single_id=args.podman_single_id,
            podman_no_cpu_limit=args.podman_no_cpu_limit,
            timeout_s=args.timeout_s,
        )
        print(json.dumps(reply, ensure_ascii=False))
        return 0 if reply["state"] == "simulated" else 1
    if args.command == "analog-bench":
        reply = run_analog_bench(
            args.task_id,
            args.source_root,
            args.candidate,
            args.output,
            backend=args.backend,
            podman_root=args.podman_root,
            podman_runroot=args.podman_runroot,
            runtime_image=args.runtime_image,
            offline_image_archive=args.offline_image_archive,
            podman_single_id=args.podman_single_id,
            podman_no_cpu_limit=args.podman_no_cpu_limit,
            ngspice=args.ngspice,
            timeout_s=args.timeout_s,
            archive_root=args.archive_root,
        )
        print(json.dumps(reply, ensure_ascii=False))
        return 0 if reply["state"] == "graded" else 1
    if args.command == "job-execute":
        execute_job(args.directory)
        return 0
    cancel = threading.Event()
    previous = {
        sig: signal.signal(sig, lambda *_: cancel.set()) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        if args.command == "pin-vabench":
            if args.output.exists():
                raise ValueError(
                    "pin output already exists; use a new path for an explicit revision"
                )
            pin = pin_vabench(args.source, args.python, args.task_id)
            atomic_json(args.output, pin)
            print(json.dumps({"pin": str(args.output), "task_id": args.task_id}))
            return 0
        elif args.command == "export-vabench":
            export_vabench(json.loads(args.pin.read_text()), args.output)
            print(json.dumps({"public_directory": str(args.output), "evaluator_exported": False}))
            return 0
        elif args.command == "submit-vabench":
            reply = submit_vabench(
                json.loads(args.pin.read_text()),
                args.submission,
                args.root,
                args.job_id,
                timeout_s=args.timeout,
                archive_root=args.archive_root,
            )
            print(json.dumps(reply, indent=2, ensure_ascii=False))
            return 0 if reply["state"] in {"running", "finished"} else 1
        elif args.command == "submit-rc":
            reply = submit_rc(
                json.loads(args.input.read_text()),
                args.root,
                args.job_id,
                ngspice=args.ngspice,
                timeout_s=args.timeout,
                archive_root=args.archive_root,
            )
            print(json.dumps(reply, indent=2, ensure_ascii=False))
            return 0 if reply["state"] in {"running", "finished"} else 1
        elif args.command == "gain-session":
            reply = create_gain_session(
                args.session, json.loads(args.input.read_text()), args.profile, args.max_simulations
            )
            print(json.dumps(reply))
            return 0
        elif args.command == "gain-request":
            data = (
                sys.stdin.read(1024 * 1024 + 1)
                if args.request == "-"
                else Path(args.request).read_text()
            )
            if len(data) > 1024 * 1024:
                raise ValueError("gain request exceeds 1 MiB")
            print(json.dumps(gain_request(args.session, json.loads(data))))
            return 0
        elif args.command == "gain-response":
            print(json.dumps(gain_response(args.session, args.id)))
            return 0
        elif args.command == "replay-benchmark":
            receipt = replay_candidate(
                args.candidate,
                args.task_package,
                json.loads(args.config.read_text()),
                args.output,
                spectre_record=args.spectre_record,
            )
            print(json.dumps(receipt, indent=2))
            return 0 if receipt["result"]["execution"] == "ok" else 1
        elif args.command == "verify-benchmark-replay":
            print(json.dumps(verify_replay(args.directory), indent=2))
            return 0
        elif args.command == "summarize-benchmark-replays":
            print(json.dumps(summarize_replays(args.directories), indent=2))
            return 0
        elif args.command == "stage-benchmark":
            data = (
                sys.stdin.read(48 * 1024 * 1024 + 1)
                if args.request == "-"
                else Path(args.request).read_text()
            )
            if len(data) > 48 * 1024 * 1024:
                raise ValueError("benchmark transfer exceeds limit")
            print(json.dumps(stage_transfer(args.root, json.loads(data), purpose=args.purpose)))
            return 0
        elif args.command == "submit-benchmark-spectre":
            reply = submit_benchmark_spectre(
                args.candidate,
                args.task_package,
                args.profile,
                args.job_id,
                purpose=args.purpose,
                isolated_public=args.isolated_public,
            )
            print(json.dumps(reply, indent=2, ensure_ascii=False))
            return 0 if reply["state"] in {"running", "finished", "archived"} else 1
        elif args.command == "submit-spectre-gain":
            reply = submit_spectre_gain(
                json.loads(args.input.read_text()), args.profile, args.job_id
            )
            print(json.dumps(reply, indent=2, ensure_ascii=False))
            return 0 if reply["state"] in {"running", "finished", "archived"} else 1
        elif args.command == "submit-spectre-rc":
            reply = submit_spectre_rc(json.loads(args.input.read_text()), args.profile, args.job_id)
            print(json.dumps(reply, indent=2, ensure_ascii=False))
            return 0 if reply["state"] in {"running", "finished", "archived"} else 1
        elif args.command in {"job-status", "job-cancel"}:
            action = inspect_job if args.command == "job-status" else cancel_job
            reply = action(args.directory)
            print(json.dumps(reply, indent=2, ensure_ascii=False))
            return 0 if reply["state"] in {"running", "finished"} else 1
        elif args.command == "job-timings":
            print(json.dumps(job_timings(args.directory), indent=2))
            return 0
        elif args.command in {"archive-status", "job-archive", "job-cleanup"}:
            action = {
                "archive-status": archived_job_status,
                "job-archive": retry_archive,
                "job-cleanup": cleanup_job,
            }[args.command]
            print(json.dumps(action(args.directory), indent=2))
            return 0
        elif args.command == "verify-archive":
            receipt = verify_archive(args.directory)
            print(
                json.dumps(
                    {
                        "state": "verified",
                        "result": receipt["completion"]["result"],
                        "timings_s": receipt["timings_s"],
                    },
                    indent=2,
                )
            )
            return 0
        elif args.command == "verify-job":
            result = verify_job(args.directory)["result"]
        elif args.command == "emx":
            result = run_emx(
                json.loads(args.input.read_text()),
                EmxConfig.load(args.config),
                args.output,
                resume=args.resume,
                cancel=cancel,
            )
        elif args.command == "ngspice-rc":
            result = run_rc(
                json.loads(args.input.read_text()),
                args.output,
                ngspice=args.ngspice,
                timeout_s=args.timeout,
                resume=args.resume,
                cancel=cancel,
            )
        elif args.command == "verify-rc":
            result = verify_rc(args.directory)
        elif args.command == "rtl-smoke":
            result = simulate(
                SimulationRequest(
                    args.design.read_text(), args.testbench.read_text(), timeout_s=args.timeout
                ),
                args.output,
                resume=args.resume,
                cancel=cancel,
            )
        elif args.command == "report":
            rows, torn = read_events(args.directory / "events.jsonl")
            state = status(args.directory)
            report = args.directory / "report.html"
            content = (
                "<!doctype html><meta charset='utf-8'><title>Chips execution report</title>"
                "<style>body{font:15px system-ui;max-width:1100px;margin:40px auto}"
                "pre{white-space:pre-wrap;background:#f3f5f7;padding:16px}</style>"
                "<h1>Chips execution report</h1>"
            )
            content += (
                "<p>Check the result verdict and grader scope. "
                "This report is not circuit certification.</p>"
            )
            content += (
                f"<p>Incomplete final event: {torn}</p><h2>Status</h2>"
                f"<pre>{html.escape(json.dumps(state, indent=2))}</pre><h2>Events</h2>"
            )
            content += "".join(
                f"<pre>{html.escape(json.dumps(row, indent=2))}</pre>" for row in rows
            )
            report.write_text(content)
            print(report.resolve())
            return 0
        else:
            while True:
                state = status(args.directory)
                print(json.dumps(state, ensure_ascii=False), flush=True)
                if (
                    args.command == "status"
                    or state["state"] in {"finished", "attention_required"}
                    or cancel.wait(1)
                ):
                    return 0
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["execution"] == "ok" and result.get("verdict") != "fail" else 1
    except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"execution": "configuration_or_evidence_error", "error": str(error)}))
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
