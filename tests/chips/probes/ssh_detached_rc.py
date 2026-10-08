"""Explicit live SSH disconnect test; only use on an authorized private server root.

Requires a deployed chips.pyz and ngspice 47. Writes one private test wrapper and
one job. The wrapper delays the simulation stage before invoking real ngspice;
it does not synthesize numerical data. Kills only this probe's local SSH client.
"""

import argparse
import json
import os
import selectors
import shlex
import subprocess
import time
from pathlib import Path

PREPARE = r"""
import json, os, sys, time
from pathlib import Path
root, job_id = Path(sys.argv[1]), sys.argv[2]
jobs = Path(sys.argv[3]) if sys.argv[3] else root / 'jobs'
archive = Path(sys.argv[4]) if sys.argv[4] else None
archive_options = {'archive_root': archive} if archive is not None else {}
bundle = Path(sys.argv[5]) if sys.argv[5] else root / 'harness/chips.pyz'
sys.path.insert(0, str(bundle))
from alphaapollo.common.execution.chips.jobs import submit_rc
from alphaapollo.common.execution.chips.journal import file_digest
work = root / 'validation' / job_id
work.mkdir(mode=0o700, parents=True, exist_ok=False)
binary = root / 'envs/ngspice/47/bin/ngspice'
tool = work / 'delayed-ngspice'
count = work / 'launches'
tool.write_text('#!' + sys.executable + '\n' +
    'import os,sys,time\nfrom pathlib import Path\n' +
    'if "-v" not in sys.argv:\n' +
    '    with Path(' + repr(str(count)) + ').open("a") as f: f.write("simulation\\n")\n' +
    '    time.sleep(8)\n' +
    'os.execv(' + repr(str(binary)) + ',[' + repr(str(binary)) + '] + sys.argv[1:])\n')
tool.chmod(0o700)
task = json.loads((root / 'harness/rc.json').read_text())
reply = submit_rc(task, jobs, job_id, ngspice=str(tool), timeout_s=30, **archive_options)
deadline = time.monotonic() + 10
while not count.exists() and time.monotonic() < deadline: time.sleep(.05)
assert count.exists(), reply
assert not (jobs / job_id / 'completion.json').exists()
print(json.dumps({'stage': 'running_before_disconnect', 'server_time': time.time(),
    'submission_state': reply['state'], 'ngspice_sha256': file_digest(binary)}), flush=True)
# Keep the submitting SSH session alive. No loop here monitors or advances the job.
time.sleep(45)
"""

CHECK = r"""
import json, sys, time
from pathlib import Path
root, job_id = Path(sys.argv[1]), sys.argv[2]
jobs = Path(sys.argv[3]) if sys.argv[3] else root / 'jobs'
archive = Path(sys.argv[4]) if sys.argv[4] else None
archive_options = {'archive_root': archive} if archive is not None else {}
bundle = Path(sys.argv[5]) if sys.argv[5] else root / 'harness/chips.pyz'
sys.path.insert(0, str(bundle))
from alphaapollo.common.execution.chips.jobs import inspect_job, submit_rc, verify_job
observed_at = time.time()
job = jobs / job_id
state = inspect_job(job)
assert state['state'] == 'finished', state
completion = verify_job(job)
assert completion['finished_at'] < observed_at
assert completion['result']['execution'] == 'ok'
assert completion['result']['verdict'] == 'pass'
archive_receipt = None
if archive is not None:
 from alphaapollo.common.execution.chips.archive import verify_archive
 archive_receipt = verify_archive(archive / job_id)
 assert archive_receipt['archived_at'] < observed_at
 assert archive_receipt['completion'] == completion
tool = root / 'validation' / job_id / 'delayed-ngspice'
task = json.loads((root / 'harness/rc.json').read_text())
retry = submit_rc(task, jobs, job_id, ngspice=str(tool), timeout_s=30, **archive_options)
assert retry['state'] == 'finished'
launches = (tool.parent / 'launches').read_text().splitlines()
assert launches == ['simulation'], launches
print(json.dumps({'observed_at': observed_at, 'finished_before_reconnect': True,
    'solver_launches': len(launches), 'completion': completion,
    'archive_receipt': archive_receipt}), flush=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--root", required=True, help="absolute private server root")
    parser.add_argument("--work-root", default="", help="absolute server-local job root")
    parser.add_argument("--archive-root", default="", help="absolute persistent archive root")
    parser.add_argument("--bundle", default="", help="absolute versioned CLI bundle")
    parser.add_argument("--python", default="/usr/local/bin/python3.12")
    parser.add_argument("--job-id", required=True, help="new ID; never reuse test directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.host.startswith("-") or not Path(args.root).is_absolute():
        parser.error("host cannot be an option; root must be absolute")
    for value in (args.work_root, args.archive_root, args.bundle):
        if value and not Path(value).is_absolute():
            parser.error("optional server paths must be absolute")
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.job_id):
        parser.error("invalid job ID")
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ControlPersist=no",
        args.host,
        shlex.join(
            [
                "env",
                "-i",
                "PATH=/usr/bin:/bin",
                "LC_ALL=C",
                args.python,
                "-S",
                "-",
                args.root,
                args.job_id,
                args.work_root,
                args.archive_root,
                args.bundle,
            ]
        ),
    ]
    with (args.output / "ssh.stderr.log").open("w") as err:
        client = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            start_new_session=True,
        )
        try:
            client.stdin.write(PREPARE.encode())
            client.stdin.close()
            with selectors.DefaultSelector() as selector:
                selector.register(client.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=25):
                    raise TimeoutError("server did not reach the injection point")
                ready = json.loads(client.stdout.readline())
            assert ready["stage"] == "running_before_disconnect", ready
            (args.output / "before-disconnect.json").write_text(json.dumps(ready, indent=2))
            client.terminate()
            exit_code = client.wait(timeout=5)
            (args.output / "client.json").write_text(json.dumps({"exit_code": exit_code}))
        finally:
            if client.poll() is None:
                client.terminate()
                client.wait(timeout=5)
    # No connection, polling, or local controller exists during this interval.
    time.sleep(12)
    check = subprocess.run(command, input=CHECK, text=True, capture_output=True, timeout=25)
    (args.output / "after-reconnect.json").write_text(check.stdout)
    (args.output / "check.stderr.log").write_text(check.stderr)
    if check.returncode:
        raise RuntimeError(f"remote verification failed: {check.stderr}")
    result = json.loads(check.stdout)
    print(
        json.dumps(
            {
                "job_id": args.job_id,
                "finished_before_reconnect": True,
                "solver_launches": result["solver_launches"],
                "verdict": result["completion"]["result"]["verdict"],
                "archived_before_reconnect": result["archive_receipt"] is not None,
            }
        )
    )


if __name__ == "__main__":
    main()
