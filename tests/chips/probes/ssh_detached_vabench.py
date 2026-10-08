"""Explicit SSH loss test around a real EVAS simulation (operator fixtures only).

A private EVAS wrapper pauses for eight seconds immediately before simulation.
Only this probe's SSH client is killed; no shared connection or server is changed.
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
import json, shlex, sys, time
from pathlib import Path
root, job_id = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root / 'harness/chips-vabench.pyz'))
from alphaapollo.common.execution.chips.jobs import submit_vabench
from alphaapollo.common.execution.chips.vabench import pin_vabench
work = root / 'validation' / job_id
work.mkdir(mode=0o700, parents=True)
real_python = root / 'envs/vaevas-0.8.7/bin/python'
real_evas = real_python.parent / 'evas'
python = work / 'python'
python.write_text('#!/bin/sh\nexec ' + shlex.quote(str(real_python)) + ' "$@"\n')
python.chmod(0o700)
evas = work / 'evas'
count = work / 'simulation-launches'
evas.write_text('#!' + str(real_python) + '\n' +
    'import os,sys,time\nfrom pathlib import Path\n' +
    'if len(sys.argv)>1 and sys.argv[1] == "simulate":\n' +
    '    with Path(' + repr(str(count)) + ').open("a") as f: f.write("simulation\\n")\n' +
    '    time.sleep(8)\n' +
    'os.execv(' + repr(str(real_evas)) + ',[' + repr(str(real_evas)) + '] + sys.argv[1:])\n')
evas.chmod(0o700)
source = root / 'sources/vaevas-0685aae-chips'
pin = pin_vabench(source, python, 'v4-001')
(work / 'pin.json').write_text(json.dumps(pin))
release = source / 'benchmark-vabench-release-v4/release/benchmarkv4-r53'
candidate = release / 'tasks/001-bang-bang-phase-detector/evaluator/solution'
state = submit_vabench(pin, candidate, root / 'jobs', job_id, timeout_s=120)
deadline = time.monotonic() + 45
while not count.exists() and time.monotonic() < deadline: time.sleep(.1)
assert count.exists(), state
assert not (root / 'jobs' / job_id / 'completion.json').exists()
print(json.dumps({'stage':'simulation_before_disconnect', 'server_time':time.time()}), flush=True)
time.sleep(55)
"""

CHECK = r"""
import json, sys, time
from pathlib import Path
root, job_id = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root / 'harness/chips-vabench.pyz'))
from alphaapollo.common.execution.chips.jobs import verify_job, submit_vabench
observed = time.time()
job = root / 'jobs' / job_id
result = verify_job(job)
assert result['finished_at'] < observed
assert result['result']['verdict'] == 'pass'
work = root / 'validation' / job_id
pin = json.loads((work / 'pin.json').read_text())
release = Path(pin['source']) / 'benchmark-vabench-release-v4/release/benchmarkv4-r53'
candidate = release / 'tasks/001-bang-bang-phase-detector/evaluator/solution'
assert submit_vabench(pin, candidate, root / 'jobs', job_id, timeout_s=120)['state'] == 'finished'
launches = (work / 'simulation-launches').read_text().splitlines()
assert launches == ['simulation'], launches
print(json.dumps({'observed_at':observed, 'finished_before_reconnect':True,
                  'simulation_launches':len(launches), 'completion':result}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import re

    if args.host.startswith("-") or not Path(args.root).is_absolute():
        parser.error("invalid host or root")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.job_id):
        parser.error("invalid job ID")
    os.umask(0o077)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
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
                "/usr/local/bin/python3.12",
                "-B",
                "-",
                args.root,
                args.job_id,
            ]
        ),
    ]
    with (args.output / "ssh.stderr.log").open("w") as err:
        client = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err
        )
        try:
            client.stdin.write(PREPARE.encode())
            client.stdin.close()
            with selectors.DefaultSelector() as selector:
                selector.register(client.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=60):
                    raise TimeoutError("simulation injection point not reached")
                ready = json.loads(client.stdout.readline())
                assert ready["stage"] == "simulation_before_disconnect"
            (args.output / "before.json").write_text(json.dumps(ready))
        finally:
            if client.poll() is None:
                client.terminate()
            code = client.wait(timeout=5)
            (args.output / "client.json").write_text(json.dumps({"exit_code": code}))
    time.sleep(20)  # Deliberately no server connection/controller in this interval.
    check = subprocess.run(command, input=CHECK, text=True, capture_output=True, timeout=60)
    (args.output / "after.json").write_text(check.stdout)
    (args.output / "check.stderr.log").write_text(check.stderr)
    if check.returncode:
        raise RuntimeError(check.stderr)
    value = json.loads(check.stdout)
    print(
        json.dumps(
            {
                "finished_before_reconnect": value["finished_before_reconnect"],
                "simulation_launches": value["simulation_launches"],
            }
        )
    )


if __name__ == "__main__":
    main()
