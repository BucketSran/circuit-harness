"""Run one command with a private local Podman store and temporary API service.

No global environment, system service, subordinate IDs or permissions are changed.
The command inherits its caller's credentials; only Podman connection variables change.
"""

import argparse
import fcntl
import json
import os
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from circuit_harness.execution.runtime.journal import atomic_json


class PodmanRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    executable: Path
    root: Path
    runroot: Path
    control_dir: Path
    ignore_chown_errors: bool = False

    @model_validator(mode="after")
    def validate_paths(self):
        paths = (self.executable, self.root, self.runroot, self.control_dir)
        if any(not p.is_absolute() or ".." in p.parts for p in paths):
            raise ValueError("runtime paths must be absolute and canonical")
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise ValueError("declare an executable Podman binary")
        stores = (self.root.resolve(), self.runroot.resolve(), self.control_dir.resolve())
        if any(
            a == b or a in b.parents or b in a.parents
            for i, a in enumerate(stores)
            for b in stores[i + 1 :]
        ):
            raise ValueError("store, runroot and control directories must be separate")
        if len(os.fsencode(self.control_dir / "api.sock")) >= 104:
            raise ValueError("control_dir is too long for a portable Unix socket path")
        return self

    def engine(self):
        argv = [str(self.executable), "--root", str(self.root), "--runroot", str(self.runroot)]
        if self.ignore_chown_errors:
            argv += ["--storage-opt", "ignore_chown_errors=true"]
        return argv


def _private_directory(path, *, new=False):
    path.mkdir(mode=0o700, parents=True, exist_ok=not new)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("runtime directories must be private and owned by the current user")


def _terminate(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5)


def _collect_resources(engine, env, kind, filters):
    command = [*engine, "ps", "-aq"] if kind == "container" else [*engine, "network", "ls", "-q"]
    for value in filters:
        command += ["--filter", value]
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=15, check=True
    )
    return result.stdout.split()


def _cleanup_resources(config, env, runner_id):
    """Reconcile only the runner-labelled public containers and registered trials."""
    engine = config.engine()
    filters = [[f"label=chips.runner={runner_id}"]]
    for registration in (config.control_dir / "projects").glob("*.json"):
        project = json.loads(registration.read_text())["project"]
        filters.append([f"label=com.docker.compose.project={project}"])
    failures = []
    for selected in filters:
        for kind in ("container", "network"):
            try:
                for resource in _collect_resources(engine, env, kind, selected):
                    command = (
                        [*engine, "rm", "-f", resource]
                        if kind == "container"
                        else [*engine, "network", "rm", resource]
                    )
                    subprocess.run(command, env=env, capture_output=True, timeout=30, check=True)
                if _collect_resources(engine, env, kind, selected):
                    failures.append({"kind": kind, "filter": selected, "error": "resources_remain"})
            except (OSError, subprocess.SubprocessError) as error:
                failures.append({"kind": kind, "filter": selected, "error": type(error).__name__})
    if not failures:
        projects = [
            json.loads(p.read_text())["project"]
            for p in (config.control_dir / "projects").glob("*.json")
        ]
        failures.extend(_reap_exec_monitors(config.root, projects))
    return failures


def _reap_exec_monitors(root, projects):
    """Reap delayed exec monitors only after their owned project is absent.

    Linux pidfds prevent PID reuse from targeting another process. No process
    name alone, substring path, shared UID, or process-group guess grants ownership.
    """
    if not Path("/proc").is_dir():
        return []
    names = {name for project in projects for name in (project + "-main-1", project + "_main_1")}
    failures = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        descriptor = None
        try:
            descriptor = os.pidfd_open(int(proc.name))
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if not args or Path(args[0]).name != "conmon" or "--exec-attach" not in args:
                continue
            if "-n" not in args or args[args.index("-n") + 1] not in names:
                continue
            if "-b" not in args:
                continue
            bundle = Path(args[args.index("-b") + 1])
            if not bundle.is_relative_to(root / "overlay-containers"):
                continue
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            import select

            poll = select.poll()
            poll.register(descriptor, select.POLLIN)
            if not poll.poll(1000):
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                if not poll.poll(1000):
                    failures.append({"kind": "exec_monitor", "pid": int(proc.name)})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        finally:
            if descriptor is not None:
                os.close(descriptor)
    return failures


def run(raw_config, command):
    """Own the child and service lifetime, retaining private cleanup evidence."""
    config = PodmanRuntimeConfig.model_validate(raw_config)
    if not command:
        raise ValueError("declare a command after --")
    for path in (config.root, config.runroot):
        _private_directory(path)
    # Prevent two launchers from sharing a store/runroot concurrently.
    with (config.root / ".chips-runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(config, command)


def _run_locked(config, command):
    _private_directory(config.control_dir, new=True)
    bin_dir = config.control_dir / "bin"
    bin_dir.mkdir(mode=0o700)
    (config.control_dir / "projects").mkdir(mode=0o700)
    runner_id = uuid.uuid4().hex
    endpoint = "unix://" + str(config.control_dir / "api.sock")
    wrapper = bin_dir / "podman"
    # JSON/repr are Python literals here, never shell interpolation.
    wrapper.write_text(
        f"#!{sys.executable}\nimport os,sys\n"
        f"engine={config.engine()!r}\na=sys.argv[1:]\n"
        f"if a and a[0]=='run': a[1:1]=['--label=chips.runner={runner_id}']\n"
        "os.execv(engine[0],engine+a)\n"
    )
    wrapper.chmod(0o700)
    env = dict(os.environ)
    for key in ("DOCKER_CONTEXT", "CONTAINER_HOST", "CONTAINER_CONNECTION", "CONTAINER_SSHKEY"):
        env.pop(key, None)
    env.update(
        PATH=str(bin_dir) + os.pathsep + env.get("PATH", os.defpath),
        DOCKER_HOST=endpoint,
        PYTHONUTF8="1",
        PODMAN_COMPOSE_WARNING_LOGS="false",
        CHIPS_PODMAN_CONTROL_DIR=str(config.control_dir),
    )
    receipt = {
        "schema_version": 1,
        "runner_id": runner_id,
        "config": config.model_dump(mode="json"),
        "endpoint": endpoint,
        "cleanup_confirmed": False,
        "returncode": None,
    }
    service = child = None
    previous = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupt)
        with (config.control_dir / "api.log").open("wb") as log:
            service = subprocess.Popen(
                [*config.engine(), "system", "service", "--time=0", endpoint],
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            receipt["service_pid"] = service.pid
            deadline = time.monotonic() + 15
            while not (config.control_dir / "api.sock").exists():
                if service.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("private Podman API did not start; inspect api.log")
                time.sleep(0.05)
            child = subprocess.Popen(command, env=env, start_new_session=True)
            receipt["child_pid"] = child.pid
            receipt["returncode"] = child.wait()
    except KeyboardInterrupt:
        receipt["returncode"] = 130
    finally:
        # Ignore repeated terminal signals while finite owned-resource cleanup runs.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            _terminate(child)
            receipt["cleanup_errors"] = _cleanup_resources(config, env, runner_id)
        finally:
            _terminate(service)
            (config.control_dir / "api.sock").unlink(missing_ok=True)
            receipt["cleanup_confirmed"] = not receipt.get("cleanup_errors", ["cleanup_not_run"])
            atomic_json(config.control_dir / "runtime.json", receipt)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return receipt["returncode"] if receipt["cleanup_confirmed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run(json.loads(args.config.read_text()), command)


if __name__ == "__main__":
    raise SystemExit(main())
