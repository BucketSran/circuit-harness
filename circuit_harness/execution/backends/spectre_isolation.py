"""Operator-selected Spectre process filesystem isolation, independent of grading.

The trusted checker remains outside this process. Network sharing is explicit for
license deployments; it does not provide network egress isolation.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from pathlib import Path

_LICENSE_ENV = {
    "CDS_LIC_FILE",
    "CDSLMD_LICENSE_FILE",
    "LM_LICENSE_FILE",
    "CDS_LIC_ONLY",
    "CDS_LIC_QUEUE",
    "CDS_AUTO_64BIT",
    "LD_LIBRARY_PATH",
    "CDS_INST_DIR",
    "CDS_SPECTRE_DIR",
}
_KEYS = {
    "schema_version",
    "bubblewrap",
    "spectre",
    "runtime_readonly_paths",
    "network",
    "license_env",
}


def _absolute(value):
    if not isinstance(value, str) or not value.startswith("/") or "\0" in value:
        raise ValueError("isolation paths must be absolute")
    path = Path(value)
    if str(path) != value or ".." in path.parts:
        raise ValueError("isolation paths must be canonical")
    return path


def _configuration(path):
    path = _absolute(str(path))
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("isolation configuration must be an owned private regular file")
    if info.st_size > 65536:
        raise ValueError("isolation configuration exceeds limit")
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or set(config) != _KEYS or config["schema_version"] != 1:
        raise ValueError("invalid Spectre isolation configuration")
    if config["network"] not in {"disabled", "shared_license"}:
        raise ValueError("declare disabled or shared_license network")
    names = config["license_env"]
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or name not in _LICENSE_ENV for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("only declared simulator/license environment names are allowed")
    for key in ("bubblewrap", "spectre"):
        tool = _absolute(config[key])
        if not tool.is_file() or not os.access(tool, os.X_OK):
            raise ValueError("declared isolation executables must exist")
    roots = config["runtime_readonly_paths"]
    if not isinstance(roots, list) or not roots or len(roots) > 32:
        raise ValueError("declare bounded runtime read-only paths")
    checked = []
    for value in roots:
        root = _absolute(value)
        if (
            root == Path("/")
            or not root.exists()
            or (root.resolve() == path.resolve() or root.resolve() in path.resolve().parents)
        ):
            raise ValueError("runtime grants must exclude the configuration and filesystem root")
        checked.append(root)
    simulator = Path(config["spectre"]).resolve()
    if not any(
        simulator == root.resolve() or simulator.is_relative_to(root.resolve()) for root in checked
    ):
        raise ValueError("Spectre must be inside a declared runtime grant")
    return config, hashlib.sha256(raw).hexdigest()


def isolated_spectre_command(config_path, arguments, *, cwd=None):
    """Build one explicit namespace launch without running a simulator.

    Every existing work file/directory is mounted read-only over the writable
    output directory. Callers must use a condition directory containing only
    candidate inputs and that condition's netlist, never a checker/task package.
    -W binds no condition directory and never executes candidate code.
    """
    config, config_sha = _configuration(config_path)
    if not arguments or any(not isinstance(arg, str) or "\0" in arg for arg in arguments):
        raise ValueError("declare simulator arguments")
    version_only = arguments == ["-W"]
    work = Path(cwd or os.getcwd()).absolute()
    if not version_only and (work.is_symlink() or not work.is_dir() or work == Path("/")):
        raise ValueError("condition directory must be a regular directory")
    if not version_only and Path(config_path).resolve().is_relative_to(work.resolve()):
        raise ValueError(
            "private isolation configuration must remain outside the condition directory"
        )
    roots = [Path(value) for value in config["runtime_readonly_paths"]]
    if not version_only and any(
        root.resolve() == work.resolve()
        or root.resolve() in work.resolve().parents
        or work.resolve() in root.resolve().parents
        for root in roots
    ):
        raise ValueError("condition directory and runtime grants must be disjoint")
    inputs = [] if version_only else list(work.iterdir())
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in inputs):
        raise ValueError("condition inputs must be regular files/directories without symlinks")
    if any(path.is_symlink() for root in inputs if root.is_dir() for path in root.rglob("*")):
        raise ValueError("condition input directories must not contain symlinks")
    argv = [
        config["bubblewrap"],
        "--die-with-parent",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        "--cap-drop",
        "ALL",
        "--tmpfs",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    if config["network"] == "disabled":
        argv.append("--unshare-net")
    for root in roots:
        argv += ["--ro-bind", str(root), str(root)]
    argv += [
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "TMPDIR",
        "/tmp",
    ]
    environment = {"PATH": "/usr/bin:/bin"}
    for name in config["license_env"]:
        if name in os.environ:
            environment[name] = os.environ[name]
            argv += ["--setenv", name, os.environ[name]]
    receipt = {
        "schema_version": 1,
        "backend": "bubblewrap_spectre_process",
        "configuration_sha256": config_sha,
        "network": config["network"],
        "version_only": version_only,
        "inputs_readonly": True,
        "mount_scope": "condition_directory_and_declared_runtime",
        "model_credentials_passed": False,
    }
    if not version_only:
        name = ".harness-spectre-isolation-" + uuid.uuid4().hex + ".json"
        receipt["input_files"] = {
            str(path.relative_to(work)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in work.rglob("*")
            if path.is_file()
        }
        receipt_path = work / name
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        inputs.append(receipt_path)
        argv += ["--bind", str(work), str(work)]
        for path in inputs:
            argv += ["--ro-bind", str(path), str(path)]
        argv += ["--chdir", str(work)]
    else:
        argv += ["--chdir", "/tmp"]
    argv += [config["spectre"], *arguments]
    return argv, environment


def exec_isolated_spectre(config_path, arguments):
    """Replace the launcher, preserving the durable job's owned process group."""
    argv, environment = isolated_spectre_command(config_path, arguments)
    os.execve(argv[0], argv, environment)
