"""Wrap the entire native Agent process in Codex's OS sandbox.

The inner Agent's workspace-write setting alone only constrains its commands.
This outer process boundary also covers native file reads and service traffic.
No model is started by constructing this policy or running its access probe.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


class NativeSandbox:
    """Create a private, fixed permission profile for one public workspace.

    Callers must supply every private session, verifier and task-source root as
    protected_paths. Only explicit public files, executable runtimes and the
    public MCP socket may be granted. The returned environment overrides belong
    to the outer launcher; set inner CODEX_HOME through argv using /usr/bin/env.
    """

    def __init__(
        self,
        *,
        codex: Path,
        directory: Path,
        workspace: Path,
        readonly_paths: tuple[Path, ...] = (),
        socket_paths: tuple[Path, ...] = (),
        protected_paths: tuple[Path, ...] = (),
        allowed_hosts: tuple[str, ...] = (),
    ):
        self.codex = Path(codex).absolute()
        self.directory = Path(directory).resolve()
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("public workspace must be a directory")
        reads = tuple(Path(p).resolve(strict=True) for p in readonly_paths)
        sockets = tuple(Path(p).resolve() for p in socket_paths)
        protected = (*[Path(p).resolve() for p in protected_paths], self.directory)
        if any(_overlaps(grant, deny) for grant in (self.workspace, *reads) for deny in protected):
            raise ValueError("public access overlaps protected material")
        if any(p == Path("/") for p in reads):
            raise ValueError("runtime grants must not expose the filesystem root")
        for host in allowed_hosts:
            if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host):
                raise ValueError("network grants require exact lowercase host names")
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.socket_paths = sockets
        quote = json.dumps
        lines = [
            'default_permissions = "harness"',
            'approval_policy = "never"',
            "[features]",
            f"network_proxy = {str(bool(allowed_hosts)).lower()}",
            "[permissions.harness.filesystem]",
            '":minimal" = "read"',
            f'{quote(str(self.workspace))} = "write"',
        ]
        lines.extend(f'{quote(str(p))} = "read"' for p in sorted(set(reads)))
        lines.extend(f'{quote(str(p))} = "deny"' for p in sorted(set(protected)))
        lines.extend(
            [
                "[permissions.harness.network]",
                f"enabled = {str(bool(allowed_hosts)).lower()}",
                "allow_local_binding = false",
                "allow_upstream_proxy = false",
                "dangerously_allow_all_unix_sockets = false",
            ]
        )
        if allowed_hosts:
            lines.append("[permissions.harness.network.domains]")
            lines.extend(f'{quote(host)} = "allow"' for host in sorted(set(allowed_hosts)))
        policy = "\n".join(lines) + "\n"
        (self.directory / "config.toml").write_text(policy)
        (self.directory / "config.toml").chmod(0o400)
        self.receipt = {
            "schema_version": 1,
            "enforcement": "outer_codex_sandbox",
            "policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
            "network_hosts": list(sorted(set(allowed_hosts))),
            "access_probe": "not_run",
        }

    def command(self, argv: list[str]) -> tuple[list[str], dict[str, str]]:
        if not argv or any(not isinstance(a, str) or "\0" in a for a in argv):
            raise ValueError("command must be a nonempty argv")
        command = [str(self.codex), "sandbox", "-P", "harness", "-C", str(self.workspace)]
        for path in self.socket_paths:
            command.extend(["--allow-unix-socket", str(path)])
        return [*command, "--", *argv], {"CODEX_HOME": str(self.directory)}

    def probe(self, *, timeout_s: float = 20) -> dict:
        """Check the selected policy with synthetic private/public files.

        The canary is in the denied policy directory. An exit status alone is
        insufficient: both the public write and absence of the private marker
        are checked. This is access evidence, not a real Agent acceptance run.
        """
        marker = os.urandom(24).hex()
        canary = self.directory / "private-canary"
        canary.write_text(marker)
        public = self.workspace / (".sandbox-probe-" + os.urandom(8).hex())
        command, overrides = self.command(
            [
                "/bin/sh",
                "-c",
                'printf public > "$1"; if cat "$2"; then exit 9; fi; test -f "$1"',
                "probe",
                str(public),
                str(canary),
            ]
        )
        try:
            result = subprocess.run(
                command,
                cwd=self.workspace,
                env={"PATH": os.environ.get("PATH", os.defpath), **overrides},
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if result.returncode != 0 or marker in result.stdout or not public.is_file():
                raise RuntimeError("native Agent access probe failed")
            self.receipt["access_probe"] = "synthetic_file_boundary_passed"
            return dict(self.receipt)
        finally:
            public.unlink(missing_ok=True)
            canary.unlink(missing_ok=True)
