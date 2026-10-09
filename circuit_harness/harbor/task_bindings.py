"""Private local task bindings shared by compilation and Harbor adapters."""

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import FinalEvaluationConfig, PublicSessionConfig, validate_final_separation


class TaskBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_path: Path
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    support: Literal["supported", "unsupported"] = "supported"
    reason: str | None = None
    session_config: Path | None = None
    final_config: Path | None = None

    @model_validator(mode="after")
    def declared_support(self):
        if not self.task_path.is_absolute():
            raise ValueError("task_path must be absolute")
        if self.support == "unsupported":
            if not self.reason or not self.reason.strip():
                raise ValueError("unsupported task requires reason")
            if self.session_config is not None or self.final_config is not None:
                raise ValueError("unsupported task cannot declare runnable configs")
        elif self.session_config is None or self.final_config is None:
            raise ValueError("supported task requires session_config and final_config")
        for path in (self.session_config, self.final_config):
            if path is not None and not path.is_absolute():
                raise ValueError("binding config paths must be absolute")
        return self


class TaskBindings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    tasks: list[TaskBinding] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_tasks(self):
        paths = [entry.task_path.resolve() for entry in self.tasks]
        identities = [(entry.task_id, entry.task_version) for entry in self.tasks]
        if len(paths) != len(set(paths)) or len(identities) != len(set(identities)):
            raise ValueError("duplicate or colliding task binding")
        return self


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in task bindings/config")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def content_identity(path):
    """Hash exact local bytes and member names; reject mutable link indirection."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("binding resources cannot be symlinks")
    path = path.resolve(strict=True)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            raise ValueError("binding resources cannot contain symlinks")
        if member.is_file():
            rows.append(
                [
                    member.relative_to(path).as_posix(),
                    hashlib.sha256(member.read_bytes()).hexdigest(),
                ]
            )
        elif not member.is_dir():
            raise ValueError("binding resources must be regular files/directories")
    return _digest(rows)


def _evas_source_identity(checkout):
    from circuit_harness.execution.backends.current_evas import EVAS_SOURCE, snapshot_repository

    with tempfile.TemporaryDirectory(prefix="harbor-evas-identity-") as directory:
        source = snapshot_repository(Path(checkout), Path(directory) / "source", EVAS_SOURCE)
        return _digest({"scope": source["scope"], "files": source["files"]})


def _resource_identity(resource, public):
    if resource == public.checkout:
        return _evas_source_identity(resource)
    if resource in (public.public_codex, public.public_python):
        resolved = Path(resource).resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("public runtime executable must resolve to a regular file")
        return _digest(
            {
                "resolved_path_sha256": hashlib.sha256(str(resolved).encode()).hexdigest(),
                "content_sha256": content_identity(resolved),
            }
        )
    return content_identity(resource)


def _outside(path, roots):
    path = Path(path).resolve()
    if any(path.is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError(
            "private binding resources must be outside all task and trial export roots"
        )


def load_task_bindings(path, *, public_roots=()):
    """Read declarations without executing tasks, models or simulators."""
    manifest = TaskBindings.model_validate(_json(path))
    roots = [*public_roots, *(entry.task_path for entry in manifest.tasks)]
    _outside(path, roots)
    for entry in manifest.tasks:
        if not entry.task_path.resolve().is_dir():
            raise ValueError("binding task_path must be an existing local directory")
        for config in (entry.session_config, entry.final_config):
            if config is not None:
                _outside(config, roots)
    return manifest


@dataclass(frozen=True)
class SelectedTaskBinding:
    binding: TaskBinding
    session_config_path: Path
    final_config_path: Path
    receipt: dict


def resolve_task_binding(
    path, task_path, *, public_roots=(), expected_manifest_sha256=None, expected_receipt=None
):
    """Select by actual canonical task path and reject any compiled input drift."""
    manifest = load_task_bindings(path, public_roots=public_roots)
    manifest_hash = content_identity(path)
    if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256:
        raise ValueError("task bindings manifest drift")
    task_path = Path(task_path).resolve()
    entry = next(
        (entry for entry in manifest.tasks if entry.task_path.resolve() == task_path), None
    )
    if entry is None:
        raise ValueError("missing task binding for actual task_path")
    if entry.support == "unsupported":
        raise ValueError(f"unsupported task {entry.task_id}: {entry.reason}")
    public = PublicSessionConfig.model_validate(_json(entry.session_config))
    final = FinalEvaluationConfig.model_validate(_json(entry.final_config))
    roots = [*public_roots, *(item.task_path for item in manifest.tasks)]
    resources = [entry.session_config, entry.final_config, final.task_package, public.materials]
    resources.extend(
        resource
        for resource in (
            public.checkout,
            public.kernel,
            public.public_task_package,
            public.public_codex,
            public.public_python,
        )
        if resource is not None
    )
    # Any future final backend resource paths remain covered without assuming EVAS semantics.
    for value in final.model_dump().values():
        if isinstance(value, Path) and value not in resources:
            resources.append(value)
    for resource in resources:
        _outside(resource, roots)
    package = _json(final.task_package / "manifest.json")
    for key in ("task_id", "task_version"):
        if public.task.get(key) != getattr(entry, key) or package.get(key) != getattr(entry, key):
            raise ValueError("binding/public/final task identity mismatch")
    validate_final_separation(public, final.task_package, final.remote)
    receipt = dict(
        schema_version=1,
        task_id=entry.task_id,
        task_version=entry.task_version,
        task_path_sha256=hashlib.sha256(str(task_path).encode()).hexdigest(),
        task_source_sha256=content_identity(task_path),
        manifest_sha256=manifest_hash,
        session_config_sha256=content_identity(entry.session_config),
        final_config_sha256=content_identity(entry.final_config),
        resource_sha256=_digest([_resource_identity(resource, public) for resource in resources]),
    )
    receipt["binding_sha256"] = _digest(receipt)
    if expected_receipt is not None and receipt != expected_receipt:
        raise ValueError("selected task binding input drift")
    return SelectedTaskBinding(
        entry, entry.session_config.resolve(), entry.final_config.resolve(), receipt
    )


def write_binding_receipt(selection, trial_dir):
    """Private runner receipt; only IDs and digests, never config paths or contents."""
    destination = Path(trial_dir) / "task-binding.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _json(destination) != selection.receipt:
            raise ValueError("environment and verifier selected different task binding")
    else:
        with destination.open("x") as stream:
            json.dump(selection.receipt, stream, indent=2)
            stream.write("\n")


def validate_job_task_bindings(job):
    """Validate full explicit local coverage, and pin inputs into native kwargs."""
    env = job.environment.kwargs
    verifier = job.verifier.kwargs
    if not env.get("task_bindings") and not verifier.get("task_bindings"):
        return
    if env.get("session_config") is not None or verifier.get("config_path") is not None:
        raise ValueError("scalar config and task_bindings are mutually exclusive")
    if (
        not env.get("task_bindings")
        or not verifier.get("task_bindings")
        or Path(env["task_bindings"]).resolve() != Path(verifier["task_bindings"]).resolve()
    ):
        raise ValueError("environment and verifier require the same task_bindings")
    if (
        job.datasets
        or not job.tasks
        or any(task.git_url is not None or task.path is None for task in job.tasks)
    ):
        raise ValueError(
            "task bindings require explicit local tasks; remote/dynamic datasets unsupported"
        )
    paths = [task.path.resolve() for task in job.tasks]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate planned task paths")
    manifest = load_task_bindings(env["task_bindings"], public_roots=[*paths, job.jobs_dir])
    mapped = {entry.task_path.resolve() for entry in manifest.tasks}
    if mapped - set(paths):
        raise ValueError("unknown task binding outside planned tasks")
    if set(paths) - mapped:
        raise ValueError("missing planned task bindings")
    receipts = {
        str(path): resolve_task_binding(
            env["task_bindings"], path, public_roots=[*paths, job.jobs_dir]
        ).receipt
        for path in paths
    }
    for kwargs in (env, verifier):
        kwargs["task_bindings_sha256"] = content_identity(env["task_bindings"])
        kwargs["task_binding_receipts"] = receipts
