"""Operator declarations for one Harbor-owned attempt."""

from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

HARBOR_VERSION = "0.23.0"


def require_harbor_version() -> None:
    if version("harbor") != HARBOR_VERSION:
        raise RuntimeError(f"Harness plugins require harbor=={HARBOR_VERSION}")


class PublicSessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    task: dict[str, Any]
    materials: Path
    checkout: Path | None = None
    kernel: Path | None = None
    public_backend: Literal["docker", "podman", "native_codex_sandbox", "remote_spectre"] = "docker"
    public_cpu_limit: float | None = Field(default=1, gt=0, le=1, strict=True)
    public_remote: dict[str, str] | None = None
    public_task_package: Path | None = None
    public_codex: Path | None = None
    public_python: Path | None = None
    image: str | None = Field(min_length=1)
    max_actions: int = Field(default=24, ge=1)
    max_simulations: int = Field(default=4, ge=1)
    simulation_timeout_s: float = Field(default=120, gt=0)
    max_output_bytes: int = Field(default=16 * 1024 * 1024, ge=1)

    @model_validator(mode="after")
    def validate_public_backend(self):
        if self.public_backend not in {"docker", "podman"} and self.public_cpu_limit != 1:
            raise ValueError("public_cpu_limit applies only to container backends")
        if self.public_backend != "remote_spectre" and (
            self.checkout is None or self.kernel is None
        ):
            raise ValueError("local EVAS backend requires checkout and kernel")
        if self.public_backend == "remote_spectre":
            if (
                self.image is not None
                or self.public_remote is None
                or self.public_task_package is None
            ):
                raise ValueError(
                    "remote public backend requires image=null and public package/endpoint"
                )
            if self.task.get("experiments") is not None:
                raise ValueError("remote public experiments unsupported")
            for endpoint in (self.public_remote,):
                for key in (
                    "python",
                    "bundle",
                    "profile",
                    "run_root",
                    "archive_root",
                    "upload_root",
                ):
                    if key not in endpoint:
                        continue  # Complete endpoint presence is checked during preparation.
                    value = endpoint[key]
                    path = PurePosixPath(value)
                    if (
                        not value.startswith("/")
                        or value.startswith("//")
                        or str(path) != value
                        or ".." in path.parts
                        or "\\" in value
                        or "\0" in value
                    ):
                        raise ValueError("remote paths must be canonical absolute POSIX paths")
        elif self.public_remote is not None or self.public_task_package is not None:
            raise ValueError("remote declarations require remote_spectre")
        elif self.public_backend == "native_codex_sandbox":
            if self.image is not None:
                raise ValueError("native public backend requires image=null")
            if any(
                path is None or not path.is_absolute()
                for path in (self.public_codex, self.public_python)
            ):
                raise ValueError(
                    "native public backend requires explicit absolute Codex/Python paths"
                )
            if self.task.get("experiments") is not None:
                raise ValueError("public experiments require a container backend")
        elif self.image is None:
            raise ValueError("Container public backend requires an immutable image")
        return self

    @classmethod
    def read(cls, path: Path) -> "PublicSessionConfig":
        return cls.model_validate_json(path.read_text())


class OpensourceEvaluationConfig(BaseModel):
    """The benchmark-declared replay settings; no commercial resource access."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    schema_version: Literal[1]
    image: str = Field(pattern=r"^(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[a-f0-9]{64}$")
    task_package_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    solver_options: dict[str, Any]
    unsupported: list[str]
    timeout_s: float = Field(gt=0, le=300)
    max_output_bytes: int = Field(ge=1, le=16 * 1024 * 1024)


def _backend_schema(backend="backend", remote="remote", opensource="opensource"):
    return {
        "allOf": [
            {
                "if": {
                    "properties": {backend: {"const": "benchmark_opensource"}},
                    "required": [backend],
                },
                "then": {
                    "required": [opensource],
                    "properties": {opensource: {"type": "object"}},
                    "not": {"required": [remote]},
                },
                "else": {
                    "required": [remote],
                    "properties": {remote: {"type": "object"}},
                    "not": {"required": [opensource]},
                },
            }
        ]
    }


class FinalEvaluationConfig(BaseModel):
    """Private verifier configuration, independent of Agent and public environment."""

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_extra=_backend_schema())
    task_package: Path
    backend: Literal["remote_spectre", "benchmark_opensource"] = "remote_spectre"
    remote: dict[str, str] | None = None
    opensource: OpensourceEvaluationConfig | None = None

    @model_serializer(mode="wrap")
    def serialize_backend(self, handler):
        values = handler(self)
        values.pop("opensource" if self.backend == "remote_spectre" else "remote", None)
        return values

    @model_validator(mode="after")
    def validate_backend(self):
        if self.backend == "remote_spectre":
            if self.remote is None or "opensource" in self.model_fields_set:
                raise ValueError("remote_spectre requires remote and forbids opensource")
        elif self.opensource is None or "remote" in self.model_fields_set:
            raise ValueError("benchmark_opensource requires opensource and forbids remote")
        return self


class HarborChipsConfig(PublicSessionConfig):
    """Compatible operator configuration for the optional native Codex path."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        json_schema_extra=_backend_schema("final_backend", "final_remote", "final_opensource"),
    )
    final_task_package: Path
    final_backend: Literal["remote_spectre", "benchmark_opensource"] = "remote_spectre"
    final_remote: dict[str, str] | None = None
    final_opensource: OpensourceEvaluationConfig | None = None
    final_timeout_s: float = Field(default=600, gt=0)
    executable: Path = Path("codex")
    auth_file: Path | None = None
    runtime_readonly_paths: tuple[Path, ...] = ()
    allowed_hosts: tuple[str, ...] = ()
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] = "medium"

    @model_serializer(mode="wrap")
    def serialize_final_backend(self, handler):
        values = handler(self)
        values.pop(
            "final_opensource" if self.final_backend == "remote_spectre" else "final_remote", None
        )
        return values

    def final_evaluation(self) -> FinalEvaluationConfig:
        values = {"task_package": self.final_task_package, "backend": self.final_backend}
        if "final_remote" in self.model_fields_set:
            values["remote"] = self.final_remote
        if "final_opensource" in self.model_fields_set:
            values["opensource"] = self.final_opensource
        return FinalEvaluationConfig.model_validate(values)

    @model_validator(mode="after")
    def validate_final_separation(self):
        final = self.final_evaluation()
        validate_final_separation(self, final.task_package, final.remote)
        return self


def validate_final_separation(public, task_package, remote):
    if public.public_backend != "remote_spectre":
        return
    if public.public_task_package == task_package:
        raise ValueError("public and final task packages must be separate")
    if remote is None:
        return
    for key in ("python", "bundle", "profile", "run_root", "archive_root", "upload_root"):
        value = remote.get(key)
        if value is None:
            continue
        path = PurePosixPath(value)
        if (
            not value.startswith("/")
            or value.startswith("//")
            or str(path) != value
            or ".." in path.parts
            or "\\" in value
            or "\0" in value
        ):
            raise ValueError("remote paths must be canonical absolute POSIX paths")
    if public.public_remote.get("host") == remote.get("host"):
        keys = ("profile", "run_root", "archive_root", "upload_root")
        for public_key in keys:
            for final_key in keys:
                a = public.public_remote.get(public_key)
                b = remote.get(final_key)
                if (
                    a is None
                    or b is None
                    or PurePosixPath(a) == PurePosixPath(b)
                    or PurePosixPath(a) in PurePosixPath(b).parents
                    or PurePosixPath(b) in PurePosixPath(a).parents
                ):
                    raise ValueError("public and final remote resources must be separate")
