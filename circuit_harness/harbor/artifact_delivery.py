"""Checked native Harbor artifact delivery, separate from public-session bundles."""

import hashlib
import json
import logging
import math
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath

from harbor.models.task.artifacts import validate_artifact_entries
from harbor.models.task.config import ArtifactConfig, TaskOS
from harbor.models.trial.paths import EnvironmentPaths
from harbor.trial.artifact_handler import ArtifactHandler, artifact_host_path
from harbor.utils.scripts import build_execution_command

from circuit_harness.execution.evaluation.candidate_bundle import MAX_CANDIDATE_BYTES, MAX_FILES
from circuit_harness.execution.runtime.journal import atomic_json, file_digest

from .config import require_harbor_version


class DeliveryError(RuntimeError):
    """Delivery could not be established; this is not a benchmark score."""


class _VerifierExecution:
    """Observe the original native entrypoint without changing its script or reward."""

    def __init__(self, environment, command, tests):
        self.environment = environment
        self.command = command
        self.tests = tests
        self.completed = False

    def __getattr__(self, name):
        return getattr(self.environment, name)

    async def exec(self, command, **kwargs):
        if command == self.command:
            if self.completed:
                raise DeliveryError("native verifier entrypoint ran more than once")
            # Native Verifier may reuse preloaded tests or overlay step tests.
            # Check what it actually staged before running any scoring code.
            with tempfile.TemporaryDirectory(prefix="artifact-checker-") as directory:
                root = Path(directory)
                manifest = await ArtifactHandler(
                    artifacts=["/tests"], logger=logging.getLogger(__name__)
                ).download_artifacts(self.environment, root, source_artifacts_dir="/logs/artifacts")
                entries = [entry for entry in manifest.entries if entry.source == "/tests"]
                if (
                    len(entries) != 1
                    or entries[0].status != "ok"
                    or _test_bytes(_tree(root / "tests")) != _test_bytes(self.tests)
                ):
                    raise DeliveryError("staged checker differs from bound task tests")
        result = await self.environment.exec(command, **kwargs)
        if command == self.command:
            if result.return_code != 0:
                raise DeliveryError("native verifier entrypoint did not finish successfully once")
            self.completed = True
        return result


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_record(path):
    metadata = path.stat()
    return {
        "sha256": file_digest(path),
        "bytes": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
    }


def _test_bytes(tree):
    # Native Verifier adds executable permission to test.sh before invoking it.
    return {
        "files": {
            name: {key: row[key] for key in ("sha256", "bytes")}
            for name, row in tree["files"].items()
        },
        "directories": tree["directories"],
    }


def _tree(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise DeliveryError("delivery tree must be a regular directory")
    files, directories = {}, []
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise DeliveryError(f"symlink in delivery: {name}")
        if path.is_dir():
            directories.append(name)
        elif path.is_file():
            files[name] = _file_record(path)
        else:
            raise DeliveryError(f"non-regular delivery file: {name}")
    if (
        len(files) + len(directories) > MAX_FILES
        or sum(row["bytes"] for row in files.values()) > MAX_CANDIDATE_BYTES
    ):
        raise DeliveryError("delivery exceeds existing candidate file/byte limits")
    return {"files": files, "directories": directories}


def _app_name(value):
    if not isinstance(value, str):
        raise DeliveryError("delivery paths must be strings below /app")
    path = PurePosixPath(value)
    if (
        not value.startswith("/app/")
        or str(path) != value.rstrip("/")
        or any(part in {".", ".."} for part in path.parts)
        or any(c in value for c in "\\\0*?[]")
    ):
        raise DeliveryError(f"noncanonical delivery path: {value}")
    return path.relative_to("/app").as_posix()


def _verifier_identity(task):
    return {
        "task_config": file_digest(task.paths.task_dir / "task.toml"),
        "instruction": file_digest(task.paths.task_dir / "instruction.md"),
        "tests": _tree(task.paths.tests_dir),
    }


class ArtifactDelivery:
    """Collect native artifacts and bind their final bytes to a trusted public baseline.

    Call after the solver and its writers have stopped. This opt-in author/runtime
    adapter does not replace Harbor Trial orchestration or public-session schemas.
    """

    def __init__(
        self, starter, artifacts, *, task_id, task_version, fixed_paths=(), verifier_task=None
    ):
        require_harbor_version()
        if not all(isinstance(value, str) and value for value in (task_id, task_version)):
            raise DeliveryError("task_id and task_version must be nonempty strings")
        self.starter = Path(starter)
        self.baseline = _tree(self.starter)
        self.artifacts = tuple(
            ArtifactConfig(source=a)
            if isinstance(a, str)
            else ArtifactConfig.model_validate(a).model_copy(deep=True)
            for a in artifacts
        )
        if not self.artifacts:
            raise DeliveryError("declare at least one artifact")
        names, destinations = [], []
        self.kinds = {}
        for artifact in self.artifacts:
            name = _app_name(artifact.source)
            artifact.source = "/app/" + name
            if artifact.destination:
                path = PurePosixPath(artifact.destination)
                if path.is_absolute() or str(path) != artifact.destination or ".." in path.parts:
                    raise DeliveryError("artifact destination must be a canonical relative path")
            if artifact.exclude or artifact.service not in (None, "main"):
                raise DeliveryError("delivery uses main-service artifacts without exclusions")
            if name in self.baseline["files"]:
                self.kinds[name] = "file"
            elif name in self.baseline["directories"]:
                self.kinds[name] = "directory"
            else:
                raise DeliveryError(f"artifact has no starter interface: {name}")
            names.append(name)
            host = artifact_host_path(Path("/artifacts"), artifact)
            if not host.is_relative_to("/artifacts"):
                raise DeliveryError("artifact destination escapes collection directory")
            destinations.append(host.as_posix())
        for scopes in (names, destinations):
            if any(
                a == b or a.startswith(b + "/") or b.startswith(a + "/")
                for i, a in enumerate(scopes)
                for b in scopes[i + 1 :]
            ):
                raise DeliveryError("overlapping delivery scopes or destinations")
        validate_artifact_entries(self.artifacts, convention_source="/logs/artifacts")
        self.fixed_paths = tuple(_app_name(value) for value in fixed_paths)
        for fixed in self.fixed_paths:
            if fixed not in self.baseline["files"] and fixed not in self.baseline["directories"]:
                raise DeliveryError(f"fixed path has no trusted baseline: {fixed}")
            if any(name == fixed or name.startswith(fixed + "/") for name in names):
                raise DeliveryError("an artifact cannot be entirely fixed")
        self.descriptor = {
            "task_id": task_id,
            "task_version": task_version,
            "artifacts": [a.model_dump(mode="json") for a in self.artifacts],
            "kinds": self.kinds,
            "fixed_paths": list(self.fixed_paths),
            "baseline": self.baseline,
            "verifier": _verifier_identity(verifier_task) if verifier_task is not None else None,
        }

    def _handler(self):
        return ArtifactHandler(artifacts=self.artifacts, logger=logging.getLogger(__name__))

    def _mutable(self, name):
        if any(name == fixed or name.startswith(fixed + "/") for fixed in self.fixed_paths):
            return False
        return any(
            name == scope or (kind == "directory" and name.startswith(scope + "/"))
            for scope, kind in self.kinds.items()
        )

    def _candidate_identity(self, tree):
        return _digest(
            {
                "task_id": self.descriptor["task_id"],
                "task_version": self.descriptor["task_version"],
                "scopes": self.kinds,
                **tree,
            }
        )

    async def collect(self, environment, destination):
        if (
            environment.os != TaskOS.LINUX
            or not isinstance(environment.session_id, str)
            or not environment.session_id
        ):
            raise DeliveryError("delivery requires a named Linux environment")
        destination = Path(destination)
        destination.mkdir(mode=0o700, parents=True, exist_ok=False)
        collected = destination / "collected"
        manifest = await self._handler().download_artifacts(
            environment, collected, source_artifacts_dir="/logs/artifacts"
        )
        snapshot = destination / "files"
        snapshot.mkdir()
        for artifact in self.artifacts:
            name = _app_name(artifact.source)
            entries = [entry for entry in manifest.entries if entry.source == artifact.source]
            if (
                len(entries) != 1
                or entries[0].status != "ok"
                or entries[0].type != self.kinds[name]
            ):
                raise DeliveryError(f"artifact collection incomplete: {artifact.source}")
            source = artifact_host_path(collected, artifact)
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if self.kinds[name] == "directory":
                _tree(source)
                shutil.copytree(source, target, symlinks=True)
            elif source.is_symlink() or not source.is_file():
                raise DeliveryError(f"artifact must be a regular file: {artifact.source}")
            else:
                shutil.copy2(source, target)
        for fixed in self.fixed_paths:
            path = snapshot / fixed
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        shutil.copytree(self.starter, destination / "baseline", symlinks=True)
        if _tree(destination / "baseline") != self.baseline or _tree(self.starter) != self.baseline:
            raise DeliveryError("starter changed during collection")
        tree = _tree(snapshot)
        if any(
            fixed.startswith(name + "/") for fixed in self.fixed_paths for name in tree["files"]
        ):
            raise DeliveryError("candidate file blocks a protected baseline path")
        receipt = {
            "schema_version": 1,
            "plan": self.descriptor,
            "source_session": environment.session_id,
            "collection_manifest_sha256": file_digest(collected / "manifest.json"),
            **tree,
            "candidate_sha256": self._candidate_identity(tree),
            "plan_sha256": _digest(self.descriptor),
        }
        receipt["receipt_sha256"] = _digest(receipt)
        atomic_json(destination / "delivery.json", receipt)
        self.verify(destination)
        return destination

    def verify(self, frozen):
        frozen = Path(frozen)
        path = frozen / "delivery.json"
        if path.is_symlink() or not path.is_file():
            raise DeliveryError("missing delivery receipt")
        receipt = json.loads(path.read_text())
        tree = _tree(frozen / "files")
        if (
            set(receipt)
            != {
                "schema_version",
                "plan",
                "source_session",
                "collection_manifest_sha256",
                "files",
                "directories",
                "candidate_sha256",
                "plan_sha256",
                "receipt_sha256",
            }
            or receipt.get("schema_version") != 1
            or not isinstance(receipt.get("source_session"), str)
            or not receipt["source_session"]
            or receipt.get("receipt_sha256")
            != _digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            or receipt.get("plan") != self.descriptor
            or _tree(frozen / "baseline") != self.baseline
            or receipt.get("collection_manifest_sha256")
            != file_digest(frozen / "collected/manifest.json")
            or receipt.get("files") != tree["files"]
            or receipt.get("directories") != tree["directories"]
            or receipt.get("candidate_sha256") != self._candidate_identity(tree)
            or receipt.get("plan_sha256") != _digest(self.descriptor)
        ):
            raise DeliveryError("frozen delivery changed or belongs to another task/plan")
        return receipt

    def _prepare_app(self, frozen, destination):
        shutil.copytree(frozen / "baseline", destination)
        for name, kind in self.kinds.items():
            source, target = frozen / "files" / name, destination / name
            if kind == "directory":
                shutil.rmtree(target)
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        # Native directory replacement clears the entire subtree. Reinsert protected
        # material from the frozen author baseline, never from collected agent bytes.
        for name in self.baseline["directories"]:
            if not self._mutable(name):
                (destination / name).mkdir(parents=True, exist_ok=True)
        for name in self.baseline["files"]:
            if not self._mutable(name):
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(frozen / "baseline" / name, target)

    async def restore(self, environment, frozen):
        frozen = Path(frozen)
        receipt = self.verify(frozen)
        if environment.session_id == receipt["source_session"]:
            raise DeliveryError("independent verifier environment required")
        if environment.os != TaskOS.LINUX:
            raise DeliveryError("delivery requires a Linux verifier environment")
        with tempfile.TemporaryDirectory(prefix="artifact-restore-") as directory:
            root = Path(directory)
            app = root / "app"
            self._prepare_app(frozen, app)
            uploads = root / "uploads"
            for artifact in self.artifacts:
                name = _app_name(artifact.source)
                target = artifact_host_path(uploads, artifact)
                target.parent.mkdir(parents=True, exist_ok=True)
                if self.kinds[name] == "directory":
                    shutil.copytree(app / name, target)
                else:
                    shutil.copy2(app / name, target)
            await self._handler().upload_artifacts(
                environment,
                uploads,
                source_artifacts_dir="/logs/artifacts",
                target_artifacts_dir="/logs/artifacts",
            )
            # Baseline files outside artifact scopes are restored through the same
            # native transfer machinery; protected files inside a directory are above.
            fixed = []
            for name in self.baseline["files"]:
                if any(name == scope or name.startswith(scope + "/") for scope in self.kinds):
                    continue
                fixed.append("/app/" + name)
            await ArtifactHandler(
                artifacts=fixed, logger=logging.getLogger(__name__)
            ).upload_artifacts(
                environment,
                root,
                source_artifacts_dir="/logs/artifacts",
                target_artifacts_dir="/logs/artifacts",
            )
            await environment.ensure_dirs(
                [
                    "/app/" + name
                    for name in self.baseline["directories"]
                    if not self._mutable(name)
                ],
                chmod=True,
            )
            expected = _tree(app)
            await self._check_placement(environment, expected)
            return expected

    async def _check_placement(self, environment, expected, *, allow_outputs=False):
        with tempfile.TemporaryDirectory(prefix="artifact-placement-") as directory:
            root = Path(directory)
            if allow_outputs:
                # Read only known inputs after grading. Simulator build products
                # and waveforms are outputs, not additions to the submitted source.
                manifest = await ArtifactHandler(
                    artifacts=["/app/" + name for name in expected["files"]],
                    logger=logging.getLogger(__name__),
                ).download_artifacts(environment, root, source_artifacts_dir="/logs/artifacts")
                entries = {entry.source: entry for entry in manifest.entries}
                for name, record in expected["files"].items():
                    entry = entries.get("/app/" + name)
                    path = root / "app" / name
                    if (
                        entry is None
                        or entry.status != "ok"
                        or entry.type != "file"
                        or path.is_symlink()
                        or not path.is_file()
                        or _file_record(path) != record
                    ):
                        raise DeliveryError(f"grader changed or lost input: {name}")
                for name in expected["directories"]:
                    if not await environment.service_is_dir("/app/" + name, user="root"):
                        raise DeliveryError(f"grader removed input directory: {name}")
                return
            manifest = await ArtifactHandler(
                artifacts=["/app"], logger=logging.getLogger(__name__)
            ).download_artifacts(environment, root, source_artifacts_dir="/logs/artifacts")
            app = [entry for entry in manifest.entries if entry.source == "/app"]
            if len(app) != 1 or app[0].status != "ok" or _tree(root / "app") != expected:
                raise DeliveryError(
                    "restored engineering tree differs from frozen delivery and baseline"
                )

    async def grade(self, verifier, frozen):
        """Restore into a separate environment, then run the supplied native verifier.

        The caller owns environment lifecycle and deadlines. Only a completed,
        finite native reward with unchanged inputs gets a delivery result receipt.
        """
        frozen = Path(frozen)
        receipt = self.verify(frozen)
        logs = verifier.trial_paths.verifier_dir
        if (
            self.descriptor["verifier"] is None
            or _verifier_identity(verifier.task) != self.descriptor["verifier"]
        ):
            raise DeliveryError("bind the expected verifier task before collecting the delivery")
        if logs.exists() and any(logs.iterdir()):
            raise DeliveryError("verifier output directory must be fresh")
        tests = _tree(verifier.task.paths.tests_dir)
        expected = await self.restore(verifier.environment, frozen)
        await verifier.environment.empty_dirs(["/logs/verifier"], chmod=True)
        logs.mkdir(parents=True, exist_ok=True)
        environment = verifier.environment
        env_paths = EnvironmentPaths.for_os(environment.os)
        command = build_execution_command(
            env_paths.tests_dir / "test.sh",
            stdout_path=env_paths.verifier_dir
            / verifier.trial_paths.test_stdout_path.relative_to(logs),
            task_os=environment.os,
        )
        execution = _VerifierExecution(environment, command, tests)
        verifier.environment = execution
        try:
            result = await verifier.verify()
        finally:
            verifier.environment = environment
        if not execution.completed:
            raise DeliveryError("native verifier did not run the expected test.sh entrypoint")
        score = result.rewards.get("reward")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise DeliveryError("native verifier did not produce a finite task reward")
        await self._check_placement(verifier.environment, expected, allow_outputs=True)
        self.verify(frozen)
        if _tree(verifier.task.paths.tests_dir) != tests:
            raise DeliveryError("verifier source changed during grading")
        atomic_json(
            logs / "delivery-result.json",
            {
                "schema_version": 1,
                "task_id": self.descriptor["task_id"],
                "task_version": self.descriptor["task_version"],
                "candidate_sha256": receipt["candidate_sha256"],
                "plan_sha256": receipt["plan_sha256"],
                "verifier_sha256": _digest(tests),
                "rewards": result.rewards,
            },
        )
        return result
