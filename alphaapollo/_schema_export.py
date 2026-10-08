"""Generate and verify JSON Schema snapshots owned by component packages."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from alphaapollo.common.artifacts.schemas import SCHEMA_MODELS as ARTIFACT_MODELS
from alphaapollo.common.execution.tools.schemas import SCHEMA_MODELS as TOOL_MODELS
from alphaapollo.common.trajectory.schemas import SCHEMA_MODELS as TRAJECTORY_MODELS
from alphaapollo.data_preprocess.prepare_atif import AtifDatasetManifest
from alphaapollo.learning.adapters.schemas import SCHEMA_MODELS as LEARNING_ADAPTER_MODELS


@dataclass(frozen=True, slots=True)
class SchemaOwner:
    package: str
    models: dict[str, type[BaseModel]]


SCHEMA_OWNERS = (
    SchemaOwner("alphaapollo.common.artifacts", ARTIFACT_MODELS),
    SchemaOwner("alphaapollo.common.execution.tools", TOOL_MODELS),
    SchemaOwner("alphaapollo.common.trajectory", TRAJECTORY_MODELS),
    SchemaOwner("alphaapollo.learning.adapters", LEARNING_ADAPTER_MODELS),
    SchemaOwner("alphaapollo.data_preprocess", {"atif_dataset_manifest": AtifDatasetManifest}),
)


def _snapshot_dir(owner: SchemaOwner) -> Path:
    module = importlib.import_module(owner.package)
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        raise RuntimeError(f"package {owner.package} has no filesystem location")
    return Path(module_path).with_name("json")


def rendered_schemas() -> dict[tuple[str, str], str]:
    return {
        (owner.package, f"{name}.json"): (
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        )
        for owner in SCHEMA_OWNERS
        for name, model in owner.models.items()
    }


def write_snapshots() -> None:
    rendered = rendered_schemas()
    for owner in SCHEMA_OWNERS:
        directory = _snapshot_dir(owner)
        directory.mkdir(parents=True, exist_ok=True)
        expected = {filename for package, filename in rendered if package == owner.package}
        for stale in directory.glob("*.json"):
            if stale.name not in expected:
                stale.unlink()
        for filename in sorted(expected):
            directory.joinpath(filename).write_text(
                rendered[(owner.package, filename)], encoding="utf-8"
            )


def check_snapshots() -> list[str]:
    mismatches: list[str] = []
    for (package, filename), expected in rendered_schemas().items():
        owner = next(item for item in SCHEMA_OWNERS if item.package == package)
        path = _snapshot_dir(owner) / filename
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            mismatches.append(f"{package}:{filename}")
            continue
        if actual != json.loads(expected):
            mismatches.append(f"{package}:{filename}")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_snapshots()
        return 0
    mismatches = check_snapshots()
    if mismatches:
        parser.error(
            "JSON Schema snapshots are missing or stale: "
            + ", ".join(mismatches)
            + "; run python -m alphaapollo._schema_export --write"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_OWNERS",
    "SchemaOwner",
    "check_snapshots",
    "main",
    "rendered_schemas",
    "write_snapshots",
]
