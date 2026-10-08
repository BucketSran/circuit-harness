"""Generate or verify the circuit configuration and trajectory JSON Schemas."""

import argparse
import json
from pathlib import Path


def rendered_schemas():
    from circuit_harness.data.prepare_atif import AtifDatasetManifest
    from circuit_harness.harbor.config import (
        FinalEvaluationConfig,
        HarborChipsConfig,
        PublicSessionConfig,
    )
    from circuit_harness.harbor.podman_runner import PodmanRuntimeConfig
    from circuit_harness.harbor.profiles import catalog_schema
    from circuit_harness.harbor.reporting import ExperimentReport
    from circuit_harness.harbor.task_bindings import TaskBindings

    schemas = {
        "data/json/atif_dataset_manifest.json": AtifDatasetManifest.model_json_schema(),
        "harbor/config.schema.json": HarborChipsConfig.model_json_schema(),
        "harbor/public_session.schema.json": PublicSessionConfig.model_json_schema(),
        "harbor/final_evaluation.schema.json": FinalEvaluationConfig.model_json_schema(),
        "harbor/podman_runner.schema.json": PodmanRuntimeConfig.model_json_schema(),
        "harbor/profiles.schema.json": catalog_schema(),
        "harbor/reporting.schema.json": ExperimentReport.model_json_schema(),
        "harbor/task_bindings.schema.json": TaskBindings.model_json_schema(),
    }
    schemas["harbor/task_bindings.schema.json"]["$schema"] = (
        "https://json-schema.org/draft/2020-12/schema"
    )
    return {
        name: json.dumps(value, indent=2, sort_keys=True) + "\n" for name, value in schemas.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    mismatches = []
    for name, expected in rendered_schemas().items():
        path = Path(__file__).parent / name
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
        else:
            try:
                matches = json.loads(path.read_text()) == json.loads(expected)
            except (OSError, ValueError):
                matches = False
            if not matches:
                mismatches.append(name)
    if mismatches:
        parser.error(
            "Stale schemas: "
            + ", ".join(mismatches)
            + "; run python -m circuit_harness._schema_export --write"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
