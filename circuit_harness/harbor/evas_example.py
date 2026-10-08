"""Compatibility entrypoint for the VA07 benchmark recipe."""

from circuit_harness.benchmarks.evas_va07 import (
    CHECKER_PACKAGE_SHA256,
    SOURCE_COMMIT,
    SOURCE_PATHS,
    SOURCE_URL,
    TASK_VERSION,
    main,
    prepare,
    result,
    run,
)

__all__ = [
    "CHECKER_PACKAGE_SHA256",
    "SOURCE_COMMIT",
    "SOURCE_PATHS",
    "SOURCE_URL",
    "TASK_VERSION",
    "main",
    "prepare",
    "result",
    "run",
]

if __name__ == "__main__":
    raise SystemExit(main())
