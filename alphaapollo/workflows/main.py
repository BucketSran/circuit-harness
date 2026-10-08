# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single command-line entry point for configured AlphaApollo Workflow runs."""

from __future__ import annotations

from collections.abc import Sequence

from alphaapollo.workflows.config import read_config_mapping
from alphaapollo.workflows.run import load_prepared_inputs, run, run_workflow
from alphaapollo.workflows.run import main as _run_main

__all__ = [
    "load_prepared_inputs",
    "main",
    "read_config_mapping",
    "run",
    "run_workflow",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and execute one configured run through the shared application runner."""

    return _run_main(argv)


if __name__ == "__main__":  # pragma: no cover - exercised through main(argv)
    raise SystemExit(main())
