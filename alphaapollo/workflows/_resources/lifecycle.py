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

"""Own composed execution resources and preserve ordered, idempotent cleanup."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from alphaapollo.workflows.records import Workflow


def _shutdown_resources(resources: Sequence[object]) -> BaseException | None:
    first_error: BaseException | None = None
    seen: set[int] = set()
    for resource in reversed(tuple(resources)):
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        for method_name in ("terminate", "close"):
            method = getattr(resource, method_name, None)
            if not callable(method):
                continue
            try:
                if method_name == "terminate":
                    _terminate_resource(method)
                else:
                    method()
            except BaseException as exc:
                first_error = first_error or exc
    return first_error


def _terminate_resource(method: Any, *, reason: str = "workflow_shutdown") -> None:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        method(reason)
        return
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    has_varargs = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if positional or has_varargs:
        method(reason)
    else:
        method()


@dataclass(slots=True)
class ExecutionResources:
    """One fully constructed and lifecycle-owned Workflow execution context."""

    workflow: Workflow
    runtimes: Mapping[str, object]
    verifiers: Mapping[str, object]
    _owned: tuple[object, ...]
    _closed: bool = False
    environment_factory: Any = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error = _shutdown_resources(self._owned)
        if error is not None:
            raise error

    def __enter__(self) -> ExecutionResources:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            self.close()
        except BaseException:
            if exc is None:
                raise
        return False
