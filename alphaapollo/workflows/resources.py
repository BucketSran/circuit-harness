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

"""Validate and compose configured Workflow resources through one public API.

Shared resource families live in _resources: Runtime and generation backends,
Verifier binding, generic Environments, and lifecycle management. This entry
point orders cross-resource validation and construction, then retains the
shared resources and the retained Robotics composition adapter.

Configuration selects declared implementations; it never imports arbitrary
Python objects. Every public composition call validates before acquisition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from alphaapollo.workflows._resources import environment as environment_resources
from alphaapollo.workflows._resources import lifecycle as lifecycle_resources
from alphaapollo.workflows._resources import runtime as runtime_resources
from alphaapollo.workflows._resources import verifier as verifier_resources
from alphaapollo.workflows._resources.lifecycle import ExecutionResources as ExecutionResources
from alphaapollo.workflows.config import (
    ConfigError,
    ResourceConfig,
    RunConfig,
    _closed_options,
    _non_empty_string,
    _positive_int,
    _thaw_json,
)
from alphaapollo.workflows.records import Workflow

__all__ = ["ExecutionResources", "compose_resources", "validate_composition_config"]


def _validate_composition_config(
    config: RunConfig,
) -> dict[str, verifier_resources._AgentVerifierSpec | None]:
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    for name, resource in config.runtimes.items():
        runtime_resources._validate_runtime_resource(name, resource)
    _validate_environment_resource(config.environment)
    for name, resource in config.verifiers.items():
        verifier_resources._validate_verifier_resource(name, resource, config)

    resolved = verifier_resources._resolve_verifier_specs(config)
    # This is a semantic fake-backend ambiguity, so classify it as a config
    # error before run_workflow creates a backend.
    verifier_resources._fake_verifier_formats(resolved)
    _validate_environment_requirement(config, resolved)
    _validate_tool_composition(config, resolved)
    return resolved


def _validate_environment_requirement(
    config: RunConfig,
    verifier_specs: Mapping[str, verifier_resources._AgentVerifierSpec | None],
) -> None:
    """Reject a missing Environment for a native Runtime.

    External runtimes may retain their self-owned tool loop under
    ``environment.type: none`` or route an exclusive bridge through the
    configured Runtime-owned Environment.
    """

    needed = runtime_resources._needed_runtime_keys(config, verifier_specs)
    alphaapollo_runtimes = sorted(
        name for name in needed if config.runtimes[name].type == "alphaapollo"
    )
    if config.environment.type == "none":
        if alphaapollo_runtimes:
            raise ConfigError(
                "environment.type 'none' requires every referenced runtime to own its loop; "
                f"these need an Environment: {alphaapollo_runtimes}"
            )
        return


def _validate_environment_resource(resource: ResourceConfig) -> None:
    if resource.type == "robotics":
        _validate_robotics_environment_options(resource.options)
        return
    if resource.type not in {"default", "text_only", "none"}:
        raise ConfigError(
            "environment.type must be 'default', 'robotics', "
            "'text_only', or 'none', "
            f"got {resource.type!r}"
        )
    environment_resources._validate_environment_resource(resource)


def _runtime_tool_schemas(
    config: RunConfig,
    verifier_specs: Mapping[str, verifier_resources._AgentVerifierSpec | None],
    runtime_name: str,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve Workflow tool grants through Common's canonical catalog."""

    from alphaapollo.common.execution.tools import ToolCatalog, list_tool_specs
    from alphaapollo.common.execution.tools.chips import EMX_TOOL_SPEC
    from alphaapollo.common.execution.tools.python import PYTHON_EXECUTE_SPEC

    common_specs = (
        *list_tool_specs(include_internal=True),
        PYTHON_EXECUTE_SPEC,
        EMX_TOOL_SPEC,
    )
    if config.environment.type == "robotics":
        from alphaapollo.common.execution.tools.robotics import ROBOTICS_TOOL_SPECS

        catalog = ToolCatalog((*common_specs, *ROBOTICS_TOOL_SPECS))
    else:
        catalog = ToolCatalog(common_specs)
    return environment_resources._tool_schemas(
        catalog,
        runtime_resources._runtime_tool_ids(config, verifier_specs, runtime_name),
        runtime_name=runtime_name,
    )


def _validate_tool_composition(
    config: RunConfig,
    verifier_specs: Mapping[str, verifier_resources._AgentVerifierSpec | None],
) -> None:
    runtime_names = runtime_resources._needed_runtime_keys(config, verifier_specs)
    native_names = [name for name in runtime_names if config.runtimes[name].type != "external"]
    schemas_by_runtime = {
        name: _runtime_tool_schemas(config, verifier_specs, name) for name in runtime_names
    }
    for name in runtime_names:
        if "emx_simulate" in runtime_resources._runtime_tool_ids(config, verifier_specs, name):
            if config.runtimes[name].type != "external" or config.environment.type != "none":
                raise ConfigError(
                    "emx_simulate requires an external runtime and environment.type='none'"
                )
    if config.environment.type == "text_only" and any(
        schemas_by_runtime[name] for name in runtime_names
    ):
        raise ConfigError("tools require environment.type='default'")
    if config.environment.type == "robotics":
        from alphaapollo.common.execution.tools.robotics import ROBOTICS_TOOL_SPECS

        allowed = {spec.tool_id for spec in ROBOTICS_TOOL_SPECS}
        for name in runtime_names:
            unsupported = sorted(
                set(runtime_resources._runtime_tool_ids(config, verifier_specs, name)) - allowed
            )
            if unsupported:
                raise ConfigError(
                    f"runtime {name!r} grants non-robotics tools to RobotEnvironment: {unsupported}"
                )
    for name in native_names:
        sampling = dict(_thaw_json(config.runtimes[name].options).get("sampling", {}))
        if sampling.get("tool_choice") is not None and not schemas_by_runtime[name]:
            raise ConfigError(f"runtime {name!r}.options.sampling.tool_choice requires role tools")
    for name in runtime_names:
        resource = config.runtimes[name]
        if resource.type != "external":
            continue
        where = f"runtimes.{name}"
        # Session keywords belong to the selected agent, and a bridged session
        # is shaped by the grants resolved above, so bind it here: a typo or an
        # agent that cannot serve the requested tools is a configuration error,
        # not a TypeError at the first subprocess.
        runtime_resources._probe_external_session(
            _thaw_json(resource.options)["agent"],
            runtime_resources._external_session_options(
                resource,
                runtime_resources._runtime_tool_ids(config, verifier_specs, name),
                where=where,
                environment_backed=config.environment.type != "none",
            ),
            where=f"{where}.options.session",
        )


# ---- Composition and public entry points ----------------------------------


def _environment_factory(resource: ResourceConfig) -> Any:
    options = _thaw_json(resource.options)

    def create(task: object) -> environment_resources._OwnedEnvironment:
        if resource.type == "robotics":
            environment = _robotics_environment(task, options)
        else:
            environment = environment_resources._build_environment(resource.type, options, task)
        return environment_resources._OwnedEnvironment(environment)

    return create


def validate_composition_config(config: RunConfig) -> None:
    """Validate all cross-resource contracts without retaining private specs."""

    _validate_composition_config(config)


def compose_resources(config: RunConfig) -> ExecutionResources:
    """Resolve one validated config into owned Runtime and Verifier registries."""

    verifier_specs = _validate_composition_config(config)
    workflow = Workflow.from_config(
        runtime_resources._effective_workflow_config(config, verifier_specs)
    )
    owned: list[object] = []
    try:
        environment_factory = _environment_factory(config.environment)
        needed_runtime_keys = runtime_resources._needed_runtime_keys(config, verifier_specs)
        runtimes: dict[str, object] = {}
        verifier_formats = verifier_resources._fake_verifier_formats(verifier_specs)
        for name in sorted(needed_runtime_keys):
            runtime, backend = runtime_resources._build_runtime(
                name,
                config.runtimes[name],
                environment_factory=(
                    None if config.environment.type == "none" else environment_factory
                ),
                verifier_formats=verifier_formats.get(name, {}),
                tools=_runtime_tool_schemas(config, verifier_specs, name),
                tool_ids=runtime_resources._runtime_tool_ids(config, verifier_specs, name),
            )
            owned.extend((backend, runtime))
            runtimes[name] = runtime

        verifiers: dict[str, object] = {}
        for name, spec in verifier_specs.items():
            resource = config.verifiers[name]
            verifier, dependencies = verifier_resources._build_verifier(resource, spec, runtimes)
            owned.extend(dependencies)
            owned.append(verifier)
            verifiers[name] = verifier
    except BaseException:
        lifecycle_resources._shutdown_resources(owned)
        raise
    return ExecutionResources(
        workflow=workflow,
        runtimes=runtimes,
        verifiers=verifiers,
        environment_factory=environment_factory,
        _owned=tuple(owned),
    )


# ---- Robotics composition adapters ----------------------------------------


class _ScriptedVLAProvider:
    def __init__(
        self,
        actions: Sequence[Mapping[str, Any]],
        *,
        provider: str = "fake-vla",
        model: str | None = None,
    ) -> None:
        from alphaapollo.common.execution.robotics import RobotAction

        try:
            self._actions = tuple(RobotAction(**_thaw_json(action)) for action in actions)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"environment fake VLA actions are invalid: {exc}") from exc
        if not self._actions:
            raise ConfigError("environment fake VLA actions must not be empty")
        self._provider = _non_empty_string(provider, "environment fake VLA provider")
        self._model = model
        if model is not None:
            _non_empty_string(model, "environment fake VLA model")

    def predict(self, request: object) -> object:
        from alphaapollo.common.execution.tools.robotics import VLARequest, VLAResult

        if not isinstance(request, VLARequest):
            raise TypeError("fake VLA provider expects VLARequest")
        return VLAResult(
            actions=self._actions[: request.max_actions],
            provider=self._provider,
            model=self._model,
        )

    def close(self) -> None:
        return None


class _ScriptedPerceptionProvider:
    def __init__(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        provider: str = "fake-perception",
        model: str | None = None,
    ) -> None:
        if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
            raise ConfigError("environment fake perception items must be a sequence")
        if any(not isinstance(item, Mapping) for item in items):
            raise ConfigError("environment fake perception items must contain mappings")
        self._items = tuple(dict(item) for item in items)
        self._provider = _non_empty_string(provider, "environment fake perception provider")
        self._model = model
        if model is not None:
            _non_empty_string(model, "environment fake perception model")

    def inspect(self, request: object) -> object:
        from alphaapollo.common.execution.tools.robotics import (
            PerceptionRequest,
            PerceptionResult,
        )

        if not isinstance(request, PerceptionRequest):
            raise TypeError("fake perception provider expects PerceptionRequest")
        return PerceptionResult(
            items=self._items,
            provider=self._provider,
            model=self._model,
        )

    def close(self) -> None:
        return None


def _validate_robotics_environment_options(value: object) -> None:
    options = _closed_options(
        value,
        allowed={
            "backend",
            "providers",
            "artifact_root",
            "max_turns",
            "max_episode_steps",
            "tool_timeout_s",
            "feedback_mode",
        },
        required={"backend", "providers"},
        where="environment.options",
    )
    for name in ("max_turns", "max_episode_steps"):
        if options.get(name) is not None:
            _positive_int(options[name], f"environment.options.{name}")
    timeout = options.get("tool_timeout_s")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ConfigError("environment.options.tool_timeout_s must be finite and positive")
    feedback_mode = options.get("feedback_mode", "differentiated")
    if feedback_mode not in {"differentiated", "generic", "missing"}:
        raise ConfigError(
            "environment.options.feedback_mode must be differentiated, generic, or missing"
        )
    artifact_root = options.get("artifact_root")
    if artifact_root is not None:
        _non_empty_string(artifact_root, "environment.options.artifact_root")
    _validate_robotics_backend(options["backend"], artifact_root=artifact_root)
    providers = _closed_options(
        options["providers"],
        allowed={"vla", "perception"},
        required={"vla", "perception"},
        where="environment.options.providers",
    )
    _validate_robotics_provider(providers["vla"], kind="vla")
    _validate_robotics_provider(providers["perception"], kind="perception")


def _validate_robotics_backend(value: object, *, artifact_root: object) -> None:
    """Validate a configured robotics backend without opening transports."""

    resource = _nested_resource(value, where="environment.options.backend")
    if resource.type == "fake":
        _fake_robot_backend(value)
        return
    if resource.type not in {"libero", "remote", "robocasa"}:
        raise ConfigError(
            "environment.options.backend.type must be 'fake', 'libero', 'remote', or 'robocasa', "
            f"got {resource.type!r}"
        )
    if artifact_root is None:
        raise ConfigError(
            f"environment.options.artifact_root is required for a {resource.type} backend"
        )
    if resource.type == "robocasa":
        options = _closed_options(
            resource.options,
            allowed={"environment_version", "enable_depth"},
            required=set(),
            where="environment.options.backend.options",
        )
        if "environment_version" in options:
            _non_empty_string(
                options["environment_version"],
                "environment.options.backend.options.environment_version",
            )
        _validate_enable_depth(options)
        return
    if resource.type == "libero":
        _validate_libero_backend_options(resource.options)
        return
    options = _closed_options(
        resource.options,
        allowed={"endpoint", "timeout_s"},
        required={"endpoint"},
        where="environment.options.backend.options",
    )
    _non_empty_string(options["endpoint"], "environment.options.backend.options.endpoint")
    timeout = options.get("timeout_s")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ConfigError(
            "environment.options.backend.options.timeout_s must be finite and positive"
        )


def _validate_enable_depth(options: Mapping[str, Any]) -> None:
    """Reject a non-bool ``enable_depth`` before a backend renders anything.

    The option gates live metric depth, which the geometry tools need. Accepting
    a truthy non-bool here would let a config read as enabling depth while the
    backend received something it never promised to honour.
    """

    if "enable_depth" not in options:
        return
    if not isinstance(options["enable_depth"], bool):
        raise ConfigError("environment.options.backend.options.enable_depth must be a boolean")


def _validate_libero_backend_options(value: object) -> None:
    """Validate the in-process LIBERO simulator options without importing LIBERO."""

    options = _closed_options(
        value,
        allowed={
            "camera_height",
            "camera_width",
            "max_episode_steps",
            "environment_version",
            "image_orientation",
            "settle_steps",
            "record_dir",
            "record_fps",
            "enable_depth",
        },
        required=set(),
        where="environment.options.backend.options",
    )
    _validate_enable_depth(options)
    for name in ("camera_height", "camera_width", "max_episode_steps", "record_fps"):
        if name in options:
            candidate = options[name]
            if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 1:
                raise ConfigError(
                    f"environment.options.backend.options.{name} must be a positive integer"
                )
    if "settle_steps" in options:
        candidate = options["settle_steps"]
        if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 0:
            raise ConfigError(
                "environment.options.backend.options.settle_steps must be a non-negative integer"
            )
    if "environment_version" in options:
        _non_empty_string(
            options["environment_version"],
            "environment.options.backend.options.environment_version",
        )
    if "record_dir" in options:
        _non_empty_string(
            options["record_dir"],
            "environment.options.backend.options.record_dir",
        )
    if "image_orientation" in options and options["image_orientation"] not in {
        "raw",
        "upright",
        "rotate_180",
    }:
        raise ConfigError(
            "environment.options.backend.options.image_orientation must be "
            "'raw', 'upright', or 'rotate_180'"
        )


def _nested_resource(value: object, *, where: str) -> ResourceConfig:
    raw = _closed_options(
        value,
        allowed={"type", "options"},
        required={"type"},
        where=where,
    )
    resource_type = _non_empty_string(raw["type"], f"{where}.type")
    options = raw.get("options", {})
    if not isinstance(options, Mapping):
        raise ConfigError(f"{where}.options must be a mapping")
    return ResourceConfig(type=resource_type, options=options)


def _robot_observation(value: object, *, where: str) -> object:
    from alphaapollo.common.execution.robotics import RobotObservation

    raw = _closed_options(
        value,
        allowed={"state", "artifact_refs", "timestamp", "backend_metadata"},
        required={"state"},
        where=where,
    )
    try:
        return RobotObservation(
            state=raw["state"],
            artifact_refs=tuple(raw.get("artifact_refs", ())),
            timestamp=raw.get("timestamp", ""),
            backend_metadata=raw.get("backend_metadata", {}),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} is invalid: {exc}") from exc


def _robot_transition(value: object, *, where: str) -> object:
    from alphaapollo.common.execution.robotics import RobotTransition

    raw = _closed_options(
        value,
        allowed={
            "observation",
            "steps_used",
            "terminated",
            "truncated",
            "success",
            "termination_reason",
            "info",
        },
        required={"observation", "steps_used", "terminated", "truncated", "success"},
        where=where,
    )
    try:
        return RobotTransition(
            observation=_robot_observation(raw["observation"], where=f"{where}.observation"),
            steps_used=raw["steps_used"],
            terminated=raw["terminated"],
            truncated=raw["truncated"],
            success=raw["success"],
            termination_reason=raw.get("termination_reason"),
            info=raw.get("info", {}),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} is invalid: {exc}") from exc


def _fake_robot_backend(value: object) -> object:
    from alphaapollo.common.execution.robotics.backends import FakeBackend

    resource = _nested_resource(value, where="environment.options.backend")
    if resource.type != "fake":
        raise ConfigError("environment.options.backend.type currently must be 'fake'")
    options = _closed_options(
        resource.options,
        allowed={"initial_observation", "transitions", "reset_info"},
        required={"initial_observation"},
        where="environment.options.backend.options",
    )
    transitions = options.get("transitions", ())
    if isinstance(transitions, (str, bytes)) or not isinstance(transitions, Sequence):
        raise ConfigError("environment.options.backend.options.transitions must be a sequence")
    return FakeBackend(
        initial_observation=_robot_observation(
            options["initial_observation"],
            where="environment.options.backend.options.initial_observation",
        ),
        transitions=tuple(
            _robot_transition(
                transition,
                where=f"environment.options.backend.options.transitions[{index}]",
            )
            for index, transition in enumerate(transitions)
        ),
        reset_info=options.get("reset_info", {}),
    )


def _libero_robot_backend(
    value: object,
    *,
    artifact_store: object,
) -> object:
    """Build the in-process LIBERO simulator backend at the composition boundary."""

    from alphaapollo.common.execution.robotics.backends import LiberoBackend
    from alphaapollo.common.execution.workspace import ArtifactStore

    if not isinstance(artifact_store, ArtifactStore):
        raise TypeError("artifact_store must be an ArtifactStore")
    resource = _nested_resource(value, where="environment.options.backend")
    if resource.type != "libero":
        raise ConfigError("environment.options.backend.type must be 'libero'")
    _validate_libero_backend_options(resource.options)
    options = dict(resource.options)
    return LiberoBackend(
        artifact_store=artifact_store,
        camera_height=options.get("camera_height", 256),
        camera_width=options.get("camera_width", 256),
        max_episode_steps=options.get("max_episode_steps"),
        environment_version=options.get("environment_version"),
        image_orientation=options.get("image_orientation", "upright"),
        settle_steps=options.get("settle_steps", 0),
        record_dir=options.get("record_dir"),
        record_fps=options.get("record_fps", 20),
        enable_depth=options.get("enable_depth", False),
    )


def _remote_robot_backend(
    value: object,
) -> object:
    """Build the generic remote backend over the canonical HTTP RPC transport."""

    from alphaapollo.common.execution.robotics.backends import (
        HttpRobotRpcTransport,
        RemoteBackend,
    )

    resource = _nested_resource(value, where="environment.options.backend")
    if resource.type != "remote":
        raise ConfigError("environment.options.backend.type must be 'remote'")
    options = _closed_options(
        resource.options,
        allowed={"endpoint", "timeout_s"},
        required={"endpoint"},
        where="environment.options.backend.options",
    )
    endpoint = options["endpoint"]
    _non_empty_string(endpoint, "environment.options.backend.options.endpoint")
    timeout = options.get("timeout_s")
    transport = HttpRobotRpcTransport(
        endpoint,
        default_timeout_s=30.0 if timeout is None else float(timeout),
    )
    try:
        return RemoteBackend(
            transport,
            timeout_s=None if timeout is None else float(timeout),
        )
    except BaseException:
        transport.close()
        raise


def _robocasa_robot_backend(
    value: object,
    *,
    artifact_store: object,
) -> object:
    """Build the optional-dependency-free-at-import RoboCasa adapter."""

    from alphaapollo.common.execution.robotics.backends import RoboCasaBackend
    from alphaapollo.common.execution.workspace import ArtifactStore

    if not isinstance(artifact_store, ArtifactStore):
        raise TypeError("artifact_store must be an ArtifactStore")
    resource = _nested_resource(value, where="environment.options.backend")
    if resource.type != "robocasa":
        raise ConfigError("environment.options.backend.type must be 'robocasa'")
    options = _closed_options(
        resource.options,
        allowed={"environment_version", "enable_depth"},
        required=set(),
        where="environment.options.backend.options",
    )
    _validate_enable_depth(options)
    return RoboCasaBackend(
        artifact_store=artifact_store,
        environment_version=options.get("environment_version"),
        enable_depth=options.get("enable_depth", False),
    )


def _fake_vla_provider(value: object) -> _ScriptedVLAProvider:
    resource = _nested_resource(value, where="environment.options.providers.vla")
    if resource.type != "fake":
        raise ConfigError("environment.options.providers.vla.type currently must be 'fake'")
    options = _closed_options(
        resource.options,
        allowed={"actions", "provider", "model"},
        required={"actions"},
        where="environment.options.providers.vla.options",
    )
    actions = options["actions"]
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise ConfigError("environment.options.providers.vla.options.actions must be a sequence")
    if any(not isinstance(action, Mapping) for action in actions):
        raise ConfigError("environment fake VLA actions must contain mappings")
    return _ScriptedVLAProvider(
        actions,
        provider=options.get("provider", "fake-vla"),
        model=options.get("model"),
    )


def _fake_perception_provider(value: object) -> _ScriptedPerceptionProvider:
    resource = _nested_resource(value, where="environment.options.providers.perception")
    if resource.type != "fake":
        raise ConfigError("environment.options.providers.perception.type currently must be 'fake'")
    options = _closed_options(
        resource.options,
        allowed={"items", "provider", "model"},
        required=set(),
        where="environment.options.providers.perception.options",
    )
    return _ScriptedPerceptionProvider(
        options.get("items", ()),
        provider=options.get("provider", "fake-perception"),
        model=options.get("model"),
    )


def _validate_robotics_provider(value: object, *, kind: str) -> None:
    resource = _nested_resource(value, where=f"environment.options.providers.{kind}")
    if resource.type == "fake":
        if kind == "vla":
            _fake_vla_provider(value)
        else:
            _fake_perception_provider(value)
        return
    if resource.type != "mcp":
        raise ConfigError(
            f"environment.options.providers.{kind}.type must be 'fake' or 'mcp', "
            f"got {resource.type!r}"
        )
    if kind == "vla":
        allowed = {"endpoint", "provider", "model", "tool_name", "timeout_s"}
        required = {"endpoint", "provider"}
    else:
        allowed = {
            "endpoint",
            "provider",
            "model",
            "segment_tool_name",
            "detect_tool_name",
            "timeout_s",
        }
        required = {"endpoint", "provider"}
    options = _closed_options(
        resource.options,
        allowed=allowed,
        required=required,
        where=f"environment.options.providers.{kind}.options",
    )
    for name in ("endpoint", "provider"):
        _non_empty_string(
            options[name],
            f"environment.options.providers.{kind}.options.{name}",
        )
    if options.get("model") is not None:
        _non_empty_string(
            options["model"],
            f"environment.options.providers.{kind}.options.model",
        )
    for name in ("tool_name", "segment_tool_name", "detect_tool_name"):
        if options.get(name) is not None:
            _non_empty_string(
                options[name],
                f"environment.options.providers.{kind}.options.{name}",
            )
    timeout = options.get("timeout_s")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ConfigError(
            f"environment.options.providers.{kind}.options.timeout_s must be finite and positive"
        )


def _mcp_vla_provider(value: object) -> object:
    from alphaapollo.common.execution.tools.robotics.mcp import (
        MCPVLAProvider,
        StreamableHTTPMCPClient,
    )

    resource = _nested_resource(value, where="environment.options.providers.vla")
    if resource.type != "mcp":
        raise ConfigError("environment.options.providers.vla.type must be 'mcp'")
    options = _closed_options(
        resource.options,
        allowed={"endpoint", "provider", "model", "tool_name", "timeout_s"},
        required={"endpoint", "provider"},
        where="environment.options.providers.vla.options",
    )
    timeout = options.get("timeout_s")
    if timeout is None:
        timeout = 30.0
    client = StreamableHTTPMCPClient(options["endpoint"], default_timeout_s=timeout)
    try:
        return MCPVLAProvider(
            client,
            provider=options["provider"],
            model=options.get("model"),
            tool_name=options.get("tool_name", "vla_act"),
        )
    except BaseException:
        client.close()
        raise


def _mcp_perception_provider(value: object) -> object:
    from alphaapollo.common.execution.tools.robotics.mcp import (
        MCPPerceptionProvider,
        StreamableHTTPMCPClient,
    )

    resource = _nested_resource(value, where="environment.options.providers.perception")
    if resource.type != "mcp":
        raise ConfigError("environment.options.providers.perception.type must be 'mcp'")
    options = _closed_options(
        resource.options,
        allowed={
            "endpoint",
            "provider",
            "model",
            "segment_tool_name",
            "detect_tool_name",
            "timeout_s",
        },
        required={"endpoint", "provider"},
        where="environment.options.providers.perception.options",
    )
    timeout = options.get("timeout_s")
    if timeout is None:
        timeout = 30.0
    client = StreamableHTTPMCPClient(options["endpoint"], default_timeout_s=timeout)
    try:
        return MCPPerceptionProvider(
            client,
            provider=options["provider"],
            model=options.get("model"),
            segment_tool_name=options.get("segment_tool_name", "segment"),
            detect_tool_name=options.get("detect_tool_name", "detect_objects"),
        )
    except BaseException:
        client.close()
        raise


def _robotics_vla_provider(value: object) -> object:
    resource = _nested_resource(value, where="environment.options.providers.vla")
    if resource.type == "fake":
        return _fake_vla_provider(value)
    if resource.type == "mcp":
        return _mcp_vla_provider(value)
    raise ConfigError(f"unsupported VLA provider type {resource.type!r}")


def _robotics_perception_provider(value: object) -> object:
    resource = _nested_resource(value, where="environment.options.providers.perception")
    if resource.type == "fake":
        return _fake_perception_provider(value)
    if resource.type == "mcp":
        return _mcp_perception_provider(value)
    raise ConfigError(f"unsupported perception provider type {resource.type!r}")


def _robotics_environment(task: object, options: Mapping[str, Any]) -> object:
    from alphaapollo.common.environment.default import ExecutorToolBridge
    from alphaapollo.common.environment.robotics import RobotEnvironment
    from alphaapollo.common.execution.tools import ToolCatalog
    from alphaapollo.common.execution.tools.robotics import (
        ROBOTICS_TOOL_SPECS,
        RoboticsToolExecutor,
        build_robotics_tool_executors,
    )
    from alphaapollo.common.execution.workspace import ArtifactStore

    artifact_root = options.get("artifact_root")
    artifact_store = None
    if artifact_root is not None:
        artifact_store = ArtifactStore(Path(str(artifact_root)))
    backend_resource = _nested_resource(options["backend"], where="environment.options.backend")
    if backend_resource.type == "fake":
        backend = _fake_robot_backend(options["backend"])
    elif backend_resource.type == "libero":
        if artifact_store is None:
            raise ConfigError("environment.options.artifact_root is required for a LIBERO backend")
        backend = _libero_robot_backend(options["backend"], artifact_store=artifact_store)
    elif backend_resource.type == "remote":
        if artifact_store is None:
            raise ConfigError("environment.options.artifact_root is required for a remote backend")
        backend = _remote_robot_backend(options["backend"])
    elif backend_resource.type == "robocasa":
        if artifact_store is None:
            raise ConfigError(
                "environment.options.artifact_root is required for a RoboCasa backend"
            )
        backend = _robocasa_robot_backend(options["backend"], artifact_store=artifact_store)
    else:  # validation normally catches this; keep the factory defensive for direct callers.
        raise ConfigError(
            "environment.options.backend.type must be 'fake', 'libero', 'remote', or 'robocasa', "
            f"got {backend_resource.type!r}"
        )
    resources: list[object] = [backend]
    try:
        providers = dict(options["providers"])
        vla_provider = _robotics_vla_provider(providers["vla"])
        resources.append(vla_provider)
        perception_provider = _robotics_perception_provider(providers["perception"])
        resources.append(perception_provider)
        executor = RoboticsToolExecutor(
            build_robotics_tool_executors(
                vla_provider=vla_provider,
                perception_provider=perception_provider,
                backend=backend,
            )
        )
        resources = [backend, executor]
        allowed_tools = tuple(getattr(task, "tools", ()) or ())
        bridge = ExecutorToolBridge(
            executor,
            allowed_tool_ids=allowed_tools,
        )
        return RobotEnvironment(
            backend=backend,
            tool_bridge=bridge,
            # Argument validation stays a precondition of execution; the
            # environment resolves requests through the catalog before
            # dispatch, keeping the shared bridge robotics-free.
            catalog=ToolCatalog(ROBOTICS_TOOL_SPECS),
            max_turns=options.get("max_turns"),
            max_episode_steps=options.get("max_episode_steps"),
            tool_timeout_s=options.get("tool_timeout_s"),
            feedback_mode=options.get("feedback_mode", "differentiated"),
            load_artifact=None if artifact_store is None else artifact_store.get,
        )
    except BaseException:
        lifecycle_resources._shutdown_resources(resources)
        raise
