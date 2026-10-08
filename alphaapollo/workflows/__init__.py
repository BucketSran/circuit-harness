"""Strict configuration-driven Reasoning workflow orchestration.

The package entry point is :mod:`alphaapollo.workflows.main`. Public contracts
are loaded on access, so importing a stdlib Chips operator does not require the
dependencies of unrelated workflow resources. No resources are constructed here.
"""

from importlib import import_module

__all__ = [
    "CONFIG_VERSION",
    "ConfigError",
    "DatasetConfig",
    "ExecutionConfig",
    "EnsembleConfig",
    "OutputConfig",
    "ResourceConfig",
    "RoleConfig",
    "RunConfig",
    "ScoringConfig",
    "StepConfig",
    "StepResult",
    "TransitionConfig",
    "Workflow",
    "WorkflowConfig",
    "WorkflowExecutionError",
    "WorkflowExecutor",
    "WorkflowInput",
    "WorkflowResult",
    "load_run_config",
    "load_workflow_config",
    "run_workflow",
]

_OWNERS = {
    "WorkflowExecutionError": "executor",
    "WorkflowExecutor": "executor",
    "StepResult": "records",
    "Workflow": "records",
    "WorkflowInput": "records",
    "WorkflowResult": "records",
    "run_workflow": "run",
}


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_OWNERS.get(name, 'config')}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
