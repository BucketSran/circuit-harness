"""Adapters from durable records to Learning runtime containers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CORE_ROLLOUT_TENSOR_FIELDS": "alphaapollo.learning.adapters.dataproto",
    "CaptureDisposition": "alphaapollo.learning.adapters.schemas",
    "CaptureStatus": "alphaapollo.learning.adapters.schemas",
    "DataProtoAdapter": "alphaapollo.learning.adapters.dataproto",
    "DataProtoRestoreError": "alphaapollo.learning.adapters.dataproto",
    "InferenceCaptureRecord": "alphaapollo.learning.adapters.schemas",
    "OUTPUT_TO_INPUT_TENSOR_FIELD": "alphaapollo.learning.adapters.dataproto",
    "ROLLOUT_TENSOR_FIELDS": "alphaapollo.learning.adapters.rollout_tensors",
    "RolloutTensorError": "alphaapollo.learning.adapters.rollout_tensors",
    "RolloutDataProtoAdapter": "alphaapollo.learning.adapters.rollout_dataproto",
    "TurnSegment": "alphaapollo.learning.adapters.rollout_tensors",
    "assemble_trajectory_row": "alphaapollo.learning.adapters.rollout_tensors",
    "build_rollout_row": "alphaapollo.learning.adapters.rollout_tensors",
    "rollout_response_row": "alphaapollo.learning.adapters.rollout_tensors",
    "stack_to_dataproto": "alphaapollo.learning.adapters.rollout_tensors",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
