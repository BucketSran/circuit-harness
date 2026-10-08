"""Restore durable inference records as v2/verl-compatible DataProto objects."""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from alphaapollo.common.artifacts.schemas import ArrayArtifact
from alphaapollo.common.execution import ArtifactStore
from alphaapollo.learning.adapters.schemas import (
    InferenceCaptureRecord,
    require_supported_inference_schema,
)

CORE_ROLLOUT_TENSOR_FIELDS = frozenset(
    {"prompts", "responses", "input_ids", "attention_mask", "position_ids"}
)
OUTPUT_TO_INPUT_TENSOR_FIELD = "__alphaapollo_output_to_input__"

_TORCH_DTYPES = frozenset(
    {
        "bool",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
    }
)


class DataProtoRestoreError(RuntimeError):
    """Raised when an inference record cannot faithfully form a DataProto."""


class DataProtoAdapter:
    """Learning-owned view over a durable inference capture record."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        dataproto_class: type[Any] | None = None,
        tensor_factory: Callable[[np.ndarray], Any] | None = None,
        required_tensor_fields: frozenset[str] = CORE_ROLLOUT_TENSOR_FIELDS,
        output_to_input_tensor_field: str | None = None,
    ) -> None:
        self._artifacts = artifact_store
        self._dataproto_class = dataproto_class
        self._tensor_factory = tensor_factory
        self._required_tensor_fields = required_tensor_fields
        if output_to_input_tensor_field is not None and not output_to_input_tensor_field.strip():
            raise ValueError("output_to_input_tensor_field must be non-empty when enabled")
        self._output_to_input_tensor_field = output_to_input_tensor_field

    def restore(self, record: InferenceCaptureRecord) -> Any:
        try:
            require_supported_inference_schema(record.schema_version)
        except ValueError as exc:
            raise DataProtoRestoreError(
                f"unsupported inference record schema {record.schema_version}"
            ) from exc
        missing = sorted(self._required_tensor_fields.difference(record.tensor_batch))
        if missing:
            raise DataProtoRestoreError(
                "record is not a complete v2 rollout: " + ", ".join(missing)
            )

        tensors = {key: self._load_tensor(spec) for key, spec in record.tensor_batch.items()}
        if record.output_to_input is not None:
            mapping = self._load_non_tensor(record.output_to_input)
            if mapping.dtype != np.dtype("int64"):
                raise DataProtoRestoreError("input row mapping must use int64 indices")
            if record.input_batch_size is None:
                raise DataProtoRestoreError("input row mapping requires input_batch_size")
            if np.any(mapping < 0) or np.any(mapping >= record.input_batch_size):
                raise DataProtoRestoreError("input row mapping contains an invalid input row")
            mapping_field = self._output_to_input_tensor_field
            if mapping_field is not None:
                if mapping_field in tensors:
                    raise DataProtoRestoreError(
                        f"output-to-input mapping field collides with tensor {mapping_field!r}"
                    )
                if self._tensor_factory is not None:
                    tensors[mapping_field] = self._tensor_factory(mapping)
                else:
                    tensors[mapping_field] = _load_torch().from_numpy(mapping.copy())
        non_tensors = {
            key: self._load_non_tensor(spec) for key, spec in record.non_tensor_batch.items()
        }
        _validate_restored_batch(record.batch_size, tensors, non_tensors)

        dataproto_class = self._dataproto_class or _load_dataproto_class()
        return dataproto_class.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info=dict(record.meta_info),
        )

    def _load_tensor(self, spec: ArrayArtifact) -> Any:
        content = self._artifacts.get(spec.artifact)
        if spec.encoding == "torch-raw":
            if spec.byte_order != sys.byteorder:
                raise DataProtoRestoreError(
                    f"tensor byte order {spec.byte_order} does not match host {sys.byteorder}"
                )
            torch = _load_torch()
            if spec.dtype not in _TORCH_DTYPES:
                raise DataProtoRestoreError(f"unsupported torch dtype {spec.dtype}")
            dtype = getattr(torch, spec.dtype, None)
            if dtype is None or not isinstance(dtype, torch.dtype):
                raise DataProtoRestoreError(f"unsupported torch dtype {spec.dtype}")
            tensor = torch.frombuffer(bytearray(content), dtype=dtype).clone()
            expected = int(np.prod(spec.shape, dtype=np.int64))
            if tensor.numel() != expected:
                raise DataProtoRestoreError("tensor byte length does not match manifest shape")
            return tensor.reshape(spec.shape)
        if spec.encoding != "numpy-npy":
            raise DataProtoRestoreError(f"tensor field cannot use {spec.encoding} encoding")
        array = _load_npy(content, spec)
        if self._tensor_factory is not None:
            return self._tensor_factory(array)
        return _load_torch().from_numpy(array.copy())

    def _load_non_tensor(self, spec: ArrayArtifact) -> np.ndarray:
        content = self._artifacts.get(spec.artifact)
        if spec.encoding == "numpy-npy":
            return _load_npy(content, spec)
        if spec.encoding == "json-array":
            try:
                value = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DataProtoRestoreError("invalid JSON array artifact") from exc
            array = _restore_object_array(value, spec.shape)
            _validate_array_manifest(array, spec, check_dtype=False)
            return array
        raise DataProtoRestoreError(f"non-tensor field cannot use {spec.encoding} encoding")


def _load_npy(content: bytes, spec: ArrayArtifact) -> np.ndarray:
    try:
        array = np.load(io.BytesIO(content), allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise DataProtoRestoreError("invalid safe numpy artifact") from exc
    _validate_array_manifest(array, spec, check_dtype=True)
    return array


def _validate_array_manifest(array: np.ndarray, spec: ArrayArtifact, *, check_dtype: bool) -> None:
    if list(array.shape) != spec.shape:
        raise DataProtoRestoreError(
            f"array shape {list(array.shape)} does not match manifest {spec.shape}"
        )
    if check_dtype and str(array.dtype) != spec.dtype:
        raise DataProtoRestoreError(
            f"array dtype {array.dtype} does not match manifest {spec.dtype}"
        )


def _restore_object_array(value: Any, shape: list[int]) -> np.ndarray:
    array = np.empty(tuple(shape), dtype=object)

    def fill(container: Any, prefix: tuple[int, ...], dimension: int) -> None:
        expected = shape[dimension]
        if not isinstance(container, list) or len(container) != expected:
            raise DataProtoRestoreError("JSON object-array nesting does not match manifest shape")
        if dimension == len(shape) - 1:
            for index, child in enumerate(container):
                array[prefix + (index,)] = child
            return
        for index, child in enumerate(container):
            fill(child, prefix + (index,), dimension + 1)

    if not shape:
        raise DataProtoRestoreError("object array must have a batch dimension")
    fill(value, (), 0)
    return array


def _validate_restored_batch(
    expected: int | None,
    tensors: dict[str, Any],
    non_tensors: dict[str, np.ndarray],
) -> None:
    observed = {
        **{f"tensor.{key}": int(value.shape[0]) for key, value in tensors.items()},
        **{f"non_tensor.{key}": int(value.shape[0]) for key, value in non_tensors.items()},
    }
    sizes = set(observed.values())
    if len(sizes) > 1 or (expected is not None and sizes and sizes != {expected}):
        detail = ", ".join(f"{key}={size}" for key, size in sorted(observed.items()))
        raise DataProtoRestoreError(
            f"restored batch does not match manifest batch_size={expected}: {detail}"
        )


def _load_dataproto_class() -> type[Any]:
    try:
        from verl.protocol import DataProto
    except ImportError as exc:
        raise ImportError(
            "verl is required to restore DataProto; install the Learning runtime or "
            "inject dataproto_class"
        ) from exc
    return DataProto


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to restore DataProto tensors; install alphaapollo[learning]"
        ) from exc
    return torch


__all__ = [
    "CORE_ROLLOUT_TENSOR_FIELDS",
    "DataProtoAdapter",
    "DataProtoRestoreError",
    "OUTPUT_TO_INPUT_TENSOR_FIELD",
]
