"""Test-only builders for durable inference records consumed by Learning.

Generation no longer owns an inference recorder.  These helpers construct the
schema records directly so the Learning restore adapter remains covered without
reintroducing the retired Common capture API.
"""

from __future__ import annotations

import io
import json
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np

from alphaapollo.common.artifacts.schemas import ArrayArtifact
from alphaapollo.common.execution import ArtifactStore
from alphaapollo.learning.adapters.schemas import InferenceCaptureRecord


def _array_artifact(
    store: ArtifactStore,
    value: Any,
    *,
    tensor: bool,
) -> ArrayArtifact:
    if tensor and hasattr(value, "detach"):
        import torch

        cpu = value.detach().cpu().contiguous()
        content = cpu.view(dtype=torch.uint8).numpy().tobytes()
        artifact = store.put(
            content,
            type_="application/x-alphaapollo-tensor",
            created_by="tests.learning.capture_fixtures",
        )
        return ArrayArtifact(
            artifact=store.ref(artifact),
            encoding="torch-raw",
            dtype=str(cpu.dtype).removeprefix("torch."),
            shape=list(cpu.shape),
            byte_order=sys.byteorder,
        )

    array = np.asarray(value)
    if array.dtype.hasobject:
        content = json.dumps(
            array.tolist(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        encoding = "json-array"
    else:
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        content = buffer.getvalue()
        encoding = "numpy-npy"
    artifact = store.put(
        content,
        type_="application/x-alphaapollo-array",
        created_by="tests.learning.capture_fixtures",
    )
    return ArrayArtifact(
        artifact=store.ref(artifact),
        encoding=encoding,
        dtype=str(array.dtype),
        shape=list(array.shape),
        byte_order=sys.byteorder,
    )


def inference_record(
    store: ArtifactStore,
    *,
    tensor_batch: Mapping[str, Any],
    non_tensor_batch: Mapping[str, Any] | None = None,
    meta_info: Mapping[str, Any] | None = None,
    input_batch_size: int | None = None,
    output_to_input: Any | None = None,
    model: str = "test-model",
) -> InferenceCaptureRecord:
    """Build an immutable schema record from explicit test arrays."""

    tensors = {
        str(name): _array_artifact(store, value, tensor=True)
        for name, value in tensor_batch.items()
    }
    non_tensors = {
        str(name): _array_artifact(store, value, tensor=False)
        for name, value in (non_tensor_batch or {}).items()
    }
    sizes = {
        *(spec.shape[0] for spec in tensors.values()),
        *(spec.shape[0] for spec in non_tensors.values()),
    }
    if len(sizes) > 1:
        raise ValueError(f"test record fields have inconsistent batch sizes: {sorted(sizes)}")
    batch_size = next(iter(sizes), None)
    mapping = (
        _array_artifact(store, np.asarray(output_to_input, dtype=np.int64), tensor=False)
        if output_to_input is not None
        else None
    )
    return InferenceCaptureRecord(
        record_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        model=model,
        input_batch_size=input_batch_size,
        batch_size=batch_size,
        output_to_input=mapping,
        tensor_batch=tensors,
        non_tensor_batch=non_tensors,
        meta_info=dict(meta_info or {}),
    )


__all__ = ["inference_record"]
