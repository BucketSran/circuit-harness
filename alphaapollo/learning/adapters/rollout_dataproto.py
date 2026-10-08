"""Pure conversion from the Learning rollout contract to DataProto."""

from __future__ import annotations

from typing import Any

from alphaapollo.learning.rollout_contract import RolloutBatch


class RolloutDataProtoAdapter:
    """Convert an already assembled rollout without changing training semantics."""

    def __init__(self, *, dataproto_class: type[Any] | None = None) -> None:
        self._dataproto_class = dataproto_class

    def convert(self, rollout: RolloutBatch) -> Any:
        dataproto_class = self._dataproto_class or _load_dataproto_class()
        return dataproto_class.from_dict(
            tensors=dict(rollout.tensors),
            non_tensors=dict(rollout.non_tensors),
            meta_info=dict(rollout.meta_info),
        )


def _load_dataproto_class() -> type[Any]:
    try:
        from verl.protocol import DataProto
    except ImportError as exc:
        raise ImportError(
            "verl is required to convert RolloutBatch; install the Learning runtime or "
            "inject dataproto_class"
        ) from exc
    return DataProto


__all__ = ["RolloutDataProtoAdapter"]
