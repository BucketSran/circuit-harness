"""Abstract base for pluggable forward-step + loss components (template method)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any


class BaseForwardStep(ABC):
    """A pluggable forward-step component.

    Factors a Megatron ``forward_step(data_iterator, model)`` into three overridable hooks
    (``get_batch`` / ``compute`` / ``loss_func``) and provides a concrete ``forward_step``
    template method with the exact signature Megatron expects:
    ``(data_iterator, model) -> (output_tensor, loss_func_partial)``. Subclasses set ``name``
    and implement the three hooks.
    """

    name: str

    @abstractmethod
    def get_batch(self, data_iterator: Any) -> dict:
        """Pull the next batch from ``data_iterator``; move tensors to cuda. Returns a dict."""
        ...

    @abstractmethod
    def compute(self, batch: dict, model: Any) -> Any:
        """Run the model forward; return the per-token loss tensor Megatron reduces."""
        ...

    @abstractmethod
    def loss_func(self, loss_mask: Any, output_tensor: Any) -> tuple[Any, Any, dict]:
        """Megatron loss contract: return ``(loss_scalar, num_tokens, report)``."""
        ...

    def forward_step(self, data_iterator: Any, model: Any) -> tuple[Any, Any]:
        """Megatron-signature forward step (concrete template method)."""
        batch = self.get_batch(data_iterator)
        output_tensor = self.compute(batch, model)
        return output_tensor, partial(self.loss_func, batch.get("loss_mask"))
