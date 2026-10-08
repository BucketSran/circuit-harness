"""Abstract base for pluggable mcore -> HuggingFace checkpoint converters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseCheckpointConverter(ABC):
    """A pluggable Megatron-dist-checkpoint -> HuggingFace converter.

    The HF directory on disk is the only crossing from pretrain to the RL/SFT side, so each
    architecture family that should reach RL needs a converter keyed by its ``name``
    (e.g. ``"qwen2"``). Subclasses set ``name``.
    """

    name: str

    @abstractmethod
    def export(self, mcore_ckpt_dir: str, hf_out_dir: str, **opts: object) -> str:
        """Convert a Megatron torch_dist checkpoint dir to a HuggingFace model dir.

        Returns ``hf_out_dir``. Writes ``config.json`` + ``model.safetensors``; tokenizer
        files must be copied into ``hf_out_dir`` separately by the caller.
        """
        ...
