"""Abstract base classes + the generic registry for the modular pretrain package.

These are the stable contracts every pluggable pretrain component implements. Concrete
components live one-file-each under ``../tokenizers``, ``../datasets``, ``../archs``,
``../forward``, ``../converters`` and register into a per-kind ``Registry``.
"""

from __future__ import annotations

from alphaapollo.learning.off_policy.pretrain.base.converter import BaseCheckpointConverter
from alphaapollo.learning.off_policy.pretrain.base.dataset import BaseDatasetProvider
from alphaapollo.learning.off_policy.pretrain.base.forward import BaseForwardStep
from alphaapollo.learning.off_policy.pretrain.base.model import BaseModelBuilder
from alphaapollo.learning.off_policy.pretrain.base.registry import Registry
from alphaapollo.learning.off_policy.pretrain.base.tokenizer import BaseTokenizer

__all__ = [
    "Registry",
    "BaseTokenizer",
    "BaseDatasetProvider",
    "BaseModelBuilder",
    "BaseForwardStep",
    "BaseCheckpointConverter",
]
