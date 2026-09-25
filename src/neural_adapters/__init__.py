"""Lazy neural adapter registry.

Importing this package does not import Transformers or download anything.
"""

from .base import AdapterConfig, BasePairAdapter, get_adapter_class

__all__ = ["AdapterConfig", "BasePairAdapter", "FakePairAdapter", "get_adapter_class"]


def __getattr__(name):
    if name == "FakePairAdapter":
        from .fake import FakePairAdapter
        return FakePairAdapter
    raise AttributeError(name)
