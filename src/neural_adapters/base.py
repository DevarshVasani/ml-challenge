"""The stable adapter boundary shared by both model owners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass
class AdapterConfig:
    adapter_type: str
    checkpoint: str = ""
    revision: str | None = None
    max_length: int = 256
    missing_text: str = ""
    device: str = "cpu"


class BasePairAdapter(ABC):
    """Adapters own model/tokenizer mechanics; the trainer owns optimization."""

    config: AdapterConfig

    @classmethod
    def from_config(cls, config: AdapterConfig, *, execute: bool = False) -> "BasePairAdapter":
        if not execute:
            raise RuntimeError("adapter construction is runtime work; use --execute on a compute machine")
        return cls(config)

    def __init__(self, config: AdapterConfig):
        self.config = config

    @abstractmethod
    def serialize_pair(self, row: Mapping[str, Any]) -> str:
        """Serialize exactly the six raw pair fields in shared order."""

    @abstractmethod
    def collate(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        """Tokenize/dynamically pad a batch and expose truncation accounting."""

    @abstractmethod
    def logits(self, batch: Any) -> Any:
        """Return floating point [batch, 2] logits; class 1 means match."""

    @abstractmethod
    def parameters(self):
        """Return trainable parameters for the shared optimizer."""

    @abstractmethod
    def to_device(self, device: str) -> "BasePairAdapter":
        return self

    def supports_gradient_checkpointing(self) -> bool:
        return False

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        if enabled and not self.supports_gradient_checkpointing():
            raise RuntimeError(f"adapter {self.config.adapter_type} does not support gradient checkpointing")

    @abstractmethod
    def save_pretrained(self, output_dir: str) -> None:
        """Save model-specific state, tokenizer revision and adapter config."""

    @classmethod
    @abstractmethod
    def load_pretrained(cls, output_dir: str, *, map_location: str = "cpu") -> "BasePairAdapter":
        raise NotImplementedError

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)


def get_adapter_class(adapter_type: str):
    key = adapter_type.lower()
    if key in {"fake", "smoke"}:
        from .fake import FakePairAdapter
        return FakePairAdapter
    if key in {"mdeberta", "mdeberta-v3"}:
        from .mdeberta import MDebertaPairAdapter
        return MDebertaPairAdapter
    if key in {"byt5", "byt5-encoder"}:
        from .byt5 import ByT5EncoderPairAdapter
        return ByT5EncoderPairAdapter
    raise ValueError(f"unknown adapter_type {adapter_type!r}; choose fake, mdeberta, or byt5")

