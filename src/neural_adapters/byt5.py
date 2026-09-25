"""Member C skeleton; intentionally does not import or download Transformers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import AdapterConfig, BasePairAdapter


class ByT5EncoderPairAdapter(BasePairAdapter):
    """Target: google/byt5-small encoder-only adaptation with mean pooling."""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        raise NotImplementedError("Member C must implement google/byt5-small encoder loading; use attention-mask-aware mean pooling")

    def serialize_pair(self, row: Mapping[str, Any]) -> str:
        raise NotImplementedError("Member C: implement shared six-field byte-pair serialization")

    def collate(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        raise NotImplementedError("Member C: implement dynamic byte padding and truncation counters")

    def logits(self, batch: Any) -> Any:
        raise NotImplementedError("Member C: return [batch, 2] logits with class 1 as match")

    def parameters(self):
        raise NotImplementedError("Member C: expose encoder and classifier parameters")

    def to_device(self, device: str):
        raise NotImplementedError("Member C: move encoder and classifier to device")

    def save_pretrained(self, output_dir: str) -> None:
        raise NotImplementedError("Member C: save encoder, tokenizer revision, and adapter configuration")

    @classmethod
    def load_pretrained(cls, output_dir: str, *, map_location: str = "cpu"):
        raise NotImplementedError("Member C: implement checkpoint reload")

